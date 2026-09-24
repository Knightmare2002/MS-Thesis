"""P4-A training loop for the shared-encoder, two-head network.

Why a separate module instead of extending `engine.fit`
------------------------------------------------------
`engine.fit` is the loop that produced every P0/P1/P2/P3/P1ML number in the
thesis. Its contract is "one model output tensor, one target tensor, one metric
dictionary". A multitask step breaks all three (two outputs, two targets, two
metric families, two loss components to log), so extending it would mean touching
the loop that guarantees the reproducibility of the existing results. This module
therefore *mirrors* `fit` - same AdamW, same SGDR + linear warm-up, same AMP,
same gradient accumulation, same early stopping, same CSV-history mechanics - and
reuses its checkpoint primitives, while `engine.py` stays untouched.

Everything that differs from `fit` is listed explicitly:

1. the batch target is the [B,6,H,W] multilabel mask; the crack target is its
   channel 0 (`Crack + ACrack`), sliced inside the loss, so both heads always see
   exactly the same crop of the same image;
2. the checkpoints are named `best_joint.pt` / `last_joint.pt`, so a P4 run
   directory can never be confused with a single-task one;
3. the history CSV carries the two loss components and the metrics of both heads;
4. model selection uses a joint score (see `joint_selection_score`).
"""

from __future__ import annotations

import time
from pathlib import Path

import torch
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, LinearLR, SequentialLR
from tqdm.auto import tqdm

from .data.class_mapping import UNIFIED_DAMAGE_CLASSES
from .engine import _append_csv, load_checkpoint, save_checkpoint
from .eval.metrics import SegmentationMetrics
from .eval.multilabel_metrics import MultilabelSegmentationMetrics
from .utils import ensure_dir

CRACK_PREFIX = "crack_"

# Keys written to history.csv on top of the mandatory ones, in this order.
MULTITASK_HISTORY_KEYS: tuple[str, ...] = (
    "loss_crack",
    "loss_multilabel",
    "joint_score",
    "crack_dice",
    "crack_iou",
    "crack_precision",
    "crack_recall",
    "macro_dice_present",
    "macro_iou_present",
    "macro_recall_present",
    "micro_dice",
    "n_classes_present",
) + tuple(f"dice_{name}" for name in UNIFIED_DAMAGE_CLASSES)

SELECTION_METRICS = ("joint_score", "crack_dice", "macro_dice_present")


def joint_selection_score(crack_dice: float, macro_dice_present: float) -> float:
    """Unweighted mean of the two task scores.

    P4-A must not be allowed to win on one task by sacrificing the other, which
    is exactly what selecting on `crack_dice` alone (or on `macro_dice_present`
    alone) would permit. The arithmetic mean of the binary Dice and of the macro
    Dice over the classes present in the monitor split is the simplest symmetric
    criterion: no tuned coefficient enters P4-A, and task weighting stays a P4-B
    question.
    """
    return 0.5 * float(crack_dice) + 0.5 * float(macro_dice_present)


def train_one_epoch_multitask(
    model, loader, criterion, optimizer, scaler, device, accumulation_steps: int = 1
) -> dict[str, float]:
    """One multitask epoch. Returns the mean total loss and its two components."""
    model.train()
    totals = {"loss": 0.0, "loss_crack": 0.0, "loss_multilabel": 0.0}
    n_batches = 0
    optimizer.zero_grad(set_to_none=True)

    for step, (images, masks) in enumerate(tqdm(loader, desc="train-mt", leave=False)):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        with torch.autocast(device_type=device.type, enabled=scaler.is_enabled()):
            outputs = model(images)  # (crack_logits, multilabel_logits), one encoder pass
            loss = criterion(outputs, masks) / accumulation_steps

        scaler.scale(loss).backward()

        if (step + 1) % accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        components = getattr(criterion, "last_components", {})
        totals["loss"] += loss.item() * accumulation_steps
        totals["loss_crack"] += float(components.get("crack", 0.0))
        totals["loss_multilabel"] += float(components.get("multilabel", 0.0))
        n_batches += 1

    return {key: value / max(n_batches, 1) for key, value in totals.items()}


@torch.no_grad()
def evaluate_multitask(model, loader, criterion, device, threshold: float = 0.5) -> dict[str, float]:
    """Patch-level evaluation of both heads in a single pass over the loader.

    The returned dictionary is flat and prefixed so nothing collides:
    `crack_*` for the binary head, the P1ML names (`macro_dice_present`,
    `micro_dice`, `dice_<class>`, ...) for the multilabel head, plus `loss`,
    `loss_crack`, `loss_multilabel` and `joint_score`.
    """
    model.eval()
    crack_meter = SegmentationMetrics(threshold=threshold)
    multilabel_meter = MultilabelSegmentationMetrics(threshold=threshold)

    totals = {"loss": 0.0, "loss_crack": 0.0, "loss_multilabel": 0.0}
    n_batches = 0

    for images, masks in tqdm(loader, desc="eval-mt", leave=False):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        crack_logits, multilabel_logits = model(images)
        total = criterion((crack_logits, multilabel_logits), masks)
        components = getattr(criterion, "last_components", {})

        totals["loss"] += float(total.item())
        totals["loss_crack"] += float(components.get("crack", 0.0))
        totals["loss_multilabel"] += float(components.get("multilabel", 0.0))
        n_batches += 1

        crack_meter.update(crack_logits, criterion.crack_target(masks))
        multilabel_meter.update(multilabel_logits, masks)

    metrics: dict[str, float] = {
        f"{CRACK_PREFIX}{key}": value for key, value in crack_meter.compute().items()
    }
    metrics.update(multilabel_meter.compute())

    for key, value in totals.items():
        metrics[key] = value / max(n_batches, 1)

    metrics["joint_score"] = joint_selection_score(
        metrics[f"{CRACK_PREFIX}dice"], metrics["macro_dice_present"]
    )
    return metrics


def _build_optimizer_and_scheduler(model, cfg):
    """AdamW + linear warm-up + SGDR, identical to `engine.fit` by construction."""
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay
    )
    cosine_scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=cfg.train.scheduler.t_0,
        T_mult=cfg.train.scheduler.t_mult,
        eta_min=cfg.train.scheduler.eta_min,
    )

    warmup_epochs = int(cfg.train.get("warmup_epochs", 0))
    warmup_start_factor = float(cfg.train.get("warmup_start_factor", 1.0))

    if warmup_epochs > 0:
        if not 0.0 < warmup_start_factor <= 1.0:
            raise ValueError("train.warmup_start_factor must be in (0, 1].")
        scheduler = SequentialLR(
            optimizer,
            schedulers=[
                LinearLR(
                    optimizer,
                    start_factor=warmup_start_factor,
                    end_factor=1.0,
                    total_iters=warmup_epochs,
                ),
                cosine_scheduler,
            ],
            milestones=[warmup_epochs],
        )
        print(
            f"[scheduler] linear warm-up for {warmup_epochs} epochs: "
            f"{warmup_start_factor:.3f} x lr -> 1.000 x lr"
        )
    else:
        scheduler = cosine_scheduler

    return optimizer, scheduler


def fit_multitask(
    model,
    train_loader,
    val_loader,
    criterion,
    cfg,
    device,
    output_dir: Path,
    evaluate_fn=None,
    selection_metric: str = "joint_score",
) -> dict:
    """Train the P4-A network and return the best validation metrics.

    Writes `history.csv`, `last_joint.pt` every epoch and `best_joint.pt` on every
    improvement of `selection_metric`.
    """
    if selection_metric not in SELECTION_METRICS:
        raise ValueError(
            f"selection_metric must be one of {SELECTION_METRICS}, got '{selection_metric}'."
        )

    evaluate_fn = evaluate_fn or evaluate_multitask
    output_dir = ensure_dir(output_dir)
    last_path, best_path = output_dir / "last_joint.pt", output_dir / "best_joint.pt"
    history_path = output_dir / "history.csv"

    model.to(device)
    optimizer, scheduler = _build_optimizer_and_scheduler(model, cfg)
    scaler = torch.amp.GradScaler(enabled=bool(cfg.train.amp) and device.type == "cuda")

    start_epoch, best_score, epochs_without_improvement = 0, -1.0, 0
    if cfg.train.get("resume") and last_path.exists():
        checkpoint = load_checkpoint(last_path, model, optimizer, scheduler, scaler, device)
        start_epoch = checkpoint["epoch"] + 1
        best_score = checkpoint["best_dice"]  # key kept for checkpoint compatibility
        epochs_without_improvement = checkpoint.get("epochs_without_improvement", 0)
        print(
            f"[fit-mt] resumed from {last_path} at epoch {start_epoch} "
            f"(best {selection_metric} {best_score:.4f})"
        )

    best_metrics: dict[str, float] = {}

    for epoch in range(start_epoch, cfg.train.epochs):
        started = time.time()
        current_lr = optimizer.param_groups[0]["lr"]

        train_losses = train_one_epoch_multitask(
            model,
            train_loader,
            criterion,
            optimizer,
            scaler,
            device,
            accumulation_steps=cfg.train.accumulation_steps,
        )
        val_metrics = evaluate_fn(
            model, val_loader, criterion, device, threshold=cfg.eval.threshold
        )
        scheduler.step()

        row = {
            "epoch": epoch,
            "lr": current_lr,
            "train_loss": train_losses["loss"],
            "train_loss_crack": train_losses["loss_crack"],
            "train_loss_multilabel": train_losses["loss_multilabel"],
            "val_loss": val_metrics["loss"],
        }
        for key in MULTITASK_HISTORY_KEYS:
            row[f"val_{key}"] = val_metrics[key]
        row["seconds"] = round(time.time() - started, 1)
        _append_csv(history_path, row)

        print(
            f"epoch {epoch:03d} | train {train_losses['loss']:.4f} "
            f"(crack {train_losses['loss_crack']:.4f} / ml {train_losses['loss_multilabel']:.4f}) "
            f"| val {val_metrics['loss']:.4f} | crack Dice {val_metrics['crack_dice']:.4f} "
            f"| macro Dice {val_metrics['macro_dice_present']:.4f} "
            f"| joint {val_metrics['joint_score']:.4f}"
        )

        if selection_metric not in val_metrics:
            raise KeyError(
                f"selection_metric '{selection_metric}' is not returned by the "
                f"evaluation function. Available: {sorted(val_metrics)}"
            )

        improved = val_metrics[selection_metric] > best_score
        if improved:
            best_score, best_metrics, epochs_without_improvement = (
                val_metrics[selection_metric],
                val_metrics,
                0,
            )
        else:
            epochs_without_improvement += 1

        save_checkpoint(
            last_path, model, optimizer, scheduler, scaler, epoch, best_score,
            epochs_without_improvement,
        )
        if improved:
            save_checkpoint(
                best_path, model, optimizer, scheduler, scaler, epoch, best_score,
                epochs_without_improvement,
            )
            print(f"  -> new best {selection_metric} {best_score:.4f}, saved {best_path.name}")
        elif epochs_without_improvement >= cfg.train.early_stopping_patience:
            print(
                f"[fit-mt] early stopping after {epochs_without_improvement} "
                "epochs without improvement"
            )
            break

    return best_metrics
