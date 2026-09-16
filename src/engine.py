"""Training / validation loop.

Colab-oriented choices:
* mixed precision (AMP) to fit 512x512 batches on a T4;
* gradient accumulation to emulate a larger batch when VRAM is the limit;
* `last.pt` written every epoch so a killed session resumes instead of restarting;
* `best.pt` selected on validation Dice, plus early stopping;
* a per-epoch CSV history that feeds the learning curves in the slides.
"""

from __future__ import annotations

import csv
import time
from pathlib import Path

import torch
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, LinearLR, SequentialLR
from tqdm.auto import tqdm

import hashlib
from typing import Any

from .eval.metrics import SegmentationMetrics
from .utils import ensure_dir


# --------------------------------------------------------------------------- #
# Single epoch
# --------------------------------------------------------------------------- #
def train_one_epoch(
    model, loader, criterion, optimizer, scaler, device, accumulation_steps: int = 1
) -> float:
    """Run one training epoch and return the mean loss."""
    model.train()
    total_loss, n_batches = 0.0, 0
    optimizer.zero_grad(set_to_none=True)

    for step, (images, masks) in enumerate(tqdm(loader, desc="train", leave=False)):
        images, masks = images.to(device, non_blocking=True), masks.to(device, non_blocking=True)

        with torch.autocast(device_type=device.type, enabled=scaler.is_enabled()):
            logits = model(images)
            # Scale the loss so accumulated gradients match a single large batch.
            loss = criterion(logits, masks) / accumulation_steps

        scaler.scale(loss).backward()

        if (step + 1) % accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        total_loss += loss.item() * accumulation_steps
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(model, loader, criterion, device, threshold: float = 0.5) -> dict[str, float]:
    """Evaluate on a loader and return {loss, iou, dice, precision, recall, ...}."""
    model.eval()
    meter = SegmentationMetrics(threshold=threshold)
    total_loss, n_batches = 0.0, 0

    for images, masks in tqdm(loader, desc="eval", leave=False):
        images, masks = images.to(device, non_blocking=True), masks.to(device, non_blocking=True)
        logits = model(images)
        total_loss += criterion(logits, masks).item()
        n_batches += 1
        meter.update(logits, masks)

    metrics = meter.compute()
    metrics["loss"] = total_loss / max(n_batches, 1)
    return metrics


# --------------------------------------------------------------------------- #
# Checkpointing
# --------------------------------------------------------------------------- #
def save_checkpoint(path: Path, model, optimizer, scheduler, scaler, epoch: int, best_dice: float, epochs_without_improvement: int = 0) -> None:
    """Persist everything needed to resume training bit-for-bit."""
    torch.save(
        {
            "epoch": epoch,
            "best_dice": best_dice,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "epochs_without_improvement": epochs_without_improvement,
        },
        path,
    )


def load_checkpoint(path: Path, model, optimizer=None, scheduler=None, scaler=None, device="cpu") -> dict:
    """Restore a checkpoint; optimizer/scheduler/scaler are optional (inference)."""
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler is not None:
        scheduler.load_state_dict(checkpoint["scheduler"])
    if scaler is not None:
        scaler.load_state_dict(checkpoint["scaler"])
    return checkpoint


def state_dict_sha256(state_dict: dict[str, torch.Tensor]) -> str:
    """Return a deterministic SHA-256 fingerprint of model parameters."""
    digest = hashlib.sha256()

    for key in sorted(state_dict):
        tensor = state_dict[key].detach().cpu().contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("utf-8"))
        digest.update(tensor.numpy().tobytes())

    return digest.hexdigest()


def initialize_model_from_checkpoint(
    model,
    checkpoint_path: str | Path,
    device: torch.device | str = "cpu",
    strict: bool = True,
) -> dict[str, Any]:
    """
    Initialize only model weights from an external checkpoint.

    This is intentionally different from `load_checkpoint()`:
    it does not restore epoch, optimizer, scheduler or AMP scaler state.
    Use it for a new sequential-transfer experiment, e.g.
    CrackSeg9K -> DACL10K.
    """
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"[transfer] source checkpoint not found: {checkpoint_path.resolve()}"
        )

    checkpoint = torch.load(checkpoint_path, map_location=device)

    if not isinstance(checkpoint, dict):
        raise TypeError(
            "[transfer] unsupported checkpoint format: expected a dictionary."
        )

    if "model" not in checkpoint:
        available = ", ".join(sorted(checkpoint.keys()))
        raise KeyError(
            "[transfer] checkpoint does not contain key 'model'. "
            f"Available keys: {available}"
        )

    source_state_dict = checkpoint["model"]

    if not isinstance(source_state_dict, dict):
        raise TypeError(
            "[transfer] checkpoint['model'] must be a PyTorch state_dict."
        )

    incompatible = model.load_state_dict(source_state_dict, strict=strict)

    if strict and (
        incompatible.missing_keys or incompatible.unexpected_keys
    ):
        raise RuntimeError(
            "[transfer] strict model initialization failed. "
            f"Missing keys: {incompatible.missing_keys}; "
            f"Unexpected keys: {incompatible.unexpected_keys}"
        )

    metadata = {
        "checkpoint_path": str(checkpoint_path.resolve()),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_best_dice": checkpoint.get("best_dice"),
        "checkpoint_keys": sorted(checkpoint.keys()),
        "strict": bool(strict),
        "missing_keys": list(incompatible.missing_keys),
        "unexpected_keys": list(incompatible.unexpected_keys),
        "source_model_sha256": state_dict_sha256(source_state_dict),
        "initialized_model_sha256": state_dict_sha256(model.state_dict()),
    }

    # Fingerprint BEFORE loading: this is what makes the check meaningful.
    factory_sha256 = state_dict_sha256(model.state_dict())

    incompatible = model.load_state_dict(source_state_dict, strict=strict)

    if metadata["initialized_model_sha256"] == factory_sha256:
        raise RuntimeError(
            "[transfer] model weights are unchanged after loading: "
            "the checkpoint did not overwrite the factory initialization."
        )

    if strict and metadata["source_model_sha256"] != metadata["initialized_model_sha256"]:
        raise RuntimeError(
            "[transfer] loaded model fingerprint differs from the source state_dict "
            "under strict loading."
        )

    print(
        "[transfer] initialized model weights from "
        f"{metadata['checkpoint_path']}"
    )
    print(
        "[transfer] source epoch="
        f"{metadata['checkpoint_epoch']} | "
        f"source best_dice={metadata['checkpoint_best_dice']}"
    )
    print(
        "[transfer] SHA-256="
        f"{metadata['initialized_model_sha256']}"
    )

    return metadata

# --------------------------------------------------------------------------- #
# Full training run
# --------------------------------------------------------------------------- #
def fit(model, train_loader, val_loader, criterion, cfg, device, output_dir: Path) -> dict:
    """Train with SGDR + early stopping. Returns the best validation metrics."""
    output_dir = ensure_dir(output_dir)
    last_path, best_path = output_dir / "last.pt", output_dir / "best.pt"
    history_path = output_dir / "history.csv"

    # The model must be on the target device before constructing/loading AdamW.
    # Otherwise Adam moment tensors restored from a checkpoint can remain on CPU
    model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay
    )
    # Warm restarts: the LR is annealed to eta_min over t_0 epochs, then reset;
    # each cycle is t_mult times longer. Helps escape the flat regions typical of
    # highly imbalanced dense tasks.
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

        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=warmup_start_factor,
            end_factor=1.0,
            total_iters=warmup_epochs,
        )
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_epochs],
        )
        print(
            f"[scheduler] linear warm-up for {warmup_epochs} epochs: "
            f"{warmup_start_factor:.3f} × lr -> 1.000 × lr"
        )
    else:
        scheduler = cosine_scheduler

    scaler = torch.amp.GradScaler(enabled=bool(cfg.train.amp) and device.type == "cuda")

    start_epoch, best_dice, epochs_without_improvement = 0, -1.0, 0
    if cfg.train.get("resume") and last_path.exists():
        checkpoint = load_checkpoint(last_path, model, optimizer, scheduler, scaler, device)
        start_epoch, best_dice = checkpoint["epoch"] + 1, checkpoint["best_dice"]
        epochs_without_improvement = checkpoint.get("epochs_without_improvement", 0)
        print(f"[fit] resumed from {last_path} at epoch {start_epoch} (best dice {best_dice:.4f})")

    best_metrics: dict[str, float] = {}

    for epoch in range(start_epoch, cfg.train.epochs):
        started = time.time()
        current_lr = optimizer.param_groups[0]["lr"]
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device,
            accumulation_steps=cfg.train.accumulation_steps,
        )
        val_metrics = evaluate(model, val_loader, criterion, device, threshold=cfg.eval.threshold)
        scheduler.step()  # per-epoch stepping matches T_0 expressed in epochs

        row = {
            "epoch": epoch,
            "lr": current_lr,
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "val_iou": val_metrics["iou"],
            "val_dice": val_metrics["dice"],
            "val_precision": val_metrics["precision"],
            "val_recall": val_metrics["recall"],
            "seconds": round(time.time() - started, 1),
        }
        _append_csv(history_path, row)
        print(
            f"epoch {epoch:03d} | train {train_loss:.4f} | val {val_metrics['loss']:.4f} "
            f"| IoU {val_metrics['iou']:.4f} | Dice {val_metrics['dice']:.4f} "
            f"| P {val_metrics['precision']:.3f} R {val_metrics['recall']:.3f}"
        )
       
        improved = val_metrics["dice"] > best_dice
        if improved:
            best_dice, best_metrics, epochs_without_improvement = val_metrics["dice"], val_metrics, 0
        else:
            epochs_without_improvement += 1

        save_checkpoint(last_path, model, optimizer, scheduler, scaler, epoch, best_dice,
                        epochs_without_improvement)
        if improved:
            save_checkpoint(best_path, model, optimizer, scheduler, scaler, epoch, best_dice,
                            epochs_without_improvement)
            print(f"  -> new best Dice {best_dice:.4f}, saved {best_path.name}")
        elif epochs_without_improvement >= cfg.train.early_stopping_patience:
            print(f"[fit] early stopping after {epochs_without_improvement} epochs without improvement")
            break

    return best_metrics


def _append_csv(path: Path, row: dict) -> None:
    """Append one row to a CSV, writing the header on first use."""
    write_header = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(row))
        if write_header:
            writer.writeheader()
        writer.writerow(row)
