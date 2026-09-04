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
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from tqdm.auto import tqdm

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
def save_checkpoint(path: Path, model, optimizer, scheduler, scaler, epoch: int, best_dice: float) -> None:
    """Persist everything needed to resume training bit-for-bit."""
    torch.save(
        {
            "epoch": epoch,
            "best_dice": best_dice,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
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


# --------------------------------------------------------------------------- #
# Full training run
# --------------------------------------------------------------------------- #
def fit(model, train_loader, val_loader, criterion, cfg, device, output_dir: Path) -> dict:
    """Train with SGDR + early stopping. Returns the best validation metrics."""
    output_dir = ensure_dir(output_dir)
    last_path, best_path = output_dir / "last.pt", output_dir / "best.pt"
    history_path = output_dir / "history.csv"

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay
    )
    # Warm restarts: the LR is annealed to eta_min over t_0 epochs, then reset;
    # each cycle is t_mult times longer. Helps escape the flat regions typical of
    # highly imbalanced dense tasks.
    scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=cfg.train.scheduler.t_0,
        T_mult=cfg.train.scheduler.t_mult,
        eta_min=cfg.train.scheduler.eta_min,
    )
    scaler = torch.amp.GradScaler(enabled=bool(cfg.train.amp) and device.type == "cuda")

    start_epoch, best_dice, epochs_without_improvement = 0, 0.0, 0
    if cfg.train.get("resume") and last_path.exists():
        checkpoint = load_checkpoint(last_path, model, optimizer, scheduler, scaler, device)
        start_epoch, best_dice = checkpoint["epoch"] + 1, checkpoint["best_dice"]
        print(f"[fit] resumed from {last_path} at epoch {start_epoch} (best dice {best_dice:.4f})")

    model.to(device)
    best_metrics: dict[str, float] = {}

    for epoch in range(start_epoch, cfg.train.epochs):
        started = time.time()
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device,
            accumulation_steps=cfg.train.accumulation_steps,
        )
        val_metrics = evaluate(model, val_loader, criterion, device, threshold=cfg.eval.threshold)
        scheduler.step()  # per-epoch stepping matches T_0 expressed in epochs

        row = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
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

        save_checkpoint(last_path, model, optimizer, scheduler, scaler, epoch, best_dice)
        if val_metrics["dice"] > best_dice:
            best_dice, best_metrics, epochs_without_improvement = val_metrics["dice"], val_metrics, 0
            save_checkpoint(best_path, model, optimizer, scheduler, scaler, epoch, best_dice)
            print(f"  -> new best Dice {best_dice:.4f}, saved {best_path.name}")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= cfg.train.early_stopping_patience:
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
