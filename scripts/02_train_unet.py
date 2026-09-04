#!/usr/bin/env python
"""Step 2 - baseline U-Net for binary crack segmentation on CrackSeg9k.

Writes to <output_dir>/runs/<run_name>/:
    config.yaml     exact config used (reproducibility)
    split.json      the image lists of train/val/test (no leakage across runs)
    history.csv     per-epoch losses, LR and validation metrics
    last.pt         resumable checkpoint (Colab-safe)
    best.pt         best checkpoint by validation Dice
    curves.png      loss and Dice/IoU learning curves

Usage
-----
    python scripts/02_train_unet.py --config configs/config.yaml --run-name unet_r34_512
    # quick smoke test before burning GPU hours:
    python scripts/02_train_unet.py --limit-train 64 --limit-val 32 --set train.epochs=1
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import yaml
from torch.utils.data import DataLoader

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.data.crackseg9k import CrackSeg9kDataset, list_pairs, split_pairs
from src.data.transforms import eval_transform, train_transform
from src.engine import fit
from src.losses import build_loss
from src.models.unet import build_model, count_parameters
from src.utils import ensure_dir, get_device, load_config, loader_kwargs, seed_everything

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the baseline U-Net on CrackSeg9k")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--set", nargs="*", default=[], help="config overrides key=value")
    parser.add_argument("--run-name", default=None, help="defaults to <arch>_<encoder>_<size>")
    parser.add_argument("--limit-train", type=int, default=None, help="use only N training images")
    parser.add_argument("--limit-val", type=int, default=None, help="use only N validation images")
    return parser.parse_args()


def plot_curves(history_path: Path, out_path: Path) -> None:
    """Loss curves + validation Dice/IoU, the two figures the slides need."""
    history = pd.read_csv(history_path)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].plot(history.epoch, history.train_loss, label="train")
    axes[0].plot(history.epoch, history.val_loss, label="val")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("BCE + Dice loss")
    axes[0].set_title("Loss")
    axes[0].legend()

    axes[1].plot(history.epoch, history.val_dice, label="Dice")
    axes[1].plot(history.epoch, history.val_iou, label="IoU")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("score")
    axes[1].set_title("Validation metrics")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config, args.set)
    seed_everything(cfg.project.seed)
    device = get_device()

    run_name = args.run_name or f"{cfg.model.arch}_{cfg.model.encoder}_{cfg.data.image_size}"
    run_dir = ensure_dir(Path(cfg.project.output_dir) / "runs" / run_name)
    with open(run_dir / "config.yaml", "w", encoding="utf-8") as fh:
        yaml.safe_dump(dict(cfg), fh, sort_keys=False)

    # ---- Data --------------------------------------------------------------
    pairs = list_pairs(cfg.data.crackseg9k.images_dir, cfg.data.crackseg9k.masks_dir)
    splits = split_pairs(
        pairs,
        val_fraction=cfg.data.crackseg9k.val_fraction,
        test_fraction=cfg.data.crackseg9k.test_fraction,
        seed=cfg.project.seed,
    )
    # Freeze the split on disk: the test set must stay untouched across runs.
    with open(run_dir / "split.json", "w", encoding="utf-8") as fh:
        json.dump({k: [str(i.name) for i, _ in v] for k, v in splits.items()}, fh, indent=2)

    train_items = splits["train"][: args.limit_train] if args.limit_train else splits["train"]
    val_items = splits["val"][: args.limit_val] if args.limit_val else splits["val"]
    print(f"[data] train {len(train_items)} | val {len(val_items)} | test {len(splits['test'])}")

    train_ds = CrackSeg9kDataset(
        train_items, train_transform(cfg.data.image_size), cfg.data.crackseg9k.mask_threshold
    )
    val_ds = CrackSeg9kDataset(
        val_items, eval_transform(cfg.data.image_size), cfg.data.crackseg9k.mask_threshold
    )

    common = loader_kwargs(cfg.data, device)
    train_loader = DataLoader(train_ds, batch_size=cfg.train.batch_size, shuffle=True, drop_last=True, **common)
    val_loader = DataLoader(val_ds, batch_size=cfg.train.batch_size, shuffle=False, **common)

    # ---- Model + loss ------------------------------------------------------
    model = build_model(cfg.model)
    total, trainable = count_parameters(model)
    print(f"[model] {cfg.model.arch}/{cfg.model.encoder} - {total/1e6:.1f}M params ({trainable/1e6:.1f}M trainable)")
    criterion = build_loss(cfg.loss).to(device)

    # ---- Train -------------------------------------------------------------
    best = fit(model, train_loader, val_loader, criterion, cfg, device, run_dir)

    history_path = run_dir / "history.csv"
    if history_path.exists():
        plot_curves(history_path, run_dir / "curves.png")
    if best:
        print("\n=== Best validation metrics ===")
        for key, value in best.items():
            print(f"{key:>28}: {value:.4f}" if isinstance(value, float) else f"{key:>28}: {value}")
    print(f"\nRun artifacts in {run_dir.resolve()}")


if __name__ == "__main__":
    main()
