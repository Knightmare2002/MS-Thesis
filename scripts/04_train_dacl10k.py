#!/usr/bin/env python
"""Step 4 / P1 - train a binary bridge-damage baseline on DACL10K.

Target:
    Crack union ACrack -> one binary crack mask.

Writes to <output_dir>/runs/<run_name>/:
    config.yaml              exact configuration used
    dataset_summary.json     official split composition and sampler settings
    history.csv              per-epoch losses and validation metrics
    last.pt                  resumable checkpoint
    best.pt                  checkpoint selected by validation Dice
    curves.png               training curves

Usage
-----
    python scripts/04_train_dacl10k.py --run-name p1_dacl10k_unet_r34_512
    python scripts/04_train_dacl10k.py --limit-train 64 --limit-val 32 --set train.epochs=1 data.num_workers=0
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
from torch.utils.data import DataLoader, WeightedRandomSampler

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.data.dacl10k import (
    Dacl10kCrackDataset,
    binary_sample_targets,
    list_samples,
    summarize_binary_targets,
)
from src.data.transforms import eval_transform, train_transform
from src.engine import fit
from src.losses import build_loss
from src.models.unet import build_model, count_parameters
from src.utils import ensure_dir, get_device, load_config, loader_kwargs, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the P1 binary bridge-damage baseline on DACL10K."
    )
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--set", nargs="*", default=[], help="config overrides key=value")
    parser.add_argument("--run-name", default=None, help="defaults to p1_<arch>_<encoder>_<size>")
    parser.add_argument("--limit-train", type=int, default=None, help="use only N train images")
    parser.add_argument("--limit-val", type=int, default=None, help="use only N validation images")
    parser.add_argument(
        "--disable-balanced-sampler",
        action="store_true",
        help="ablation/debug: use ordinary shuffled sampling instead of weighted sampling",
    )
    return parser.parse_args()


def plot_curves(history_path: Path, out_path: Path) -> None:
    """Save training/validation loss and validation Dice/IoU curves."""
    history = pd.read_csv(history_path)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].plot(history.epoch, history.train_loss, label="train")
    axes[0].plot(history.epoch, history.val_loss, label="val")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("BCE + Dice loss")
    axes[0].set_title("P1 loss")
    axes[0].legend()

    axes[1].plot(history.epoch, history.val_dice, label="Dice")
    axes[1].plot(history.epoch, history.val_iou, label="IoU")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("score")
    axes[1].set_title("P1 validation metrics")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def build_train_sampler(targets: list[int], desired_positive_fraction: float) -> WeightedRandomSampler:
    """Create a replacement sampler with the requested expected positive fraction."""
    if not 0.0 < desired_positive_fraction < 1.0:
        raise ValueError("p1.train_positive_fraction must be strictly between 0 and 1.")

    n_positive = sum(targets)
    n_negative = len(targets) - n_positive
    if n_positive == 0 or n_negative == 0:
        raise RuntimeError(
            "Balanced sampling requires both positive and negative DACL10K train samples."
        )

    positive_weight = desired_positive_fraction / n_positive
    negative_weight = (1.0 - desired_positive_fraction) / n_negative
    weights = [positive_weight if target else negative_weight for target in targets]

    return WeightedRandomSampler(
        weights=weights,
        num_samples=len(weights),
        replacement=True,
    )


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config, args.set)
    seed_everything(cfg.project.seed)
    device = get_device()

    run_name = args.run_name or f"p1_{cfg.model.arch}_{cfg.model.encoder}_{cfg.data.image_size}"
    run_dir = ensure_dir(Path(cfg.project.output_dir) / "runs" / run_name)

    with open(run_dir / "config.yaml", "w", encoding="utf-8") as fh:
        yaml.safe_dump(dict(cfg), fh, sort_keys=False)

    labels = list(cfg.data.dacl10k.get("crack_labels", ["Crack", "ACrack"]))
    train_samples = list_samples(cfg.data.dacl10k.root, cfg.data.dacl10k.train_split)
    val_samples = list_samples(cfg.data.dacl10k.root, cfg.data.dacl10k.val_split)

    if args.limit_train:
        train_samples = train_samples[: args.limit_train]
    if args.limit_val:
        val_samples = val_samples[: args.limit_val]

    train_targets = binary_sample_targets(train_samples, labels)
    val_targets = binary_sample_targets(val_samples, labels)
    train_summary = summarize_binary_targets(train_targets)
    val_summary = summarize_binary_targets(val_targets)

    use_sampler = bool(cfg.data.p1.use_weighted_sampler) and not args.disable_balanced_sampler
    summary = {
        "task": str(cfg.data.p1.task_name),
        "target_labels": labels,
        "official_train_split": str(cfg.data.dacl10k.train_split),
        "official_val_split": str(cfg.data.dacl10k.val_split),
        "train": train_summary,
        "validation": val_summary,
        "weighted_sampler": {
            "enabled": use_sampler,
            "desired_positive_fraction": float(cfg.data.p1.train_positive_fraction) if use_sampler else None,
        },
    }
    with open(run_dir / "dataset_summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    print(
        f"[data] train {train_summary['n_images']} "
        f"(positive {train_summary['positive_fraction']:.1%}) | "
        f"val {val_summary['n_images']} "
        f"(positive {val_summary['positive_fraction']:.1%})"
    )
    if use_sampler:
        print(
            f"[sampler] weighted replacement sampling; expected positive share "
            f"{cfg.data.p1.train_positive_fraction:.1%}"
        )

    train_ds = Dacl10kCrackDataset(
        train_samples,
        train_transform(cfg.data.image_size),
        labels=labels,
    )
    val_ds = Dacl10kCrackDataset(
        val_samples,
        eval_transform(cfg.data.image_size),
        labels=labels,
    )

    common = loader_kwargs(cfg.data, device)
    if use_sampler:
        sampler = build_train_sampler(train_targets, float(cfg.data.p1.train_positive_fraction))
        train_loader = DataLoader(
            train_ds,
            batch_size=cfg.train.batch_size,
            sampler=sampler,
            drop_last=True,
            **common,
        )
    else:
        train_loader = DataLoader(
            train_ds,
            batch_size=cfg.train.batch_size,
            shuffle=True,
            drop_last=True,
            **common,
        )

    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        **common,
    )

    model = build_model(cfg.model)
    total, trainable = count_parameters(model)
    print(
        f"[model] {cfg.model.arch}/{cfg.model.encoder} - "
        f"{total / 1e6:.1f}M params ({trainable / 1e6:.1f}M trainable)"
    )
    criterion = build_loss(cfg.loss).to(device)

    best = fit(model, train_loader, val_loader, criterion, cfg, device, run_dir)

    history_path = run_dir / "history.csv"
    if history_path.exists():
        plot_curves(history_path, run_dir / "curves.png")

    if best:
        print("\n=== Best P1 validation metrics ===")
        for key, value in best.items():
            print(f"{key:>28}: {value:.4f}" if isinstance(value, float) else f"{key:>28}: {value}")

    print(f"\nRun artifacts in {run_dir.resolve()}")


if __name__ == "__main__":
    main()