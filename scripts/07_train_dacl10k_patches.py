#!/usr/bin/env python
"""Step 7 / P1-B - native-resolution patch training on DACL10K.

Unlike P1-A, this script never globally resizes full bridge images. It trains
the same U-Net on random native-resolution patches and keeps official DACL10K
train/validation split membership unchanged.

Validation during training is patch-level and deterministic. Full-image
sliding-window validation is performed after training by
08_evaluate_dacl10k_sliding.py.

Writes to <output_dir>/runs/<run_name>/:
    config.yaml
    dataset_summary.json
    history.csv
    last.pt
    best.pt
    curves.png

Usage
-----
    python scripts/07_train_dacl10k_patches.py --run-name p1b_dacl10k_patch_unet_r34_512
    python scripts/07_train_dacl10k_patches.py --limit-train 64 --limit-val 32 --set train.epochs=1 data.num_workers=0
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import yaml
from torch.utils.data import DataLoader

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.data.dacl10k import (
    Dacl10kCrackPatchDataset,
    Dacl10kCrackCenterPatchDataset,
    binary_sample_targets,
    list_samples,
    summarize_binary_targets,
)
from src.data.transforms import patch_eval_transform, patch_train_transform
from src.engine import fit
from src.losses import build_loss
from src.models.unet import build_model, count_parameters
from src.utils import ensure_dir, get_device, load_config, loader_kwargs, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train P1-B DACL10K binary crack segmentation on native-resolution patches."
    )
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--set", nargs="*", default=[], help="config overrides key=value")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--limit-train", type=int, default=None)
    parser.add_argument("--limit-val", type=int, default=None)
    return parser.parse_args()


def plot_curves(history_path: Path, out_path: Path) -> None:
    """Save loss and validation overlap curves."""
    history = pd.read_csv(history_path)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].plot(history.epoch, history.train_loss, label="train")
    axes[0].plot(history.epoch, history.val_loss, label="val patches")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("BCE + Dice loss")
    axes[0].set_title("P1-B patch loss")
    axes[0].legend()

    axes[1].plot(history.epoch, history.val_dice, label="Dice")
    axes[1].plot(history.epoch, history.val_iou, label="IoU")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("score")
    axes[1].set_title("P1-B validation patches")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def patch_sampling_sanity_check(
    dataset,
    n_samples: int,
    min_positive_pixels: int,
    ) -> dict:
        """Measure patch composition on indices sampled across the whole dataset."""
        n_samples = min(int(n_samples), len(dataset))

        # Equally spaced indices cover both the positive and negative index ranges.
        indices = np.linspace(
            0,
            len(dataset) - 1,
            num=n_samples,
            dtype=int,
        )

        crack_pixels = []
        positive_patches = 0
        negative_patches = 0
        intermediate_patches = 0

        for index in indices:
            _, mask = dataset[int(index)]
            n_crack_pixels = int(mask.sum().item())
            crack_pixels.append(n_crack_pixels)

            if n_crack_pixels >= min_positive_pixels:
                positive_patches += 1
            elif n_crack_pixels == 0:
                negative_patches += 1
            else:
                intermediate_patches += 1

        return {
            "n_samples": n_samples,
            "positive_patches": positive_patches,
            "negative_patches": negative_patches,
            "intermediate_patches": intermediate_patches,
            "positive_fraction_observed": positive_patches / n_samples,
            "negative_fraction_observed": negative_patches / n_samples,
            "intermediate_fraction_observed": intermediate_patches / n_samples,
            "mean_crack_pixels": sum(crack_pixels) / n_samples,
            "min_crack_pixels": min(crack_pixels),
            "max_crack_pixels": max(crack_pixels),
        }


def make_validation_dataset(samples, cfg, labels):
    """Create one deterministic center patch per validation image."""
    return Dacl10kCrackCenterPatchDataset(
        samples=samples,
        transform=patch_eval_transform(),
        patch_size=cfg.data.p1_patch.patch_size,
        labels=labels,
    )


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config, args.set)
    seed_everything(cfg.project.seed)
    device = get_device()

    run_name = args.run_name or (
        f"p1b_patch_{cfg.model.arch}_{cfg.model.encoder}_{cfg.data.p1_patch.patch_size}"
    )
    run_dir = ensure_dir(Path(cfg.project.output_dir) / "runs" / run_name)

    with open(run_dir / "config.yaml", "w", encoding="utf-8") as fh:
        yaml.safe_dump(dict(cfg), fh, sort_keys=False)

    labels = list(cfg.data.dacl10k.get("crack_labels", ["Crack", "ACrack"]))
    train_samples = list_samples(cfg.data.dacl10k.root, cfg.data.dacl10k.train_split)
    val_samples = list_samples(cfg.data.dacl10k.root, cfg.data.dacl10k.val_split)
    hnm_cfg = cfg.data.hard_negative_mining

    if args.limit_train:
        train_samples = train_samples[: args.limit_train]
    if args.limit_val:
        val_samples = val_samples[: args.limit_val]

    train_summary = summarize_binary_targets(binary_sample_targets(train_samples, labels))
    val_summary = summarize_binary_targets(binary_sample_targets(val_samples, labels))

    print(
            f"[data] train images {train_summary['n_images']} | "
            f"val images {val_summary['n_images']} | "
            f"patch size {cfg.data.p1_patch.patch_size} | "
            f"train patches/epoch {len(train_samples) * cfg.data.p1_patch.patches_per_image}"
        )
    
    if bool(hnm_cfg.enabled):
        pool_path = Path(hnm_cfg.pool_path)

        if not pool_path.is_file():
            raise FileNotFoundError(
                "Hard-negative mining is enabled, but the pool does not exist:\n"
                f"  {pool_path}\n"
                "Generate it first using scripts/08_mine_hard_negatives.py."
            )

        if not Path(hnm_cfg.source_checkpoint).is_file():
            print(
                "[warning] HNM source checkpoint does not exist at the configured path:\n"
                f"  {hnm_cfg.source_checkpoint}\n"
                "The pool can still be used if it was previously generated, "
                "but verify experiment provenance."
            )

    train_ds = Dacl10kCrackPatchDataset(
        samples=train_samples,
        transform=patch_train_transform(),
        patch_size=cfg.data.p1_patch.patch_size,
        positive_patch_fraction=cfg.data.p1_patch.positive_patch_fraction,
        min_positive_pixels=cfg.data.p1_patch.min_positive_pixels,
        max_negative_pixels=cfg.data.p1_patch.max_negative_pixels,
        max_crop_attempts=cfg.data.p1_patch.max_crop_attempts,
        patches_per_image=cfg.data.p1_patch.patches_per_image,
        labels=labels,

        hard_negative_pool_path=(hnm_cfg.pool_path if hnm_cfg.enabled else None),
        hard_negative_fraction=(
            float(hnm_cfg.hard_negative_fraction)
            if bool(hnm_cfg.enabled)
            else 0.0
        )
    )
    val_ds = make_validation_dataset(val_samples, cfg, labels)

    sampling_check = patch_sampling_sanity_check(
        dataset=train_ds,
        n_samples=cfg.data.p1_patch.sanity_check_samples,
        min_positive_pixels=cfg.data.p1_patch.min_positive_pixels,
    )

    if bool(hnm_cfg.enabled):
        expected_hard_negative = (
            train_ds.n_hard_negative_patches / len(train_ds)
        )
        expected_random_negative = (
            train_ds.n_random_negative_patches / len(train_ds)
        )

        print(
            f"[hnm] pool patches {len(train_ds.hard_negative_pool)} | "
            f"hard-negative slots {train_ds.n_hard_negative_patches} "
            f"({expected_hard_negative:.1%}) | "
            f"random-negative slots {train_ds.n_random_negative_patches} "
            f"({expected_random_negative:.1%})"
    )

    print(
        f"[sampling] positive patches "
        f"{sampling_check['positive_fraction_observed']:.1%} | "
        f"negative patches "
        f"{sampling_check['negative_fraction_observed']:.1%} | "
        f"intermediate patches "
        f"{sampling_check['intermediate_fraction_observed']:.1%} | "
        f"mean crack pixels {sampling_check['mean_crack_pixels']:.1f}"
    )

    expected_positive = float(cfg.data.p1_patch.positive_patch_fraction)
    observed_positive = sampling_check["positive_fraction_observed"]
    if abs(observed_positive - expected_positive) > 0.03:
        print(
            f"[sampling] WARNING: expected about {expected_positive:.1%} positive patches, "
            f"observed {observed_positive:.1%}. Geometric augmentation can push a marginal "
            f"patch below min_positive_pixels; investigate only if the gap is large."
        )

    summary = {
        "task": str(cfg.data.p1_patch.task_name),
        "target_labels": labels,
        "official_train_split": str(cfg.data.dacl10k.train_split),
        "official_val_split": str(cfg.data.dacl10k.val_split),
        "train_images": train_summary,
        "validation_images": val_summary,
        "patch_sampling": {
            "patch_size": int(cfg.data.p1_patch.patch_size),
            "patches_per_image": int(cfg.data.p1_patch.patches_per_image),
            "positive_patch_fraction": float(cfg.data.p1_patch.positive_patch_fraction),
            "min_positive_pixels": int(cfg.data.p1_patch.min_positive_pixels),
            "max_negative_pixels": int(cfg.data.p1_patch.max_negative_pixels),
            "max_crop_attempts": int(cfg.data.p1_patch.max_crop_attempts),
        },
        
        "hard_negative_mining": {
            "enabled": bool(hnm_cfg.enabled),

            "source_checkpoint": (
                str(hnm_cfg.source_checkpoint)
                if bool(hnm_cfg.enabled)
                else None
            ),
            "pool_path": (
                str(hnm_cfg.pool_path)
                if bool(hnm_cfg.enabled)
                else None
            ),
            "pool_size": (
                int(len(train_ds.hard_negative_pool))
                if bool(hnm_cfg.enabled)
                else 0
            ),

            "hard_negative_fraction_among_negative": (
                float(hnm_cfg.hard_negative_fraction)
                if bool(hnm_cfg.enabled)
                else 0.0
            ),
            "n_hard_negative_patch_slots": (
                int(train_ds.n_hard_negative_patches)
                if bool(hnm_cfg.enabled)
                else 0
            ),
            "n_random_negative_patch_slots": (
                int(train_ds.n_random_negative_patches)
                if bool(hnm_cfg.enabled)
                else int(train_ds.n_negative_patches)
            ),

            "candidate_stride": (
                int(hnm_cfg.candidate_stride)
                if bool(hnm_cfg.enabled)
                else None
            ),
            "top_k_per_image": (
                int(hnm_cfg.top_k_per_image)
                if bool(hnm_cfg.enabled)
                else None
            ),
            "max_window_overlap": (
                float(hnm_cfg.max_window_overlap)
                if bool(hnm_cfg.enabled)
                else None
            ),
            "probability_threshold": (
                float(hnm_cfg.probability_threshold)
                if bool(hnm_cfg.enabled)
                else None
            ),
            "min_predicted_fraction": (
                float(hnm_cfg.min_predicted_fraction)
                if bool(hnm_cfg.enabled)
                else None
            ),
            "negative_images_only": (
                bool(hnm_cfg.negative_images_only)
                if bool(hnm_cfg.enabled)
                else None
            ),
            "refresh_each_epoch": (
                bool(hnm_cfg.refresh_each_epoch)
                if bool(hnm_cfg.enabled)
                else None
            ),
            "seed": (
                int(hnm_cfg.seed)
                if bool(hnm_cfg.enabled)
                else None
            ),
        },

        "sampling_sanity_check": sampling_check,
        "validation_note": (
            "Training-time validation uses one random-free normalized patch per image "
            "only as an optimization monitor. Full-image metrics require sliding-window "
            "inference and are written by scripts/08_evaluate_dacl10k_sliding.py."
        ),
    }
    with open(run_dir / "dataset_summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)


    common = loader_kwargs(cfg.data, device)
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
        print("\n=== Best training-monitor validation metrics ===")
        for key, value in best.items():
            print(f"{key:>28}: {value:.4f}" if isinstance(value, float) else f"{key:>28}: {value}")

    print(f"\nPatch-training artifacts in {run_dir.resolve()}")
    print("Run scripts/08_evaluate_dacl10k_sliding.py for full-resolution validation.")


if __name__ == "__main__":
    main()