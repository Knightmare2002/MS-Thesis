#!/usr/bin/env python
r"""Step 14 / P4-A - multitask training on DACL10K (shared encoder, two heads).

    x --> ResNet-34 (shared) --> crack decoder/head      --> [B,1,H,W]
                            \-> multilabel decoder/head  --> [B,6,H,W]

Controlled comparison
---------------------
Everything except the architecture is copied from configs/config_p1ml.yaml, which
in turn mirrors P1-B4: official DACL10K train/validation split, native 512 patch
sampling with a 60% positive quota on the union of the 6 damage channels,
identical transforms, AdamW(3e-4, wd 1e-4), AMP, accumulation 2, SGDR with 2
warm-up epochs, 30 epochs, early stopping patience 10, threshold 0.5.
The crack target is channel 0 (Crack + ACrack) of the same multilabel mask, so
the two heads always see the same crop and the taxonomy is never duplicated.

Writes to <output_dir>/runs/<run_name>/:
    config.yaml  dataset_summary.json  class_weights.json  history.csv
    last_joint.pt  best_joint.pt  curves.png  best_monitor_metrics.json

Usage
-----
    python scripts/14_train_dacl10k_multitask.py --run-name p4a_unetpp_r34_shared_encoder

    # smoke run
    python scripts/14_train_dacl10k_multitask.py --limit-train 32 --limit-val 16 \
        --set train.epochs=1 data.num_workers=0 data.p4a_patch.class_weights.max_images=16
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from torch.utils.data import DataLoader

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.data.class_mapping import (
    UNIFIED_DAMAGE_CLASSES,
    assert_unified_damage_taxonomy,
    unified_damage_label_groups,
)
from src.data.dacl10k import (
    Dacl10kMultilabelCenterPatchDataset,
    Dacl10kMultilabelPatchDataset,
    list_samples,
    multilabel_sample_targets,
    summarize_multilabel_targets,
)
from src.data.stats import resolve_multilabel_class_weights
from src.data.transforms import (
    patch_eval_transform_multilabel,
    patch_train_transform_multilabel,
)
from src.engine_multitask import evaluate_multitask, fit_multitask
from src.losses import build_multitask_loss
from src.models.multitask import build_multitask_model
from src.provenance import save_resolved_run_config, to_jsonable
from src.utils import (
    ensure_dir,
    get_device,
    load_config,
    loader_kwargs,
    multilabel_patch_cfg,
    seed_everything,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the P4-A multitask network (crack + 6-channel damage) on DACL10K."
    )
    parser.add_argument("--config", default="configs/config_p4a.yaml")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--limit-train", type=int, default=None)
    parser.add_argument("--limit-val", type=int, default=None)
    parser.add_argument(
        "--recompute-class-weights",
        action="store_true",
        help="ignore the cached pos_weight vector and recompute it from the train split",
    )
    parser.add_argument("--set", nargs="*", default=[], help="config overrides key=value")
    return parser.parse_args()


def plot_curves(history_path: Path, out_path: Path) -> None:
    """Loss components, crack-head metrics and multilabel-head metrics."""
    history = pd.read_csv(history_path)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    axes[0].plot(history.epoch, history.train_loss, label="train total")
    axes[0].plot(history.epoch, history.train_loss_crack, label="train crack", linestyle="--")
    axes[0].plot(
        history.epoch, history.train_loss_multilabel, label="train multilabel", linestyle=":"
    )
    axes[0].plot(history.epoch, history.val_loss, label="val total")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("loss")
    axes[0].set_title("P4-A joint loss")
    axes[0].legend(fontsize=8)

    axes[1].plot(history.epoch, history.val_crack_dice, label="Dice")
    axes[1].plot(history.epoch, history.val_crack_iou, label="IoU")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("score")
    axes[1].set_title("crack head (val patches)")
    axes[1].legend(fontsize=8)

    axes[2].plot(history.epoch, history.val_macro_dice_present, label="macro Dice (present)")
    axes[2].plot(history.epoch, history.val_macro_iou_present, label="macro IoU (present)")
    axes[2].plot(history.epoch, history.val_micro_dice, label="micro Dice", linestyle="--")
    if "val_joint_score" in history:
        axes[2].plot(history.epoch, history.val_joint_score, label="joint score", linewidth=2)
    axes[2].set_xlabel("epoch")
    axes[2].set_ylabel("score")
    axes[2].set_title("multilabel head (val patches)")
    axes[2].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def patch_sampling_sanity_check(dataset, n_samples: int, min_positive_pixels: int) -> dict:
    """Patch composition on indices spread across the dataset.

    Identical definition to P1ML (positive = union of the 6 channels), with one
    extra P4-A quantity: the fraction of sampled patches whose *crack* channel is
    non-empty. A joint run whose crack channel is almost always empty would be
    training the crack head on negatives only, and the comparison against P1-B4
    would be meaningless.
    """
    n_samples = min(int(n_samples), len(dataset))
    indices = np.linspace(0, len(dataset) - 1, num=n_samples, dtype=int)

    n_classes = len(UNIFIED_DAMAGE_CLASSES)
    union_pixels: list[int] = []
    crack_pixels: list[int] = []
    per_class_patches = [0] * n_classes
    positive_patches = negative_patches = intermediate_patches = 0
    crack_positive_patches = 0

    for index in indices:
        _, mask = dataset[int(index)]
        union = mask.amax(dim=0) > 0.5
        n_positive = int(union.sum().item())
        union_pixels.append(n_positive)

        if n_positive >= min_positive_pixels:
            positive_patches += 1
        elif n_positive == 0:
            negative_patches += 1
        else:
            intermediate_patches += 1

        channel_positive = mask.flatten(1).sum(dim=1)
        for channel in range(n_classes):
            per_class_patches[channel] += int(channel_positive[channel].item() > 0)

        n_crack = int(channel_positive[0].item())
        crack_pixels.append(n_crack)
        crack_positive_patches += int(n_crack >= min_positive_pixels)

    return {
        "n_samples": n_samples,
        "positive_patches": positive_patches,
        "negative_patches": negative_patches,
        "intermediate_patches": intermediate_patches,
        "positive_fraction_observed": positive_patches / n_samples,
        "negative_fraction_observed": negative_patches / n_samples,
        "intermediate_fraction_observed": intermediate_patches / n_samples,
        "mean_union_damage_pixels": sum(union_pixels) / n_samples,
        "crack_positive_patch_fraction": crack_positive_patches / n_samples,
        "mean_crack_pixels": sum(crack_pixels) / n_samples,
        "patch_fraction_per_class": {
            name: per_class_patches[channel] / n_samples
            for channel, name in enumerate(UNIFIED_DAMAGE_CLASSES)
        },
    }


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config, args.set)
    seed_everything(cfg.project.seed)
    device = get_device()

    assert_unified_damage_taxonomy()

    if int(cfg.model.get("multilabel_classes", 0)) != len(UNIFIED_DAMAGE_CLASSES):
        raise ValueError(
            f"P4-A requires model.multilabel_classes={len(UNIFIED_DAMAGE_CLASSES)}, "
            f"got {cfg.model.get('multilabel_classes')}."
        )
    if int(cfg.model.get("crack_classes", 1)) != 1:
        raise ValueError("P4-A requires model.crack_classes=1.")
    if bool(cfg.data.hard_negative_mining.get("enabled", False)):
        raise ValueError(
            "Hard negative mining is a P4-B variable: set "
            "data.hard_negative_mining.enabled=false for the controlled P4-A run."
        )

    patch_cfg = multilabel_patch_cfg(cfg)

    run_name = args.run_name or f"p4a_{cfg.model.arch}_{cfg.model.encoder}_{patch_cfg.patch_size}"
    run_dir = ensure_dir(Path(cfg.project.output_dir) / "runs" / run_name)

    train_samples = list_samples(cfg.data.dacl10k.root, cfg.data.dacl10k.train_split)
    val_samples = list_samples(cfg.data.dacl10k.root, cfg.data.dacl10k.val_split)

    if args.limit_train:
        train_samples = train_samples[: args.limit_train]
    if args.limit_val:
        val_samples = val_samples[: args.limit_val]

    train_summary = summarize_multilabel_targets(multilabel_sample_targets(train_samples))
    val_summary = summarize_multilabel_targets(multilabel_sample_targets(val_samples))

    print(
        f"[data] train images {train_summary['n_images']} | "
        f"val images {val_summary['n_images']} | patch size {patch_cfg.patch_size} | "
        f"train patches/epoch {len(train_samples) * patch_cfg.patches_per_image}"
    )
    for name in UNIFIED_DAMAGE_CLASSES:
        print(
            f"[data] {name:>13}: train images {train_summary['n_images_per_class'][name]:5d} "
            f"({train_summary['image_fraction_per_class'][name]:6.1%}) | "
            f"val images {val_summary['n_images_per_class'][name]:5d}"
        )

    class_weights = resolve_multilabel_class_weights(
        cfg=cfg,
        train_samples=train_samples,
        run_dir=run_dir,
        recompute=args.recompute_class_weights,
        patch_cfg=patch_cfg,
    )

    train_ds = Dacl10kMultilabelPatchDataset(
        samples=train_samples,
        transform=patch_train_transform_multilabel(),
        patch_size=patch_cfg.patch_size,
        positive_patch_fraction=patch_cfg.positive_patch_fraction,
        min_positive_pixels=patch_cfg.min_positive_pixels,
        max_negative_pixels=patch_cfg.max_negative_pixels,
        max_crop_attempts=patch_cfg.max_crop_attempts,
        patches_per_image=patch_cfg.patches_per_image,
    )
    val_ds = Dacl10kMultilabelCenterPatchDataset(
        samples=val_samples,
        transform=patch_eval_transform_multilabel(),
        patch_size=patch_cfg.patch_size,
    )

    sampling_check = patch_sampling_sanity_check(
        dataset=train_ds,
        n_samples=patch_cfg.sanity_check_samples,
        min_positive_pixels=patch_cfg.min_positive_pixels,
    )
    print(
        f"[sampling] positive {sampling_check['positive_fraction_observed']:.1%} | "
        f"negative {sampling_check['negative_fraction_observed']:.1%} | "
        f"intermediate {sampling_check['intermediate_fraction_observed']:.1%} | "
        f"crack-positive {sampling_check['crack_positive_patch_fraction']:.1%}"
    )
    for name, fraction in sampling_check["patch_fraction_per_class"].items():
        print(f"[sampling] {name:>13}: present in {fraction:6.1%} of the sampled patches")

    expected_positive = float(patch_cfg.positive_patch_fraction)
    if abs(sampling_check["positive_fraction_observed"] - expected_positive) > 0.03:
        print(
            f"[sampling] WARNING: expected about {expected_positive:.1%} positive patches, "
            f"observed {sampling_check['positive_fraction_observed']:.1%}."
        )

    common = loader_kwargs(cfg.data, device)
    train_loader = DataLoader(
        train_ds, batch_size=cfg.train.batch_size, shuffle=True, drop_last=True, **common
    )
    val_loader = DataLoader(val_ds, batch_size=cfg.train.batch_size, shuffle=False, **common)

    model = build_multitask_model(cfg.model)
    report = model.parameter_report()
    print(
        f"[model] P4-A {cfg.model.arch}/{cfg.model.encoder} - "
        f"{report['total'] / 1e6:.1f}M params | shared encoder "
        f"{report['encoder_shared'] / 1e6:.1f}M ({report['shared_fraction']:.1%}) | "
        f"crack decoder {report['crack_decoder'] / 1e6:.1f}M | "
        f"multilabel decoder {report['multilabel_decoder'] / 1e6:.1f}M"
    )

    last_checkpoint_path = run_dir / "last_joint.pt"
    resume_requested = bool(cfg.train.get("resume", False))
    if last_checkpoint_path.exists() and not resume_requested:
        raise RuntimeError(
            "[run] existing last_joint.pt found, but train.resume=false. "
            "Use resume=true, choose a new run name, or delete the run directory."
        )

    save_resolved_run_config(
        run_dir=run_dir,
        cfg=cfg,
        transfer_metadata=None,
        resumed=resume_requested and last_checkpoint_path.exists(),
    )

    criterion = build_multitask_loss(cfg, pos_weight=class_weights["pos_weight"]).to(device)
    selection_metric = str(cfg.train.get("selection_metric", "joint_score"))
    print(f"[fit] checkpoint selection and early stopping on '{selection_metric}'")

    summary = {
        "task": str(patch_cfg.task_name),
        "task_type": "multitask_crack_binary_plus_multilabel_damage",
        "pipeline": "P4-A",
        "heads": {
            "crack": {
                "channels": 1,
                "target": "channel 0 of the unified mask (Crack + ACrack)",
                "comparable_baseline": "P1-B4 (binary DACL10K crack)",
            },
            "multilabel": {
                "channels": len(UNIFIED_DAMAGE_CLASSES),
                "class_names": list(UNIFIED_DAMAGE_CLASSES),
                "comparable_baseline": "P1ML-A (multilabel DACL10K)",
            },
        },
        "shared_encoder": {
            "encoder": str(cfg.model.encoder),
            "weights": str(cfg.model.encoder_weights),
            "parameters": report,
        },
        "class_mapping_19_to_6": unified_damage_label_groups(),
        "official_train_split": str(cfg.data.dacl10k.train_split),
        "official_val_split": str(cfg.data.dacl10k.val_split),
        "train_images": train_summary,
        "validation_images": val_summary,
        "patch_sampling": {
            "patch_size": int(patch_cfg.patch_size),
            "patches_per_image": int(patch_cfg.patches_per_image),
            "positive_patch_fraction": float(patch_cfg.positive_patch_fraction),
            "min_positive_pixels": int(patch_cfg.min_positive_pixels),
            "max_negative_pixels": int(patch_cfg.max_negative_pixels),
            "max_crop_attempts": int(patch_cfg.max_crop_attempts),
            "positive_criterion": "union of the 6 damage channels (same as P1ML)",
        },
        "hard_negative_mining": {"enabled": False},
        "loss": {
            "name": str(cfg.loss.name),
            "crack_weight": float(cfg.loss.get("crack_weight", 1.0)),
            "multilabel_weight": float(cfg.loss.get("multilabel_weight", 1.0)),
            "crack_pos_weight": cfg.loss.crack.get("pos_weight"),
            "multilabel_pos_weight": class_weights["pos_weight"],
            "multilabel_pos_weight_raw": class_weights["pos_weight_raw"],
            "task_weighting_note": "fixed 1.0 / 1.0 in P4-A; tuning belongs to P4-B",
        },
        "selection_metric": selection_metric,
        "sampling_sanity_check": sampling_check,
        "validation_note": (
            "Training-time validation uses one deterministic damage-centred patch per "
            "image as an optimization monitor only. Full-resolution metrics for both "
            "heads are written by scripts/15_evaluate_dacl10k_multitask_sliding.py."
        ),
    }
    with open(run_dir / "dataset_summary.json", "w", encoding="utf-8") as fh:
        json.dump(to_jsonable(summary), fh, indent=2, ensure_ascii=False, sort_keys=False)

    best = fit_multitask(
        model,
        train_loader,
        val_loader,
        criterion,
        cfg,
        device,
        run_dir,
        evaluate_fn=evaluate_multitask,
        selection_metric=selection_metric,
    )

    history_path = run_dir / "history.csv"
    if history_path.exists():
        plot_curves(history_path, run_dir / "curves.png")

    if best:
        print("\n=== Best training-monitor validation metrics (patch level) ===")
        for key in (
            "loss",
            "loss_crack",
            "loss_multilabel",
            "joint_score",
            "crack_dice",
            "crack_iou",
            "macro_dice_present",
            "macro_iou_present",
            "micro_dice",
        ):
            if key in best:
                print(f"{key:>24}: {best[key]:.4f}")
        for name in UNIFIED_DAMAGE_CLASSES:
            key = f"dice_{name}"
            if key in best:
                print(f"{key:>24}: {best[key]:.4f}")
        with open(run_dir / "best_monitor_metrics.json", "w", encoding="utf-8") as fh:
            json.dump(to_jsonable(best), fh, indent=2, ensure_ascii=False)

    print(f"\nP4-A training artifacts in {run_dir.resolve()}")
    print("Next: scripts/15_evaluate_dacl10k_multitask_sliding.py (full-resolution, both heads).")


if __name__ == "__main__":
    main()
