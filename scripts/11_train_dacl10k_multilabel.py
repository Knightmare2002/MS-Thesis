#!/usr/bin/env python
"""Step 11 / P1ML - native-resolution multilabel damage training on DACL10K.

Same experimental protocol as 07_train_dacl10k_patches.py (official split, no
global resize, patch sampling with a positive quota, AMP, SGDR + warm-up, early
stopping, CSV history, curves) applied to 6 independent damage channels:

    0 crack        (Crack, ACrack)
    1 spalling     (Rockpocket, Cavity, Spalling)
    2 corrosion    (Rust, ExposedRebars, WConccor)
    3 moisture     (Wetspot, Efflorescence)
    4 delamination (Hollowareas)
    5 surface      (Graffiti, Weathering, Restformwork)

Sigmoid + per-channel BCE (vector pos_weight) + per-channel Soft Dice. Model
selection and early stopping use macro Dice over the classes present in the
monitor split.

Writes to <output_dir>/runs/<run_name>/:
    config.yaml  dataset_summary.json  class_weights.json  history.csv
    last.pt  best.pt  curves.png  [partial_load_report.json  transfer_metadata.json]

Usage
-----
    # A) from ImageNet
    python scripts/11_train_dacl10k_multilabel.py --run-name p1ml_unetpp_r34_imagenet

    # B) from the P1-B4 binary checkpoint
    python scripts/11_train_dacl10k_multilabel.py --run-name p1ml_unetpp_r34_from_p1b4 \
        --set transfer.init_checkpoint=outputs/runs/p1b4_unetplusplus_r34_minpos256_pw5_hnm/best.pt

    # smoke run
    python scripts/11_train_dacl10k_multilabel.py --limit-train 32 --limit-val 16 \
        --set train.epochs=1 data.num_workers=0 data.p1ml_patch.class_weights.max_images=16
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
from src.data.stats import dacl10k_multilabel_pixel_stats, multilabel_pos_weights
from src.data.transforms import (
    patch_eval_transform_multilabel,
    patch_train_transform_multilabel,
)

from src.engine import fit, initialize_model_from_checkpoint_partial

from src.eval.multilabel_metrics import evaluate_multilabel

from src.losses import build_multilabel_loss

from src.models.unet import build_model, count_parameters

from src.provenance import save_resolved_run_config, save_transfer_metadata, to_jsonable

from src.utils import ensure_dir, get_device, load_config, loader_kwargs, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train P1ML DACL10K multilabel damage segmentation on native-resolution patches."
    )
    parser.add_argument("--config", default="configs/config_p1ml.yaml")
    parser.add_argument("--set", nargs="*", default=[], help="config overrides key=value")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--limit-train", type=int, default=None)
    parser.add_argument("--limit-val", type=int, default=None)
    parser.add_argument(
        "--recompute-class-weights",
        action="store_true",
        help="ignore the cached pos_weight vector and recompute it from the train split",
    )
    return parser.parse_args()


def plot_curves(history_path: Path, out_path: Path) -> None:
    """Save loss and validation overlap curves (macro on present classes + micro)."""
    history = pd.read_csv(history_path)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].plot(history.epoch, history.train_loss, label="train")
    axes[0].plot(history.epoch, history.val_loss, label="val patches")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("BCE + Dice loss (multilabel)")
    axes[0].set_title("P1ML patch loss")
    axes[0].legend()

    if "val_macro_dice_present" in history:
        axes[1].plot(history.epoch, history.val_macro_dice_present, label="macro Dice (present)")
    if "val_macro_iou_present" in history:
        axes[1].plot(history.epoch, history.val_macro_iou_present, label="macro IoU (present)")
    axes[1].plot(history.epoch, history.val_dice, label="micro Dice", linestyle="--")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("score")
    axes[1].set_title("P1ML validation patches")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def patch_sampling_sanity_check(dataset, n_samples: int, min_positive_pixels: int) -> dict:
    """Measure multilabel patch composition on indices sampled across the dataset.

    `positive` is defined on the union of the 6 channels, exactly like the
    sampler; per-channel occupancy is also reported because a positive quota on
    the union can still starve the rare channels.
    """
    n_samples = min(int(n_samples), len(dataset))
    indices = np.linspace(0, len(dataset) - 1, num=n_samples, dtype=int)

    n_classes = len(UNIFIED_DAMAGE_CLASSES)
    union_pixels: list[int] = []
    per_class_patches = [0] * n_classes
    positive_patches = negative_patches = intermediate_patches = 0

    for index in indices:
        _, mask = dataset[int(index)]
        union = (mask.amax(dim=0) > 0.5)
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

    return {
        "n_samples": n_samples,
        "positive_patches": positive_patches,
        "negative_patches": negative_patches,
        "intermediate_patches": intermediate_patches,
        "positive_fraction_observed": positive_patches / n_samples,
        "negative_fraction_observed": negative_patches / n_samples,
        "intermediate_fraction_observed": intermediate_patches / n_samples,
        "mean_union_damage_pixels": sum(union_pixels) / n_samples,
        "min_union_damage_pixels": min(union_pixels),
        "max_union_damage_pixels": max(union_pixels),
        "patch_fraction_per_class": {
            name: per_class_patches[channel] / n_samples
            for channel, name in enumerate(UNIFIED_DAMAGE_CLASSES)
        },
    }


def resolve_class_weights(cfg, train_samples, run_dir: Path, recompute: bool) -> dict:
    """Load the cached pos_weight vector or estimate it on the official train split."""
    weights_cfg = cfg.data.p1ml_patch.class_weights
    cache_path = Path(str(weights_cfg.cache_path))
    max_images = weights_cfg.get("max_images")

    if cache_path.is_file() and not recompute:
        with open(cache_path, encoding="utf-8") as fh:
            payload = json.load(fh)
        print(f"[class-weights] reusing cached vector from {cache_path}")
    else:
        print("[class-weights] estimating per-channel pixel frequency on the train split...")
        stats = dacl10k_multilabel_pixel_stats(
            train_samples,
            max_images=int(max_images) if max_images else None,
            seed=int(cfg.project.seed),
        )
        payload = multilabel_pos_weights(
            stats,
            clip_min=float(weights_cfg.clip_min),
            clip_max=float(weights_cfg.clip_max),
            power=float(weights_cfg.power)
        )
        payload["pixel_statistics"] = stats
        ensure_dir(cache_path.parent)
        with open(cache_path, "w", encoding="utf-8") as fh:
            json.dump(to_jsonable(payload), fh, indent=2, ensure_ascii=False)
        print(f"[class-weights] cached to {cache_path}")

    if list(payload["class_names"]) != list(UNIFIED_DAMAGE_CLASSES):
        raise RuntimeError(
            "[class-weights] cached channel order does not match UNIFIED_DAMAGE_CLASSES: "
            f"{payload['class_names']}"
        )

    # A copy always lives next to the checkpoints, so a run is self-contained.
    with open(run_dir / "class_weights.json", "w", encoding="utf-8") as fh:
        json.dump(to_jsonable(payload), fh, indent=2, ensure_ascii=False)

    for name, raw, clipped in zip(
        payload["class_names"], payload["pos_weight_raw"], payload["pos_weight"]
    ):
        print(f"[class-weights] {name:>13}: pos_weight {clipped:7.3f} (raw {raw:10.1f})")

    return payload


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config, args.set)
    seed_everything(cfg.project.seed)
    device = get_device()

    assert_unified_damage_taxonomy()

    if int(cfg.model.classes) != len(UNIFIED_DAMAGE_CLASSES):
        raise ValueError(
            f"P1ML requires model.classes={len(UNIFIED_DAMAGE_CLASSES)}, got {cfg.model.classes}."
        )
    if bool(cfg.data.hard_negative_mining.get("enabled", False)):
        raise ValueError(
            "Hard negative mining is not supported by the first P1ML pipeline: "
            "set data.hard_negative_mining.enabled=false."
        )

    run_name = args.run_name or (
        f"p1ml_{cfg.model.arch}_{cfg.model.encoder}_{cfg.data.p1ml_patch.patch_size}"
    )
    run_dir = ensure_dir(Path(cfg.project.output_dir) / "runs" / run_name)

    patch_cfg = cfg.data.p1ml_patch

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
        f"val images {val_summary['n_images']} | "
        f"patch size {patch_cfg.patch_size} | "
        f"train patches/epoch {len(train_samples) * patch_cfg.patches_per_image}"
    )
    for name in UNIFIED_DAMAGE_CLASSES:
        print(
            f"[data] {name:>13}: train images {train_summary['n_images_per_class'][name]:5d} "
            f"({train_summary['image_fraction_per_class'][name]:6.1%}) | "
            f"val images {val_summary['n_images_per_class'][name]:5d}"
        )

    class_weights = resolve_class_weights(cfg, train_samples, run_dir, args.recompute_class_weights)

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
        f"mean union damage pixels {sampling_check['mean_union_damage_pixels']:.1f}"
    )
    for name, fraction in sampling_check["patch_fraction_per_class"].items():
        print(f"[sampling] {name:>13}: present in {fraction:6.1%} of the sampled patches")

    expected_positive = float(patch_cfg.positive_patch_fraction)
    observed_positive = sampling_check["positive_fraction_observed"]
    if abs(observed_positive - expected_positive) > 0.03:
        print(
            f"[sampling] WARNING: expected about {expected_positive:.1%} positive patches, "
            f"observed {observed_positive:.1%}."
        )

    summary = {
        "task": str(patch_cfg.task_name),
        "task_type": "multilabel_damage_segmentation",
        "n_classes": len(UNIFIED_DAMAGE_CLASSES),
        "class_names": list(UNIFIED_DAMAGE_CLASSES),
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
            "positive_criterion": "union of the 6 damage channels",
        },
        "hard_negative_mining": {"enabled": False},
        "class_weights": {
            "pos_weight": class_weights["pos_weight"],
            "pos_weight_raw": class_weights["pos_weight_raw"],
            "clip_min": class_weights["clip_min"],
            "clip_max": class_weights["clip_max"],
            "estimated_on": "official DACL10K train split only",
        },
        "selection_metric": str(cfg.train.get("selection_metric", "macro_dice_present")),
        "sampling_sanity_check": sampling_check,
        "validation_note": (
            "Training-time validation uses one deterministic damage-centred patch per "
            "image as an optimization monitor only. Full-image multilabel metrics are "
            "written by scripts/12_evaluate_dacl10k_multilabel_sliding.py."
        ),
    }
    with open(run_dir / "dataset_summary.json", "w", encoding="utf-8") as fh:
        json.dump(to_jsonable(summary), fh, indent=2, ensure_ascii=False, sort_keys=False)

    common = loader_kwargs(cfg.data, device)
    train_loader = DataLoader(
        train_ds, batch_size=cfg.train.batch_size, shuffle=True, drop_last=True, **common
    )
    val_loader = DataLoader(val_ds, batch_size=cfg.train.batch_size, shuffle=False, **common)

    model = build_model(cfg.model)

    transfer_cfg = cfg.get("transfer", {})
    transfer_metadata: dict | None = None

    last_checkpoint_path = run_dir / "last.pt"
    resume_requested = bool(cfg.train.get("resume", False))
    last_checkpoint_exists = last_checkpoint_path.exists()

    if last_checkpoint_exists and not resume_requested:
        raise RuntimeError(
            "[run] existing last.pt found, but train.resume=false. "
            "Use resume=true, choose a new run name, or delete the run directory."
        )

    local_resume_exists = resume_requested and last_checkpoint_exists

    if local_resume_exists:
        print(
            "[transfer] existing local last.pt found: fit() will resume this P1ML run; "
            "external initialization is skipped."
        )
    elif transfer_cfg.get("init_checkpoint"):
        transfer_metadata = initialize_model_from_checkpoint_partial(
            model=model,
            checkpoint_path=transfer_cfg["init_checkpoint"],
            device=device,
        )
        with open(run_dir / "partial_load_report.json", "w", encoding="utf-8") as fh:
            json.dump(to_jsonable(transfer_metadata), fh, indent=2, ensure_ascii=False)
        print(f"[transfer] partial-load report saved to {run_dir / 'partial_load_report.json'}")
    else:
        print(
            "[transfer] no init_checkpoint configured: training starts from the "
            f"model factory (encoder_weights={cfg.model.encoder_weights})."
        )

    save_transfer_metadata(run_dir=run_dir, cfg=cfg, metadata=transfer_metadata)
    save_resolved_run_config(
        run_dir=run_dir, cfg=cfg, transfer_metadata=transfer_metadata, resumed=local_resume_exists
    )

    total, trainable = count_parameters(model)
    print(
        f"[model] {cfg.model.arch}/{cfg.model.encoder} - {total / 1e6:.1f}M params "
        f"({trainable / 1e6:.1f}M trainable) - {cfg.model.classes} output channels"
    )

    criterion = build_multilabel_loss(cfg, pos_weight=class_weights["pos_weight"]).to(device)

    selection_metric = str(cfg.train.get("selection_metric", "macro_dice_present"))
    print(f"[fit] checkpoint selection and early stopping on '{selection_metric}'")

    best = fit(
        model,
        train_loader,
        val_loader,
        criterion,
        cfg,
        device,
        run_dir,
        evaluate_fn=evaluate_multilabel,
        selection_metric=selection_metric,
        extra_history_keys=(
            "macro_dice_present",
            "macro_iou_present",
            "macro_recall_present",
            "micro_dice",
            "n_classes_present",
        )
        + tuple(f"dice_{name}" for name in UNIFIED_DAMAGE_CLASSES),
    )

    history_path = run_dir / "history.csv"
    if history_path.exists():
        plot_curves(history_path, run_dir / "curves.png")

    if best:
        print("\n=== Best training-monitor validation metrics (patch level) ===")
        for key in ("loss", "macro_dice_present", "macro_iou_present", "micro_dice"):
            if key in best:
                print(f"{key:>28}: {best[key]:.4f}")
        for name in UNIFIED_DAMAGE_CLASSES:
            key = f"dice_{name}"
            if key in best:
                print(f"{key:>28}: {best[key]:.4f}")
        with open(run_dir / "best_monitor_metrics.json", "w", encoding="utf-8") as fh:
            json.dump(to_jsonable(best), fh, indent=2, ensure_ascii=False)

    print(f"\nP1ML training artifacts in {run_dir.resolve()}")
    print("Run scripts/12_evaluate_dacl10k_multilabel_sliding.py for full-resolution validation.")


if __name__ == "__main__":
    main()
