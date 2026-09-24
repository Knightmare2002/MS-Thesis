#!/usr/bin/env python
"""Step 15 / P4-A - full-resolution sliding-window evaluation of both heads.

Protocol identical to scripts/08 (binary) and scripts/12 (multilabel): native
resolution, no global resize, 512 windows with stride 256, Gaussian blending,
standard threshold 0.5 plus the global sweep [0.3, 0.4, 0.5, 0.6, 0.7]. Both
heads are evaluated in the same pass over the images, with a single encoder
forward per window, so the numbers are exactly those of the joint model.

Writes to <run_dir>/eval_multitask_sliding/:
    metrics_dacl10k_val_crack_sliding.csv               (crack head, one row per threshold)
    metrics_dacl10k_val_multilabel_sliding.csv          (multilabel aggregate, one row per threshold)
    metrics_dacl10k_val_multilabel_per_class.csv        (one row per threshold x class)
    validation_composition.json
    qualitative_overview.png                            (compact, all classes at a glance)
    qualitative_crack.png
    qualitative_multilabel_<class>.png                  (one file per damage class)

The two multilabel CSVs keep the exact names and schema of scripts/12, so the
P1ML-A vs P4-A comparison and scripts/17 read them identically.

Usage
-----
    python scripts/15_evaluate_dacl10k_multitask_sliding.py \
        --run-dir outputs/runs/p4a_unetpp_r34_shared_encoder
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.data.class_mapping import UNIFIED_DAMAGE_CLASSES, unified_damage_label_groups
from src.data.dacl10k import (
    list_samples,
    load_annotation,
    multilabel_sample_targets,
    rasterize_unified_damage,
    summarize_multilabel_targets,
)
from src.data.transforms import IMAGENET_MEAN, IMAGENET_STD
from src.engine import load_checkpoint
from src.eval.metrics import SegmentationMetrics
from src.eval.multilabel_metrics import MultilabelSegmentationMetrics
from src.eval.sliding_window import predict_sliding_window_multitask
from src.models.multitask import build_multitask_model
from src.utils import ensure_dir, get_device, load_config, multilabel_patch_cfg, seed_everything

# Error-coded overlay, identical for every panel of every figure: the reader
# never has to guess which class "won" a pixel, because each panel shows exactly
# one channel. No priority-based compositing is performed anywhere.
OVERLAY_COLORS = {
    "true_positive": (40, 200, 90),
    "false_positive": (235, 60, 50),
    "false_negative": (60, 120, 255),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Full-resolution sliding-window evaluation of a P4-A multitask run."
    )
    parser.add_argument("--run-dir", required=True)
    parser.add_argument(
        "--checkpoint",
        default="best_joint.pt",
        help="checkpoint inside --run-dir (default: best_joint.pt)",
    )
    parser.add_argument("--split", default=None, help="DACL10K split (default: the run's val split)")
    parser.add_argument("--limit", type=int, default=None, help="debug: evaluate only N images")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--skip-figures",
        action="store_true",
        help="metrics only (the qualitative figures require a second inference pass)",
    )
    return parser.parse_args()


def load_rgb_and_masks(sample) -> tuple[np.ndarray, np.ndarray]:
    """Return the native-resolution RGB image and its [6,H,W] unified damage mask.

    The crack target is channel 0 of this very mask (Crack + ACrack): the same
    single source of truth used during training, so the crack head is scored on
    exactly what it was optimised for.
    """
    image_path, annotation_path = sample

    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise RuntimeError(f"Unreadable image: {image_path}")
    image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    annotation = load_annotation(annotation_path)
    mask = rasterize_unified_damage(annotation, shape=image.shape[:2]).astype(np.float32)
    return image, mask


def overlay_error_map(image: np.ndarray, ground_truth: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    """Dim the image and paint TP / FP / FN of ONE binary channel."""
    canvas = (0.35 * image.astype(np.float32)).astype(np.uint8)

    gt = ground_truth > 0.5
    pred = prediction > 0.5

    canvas[gt & pred] = OVERLAY_COLORS["true_positive"]
    canvas[~gt & pred] = OVERLAY_COLORS["false_positive"]
    canvas[gt & ~pred] = OVERLAY_COLORS["false_negative"]
    return canvas


def select_qualitative_indices(targets: list[list[int]], n_samples: int) -> list[int]:
    """Deterministically pick images maximising damage-class diversity (as in scripts/12)."""
    ranked = sorted(range(len(targets)), key=lambda index: -sum(targets[index]))
    chosen: list[int] = []
    covered: set[int] = set()

    for index in ranked:
        new = {
            channel
            for channel, flag in enumerate(targets[index])
            if flag and channel not in covered
        }
        if new or len(chosen) < 2:
            chosen.append(index)
            covered |= {channel for channel, flag in enumerate(targets[index]) if flag}
        if len(chosen) >= n_samples:
            break

    for index in (i for i, row in enumerate(targets) if not any(row)):
        if len(chosen) >= n_samples:
            break
        chosen.append(index)

    return chosen[:n_samples]


@torch.no_grad()
def predict_one(model, image, patch_cfg, device) -> tuple[np.ndarray, np.ndarray]:
    """Joint sliding-window inference for one image: (crack [H,W], multilabel [C,H,W])."""
    crack, multilabel = predict_sliding_window_multitask(
        model=model,
        image=image,
        device=device,
        patch_size=int(patch_cfg.patch_size),
        stride=int(patch_cfg.eval_stride),
        batch_size=int(patch_cfg.eval_batch_size),
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD,
        n_classes=len(UNIFIED_DAMAGE_CLASSES),
        blend_mode=str(patch_cfg.blend_mode),
    )
    return crack.numpy(), multilabel.numpy()


@torch.no_grad()
def evaluate_sliding_multitask(model, samples, cfg, device, patch_cfg):
    """Both heads, every threshold of the sweep, one pass over the images."""
    thresholds = list(cfg.eval.threshold_sweep)
    if float(cfg.eval.threshold) not in thresholds:
        thresholds = sorted(thresholds + [float(cfg.eval.threshold)])

    crack_meters = {t: SegmentationMetrics(threshold=t) for t in thresholds}
    multilabel_meters = {t: MultilabelSegmentationMetrics(threshold=t) for t in thresholds}

    for index, sample in enumerate(samples, start=1):
        image, mask = load_rgb_and_masks(sample)
        crack_probability, multilabel_probability = predict_one(model, image, patch_cfg, device)

        # The blended outputs are probabilities: invert the sigmoid so both meters
        # keep the single thresholding code path they use everywhere else.
        crack_logits = torch.logit(
            torch.from_numpy(crack_probability).clamp(1e-6, 1.0 - 1e-6)
        ).unsqueeze(0).unsqueeze(0)
        multilabel_logits = torch.logit(
            torch.from_numpy(multilabel_probability).clamp(1e-6, 1.0 - 1e-6)
        ).unsqueeze(0)

        crack_target = torch.from_numpy(mask[0:1]).unsqueeze(0)
        multilabel_target = torch.from_numpy(mask).unsqueeze(0)

        for threshold in thresholds:
            crack_meters[threshold].update(crack_logits, crack_target)
            multilabel_meters[threshold].update(multilabel_logits, multilabel_target)

        if index % 25 == 0 or index == len(samples):
            print(f"[eval] processed {index}/{len(samples)} images")

    crack_rows = [
        {"threshold": threshold, **meter.compute()} for threshold, meter in crack_meters.items()
    ]

    summary_rows, per_class_rows = [], []
    for threshold, meter in multilabel_meters.items():
        aggregated = meter.compute()
        summary_rows.append(
            {
                "threshold": threshold,
                **{
                    key: value
                    for key, value in aggregated.items()
                    if not any(key.endswith(f"_{name}") for name in UNIFIED_DAMAGE_CLASSES)
                },
            }
        )
        for name, values in meter.per_class().items():
            per_class_rows.append({"threshold": threshold, "class": name, **values})

    crack = pd.DataFrame(crack_rows).sort_values("threshold").reset_index(drop=True)
    summary = pd.DataFrame(summary_rows).sort_values("threshold").reset_index(drop=True)
    per_class = (
        pd.DataFrame(per_class_rows).sort_values(["threshold", "class"]).reset_index(drop=True)
    )
    return crack, summary, per_class


@torch.no_grad()
def save_qualitative_figures(model, samples, targets, cfg, device, patch_cfg, eval_dir) -> None:
    """Compact overview + one readable file per damage class (+ one for the crack head).

    Deliberately *not* a priority-based composite: every panel shows a single
    channel, colour-coded TP / FP / FN, so a false positive of `surface` can never
    hide a missed `crack`. Inference is run once per selected image and reused by
    all the figures.
    """
    threshold = float(cfg.eval.threshold)
    indices = select_qualitative_indices(targets, int(cfg.eval.n_qualitative_samples))
    if not indices:
        raise RuntimeError("No sample available for qualitative evaluation.")

    cached = []
    for index in indices:
        image, mask = load_rgb_and_masks(samples[index])
        crack_probability, multilabel_probability = predict_one(model, image, patch_cfg, device)
        cached.append((image, mask, crack_probability, multilabel_probability))

    legend = [
        mpatches.Patch(color=np.array(color) / 255.0, label=label.replace("_", " "))
        for label, color in OVERLAY_COLORS.items()
    ]

    # ---- compact overview: rows = images, columns = image + crack + 6 classes ----
    n_columns = 2 + len(UNIFIED_DAMAGE_CLASSES)
    fig, axes = plt.subplots(
        len(cached), n_columns, figsize=(2.05 * n_columns, 2.05 * len(cached))
    )
    axes = np.atleast_2d(axes)

    for row, (image, mask, crack_probability, multilabel_probability) in enumerate(cached):
        panels = [(image, "image")]
        panels.append(
            (
                overlay_error_map(image, mask[0], crack_probability > threshold),
                "crack head",
            )
        )
        for channel, name in enumerate(UNIFIED_DAMAGE_CLASSES):
            panels.append(
                (
                    overlay_error_map(
                        image, mask[channel], multilabel_probability[channel] > threshold
                    ),
                    f"ml: {name}",
                )
            )

        for column, (content, title) in enumerate(panels):
            axes[row, column].imshow(content)
            axes[row, column].set_axis_off()
            if row == 0:
                axes[row, column].set_title(title, fontsize=8)

    fig.suptitle(
        f"P4-A validation overview @ threshold {threshold:.2f} - one channel per panel",
        fontsize=11,
    )
    fig.legend(handles=legend, loc="lower center", ncol=3, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0.035, 1, 0.965))
    fig.savefig(eval_dir / "qualitative_overview.png", dpi=150)
    plt.close(fig)

    # ---- one file per channel: image | ground truth | prediction | error map ----
    def save_channel_figure(name: str, gt_getter, prediction_getter, out_path: Path) -> None:
        fig, axes = plt.subplots(len(cached), 4, figsize=(13.5, 3.4 * len(cached)))
        axes = np.atleast_2d(axes)

        for row, item in enumerate(cached):
            image = item[0]
            ground_truth = gt_getter(item)
            prediction = prediction_getter(item)
            panels = [
                (image, "image"),
                (ground_truth, "ground truth"),
                (prediction, f"prediction @ {threshold:.2f}"),
                (overlay_error_map(image, ground_truth, prediction), "TP / FP / FN"),
            ]
            for column, (content, title) in enumerate(panels):
                if content.ndim == 2:
                    axes[row, column].imshow(content, cmap="gray", vmin=0, vmax=1)
                else:
                    axes[row, column].imshow(content)
                axes[row, column].set_axis_off()
                if row == 0:
                    axes[row, column].set_title(title, fontsize=10)

        fig.suptitle(f"P4-A - {name}", fontsize=12)
        fig.legend(handles=legend, loc="lower center", ncol=3, frameon=False, fontsize=9)
        fig.tight_layout(rect=(0, 0.035, 1, 0.97))
        fig.savefig(out_path, dpi=150)
        plt.close(fig)

    save_channel_figure(
        name="crack head (binary)",
        gt_getter=lambda item: item[1][0],
        prediction_getter=lambda item: (item[2] > threshold).astype(np.float32),
        out_path=eval_dir / "qualitative_crack.png",
    )

    for channel, name in enumerate(UNIFIED_DAMAGE_CLASSES):
        save_channel_figure(
            name=f"multilabel head - {name}",
            gt_getter=lambda item, c=channel: item[1][c],
            prediction_getter=lambda item, c=channel: (item[3][c] > threshold).astype(np.float32),
            out_path=eval_dir / f"qualitative_multilabel_{name}.png",
        )


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    cfg = load_config(run_dir / "config.yaml")
    seed_everything(cfg.project.seed)
    device = get_device()

    if int(cfg.model.get("multilabel_classes", 0)) != len(UNIFIED_DAMAGE_CLASSES):
        raise ValueError(
            "This run directory does not describe a P4 multitask run "
            "(model.multilabel_classes missing or != 6). Use scripts/12 for P1ML runs."
        )

    patch_cfg = multilabel_patch_cfg(cfg)
    split = args.split or str(cfg.data.dacl10k.val_split)
    samples = list_samples(cfg.data.dacl10k.root, split)
    if args.limit:
        samples = samples[: args.limit]

    targets = multilabel_sample_targets(samples)
    composition = summarize_multilabel_targets(targets)

    model = build_multitask_model(cfg.model).to(device)

    checkpoint_path = run_dir / args.checkpoint
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = load_checkpoint(checkpoint_path, model, device=device)
    checkpoint_epoch = checkpoint.get("epoch", "unknown")
    checkpoint_best_metric = checkpoint.get("best_dice", None)

    print("\n=== Checkpoint used for the P4-A sliding-window evaluation ===")
    print(f"checkpoint path : {checkpoint_path.resolve()}")
    print(f"saved epoch     : {checkpoint_epoch}")
    if checkpoint_best_metric is not None:
        print(f"best monitor    : {float(checkpoint_best_metric):.4f} "
              f"({cfg.train.get('selection_metric', 'joint_score')})")
    if checkpoint_path.name != "best_joint.pt":
        print(
            "[warning] This is not best_joint.pt. Do not report these metrics as the "
            "official P4-A result unless this choice is intentional."
        )
    print("==============================================================\n")

    print(
        f"[eval] DACL10K {split}: {composition['n_images']} images | "
        f"patch {patch_cfg.patch_size} | stride {patch_cfg.eval_stride} | "
        f"blend {patch_cfg.blend_mode} | heads: crack(1) + multilabel({len(UNIFIED_DAMAGE_CLASSES)})"
    )

    crack, summary, per_class = evaluate_sliding_multitask(model, samples, cfg, device, patch_cfg)

    for frame in (crack, summary, per_class):
        frame.insert(0, "dataset", f"dacl10k_{split}_multitask_sliding")
        frame.insert(1, "patch_size", int(patch_cfg.patch_size))
        frame.insert(2, "stride", int(patch_cfg.eval_stride))
        frame.insert(3, "blend_mode", str(patch_cfg.blend_mode))
        frame.insert(4, "checkpoint_name", checkpoint_path.name)
        frame.insert(5, "checkpoint_epoch", checkpoint_epoch)

    eval_dir = ensure_dir(
        Path(args.output_dir) if args.output_dir is not None else run_dir / "eval_multitask_sliding"
    )
    crack.to_csv(eval_dir / "metrics_dacl10k_val_crack_sliding.csv", index=False)
    summary.to_csv(eval_dir / "metrics_dacl10k_val_multilabel_sliding.csv", index=False)
    per_class.to_csv(eval_dir / "metrics_dacl10k_val_multilabel_per_class.csv", index=False)

    with open(eval_dir / "validation_composition.json", "w", encoding="utf-8") as fh:
        json.dump(
            {
                "pipeline": "P4-A",
                "split": split,
                "class_names": list(UNIFIED_DAMAGE_CLASSES),
                "class_mapping_19_to_6": unified_damage_label_groups(),
                "crack_target": "channel 0 of the unified mask (Crack + ACrack)",
                "inference": {
                    "patch_size": int(patch_cfg.patch_size),
                    "stride": int(patch_cfg.eval_stride),
                    "batch_size": int(patch_cfg.eval_batch_size),
                    "blend_mode": str(patch_cfg.blend_mode),
                    "shared_encoder_forward_passes_per_window": 1,
                },
                "checkpoint": {
                    "path": str(checkpoint_path),
                    "name": checkpoint_path.name,
                    "epoch": int(checkpoint_epoch) if checkpoint_epoch != "unknown" else None,
                    "best_monitor_metric": (
                        float(checkpoint_best_metric) if checkpoint_best_metric is not None else None
                    ),
                    "selection_metric": str(cfg.train.get("selection_metric", "joint_score")),
                },
                **composition,
            },
            fh,
            indent=2,
            ensure_ascii=False,
        )

    if not args.skip_figures:
        save_qualitative_figures(
            model=model,
            samples=samples,
            targets=targets,
            cfg=cfg,
            device=device,
            patch_cfg=patch_cfg,
            eval_dir=eval_dir,
        )

    print("\n--- crack head (binary, full resolution) ---")
    print(
        crack[
            ["threshold", "dice", "iou", "precision", "recall", "false_alarm_rate_empty_gt"]
        ].to_string(index=False)
    )
    print("\n--- multilabel head (6 channels, full resolution) ---")
    print(
        summary[
            [
                "threshold",
                "macro_dice_present",
                "macro_iou_present",
                "micro_dice",
                "macro_precision_present",
                "macro_recall_present",
                "n_classes_present",
            ]
        ].to_string(index=False)
    )
    print()
    print(
        per_class[per_class.threshold == float(cfg.eval.threshold)][
            ["class", "dice", "iou", "precision", "recall", "support_pixels", "n_images_present"]
        ].to_string(index=False)
    )
    print(f"\nP4-A evaluation artifacts in {eval_dir.resolve()}")
    print(
        "Optional, additive: scripts/17_calibrate_multilabel_thresholds.py "
        f"--run-dir {run_dir} (per-channel thresholds, standard CSVs untouched)."
    )


if __name__ == "__main__":
    main()
