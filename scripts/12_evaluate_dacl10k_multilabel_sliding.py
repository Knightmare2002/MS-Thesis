#!/usr/bin/env python
"""Step 12 / P1ML - full-resolution multilabel DACL10K evaluation with sliding windows.

Same protocol as 08_evaluate_dacl10k_sliding.py (native resolution, no global
resize, Gaussian-blended overlapping patches, threshold sweep) applied to the 6
unified damage channels.

Writes to <run_dir>/eval_multilabel_sliding/:
    metrics_dacl10k_val_multilabel_sliding.csv        (macro/micro per threshold)
    metrics_dacl10k_val_multilabel_per_class.csv      (one row per threshold x class)
    validation_composition.json
    qualitative_dacl10k_val_multilabel_sliding.png

Usage
-----
    python scripts/12_evaluate_dacl10k_multilabel_sliding.py \
        --run-dir outputs/runs/p1ml_unetpp_r34_imagenet
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
from src.eval.multilabel_metrics import MultilabelSegmentationMetrics
from src.eval.sliding_window import predict_sliding_window_multilabel
from src.models.unet import build_model
from src.utils import ensure_dir, get_device, load_config, seed_everything

# One colour per unified damage class, used consistently in the overlays.
CLASS_COLORS = {
    "crack": (255, 0, 0),
    "spalling": (255, 128, 0),
    "corrosion": (255, 0, 255),
    "moisture": (0, 128, 255),
    "delamination": (0, 255, 255),
    "surface": (0, 255, 0),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Full-resolution multilabel sliding-window evaluation of a P1ML run."
    )
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--checkpoint", default="best.pt")
    parser.add_argument("--split", default=None, help="DACL10K split (default: the run's val split)")
    parser.add_argument("--limit", type=int, default=None, help="debug: evaluate only N images")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for evaluation artifacts. Defaults to <run-dir>/eval_multilabel_sliding.",
    )
    return parser.parse_args()


def load_rgb_and_multilabel_mask(sample) -> tuple[np.ndarray, np.ndarray]:
    """Load original-resolution RGB image and the aligned [6,H,W] damage mask."""
    image_path, annotation_path = sample

    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise RuntimeError(f"Unreadable image: {image_path}")
    image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    annotation = load_annotation(annotation_path)
    mask = rasterize_unified_damage(annotation, shape=image.shape[:2]).astype(np.float32)

    return image, mask


def select_qualitative_indices_per_class(
    samples,
    n_per_class: int = 2,
) -> dict[str, list[int]]:
    """Select up to n_per_class GT-positive images for each damage channel.

    Selection is deterministic and prioritises images containing the largest
    annotated area of the corresponding class. This guarantees that each
    per-class qualitative figure shows meaningful GT evidence rather than
    arbitrary samples selected by global label diversity.
    """
    if n_per_class < 1:
        raise ValueError("n_per_class must be >= 1.")

    selected: dict[str, list[int]] = {}

    for channel, class_name in enumerate(UNIFIED_DAMAGE_CLASSES):
        candidates: list[tuple[float, int]] = []

        for index, sample in enumerate(samples):
            _, mask = load_rgb_and_multilabel_mask(sample)
            n_positive_pixels = float(mask[channel].sum())

            if n_positive_pixels > 0:
                candidates.append((n_positive_pixels, index))

        candidates.sort(key=lambda item: (-item[0], item[1]))
        selected[class_name] = [
            index for _, index in candidates[:n_per_class]
        ]

    return selected

def _overlay_single_channel(
    image: np.ndarray,
    mask: np.ndarray,
    color: tuple[int, int, int],
    alpha: float = 0.50,
) -> np.ndarray:
    """Overlay one binary channel on an RGB image without hiding other channels.

    The caller invokes this function separately for each multilabel channel,
    therefore overlapping labels remain visible in their respective panels.
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected RGB image [H,W,3], got {image.shape}.")
    if mask.shape != image.shape[:2]:
        raise ValueError(
            f"Mask/image spatial mismatch: mask {mask.shape}, image {image.shape[:2]}."
        )

    overlay = image.copy()
    positive = mask > 0.5
    color_array = np.asarray(color, dtype=np.float32)

    overlay[positive] = (
        (1.0 - alpha) * overlay[positive].astype(np.float32)
        + alpha * color_array
    ).astype(np.uint8)

    return overlay


def _save_class_qualitative_figure(
    model,
    samples,
    class_name: str,
    sample_indices: list[int],
    cfg,
    device,
    threshold: float,
    out_path: Path,
) -> None:
    """Save one compact qualitative figure for one multilabel channel."""
    if not sample_indices:
        return

    channel = list(UNIFIED_DAMAGE_CLASSES).index(class_name)
    patch_cfg = cfg.data.p1ml_patch
    color = CLASS_COLORS[class_name]

    fig, axes = plt.subplots(
        len(sample_indices),
        3,
        figsize=(14, 4.6 * len(sample_indices)),
        squeeze=False,
    )

    for row, sample_index in enumerate(sample_indices):
        image, mask = load_rgb_and_multilabel_mask(samples[sample_index])

        probability = predict_sliding_window_multilabel(
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
        ).numpy()

        prediction = (probability[channel] > threshold).astype(np.float32)

        gt_mask = mask[channel]
        gt_overlay = _overlay_single_channel(
            image=image,
            mask=gt_mask,
            color=color,
            alpha=0.50,
        )
        pred_overlay = _overlay_single_channel(
            image=image,
            mask=prediction,
            color=color,
            alpha=0.50,
        )

        gt_pixels = int(gt_mask.sum())
        pred_pixels = int(prediction.sum())

        panels = [
            (image, "RGB image"),
            (gt_overlay, f"Ground truth — {class_name}"),
            (pred_overlay, f"Prediction — {class_name}"),
        ]

        for column, (content, title) in enumerate(panels):
            axes[row, column].imshow(content)
            axes[row, column].set_axis_off()

            if row == 0:
                axes[row, column].set_title(title, fontsize=12)

        axes[row, 0].text(
            0.01,
            0.98,
            f"Sample {sample_index}\nGT pixels: {gt_pixels:,}\nPredicted pixels: {pred_pixels:,}",
            transform=axes[row, 0].transAxes,
            ha="left",
            va="top",
            color="white",
            fontsize=10,
            fontweight="bold",
            bbox={
                "facecolor": "black",
                "alpha": 0.65,
                "edgecolor": "none",
                "pad": 3,
            },
        )

    color_patch = mpatches.Patch(
        color=np.asarray(color, dtype=np.float32) / 255.0,
        label=class_name,
    )
    fig.legend(
        handles=[color_patch],
        loc="lower center",
        frameon=False,
        fontsize=11,
    )

    fig.suptitle(
        f"DACL10K multilabel qualitative evaluation — {class_name}",
        fontsize=15,
        y=0.99,
    )
    fig.tight_layout(rect=(0.0, 0.04, 1.0, 0.96))
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


@torch.no_grad()
def save_qualitative_figure(
    model,
    samples,
    targets,
    cfg,
    device,
    threshold,
    out_path,
) -> None:
    """Save a compact overview plus one qualitative figure per damage class.

    Outputs:
    - qualitative_overview_multilabel.png:
      6 rows x 3 columns, one representative image per class.
    - qualitative_per_class/qualitative_<class>.png:
      1-2 selected GT-positive examples for each individual class.

    No priority-based channel compositing is used: each class is visualised
    independently, preserving multilabel semantics.
    """
    del targets  # Selection is now based on native-resolution GT pixel support.

    out_path = Path(out_path)
    out_dir = out_path.parent
    per_class_dir = ensure_dir(out_dir / "qualitative_per_class")

    n_per_class = min(2, int(cfg.eval.n_qualitative_samples))
    selected = select_qualitative_indices_per_class(
        samples=samples,
        n_per_class=n_per_class,
    )

    patch_cfg = cfg.data.p1ml_patch
    n_classes = len(UNIFIED_DAMAGE_CLASSES)

    # ------------------------------------------------------------------
    # A. One compact overview: exactly one representative sample per class.
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(
        n_classes,
        3,
        figsize=(15, 4.2 * n_classes),
        squeeze=False,
    )

    for channel, class_name in enumerate(UNIFIED_DAMAGE_CLASSES):
        candidates = selected[class_name]

        if not candidates:
            for column in range(3):
                axes[channel, column].set_axis_off()

            axes[channel, 0].text(
                0.5,
                0.5,
                f"No validation image with GT-positive '{class_name}'",
                ha="center",
                va="center",
                transform=axes[channel, 0].transAxes,
                fontsize=11,
            )
            continue

        sample_index = candidates[0]
        image, mask = load_rgb_and_multilabel_mask(samples[sample_index])

        probability = predict_sliding_window_multilabel(
            model=model,
            image=image,
            device=device,
            patch_size=int(patch_cfg.patch_size),
            stride=int(patch_cfg.eval_stride),
            batch_size=int(patch_cfg.eval_batch_size),
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD,
            n_classes=n_classes,
            blend_mode=str(patch_cfg.blend_mode),
        ).numpy()

        gt_mask = mask[channel]
        prediction = (probability[channel] > threshold).astype(np.float32)
        color = CLASS_COLORS[class_name]

        gt_overlay = _overlay_single_channel(
            image=image,
            mask=gt_mask,
            color=color,
            alpha=0.50,
        )
        pred_overlay = _overlay_single_channel(
            image=image,
            mask=prediction,
            color=color,
            alpha=0.50,
        )

        panels = [
            (image, "RGB image"),
            (gt_overlay, "Ground truth"),
            (pred_overlay, f"Prediction @ {threshold:.2f}"),
        ]

        for column, (content, title) in enumerate(panels):
            axes[channel, column].imshow(content)
            axes[channel, column].set_axis_off()

            if channel == 0:
                axes[channel, column].set_title(title, fontsize=12)

        axes[channel, 0].set_ylabel(
            f"{class_name}\n(sample {sample_index})",
            rotation=0,
            ha="right",
            va="center",
            fontsize=11,
            fontweight="bold",
            labelpad=18,
        )

    legend_handles = [
        mpatches.Patch(
            color=np.asarray(CLASS_COLORS[name], dtype=np.float32) / 255.0,
            label=name,
        )
        for name in UNIFIED_DAMAGE_CLASSES
    ]

    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=len(UNIFIED_DAMAGE_CLASSES),
        frameon=False,
        fontsize=10,
    )
    fig.suptitle(
        "DACL10K multilabel qualitative evaluation — compact class-wise overview",
        fontsize=15,
        y=0.995,
    )
    fig.tight_layout(rect=(0.05, 0.035, 1.0, 0.98))
    fig.savefig(
        out_dir / "qualitative_overview_multilabel.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(fig)

    # ------------------------------------------------------------------
    # B. Six readable class-specific figures, each with up to 2 examples.
    # ------------------------------------------------------------------
    for class_name in UNIFIED_DAMAGE_CLASSES:
        _save_class_qualitative_figure(
            model=model,
            samples=samples,
            class_name=class_name,
            sample_indices=selected[class_name],
            cfg=cfg,
            device=device,
            threshold=threshold,
            out_path=per_class_dir / f"qualitative_{class_name}.png",
        )

    print(
        "[qualitative] saved compact overview plus per-class figures to "
        f"{out_dir.resolve()}"
    )


@torch.no_grad()
def evaluate_sliding_multilabel(model, samples, cfg, device) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Full-resolution multilabel metrics for every threshold in the sweep."""
    thresholds = list(cfg.eval.threshold_sweep)
    patch_cfg = cfg.data.p1ml_patch
    n_classes = len(UNIFIED_DAMAGE_CLASSES)

    meters = {
        threshold: MultilabelSegmentationMetrics(threshold=threshold)
        for threshold in thresholds
    }

    for index, sample in enumerate(samples, start=1):
        image, mask = load_rgb_and_multilabel_mask(sample)
        probability = predict_sliding_window_multilabel(
            model=model,
            image=image,
            device=device,
            patch_size=int(patch_cfg.patch_size),
            stride=int(patch_cfg.eval_stride),
            batch_size=int(patch_cfg.eval_batch_size),
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD,
            n_classes=n_classes,
            blend_mode=str(patch_cfg.blend_mode),
        )

        # The blended output is a probability: invert the sigmoid so the meters
        # keep a single thresholding code path.
        logits = torch.logit(probability.clamp(1e-6, 1.0 - 1e-6)).unsqueeze(0)
        target = torch.from_numpy(mask).unsqueeze(0)

        for meter in meters.values():
            meter.update(logits, target)

        if index % 25 == 0 or index == len(samples):
            print(f"[eval] processed {index}/{len(samples)} images")

    summary_rows, per_class_rows = [], []
    for threshold, meter in meters.items():
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

    summary = pd.DataFrame(summary_rows).sort_values("threshold").reset_index(drop=True)
    per_class = (
        pd.DataFrame(per_class_rows)
        .sort_values(["threshold", "class"])
        .reset_index(drop=True)
    )
    return summary, per_class


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    cfg = load_config(run_dir / "config.yaml")
    seed_everything(cfg.project.seed)
    device = get_device()

    if int(cfg.model.classes) != len(UNIFIED_DAMAGE_CLASSES):
        raise ValueError(
            f"This run emits {cfg.model.classes} channels: it is not a P1ML multilabel run."
        )

    split = args.split or str(cfg.data.dacl10k.val_split)
    samples = list_samples(cfg.data.dacl10k.root, split)
    if args.limit:
        samples = samples[: args.limit]

    targets = multilabel_sample_targets(samples)
    composition = summarize_multilabel_targets(targets)

    model = build_model(cfg.model).to(device)

    checkpoint_path = run_dir / args.checkpoint
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = load_checkpoint(checkpoint_path, model, device=device)
    checkpoint_epoch = checkpoint.get("epoch", "unknown")
    checkpoint_best_metric = checkpoint.get("best_dice", None)

    print("\n=== Checkpoint used for multilabel sliding-window evaluation ===")
    print(f"checkpoint path : {checkpoint_path.resolve()}")
    print(f"saved epoch     : {checkpoint_epoch}")
    if checkpoint_best_metric is not None:
        print(f"best monitor    : {float(checkpoint_best_metric):.4f} (selection metric of the run)")
    if checkpoint_path.name != "best.pt":
        print(
            "[warning] This is not best.pt. Do not use these metrics as the official "
            "final result unless this choice is intentional."
        )
    print("===============================================================\n")

    patch_cfg = cfg.data.p1ml_patch
    print(
        f"[eval] DACL10K {split}: {composition['n_images']} images | "
        f"patch {patch_cfg.patch_size} | stride {patch_cfg.eval_stride} | "
        f"blend {patch_cfg.blend_mode} | {len(UNIFIED_DAMAGE_CLASSES)} channels"
    )

    summary, per_class = evaluate_sliding_multilabel(model, samples, cfg, device)

    for frame in (summary, per_class):
        frame.insert(0, "dataset", f"dacl10k_{split}_multilabel_sliding")
        frame.insert(1, "patch_size", int(patch_cfg.patch_size))
        frame.insert(2, "stride", int(patch_cfg.eval_stride))
        frame.insert(3, "blend_mode", str(patch_cfg.blend_mode))
        frame.insert(4, "checkpoint_name", checkpoint_path.name)
        frame.insert(5, "checkpoint_epoch", checkpoint_epoch)

    eval_dir = ensure_dir(
        Path(args.output_dir) if args.output_dir is not None else run_dir / "eval_multilabel_sliding"
    )
    summary.to_csv(eval_dir / "metrics_dacl10k_val_multilabel_sliding.csv", index=False)
    per_class.to_csv(eval_dir / "metrics_dacl10k_val_multilabel_per_class.csv", index=False)

    with open(eval_dir / "validation_composition.json", "w", encoding="utf-8") as fh:
        json.dump(
            {
                "split": split,
                "class_names": list(UNIFIED_DAMAGE_CLASSES),
                "class_mapping_19_to_6": unified_damage_label_groups(),
                "inference": {
                    "patch_size": int(patch_cfg.patch_size),
                    "stride": int(patch_cfg.eval_stride),
                    "batch_size": int(patch_cfg.eval_batch_size),
                    "blend_mode": str(patch_cfg.blend_mode),
                },
                "checkpoint": {
                    "path": str(checkpoint_path),
                    "name": checkpoint_path.name,
                    "epoch": int(checkpoint_epoch) if checkpoint_epoch != "unknown" else None,
                    "best_monitor_metric": (
                        float(checkpoint_best_metric) if checkpoint_best_metric is not None else None
                    ),
                },
                **composition,
            },
            fh,
            indent=2,
            ensure_ascii=False,
        )

    save_qualitative_figure(
        model=model,
        samples=samples,
        targets=targets,
        cfg=cfg,
        device=device,
        threshold=float(cfg.eval.threshold),
        out_path=eval_dir / "qualitative_overview_multilabel.png",
    )

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
    print(f"\nMultilabel sliding-window artifacts in {eval_dir.resolve()}")


if __name__ == "__main__":
    main()
