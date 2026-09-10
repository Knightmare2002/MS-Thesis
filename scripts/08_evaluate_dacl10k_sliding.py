#!/usr/bin/env python
"""Step 8 / P1-B - full-resolution DACL10K evaluation with sliding windows.

The model is trained on native-resolution patches. Every validation image is
evaluated at original resolution, without global resizing, by averaging
overlapping patch probability maps.

Writes to <run_dir>/eval_sliding/:
    metrics_dacl10k_val_sliding.csv
    validation_composition.json
    qualitative_dacl10k_val_sliding.png

Usage
-----
    python scripts/08_evaluate_dacl10k_sliding.py --run-dir outputs/runs/p1b_patch_unet_r34_512
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.data.dacl10k import (
    binary_sample_targets,
    list_samples,
    load_annotation,
    rasterize_binary,
    summarize_binary_targets,
)
from src.data.transforms import IMAGENET_MEAN, IMAGENET_STD
from src.engine import load_checkpoint
from src.eval.metrics import SegmentationMetrics
from src.eval.sliding_window import predict_sliding_window
from src.models.unet import build_model
from src.utils import ensure_dir, get_device, load_config, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate P1-B on DACL10K validation using full-resolution sliding-window inference."
    )
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--checkpoint", default="best.pt")
    parser.add_argument("--limit", type=int, default=None, help="debug: evaluate only N images")
    return parser.parse_args()


def load_rgb_and_mask(sample, labels):
    """Load original-resolution RGB image and aligned binary Crack/ACrack mask."""
    image_path, annotation_path = sample

    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise RuntimeError(f"Unreadable image: {image_path}")
    image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    annotation = load_annotation(annotation_path)
    mask = rasterize_binary(
        annotation,
        labels=labels,
        shape=image.shape[:2],
    ).astype(np.float32)

    return image, mask


def tensor_for_metric(array: np.ndarray) -> torch.Tensor:
    """Convert [H,W] NumPy data into metric-compatible [1,1,H,W] tensor."""
    return torch.from_numpy(array).unsqueeze(0).unsqueeze(0)


def denormalized_display(image: np.ndarray) -> np.ndarray:
    """Input is already RGB uint8; this helper documents display semantics."""
    return image.copy()


def select_qualitative_indices(targets: list[int], n_samples: int) -> list[int]:
    """Select a balanced deterministic set of positive and negative examples."""
    positives = [index for index, target in enumerate(targets) if target == 1]
    negatives = [index for index, target in enumerate(targets) if target == 0]

    n_positive = min((n_samples + 1) // 2, len(positives))
    n_negative = min(n_samples - n_positive, len(negatives))
    return positives[:n_positive] + negatives[:n_negative]


def save_qualitative_figure(
    model,
    samples,
    targets,
    labels,
    cfg,
    device,
    threshold: float,
    out_path: Path,
) -> None:
    """Save original-resolution qualitative examples using sliding-window predictions."""
    indices = select_qualitative_indices(targets, int(cfg.eval.n_qualitative_samples))
    if not indices:
        raise RuntimeError("No validation samples available for qualitative evaluation.")

    fig, axes = plt.subplots(len(indices), 4, figsize=(12, 3 * len(indices)))
    axes = np.atleast_2d(axes)

    for row, index in enumerate(indices):
        image, mask = load_rgb_and_mask(samples[index], labels)
        probability = predict_sliding_window(
            model=model,
            image=image,
            device=device,
            patch_size=int(cfg.data.p1_patch.patch_size),
            stride=int(cfg.data.p1_patch.eval_stride),
            batch_size=int(cfg.data.p1_patch.eval_batch_size),
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD,
            blend_mode=str(cfg.data.p1_patch.blend_mode),
        ).numpy()

        prediction = probability > threshold
        ground_truth = mask > 0.5
        overlay = denormalized_display(image)
        overlay[prediction] = [255, 0, 0]
        overlay[ground_truth] = [0, 255, 0]

        contents = [
            (image, "image", None),
            (ground_truth, "ground truth", "gray"),
            (prediction, "prediction", "gray"),
            (overlay, "overlay (GT green / pred red)", None),
        ]
        for column, (content, title, cmap) in enumerate(contents):
            axes[row, column].imshow(content, cmap=cmap)
            axes[row, column].set_axis_off()
            if row == 0:
                axes[row, column].set_title(title, fontsize=10)

    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


@torch.no_grad()
def evaluate_sliding(model, samples, labels, cfg, device) -> pd.DataFrame:
    """Compute complete full-resolution validation metrics for every threshold."""
    thresholds = list(cfg.eval.threshold_sweep)
    meters = {threshold: SegmentationMetrics(threshold=threshold) for threshold in thresholds}

    for index, sample in enumerate(samples, start=1):
        image, mask = load_rgb_and_mask(sample, labels)
        probability = predict_sliding_window(
            model=model,
            image=image,
            device=device,
            patch_size=int(cfg.data.p1_patch.patch_size),
            stride=int(cfg.data.p1_patch.eval_stride),
            batch_size=int(cfg.data.p1_patch.eval_batch_size),
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD,
            blend_mode=str(cfg.data.p1_patch.blend_mode),
        )

        logits = torch.logit(probability.clamp(1e-6, 1.0 - 1e-6))
        target = tensor_for_metric(mask)
        for meter in meters.values():
            meter.update(logits.unsqueeze(0), target)

        if index % 25 == 0 or index == len(samples):
            print(f"[eval] processed {index}/{len(samples)} images")

    rows = [{"threshold": threshold, **meter.compute()} for threshold, meter in meters.items()]
    return pd.DataFrame(rows).sort_values("threshold").reset_index(drop=True)


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    cfg = load_config(run_dir / "config.yaml")
    seed_everything(cfg.project.seed)
    device = get_device()

    labels = list(cfg.data.dacl10k.get("crack_labels", ["Crack", "ACrack"]))
    samples = list_samples(cfg.data.dacl10k.root, cfg.data.dacl10k.val_split)
    if args.limit:
        samples = samples[: args.limit]

    targets = binary_sample_targets(samples, labels)
    composition = summarize_binary_targets(targets)

    model = build_model(cfg.model).to(device)
    load_checkpoint(run_dir / args.checkpoint, model, device=device)

    print(
        f"[eval] DACL10K {cfg.data.dacl10k.val_split}: {composition['n_images']} images | "
        f"patch {cfg.data.p1_patch.patch_size} | stride {cfg.data.p1_patch.eval_stride} | "
        f"blend {cfg.data.p1_patch.blend_mode}"
    )

    results = evaluate_sliding(model, samples, labels, cfg, device)
    results.insert(0, "dataset", f"dacl10k_{cfg.data.dacl10k.val_split}_sliding")
    results.insert(1, "target", "+".join(labels))
    results.insert(2, "patch_size", int(cfg.data.p1_patch.patch_size))
    results.insert(3, "stride", int(cfg.data.p1_patch.eval_stride))
    results.insert(4, "blend_mode", str(cfg.data.p1_patch.blend_mode))

    eval_dir = ensure_dir(run_dir / "eval_sliding")
    results.to_csv(eval_dir / "metrics_dacl10k_val_sliding.csv", index=False)

    with open(eval_dir / "validation_composition.json", "w", encoding="utf-8") as fh:
        json.dump(
            {
                "split": str(cfg.data.dacl10k.val_split),
                "target_labels": labels,
                "inference": {
                    "patch_size": int(cfg.data.p1_patch.patch_size),
                    "stride": int(cfg.data.p1_patch.eval_stride),
                    "batch_size": int(cfg.data.p1_patch.eval_batch_size),
                    "blend_mode": str(cfg.data.p1_patch.blend_mode),
                },
                **composition,
            },
            fh,
            indent=2,
        )

    save_qualitative_figure(
        model=model,
        samples=samples,
        targets=targets,
        labels=labels,
        cfg=cfg,
        device=device,
        threshold=float(cfg.eval.threshold),
        out_path=eval_dir / "qualitative_dacl10k_val_sliding.png",
    )

    print(
        results[
            [
                "threshold",
                "iou",
                "dice",
                "precision",
                "recall",
                "dice_image_mean",
                "false_alarm_rate_empty_gt",
            ]
        ].to_string(index=False)
    )
    print(f"\nSliding-window evaluation artifacts in {eval_dir.resolve()}")


if __name__ == "__main__":
    main()