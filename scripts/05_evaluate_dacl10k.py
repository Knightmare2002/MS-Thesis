#!/usr/bin/env python
"""Step 5 / P1 - evaluate the DACL10K bridge-damage baseline.

The script:
1. Loads the best P1 checkpoint.
2. Evaluates only the official DACL10K validation split.
3. Produces a threshold sweep without changing the frozen model.
4. Saves complete metrics, a positive/negative breakdown, and qualitative masks.

Usage
-----
    python scripts/05_evaluate_dacl10k.py --run-dir outputs/runs/p1_dacl10k_unet_r34_512
    python scripts/05_evaluate_dacl10k.py --run-dir outputs/runs/p1_dacl10k_unet_r34_512 --limit 64
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.data.dacl10k import (
    Dacl10kCrackDataset,
    binary_sample_targets,
    list_samples,
    summarize_binary_targets,
)
from src.data.transforms import IMAGENET_MEAN, IMAGENET_STD, eval_transform
from src.engine import load_checkpoint
from src.eval.metrics import SegmentationMetrics
from src.models.unet import build_model
from src.utils import ensure_dir, get_device, load_config, loader_kwargs, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the P1 DACL10K bridge-damage baseline.")
    parser.add_argument("--run-dir", required=True, help="directory containing config.yaml and best.pt")
    parser.add_argument("--checkpoint", default="best.pt")
    parser.add_argument("--limit", type=int, default=None, help="evaluate only N validation images")
    return parser.parse_args()


@torch.no_grad()
def evaluate_with_sweep(model, loader, device, thresholds: list[float]) -> pd.DataFrame:
    """Compute all threshold-dependent metrics in one forward pass over DACL10K."""
    model.eval()
    meters = {threshold: SegmentationMetrics(threshold=threshold) for threshold in thresholds}

    for images, masks in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        logits = model(images)

        for meter in meters.values():
            meter.update(logits, masks)

    rows = [{"threshold": threshold, **meter.compute()} for threshold, meter in meters.items()]
    return pd.DataFrame(rows).sort_values("threshold").reset_index(drop=True)


def denormalize(image: torch.Tensor) -> np.ndarray:
    """Convert one normalized [3,H,W] tensor to a displayable uint8 RGB image."""
    rgb = image.cpu().numpy().transpose(1, 2, 0)
    rgb = rgb * np.asarray(IMAGENET_STD) + np.asarray(IMAGENET_MEAN)
    return (np.clip(rgb, 0.0, 1.0) * 255).astype(np.uint8)


@torch.no_grad()
def qualitative_grid(
    model,
    dataset,
    targets: list[int],
    device,
    threshold: float,
    n_samples: int,
    out_path: Path,
) -> None:
    """Save informative positive and negative validation examples.

    Rows contain image, ground truth, prediction and overlay. Positive samples
    visualize crack recovery; negative samples visualize false-alarm behavior.
    """
    positive_indices = [index for index, target in enumerate(targets) if target == 1]
    negative_indices = [index for index, target in enumerate(targets) if target == 0]

    n_positive = min(max(1, n_samples // 2), len(positive_indices))
    n_negative = min(n_samples - n_positive, len(negative_indices))
    selected = positive_indices[:n_positive] + negative_indices[:n_negative]

    if not selected:
        raise RuntimeError("No DACL10K samples available for qualitative evaluation.")

    fig, axes = plt.subplots(len(selected), 4, figsize=(12, 3 * len(selected)))
    axes = np.atleast_2d(axes)

    for row, index in enumerate(selected):
        image, mask = dataset[index]
        logits = model(image.unsqueeze(0).to(device))
        prediction = (torch.sigmoid(logits)[0, 0] > threshold).cpu().numpy()

        rgb = denormalize(image)
        gt = mask[0].numpy() > 0.5

        overlay = rgb.copy()
        overlay[prediction] = [255, 0, 0]
        overlay[gt] = [0, 255, 0]

        contents = [
            (rgb, "image", None),
            (gt, "ground truth", "gray"),
            (prediction, "prediction", "gray"),
            (overlay, "overlay (GT green / pred red)", None),
        ]
        for col, (content, title, cmap) in enumerate(contents):
            axes[row, col].imshow(content, cmap=cmap)
            axes[row, col].set_axis_off()
            if row == 0:
                axes[row, col].set_title(title, fontsize=10)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


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
    split_summary = summarize_binary_targets(targets)

    dataset = Dacl10kCrackDataset(
        samples,
        eval_transform(cfg.data.image_size),
        labels=labels,
    )
    loader = DataLoader(
        dataset,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        **loader_kwargs(cfg.data, device),
    )

    model = build_model(cfg.model).to(device)
    load_checkpoint(run_dir / args.checkpoint, model, device=device)

    thresholds = list(cfg.eval.threshold_sweep)
    results = evaluate_with_sweep(model, loader, device, thresholds)
    results.insert(0, "dataset", f"dacl10k_{cfg.data.dacl10k.val_split}")
    results.insert(1, "target", "+".join(labels))

    print(
        f"[eval] DACL10K {cfg.data.dacl10k.val_split}: {split_summary['n_images']} images | "
        f"positive {split_summary['positive_fraction']:.1%} | "
        f"negative {split_summary['negative_fraction']:.1%}"
    )
    print(results[["threshold", "iou", "dice", "precision", "recall", "false_alarm_rate_empty_gt"]].to_string(index=False))

    eval_dir = ensure_dir(run_dir / "eval")
    results.to_csv(eval_dir / "metrics_dacl10k_val.csv", index=False)

    with open(eval_dir / "validation_composition.json", "w", encoding="utf-8") as fh:
        json.dump(
            {
                "split": str(cfg.data.dacl10k.val_split),
                "target_labels": labels,
                **split_summary,
            },
            fh,
            indent=2,
        )

    qualitative_grid(
        model=model,
        dataset=dataset,
        targets=targets,
        device=device,
        threshold=float(cfg.eval.threshold),
        n_samples=int(cfg.eval.n_qualitative_samples),
        out_path=eval_dir / "qualitative_dacl10k_val.png",
    )

    print(f"\nEvaluation artifacts in {eval_dir.resolve()}")


if __name__ == "__main__":
    main()