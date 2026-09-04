#!/usr/bin/env python
"""Step 3 - evaluate the trained U-Net and produce the slide material.

What it does
------------
1. Loads `best.pt` from a run directory.
2. Evaluates on the held-out CrackSeg9k test split (same deterministic split).
3. Sweeps the decision threshold in one pass over the data (calibration table).
4. Optionally evaluates *zero-shot* on the dacl10k validation split, using only
   the Crack + ACrack channels: a first, cheap domain-shift measurement that
   motivates the dual-branch design without touching the private UAV test set.
5. Saves a qualitative grid (image | ground truth | prediction | overlay).

Usage
-----
    python scripts/03_evaluate.py --run-dir outputs/runs/unet_r34_512
    python scripts/03_evaluate.py --run-dir outputs/runs/unet_r34_512 --cross-dataset
"""

from __future__ import annotations

import argparse
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

from src.data.crackseg9k import CrackSeg9kDataset, list_pairs, split_pairs
from src.data.dacl10k import Dacl10kCrackDataset, list_samples
from src.data.transforms import IMAGENET_MEAN, IMAGENET_STD, eval_transform
from src.engine import load_checkpoint
from src.eval.metrics import SegmentationMetrics
from src.losses import build_loss
from src.models.unet import build_model
from src.utils import ensure_dir, get_device, load_config, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the baseline U-Net")
    parser.add_argument("--run-dir", required=True, help="directory containing best.pt and config.yaml")
    parser.add_argument("--checkpoint", default="best.pt")
    parser.add_argument("--cross-dataset", action="store_true", help="also evaluate zero-shot on dacl10k")
    parser.add_argument("--limit", type=int, default=None, help="evaluate only N images (debug)")
    return parser.parse_args()


# --------------------------------------------------------------------------- #
# Evaluation with a threshold sweep in a single pass
# --------------------------------------------------------------------------- #
@torch.no_grad()
def evaluate_with_sweep(model, loader, device, thresholds: list[float]) -> pd.DataFrame:
    """Return one metrics row per threshold, computed in a single pass over the loader."""
    model.eval()
    meters = {t: SegmentationMetrics(threshold=t) for t in thresholds}
    for images, masks in loader:
        images, masks = images.to(device), masks.to(device)
        logits = model(images)
        for meter in meters.values():
            meter.update(logits, masks)
    rows = [{"threshold": t, **meter.compute()} for t, meter in meters.items()]
    return pd.DataFrame(rows).sort_values("threshold").reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Qualitative figure
# --------------------------------------------------------------------------- #
def denormalize(tensor: torch.Tensor) -> np.ndarray:
    """Undo the ImageNet normalisation for display: [3,H,W] tensor -> [H,W,3] uint8."""
    image = tensor.cpu().numpy().transpose(1, 2, 0)
    image = image * np.array(IMAGENET_STD) + np.array(IMAGENET_MEAN)
    return (np.clip(image, 0, 1) * 255).astype(np.uint8)


@torch.no_grad()
def qualitative_grid(model, dataset, device, n_samples: int, threshold: float, out_path: Path) -> None:
    """Save a grid: input | ground truth | prediction | overlay.

    Samples are taken from the images with the most crack pixels, so the figure
    is informative instead of showing four empty masks.
    """
    ratios = []
    for i in range(len(dataset)):
        _, mask = dataset[i]
        ratios.append(mask.mean().item())
        if i >= 200:  # scanning the whole test set would be wasteful
            break
    indices = np.argsort(ratios)[::-1][:n_samples]

    fig, axes = plt.subplots(len(indices), 4, figsize=(12, 3 * len(indices)))
    axes = np.atleast_2d(axes)
    for row, index in enumerate(indices):
        image, mask = dataset[int(index)]
        logits = model(image.unsqueeze(0).to(device))
        prediction = (torch.sigmoid(logits)[0, 0] > threshold).cpu().numpy()
        rgb = denormalize(image)

        overlay = rgb.copy()
        overlay[prediction] = [255, 0, 0]        # false/true positives in red
        overlay[mask[0].numpy() > 0.5] = [0, 255, 0]  # ground truth in green (drawn last)

        for col, (content, title, cmap) in enumerate(
            [
                (rgb, "image", None),
                (mask[0].numpy(), "ground truth", "gray"),
                (prediction, "prediction", "gray"),
                (overlay, "overlay (GT green / pred red)", None),
            ]
        ):
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

    # ---- Model -------------------------------------------------------------
    model = build_model(cfg.model).to(device)
    load_checkpoint(run_dir / args.checkpoint, model, device=device)
    criterion = build_loss(cfg.loss).to(device)
    thresholds = list(cfg.eval.threshold_sweep)
    common = dict(num_workers=cfg.data.num_workers, pin_memory=(device.type == "cuda"))
    eval_tf = eval_transform(cfg.data.image_size)

    # ---- In-domain test set ------------------------------------------------
    pairs = list_pairs(cfg.data.crackseg9k.images_dir, cfg.data.crackseg9k.masks_dir)
    splits = split_pairs(
        pairs,
        val_fraction=cfg.data.crackseg9k.val_fraction,
        test_fraction=cfg.data.crackseg9k.test_fraction,
        seed=cfg.project.seed,
    )
    test_items = splits["test"][: args.limit] if args.limit else splits["test"]
    test_ds = CrackSeg9kDataset(test_items, eval_tf, cfg.data.crackseg9k.mask_threshold)
    test_loader = DataLoader(test_ds, batch_size=cfg.train.batch_size, shuffle=False, **common)

    print(f"[eval] CrackSeg9k test: {len(test_ds)} images")
    in_domain = evaluate_with_sweep(model, test_loader, device, thresholds)
    in_domain.insert(0, "dataset", "crackseg9k_test")
    print(in_domain[["threshold", "iou", "dice", "precision", "recall"]].to_string(index=False))

    all_results = [in_domain]

    # ---- Zero-shot cross-dataset (domain shift) ----------------------------
    if args.cross_dataset:
        samples = list_samples(cfg.data.dacl10k.root, cfg.data.dacl10k.val_split)
        samples = samples[: args.limit] if args.limit else samples
        cross_ds = Dacl10kCrackDataset(samples, eval_tf)
        cross_loader = DataLoader(cross_ds, batch_size=cfg.train.batch_size, shuffle=False, **common)
        print(f"\n[eval] dacl10k {cfg.data.dacl10k.val_split} (zero-shot, Crack+ACrack): {len(cross_ds)} images")
        cross = evaluate_with_sweep(model, cross_loader, device, thresholds)
        cross.insert(0, "dataset", f"dacl10k_{cfg.data.dacl10k.val_split}_zeroshot")
        print(cross[["threshold", "iou", "dice", "precision", "recall"]].to_string(index=False))
        all_results.append(cross)

    # ---- Persist -----------------------------------------------------------
    eval_dir = ensure_dir(run_dir / "eval")
    results = pd.concat(all_results, ignore_index=True)
    results.to_csv(eval_dir / "metrics.csv", index=False)

    qualitative_grid(
        model, test_ds, device,
        n_samples=cfg.eval.n_qualitative_samples,
        threshold=cfg.eval.threshold,
        out_path=eval_dir / "qualitative_crackseg9k.png",
    )
    print(f"\nEvaluation artifacts in {eval_dir.resolve()}")


if __name__ == "__main__":
    main()
