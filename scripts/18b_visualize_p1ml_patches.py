#!/usr/bin/env python3
"""Visualize P1ML's *pre-augmentation* crops from one DACL10K image.

Run from the repository root:
  python scripts/18_visualize_p1ml_patches.py --image-stem IMAGE_STEM
  python scripts/18_visualize_p1ml_patches.py --image-stem IMAGE_STEM --n-positive 3 --n-negative 2 --seed 42

Uses Dacl10kMultilabelPatchDataset's actual _positive_crop/_negative_crop methods.
The requested image must have at least min_positive_pixels union damage pixels.
Negative examples may contain damage if the sampler exhausts its attempts.
No model/checkpoint/training files are modified.
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
from matplotlib.patches import Rectangle
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import src.data.dacl10k as dacl
from src.data.class_mapping import UNIFIED_DAMAGE_CLASSES
from src.utils import load_config


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--image-stem", required=True, help="Exact JPG filename without .jpg")
    p.add_argument("--split", default="train", help="Dataset split; default train")
    p.add_argument("--config", default="configs/config_p1ml.yaml")
    p.add_argument("--n-positive", type=int, default=3)
    p.add_argument("--n-negative", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", default=None)
    return p.parse_args()


def write_rgb(path: Path, image: np.ndarray):
    if not cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR)):
        raise OSError(f"Could not save {path}")


def capture_crop(dataset, kind, image, mask):
    """Record actual _crop_at calls; match final returned crop (including fallback)."""
    original = dacl._crop_at
    candidates = []

    def traced(im, ma, top, left, size):
        crop = original(im, ma, top, left, size)
        candidates.append((int(top), int(left), crop))
        return crop

    dacl._crop_at = traced
    try:
        patch_image, patch_mask = (
            dataset._positive_crop(image, mask) if kind == "positive"
            else dataset._negative_crop(image, mask)
        )
    finally:
        dacl._crop_at = original

    matches = [(top, left) for top, left, (ci, cm) in candidates
               if np.array_equal(ci, patch_image) and np.array_equal(cm, patch_mask)]
    if not matches:
        raise RuntimeError("Cannot locate the returned crop among _crop_at calls; sampler implementation may have changed.")
    top, left = matches[-1]
    if patch_image.shape[:2] != (dataset.patch_size, dataset.patch_size):
        raise RuntimeError(f"Unexpected crop shape: {patch_image.shape}")
    return patch_image, patch_mask, top, left


def main():
    a = parse_args()
    if a.n_positive < 0 or a.n_negative < 0 or a.n_positive + a.n_negative == 0:
        raise ValueError("Request at least one patch; counts must be non-negative.")
    cfg = load_config(a.config)
    pc = cfg.data.p1ml_patch
    samples = dacl.list_samples(cfg.data.dacl10k.root, a.split)
    selected = [pair for pair in samples if pair[0].stem == a.image_stem]
    if len(selected) != 1:
        raise ValueError(f"Expected exactly one image with stem {a.image_stem!r} in {a.split}; found {len(selected)}")

    dataset = dacl.Dacl10kMultilabelPatchDataset(
        samples=selected, transform=lambda **kwargs: kwargs,
        patch_size=int(pc.patch_size),
        positive_patch_fraction=float(pc.positive_patch_fraction),
        min_positive_pixels=int(pc.min_positive_pixels),
        max_negative_pixels=int(pc.max_negative_pixels),
        max_crop_attempts=int(pc.max_crop_attempts),
        patches_per_image=int(pc.patches_per_image),
    )
    image, mask = dataset._load_sample(0)
    raw = cv2.imread(str(selected[0][0]), cv2.IMREAD_COLOR)
    if raw is None:
        raise OSError(f"Cannot read {selected[0][0]}")
    original_height, original_width = raw.shape[:2]
    output = Path(a.output_dir) if a.output_dir else Path(cfg.project.output_dir) / "visualizations" / "p1ml_patches" / a.image_stem
    output.mkdir(parents=True, exist_ok=True)
    write_rgb(output / "reference.png", image)

    rows = []
    rng_state = np.random.get_state()
    np.random.seed(a.seed)
    try:
        for kind, count in (("positive", a.n_positive), ("negative", a.n_negative)):
            for number in range(1, count + 1):
                patch_image, patch_mask, top, left = capture_crop(dataset, kind, image, mask)
                union = patch_mask.max(axis=2) > 0
                union_pixels = int(union.sum())
                label = f"{kind}_{number:02d}"
                write_rgb(output / f"{label}.png", patch_image)
                cv2.imwrite(str(output / f"{label}_union_mask.png"), union.astype(np.uint8) * 255)
                per_class = {name: int((patch_mask[..., i] > 0).sum()) for i, name in enumerate(UNIFIED_DAMAGE_CLASSES)}
                rows.append({"name": label, "requested": kind, "top": top, "left": left,
                             "patch_size": dataset.patch_size, "union_pixels": union_pixels,
                             "meets_positive_min": union_pixels >= dataset.min_positive_pixels,
                             "meets_negative_max": union_pixels <= dataset.max_negative_pixels,
                             "pixels_per_class": per_class})
    finally:
        np.random.set_state(rng_state)

    colors = {"positive": "#e05252", "negative": "#198eaa"}
    fig, axes = plt.subplots(1, len(rows) + 1, figsize=(5 + 3.2 * len(rows), 5),
                             gridspec_kw={"width_ratios": [1.8] + [1] * len(rows)})
    axes[0].imshow(image)
    axes[0].set_title(f"Reference: {a.image_stem}\n{original_width}x{original_height} original")
    for row, ax in zip(rows, axes[1:]):
        rgb = cv2.cvtColor(cv2.imread(str(output / f"{row['name']}.png")), cv2.COLOR_BGR2RGB)
        ax.imshow(rgb)
        ax.set_title(f"{row['name']}\nunion: {row['union_pixels']} px", fontsize=10)
        ax.axis("off")
        axes[0].add_patch(Rectangle((row["left"], row["top"]), row["patch_size"], row["patch_size"],
                                     fill=False, linewidth=2, edgecolor=colors[row["requested"]]))
        axes[0].text(row["left"], row["top"], row["name"], fontsize=8, color="white",
                     bbox={"facecolor": colors[row["requested"]], "pad": 2, "edgecolor": "none"})
    axes[0].set_xlim(0, image.shape[1])
    axes[0].set_ylim(image.shape[0], 0)
    axes[0].axis("off")
    fig.tight_layout()
    fig.savefig(output / "overview.png", dpi=200, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    metadata = {"image": str(selected[0][0]), "annotation": str(selected[0][1]),
                "split": a.split, "seed": a.seed, "original_hw": [original_height, original_width],
                "reference_hw": list(image.shape[:2]), "min_positive_pixels": dataset.min_positive_pixels,
                "max_negative_pixels": dataset.max_negative_pixels, "patches": rows,
                "note": "Raw sampler crops before training augmentation; reference padded if source is smaller than patch."}
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved {len(rows)} patches, overview and metadata in {output}")
    for row in rows:
        print(f"{row['name']}: (x={row['left']}, y={row['top']}), union={row['union_pixels']} px")


if __name__ == "__main__":
    main()
