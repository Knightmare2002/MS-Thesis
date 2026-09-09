#!/usr/bin/env python
"""Export a DACL10K example for the EDA presentation.

Creates one PNG with:
1. original RGB image;
2. Crack/ACrack polygon annotations overlaid on the image;
3. binary Crack + ACrack raster mask.

Usage:
    python scripts/01_export_dacl10k_crack_example.py ^
      --config configs/config.yaml ^
      --split train ^
      --image-name YOUR_IMAGE_NAME.jpg
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.data.class_mapping import CRACK_LIKE_DACL10K
from src.data.dacl10k import load_annotation, rasterize_binary
from src.utils import ensure_dir, load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a DACL10K Crack/ACrack annotation example."
    )
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument(
        "--split",
        choices=["train", "validation"],
        default="train",
    )
    parser.add_argument(
        "--image-name",
        required=True,
        help="Exact JPG filename, e.g. image_000123.jpg",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional output directory override.",
    )
    return parser.parse_args()


def polygon_points(shape: dict) -> np.ndarray | None:
    points = shape.get("points", [])
    if len(points) < 3:
        return None
    return np.asarray(points, dtype=np.float32)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    root = Path(cfg.data.dacl10k.root)
    image_path = root / "images" / args.split / args.image_name
    annotation_path = root / "annotations" / args.split / f"{image_path.stem}.json"

    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")
    if not annotation_path.is_file():
        raise FileNotFoundError(
            f"Annotation not found: {annotation_path}"
        )

    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise RuntimeError(f"Could not decode image: {image_path}")

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    height, width = image_rgb.shape[:2]

    annotation = load_annotation(annotation_path)
    mask = rasterize_binary(
        annotation,
        labels=CRACK_LIKE_DACL10K,
        shape=(height, width),
    )

    crack_pixels = int(mask.sum())
    crack_ratio = float(mask.mean()) * 100

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(image_rgb)
    axes[0].set_title("Original image", fontweight="bold")
    axes[0].axis("off")

    axes[1].imshow(image_rgb)
    for shape in annotation.get("shapes", []):
        if shape.get("label") not in CRACK_LIKE_DACL10K:
            continue

        points = polygon_points(shape)
        if points is None:
            continue

        closed = np.vstack([points, points[0]])
        axes[1].plot(
            closed[:, 0],
            closed[:, 1],
            color="#e53e3e",
            linewidth=1.5,
        )

    axes[1].set_title(
        "Crack / ACrack polygons",
        fontweight="bold",
    )
    axes[1].axis("off")

    axes[2].imshow(mask, cmap="gray", vmin=0, vmax=1)
    axes[2].set_title(
        f"Binary crack mask\n{crack_ratio:.3f}% positive pixels",
        fontweight="bold",
    )
    axes[2].axis("off")

    fig.suptitle(
        f"DACL10K annotation-to-mask conversion — {args.image_name}",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout()

    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else Path(cfg.project.output_dir) / "eda" / "figures"
    )
    ensure_dir(output_dir)

    output_path = output_dir / f"{image_path.stem}_crack_mask_example.png"
    fig.savefig(
        output_path,
        dpi=250,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)

    print(f"Image:        {image_path}")
    print(f"Annotation:   {annotation_path}")
    print(f"Crack pixels: {crack_pixels}")
    print(f"Crack ratio:  {crack_ratio:.3f}%")
    print(f"Saved figure: {output_path.resolve()}")


if __name__ == "__main__":
    main()