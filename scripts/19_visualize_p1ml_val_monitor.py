#!/usr/bin/env python3
"""Export ONLY the original validation image with its actual epoch-monitor crop outlined.

Run from the repository root:
    python scripts/19_visualize_p1ml_val_monitor.py \
        --image-stem dacl10k_v2_validation_0816

Output: output/patches/<image-stem>/val_monitor_reference.png
The dataset's own deterministic center-crop logic selects the rectangle.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import src.data.dacl10k as dacl
from src.utils import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--image-stem", required=True, help="Exact validation JPG stem, without .jpg")
    parser.add_argument("--config", default="configs/config_p1ml.yaml")
    parser.add_argument("--output-dir", default="output/patches")
    args = parser.parse_args()

    cfg = load_config(args.config)
    split = str(cfg.data.dacl10k.val_split)
    matches = [sample for sample in dacl.list_samples(cfg.data.dacl10k.root, split)
               if sample[0].stem == args.image_stem]
    if len(matches) != 1:
        parser.error(f"Expected exactly one image {args.image_stem!r} in {split}; found {len(matches)}")

    image_path, _ = matches[0]
    reference = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if reference is None:
        raise OSError(f"Cannot read {image_path}")
    height, width = reference.shape[:2]
    size = int(cfg.data.p1ml_patch.patch_size)

    def raw_transform(image, mask):
        return {"image": image, "mask": torch.from_numpy(mask.copy())}

    dataset = dacl.Dacl10kMultilabelCenterPatchDataset(
        samples=matches, transform=raw_transform, patch_size=size
    )
    original_crop_at = dacl._crop_at
    crop_calls: list[tuple[int, int, np.ndarray, np.ndarray]] = []

    def traced_crop(image, mask, top, left, patch_size):
        crop_image, crop_mask = original_crop_at(image, mask, top, left, patch_size)
        crop_calls.append((int(top), int(left), crop_image, crop_mask))
        return crop_image, crop_mask

    dacl._crop_at = traced_crop
    try:
        patch, mask_tensor = dataset[0]
    finally:
        dacl._crop_at = original_crop_at

    if not isinstance(patch, np.ndarray) or patch.shape[:2] != (size, size):
        raise RuntimeError(f"Unexpected image patch shape: {getattr(patch, 'shape', None)}")
    mask = mask_tensor.numpy()
    positions = [(top, left) for top, left, crop, crop_mask in crop_calls
                 if np.array_equal(crop, patch) and np.array_equal(crop_mask, mask)]
    if len(positions) != 1:
        raise RuntimeError("Could not uniquely match monitor patch to a crop position; check _crop_at API.")
    top, left = positions[0]

    if width < size or height < size:
        # Padding is used internally by the dataset; the complete 512px square
        # would not lie in the *original* image, so refuse a misleading overlay.
        raise ValueError(
            f"Source image is {width}x{height}, smaller than the {size}x{size} monitor patch. "
            "Choose an image at least as large as the patch on both axes."
        )
    if not (0 <= left <= width - size and 0 <= top <= height - size):
        raise RuntimeError(f"Crop ({left}, {top}, {size}) falls outside {width}x{height} source image.")

    annotated = reference.copy()
    thickness = max(3, round(min(width, height) / 300))
    cv2.rectangle(annotated, (left, top), (left + size - 1, top + size - 1),
                  (0, 0, 255), thickness, cv2.LINE_AA)
    destination = Path(args.output_dir) / args.image_stem
    destination.mkdir(parents=True, exist_ok=True)
    target = destination / "val_monitor_reference.png"
    if not cv2.imwrite(str(target), annotated):
        raise OSError(f"Cannot save {target}")
    print(f"Saved {target}; actual monitor crop: x={left}, y={top}, {size}x{size}")


if __name__ == "__main__":
    main()
