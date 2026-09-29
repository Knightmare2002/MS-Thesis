#!/usr/bin/env python3
"""Split one image into four non-overlapping quadrants for presentation.

Example (run from the repository root):
    python scripts/18_split_image_into_4_patches.py \
        --image datasets/dacl10k/images/train/IMAGE.jpg

Outputs go to output/patches/IMAGE/ by default. No resizing or training code.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True, help="Input JPG or PNG image")
    parser.add_argument("--output-dir", type=Path, default=Path("output/patches"))
    args = parser.parse_args()

    image = cv2.imread(str(args.image), cv2.IMREAD_UNCHANGED)
    if image is None:
        parser.error(f"Cannot read image: {args.image}")
    height, width = image.shape[:2]
    if height < 2 or width < 2:
        parser.error(f"Image must be at least 2x2 pixels; got {width}x{height}")

    mid_y, mid_x = height // 2, width // 2
    areas = {
        "patch_01_top_left": (0, mid_y, 0, mid_x),
        "patch_02_top_right": (0, mid_y, mid_x, width),
        "patch_03_bottom_left": (mid_y, height, 0, mid_x),
        "patch_04_bottom_right": (mid_y, height, mid_x, width),
    }

    folder = args.output_dir / args.image.stem
    folder.mkdir(parents=True, exist_ok=True)

    for name, (y0, y1, x0, x1) in areas.items():
        patch = image[y0:y1, x0:x1]
        destination = folder / f"{name}.png"
        if not cv2.imwrite(str(destination), patch):
            raise OSError(f"Cannot save {destination}")
        print(f"{destination}: {x1-x0}x{y1-y0} px; x=[{x0},{x1}), y=[{y0},{y1})")

    original = folder / "reference.png"
    if not cv2.imwrite(str(original), image):
        raise OSError(f"Cannot save {original}")

    marked = image.copy()
    color = (0, 0, 255) if marked.ndim == 3 else 255
    thickness = max(2, min(height, width) // 350)
    cv2.line(marked, (mid_x, 0), (mid_x, height - 1), color, thickness)
    cv2.line(marked, (0, mid_y), (width - 1, mid_y), color, thickness)
    for number, (y0, y1, x0, x1) in enumerate(areas.values(), start=1):
        label = f"{number:02d}"
        text_x, text_y = x0 + 12, min(y0 + 35, y1 - 5)
        cv2.putText(marked, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, color, max(2, thickness), cv2.LINE_AA)
    overview = folder / "reference_with_grid.png"
    if not cv2.imwrite(str(overview), marked):
        raise OSError(f"Cannot save {overview}")
    print(f"Saved the original, grid overview and four patches in {folder}")


if __name__ == "__main__":
    main()
