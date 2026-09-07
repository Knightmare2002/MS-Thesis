"""Exploratory data analysis: the numbers that justify the experimental design.

Everything returns a pandas DataFrame so the scripts only handle IO/plots.
Image headers are read without decoding pixels when possible (PIL lazy open),
which keeps a full pass over ~10k images in the order of seconds.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm.auto import tqdm

from .class_mapping import DACL10K_CLASSES, DACL10K_DAMAGE
from .dacl10k import load_annotation, present_labels, rasterize_binary


# --------------------------------------------------------------------------- #
# CrackSeg9k
# --------------------------------------------------------------------------- #
def crackseg9k_stats(pairs: list[tuple[Path, Path]], sample: int | None = None, seed: int = 42) -> pd.DataFrame:
    """Per-image stats: resolution and share of crack pixels.

    Columns: image, width, height, aspect_ratio, crack_pixels, total_pixels,
             crack_ratio, is_empty
    """
    if sample and sample < len(pairs):
        rng = np.random.default_rng(seed)
        pairs = [pairs[i] for i in rng.choice(len(pairs), size=sample, replace=False)]

    rows = []
    for image_path, mask_path in tqdm(pairs, desc="EDA CrackSeg9k"):
        with Image.open(image_path) as img:
            width, height = img.size
        # The mask must be decoded to count positive pixels.
        mask = np.asarray(Image.open(mask_path).convert("L")) > 0
        crack_pixels = int(mask.sum())
        total = int(mask.size)
        rows.append(
            {
                "image": image_path.name,
                "width": width,
                "height": height,
                "aspect_ratio": round(width / height, 4),
                "crack_pixels": crack_pixels,
                "total_pixels": total,
                "crack_ratio": crack_pixels / total,
                "is_empty": crack_pixels == 0,
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# dacl10k
# --------------------------------------------------------------------------- #
def dacl10k_stats(
    samples: list[tuple[Path, Path]],
    compute_crack_ratio: bool = True,
    sample: int | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    """Per-image stats + one boolean column per class (multi-label presence).

    `compute_crack_ratio=True` also rasterises the crack channel to measure the
    positive-pixel share, i.e. how comparable dacl10k cracks are to CrackSeg9k.
    """
    if sample and sample < len(samples):
        rng = np.random.default_rng(seed)
        samples = [samples[i] for i in rng.choice(len(samples), size=sample, replace=False)]

    rows = []
    for image_path, ann_path in tqdm(samples, desc="EDA dacl10k"):
        annotation = load_annotation(ann_path)
        with Image.open(image_path) as image:
            width, height = image.size

        json_width = int(annotation["imageWidth"])
        json_height = int(annotation["imageHeight"])
        labels = present_labels(annotation)

        row: dict[str, object] = {
            "image": image_path.name,
            "width": width,
            "height": height,
            "json_width": json_width,
            "json_height": json_height,
            "size_matches_json": (
                width == json_width
                and height == json_height
            ),
            "aspect_ratio": round(width / height, 4),
            "n_shapes": len(annotation.get("shapes", [])),
            "n_classes": len(labels),
            "n_damage_classes": len(labels & set(DACL10K_DAMAGE)),
        }
        row.update({f"has_{c}": (c in labels) for c in DACL10K_CLASSES})

        if compute_crack_ratio:
            mask = rasterize_binary(annotation, shape=(height, width))
            row["crack_pixels"] = int(mask.sum())
            row["crack_ratio"] = float(mask.mean())
        rows.append(row)

    return pd.DataFrame(rows)


def dacl10k_class_frequency(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse the has_* columns into a class frequency table sorted by count."""
    counts = {c: int(df[f"has_{c}"].sum()) for c in DACL10K_CLASSES if f"has_{c}" in df}
    out = pd.DataFrame(
        {
            "class": list(counts),
            "n_images": list(counts.values()),
            "group": ["damage" if c in DACL10K_DAMAGE else "object" for c in counts],
        }
    )
    out["share_of_images"] = out["n_images"] / max(len(df), 1)
    # Imbalance factor w.r.t. the most frequent class: the number to show on a slide.
    out["imbalance_vs_max"] = out["n_images"].max() / out["n_images"].replace(0, np.nan)
    return out.sort_values("n_images", ascending=False).reset_index(drop=True)


def describe_numeric(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Compact summary (count/mean/std/min/quartiles/max) for selected columns."""
    available = [c for c in columns if c in df]
    return df[available].describe(percentiles=[0.25, 0.5, 0.75, 0.95]).T
