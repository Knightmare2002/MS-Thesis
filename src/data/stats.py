"""Exploratory data analysis: the numbers that justify the experimental design.

Everything returns a pandas DataFrame so the scripts only handle IO/plots.
Image headers are read without decoding pixels when possible (PIL lazy open), which keeps a full pass over ~10k images in the order of seconds.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm.auto import tqdm

from .class_mapping import DACL10K_CLASSES, DACL10K_DAMAGE, UNIFIED_DAMAGE_CLASSES
from .dacl10k import (
    load_annotation,
    present_labels,
    rasterize_binary,
    rasterize_unified_damage,
)


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


# --------------------------------------------------------------------------- #
# P1ML: per-channel pixel statistics and pos_weight vector
# --------------------------------------------------------------------------- #
def dacl10k_multilabel_pixel_stats(
    samples: list[tuple[Path, Path]],
    max_images: int | None = None,
    seed: int = 42,
    ) -> dict:
    """Pixel-level frequency of the 6 unified damage classes.

    Computed on the official DACL10K **train** split only: using validation pixels to set the loss weights would leak the evaluation distribution into the objective. Masks are rasterised at native resolution, so the ratios match
    exactly what the patch sampler will see.

    `max_images` subsamples the split (reproducibly) when a full pass is too expensive; the value is recorded in the returned dictionary so the weights remain traceable.
    """
    if not samples:
        raise ValueError("Cannot compute class statistics on an empty split.")

    if max_images and max_images < len(samples):
        rng = np.random.default_rng(seed)
        indices = sorted(rng.choice(len(samples), size=max_images, replace=False).tolist())
        selected = [samples[i] for i in indices]
    else:
        selected = list(samples)

    n_classes = len(UNIFIED_DAMAGE_CLASSES)
    positive_pixels = np.zeros(n_classes, dtype=np.float64)
    images_present = np.zeros(n_classes, dtype=np.int64)
    total_pixels = 0.0
    union_positive_pixels = 0.0

    for _, annotation_path in tqdm(selected, desc="EDA dacl10k multilabel"):
        annotation = load_annotation(annotation_path)
        masks = rasterize_unified_damage(annotation) > 0

        total_pixels += float(masks.shape[1] * masks.shape[2])
        union_positive_pixels += float(masks.any(axis=0).sum())

        for channel in range(n_classes):
            count = float(masks[channel].sum())
            positive_pixels[channel] += count
            images_present[channel] += int(count > 0)

    return {
        "split_size": len(samples),
        "n_images_used": len(selected),
        "max_images": max_images,
        "seed": seed,
        "class_names": list(UNIFIED_DAMAGE_CLASSES),
        "total_pixels": total_pixels,
        "union_positive_pixels": union_positive_pixels,
        "union_positive_fraction": union_positive_pixels / max(total_pixels, 1.0),
        "positive_pixels": positive_pixels.tolist(),
        "negative_pixels": (total_pixels - positive_pixels).tolist(),
        "positive_fraction": (positive_pixels / max(total_pixels, 1.0)).tolist(),
        "n_images_present": images_present.tolist(),
    }


def multilabel_pos_weights(
    stats: dict,
    clip_min: float = 1.0,
    clip_max: float = 20.0,
    ) -> dict:
    """Turn per-channel pixel statistics into a clipped pos_weight vector.

    For each channel c the unclipped value is the inverse positive prior

        w_c = N_neg,c / N_pos,c ,

    i.e. the weight that equalises the contribution of positive and negative
    pixels in BCEWithLogits. Raw values reach O(10^3) for `delamination`, which
    makes the gradient explode and the model predict everything as damage, hence
    the clipping to [clip_min, clip_max]; both raw and clipped values are kept so
    the thesis can report how much each channel was capped.
    """
    if clip_max < clip_min:
        raise ValueError("clip_max must be >= clip_min.")

    positive = np.asarray(stats["positive_pixels"], dtype=np.float64)
    negative = np.asarray(stats["negative_pixels"], dtype=np.float64)

    if (positive <= 0).any():
        empty = [
            name
            for name, count in zip(stats["class_names"], positive.tolist())
            if count <= 0
        ]
        raise RuntimeError(
            f"Classes with zero positive pixels in the train split: {empty}. "
            "pos_weight would be undefined."
        )

    raw = negative / positive
    clipped = np.clip(raw, clip_min, clip_max)

    return {
        "class_names": list(stats["class_names"]),
        "pos_weight_raw": raw.tolist(),
        "pos_weight": clipped.tolist(),
        "clip_min": float(clip_min),
        "clip_max": float(clip_max),
        "n_clipped": int((raw > clip_max).sum() + (raw < clip_min).sum()),
        "source": {
            "split_size": stats["split_size"],
            "n_images_used": stats["n_images_used"],
            "positive_fraction": stats["positive_fraction"],
            "n_images_present": stats["n_images_present"],
        },
    }
