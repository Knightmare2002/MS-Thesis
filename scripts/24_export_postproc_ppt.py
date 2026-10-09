#!/usr/bin/env python
"""Export presentation figures for the multilabel post-processing pipeline."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.data.class_mapping import UNIFIED_DAMAGE_CLASSES
from src.eval.postprocess import min_area_pixels, remove_small_components
from src.utils import (
    ensure_dir,
    get_device,
    load_config,
    multilabel_patch_cfg,
    seed_everything,
)

COLORS = {
    "crack": (255, 0, 0),
    "spalling": (255, 128, 0),
    "corrosion": (255, 0, 255),
    "moisture": (0, 128, 255),
    "delamination": (0, 255, 255),
    "surface": (0, 255, 0),
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--image", required=True, help="Original RGB image.")
    parser.add_argument(
        "--operating-points",
        required=True,
        help="operating_points_frozen.json produced by script 20.",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Default: checkpoint recorded in operating-points JSON.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--alpha", type=float, default=0.65)
    parser.add_argument("--dpi", type=int, default=250)
    return parser.parse_args()


def load_step17():
    path = Path(__file__).with_name("17_calibrate_multilabel_thresholds.py")
    spec = importlib.util.spec_from_file_location("ppt_step17", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def overlay(image, color, opacity):
    """Opacity can be a scalar field [H,W], including probabilities."""
    opacity = np.asarray(opacity, dtype=np.float32)[..., None]
    result = (
        image.astype(np.float32) * (1.0 - opacity)
        + np.asarray(color, dtype=np.float32) * opacity
    )
    return np.clip(result, 0, 255).astype(np.uint8)


def component_overlay(image, labels, alpha, seed):
    """Assign a reproducible categorical color to each foreground component."""
    rng = np.random.default_rng(seed)
    palette = rng.integers(
        45, 256, size=(int(labels.max()) + 1, 3), dtype=np.uint8
    )
    positive = labels > 0
    result = image.copy()
    result[positive] = (
        (1.0 - alpha) * image[positive].astype(np.float32)
        + alpha * palette[labels[positive]].astype(np.float32)
    ).astype(np.uint8)
    return result


def save_grid(panels, titles, heading, path, dpi):
    fig, axes = plt.subplots(2, 3, figsize=(15, 10), squeeze=False)
    try:
        for ax, panel, title in zip(axes.flat, panels, titles):
            ax.imshow(panel, interpolation="nearest")
            ax.set_title(title, fontsize=12)
            ax.set_axis_off()
        fig.suptitle(heading, fontsize=17)
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    finally:
        plt.close(fig)


@torch.no_grad()
def main():
    args = parse_args()
    if not 0.0 <= args.alpha <= 1.0:
        raise ValueError("--alpha must be in [0,1].")
    if args.dpi < 1:
        raise ValueError("--dpi must be positive.")

    run_dir = Path(args.run_dir)
    output_dir = ensure_dir(args.output_dir)
    individual_dir = ensure_dir(output_dir / "per_channel")
    cfg = load_config(run_dir / "config.yaml")
    seed_everything(cfg.project.seed)

    with open(args.operating_points, encoding="utf-8") as handle:
        frozen = json.load(handle)

    points = frozen["frozen_fold0"]["threshold_min_area"]
    if set(points) != set(UNIFIED_DAMAGE_CLASSES):
        raise ValueError("Operating-point class names do not match the taxonomy.")

    source_checkpoint = frozen["checkpoint"]["name"]
    checkpoint_name = args.checkpoint or source_checkpoint
    if checkpoint_name != source_checkpoint:
        raise ValueError(
            "Checkpoint differs from the one used for calibration: "
            f"{checkpoint_name} vs {source_checkpoint}."
        )

    bgr = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"Unreadable image: {args.image}")
    image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    step17 = load_step17()
    predict, checkpoint_path, checkpoint, family = step17.build_predictor(
        cfg, run_dir, checkpoint_name, get_device()
    )
    recorded_epoch = frozen["checkpoint"].get("epoch")
    if recorded_epoch is not None and (
        str(checkpoint.get("epoch", "unknown")) != str(recorded_epoch)
    ):
        raise ValueError(
            "Checkpoint epoch differs from the calibration JSON. "
            "Use the original calibrated checkpoint."
        )

    probability = np.asarray(
        predict(image, multilabel_patch_cfg(cfg)), dtype=np.float32
    )
    expected_shape = (len(UNIFIED_DAMAGE_CLASSES), *image.shape[:2])
    if probability.shape != expected_shape:
        raise ValueError(
            f"Expected probability shape {expected_shape}, got {probability.shape}."
        )
    if not np.isfinite(probability).all():
        raise ValueError("Probability maps contain NaN or infinity.")
    if probability.min() < 0 or probability.max() > 1:
        raise ValueError("Predictor must return probabilities in [0,1].")

    raw_masks = np.zeros(expected_shape, dtype=bool)
    final_masks = np.zeros(expected_shape, dtype=bool)
    probability_panels, component_panels, final_panels = [], [], []
    probability_titles, component_titles, final_titles = [], [], []
    report = []

    Image.fromarray(image).save(output_dir / "00_rgb.png")

    for channel, name in enumerate(UNIFIED_DAMAGE_CLASSES):
        tau = float(points[name]["threshold"])
        fraction = float(points[name]["min_area_fraction"])
        if not 0.0 < tau < 1.0 or not 0.0 <= fraction <= 1.0:
            raise ValueError(f"Invalid operating point for {name}.")

        area_px = min_area_pixels(fraction, image.shape[:2])
        binary = probability[channel] > tau
        n, labels, stats, _ = cv2.connectedComponentsWithStats(
            binary.astype(np.uint8), connectivity=8
        )
        areas = stats[1:, cv2.CC_STAT_AREA]
        kept = int((areas >= area_px).sum())
        cleaned = remove_small_components(binary, area_px)

        raw_masks[channel] = binary
        final_masks[channel] = cleaned

        prob_panel = overlay(
            image, COLORS[name], args.alpha * probability[channel]
        )
        cc_panel = component_overlay(
            image, labels, args.alpha, seed=42 + channel
        )
        final_panel = overlay(
            image, COLORS[name], args.alpha * cleaned.astype(np.float32)
        )

        probability_panels.append(prob_panel)
        component_panels.append(cc_panel)
        final_panels.append(final_panel)

        probability_titles.append(f"{name} | opacity proportional to probability")
        component_titles.append(
            f"{name} | tau={tau:.2f} | {n - 1} components"
        )
        final_titles.append(
            f"{name} | A={area_px:,} px | {kept}/{n - 1} retained"
        )

        for suffix, panel in (
            ("probability_overlay", prob_panel),
            ("components_overlay", cc_panel),
            ("filtered_overlay", final_panel),
        ):
            Image.fromarray(panel).save(individual_dir / f"{name}_{suffix}.png")

        Image.fromarray(binary.astype(np.uint8) * 255).save(
            individual_dir / f"{name}_threshold_mask.png"
        )
        Image.fromarray(cleaned.astype(np.uint8) * 255).save(
            individual_dir / f"{name}_filtered_mask.png"
        )

        report.append({
            "class": name,
            "threshold": tau,
            "min_area_fraction": fraction,
            "min_area_pixels": area_px,
            "components_before": int(n - 1),
            "components_after": kept,
            "pixels_before": int(binary.sum()),
            "pixels_after": int(cleaned.sum()),
        })

    save_grid(
        probability_panels, probability_titles,
        "Six probability maps — opacity encodes probability",
        output_dir / "01_probability_maps.png", args.dpi,
    )
    save_grid(
        component_panels, component_titles,
        "Connected components after per-class thresholding — 8-connectivity",
        output_dir / "02_connected_components.png", args.dpi,
    )
    save_grid(
        final_panels, final_titles,
        "Six post-processed masks — small components removed",
        output_dir / "03_filtered_damage_masks.png", args.dpi,
    )

    np.savez_compressed(
        output_dir / "prediction_arrays.npz",
        probability=probability,
        threshold_masks=raw_masks,
        filtered_masks=final_masks,
        class_names=np.asarray(UNIFIED_DAMAGE_CLASSES),
    )
    with open(output_dir / "figure_metadata.json", "w", encoding="utf-8") as handle:
        json.dump({
            "image": str(Path(args.image).resolve()),
            "image_shape": list(image.shape[:2]),
            "run_family": family,
            "checkpoint": str(checkpoint_path.resolve()),
            "checkpoint_epoch": checkpoint.get("epoch", "unknown"),
            "operating_points_source": str(Path(args.operating_points).resolve()),
            "connectivity": 8,
            "probability_visualization": "class color with alpha * probability",
            "components": report,
        }, handle, indent=2)

    for row in report:
        print(
            f"{row['class']:>13}: tau={row['threshold']:.2f}, "
            f"A={row['min_area_pixels']} px, "
            f"components {row['components_before']} -> "
            f"{row['components_after']}"
        )
    print(f"[ppt] Figures saved to {output_dir.resolve()}")


if __name__ == "__main__":
    main()