#!/usr/bin/env python
"""Step 1 - EDA on CrackSeg9k and dacl10k.

Produces (under <output_dir>/eda/):
    crackseg9k_per_image.csv        per-image resolution + crack pixel ratio
    crackseg9k_summary.csv          describe() of the numeric columns
    dacl10k_per_image.csv           per-image stats + multi-label presence flags
    dacl10k_class_frequency.csv     class frequency, imbalance factor
    dataset_comparison.csv          the CrackSeg9k vs dacl10k table for the slides
    figures/*.png                   plots ready to paste into the PPT

Usage
-----
    python scripts/01_eda.py --config configs/config.yaml
    python scripts/01_eda.py --config configs/config.yaml --sample 500   # quick pass
    python scripts/01_eda.py --config configs/config.yaml --skip dacl10k
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: write PNGs, never open a window
import matplotlib.pyplot as plt
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.data import crackseg9k as cs9k
from src.data import dacl10k as d10k
from src.data.stats import (
    crackseg9k_stats,
    dacl10k_class_frequency,
    dacl10k_stats,
    describe_numeric,
)
from src.utils import ensure_dir, load_config, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="EDA on CrackSeg9k and dacl10k")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--set", nargs="*", default=[], help="config overrides key=value")
    parser.add_argument("--sample", type=int, default=None, help="analyse only N random images per dataset")
    parser.add_argument("--skip", nargs="*", default=[], choices=["crackseg9k", "dacl10k"])
    return parser.parse_args()


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #
def plot_crack_ratio(df: pd.DataFrame, title: str, path: Path) -> None:
    """Histogram of the crack-pixel share (log y: the distribution is very skewed)."""
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(df["crack_ratio"] * 100, bins=60, color="#2b6cb0")
    ax.set_yscale("log")
    ax.set_xlabel("crack pixels per image [%]")
    ax.set_ylabel("number of images (log)")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_resolutions(df: pd.DataFrame, title: str, path: Path) -> None:
    """Scatter of width vs height: shows whether a single resize is acceptable."""
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(df["width"], df["height"], s=6, alpha=0.3, color="#2f855a")
    ax.set_xlabel("width [px]")
    ax.set_ylabel("height [px]")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_class_frequency(freq: pd.DataFrame, path: Path) -> None:
    """Horizontal bar chart of dacl10k class frequency, damages vs components."""
    colors = ["#c53030" if g == "damage" else "#4a5568" for g in freq["group"]]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(freq["class"][::-1], freq["n_images"][::-1], color=colors[::-1])
    ax.set_xlabel("number of images containing the class")
    ax.set_title("dacl10k - class frequency (red = damage, grey = component)")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    args = parse_args()
    cfg = load_config(args.config, args.set)
    seed_everything(cfg.project.seed)

    eda_dir = ensure_dir(Path(cfg.project.output_dir) / "eda")
    figures_dir = ensure_dir(eda_dir / "figures")
    comparison_rows = []

    # ---- CrackSeg9k --------------------------------------------------------
    if "crackseg9k" not in args.skip:
        pairs = cs9k.list_pairs(cfg.data.crackseg9k.images_dir, cfg.data.crackseg9k.masks_dir)
        print(f"[CrackSeg9k] {len(pairs)} image/mask pairs")
        df = crackseg9k_stats(pairs, sample=args.sample, seed=cfg.project.seed)
        df.to_csv(eda_dir / "crackseg9k_per_image.csv", index=False)
        describe_numeric(df, ["width", "height", "crack_ratio"]).to_csv(
            eda_dir / "crackseg9k_summary.csv"
        )
        plot_crack_ratio(df, "CrackSeg9k - crack pixel share", figures_dir / "crackseg9k_crack_ratio.png")
        plot_resolutions(df, "CrackSeg9k - resolutions", figures_dir / "crackseg9k_resolutions.png")

        comparison_rows.append(
            {
                "dataset": "CrackSeg9k",
                "task": "binary crack segmentation",
                "n_images": len(df),
                "median_resolution": f"{int(df.width.median())}x{int(df.height.median())}",
                "mean_crack_ratio_%": round(df.crack_ratio.mean() * 100, 3),
                "median_crack_ratio_%": round(df.crack_ratio.median() * 100, 3),
                "empty_masks_%": round(df.is_empty.mean() * 100, 2),
                "n_classes": 1,
            }
        )
        print(df[["width", "height", "crack_ratio"]].describe())

    # ---- dacl10k -----------------------------------------------------------
    if "dacl10k" not in args.skip:
        samples = d10k.list_samples(cfg.data.dacl10k.root, cfg.data.dacl10k.train_split)
        print(f"[dacl10k/{cfg.data.dacl10k.train_split}] {len(samples)} annotated images")
        df = dacl10k_stats(samples, compute_crack_ratio=True, sample=args.sample, seed=cfg.project.seed)
        df.to_csv(eda_dir / "dacl10k_per_image.csv", index=False)

        freq = dacl10k_class_frequency(df)
        freq.to_csv(eda_dir / "dacl10k_class_frequency.csv", index=False)
        plot_class_frequency(freq, figures_dir / "dacl10k_class_frequency.png")
        plot_crack_ratio(df, "dacl10k - crack pixel share (Crack + ACrack)", figures_dir / "dacl10k_crack_ratio.png")
        plot_resolutions(df, "dacl10k - resolutions", figures_dir / "dacl10k_resolutions.png")

        comparison_rows.append(
            {
                "dataset": "dacl10k",
                "task": "multi-label damage + component segmentation",
                "n_images": len(df),
                "median_resolution": f"{int(df.width.median())}x{int(df.height.median())}",
                "mean_crack_ratio_%": round(df.crack_ratio.mean() * 100, 3),
                "median_crack_ratio_%": round(df.crack_ratio.median() * 100, 3),
                "empty_masks_%": round((df.crack_pixels == 0).mean() * 100, 2),
                "n_classes": 19,
            }
        )
        print(freq.head(10).to_string(index=False))

    # ---- Comparison table for the slides -----------------------------------
    if comparison_rows:
        comparison = pd.DataFrame(comparison_rows)
        comparison.to_csv(eda_dir / "dataset_comparison.csv", index=False)
        print("\n=== Dataset comparison ===")
        print(comparison.to_string(index=False))

    print(f"\nArtifacts written to {eda_dir.resolve()}")


if __name__ == "__main__":
    main()
