#!/usr/bin/env python
"""Step 6 - create a compact P0 versus P1 comparison table.

P0:
    CrackSeg9K in-domain test result.

P1:
    DACL10K official validation result.

The datasets and splits are not identical, therefore this script does not claim
a direct statistical ranking. It creates a transparent experimental summary for
the thesis and presentation.

Usage
-----
    python scripts/06_compare_p0_p1.py ^
        --p0-run-dir outputs/runs/unet_r34_512 ^
        --p1-run-dir outputs/runs/p1_dacl10k_unet_r34_512
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))


METRIC_COLUMNS = [
    "dataset",
    "threshold",
    "dice",
    "iou",
    "precision",
    "recall",
    "dice_image_mean",
    "iou_image_mean",
    "n_images_with_crack",
    "n_images_empty_gt",
    "false_alarm_rate_empty_gt",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare P0 and P1 evaluation tables.")
    parser.add_argument("--p0-run-dir", required=True, help="P0 CrackSeg9K run directory")
    parser.add_argument("--p1-run-dir", required=True, help="P1 DACL10K run directory")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="operating threshold to extract from both sweeps",
    )
    return parser.parse_args()


def select_threshold(metrics_path: Path, threshold: float) -> pd.Series:
    """Load a metrics table and select exactly one operating threshold."""
    if not metrics_path.exists():
        raise FileNotFoundError(f"Metrics file not found: {metrics_path}")

    table = pd.read_csv(metrics_path)
    selected = table.loc[(table["threshold"] - threshold).abs() < 1e-9]

    if len(selected) != 1:
        available = ", ".join(map(str, sorted(table["threshold"].unique())))
        raise ValueError(
            f"Expected one row at threshold {threshold} in {metrics_path}; "
            f"available thresholds: {available}"
        )
    return selected.iloc[0]


def main() -> None:
    args = parse_args()
    p0_dir = Path(args.p0_run_dir)
    p1_dir = Path(args.p1_run_dir)

    p0 = select_threshold(p0_dir / "eval" / "metrics.csv", args.threshold)
    p1 = select_threshold(p1_dir / "eval" / "metrics_dacl10k_val.csv", args.threshold)

    comparison = pd.DataFrame(
        [
            {
                "phase": "P0",
                "protocol": "CrackSeg9K held-out random image-level test",
                "task": "binary crack segmentation",
                **{column: p0.get(column) for column in METRIC_COLUMNS},
            },
            {
                "phase": "P1",
                "protocol": "DACL10K official validation split",
                "task": "binary Crack + ACrack segmentation",
                **{column: p1.get(column) for column in METRIC_COLUMNS},
            },
        ]
    )

    output_dir = p1_dir / "comparison"
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "p0_p1_operating_point.csv"
    comparison.to_csv(out_path, index=False)

    display_columns = [
        "phase",
        "dataset",
        "threshold",
        "dice",
        "iou",
        "precision",
        "recall",
        "false_alarm_rate_empty_gt",
    ]
    print(comparison[display_columns].to_string(index=False))
    print(f"\nComparison written to {out_path.resolve()}")
    print(
        "\nInterpretation rule: P0 and P1 use different datasets and split protocols. "
        "Read this table as a domain/protocol baseline summary, not as a direct "
        "same-test-set model ranking."
    )


if __name__ == "__main__":
    main()