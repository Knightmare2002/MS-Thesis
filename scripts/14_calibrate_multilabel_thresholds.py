"""Select validation-calibrated thresholds per multilabel damage channel.

Uses the per-class CSV already produced by
scripts/12_evaluate_dacl10k_multilabel_sliding.py. No inference and no
retraining are performed here.

The selected thresholds are validation-calibrated operating points, not
blind-test results.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select per-class Dice-optimal thresholds from a P1ML evaluation CSV."
    )
    parser.add_argument(
        "--eval-dir",
        required=True,
        help="Directory containing metrics_dacl10k_val_multilabel_per_class.csv",
    )
    parser.add_argument(
        "--metric",
        default="dice",
        choices=("dice", "iou", "precision", "recall"),
        help="Metric maximised independently for every class.",
    )
    parser.add_argument(
        "--output-name",
        default="thresholds_per_class_validation_calibrated.json",
        help="Output JSON filename written inside --eval-dir.",
    )
    return parser.parse_args()


def to_jsonable(value):
    """Convert NumPy/Pandas scalar values recursively into JSON-native types."""
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}

    if isinstance(value, list):
        return [to_jsonable(item) for item in value]

    if isinstance(value, tuple):
        return [to_jsonable(item) for item in value]

    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, np.floating):
        return float(value)

    if isinstance(value, np.bool_):
        return bool(value)

    return value


def main() -> None:
    args = parse_args()
    eval_dir = Path(args.eval_dir)

    csv_path = eval_dir / "metrics_dacl10k_val_multilabel_per_class.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(
            f"Per-class metric CSV not found: {csv_path}. "
            "Run scripts/12_evaluate_dacl10k_multilabel_sliding.py first."
        )

    table = pd.read_csv(csv_path)

    required = {"threshold", "class", args.metric}
    missing = required - set(table.columns)
    if missing:
        raise ValueError(
            f"CSV does not contain required columns {sorted(missing)}. "
            f"Available columns: {list(table.columns)}"
        )

    selected_rows = (
        table.sort_values(
            ["class", args.metric, "threshold"],
            ascending=[True, False, True],
        )
        .groupby("class", as_index=False)
        .first()
        .sort_values("class")
        .reset_index(drop=True)
    )

    thresholds = {
        row["class"]: float(row["threshold"])
        for _, row in selected_rows.iterrows()
    }

    selected_metrics = []
    for _, row in selected_rows.iterrows():
        selected_metrics.append(
            {
                "class": str(row["class"]),
                "threshold": float(row["threshold"]),
                "selected_metric": args.metric,
                "selected_value": float(row[args.metric]),
                "dice": float(row["dice"]),
                "iou": float(row["iou"]),
                "precision": float(row["precision"]),
                "recall": float(row["recall"]),
                "support_pixels": float(row["support_pixels"]),
                "predicted_pixels": float(row["predicted_pixels"]),
                "false_alarm_rate_empty_gt": float(row["false_alarm_rate_empty_gt"]),
            }
        )

    metadata_columns = [
        "dataset",
        "patch_size",
        "stride",
        "blend_mode",
        "checkpoint_name",
        "checkpoint_epoch",
    ]
    metadata = {
        column: table.iloc[0][column]
        for column in metadata_columns
        if column in table.columns
    }

    payload = {
        "calibration_split_note": (
            "Thresholds selected on the evaluation validation split. "
            "They are validation-calibrated operating thresholds, not blind-test results."
        ),
        "selection_metric": args.metric,
        "candidate_thresholds": sorted(float(value) for value in table["threshold"].unique()),
        "thresholds_per_class": thresholds,
        "selected_rows": selected_metrics,
        "evaluation_metadata": metadata,
    }

    output_path = eval_dir / args.output_name
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(
            to_jsonable(payload),
            handle,
            indent=2,
            ensure_ascii=False,
        )

    print("\n=== Per-class validation-calibrated thresholds ===")
    print(
        selected_rows[
            [
                "class",
                "threshold",
                "dice",
                "iou",
                "precision",
                "recall",
                "false_alarm_rate_empty_gt",
            ]
        ].to_string(index=False)
    )
    print(f"\nSaved: {output_path.resolve()}")


if __name__ == "__main__":
    main()