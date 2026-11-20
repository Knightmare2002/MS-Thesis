import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


parser = argparse.ArgumentParser()
parser.add_argument("folder", type=Path)
args = parser.parse_args()

if not args.folder.is_dir():
    parser.error(f"Cartella non trovata: {args.folder}")

rows = []

for path in sorted(args.folder.glob("*.json")):
    data = json.loads(path.read_text(encoding="utf-8-sig"))

    for shape_index, shape in enumerate(data.get("shapes", [])):
        if shape.get("label", "").strip().lower() != "crack_measure":
            continue

        points = np.asarray(shape.get("points", []), dtype=float)

        if (
            shape.get("shape_type") != "line"
            or points.shape != (2, 2)
            or not np.isfinite(points).all()
        ):
            raise ValueError(
                f"{path.name}, shape {shape_index}: attesa una linea con due punti"
            )

        (x1, y1), (x2, y2) = points
        width = float(np.hypot(x2 - x1, y2 - y1))

        if width <= 0:
            raise ValueError(
                f"{path.name}, shape {shape_index}: distanza nulla"
            )

        rows.append({
            "view": path.stem,
            "shape_index": shape_index,
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
            "width_px": width,
        })

if not rows:
    raise SystemExit("Nessuna linea crack_measure trovata.")

measurements = pd.DataFrame(rows)

summary = measurements.groupby("view")["width_px"].agg(
    n="size",
    min_px="min",
    q10_px=lambda s: s.quantile(0.10),
    q25_px=lambda s: s.quantile(0.25),
    median_px="median",
    q75_px=lambda s: s.quantile(0.75),
    q90_px=lambda s: s.quantile(0.90),
    max_px="max",
).reset_index()

measurements.to_csv(
    args.folder / "crack_measurements_px.csv", index=False
)
summary.to_csv(
    args.folder / "crack_width_summary_px.csv", index=False
)

print(summary.to_string(index=False))