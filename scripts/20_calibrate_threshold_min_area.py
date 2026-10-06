#!/usr/bin/env python
"""Step 20 - joint per-class calibration of threshold and minimum component area.

Why
---
`false_alarm_rate_empty_gt` flags an image as a false alarm as soon as ONE pixel
of a class is predicted where the GT is empty. Isolated specks therefore dominate
the FAR and push the per-class threshold up, at the expense of recall. An
inspection alarm only makes sense above a minimum area, so each class c gets an
operating point (tau_c, A_c): probability > tau_c, then 8-connected components
smaller than A_c (fraction of the image area) are removed.

Protocol (additive: no standard or calibrated artifact is touched)
-----------------------------------------------------------------
1. one sliding-window inference over the validation split; per image the counts
   of every (tau, A) grid point are stored in `postproc_counts.npz`;
2. the images are split into two seeded halves H0 / H1;
3. fold 0: select on H0, report on H1; fold 1: the reverse;
4. two modes per fold: `threshold_only` (A = 0, equivalent to scripts/17) and
   `threshold_min_area`; the gain of the filter is read on the held-out half;
5. the fold-0 `threshold_min_area` operating points are frozen for held-out use
   (testdev, Lincoln Street).

Selection per class: argmax Dice; optional `--max-far` keeps only grid points
with FAR <= max_far (if none, the minimum-FAR point is taken and flagged).
Ties -> larger area, then higher threshold (conservative, deterministic).

Usage
-----
    python scripts/20_calibrate_threshold_min_area.py --run-dir outputs/runs/<run>
    python scripts/20_calibrate_threshold_min_area.py --run-dir outputs/runs/<run> \
        --from-cache --max-far 0.5 --output-dir outputs/runs/<run>/eval_multilabel_postproc_far050
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.data.class_mapping import UNIFIED_DAMAGE_CLASSES
from src.data.dacl10k import list_samples
from src.eval.postprocess import COUNT_FIELDS, image_counts, metrics_from_counts
from src.utils import ensure_dir, get_device, load_config, multilabel_patch_cfg, seed_everything

DEFAULT_AREA_FRACTIONS = (0.0, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3)
CACHE_NAME = "postproc_counts.npz"
MODES = ("threshold_only", "threshold_min_area")
NOTE = (
    "Operating points selected on one validation half and reported on the other. "
    "Held-out claims must reuse the frozen fold-0 points without re-selection."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Joint per-class threshold + min-area calibration.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--checkpoint", default=None, help="default: best.pt (P1ML) / best_joint.pt (P4)")
    parser.add_argument("--split", default=None)
    parser.add_argument("--limit", type=int, default=None, help="debug: first N images")
    parser.add_argument("--thresholds", type=float, nargs="+", default=None,
                        help="default: eval.threshold_sweep of the run")
    parser.add_argument("--min-area-fractions", type=float, nargs="+", default=list(DEFAULT_AREA_FRACTIONS))
    parser.add_argument("--max-far", type=float, default=None, help="optional per-class FAR constraint")
    parser.add_argument("--seed", type=int, default=None, help="split seed (default: project.seed)")
    parser.add_argument("--output-dir", default=None, help="default: <run-dir>/eval_multilabel_postproc")
    parser.add_argument("--from-cache", action="store_true", help=f"reuse {CACHE_NAME}, skip inference")
    return parser.parse_args()


def load_step17():
    """Reuse predictor and data loading of scripts/17 (file name starts with a digit)."""
    path = Path(__file__).with_name("17_calibrate_multilabel_thresholds.py")
    spec = importlib.util.spec_from_file_location("step17_calibration", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_inference(args, cfg, run_dir: Path, thresholds, area_fractions, cache_path: Path) -> dict:
    step17 = load_step17()
    patch_cfg = multilabel_patch_cfg(cfg)
    predict, checkpoint_path, checkpoint, family = step17.build_predictor(
        cfg, run_dir, args.checkpoint, get_device()
    )
    split = args.split or str(cfg.data.dacl10k.val_split)
    samples = list_samples(cfg.data.dacl10k.root, split)
    if args.limit:
        samples = samples[: args.limit]
    print(f"[postproc] {family} | {checkpoint_path.name} | {split}: {len(samples)} images")
    print(f"[postproc] grid: {len(thresholds)} thresholds x {len(area_fractions)} min-area fractions")

    counts, names, shapes = [], [], []
    for index, sample in enumerate(samples, start=1):
        image, mask = step17.load_rgb_and_mask(sample)
        probability = predict(image, patch_cfg)
        if probability.shape != mask.shape:
            raise RuntimeError(f"Shape mismatch {probability.shape} vs {mask.shape} for {sample[0]}")
        counts.append(image_counts(probability, mask, thresholds, area_fractions))
        names.append(Path(sample[0]).name)
        shapes.append(mask.shape[1:])
        if index % 25 == 0 or index == len(samples):
            print(f"[postproc] processed {index}/{len(samples)} images")

    payload = {
        "counts": np.stack(counts),                     # [N, C, T, A, F]
        "thresholds": np.asarray(thresholds, dtype=np.float64),
        "area_fractions": np.asarray(area_fractions, dtype=np.float64),
        "image_names": np.asarray(names),
        "image_shapes": np.asarray(shapes, dtype=np.int64),
        "count_fields": np.asarray(COUNT_FIELDS),
        "checkpoint_name": np.asarray(checkpoint_path.name),
        "checkpoint_epoch": np.asarray(str(checkpoint.get("epoch", "unknown"))),
        "split": np.asarray(split),
    }
    np.savez_compressed(cache_path, **payload)
    print(f"[postproc] counts cached to {cache_path}")
    return payload


def select_class(counts_c: np.ndarray, thresholds, area_fractions, mode: str, max_far):
    """counts_c: summed [T, A, F] for one class on the calibration half."""
    metrics = metrics_from_counts(counts_c)
    area_indices = [0] if mode == "threshold_only" else range(len(area_fractions))
    candidates = [(t, a) for t in range(len(thresholds)) for a in area_indices]

    feasible = [x for x in candidates
                if max_far is None or metrics["false_alarm_rate_empty_gt"][x] <= max_far]
    if feasible:
        best = max(feasible, key=lambda x: (round(float(metrics["dice"][x]), 6),
                                            area_fractions[x[1]], thresholds[x[0]]))
    else:
        best = min(candidates, key=lambda x: (float(metrics["false_alarm_rate_empty_gt"][x]),
                                              -float(metrics["dice"][x])))
    return best, bool(feasible)


def summarize(per_class_counts: np.ndarray) -> dict:
    """per_class_counts: [C, F] at the selected points -> macro / micro metrics."""
    per_class = metrics_from_counts(per_class_counts)
    present = per_class["support_pixels"] > 0
    summed = metrics_from_counts(per_class_counts.sum(0))
    out = {f"macro_{k}_present": float(per_class[k][present].mean())
           for k in ("dice", "iou", "precision", "recall")}
    out.update({
        "micro_dice": float(summed["dice"]),
        "false_alarm_rate_empty_gt": float(summed["false_alarm_rate_empty_gt"]),
        "macro_object_precision": float(per_class["object_precision"].mean()),
        "macro_object_recall": float(per_class["object_recall"][present].mean()),
        "macro_object_recall_large": float(per_class["object_recall_large"][present].mean()),
    })
    return out


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    cfg = load_config(run_dir / "config.yaml")
    seed = int(args.seed if args.seed is not None else cfg.project.seed)
    seed_everything(seed)
    output_dir = ensure_dir(Path(args.output_dir) if args.output_dir else run_dir / "eval_multilabel_postproc")
    cache_path = output_dir / CACHE_NAME

    if args.from_cache:
        data = dict(np.load(cache_path, allow_pickle=False))
        print(f"[postproc] reusing {cache_path}")
    else:
        thresholds = sorted(args.thresholds or list(cfg.eval.threshold_sweep))
        area_fractions = sorted(set([0.0, *args.min_area_fractions]))
        data = run_inference(args, cfg, run_dir, thresholds, area_fractions, cache_path)

    counts = data["counts"]
    thresholds = data["thresholds"].tolist()
    area_fractions = data["area_fractions"].tolist()
    if area_fractions[0] != 0.0:
        raise ValueError("The min-area grid must contain 0 (threshold-only baseline).")

    permutation = np.random.default_rng(seed).permutation(counts.shape[0])
    halves = (np.sort(permutation[: len(permutation) // 2]), np.sort(permutation[len(permutation) // 2:]))

    per_class_rows, summary_rows, frozen = [], [], {}
    for fold, (calib, test) in enumerate((halves, halves[::-1])):
        calib_sum, test_sum = counts[calib].sum(0), counts[test].sum(0)  # [C, T, A, F]
        for mode in MODES:
            selected = {}
            for c, name in enumerate(UNIFIED_DAMAGE_CLASSES):
                (t, a), feasible = select_class(calib_sum[c], thresholds, area_fractions, mode, args.max_far)
                selected[name] = {"threshold": thresholds[t], "min_area_fraction": area_fractions[a],
                                  "far_constraint_met": feasible}
                for part, source in (("calib", calib_sum), ("test", test_sum)):
                    m = metrics_from_counts(source[c, t, a])
                    per_class_rows.append({
                        "fold": fold, "mode": mode, "part": part, "class": name,
                        "threshold": thresholds[t], "min_area_fraction": area_fractions[a],
                        "far_constraint_met": feasible, **{k: float(v) for k, v in m.items()},
                    })
            for part, source in (("calib", calib_sum), ("test", test_sum)):
                chosen = np.stack([
                    source[c, thresholds.index(selected[n]["threshold"]),
                           area_fractions.index(selected[n]["min_area_fraction"])]
                    for c, n in enumerate(UNIFIED_DAMAGE_CLASSES)
                ])
                summary_rows.append({"fold": fold, "mode": mode, "part": part,
                                     "n_images": int(len(calib if part == "calib" else test)),
                                     **summarize(chosen)})
            if fold == 0:
                frozen[mode] = selected

    per_class_df = pd.DataFrame(per_class_rows)
    summary_df = pd.DataFrame(summary_rows)
    per_class_df.to_csv(output_dir / "metrics_postproc_per_class.csv", index=False)
    summary_df.to_csv(output_dir / "metrics_postproc_summary.csv", index=False)

    with open(output_dir / "operating_points_frozen.json", "w", encoding="utf-8") as fh:
        json.dump({
            "pipeline": "joint per-class threshold + min-area calibration (additive)",
            "run_dir": str(run_dir.resolve()),
            "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "checkpoint": {"name": str(data["checkpoint_name"]), "epoch": str(data["checkpoint_epoch"])},
            "split": str(data["split"]),
            "split_seed": seed,
            "calibration_images": data["image_names"][halves[0]].tolist(),
            "thresholds_grid": thresholds,
            "min_area_fraction_grid": area_fractions,
            "selection": "argmax Dice; ties -> larger area, higher threshold",
            "max_far": args.max_far,
            "frozen_fold0": frozen,
            "note": NOTE,
        }, fh, indent=2, ensure_ascii=False)

    view = ["fold", "mode", "macro_dice_present", "macro_precision_present", "macro_recall_present",
            "false_alarm_rate_empty_gt", "macro_object_precision", "macro_object_recall"]
    test_view = summary_df[summary_df.part == "test"]
    print("\n--- held-out half (selection made on the other half) ---")
    print(test_view[view].round(4).to_string(index=False))
    print("\n--- mean over the two folds (held-out) ---")
    print(test_view.groupby("mode")[view[2:]].mean().round(4).to_string())
    print("\n--- frozen fold-0 operating points (threshold_min_area) ---")
    for name, point in frozen["threshold_min_area"].items():
        flag = "" if point["far_constraint_met"] else "  (FAR constraint NOT met)"
        print(f"{name:>13}: tau = {point['threshold']:.2f} | A = {point['min_area_fraction']:.0e}{flag}")
    print(f"\n[postproc] {NOTE}\n[postproc] artifacts in {output_dir.resolve()}")


if __name__ == "__main__":
    main()
