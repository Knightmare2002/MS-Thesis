"""Evaluate frozen methods on T; coverage, common support, strata, paired region bootstrap.

Threshold JSON: {method: {class: threshold}}. Bins JSON:
{"g_edges":[...], "incidence_edges":[0,30,60,90]}; edges must be frozen on C/geometry.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path
import numpy as np

from .classes import DAMAGE_CLASSES
from .experiment_data import read_json, write_json, regions
from .experiment_metrics import binary_metrics, paired_bootstrap
from .fusion_benchmark import METHODS
from .run_state import sha256_file


def log(msg: str, t0: float) -> None:
    print(f"[{time.time() - t0:6.1f}s] {msg}", flush=True)



def per_unit_metrics(d, scores, methods, thresholds, class_index, base, common,
                     reference, unit):
    records = {}
    for value in np.unique(unit[d["split"] == "T"]):
        member = (d["split"] == "T") & (unit == value)
        known = member & base
        ok = member & common
        known_area = float(d["area"][known].sum())
        row = {
            "n_total": int(member.sum()), "n_labelled": int(known.sum()),
            "n_unknown": int((member & (d["labels"][:, class_index] < 0)).sum()),
            "n_common": int(ok.sum()), "labelled_area": known_area,
            "common_area": float(d["area"][ok].sum()),
            "positive_labelled_area": float(d["area"][known & (d["labels"][:, class_index] == 1)].sum()),
            "area_coverage": {}, "methods": {},
            "status": "ok" if ok.any() else "no_common_support"
        }
        for method in methods:
            available = known & np.isfinite(scores[method][:, class_index])
            row["area_coverage"][method] = (
                float(d["area"][available].sum()) / known_area if known_area else None)
        if ok.any():
            y = d["labels"][ok, class_index]
            w = d["area"][ok]
            baseline = binary_metrics(y, scores[reference][ok, class_index], w, thresholds[reference])
            for method in methods:
                p = scores[method][ok, class_index]
                pred = p >= thresholds[method]
                metrics = binary_metrics(y, p, w, thresholds[method])
                tp = float(w[(y == 1) & pred].sum())
                fp = float(w[(y == 0) & pred].sum())
                fn = float(w[(y == 1) & ~pred].sum())
                tn = float(w[(y == 0) & ~pred].sum())
                row["methods"][method] = {
                    "metrics": metrics,
                    "delta": {k: metrics[k] - baseline[k] for k in metrics},
                    "threshold": thresholds[method],
                    "confusion_area": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
                    "precision": tp / (tp + fp) if tp + fp else None,
                    "recall": tp / (tp + fn) if tp + fn else None
                }
        records[str(value)] = row
    return records

def main(argv=None):
    t0 = time.time()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scores", type=Path, required=True)
    ap.add_argument("--regions", required=True)
    ap.add_argument("--buffer", type=float, required=True)
    ap.add_argument("--thresholds", required=True, help="explicit frozen per-method/per-class thresholds")
    ap.add_argument("--bins", required=True)
    ap.add_argument("--methods", nargs="+", default=list(METHODS))
    ap.add_argument("--reference", default="max")
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-region", action="store_true", help="metrics for each region/group")
    ap.add_argument("--per-block", action="store_true", help="diagnostics for each block; bootstrap remains by region")
    args = ap.parse_args(argv)
    if args.bootstrap < 0:
        ap.error("--bootstrap must be >= 0")

    log(f"Loading regions from {args.regions} (buffer={args.buffer})...", t0)
    d = regions(args.regions, args.buffer)
    log(f"Loaded {len(d['source_index']):,} points.", t0)
    blocks = None
    if args.per_block:
        with np.load(args.regions, allow_pickle=False) as f:
            if "block" not in f.files:
                raise ValueError("--per-block requires block array in regions NPZ")
            order = np.argsort(f["source_index"])
            if not np.array_equal(f["source_index"][order], d["source_index"]):
                raise ValueError("Block source-index mismatch")
            blocks = f["block"][order]


    log("Verifying SHA-256 provenance hashes for scores, regions, and thresholds...", t0)
    meta = read_json(args.scores.with_suffix(".json"))
    known = set(meta.get("methods", METHODS)) | {"oracle_best_view"}
    if args.reference not in args.methods or not set(args.methods) <= known:
        raise ValueError("Choose methods present in the score file and include the reference.")
    scores_hash = sha256_file(args.scores)
    regions_hash = sha256_file(args.regions)
    if (meta["score_sha256"] != scores_hash or
            meta["provenance"]["regions_sha256"] != regions_hash):
        raise ValueError("Score/region provenance mismatch.")

    thresholds, bins = read_json(args.thresholds), read_json(args.bins)
    excluded = meta.get("parameters", {}).get("excluded_classes", {})
    if excluded != thresholds.get("_provenance", {}).get("excluded_classes", {}):
        raise ValueError("Thresholds and scores use different H3 class scopes.")
    if "_provenance" in thresholds:
        tp = thresholds["_provenance"]
        if (tp["split"] != "C_cal" or tp["regions_sha256"] != regions_hash or
                tp["params_sha256"] != meta["params_sha256"]):
            raise ValueError("Thresholds were frozen for different regions or model parameters.")
    log("Provenance verification passed.", t0)

    log(f"Loading score arrays from {args.scores}...", t0)
    with np.load(args.scores, allow_pickle=False) as f:
        scores = {k: f[k] for k in f.files}
    if not np.array_equal(scores["source_index"], d["source_index"]):
        raise ValueError("Source-index alignment mismatch.")
    log(f"Loaded {len(scores)} arrays; evaluating {len(args.methods)} methods (reference='{args.reference}').", t0)

    result = {"scores_sha256": scores_hash, "thresholds_sha256": sha256_file(args.thresholds),
              "bins_sha256": sha256_file(args.bins), "split": "T", "seed": args.seed,
              "reference": args.reference, "classes": {},
              "note": "CI is candidate minus reference; NLL/Brier lower is better. AP is step PR area."}

    n_classes = len(DAMAGE_CLASSES)
    for c, name in enumerate(DAMAGE_CLASSES):
        if name in excluded:
            result["classes"][name] = {
                "status": "excluded",
                "reasons": excluded[name],
            }
            continue
        log(f"[{c + 1}/{n_classes}] Evaluating class '{name}' on split T...", t0)
        base = (d["split"] == "T") & (d["labels"][:, c] >= 0)
        if not base.any():
            log(f"  -> Skipping '{name}': no labelled T points found.", t0)
            result["classes"][name] = {"status": "no T labels"}
            continue

        available = {m: base & np.isfinite(scores[m][:, c]) for m in args.methods}
        common = np.logical_and.reduce(list(available.values()))
        n_common = int(common.sum())
        n_regions = len(np.unique(d["region"][common]))
        log(f"  Common support: {n_common:,} points across {n_regions} regions.", t0)

        coverage = {m: float(d["area"][ok].sum() / d["area"][base].sum()) for m, ok in available.items()}
        table = {"area_coverage": coverage, "common_area": float(d["area"][common].sum()),
                 "labelled_area": float(d["area"][base].sum()), "n_regions": n_regions,
                 "methods": {}, "strata": {}}
        thrs = {}
        for m in args.methods:
            thrs[m] = .5 if m == "oracle_best_view" else float(thresholds[m][name])
            if not 0 < thrs[m] < 1:
                raise ValueError("Thresholds must be in (0,1).")

        y, area, r = d["labels"][common, c], d["area"][common], d["region"][common]
        ref = scores[args.reference][common, c]
        baseline = binary_metrics(y, ref, area, thrs[args.reference])

        for idx_m, m in enumerate(args.methods, 1):
            log(f"  Method [{idx_m}/{len(args.methods)}] '{m}' (thr={thrs[m]:.3f}, bootstrap={args.bootstrap})...", t0)
            p = scores[m][common, c]
            metrics = binary_metrics(y, p, area, thrs[m])
            table["methods"][m] = {"metrics": metrics, "delta": {k: metrics[k] - baseline[k] for k in metrics}}
            if args.bootstrap > 0 and len(np.unique(r)) >= 2:
                table["methods"][m]["paired_region_bootstrap"] = paired_bootstrap(
                    y, p, ref, area, r, thrs[m], thrs[args.reference], args.bootstrap, args.seed)

        if args.per_region:
            table["per_region"] = per_unit_metrics(
                d, scores, args.methods, thrs, c, base, common, args.reference, d["region"])
        if args.per_block:
            table["per_block"] = per_unit_metrics(
                d, scores, args.methods, thrs, c, base, common, args.reference, blocks)

        log(f"  Computing stratified metrics (g_min & incidence) for '{name}'...", t0)
        for field, key, value in (("g_min", "g_edges", scores["g_min"]),
                                  ("incidence", "incidence_edges",
                                   np.rad2deg(np.arccos(np.clip(scores["cos_median"], 0, 1))))):
            edges = np.asarray(bins[key], float)
            if len(edges) < 2 or not np.isfinite(edges).all() or np.any(np.diff(edges) <= 0):
                raise ValueError("Frozen bin edges must be finite and strictly increasing.")
            rows = []
            for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
                ok = common & (value >= lo) & (value <= hi if i == len(edges) - 2 else value < hi)
                rows.append({"low": lo, "high": hi, "n": int(ok.sum()), "area": float(d["area"][ok].sum()),
                             "methods": {m: binary_metrics(d["labels"][ok, c], scores[m][ok, c],
                                                          d["area"][ok], thrs[m]) for m in args.methods}})
            table["strata"][field] = rows
            table["strata"][field + "_outside_n"] = int(np.sum(common & ((value < edges[0]) | (value > edges[-1]))))
            slopes = {}
            for m in args.methods:
                slopes[m] = {}
                for metric in ("iou", "f1", "ap", "nll", "brier"):
                    valid = [r for r in rows if np.isfinite(r["methods"][m][metric]) and r["area"] > 0]
                    if len(valid) >= 3:
                        x = np.array([(r["low"] + r["high"]) / 2 for r in valid])
                        yv = np.array([r["methods"][m][metric] for r in valid])
                        w = np.array([r["area"] for r in valid])
                        xc = x - np.average(x, weights=w)
                        slopes[m][metric] = float(np.sum(w * xc * yv) / np.sum(w * xc * xc))
                    else:
                        slopes[m][metric] = None
            table["strata"][field + "_descriptive_linear_slopes"] = slopes
            table["strata"]["slope_note"] = "Area-weighted bin-midpoint OLS; >=3 populated bins; descriptive, not causal or a significance test."
        result["classes"][name] = table

    log(f"Writing evaluation report to {args.out}...", t0)
    write_json(args.out, result)
    log("Done!", t0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())