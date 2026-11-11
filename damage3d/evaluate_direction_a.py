"""Evaluate frozen methods on T; coverage, common support, strata, paired region bootstrap.

Threshold JSON: {method: {class: threshold}}. Bins JSON:
{"g_edges":[...], "incidence_edges":[0,30,60,90]}; edges must be frozen on C/geometry.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np

from .classes import DAMAGE_CLASSES
from .experiment_data import read_json, write_json, regions
from .experiment_metrics import binary_metrics, paired_bootstrap
from .fusion_benchmark import METHODS
from .run_state import sha256_file


def main(argv=None):
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
    args = ap.parse_args(argv)
    if args.reference not in args.methods or not set(args.methods) <= set(METHODS) | {"oracle_best_view"}:
        raise ValueError("Choose known methods and include the reference.")
    d = regions(args.regions, args.buffer)
    meta = read_json(args.scores.with_suffix(".json"))
    if (meta["score_sha256"] != sha256_file(args.scores) or
            meta["provenance"]["regions_sha256"] != sha256_file(args.regions)):
        raise ValueError("Score/region provenance mismatch.")
    thresholds, bins = read_json(args.thresholds), read_json(args.bins)
    if "_provenance" in thresholds:
        tp = thresholds["_provenance"]
        if (tp["split"] != "C_cal" or tp["regions_sha256"] != sha256_file(args.regions) or
                tp["params_sha256"] != meta["params_sha256"]):
            raise ValueError("Thresholds were frozen for different regions or model parameters.")
    with np.load(args.scores, allow_pickle=False) as f:
        scores = {k: f[k] for k in f.files}
    if not np.array_equal(scores["source_index"], d["source_index"]):
        raise ValueError("Source-index alignment mismatch.")
    result = {"scores_sha256": sha256_file(args.scores), "thresholds_sha256": sha256_file(args.thresholds),
              "bins_sha256": sha256_file(args.bins), "split": "T", "seed": args.seed,
              "reference": args.reference, "classes": {},
              "note": "CI is candidate minus reference; NLL/Brier lower is better. AP is step PR area."}
    for c, name in enumerate(DAMAGE_CLASSES):
        base = (d["split"] == "T") & (d["labels"][:, c] >= 0)
        if not base.any():
            result["classes"][name] = {"status": "no T labels"}
            continue
        available = {m: base & np.isfinite(scores[m][:, c]) for m in args.methods}
        common = np.logical_and.reduce(list(available.values()))
        coverage = {m: float(d["area"][ok].sum()/d["area"][base].sum()) for m, ok in available.items()}
        table = {"area_coverage": coverage, "common_area": float(d["area"][common].sum()),
                 "labelled_area": float(d["area"][base].sum()), "n_regions": len(np.unique(d["region"][common])),
                 "methods": {}, "strata": {}}
        thrs = {}
        for m in args.methods:
            thrs[m] = .5 if m == "oracle_best_view" else float(thresholds[m][name])
            if not 0 < thrs[m] < 1:
                raise ValueError("Thresholds must be in (0,1).")
        y, area, r = d["labels"][common, c], d["area"][common], d["region"][common]
        ref = scores[args.reference][common, c]
        for m in args.methods:
            p = scores[m][common, c]
            metrics = binary_metrics(y, p, area, thrs[m])
            baseline = binary_metrics(y, ref, area, thrs[args.reference])
            table["methods"][m] = {"metrics": metrics, "delta": {k: metrics[k]-baseline[k] for k in metrics}}
            if len(np.unique(r)) >= 2:
                table["methods"][m]["paired_region_bootstrap"] = paired_bootstrap(
                    y, p, ref, area, r, thrs[m], thrs[args.reference], args.bootstrap, args.seed)
        for field, key, value in (("g_min", "g_edges", scores["g_min"]),
                                  ("incidence", "incidence_edges",
                                   np.rad2deg(np.arccos(np.clip(scores["cos_median"], 0, 1))))):
            edges = np.asarray(bins[key], float)
            if len(edges) < 2 or not np.isfinite(edges).all() or np.any(np.diff(edges) <= 0):
                raise ValueError("Frozen bin edges must be finite and strictly increasing.")
            rows = []
            for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
                ok = common & (value >= lo) & (value <= hi if i == len(edges)-2 else value < hi)
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
                        x = np.array([(r["low"]+r["high"])/2 for r in valid])
                        yv = np.array([r["methods"][m][metric] for r in valid])
                        w = np.array([r["area"] for r in valid])
                        xc = x - np.average(x, weights=w)
                        slopes[m][metric] = float(np.sum(w*xc*yv)/np.sum(w*xc*xc))
                    else:
                        slopes[m][metric] = None
            table["strata"][field+"_descriptive_linear_slopes"] = slopes
            table["strata"]["slope_note"] = "Area-weighted bin-midpoint OLS; >=3 populated bins; descriptive, not causal or a significance test."
        result["classes"][name] = table
    write_json(args.out, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
