"""T9: parameter-free baselines; primary = ALL views, --lavo = leave-annotated-views-out.

Primary (default): every logged view is used, annotated ones included (deployment setting:
the annotated views are the best views of the ROI and exist at inference time).
Sensitivity (--lavo): for every ROI point all cameras annotated in its GROUP are removed,
which removes the GT-source image and often the best views; declare it as such.
Methods (scores in [0,1], NaN when fewer than --min-views used views):
  best_view       view with minimum effective g = gsd_frontal*map_scale/sqrt(max(cos,min_cos))
  random_view     one non-annotated view drawn uniformly (seeded, per point)
  mean, max       over the non-annotated views
  naive_bayes     sigmoid(logit rho + sum_v [logit p_v - logit rho]), rho = calibration prevalence
  annotated_view  annotated view (of the point's own block) with min g: the 2D-H1 view lifted to 3D
--dacl-thresholds also writes <out>_thresholds_dacl.json: the frozen calibrated DACL thresholds
of the T13 run for every method (zero-shot, as in H1). C_cal-frozen thresholds via
freeze_direction_a are impossible for spalling (no C_cal positives): AP is the primary metric there.
Also writes g_min / cos_median / count over the non-annotated views, so evaluate_direction_a
(strata) and freeze_direction_a (C_cal thresholds) accept the file unchanged in format.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from damage3d.classes import DAMAGE_CLASSES, NUM_CLASSES  # noqa: E402
from damage3d.experiment_data import read_json, write_json, regions, provenance  # noqa: E402
from damage3d.observations import iter_observations, probs_of  # noqa: E402
from damage3d.run_state import sha256_file  # noqa: E402

METHODS = ("best_view", "random_view", "mean", "max", "naive_bayes", "annotated_view")


def logit(p):
    return np.log(p) - np.log1p(-p)


def segments(pos):
    """Start index and length of runs of equal sorted ``pos``."""
    starts = np.r_[0, np.flatnonzero(pos[1:] != pos[:-1]) + 1]
    return starts, np.diff(np.r_[starts, len(pos)])


def pick(pos, key):
    """Index of the record with minimum ``key`` for each point (records sorted by pos)."""
    order = np.lexsort((key, pos))
    starts, _ = segments(pos[order])
    return order[starts]


def compute(run, d, rows, rho, p_clip, min_cos, min_views, seed, lavo=False):
    meta = read_json(Path(run) / "observations/meta.json")
    cam_of = {c["stem"]: int(c["camera"]) for c in meta["cameras"]}
    n_cam = max(cam_of.values()) + 1
    groups = np.unique(d["region"])
    gidx = np.searchsorted(groups, d["region"])
    excl = np.zeros((len(groups), n_cam), bool)
    own = np.zeros((len(d["source_index"]), 2), np.int64)  # annotated cameras of the point's block
    for r in rows.values():
        for s in (r["view1"].strip(), r["view2"].strip()):
            if s not in cam_of:
                raise ValueError(f"Annotated view {s} not in run cameras.")
            if lavo and r["group"] in groups:
                excl[np.searchsorted(groups, r["group"]), cam_of[s]] = True
    for b, r in rows.items():
        m = d["block"] == b
        own[m] = cam_of[r["view1"].strip()], cam_of[r["view2"].strip()]

    n = len(d["source_index"])
    out = {m: np.full((n, NUM_CLASSES), np.nan, np.float32) for m in METHODS}
    out.update(g_min=np.full(n, np.nan), cos_median=np.full(n, np.nan), count=np.zeros(n, np.int32))
    lrho = logit(rho)
    rng = np.random.default_rng(seed)
    for rec in iter_observations(run, d["source_index"]):
        pos = np.searchsorted(d["source_index"], rec["source_index"])
        cos = np.abs(rec["cos_theta"].astype(np.float64))
        g = rec["gsd_frontal"] * rec["map_scale"] / np.sqrt(np.maximum(cos, min_cos))
        p = np.clip(probs_of(rec).astype(np.float64), p_clip, 1 - p_clip)
        cam = rec["camera"].astype(np.int64)
        # circular reference: own annotated views only
        a = (cam == own[pos, 0]) | (cam == own[pos, 1])
        if a.any():
            k = pick(pos[a], g[a])
            out["annotated_view"][pos[a][k]] = p[a][k]
        keep = ~excl[gidx[pos], cam]
        pos, cos, g, p = pos[keep], cos[keep], g[keep], p[keep]
        if not len(pos):
            continue
        starts, cnt = segments(pos)
        pts = pos[starts]
        out["count"][pts] = cnt
        ok = cnt >= min_views
        out["g_min"][pts] = np.minimum.reduceat(g, starts)
        oc = np.lexsort((cos, pos))
        cs = cos[oc]
        out["cos_median"][pts] = (cs[starts + (cnt - 1) // 2] + cs[starts + cnt // 2]) / 2
        sel = pts[ok]
        out["mean"][sel] = (np.add.reduceat(p, starts, axis=0) / cnt[:, None])[ok]
        out["max"][sel] = np.maximum.reduceat(p, starts, axis=0)[ok]
        llr = np.add.reduceat(logit(p) - lrho, starts, axis=0) + lrho
        out["naive_bayes"][sel] = (1 / (1 + np.exp(-llr)))[ok]
        out["best_view"][sel] = p[pick(pos, g)][ok]
        out["random_view"][sel] = p[pick(pos, rng.random(len(pos)))][ok]
    few = out["count"] < min_views
    out["annotated_view"][few] = np.nan  # same support as the deployable methods
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--regions", type=Path, required=True, help="lift_gt.py output")
    ap.add_argument("--splits", type=Path, required=True)
    ap.add_argument("--buffer", type=float, required=True)
    ap.add_argument("--lavo", action="store_true", help="sensitivity: drop the annotated views of the group")
    ap.add_argument("--min-views", type=int, default=2, help="common support: >= N used views")
    ap.add_argument("--p-clip", type=float, default=1e-4)
    ap.add_argument("--min-cos", type=float, default=0.0872)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dacl-thresholds", type=Path, help="probability_calibration.json of the T13 run")
    ap.add_argument("--out", type=Path, required=True, help="scores .npz (+ .json, _params.json)")
    a = ap.parse_args(argv)
    params_path = a.out.with_name(a.out.stem + "_params.json")
    for p in (a.out, a.out.with_suffix(".json"), params_path):
        if p.exists():
            raise FileExistsError(f"{p}: choose a new output.")
    d = regions(a.regions, a.buffer)
    with np.load(a.regions, allow_pickle=False) as f:  # regions() returns the contract keys only
        order = np.argsort(f["source_index"])
        d["block"] = f["block"][order].astype(int)
    with open(a.splits, newline="", encoding="utf-8-sig") as f:
        rows = {int(r["block_id"]): r for r in csv.DictReader(f)}
    prov = provenance(a.run, a.regions)
    prev = read_json(a.run / "observations/meta.json").get("calibration_prevalence")
    if not prev:
        raise ValueError("meta.json has no calibration_prevalence: run with --calibration (files+platt).")
    rho = np.array([prev[c] for c in DAMAGE_CLASSES])
    params = {"methods": list(METHODS),
              "views": "lavo: annotated cameras of the point's group removed" if a.lavo else "all",
              "min_views": a.min_views, "p_clip": a.p_clip, "min_cos": a.min_cos, "seed": a.seed,
              "rho": dict(zip(DAMAGE_CLASSES, rho.tolist()))}
    write_json(params_path, params)
    out = compute(a.run, d, rows, rho, a.p_clip, a.min_cos, a.min_views, a.seed, a.lavo)
    with a.out.open("wb") as f:
        np.savez_compressed(f, source_index=d["source_index"], classes=np.array(DAMAGE_CLASSES), **out)
    write_json(a.out.with_suffix(".json"), {
        "provenance": prov, "params_sha256": sha256_file(params_path), "parameters": params,
        "score_sha256": sha256_file(a.out), "fraction": 1, "seed": a.seed, "buffer": a.buffer,
        "methods": list(METHODS), "oracle": False,
        "note": ("LAVO sensitivity: GT-source views removed." if a.lavo else
                 "All views: annotated views share the image (and projection) with the GT; "
                 "they favour view-centred methods, so fusion gains are conservative.")})
    if a.dacl_thresholds:
        from src.eval.probability_calibration import thresholds_from_json
        thr = dict(zip(DAMAGE_CLASSES, thresholds_from_json(a.dacl_thresholds, DAMAGE_CLASSES).tolist()))
        write_json(a.out.with_name(a.out.stem + "_thresholds_dacl.json"),
                   {m: thr for m in METHODS})
    cov = {m: float(np.isfinite(out[m][:, 0]).mean()) for m in METHODS}
    print(json.dumps({"point_coverage": cov, "count_median": float(np.median(out["count"]))}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
