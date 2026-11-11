"""Offline baselines/ablations on identical visibility observations; nested camera subsets."""
from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import numpy as np

from .classes import DAMAGE_CLASSES, NUM_CLASSES
from .detectability import FusionModel
from .experiment_data import (read_json, write_json, regions, provenance, match_rows,
                              validate_fit_provenance, validate_detector)
from .observations import iter_observations, probs_of
from .run_state import sha256_file

METHODS = ("mean", "max", "angle_range", "distance_range", "distance_weighted",
           "naive_bayes", "tempered_bayes", "censored_untempered", "censored",
           "pi_weighted_mean", "pi_weighted_llr")


def camera_subset(cameras, fraction, seed):
    """Same permutation across fractions: global camera subset, not per-point sampling."""
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be in (0,1].")
    cams = np.sort(np.unique(cameras))
    return np.sort(np.random.default_rng(seed).permutation(cams)[:max(1, int(np.ceil(len(cams)*fraction)))])


def weighted_mean(p, w):
    w = np.broadcast_to(np.asarray(w), p.shape)
    valid = np.isfinite(p)
    den = np.where(valid, w, 0).sum(axis=0)
    num = np.where(valid, p*w, 0).sum(axis=0)
    return np.divide(num, den, out=np.full(p.shape[1], np.nan), where=den > 0)


def point_scores(obs, model):
    """One point, cameras in canonical order. Distances are Euclidean source range."""
    p = probs_of(obs).astype(float)
    count = np.isfinite(p).sum(axis=0)[None, :]
    g = model.view_gsd(obs["gsd_frontal"], obs["cos_theta"], obs["map_scale"])
    pi = model.pod(g)
    plain = replace(model, mode="llr")
    lam = plain.view_llr(p, None)
    cens = model.view_llr(p, g)
    raw = np.nansum(lam, axis=0, keepdims=True)
    mix = np.nansum(cens, axis=0, keepdims=True)
    weighted = np.nansum(pi*lam, axis=0, keepdims=True)
    untempered = replace(model, alpha=np.ones(NUM_CLASSES))
    dist = obs["range_source"].astype(float)
    if not np.isfinite(dist).all() or (dist <= 0).any():
        raise ValueError("Distance baselines require positive Euclidean range_source.")
    span = np.ptp(dist)
    # Exact tie convention declared: equal distances -> equal weights; strict range -> no support.
    dw = np.ones(len(dist)) if span == 0 else 1 - (dist - dist.min())/span
    maxp = np.max(np.where(np.isfinite(p), p, -np.inf), axis=0)
    maxp[~np.isfinite(maxp)] = np.nan
    result = {
        "mean": weighted_mean(p, np.ones((len(p), 1))),
        "max": maxp,
        "angle_range": weighted_mean(p, (obs["cos_theta"] >= np.cos(np.deg2rad(50)))[:, None]),
        "distance_range": weighted_mean(p, (dist < np.median(dist))[:, None]),
        "distance_weighted": weighted_mean(p, dw[:, None]),
        "naive_bayes": untempered.posterior(raw, count)[0],
        "tempered_bayes": model.posterior(raw, count)[0],
        "censored_untempered": untempered.posterior(mix, count)[0],
        "censored": model.posterior(mix, count)[0],
        "pi_weighted_mean": weighted_mean(p, pi),
        "pi_weighted_llr": model.posterior(weighted, count)[0]}
    return result, raw[0], mix[0], count[0], float(np.min(g)), float(np.median(obs["cos_theta"])), weighted[0]


def compute(run, d, model, cameras, uncensored_alpha=None, oracle=False, weighted_alpha=None):
    n = len(d["source_index"])
    out = {k: np.full((n, NUM_CLASSES), np.nan, float) for k in METHODS}
    if oracle:
        out["oracle_best_view"] = np.full((n, NUM_CLASSES), np.nan)
    out.update(llr=np.zeros((n, NUM_CLASSES)), censored_llr=np.zeros((n, NUM_CLASSES)),
               pi_llr=np.zeros((n, NUM_CLASSES)),
               count=np.zeros((n, NUM_CLASSES), dtype=np.uint32),
               g_min=np.full(n, np.nan), cos_median=np.full(n, np.nan))
    seen = set()
    for obs in iter_observations(run, d["source_index"]):
        obs = obs[np.isin(obs["camera"], cameras)]
        if not len(obs):
            continue
        ids, start, count = np.unique(obs["source_index"], return_index=True, return_counts=True)
        rows = match_rows(ids, d)
        if seen.intersection(ids.tolist()):
            raise ValueError("A source point appears in multiple point chunks.")
        seen.update(ids.tolist())
        for row, a, length in zip(rows, start, count):
            scores, llr, mix, cnt, g, cos, weighted = point_scores(obs[a:a+length], model)
            for key, value in scores.items():
                out[key][row] = value
            if oracle:
                pp = probs_of(obs[a:a+length])
                for c in range(NUM_CLASSES):
                    valid = pp[:, c][np.isfinite(pp[:, c])]
                    if len(valid) and d["labels"][row, c] >= 0:
                        out["oracle_best_view"][row, c] = (valid.max() if d["labels"][row, c] else valid.min())
            if uncensored_alpha is not None:
                baseline = replace(model, alpha=np.asarray(uncensored_alpha))
                out["tempered_bayes"][row] = baseline.posterior(llr[None], cnt[None])[0]
            if weighted_alpha is not None:
                baseline = replace(model, alpha=np.asarray(weighted_alpha))
                out["pi_weighted_llr"][row] = baseline.posterior(weighted[None], cnt[None])[0]
            out["pi_llr"][row] = weighted
            out["llr"][row], out["censored_llr"][row], out["count"][row] = llr, mix, cnt
            out["g_min"][row], out["cos_median"][row] = g, cos
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True)
    ap.add_argument("--regions", required=True)
    ap.add_argument("--buffer", type=float, required=True)
    ap.add_argument("--params", required=True)
    ap.add_argument("--fraction", type=float, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--oracle", action="store_true", help="diagnostic only: reads labels to select each point/class view")
    args = ap.parse_args(argv)
    if args.out.exists() or args.out.with_suffix(".json").exists():
        raise FileExistsError("Choose a new output (never replace a frozen experiment).")
    d = regions(args.regions, args.buffer)
    prov = provenance(args.run, args.regions)
    params = read_json(args.params)
    validate_detector(args.run, params)
    validate_fit_provenance(params, prov)
    model = FusionModel.from_dict("censored_llr", params)
    meta = read_json(Path(args.run) / "observations/meta.json")
    cameras = camera_subset([c["camera"] for c in meta["cameras"]], args.fraction, args.seed)
    uncensored_alpha = ([params["tempering_diagnostics"][c]["uncensored_alpha"] for c in DAMAGE_CLASSES]
                        if "tempering_diagnostics" in params else None)
    weighted_alpha = ([params["tempering_diagnostics"][c]["weighted_alpha"] for c in DAMAGE_CLASSES]
                      if "tempering_diagnostics" in params else None)
    out = compute(args.run, d, model, cameras, uncensored_alpha, args.oracle, weighted_alpha)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("wb") as f:
        np.savez_compressed(f, source_index=d["source_index"], classes=np.array(DAMAGE_CLASSES), **out)
    write_json(args.out.with_suffix(".json"), {
        "provenance": prov, "params_sha256": sha256_file(args.params),
        "score_sha256": sha256_file(args.out), "parameters": params,
        "camera_subset": cameras, "fraction": args.fraction, "seed": args.seed,
        "buffer": args.buffer, "methods": list(METHODS) + (["oracle_best_view"] if args.oracle else []),
        "oracle": args.oracle,
        "note": "Unsigned-normal geometric baselines. Only optional oracle_best_view reads labels; it is not deployable."})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
