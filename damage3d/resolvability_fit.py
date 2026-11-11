"""Fit class-specific latent resolvability from C_fit leave-view/group-out pseudo-positives.

python -m damage3d.resolvability_fit --help
See DIRECTION_A_METHOD.md: recall is b + (a-b)*pi, NOT pi itself.
"""
from __future__ import annotations

import argparse
import numpy as np
from scipy.optimize import minimize

from .classes import DAMAGE_CLASSES
from .detectability import FusionModel, sigmoid
from .experiment_data import read_json, write_json, regions, provenance, match_rows, validate_detector
from .observations import iter_observations, probs_of


def held_out_samples(obs, model, cfg, class_index):
    """Return candidate indices; the candidate's outcome never enters its reference."""
    g = model.view_gsd(obs["gsd_frontal"], obs["cos_theta"], obs["map_scale"])
    p = probs_of(obs)[:, class_index]
    k = cfg["top_k"]
    if not isinstance(k, int) or k < 1 or cfg["min_resolution_ratio"] <= 1:
        raise ValueError("top_k must be integer >=1, min_resolution_ratio >1.")
    if not 0 < cfg["reference_threshold"] < 1:
        raise ValueError("reference_threshold must be in (0,1).")
    groups = cfg.get("camera_groups", {})
    group = np.array([str(groups.get(str(int(c)), f"camera_{c}")) for c in obs["camera"]])
    result = []
    _, starts, sizes = np.unique(obs["source_index"], return_index=True, return_counts=True)
    for start, size in zip(starts, sizes):
        idx = np.arange(start, start + size)
        ordered = idx[np.lexsort((obs["camera"][idx], g[idx]))]
        for j in idx:
            if not np.isfinite(p[j]):
                continue
            eligible = ordered[(group[ordered] != group[j]) & np.isfinite(p[ordered]) &
                               (g[ordered] <= g[j] / cfg["min_resolution_ratio"])]
            # One reference per group, not k nearly identical images in the same group.
            _, first = np.unique(group[eligible], return_index=True)
            ref = eligible[np.sort(first)][:k]
            if len(ref) == k and np.all(p[ref] >= cfg["reference_threshold"]):
                result.append(j)
    return np.asarray(result, dtype=int), g


def balanced_weights(point, region):
    """Each region has total mass 1; each pseudo-positive point shares it equally."""
    w = np.zeros(len(point), float)
    for r in np.unique(region):
        mask = region == r
        _, inverse, counts = np.unique(point[mask], return_inverse=True, return_counts=True)
        w[mask] = 1.0 / counts[inverse] / len(counts)
    return w / w.sum()


def fit_curve(g, hit, weight, a, b, min_samples=50):
    """Weighted Bernoulli MLE with FIXED resolved TPR a and background FPR b."""
    g, y, w = np.asarray(g, float), np.asarray(hit, float), np.asarray(weight, float)
    if g.ndim != 1 or y.shape != g.shape or w.shape != g.shape:
        raise ValueError("g, hits, weights must be equal length vectors.")
    if (not np.isfinite(g).all() or (g <= 0).any() or not np.isin(y, [0, 1]).all()
            or not np.isfinite(w).all() or (w <= 0).any()):
        raise ValueError("Invalid fitting samples.")
    if not 0 <= b < a <= 1:
        raise ValueError("Fixed hit-rate anchors require 0 <= b < a <= 1.")
    if len(g) < min_samples or len(np.unique(y)) < 2 or np.ptp(np.log(g)) < np.log(2):
        raise ValueError("Insufficient support: need hits, misses and >=2x GSD span.")
    x = np.log(g)
    bounds = [(x.min(), x.max()), (np.log(.05), np.log(5.0))]
    w = w / w.sum()

    def objective(t):
        q = np.clip(b + (a-b) * sigmoid((t[0] - x) / np.exp(t[1])), 1e-12, 1-1e-12)
        return -np.sum(w * (y*np.log(q) + (1-y)*np.log1p(-q)))

    fits = [minimize(objective, [np.quantile(x, q), np.log(s)], method="L-BFGS-B",
                     bounds=bounds) for q in (.25, .5, .75) for s in (.2, 1.)]
    fit = min(fits, key=lambda f: f.fun)
    if not fit.success:
        raise ValueError(f"Resolvability optimization failed: {fit.message}")
    if any(abs(t-lo) < 1e-4 or abs(t-hi) < 1e-4 for t, (lo, hi) in zip(fit.x, bounds)):
        raise ValueError("Curve optimum touches bounds: transition not identified. Do not extrapolate a fit.")
    # Compare with the best constant probability. Improvement is a diagnostic, not a significance test.
    q0 = np.clip(np.sum(w*y), 1e-12, 1-1e-12)
    null = -np.sum(w * (y*np.log(q0)+(1-y)*np.log1p(-q0)))
    if null - fit.fun <= 1e-6:
        raise ValueError("No identifiable decreasing trend against constant-hit-rate model.")
    bins = []
    edges = np.linspace(x.min(), x.max(), 9)
    for lo, hi in zip(edges[:-1], edges[1:]):
        use = (x >= lo) & (x <= hi if hi == edges[-1] else x < hi)
        bins.append({"g_low": float(np.exp(lo)), "g_high": float(np.exp(hi)),
                     "n": int(use.sum()), "hit_rate": float(y[use].mean()) if use.any() else None})
    return {"g50": float(np.exp(fit.x[0])), "beta": float(np.exp(fit.x[1]))}, {
        "n": len(g), "g_min": float(g.min()), "g_max": float(g.max()), "nll": float(fit.fun),
        "constant_nll": float(null), "a": a, "b": b, "bins": bins,
        "beta_bounds": [.05, 5.0], "g50_bounds": [float(g.min()), float(g.max())]}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True)
    ap.add_argument("--regions", required=True)
    ap.add_argument("--buffer", type=float, required=True)
    ap.add_argument("--params", required=True, help="base fusion JSON from fusion_params")
    ap.add_argument("--config", required=True, help="frozen per-class anchors/reference choices")
    ap.add_argument("--out", required=True)
    ap.add_argument("--bootstrap", type=int, default=0,
                    help="optional region bootstrap of g50/beta, conditional on fixed anchors (e.g. 200)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    d = regions(args.regions, args.buffer)
    prov = provenance(args.run, args.regions, args.config)
    cfg, params = read_json(args.config), read_json(args.params)
    validate_detector(args.run, params)
    if cfg.get("calibration_sha256") and cfg["calibration_sha256"] != params.get("calibration_sha256"):
        raise ValueError("Anchor calibration does not match the inference model.")
    geom = FusionModel.from_dict("llr", params)
    selected = d["source_index"][d["split"] == "C_fit"]
    if not len(selected):
        raise ValueError("No C_fit points.")
    samples = {c: [] for c in DAMAGE_CLASSES}
    for obs in iter_observations(args.run, selected):
        if not len(obs):
            continue
        for c, name in enumerate(DAMAGE_CLASSES):
            conf = {**cfg["common"], **cfg["classes"][name]}
            if not 0 < conf["threshold"] < 1 or not conf.get("anchor_source"):
                raise ValueError("Declare threshold and anchor_source for every class.")
            ids, g = held_out_samples(obs, geom, conf, c)
            rows = match_rows(obs["source_index"][ids], d)
            samples[name].append((g[ids], probs_of(obs)[ids, c] >= conf["threshold"],
                                  obs["source_index"][ids], d["region"][rows]))
    diagnostics = {}
    for name in DAMAGE_CLASSES:
        if not samples[name]:
            raise ValueError(f"{name}: no eligible pseudo-positive observations.")
        g, y, point, region = [np.concatenate([s[j] for s in samples[name]]) for j in range(4)]
        conf = {**cfg["common"], **cfg["classes"][name]}
        if len(np.unique(region)) < conf.get("min_regions", 3):
            raise ValueError(f"{name}: too few C_fit regions.")
        try:
            fit, diag = fit_curve(g, y, balanced_weights(point, region), conf["resolved_tpr"],
                                  conf["background_fpr"], conf.get("min_samples", 50))
        except ValueError as e:
            raise ValueError(f"{name}: {e}") from e
        params["classes"][name].update(fit)
        params["classes"][name]["alpha"] = 1.0
        diagnostics[name] = {**diag, "n_points": len(np.unique(point)), "n_regions": len(np.unique(region)),
                             "selection": conf}
        if args.bootstrap < 0:
            raise ValueError("bootstrap must be nonnegative.")
        if args.bootstrap:
            groups, inverse = np.unique(region, return_inverse=True)
            rng = np.random.default_rng(args.seed)
            values = []
            weights = balanced_weights(point, region)
            for _ in range(args.bootstrap):
                multiplicity = np.bincount(rng.integers(0, len(groups), len(groups)), minlength=len(groups))
                w = weights * multiplicity[inverse]
                use = w > 0
                try:
                    f, _ = fit_curve(g[use], y[use], w[use], conf["resolved_tpr"],
                                     conf["background_fpr"], conf.get("min_samples", 50))
                    values.append([f["g50"], f["beta"]])
                except ValueError:
                    pass  # Report failures; never silently replace with original optimum.
            diagnostics[name]["region_bootstrap"] = {
                "requested": args.bootstrap, "valid": len(values), "seed": args.seed,
                "ci95_g50_beta": np.quantile(values, [.025, .975], axis=0).tolist() if values else None,
                "conditional_on": "fixed anchors, selected pseudo-positives and camera groups"}
    params.update(fit_provenance=prov, fit_split="C_fit", fit_buffer=args.buffer,
                  fit_diagnostics=diagnostics, fit_kind="anchored_leave_group_out_pseudo_positive")
    FusionModel.from_dict("censored_llr", params)
    write_json(args.out, params)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
