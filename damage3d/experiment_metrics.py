"""Area-weighted metrics, paired region bootstrap and crack-axis tolerance metric."""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


def binary_metrics(y, p, area, threshold=.5):
    """No unknown labels or missing scores allowed: caller declares common support."""
    y, p, w = np.asarray(y, float), np.asarray(p, float), np.asarray(area, float)
    if (y.ndim != 1 or p.shape != y.shape or w.shape != y.shape or
            not np.isin(y, [0, 1]).all() or not np.isfinite(p).all() or
            np.any((p < 0) | (p > 1)) or not np.isfinite(w).all() or np.any(w <= 0)):
        raise ValueError("Metrics require valid aligned binary labels, scores and positive areas.")
    if not len(y):
        return {k: np.nan for k in ("iou", "f1", "ap", "nll", "brier")}
    pred = p >= threshold
    tp = np.sum(w * y * pred)
    fp = np.sum(w * (1-y) * pred)
    fn = np.sum(w * y * ~pred)
    iou = tp/(tp+fp+fn) if tp+fp+fn else np.nan
    f1 = 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else np.nan
    # Equal-score ties are one PR step: invariant to input order.
    order = np.argsort(-p, kind="stable")
    ys, ws, ps = y[order], w[order], p[order]
    ends = np.r_[np.flatnonzero(ps[:-1] != ps[1:]), len(ps)-1]
    tps, total = np.cumsum(ys*ws)[ends], np.cumsum(ws)[ends]
    pos = np.sum(w*y)
    ap = np.sum(np.diff(np.r_[0, tps])/pos * tps/total) if pos > 0 else np.nan
    pc = np.clip(p, 1e-7, 1-1e-7)
    return {"iou": float(iou), "f1": float(f1), "ap": float(ap),
            "nll": float(np.average(-y*np.log(pc)-(1-y)*np.log1p(-pc), weights=w)),
            "brier": float(np.average((p-y)**2, weights=w))}


def paired_bootstrap(y, candidate, baseline, area, region, threshold_candidate=.5,
                     threshold_baseline=.5, repeats=1000, seed=0):
    """Resample independent regions, same draw for both methods; never individual points."""
    groups, inverse = np.unique(region, return_inverse=True)
    if len(groups) < 2 or repeats < 1:
        raise ValueError("Bootstrap needs >=2 independent regions and repeats >=1.")
    rng = np.random.default_rng(seed)
    deltas = {k: [] for k in ("iou", "f1", "ap", "nll", "brier")}
    for _ in range(repeats):
        counts = np.bincount(rng.integers(0, len(groups), len(groups)), minlength=len(groups))
        w = area * counts[inverse]
        keep = w > 0
        a = binary_metrics(y[keep], candidate[keep], w[keep], threshold_candidate)
        b = binary_metrics(y[keep], baseline[keep], w[keep], threshold_baseline)
        for k in deltas:
            deltas[k].append(a[k] - b[k])
    result = {}
    for k, values in deltas.items():
        finite = np.asarray(values)[np.isfinite(values)]
        result[k] = {"ci95": np.quantile(finite, [.025, .975]).tolist() if len(finite) else None,
                     "valid_replicates": len(finite), "requested_replicates": repeats}
    return result


def axis_iou(truth, prediction, tolerances):
    """Benz-style directed NN counts on UNIFORMLY SAMPLED MEDIAL AXES, same metric units.

    Inputs must be extracted axes, not positive dense surface points. This function does
    not infer crack width or perform skeletonization. Empty/empty is undefined (NaN).
    """
    t, p = np.asarray(truth, float), np.asarray(prediction, float)
    if t.ndim != 2 or p.ndim != 2 or t.shape[1:] != (3,) or p.shape[1:] != (3,):
        raise ValueError("Axis coordinates must be explicit N,3 arrays.")
    tau = np.asarray(tolerances, float)
    if (tau.ndim != 1 or not len(tau) or not np.isfinite(tau).all() or
            not np.isfinite(t).all() or not np.isfinite(p).all() or np.any(tau <= 0)):
        raise ValueError("Finite coordinates and positive tolerances required.")
    dt = cKDTree(p).query(t)[0] if len(p) else np.full(len(t), np.inf)
    dp = cKDTree(t).query(p)[0] if len(t) else np.full(len(p), np.inf)
    rows = []
    for v in tau:
        tp, fn, fp = int(np.sum(dt <= v)), int(np.sum(dt > v)), int(np.sum(dp > v))
        rows.append({"tolerance": float(v), "tp": tp, "fn": fn, "fp": fp,
                     "iou": tp/(tp+fn+fp) if tp+fn+fp else np.nan})
    return rows
