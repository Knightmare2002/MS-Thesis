"""Per-class Platt calibration of multilabel sigmoid outputs (single source of truth).

Model
-----
For class c a raw probability p is mapped to

    p' = sigmoid(a_c * logit(p) + b_c),        a_c > 0.

* ``a_c`` (= 1 / temperature) corrects over/under-confidence;
* ``b_c`` corrects the prior shift introduced by ``pos_weight`` in the BCE
  loss and by class-balanced patch sampling (temperature scaling alone,
  b_c = 0, cannot remove it).

Since a_c > 0 the map is strictly increasing: for ONE model, rankings, AP,
masks and Dice are unchanged as long as each threshold is mapped with the
same function (``map_thresholds``). Calibration matters only where
probabilities are combined: averaging over models (ensemble) and over views
(damage3d fusion).

Fitting
-------
``fit_platt_from_histograms`` minimises the binary NLL on the positive /
negative probability histograms written by scripts/25 (uniform bins on
[0, 1]); each bin is represented by the logit of its centre. Newton-Raphson on
two parameters, with backtracking.

JSON formats (read by scripts/23, scripts/28 and damage3d)
----------------------------------------------------------
* single model: ``format = "p1ml_platt_calibration_v1"`` with ``params``
  {class: {a, b}} and ``thresholds_calibrated`` {class: tau'};
* ensemble: ``format = "p1ml_platt_ensemble_v1"`` with ``members`` (each with
  its own ``params`` and normalised ``weight``) and ``thresholds`` {class: tau}.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

FORMAT_SINGLE = "p1ml_platt_calibration_v1"
FORMAT_ENSEMBLE = "p1ml_platt_ensemble_v1"
EPS = 1e-6


# --------------------------------------------------------------------------- #
# Elementary maps
# --------------------------------------------------------------------------- #
def logit(p: np.ndarray, eps: float = EPS) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=np.float64), eps, 1.0 - eps)
    return np.log(p) - np.log1p(-p)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.tanh(0.5 * np.asarray(x, dtype=np.float64)))  # overflow-free


def apply_platt(probs: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """probs (C, ...) in [0, 1] -> calibrated float32 (C, ...); NaN stays NaN."""
    probs = np.asarray(probs)
    shape = (-1,) + (1,) * (probs.ndim - 1)
    a = np.asarray(a, dtype=np.float64).reshape(shape)
    b = np.asarray(b, dtype=np.float64).reshape(shape)
    out = sigmoid(a * logit(probs) + b).astype(np.float32)
    out[~np.isfinite(probs)] = np.nan
    return out


def map_thresholds(thresholds: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Raw-scale thresholds -> calibrated scale (same decision for every pixel)."""
    return sigmoid(np.asarray(a) * logit(thresholds) + np.asarray(b))


# --------------------------------------------------------------------------- #
# Fitting and diagnostics on histograms
# --------------------------------------------------------------------------- #
def bin_centers(bins: int) -> np.ndarray:
    return (np.arange(bins, dtype=np.float64) + 0.5) / bins


def _nll(a: float, b: float, z: np.ndarray, n1: np.ndarray, n0: np.ndarray) -> float:
    s = np.clip(sigmoid(a * z + b), 1e-12, 1.0 - 1e-12)
    return float(-(n1 * np.log(s) + n0 * np.log1p(-s)).sum() / (n1.sum() + n0.sum()))


def fit_platt_from_histograms(hist_pos: np.ndarray, hist_neg: np.ndarray,
                              max_iter: int = 100, tol: float = 1e-10) -> tuple[float, float, dict]:
    """Return (a, b, info) for one class from its positive/negative histograms."""
    n1 = np.asarray(hist_pos, dtype=np.float64)
    n0 = np.asarray(hist_neg, dtype=np.float64)
    if n1.sum() == 0 or n0.sum() == 0:
        raise ValueError("Platt fit needs both positive and negative pixels.")
    z = logit(bin_centers(len(n1)))
    keep = (n1 + n0) > 0
    z, n1, n0 = z[keep], n1[keep], n0[keep]
    total = n1.sum() + n0.sum()
    a, b = 1.0, 0.0
    loss = _nll(a, b, z, n1, n0)
    converged = False
    for it in range(1, max_iter + 1):
        s = sigmoid(a * z + b)
        residual = (n1 + n0) * s - n1                      # Expected minus observed counts in each bin
        grad = np.array([(residual * z).sum(), residual.sum()]) / total # Gradient of the loss w.r.t. a and b
        bin_curvature = (n1 + n0) * s * (1.0 - s)
        hessian = np.array([
                        [(bin_curvature * z * z).sum(), (bin_curvature * z).sum()], 
                        [(bin_curvature * z).sum(), bin_curvature.sum()]
                ]) / total
        step = np.linalg.solve(hessian + 1e-12 * np.eye(2), grad) # Newton-Raphson step (correction for a and b)
        t = 1.0 # Backtracking line search to ensure loss decreases (How much of that step to take)
        while True:
            a_new, b_new = a - t * step[0], b - t * step[1]
            new_loss = _nll(a_new, b_new, z, n1, n0)
            if new_loss <= loss or t < 1e-8:
                break
            t *= 0.5
        a, b, delta, loss = a_new, b_new, loss - new_loss, new_loss
        if abs(delta) < tol:
            converged = True
            break
    if a <= 0:
        raise ValueError(f"Fitted slope a={a:.4f} <= 0: the channel ranks positives below negatives.")
    return float(a), float(b), {"iterations": it, "converged": converged, "nll": loss}


def nll_from_histograms(hist_pos, hist_neg, a: float = 1.0, b: float = 0.0) -> float:
    z = logit(bin_centers(len(hist_pos)))
    return _nll(a, b, z, np.asarray(hist_pos, float), np.asarray(hist_neg, float))


def ece_from_histograms(hist_pos, hist_neg, a: float = 1.0, b: float = 0.0, n_bins: int = 15) -> float:
    """Expected calibration error (equal-width bins on the calibrated probability)."""
    n1 = np.asarray(hist_pos, float)
    n0 = np.asarray(hist_neg, float)
    s = sigmoid(a * logit(bin_centers(len(n1))) + b)
    group = np.minimum((s * n_bins).astype(int), n_bins - 1)
    n = n1 + n0
    ece = 0.0
    for k in range(n_bins):
        m = group == k
        if n[m].sum() > 0:
            ece += abs(n1[m].sum() / n[m].sum() - (s[m] * n[m]).sum() / n[m].sum()) * n[m].sum()
    return float(ece / n.sum())


def reliability_curve(hist_pos, hist_neg, a: float = 1.0, b: float = 0.0, n_bins: int = 15):
    """(mean predicted, observed frequency, pixel count) per equal-width bin."""
    n1 = np.asarray(hist_pos, float)
    n0 = np.asarray(hist_neg, float)
    s = sigmoid(a * logit(bin_centers(len(n1))) + b)
    group = np.minimum((s * n_bins).astype(int), n_bins - 1)
    n = n1 + n0
    rows = []
    for k in range(n_bins):
        m = group == k
        if n[m].sum() > 0:
            rows.append(((s[m] * n[m]).sum() / n[m].sum(), n1[m].sum() / n[m].sum(), n[m].sum()))
    return np.array(rows, dtype=np.float64).reshape(-1, 3)


# --------------------------------------------------------------------------- #
# JSON I/O
# --------------------------------------------------------------------------- #
def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def params_arrays(params: dict, classes) -> tuple[np.ndarray, np.ndarray]:
    missing = [c for c in classes if c not in params]
    if missing:
        raise KeyError(f"Platt parameters missing for classes {missing}.")
    a = np.array([float(params[c]["a"]) for c in classes], dtype=np.float64)
    b = np.array([float(params[c]["b"]) for c in classes], dtype=np.float64)
    if np.any(a <= 0):
        raise ValueError("Platt slopes must be > 0.")
    return a, b


def load_platt(path: Path, classes) -> tuple[np.ndarray, np.ndarray, dict]:
    """Single-model calibration file -> (a, b, payload)."""
    payload = read_json(path)
    if payload.get("format") != FORMAT_SINGLE:
        raise ValueError(f"{path}: format '{payload.get('format')}' != '{FORMAT_SINGLE}'.")
    missing = [c for c in classes if c not in payload.get("classes", [])]
    if missing:  # selection by name: a 6-class file serves the 5 reported 3D classes
        raise ValueError(f"{path}: classes {missing} not in {payload.get('classes')}.")
    a, b = params_arrays(payload["params"], classes)
    return a, b, payload


def thresholds_from_json(path: Path, classes) -> np.ndarray:
    """Frozen thresholds on the CALIBRATED scale, from a single-model or an ensemble file."""
    payload = read_json(path)
    fmt = payload.get("format")
    if fmt == FORMAT_SINGLE:
        thr = payload["thresholds_calibrated"]
    elif fmt == FORMAT_ENSEMBLE:
        thr = payload["thresholds"]
    else:
        raise ValueError(f"{path}: unknown calibration format '{fmt}'.")
    values = np.array([float(thr[c]) for c in classes], dtype=np.float32)
    if np.any((values <= 0) | (values > 1)):
        raise ValueError(f"{path}: thresholds must be in (0, 1].")
    return values
