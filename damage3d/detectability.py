"""Resolution-aware (censored) log-likelihood-ratio fusion (report, direction A).

Notation of the report. Per point i, view v and class c:

    cos theta_iv = |n_i^T (C_v - P_i)| / ||C_v - P_i||
    g_iv         = (Z_iv / f_v) * s * sigma_v / sqrt(cos theta_iv)
                   (s = chunk -> source scale, sigma_v = W / W_m map reduction factor;
                    1/sqrt(cos) = geometric mean of the anisotropic footprint (Z/f, Z/(f cos)))
    lambda_ivc   = logit p_ivc - logit rho_c
                   (rho_c = prevalence implied by the Platt calibration, see damage3d.fusion_params)
    pi_c(g)      = sigmoid(-(log g - log g50_c) / beta_c)       (beta_c > 0, pi_c(g50_c) = 0.5)
    lambda~_ivc  = log(1 + pi_c(g_iv) (exp(lambda_ivc) - 1))
                 = logaddexp(log pi + lambda, log(1 - pi))        (stable form)
    L_ic         = logit rho_c + alpha_c * sum_v lambda~_ivc,     s_ic = sigmoid(L_ic)

Mixture model: with probability pi the view resolves the defect and answers like
the calibrated detector (likelihood ratio exp(lambda)); otherwise it answers like
on the background (ratio 1). pi -> 1 gives lambda (plain naive Bayes, fusion
"llr"); pi -> 0 gives 0 (the view is censored).

alpha_c in (0, 1] is a per-class TEMPERED posterior: a single exponent that
compensates on average for the correlation between views (naive Bayes assumes
conditionally independent views). It does not model the correlation of
individual view pairs. The optional quality weight omega_iv of the report is not
implemented (omega = 1).

g is expressed in SOURCE units per MAP pixel: g50_c refers to these units
(estimate it on the same project or convert it once the metric scale is fixed).

JSON (``--fusion-params``, written by ``damage3d.fusion_params``)::

    {"gsd": "effective",             # or "frontal" (cos theta ignored)
     "gsd_units": "source_units_per_map_px",
     "p_clip": 1e-4, "min_cos": 0.0872,
     "classes": {"crack": {"rho": 0.0121, "g50": 0.004, "beta": 0.3, "alpha": 1.0}, ...}}

Every mapped class needs ``rho``. ``g50``/``beta`` are needed only by
``censored_llr`` (missing parameters are an error); ``alpha``
defaults to 1. Optional ``base_rate`` replaces rho_c in the offset of L_ic only
(different prevalence on the bridge); the evidence always uses rho_c.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .classes import CLOSED_CLASSES, DAMAGE_CLASSES, NUM_CLASSES

FUSION_MODES = ("mean", "llr", "censored_llr")


def _logit(p):
    return np.log(p) - np.log1p(-p)


def sigmoid(z):
    """Overflow-safe logistic, including extreme log odds."""
    return np.exp(-np.logaddexp(0.0, -np.asarray(z, np.float64)))


@dataclass(frozen=True)
class FusionModel:
    mode: str                 # llr | censored_llr
    rho: np.ndarray           # (C,) prevalence of the calibration set (evidence reference)
    g50: np.ndarray           # (C,) GSD with pi = 0.5; NaN = no censoring for the class
    beta: np.ndarray          # (C,) logistic scale in log g
    alpha: np.ndarray         # (C,) tempering exponent in (0, 1]
    base_rate: np.ndarray     # (C,) offset of the posterior (default rho)
    p_clip: float = 1e-4
    min_cos: float = 0.0872   # cos 85 deg
    gsd: str = "effective"
    gsd_units: str = "source_units_per_map_px"

    # ---- construction ------------------------------------------------------
    @staticmethod
    def from_dict(mode: str, d: dict) -> "FusionModel":
        if mode not in ("llr", "censored_llr"):
            raise ValueError(f"FusionModel is for llr/censored_llr, got '{mode}'.")
        classes = d.get("classes", {})
        unknown = set(classes) - set(DAMAGE_CLASSES) - set(CLOSED_CLASSES)
        if unknown:
            raise ValueError(f"Unknown classes in fusion params: {sorted(unknown)}")
        rho, g50, beta, alpha, base = (np.full(NUM_CLASSES, np.nan) for _ in range(5))
        for c, name in enumerate(DAMAGE_CLASSES):
            entry = classes.get(name, {})
            if set(entry) - {"rho", "base_rate", "alpha", "g50", "beta"}:
                raise ValueError(f"Unknown/legacy parameters for {name}: {sorted(entry)}")
            if "rho" not in entry:
                raise ValueError(f"Fusion params: class '{name}' needs 'rho' (calibration prevalence).")
            rho[c] = float(entry["rho"])
            base[c] = float(entry.get("base_rate", entry["rho"]))
            alpha[c] = float(entry.get("alpha", 1.0))
            if mode == "censored_llr":
                if not {"g50", "beta"} <= entry.keys():
                    raise ValueError(f"{name}: censored_llr requires g50 and beta; no silent fallback.")
                g50[c], beta[c] = float(entry["g50"]), float(entry["beta"])
        if not np.isfinite(np.concatenate([rho, base, alpha])).all():
            raise ValueError("rho, base_rate and alpha must be finite.")
        if mode == "censored_llr" and not np.isfinite(np.concatenate([g50, beta])).all():
            raise ValueError("g50 and beta must be finite.")
        if np.any((rho <= 0) | (rho >= 1)) or np.any((base <= 0) | (base >= 1)):
            raise ValueError("rho and base_rate must be in (0, 1).")
        if np.any(g50[np.isfinite(g50)] <= 0) or np.any(beta[np.isfinite(beta)] <= 0):
            raise ValueError("g50 and beta must be > 0 (detectability decreases with the GSD).")
        if np.any((alpha <= 0) | (alpha > 1)):
            raise ValueError("alpha (tempering) must be in (0, 1].")
        gsd = d.get("gsd", "effective")
        if gsd not in ("effective", "frontal"):
            raise ValueError("gsd must be 'effective' or 'frontal'.")
        if d.get("gsd_units", "source_units_per_map_px") != "source_units_per_map_px":
            raise ValueError("Expected source_units_per_map_px; convert parameters explicitly.")
        if "kappa" in d:
            raise ValueError("Legacy kappa is not supported; use class alpha.")
        clip, floor = float(d.get("p_clip", 1e-4)), float(d.get("min_cos", 0.0872))
        if not 0 < clip < 0.5 or not 0 < floor <= 1:
            raise ValueError("p_clip must be in (0,.5), min_cos in (0,1].")
        return FusionModel(mode, rho, g50, beta, alpha, base, float(d.get("p_clip", 1e-4)),
                           float(d.get("min_cos", 0.0872)), gsd, d.get("gsd_units", "source_units_per_map_px"))

    @staticmethod
    def load(mode: str, path: Path) -> "FusionModel":
        return FusionModel.from_dict(mode, json.loads(Path(path).read_text(encoding="utf-8")))

    def describe(self) -> dict:
        classes = {}
        for c, n in enumerate(DAMAGE_CLASSES):
            entry = {"rho": float(self.rho[c]), "base_rate": float(self.base_rate[c]), "alpha": float(self.alpha[c])}
            if self.mode == "censored_llr" and np.isfinite(self.g50[c]):
                entry.update(g50=float(self.g50[c]), beta=float(self.beta[c]))
            classes[n] = entry
        return {"implementation": "direction_a_v3", "mode": self.mode, "gsd": self.gsd, "gsd_units": self.gsd_units, "p_clip": self.p_clip,
                "min_cos": self.min_cos, "classes": classes}

    def uncensored_classes(self) -> list[str]:
        if self.mode != "censored_llr":
            return []
        return [n for c, n in enumerate(DAMAGE_CLASSES) if not np.isfinite(self.g50[c])]

    @property
    def needs_normals(self) -> bool:
        return self.mode == "censored_llr" and self.gsd == "effective"

    # ---- per-view quantities -----------------------------------------------
    def view_gsd(self, gsd_frontal: np.ndarray, cos_theta: np.ndarray | None,
                 map_scale: np.ndarray | float = 1.0) -> np.ndarray:
        """g_iv = gsd_frontal * sigma_v / sqrt(cos theta); gsd_frontal = Z * s / f (per image px)."""
        g = np.asarray(gsd_frontal, np.float64) * np.asarray(map_scale, np.float64)
        if np.any(~np.isfinite(g) | (g <= 0)):
            raise ValueError("GSD and map scale must be finite and positive.")
        if self.gsd == "frontal":
            return g
        if cos_theta is None:
            raise ValueError("Effective GSD requires valid normals.")
        cos = np.asarray(cos_theta, np.float64)
        if np.any(~np.isfinite(cos) | (np.abs(cos) > 1.00001)):
            raise ValueError("Invalid normal/incidence: repair geometry before censored fusion.")
        return g / np.sqrt(np.maximum(np.abs(cos), self.min_cos))

    def pod(self, g: np.ndarray) -> np.ndarray:
        """(N,) GSD -> (N, C) probability that the view resolves each class."""
        if g is None or np.any(~np.isfinite(g)) or np.any(np.asarray(g) <= 0):
            raise ValueError("POD requires finite positive GSD.")
        logg = np.log(np.asarray(g, np.float64))[:, None]
        with np.errstate(invalid="ignore"):
            z = -(logg - np.log(self.g50)[None, :]) / self.beta[None, :]
        pi = sigmoid(z)
        return np.where(np.isfinite(self.g50)[None, :], pi, 1.0)

    def view_llr(self, probs: np.ndarray, g: np.ndarray | None) -> np.ndarray:
        """(N, C) calibrated probabilities -> (N, C) float64 evidence lambda~ (NaN stays NaN)."""
        p = np.clip(np.asarray(probs, np.float64), self.p_clip, 1.0 - self.p_clip)
        lam = _logit(p) - _logit(self.rho)[None, :]
        if self.mode == "llr":
            return lam
        pi = self.pod(g)
        with np.errstate(divide="ignore", invalid="ignore"):
            out = np.logaddexp(np.log(pi) + lam, np.log1p(-pi))
        return np.where(pi >= 1.0, lam, out)

    def posterior(self, llr_sum: np.ndarray, count: np.ndarray) -> np.ndarray:
        """(n, C) summed evidence -> s_ic = sigmoid(logit b_c + alpha_c * sum), NaN where no valid view."""
        z = _logit(self.base_rate)[None, :] + self.alpha[None, :] * np.asarray(llr_sum, np.float64)
        post = sigmoid(z)
        return np.where(count > 0, post, np.nan).astype(np.float32)


def load_fusion(mode: str, params: Path | None) -> FusionModel | None:
    if mode not in FUSION_MODES:
        raise ValueError(f"Unknown fusion '{mode}' (choose {FUSION_MODES}).")
    if mode == "mean":
        if params is not None:
            raise ValueError("--fusion-params is used only by --fusion llr/censored_llr.")
        return None
    if params is None:
        raise ValueError(f"--fusion {mode} requires --fusion-params <json> (class rho"
                         f"{', g50, beta' if mode == 'censored_llr' else ''}).")
    return FusionModel.load(mode, params)
