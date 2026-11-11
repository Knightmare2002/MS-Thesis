"""Fit alpha_c in [alpha_min,1] on spatially disjoint C_cal labels only."""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
from scipy.optimize import minimize_scalar

from .classes import DAMAGE_CLASSES
from .detectability import FusionModel, _logit
from .experiment_data import read_json, write_json, regions
from .run_state import sha256_file


def fit_alpha(evidence, y, weight, base_rate, alpha_min=1e-4):
    if not 0 < alpha_min < 1:
        raise ValueError("alpha_min must be in (0,1).")
    x, y, w = np.asarray(evidence), np.asarray(y), np.asarray(weight)
    if (x.ndim != 1 or y.shape != x.shape or w.shape != x.shape or
            not np.isfinite(x).all() or not np.isfinite(w).all() or (w <= 0).any() or
            not np.isin(y, [0, 1]).all() or len(np.unique(y)) < 2 or not 0 < base_rate < 1):
        raise ValueError("Tempering requires finite evidence, positive weights and both labels.")
    if np.all(x == 0):
        raise ValueError("Tempering is unidentifiable with zero evidence.")
    def loss(a):
        z = _logit(base_rate) + a*x
        return float(np.average(np.logaddexp(0, z) - y*z, weights=w))
    fit = minimize_scalar(loss, bounds=(alpha_min, 1), method="bounded",
                          options={"xatol": 1e-10})
    if not fit.success:
        raise ValueError("Tempering optimizer failed.")
    best = min((alpha_min, fit.x, 1.0), key=loss)
    return float(best), {"nll_before": loss(1.0), "nll_after": loss(best),
                         "at_lower_bound": best == alpha_min, "n": len(y)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scores", type=Path, required=True)
    ap.add_argument("--regions", required=True)
    ap.add_argument("--buffer", type=float, required=True)
    ap.add_argument("--params", required=True)
    ap.add_argument("--alpha-min", type=float, default=1e-4)
    ap.add_argument("--min-regions", type=int, default=3)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    d = regions(args.regions, args.buffer)
    params, meta = read_json(args.params), read_json(args.scores.with_suffix(".json"))
    if meta.get("oracle"):
        raise ValueError("Use non-oracle scores for fitting.")
    if (meta["provenance"]["regions_sha256"] != sha256_file(args.regions) or
            meta["params_sha256"] != sha256_file(args.params) or
            meta["score_sha256"] != sha256_file(args.scores)):
        raise ValueError("Score/region/parameter provenance mismatch.")
    model = FusionModel.from_dict("censored_llr", params)
    diagnostics = {}
    with np.load(args.scores, allow_pickle=False) as scores:
        if not np.array_equal(d["source_index"], scores["source_index"]):
            raise ValueError("Source-index alignment mismatch.")
        for c, name in enumerate(DAMAGE_CLASSES):
            use = (d["split"] == "C_cal") & (d["labels"][:, c] >= 0) & (scores["count"][:, c] > 0)
            if len(np.unique(d["region"][use])) < args.min_regions:
                raise ValueError(f"{name}: insufficient labelled C_cal regions.")
            alpha, diag = fit_alpha(scores["censored_llr"][use, c], d["labels"][use, c],
                                    d["area"][use], model.base_rate[c], args.alpha_min)
            params["classes"][name]["alpha"] = alpha
            # Separate optimum for the uncensored comparator: do not handicap it.
            plain, pd = fit_alpha(scores["llr"][use, c], d["labels"][use, c],
                                 d["area"][use], model.base_rate[c], args.alpha_min)
            weighted, wd = fit_alpha(scores["pi_llr"][use, c], d["labels"][use, c],
                                     d["area"][use], model.base_rate[c], args.alpha_min)
            diagnostics[name] = {**diag, "uncensored_alpha": plain, "uncensored_fit": pd,
                                 "weighted_alpha": weighted, "weighted_fit": wd}
    params.update(tempering_split="C_cal", tempering_diagnostics=diagnostics,
                  tempering_scores_sha256=sha256_file(args.scores))
    write_json(args.out, params)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
