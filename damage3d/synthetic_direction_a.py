"""Known-model and misspecified stress experiments; not semi-synthetic image/bridge validation."""
from __future__ import annotations
import argparse
import numpy as np
from .classes import DAMAGE_CLASSES, NUM_CLASSES
from .detectability import FusionModel, sigmoid
from .experiment_metrics import binary_metrics
from .experiment_data import write_json


def experiment(seed=0, n=10000, views=16):
    rng = np.random.default_rng(seed)
    params = {"classes": {c: {"rho": .1, "g50": 1., "beta": .4, "alpha": 1.} for c in DAMAGE_CLASSES}}
    model = FusionModel.from_dict("censored_llr", params)
    y = rng.random(n) < .1
    g = np.exp(rng.uniform(-2, 2, (n, views)))
    pi = sigmoid(-np.log(g)/.4)
    rows = {}
    # Gaussian log-LR with variance 2m, means +/-m has exactly f1/f0=exp(lambda).
    for scenario in ("independent_correct", "duplicated_views", "background_shift", "no_resolution"):
        resolves = rng.random((n, views)) < (0 if scenario == "no_resolution" else pi)
        lam = rng.normal(np.where(y[:, None] & resolves, 2., -2.), 2.)
        if scenario == "duplicated_views":
            lam[:] = lam[:, :1]
            gg = np.repeat(g[:, :1], views, axis=1)
        else:
            gg = g
        if scenario == "background_shift":
            lam[~y] += 2  # breaks the assumed stationary f0
        p = sigmoid(np.log(.1/.9)+lam)
        evidence = np.zeros((n, NUM_CLASSES))
        plain = np.zeros_like(evidence)
        for v in range(views):
            pp = np.repeat(p[:, v, None], NUM_CLASSES, axis=1)
            evidence += model.view_llr(pp, gg[:, v])
            plain += lam[:, v, None]
        count = np.full_like(evidence, views)
        scores = {"mean": p.mean(axis=1), "max": p.max(axis=1),
                  "naive_bayes": model.posterior(plain, count)[:, 0],
                  "censored": model.posterior(evidence, count)[:, 0]}
        rows[scenario] = {k: binary_metrics(y, s, np.ones(n)) for k, s in scores.items()}
    return {"seed": seed, "n": n, "views": views, "parameters": params, "results": rows,
            "note": "Conditional model check and deliberate violations; no evidence of bridge accuracy."}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n", type=int, default=10000)
    ap.add_argument("--views", type=int, default=16)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if a.n < 100 or a.views < 1:
        raise ValueError("n>=100 and views>=1 required.")
    write_json(a.out, experiment(a.seed, a.n, a.views))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
