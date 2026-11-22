"""Freeze GSD/incidence bin edges for evaluate_direction_a before viewing T metrics.

g_edges: quintiles of g_min over C_fit+C_cal points (geometry only, no labels or
metrics), with ends 0 and the global finite maximum so no T point falls outside.
incidence_edges: fixed in degrees.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from .experiment_data import regions
from .run_state import sha256_file

QUANTILES = (.2, .4, .6, .8)
INCIDENCE_EDGES = [0, 15, 30, 45, 60, 90]


def log(msg: str, t0: float) -> None:
    print(f"[{time.time() - t0:6.1f}s] {msg}", flush=True)


def main(argv=None):
    t0 = time.time()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scores", type=Path, required=True, help="baselines.npz (with g_min)")
    ap.add_argument("--regions", type=Path, required=True)
    ap.add_argument("--buffer", type=float, required=True)
    ap.add_argument("--mm-per-unit", type=float, default=None, help="only for the printed mm/px edges")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    if args.out.exists():
        raise FileExistsError(f"{args.out} exists; bins must be frozen once.")

    log(f"Loading regions from {args.regions} and validating buffer={args.buffer}...", t0)
    d = regions(args.regions, args.buffer)
    n_total = len(d["source_index"])
    log(f"Loaded {n_total:,} points from regions.", t0)

    log(f"Loading g_min from {args.scores} and checking index alignment...", t0)
    with np.load(args.scores, allow_pickle=False) as f:
        if not np.array_equal(f["source_index"], d["source_index"]):
            raise ValueError("Source-index alignment mismatch between scores and regions.")
        g = f["g_min"]
    log("Source-index alignment verified.", t0)

    log("Filtering C_fit + C_cal points with finite g_min...", t0)
    finite_g = np.isfinite(g)
    C = np.isin(d["split"], ["C_fit", "C_cal"]) & finite_g
    if not C.any():
        raise ValueError("No C_fit/C_cal points with finite g_min.")
    log(f"Found {int(C.sum()):,} valid C_fit+C_cal points (out of {int(finite_g.sum()):,} total finite g_min points).", t0)

    log(f"Computing quantiles {list(QUANTILES)} and global max...", t0)
    q = np.quantile(g[C], QUANTILES)
    g_max = float(g[finite_g].max())
    edges = sorted({0.0, *map(float, q), g_max * 1.0001})

    log("Computing SHA-256 provenance hashes for scores and regions...", t0)
    scores_hash = sha256_file(args.scores)
    regions_hash = sha256_file(args.regions)

    out = {"g_edges": edges, "incidence_edges": INCIDENCE_EDGES,
           "_provenance": {"g_rule": f"quantiles {list(QUANTILES)} of g_min on C_fit+C_cal "
                                     "(geometry only); ends 0 and global max",
                           "incidence_rule": "fixed 15 deg steps, last bin 60-90",
                           "g_units": "source units per map px",
                           "n_C": int(C.sum()),
                           "scores_sha256": scores_hash,
                           "regions_sha256": regions_hash}}

    log(f"Writing frozen bin edges to {args.out}...", t0)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))

    log("Done! Summary of frozen bins:", t0)
    print("g_edges [units/px]:", np.round(edges, 6), flush=True)
    if args.mm_per_unit:
        print("g_edges [mm/px]:   ", np.round(np.array(edges) * args.mm_per_unit, 3), flush=True)
    print("incidence_edges [deg]:", INCIDENCE_EDGES, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())