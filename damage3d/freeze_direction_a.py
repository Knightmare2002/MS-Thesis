"""Freeze per-method F1 thresholds on C_cal only, using a declared 0.01..0.99 grid."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
from .classes import DAMAGE_CLASSES
from .experiment_data import read_json, write_json, regions
from .experiment_metrics import binary_metrics
from .fusion_benchmark import METHODS
from .run_state import sha256_file


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scores", type=Path, required=True)
    ap.add_argument("--regions", required=True)
    ap.add_argument("--buffer", type=float, required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    d = regions(a.regions, a.buffer)
    meta = read_json(a.scores.with_suffix(".json"))
    if meta.get("oracle"):
        raise ValueError("Use non-oracle scores for threshold selection.")
    if (meta["provenance"]["regions_sha256"] != sha256_file(a.regions) or
            meta["score_sha256"] != sha256_file(a.scores) or meta["fraction"] != 1):
        raise ValueError("Need aligned, full-view C_cal score provenance.")
    result = {}
    with np.load(a.scores, allow_pickle=False) as s:
        if not np.array_equal(s["source_index"], d["source_index"]):
            raise ValueError("Source index mismatch.")
        for m in METHODS:
            result[m] = {}
            for c, name in enumerate(DAMAGE_CLASSES):
                use = (d["split"] == "C_cal") & (d["labels"][:, c] >= 0) & np.isfinite(s[m][:, c])
                if len(np.unique(d["labels"][use, c])) < 2:
                    raise ValueError(f"{m}/{name}: C_cal must contain both labels.")
                grid = np.arange(1, 100)/100
                f1 = [binary_metrics(d["labels"][use, c], s[m][use, c], d["area"][use], t)["f1"] for t in grid]
                # Predeclared conservative tie break: highest threshold among equal F1.
                result[m][name] = float(grid[np.flatnonzero(np.isclose(f1, np.max(f1), rtol=0, atol=1e-12))[-1]])
    result["_provenance"] = {"split": "C_cal", "regions_sha256": sha256_file(a.regions),
                             "params_sha256": meta["params_sha256"], "scores_sha256": sha256_file(a.scores),
                             "objective": "area-weighted F1; grid .01:.01:.99; ties highest threshold"}
    write_json(a.out, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
