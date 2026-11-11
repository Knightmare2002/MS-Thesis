"""Evaluate already extracted/resampled 3D medial axes. Not a point-cloud skeletonizer."""
import argparse
import numpy as np
from .experiment_metrics import axis_iou
from .experiment_data import write_json
from .run_state import sha256_file


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--truth", required=True, help="N,3 NPY, uniformly sampled GT axes")
    ap.add_argument("--prediction", required=True, help="M,3 NPY, same sampling step")
    ap.add_argument("--metres-per-unit", required=True, type=float)
    ap.add_argument("--sampling-mm", required=True, type=float,
                    help="declared equal sampling step; arrays cannot prove axis topology")
    ap.add_argument("--tolerances-cm", nargs="+", type=float, default=[.5, 1, 2, 4, 8])
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if (not np.isfinite(a.metres_per_unit) or a.metres_per_unit <= 0 or
            not np.isfinite(a.sampling_mm) or a.sampling_mm <= 0):
        raise ValueError("Positive measured scale and sampling interval required.")
    truth = np.load(a.truth, allow_pickle=False) * a.metres_per_unit
    pred = np.load(a.prediction, allow_pickle=False) * a.metres_per_unit
    write_json(a.out, {"results": axis_iou(truth, pred, np.array(a.tolerances_cm)/100),
                      "units": "metres", "sampling_mm_declared": a.sampling_mm,
                      "truth_sha256": sha256_file(a.truth), "prediction_sha256": sha256_file(a.prediction),
                      "note": "Nearest-neighbour counts on supplied medial axes; not width estimation."})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
