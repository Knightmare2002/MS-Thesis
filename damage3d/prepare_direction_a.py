"""Prepare a pre-registration template from measured DACL10K degradation counts.

Creates no fitted bridge parameters. Defaults for reference selection are declared design
choices, not empirically optimized values. Review/freeze the output before using C_fit.
"""
from __future__ import annotations
import argparse
from .classes import DAMAGE_CLASSES
from .experiment_data import read_json, write_json
from .run_state import sha256_file


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--degradation", required=True)
    ap.add_argument("--reference-threshold", type=float, default=.9)
    ap.add_argument("--top-k", type=int, default=2)
    ap.add_argument("--min-resolution-ratio", type=float, default=1.25)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if not 0 < a.reference_threshold < 1 or a.top_k < 1 or a.min_resolution_ratio <= 1:
        raise ValueError("Invalid reference selection settings.")
    data = read_json(a.degradation)
    if data["classes"] != list(DAMAGE_CLASSES):
        raise ValueError("Degradation class order mismatch.")
    classes = {}
    for j, c in enumerate(DAMAGE_CLASSES):
        anchor = data["aggregate"]["1"][c]
        tpr, fpr = anchor["recall"], anchor["fpr"]
        if tpr is None or fpr is None or not 0 <= fpr < tpr <= 1:
            raise ValueError(f"{c}: valid native-resolution positive/negative anchors required.")
        classes[c] = {"threshold": data["thresholds"][j],
                      "reference_threshold": max(a.reference_threshold, data["thresholds"][j]),
                      "resolved_tpr": tpr, "background_fpr": fpr,
                      "anchor_source": f"{a.degradation} / aggregate / 1 / {c}"}
    write_json(a.out, {"common": {"top_k": a.top_k, "min_resolution_ratio": a.min_resolution_ratio,
                                 "min_regions": 3, "min_samples": 50},
                      "classes": classes, "degradation_sha256": sha256_file(a.degradation),
                      "calibration_sha256": data["calibration_sha256"],
                      "warning": "DACL10K native-scale anchors are transferable reference rates, "
                                 "not proof of fully resolved bridge score distributions."})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
