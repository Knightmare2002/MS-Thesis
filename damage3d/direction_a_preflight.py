"""Audit class support without fitting; no T label analysis."""
import argparse
import json
import numpy as np
from .classes import DAMAGE_CLASSES

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--regions", required=True)
    a = ap.parse_args()
    with np.load(a.regions, allow_pickle=False) as f:
        split, region, labels = f["split"].astype(str), f["region"], f["labels"]
        if f["classes"].tolist() != list(DAMAGE_CLASSES):
            raise ValueError("Unexpected class order")
        report = {}
        for s in ("C_fit", "C_cal"):
            mask = split == s
            report[s] = {"n_regions": len(np.unique(region[mask])), "classes": {}}
            for j,c in enumerate(DAMAGE_CLASSES):
                known = mask & (labels[:,j] >= 0)
                pos = known & (labels[:,j] == 1)
                neg = known & (labels[:,j] == 0)
                report[s]["classes"][c] = {
                    "positive_points": int(pos.sum()), "negative_points": int(neg.sum()),
                    "positive_regions": np.unique(region[pos]).tolist(),
                    "labelled_regions": np.unique(region[known]).tolist(),
                    "tempering_label_support": bool(pos.any() and neg.any() and len(np.unique(region[known])) >= 3)
                }
    print(json.dumps(report, indent=2))
    if not all(x["tempering_label_support"] for x in report["C_cal"]["classes"].values()):
        raise SystemExit("C_cal support insufficient for existing all-class tempering; no defaults substituted.")

if __name__ == "__main__":
    main()
