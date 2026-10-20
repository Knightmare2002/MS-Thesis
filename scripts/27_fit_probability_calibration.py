#!/usr/bin/env python
"""Step 27 - per-class Platt calibration of a P1ML run (no inference).

Fits p' = sigmoid(a_c * logit(p) + b_c) for every damage channel from the
validation probability histograms of scripts/25 (``eval_curves/curves_histograms.npz``)
and maps the frozen per-class thresholds of scripts/17 to the calibrated
scale. Nothing is hard-coded: a_c, b_c are estimated by maximum likelihood on
the histograms of the run passed on the command line.

Why
---
* a single model: the map is strictly increasing, so masks/Dice/AP are
  unchanged when the thresholds are mapped with the same function (checked
  below: the predicted-pixel counts at tau and at tau' coincide);
* damage3d averages probabilities over views and scripts/28 over models:
  averages are meaningful only on a common, calibrated scale.

Inputs (all from the same checkpoint; the epoch is checked)
------
    <run>/eval_curves/curves_histograms.npz                              (scripts/25)
    <run>/eval_multilabel_sliding_calibrated/thresholds_per_class_validation_calibrated.json  (scripts/17)

Outputs
-------
    <run>/eval_curves/probability_calibration.json    (read by scripts/23, scripts/28, damage3d)
    <run>/eval_curves/reliability_platt.png

Usage
-----
    python scripts/27_fit_probability_calibration.py --run-dir outputs/runs/<run>
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.data.class_mapping import UNIFIED_DAMAGE_CLASSES
from src.eval.probability_calibration import (FORMAT_SINGLE, ece_from_histograms, fit_platt_from_histograms,
                                              map_thresholds, nll_from_histograms, reliability_curve,
                                              sha256_file)

THRESHOLDS_17 = Path("eval_multilabel_sliding_calibrated") / "thresholds_per_class_validation_calibrated.json"
HISTOGRAMS_25 = Path("eval_curves") / "curves_histograms.npz"
OUTPUT_NAME = Path("eval_curves") / "probability_calibration.json"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Per-class Platt calibration from scripts/25 histograms.")
    p.add_argument("--run-dir", required=True)
    p.add_argument("--checkpoint", default="best.pt", help="hashed (if present) to bind maps to this calibration")
    p.add_argument("--histograms", default=None, help=f"default: <run>/{HISTOGRAMS_25.as_posix()}")
    p.add_argument("--thresholds-json", default=None, help=f"default: <run>/{THRESHOLDS_17.as_posix()}")
    p.add_argument("--output", default=None, help=f"default: <run>/{OUTPUT_NAME.as_posix()}")
    p.add_argument("--ece-bins", type=int, default=15)
    p.add_argument("--allow-epoch-mismatch", action="store_true",
                   help="do not stop if histograms and thresholds come from different epochs")
    return p.parse_args()


def predicted_pixels(hist_pos: np.ndarray, hist_neg: np.ndarray, threshold: float) -> tuple[int, int]:
    """(TP, FP) at p >= threshold; exact when threshold * bins is an integer."""
    k = int(round(threshold * len(hist_pos)))
    return int(hist_pos[k:].sum()), int(hist_neg[k:].sum())


def dice_at(hist_pos, hist_neg, threshold: float) -> float:
    tp, fp = predicted_pixels(hist_pos, hist_neg, threshold)
    fn = int(hist_pos.sum()) - tp
    return 2 * tp / max(2 * tp + fp + fn, 1)


def plot_reliability(hist_pos, hist_neg, params, out_path: Path, title: str, n_bins: int) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    for ax, (c, name) in zip(axes.ravel(), enumerate(UNIFIED_DAMAGE_CLASSES)):
        a, b = params[name]["a"], params[name]["b"]
        for (aa, bb), label, style in (((1.0, 0.0), "raw", "o--"), ((a, b), "Platt", "s-")):
            curve = reliability_curve(hist_pos[c], hist_neg[c], aa, bb, n_bins)
            ece = ece_from_histograms(hist_pos[c], hist_neg[c], aa, bb, n_bins)
            ax.plot(curve[:, 0], curve[:, 1], style, ms=3, label=f"{label} (ECE {ece:.3f})")
        ax.plot([0, 1], [0, 1], color="grey", lw=0.8)
        ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="mean predicted probability", ylabel="observed frequency",
               title=f"{name}: a={a:.2f}, b={b:+.2f}")
        ax.legend(fontsize=8, loc="upper left")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    hist_path = Path(args.histograms) if args.histograms else run_dir / HISTOGRAMS_25
    thr_path = Path(args.thresholds_json) if args.thresholds_json else run_dir / THRESHOLDS_17
    out_path = Path(args.output) if args.output else run_dir / OUTPUT_NAME
    for path, step in ((hist_path, "scripts/25"), (thr_path, "scripts/17")):
        if not path.is_file():
            raise SystemExit(f"{path} not found: run {step} on this run first.")

    data = dict(np.load(hist_path, allow_pickle=False))
    classes = [str(c) for c in data["class_names"].tolist()]
    if classes != list(UNIFIED_DAMAGE_CLASSES):
        raise SystemExit(f"Class order in {hist_path} {classes} != {UNIFIED_DAMAGE_CLASSES}.")
    hist_pos, hist_neg = data["hist_pos"].astype(np.int64), data["hist_neg"].astype(np.int64)

    thr_payload = json.loads(thr_path.read_text(encoding="utf-8"))
    raw_thr = np.array([float(thr_payload["thresholds"][f"threshold_{c}"]) for c in classes])
    epoch_hist = str(data["checkpoint_epoch"])
    epoch_thr = str(thr_payload.get("checkpoint", {}).get("epoch", "unknown"))
    if epoch_hist != epoch_thr and not args.allow_epoch_mismatch:
        raise SystemExit(f"Checkpoint epoch mismatch: histograms {epoch_hist} vs thresholds {epoch_thr}.")
    if str(data["split"]) != str(thr_payload.get("split")):
        raise SystemExit(f"Split mismatch: histograms {data['split']} vs thresholds {thr_payload.get('split')}.")

    checkpoint_path = run_dir / args.checkpoint
    checkpoint_sha = sha256_file(checkpoint_path) if checkpoint_path.is_file() else None
    if checkpoint_sha is None:
        print(f"[platt] WARNING: {checkpoint_path} not found; maps cannot be bound to this calibration by hash.")

    params, diagnostics = {}, {}
    for c, name in enumerate(classes):
        a, b, info = fit_platt_from_histograms(hist_pos[c], hist_neg[c])
        params[name] = {"a": a, "b": b}
        diagnostics[name] = {
            "temperature": 1.0 / a,
            "n_positive_pixels": int(hist_pos[c].sum()), "n_negative_pixels": int(hist_neg[c].sum()),
            "nll_raw": nll_from_histograms(hist_pos[c], hist_neg[c]), "nll_platt": info["nll"],
            "ece_raw": ece_from_histograms(hist_pos[c], hist_neg[c], n_bins=args.ece_bins),
            "ece_platt": ece_from_histograms(hist_pos[c], hist_neg[c], a, b, n_bins=args.ece_bins),
            "newton_iterations": info["iterations"], "converged": info["converged"],
        }

    a_vec = np.array([params[c]["a"] for c in classes])
    b_vec = np.array([params[c]["b"] for c in classes])
    cal_thr = map_thresholds(raw_thr, a_vec, b_vec)
    for c, name in enumerate(classes):
        diagnostics[name]["dice_at_threshold_from_histograms"] = dice_at(hist_pos[c], hist_neg[c], raw_thr[c])

    payload = {
        "format": FORMAT_SINGLE,
        "method": "platt (per class): p' = sigmoid(a * logit(p) + b), a > 0; maximum likelihood on validation pixels",
        "classes": classes,
        "run_dir": str(run_dir.resolve()),
        "run_name": str(data["run_name"]),
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "checkpoint": {"name": args.checkpoint, "epoch": epoch_hist, "sha256": checkpoint_sha},
        "fit": {"split": str(data["split"]), "n_images": int(data["n_images"]),
                "histograms": str(hist_path.resolve()), "bins": int(hist_pos.shape[1]),
                "bin_representative": "logit of the bin centre"},
        "params": params,
        "thresholds_source": str(thr_path.resolve()),
        "thresholds_raw": {c: float(t) for c, t in zip(classes, raw_thr)},
        "thresholds_calibrated": {c: float(t) for c, t in zip(classes, cal_thr)},
        "diagnostics": diagnostics,
        "note": ("Monotone per-class map: with thresholds_calibrated the binary masks of THIS model are "
                 "identical to the raw masks at thresholds_raw. Probability maps stay RAW on disk; the "
                 "calibration is applied when maps are read (damage3d) or combined (scripts/28)."),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    plot_reliability(hist_pos, hist_neg, params, out_path.with_name("reliability_platt.png"),
                     f"{data['run_name']} (epoch {epoch_hist}) - validation reliability", args.ece_bins)

    print(f"[platt] {data['run_name']} | epoch {epoch_hist} | {int(data['n_images'])} validation images")
    print(f"{'class':13s} {'a':>6s} {'b':>7s} {'T':>5s} {'ECE raw':>8s} {'ECE cal':>8s} {'tau raw':>8s} {'tau cal':>8s}")
    for c, name in enumerate(classes):
        d = diagnostics[name]
        print(f"{name:13s} {params[name]['a']:6.3f} {params[name]['b']:+7.3f} {d['temperature']:5.2f} "
              f"{d['ece_raw']:8.4f} {d['ece_platt']:8.4f} {raw_thr[c]:8.2f} {cal_thr[c]:8.3f}")
    print(f"\n[platt] written {out_path}")


if __name__ == "__main__":
    main()
