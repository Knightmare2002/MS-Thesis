#!/usr/bin/env python
"""Step 25 - pixel precision-recall curves and multilabel confusion matrix.

Why these two and not ROC
-------------------------
* PR curve / Average Precision (AP): threshold-free ranking quality of each
  channel. Model comparisons stop depending on threshold calibration and on
  post-processing (AP is computed on the RAW probabilities).
* Multilabel confusion matrix: a K x K matrix is undefined when a pixel can
  carry several labels or none. The multilabel analogue used here is a
  co-activation matrix: row = GT class (plus "background" = no damage label),
  column = predicted channel (plus "none"); entry = share of the row's pixels
  predicted by that channel at the frozen operating points of scripts/20. The
  GT co-occurrence matrix is saved too, so that excess = activation - GT
  overlap isolates genuine confusions from legitimate multilabel overlap.
* ROC-AUC is not produced: with damage pixels at a few percent of all pixels,
  FPR is driven by the huge TN count, so ROC-AUC is inflated and hardly
  separates models. The histograms saved here would allow it if ever needed.

Exactness
---------
Probabilities are binned into `--bins` uniform bins per class (positives and
negatives separately), so curves are exact up to the bin width (1e-3 default)
over billions of pixels with constant memory.

Usage
-----
    # one run: inference + CSV + figures
    python scripts/25_pr_curves_confusion.py --run-dir outputs/runs/<run> \
        --operating-points outputs/runs/<run>/eval_multilabel_postproc_objrec05/operating_points_frozen.json

    # overlay several runs from their saved histograms (no inference)
    python scripts/25_pr_curves_confusion.py --compare runA/eval_curves/curves_histograms.npz \
        runB/eval_curves/curves_histograms.npz --labels T4 T9 --output-dir outputs/compare_t4_t9
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.data.class_mapping import UNIFIED_DAMAGE_CLASSES
from src.data.dacl10k import list_samples
from src.eval.postprocess import apply_operating_points
from src.utils import ensure_dir, get_device, load_config, multilabel_patch_cfg, seed_everything

ROWS = [*UNIFIED_DAMAGE_CLASSES, "background"]
COLS = [*UNIFIED_DAMAGE_CLASSES, "none"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PR curves (AP) and multilabel confusion matrix.")
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--split", default=None)
    parser.add_argument("--limit", type=int, default=None, help="debug: first N images")
    parser.add_argument("--operating-points", default=None, help="operating_points_frozen.json of scripts/20")
    parser.add_argument("--mode", default="threshold_min_area", choices=("threshold_min_area", "threshold_only"))
    parser.add_argument("--bins", type=int, default=1000)
    parser.add_argument("--output-dir", default=None, help="default: <run>/eval_curves")
    parser.add_argument("--compare", nargs="+", default=None, help="curves_histograms.npz files to overlay")
    parser.add_argument("--labels", nargs="+", default=None, help="legend labels for --compare")
    return parser.parse_args()


def load_step17():
    path = Path(__file__).with_name("17_calibrate_multilabel_thresholds.py")
    spec = importlib.util.spec_from_file_location("step17_calibration", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --------------------------------------------------------------------------- #
# Accumulation (pure numpy, testable)
# --------------------------------------------------------------------------- #
def new_accumulator(n_classes: int, bins: int) -> dict:
    return {
        "hist_pos": np.zeros((n_classes, bins), dtype=np.int64),
        "hist_neg": np.zeros((n_classes, bins), dtype=np.int64),
        "activation": np.zeros((n_classes + 1, n_classes + 1), dtype=np.int64),  # ROWS x COLS
        "gt_overlap": np.zeros((n_classes + 1, n_classes), dtype=np.int64),       # ROWS x classes
        "row_support": np.zeros(n_classes + 1, dtype=np.int64),
    }


def accumulate(acc: dict, probability: np.ndarray, target: np.ndarray, kept: np.ndarray) -> None:
    """probability [C,H,W] float, target [C,H,W] {0,1}, kept [C,H,W] bool (operating masks)."""
    n_classes, bins = acc["hist_pos"].shape
    gt = target > 0
    index = np.minimum((probability * bins).astype(np.int64), bins - 1)
    for c in range(n_classes):
        acc["hist_pos"][c] += np.bincount(index[c][gt[c]], minlength=bins)
        acc["hist_neg"][c] += np.bincount(index[c][~gt[c]], minlength=bins)

    rows = [*gt, ~gt.any(0)]
    none = ~kept.any(0)
    for r, row_mask in enumerate(rows):
        n = int(row_mask.sum())
        if n == 0:
            continue
        acc["row_support"][r] += n
        for k in range(n_classes):
            acc["activation"][r, k] += int((kept[k] & row_mask).sum())
            acc["gt_overlap"][r, k] += int((gt[k] & row_mask).sum())
        acc["activation"][r, n_classes] += int((none & row_mask).sum())


def pr_from_histograms(pos: np.ndarray, neg: np.ndarray) -> dict:
    """Curve over descending thresholds t_b = b / bins (pixel predicted if p >= t_b)."""
    bins = len(pos)
    tp = np.cumsum(pos[::-1])[::-1].astype(np.float64)
    fp = np.cumsum(neg[::-1])[::-1].astype(np.float64)
    total_pos = max(float(pos.sum()), 1.0)
    recall = tp / total_pos
    precision = np.divide(tp, tp + fp, out=np.ones_like(tp), where=(tp + fp) > 0)
    # AP = sum_b (R_b - R_{b+1}) * P_b, thresholds visited from high to low.
    recall_next = np.append(recall[1:], 0.0)
    ap = float(np.sum((recall - recall_next) * precision))
    dice = np.divide(2 * precision * recall, precision + recall,
                     out=np.zeros_like(tp), where=(precision + recall) > 0)
    best = int(np.argmax(dice))
    return {
        "thresholds": np.arange(bins) / bins,
        "precision": precision,
        "recall": recall,
        "ap": ap,
        "prevalence": float(pos.sum()) / max(float(pos.sum() + neg.sum()), 1.0),
        "best_dice": float(dice[best]),
        "best_threshold": best / bins,
    }


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def plot_pr(entries, out_path: Path) -> pd.DataFrame:
    """entries: list of (label, npz dict). Returns the AP table."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 9.5), squeeze=False)
    rows = []
    for c, name in enumerate(UNIFIED_DAMAGE_CLASSES):
        ax = axes[c // 3, c % 3]
        for label, data in entries:
            curve = pr_from_histograms(data["hist_pos"][c], data["hist_neg"][c])
            line, = ax.plot(curve["recall"], curve["precision"], lw=1.8,
                            label=f"{label} (AP {curve['ap']:.3f})")
            tau = float(data["thresholds_op"][c])
            b = min(int(round(tau * len(curve["thresholds"]))), len(curve["thresholds"]) - 1)
            ax.plot(curve["recall"][b], curve["precision"][b], "o", color=line.get_color(), ms=6)
            rows.append({"run": label, "class": name, "ap": curve["ap"], "prevalence": curve["prevalence"],
                         "best_dice_pixel": curve["best_dice"], "best_threshold": curve["best_threshold"],
                         "tau_operating": tau})
        ax.axhline(curve["prevalence"], color="grey", ls=":", lw=1)
        ax.set(title=name, xlabel="recall", ylabel="precision", xlim=(0, 1), ylim=(0, 1))
        ax.grid(alpha=0.3)
        ax.legend(loc="upper right", fontsize=8)
    fig.suptitle("Pixel precision-recall (raw probabilities; dot = operating threshold; dotted = prevalence)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return pd.DataFrame(rows)


def confusion_tables(data) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    support = np.maximum(data["row_support"], 1)[:, None]
    activation = pd.DataFrame(data["activation"] / support, index=ROWS, columns=COLS)
    overlap = pd.DataFrame(data["gt_overlap"] / support, index=ROWS, columns=UNIFIED_DAMAGE_CLASSES)
    excess = activation[UNIFIED_DAMAGE_CLASSES] - overlap
    for c in range(len(UNIFIED_DAMAGE_CLASSES)):
        excess.iat[c, c] = 0.0  # diagonal = recall - 1, already in the activation panel
    return activation, overlap, excess


def plot_confusion(data, label: str, out_path: Path) -> None:
    activation, _, excess = confusion_tables(data)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6.5))
    panels = ((activation, "Blues", (0, 1), "share of GT-row pixels predicted by each channel"),
              (excess, "RdBu_r", (-0.5, 0.5), "excess = activation - GT co-occurrence"))
    for ax, (table, cmap, (vmin, vmax), title) in zip(axes, panels):
        im = ax.imshow(table.values, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_xticks(range(table.shape[1]), table.columns, rotation=45, ha="right")
        ax.set_yticks(range(table.shape[0]), table.index)
        for (i, j), v in np.ndenumerate(table.values):
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if abs(v) > 0.6 * max(abs(vmin), vmax) else "black")
        ax.set(xlabel="predicted channel", ylabel="ground truth", title=title)
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle(f"Multilabel confusion at the operating points - {label}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- #
def run_inference(args) -> Path:
    run_dir = Path(args.run_dir)
    cfg = load_config(run_dir / "config.yaml")
    seed_everything(cfg.project.seed)
    output_dir = ensure_dir(Path(args.output_dir) if args.output_dir else run_dir / "eval_curves")

    op_path = Path(args.operating_points) if args.operating_points else (
        run_dir / "eval_multilabel_postproc_objrec05" / "operating_points_frozen.json")
    if not op_path.is_file():
        raise FileNotFoundError(f"{op_path} not found: run scripts/20 first or pass --operating-points.")
    with open(op_path, encoding="utf-8") as fh:
        points = json.load(fh)["frozen_fold0"][args.mode]
    thresholds = [float(points[n]["threshold"]) for n in UNIFIED_DAMAGE_CLASSES]
    areas = [float(points[n]["min_area_fraction"]) for n in UNIFIED_DAMAGE_CLASSES]

    step17 = load_step17()
    patch_cfg = multilabel_patch_cfg(cfg)
    predict, checkpoint_path, checkpoint, family = step17.build_predictor(
        cfg, run_dir, args.checkpoint, get_device())
    split = args.split or str(cfg.data.dacl10k.val_split)
    samples = list_samples(cfg.data.dacl10k.root, split)
    if args.limit:
        samples = samples[: args.limit]
    print(f"[curves] {family} | {checkpoint_path.name} | {split}: {len(samples)} images | {op_path}")

    acc = new_accumulator(len(UNIFIED_DAMAGE_CLASSES), int(args.bins))
    for index, sample in enumerate(samples, start=1):
        image, mask = step17.load_rgb_and_mask(sample)
        probability = predict(image, patch_cfg)
        kept = apply_operating_points(probability, thresholds, areas)
        accumulate(acc, probability, mask, kept)
        if index % 25 == 0 or index == len(samples):
            print(f"[curves] processed {index}/{len(samples)} images")

    npz_path = output_dir / "curves_histograms.npz"
    np.savez_compressed(
        npz_path, **acc,
        thresholds_op=np.asarray(thresholds), min_area_op=np.asarray(areas),
        class_names=np.asarray(UNIFIED_DAMAGE_CLASSES), run_name=np.asarray(run_dir.name),
        checkpoint_epoch=np.asarray(str(checkpoint.get("epoch", "unknown"))), split=np.asarray(split),
        n_images=np.asarray(len(samples)), operating_points_file=np.asarray(str(op_path)),
    )
    return npz_path


def main() -> None:
    args = parse_args()
    if args.compare:
        files = [Path(p) for p in args.compare]
        labels = args.labels or [Path(p).parent.parent.name for p in files]
        if len(labels) != len(files):
            raise ValueError("--labels must match --compare.")
        output_dir = ensure_dir(Path(args.output_dir or "outputs/compare_curves"))
    elif args.run_dir:
        files = [run_inference(args)]
        labels = args.labels or [Path(args.run_dir).name]
        output_dir = files[0].parent
    else:
        raise SystemExit("Pass --run-dir (inference) or --compare (overlay saved histograms).")

    entries = [(label, dict(np.load(f, allow_pickle=False))) for label, f in zip(labels, files)]
    ap_table = plot_pr(entries, output_dir / "pr_curves.png")
    ap_table.to_csv(output_dir / "average_precision.csv", index=False)

    for label, data in entries:
        tag = label.replace(" ", "_")
        activation, overlap, excess = confusion_tables(data)
        activation.to_csv(output_dir / f"confusion_activation_{tag}.csv")
        overlap.to_csv(output_dir / f"confusion_gt_overlap_{tag}.csv")
        excess.to_csv(output_dir / f"confusion_excess_{tag}.csv")
        plot_confusion(data, label, output_dir / f"confusion_matrix_{tag}.png")

    pd.set_option("display.width", 200)
    summary = ap_table.pivot(index="class", columns="run", values="ap").loc[UNIFIED_DAMAGE_CLASSES]
    summary.loc["mAP"] = summary.mean()
    print("\n--- Average Precision (raw probabilities) ---")
    print(summary.round(3).to_string())
    for label, data in entries:
        print(f"\n--- excess activation (activation - GT co-occurrence): {label} ---")
        print(confusion_tables(data)[2].round(2).to_string())
    print(f"\n[curves] artifacts in {output_dir.resolve()}")


if __name__ == "__main__":
    main()
