#!/usr/bin/env python
"""Step 21 - sub-class diagnosis of the merged multilabel channels (no training).

Question
--------
Several unified channels merge DACL10K labels that look very different
(moisture = Wetspot + Efflorescence, surface = Graffiti + Weathering +
Restformwork, spalling = Rockpocket + Cavity + Spalling). If the model serves
one sub-class well and another badly, the merged class definition, not the
network, limits the channel, and splitting it into separate outputs is
justified. If all sub-classes behave alike, splitting will not help.

What is measured (per DACL10K sub-class s with parent channel c)
---------------------------------------------------------------
Predictions use the frozen operating points (tau_c, A_c) of scripts/20.
Precision and FAR cannot be split by sub-class (the model emits one channel per
parent), so every metric is recall-type, computed on the GT pixels of s:

* recall_threshold_only : |(p_c > tau_c) & GT_s| / |GT_s|
* recall_postproc       : same, after the min-area filter (the operating mask)
* recall_exclusive      : recall_postproc on the pixels of s that belong to no
                          other sub-class of the same parent (no shared credit)
* soft_recall           : mean p_c over GT_s pixels (threshold-free)
* object_recall         : share of GT_s components touched by a kept component
* object_recall_large   : same, on GT_s components with area >= A_c
* image_detection_rate  : share of images containing s where >= 1 kept
                          component of c touches GT_s
* activation matrix     : share of GT_s pixels predicted by EACH of the 6 channels
                          (and by none): tells a confusion with another class
                          apart from a plain miss

Usage
-----
    python scripts/21_diagnose_subclasses.py --run-dir outputs/runs/<run>
    python scripts/21_diagnose_subclasses.py --run-dir outputs/runs/<run> --subset heldout
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.data.class_mapping import DACL10K_TO_UNIFIED, UNIFIED_DAMAGE_CLASSES
from src.data.dacl10k import _fill, list_samples, load_annotation
from src.eval.postprocess import min_area_pixels, remove_small_components
from src.utils import ensure_dir, get_device, load_config, multilabel_patch_cfg, seed_everything

DEFAULT_LABELS = (
    "Wetspot", "Efflorescence",
    "Rockpocket", "Cavity", "Spalling",
    "Graffiti", "Weathering", "Restformwork",
)
ACCUMULATOR_FIELDS = (
    "n_images", "n_images_detected", "support_pixels", "tp_threshold_only", "tp_postproc",
    "support_exclusive", "tp_exclusive", "prob_sum",
    "n_components", "n_components_detected", "n_components_large", "n_components_large_detected",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sub-class recall diagnosis of merged channels.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--split", default=None)
    parser.add_argument("--limit", type=int, default=None, help="debug: first N images")
    parser.add_argument("--operating-points", default=None,
                        help="operating_points_frozen.json of scripts/20 "
                             "(default: <run>/eval_multilabel_postproc_objrec05/...)")
    parser.add_argument("--mode", default="threshold_min_area", choices=("threshold_min_area", "threshold_only"))
    parser.add_argument("--subset", default="all", choices=("all", "heldout"),
                        help="heldout = only the images NOT used to select the operating points")
    parser.add_argument("--labels", nargs="+", default=list(DEFAULT_LABELS),
                        help="DACL10K labels to diagnose; 'all' = every mapped damage label")
    parser.add_argument("--output-dir", default=None, help="default: <run>/eval_subclass_diagnosis")
    return parser.parse_args()


def load_step17():
    path = Path(__file__).with_name("17_calibrate_multilabel_thresholds.py")
    spec = importlib.util.spec_from_file_location("step17_calibration", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def resolve_labels(requested) -> dict[str, int]:
    """{DACL10K label -> parent channel index}, validated against the mapping."""
    mapped = {label: parent for label, parent in DACL10K_TO_UNIFIED.items() if parent is not None}
    labels = list(mapped) if requested == ["all"] else list(requested)
    unknown = [label for label in labels if label not in mapped]
    if unknown:
        raise ValueError(f"Labels not mapped to a damage channel: {unknown}. Known: {sorted(mapped)}")
    return {label: UNIFIED_DAMAGE_CLASSES.index(mapped[label]) for label in labels}


def label_masks(annotation: dict, labels, shape) -> dict[str, np.ndarray]:
    """Boolean [H,W] mask per DACL10K label (same polygon fill as the training targets)."""
    masks = {label: np.zeros(shape, dtype=np.uint8) for label in labels}
    for shape_dict in annotation.get("shapes", []):
        label = shape_dict.get("label")
        if label in masks:
            _fill(masks[label], shape_dict.get("points", []))
    return {label: mask.astype(bool) for label, mask in masks.items()}


def operating_masks(probability, thresholds, area_fractions):
    """Threshold-only and post-processed [C,H,W] boolean masks."""
    thresholded = probability > np.asarray(thresholds, dtype=np.float32)[:, None, None]
    kept = np.stack([
        remove_small_components(thresholded[c], min_area_pixels(area_fractions[c], probability.shape[1:]))
        for c in range(probability.shape[0])
    ])
    return thresholded, kept


def accumulate(acc, activation, probability, thresholded, kept, masks, label_parent, area_px) -> None:
    """Update the per-label accumulators with one image (pure numpy, testable)."""
    for label, parent in label_parent.items():
        gt = masks[label]
        support = int(gt.sum())
        if support == 0:
            continue
        siblings = [masks[o] for o, p in label_parent.items() if p == parent and o != label]
        exclusive = gt & ~np.logical_or.reduce(siblings) if siblings else gt

        row = acc[label]
        row["n_images"] += 1
        row["support_pixels"] += support
        row["tp_threshold_only"] += int((thresholded[parent] & gt).sum())
        hit = kept[parent] & gt
        row["tp_postproc"] += int(hit.sum())
        row["support_exclusive"] += int(exclusive.sum())
        row["tp_exclusive"] += int((kept[parent] & exclusive).sum())
        row["prob_sum"] += float(probability[parent][gt].sum())
        row["n_images_detected"] += int(hit.any())

        n, components, stats, _ = cv2.connectedComponentsWithStats(gt.astype(np.uint8), connectivity=8)
        areas = stats[1:, cv2.CC_STAT_AREA]
        detected = np.zeros(n, dtype=bool)
        detected[np.unique(components[hit])] = True
        detected = detected[1:]
        large = areas >= area_px[parent]
        row["n_components"] += n - 1
        row["n_components_detected"] += int(detected.sum())
        row["n_components_large"] += int(large.sum())
        row["n_components_large_detected"] += int((detected & large).sum())

        activation[label] += np.append((kept & gt[None]).reshape(kept.shape[0], -1).sum(1),
                                       int((~kept.any(0) & gt).sum()))


def finalize(acc, activation, label_parent) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    parent_support = {}
    for label, parent in label_parent.items():
        parent_support[parent] = parent_support.get(parent, 0) + acc[label]["support_pixels"]
    for label, parent in label_parent.items():
        a = acc[label]
        s = max(a["support_pixels"], 1)
        rows.append({
            "parent": UNIFIED_DAMAGE_CLASSES[parent],
            "subclass": label,
            "n_images": a["n_images"],
            "support_pixels": a["support_pixels"],
            "share_of_parent_pixels": a["support_pixels"] / max(parent_support[parent], 1),
            "recall_threshold_only": a["tp_threshold_only"] / s,
            "recall_postproc": a["tp_postproc"] / s,
            "recall_exclusive": a["tp_exclusive"] / max(a["support_exclusive"], 1),
            "exclusive_share": a["support_exclusive"] / s,
            "soft_recall": a["prob_sum"] / s,
            "object_recall": a["n_components_detected"] / max(a["n_components"], 1),
            "object_recall_large": a["n_components_large_detected"] / max(a["n_components_large"], 1),
            "image_detection_rate": a["n_images_detected"] / max(a["n_images"], 1),
        })
    table = pd.DataFrame(rows).sort_values(["parent", "subclass"]).reset_index(drop=True)
    matrix = pd.DataFrame(
        {label: activation[label] / max(acc[label]["support_pixels"], 1) for label in label_parent},
        index=[*UNIFIED_DAMAGE_CLASSES, "none"],
    ).T
    return table, matrix.loc[table["subclass"]]


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    cfg = load_config(run_dir / "config.yaml")
    seed_everything(cfg.project.seed)
    output_dir = ensure_dir(Path(args.output_dir) if args.output_dir else run_dir / "eval_subclass_diagnosis")

    op_path = Path(args.operating_points) if args.operating_points else (
        run_dir / "eval_multilabel_postproc_objrec05" / "operating_points_frozen.json")
    with open(op_path, encoding="utf-8") as fh:
        op_payload = json.load(fh)
    points = op_payload["frozen_fold0"][args.mode]
    thresholds = [float(points[name]["threshold"]) for name in UNIFIED_DAMAGE_CLASSES]
    area_fractions = [float(points[name]["min_area_fraction"]) for name in UNIFIED_DAMAGE_CLASSES]

    label_parent = resolve_labels(args.labels)
    step17 = load_step17()
    patch_cfg = multilabel_patch_cfg(cfg)
    predict, checkpoint_path, checkpoint, family = step17.build_predictor(
        cfg, run_dir, args.checkpoint, get_device())

    split = args.split or str(cfg.data.dacl10k.val_split)
    samples = list_samples(cfg.data.dacl10k.root, split)
    if args.subset == "heldout":
        calibration = set(op_payload["calibration_images"])
        samples = [s for s in samples if Path(s[0]).name not in calibration]
    if args.limit:
        samples = samples[: args.limit]

    print(f"[subclass] {family} | {checkpoint_path.name} | {split} ({args.subset}): {len(samples)} images")
    print(f"[subclass] operating points ({args.mode}) from {op_path}")
    for name, t, a in zip(UNIFIED_DAMAGE_CLASSES, thresholds, area_fractions):
        print(f"[subclass] {name:>13}: tau = {t:.2f} | A = {a:.0e}")

    acc = {label: dict.fromkeys(ACCUMULATOR_FIELDS, 0) for label in label_parent}
    activation = {label: np.zeros(len(UNIFIED_DAMAGE_CLASSES) + 1, dtype=np.int64) for label in label_parent}

    for index, sample in enumerate(samples, start=1):
        image, _ = step17.load_rgb_and_mask(sample)
        probability = predict(image, patch_cfg)
        masks = label_masks(load_annotation(sample[1]), label_parent, image.shape[:2])
        if any(masks[label].any() for label in label_parent):
            thresholded, kept = operating_masks(probability, thresholds, area_fractions)
            area_px = [min_area_pixels(a, probability.shape[1:]) for a in area_fractions]
            accumulate(acc, activation, probability, thresholded, kept, masks, label_parent, area_px)
        if index % 25 == 0 or index == len(samples):
            print(f"[subclass] processed {index}/{len(samples)} images")

    table, matrix = finalize(acc, activation, label_parent)
    table.to_csv(output_dir / "subclass_diagnosis.csv", index=False)
    matrix.to_csv(output_dir / "subclass_activation_matrix.csv")
    with open(output_dir / "subclass_diagnosis_meta.json", "w", encoding="utf-8") as fh:
        json.dump({
            "run_dir": str(run_dir.resolve()),
            "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "checkpoint": {"name": checkpoint_path.name, "epoch": str(checkpoint.get("epoch", "unknown"))},
            "split": split, "subset": args.subset, "n_images": len(samples),
            "operating_points_file": str(op_path.resolve()), "mode": args.mode,
            "thresholds": dict(zip(UNIFIED_DAMAGE_CLASSES, thresholds)),
            "min_area_fractions": dict(zip(UNIFIED_DAMAGE_CLASSES, area_fractions)),
            "labels": label_parent,
        }, fh, indent=2)

    view = ["parent", "subclass", "n_images", "share_of_parent_pixels", "recall_threshold_only",
            "recall_postproc", "recall_exclusive", "soft_recall", "object_recall",
            "object_recall_large", "image_detection_rate"]
    pd.set_option("display.width", 250)
    print("\n--- sub-class recall of the parent channel ---")
    print(table[view].round(3).to_string(index=False))
    print("\n--- share of GT_s pixels predicted by each channel (post-processed) ---")
    print(matrix.round(3).to_string())
    print(f"\n[subclass] artifacts in {output_dir.resolve()}")


if __name__ == "__main__":
    main()
