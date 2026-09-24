#!/usr/bin/env python
"""Step 17 - per-channel threshold calibration for a multilabel run (P1ML or P4).

Why
---
A single global threshold is optimal only if every channel had the same
probability calibration, which is false by construction: the 6 unified damage
classes differ by more than two orders of magnitude in pixel prior, and a
`pos_weight` clipped to [1, 20] shifts each channel's logit distribution by a
different amount. Reporting a per-channel operating point therefore separates
*ranking quality* from *thresholding*, which is the honest way to compare
P1ML-A with P4-A.

Contract: ADDITIVE, never substitutive
--------------------------------------
The standard evaluation is untouched. These files are read, never rewritten:
    metrics_dacl10k_val_multilabel_sliding.csv
    metrics_dacl10k_val_multilabel_per_class.csv
These files are created:
    thresholds_per_class_validation_calibrated.json
    metrics_dacl10k_val_multilabel_per_class_calibrated.csv
    metrics_dacl10k_val_multilabel_sliding_calibrated.csv

Procedure
---------
1. read the standard per-class sweep CSV of the run;
2. per class, t_c = argmax_t Dice_c(t); ties are resolved towards the *highest*
   threshold (fewer predicted pixels, deterministic, conservative);
3. reload checkpoint + config and re-run the full-resolution sliding window on
   the validation split (nothing is re-derived from the stored global-threshold
   counts, which would be arithmetically invalid);
4. apply `probability[c] > t_c` independently per channel;
5. recompute TP/FP/FN/TN from scratch, then the per-class metrics and ONE
   aggregate row with mixed thresholds;
6. rows computed with different thresholds are never averaged or concatenated
   into a single metric: the calibrated aggregate comes from the mixed-threshold
   counts, not from combining the standard rows.

Both run families are supported:
* P4 multitask runs (`model.multilabel_classes` present) -> default checkpoint
  `best_joint.pt`, multilabel head only (the crack head keeps its own separate
  global sweep in scripts/15);
* P1ML single-head runs (`model.classes == 6`) -> default checkpoint `best.pt`.

Usage
-----
    python scripts/17_calibrate_multilabel_thresholds.py \
        --run-dir outputs/runs/p4a_unetpp_r34_shared_encoder

    python scripts/17_calibrate_multilabel_thresholds.py \
        --run-dir outputs/runs/p1ml_unetpp_r34_imagenet
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.data.class_mapping import UNIFIED_DAMAGE_CLASSES
from src.data.dacl10k import list_samples, load_annotation, rasterize_unified_damage
from src.data.transforms import IMAGENET_MEAN, IMAGENET_STD
from src.engine import load_checkpoint
from src.eval.multilabel_metrics import (
    MultilabelSegmentationMetrics,
    select_thresholds_per_class,
)
from src.eval.sliding_window import (
    predict_sliding_window_multilabel,
    predict_sliding_window_multitask,
)
from src.utils import ensure_dir, get_device, load_config, multilabel_patch_cfg, seed_everything

STANDARD_PER_CLASS_CSV = "metrics_dacl10k_val_multilabel_per_class.csv"
STANDARD_SUMMARY_CSV = "metrics_dacl10k_val_multilabel_sliding.csv"
CALIBRATED_PER_CLASS_CSV = "metrics_dacl10k_val_multilabel_per_class_calibrated.csv"
CALIBRATED_SUMMARY_CSV = "metrics_dacl10k_val_multilabel_sliding_calibrated.csv"
THRESHOLDS_JSON = "thresholds_per_class_validation_calibrated.json"

CALIBRATION_NOTE = (
    "validation-calibrated thresholds: selected on the same split they are "
    "evaluated on, therefore OPTIMISTIC. Not a blind-test result. Any held-out "
    "claim (testdev/testchallenge or the Lincoln Street case study) must reuse "
    "these frozen thresholds without re-selecting them."
)

# (evaluation subdirectory, default checkpoint) per run family.
EVAL_DIR_CANDIDATES = ("eval_multitask_sliding", "eval_multilabel_sliding")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Additive per-channel threshold calibration for a P1ML or P4 multilabel run."
    )
    parser.add_argument("--run-dir", required=True)
    parser.add_argument(
        "--eval-dir",
        default=None,
        help=f"directory holding the standard CSVs (default: first existing of {EVAL_DIR_CANDIDATES})",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="checkpoint inside --run-dir (default: best_joint.pt for P4, best.pt for P1ML)",
    )
    parser.add_argument("--split", default=None, help="DACL10K split (default: the run's val split)")
    parser.add_argument("--limit", type=int, default=None, help="debug: evaluate only N images")
    parser.add_argument(
        "--metric",
        default=None,
        help="per-class selection metric (default: eval.calibration.metric or 'dice')",
    )
    return parser.parse_args()


def resolve_eval_dir(run_dir: Path, explicit: str | None) -> Path:
    """Locate the directory holding the standard (global-threshold) CSVs."""
    if explicit is not None:
        eval_dir = Path(explicit)
        if not (eval_dir / STANDARD_PER_CLASS_CSV).is_file():
            raise FileNotFoundError(f"{eval_dir / STANDARD_PER_CLASS_CSV} not found.")
        return eval_dir

    for name in EVAL_DIR_CANDIDATES:
        candidate = run_dir / name
        if (candidate / STANDARD_PER_CLASS_CSV).is_file():
            return candidate

    raise FileNotFoundError(
        f"No standard per-class sweep CSV found under {run_dir}. Run the standard "
        "evaluation first (scripts/15 for P4, scripts/12 for P1ML)."
    )


def build_predictor(cfg, run_dir: Path, checkpoint_name: str | None, device):
    """Return `(predict_fn, checkpoint_path, checkpoint, run_family)`.

    `predict_fn(image, patch_cfg) -> multilabel probability [C,H,W]` hides whether
    the multilabel channels come from a single-head P1ML network or from the
    multilabel head of a P4 multitask network, so the calibration procedure below
    is literally the same code for both families.
    """
    is_multitask = cfg.model.get("multilabel_classes") is not None

    if is_multitask:
        from src.models.multitask import build_multitask_model

        model = build_multitask_model(cfg.model).to(device)
        family, default_checkpoint = "P4_multitask", "best_joint.pt"
    else:
        from src.models.unet import build_model

        if int(cfg.model.classes) != len(UNIFIED_DAMAGE_CLASSES):
            raise ValueError(
                f"This run emits {cfg.model.classes} channels: it is not a multilabel run."
            )
        model = build_model(cfg.model).to(device)
        family, default_checkpoint = "P1ML_single_head", "best.pt"

    checkpoint_path = run_dir / (checkpoint_name or default_checkpoint)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    checkpoint = load_checkpoint(checkpoint_path, model, device=device)

    def predict(image: np.ndarray, patch_cfg) -> np.ndarray:
        kwargs = dict(
            model=model,
            image=image,
            device=device,
            patch_size=int(patch_cfg.patch_size),
            stride=int(patch_cfg.eval_stride),
            batch_size=int(patch_cfg.eval_batch_size),
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD,
            n_classes=len(UNIFIED_DAMAGE_CLASSES),
            blend_mode=str(patch_cfg.blend_mode),
        )
        if is_multitask:
            _, multilabel = predict_sliding_window_multitask(**kwargs)
            return multilabel.numpy()
        return predict_sliding_window_multilabel(**kwargs).numpy()

    return predict, checkpoint_path, checkpoint, family


def load_rgb_and_mask(sample) -> tuple[np.ndarray, np.ndarray]:
    image_path, annotation_path = sample

    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise RuntimeError(f"Unreadable image: {image_path}")
    image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    annotation = load_annotation(annotation_path)
    mask = rasterize_unified_damage(annotation, shape=image.shape[:2]).astype(np.float32)
    return image, mask


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    cfg = load_config(run_dir / "config.yaml")
    seed_everything(cfg.project.seed)
    device = get_device()

    patch_cfg = multilabel_patch_cfg(cfg)
    eval_dir = resolve_eval_dir(run_dir, args.eval_dir)

    calibration_cfg = cfg.eval.get("calibration", {}) or {}
    metric = str(args.metric or calibration_cfg.get("metric", "dice"))

    # ---------------- 1-2) selection from the standard sweep ----------------
    standard_per_class = pd.read_csv(eval_dir / STANDARD_PER_CLASS_CSV)
    required = {"class", "threshold", metric}
    missing = required - set(standard_per_class.columns)
    if missing:
        raise ValueError(f"{STANDARD_PER_CLASS_CSV} lacks the columns {sorted(missing)}.")

    selection = select_thresholds_per_class(
        standard_per_class[["class", "threshold", metric]].to_dict("records"),
        class_names=UNIFIED_DAMAGE_CLASSES,
        metric=metric,
    )
    thresholds = {name: float(values["threshold"]) for name, values in selection.items()}

    print(f"[calibration] source sweep : {eval_dir / STANDARD_PER_CLASS_CSV}")
    print(f"[calibration] criterion    : argmax {metric}, ties -> highest threshold")
    for name in UNIFIED_DAMAGE_CLASSES:
        values = selection[name]
        print(
            f"[calibration] {name:>13}: t = {values['threshold']:.2f} "
            f"({metric} {values[f'{metric}_at_selected_threshold']:.4f}, "
            f"{values['n_tied_candidates']} tied)"
        )

    # ---------------- 3) reload the model and re-run inference ----------------
    predict, checkpoint_path, checkpoint, family = build_predictor(
        cfg, run_dir, args.checkpoint, device
    )
    split = args.split or str(cfg.data.dacl10k.val_split)
    samples = list_samples(cfg.data.dacl10k.root, split)
    if args.limit:
        samples = samples[: args.limit]

    print(
        f"[calibration] run family   : {family}\n"
        f"[calibration] checkpoint   : {checkpoint_path}\n"
        f"[calibration] split        : {split} ({len(samples)} images), "
        f"patch {patch_cfg.patch_size} / stride {patch_cfg.eval_stride}"
    )

    # ---------------- 4-5) mixed thresholds, counts recomputed from scratch ----------------
    meter = MultilabelSegmentationMetrics(
        threshold=[thresholds[name] for name in UNIFIED_DAMAGE_CLASSES],
        class_names=list(UNIFIED_DAMAGE_CLASSES),
    )

    for index, sample in enumerate(samples, start=1):
        image, mask = load_rgb_and_mask(sample)
        probability = predict(image, patch_cfg)

        logits = torch.logit(
            torch.from_numpy(probability).clamp(1e-6, 1.0 - 1e-6)
        ).unsqueeze(0)
        meter.update(logits, torch.from_numpy(mask).unsqueeze(0))

        if index % 25 == 0 or index == len(samples):
            print(f"[calibration] processed {index}/{len(samples)} images")

    aggregated = meter.compute()
    per_class_values = meter.per_class()

    # ---------------- 6) additive artifacts ----------------
    shared_columns = {
        "dataset": f"dacl10k_{split}_multilabel_sliding_calibrated",
        "run_family": family,
        "threshold_mode": "per_class_validation_calibrated",
        "selection_metric": metric,
        "tie_break": "highest threshold among the tied maxima",
        "patch_size": int(patch_cfg.patch_size),
        "stride": int(patch_cfg.eval_stride),
        "blend_mode": str(patch_cfg.blend_mode),
        "checkpoint_name": checkpoint_path.name,
        "checkpoint_epoch": checkpoint.get("epoch", "unknown"),
        "split": split,
        "n_images": len(samples),
        "note": CALIBRATION_NOTE,
    }

    per_class_rows = [
        {
            **shared_columns,
            "class": name,
            "threshold": thresholds[name],
            **per_class_values[name],
            f"{metric}_global_sweep_best": selection[name][f"{metric}_at_selected_threshold"],
            "n_tied_candidates": selection[name]["n_tied_candidates"],
        }
        for name in UNIFIED_DAMAGE_CLASSES
    ]
    calibrated_per_class = pd.DataFrame(per_class_rows)

    summary_row = {
        **shared_columns,
        **{
            key: value
            for key, value in aggregated.items()
            if not any(key.endswith(f"_{name}") for name in UNIFIED_DAMAGE_CLASSES)
        },
        **{f"threshold_{name}": thresholds[name] for name in UNIFIED_DAMAGE_CLASSES},
    }
    calibrated_summary = pd.DataFrame([summary_row])

    calibrated_per_class.to_csv(eval_dir / CALIBRATED_PER_CLASS_CSV, index=False)
    calibrated_summary.to_csv(eval_dir / CALIBRATED_SUMMARY_CSV, index=False)

    payload = {
        "pipeline": "per-channel threshold calibration (additive)",
        "run_dir": str(run_dir.resolve()),
        "run_family": family,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "checkpoint": {
            "path": str(checkpoint_path.resolve()),
            "name": checkpoint_path.name,
            "epoch": checkpoint.get("epoch", "unknown"),
            "best_monitor_metric": (
                float(checkpoint["best_dice"]) if "best_dice" in checkpoint else None
            ),
            "selection_metric_of_the_run": str(cfg.train.get("selection_metric", "unknown")),
        },
        "split": split,
        "n_images": len(samples),
        "source_sweep_csv": str((eval_dir / STANDARD_PER_CLASS_CSV).resolve()),
        "candidate_thresholds": sorted(standard_per_class["threshold"].unique().tolist()),
        "selection_metric": metric,
        "tie_break": "highest threshold among the tied maxima",
        "inference": {
            "patch_size": int(patch_cfg.patch_size),
            "stride": int(patch_cfg.eval_stride),
            "batch_size": int(patch_cfg.eval_batch_size),
            "blend_mode": str(patch_cfg.blend_mode),
        },
        "thresholds": {f"threshold_{name}": thresholds[name] for name in UNIFIED_DAMAGE_CLASSES},
        "per_class_selection": selection,
        "calibrated_metrics": {
            "macro_dice_present": aggregated["macro_dice_present"],
            "macro_iou_present": aggregated["macro_iou_present"],
            "macro_precision_present": aggregated["macro_precision_present"],
            "macro_recall_present": aggregated["macro_recall_present"],
            "micro_dice": aggregated["micro_dice"],
            "micro_iou": aggregated["micro_iou"],
            "micro_precision": aggregated["micro_precision"],
            "micro_recall": aggregated["micro_recall"],
            "false_alarm_rate_empty_gt": aggregated["false_alarm_rate_empty_gt"],
            "per_class": {
                name: {
                    "threshold": thresholds[name],
                    "dice": per_class_values[name]["dice"],
                    "iou": per_class_values[name]["iou"],
                    "precision": per_class_values[name]["precision"],
                    "recall": per_class_values[name]["recall"],
                    "false_alarm_rate_empty_gt": per_class_values[name][
                        "false_alarm_rate_empty_gt"
                    ],
                    "support_pixels": per_class_values[name]["support_pixels"],
                }
                for name in UNIFIED_DAMAGE_CLASSES
            },
        },
        "standard_artifacts_preserved": [STANDARD_SUMMARY_CSV, STANDARD_PER_CLASS_CSV],
        "note": CALIBRATION_NOTE,
    }
    with open(eval_dir / THRESHOLDS_JSON, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)

    print("\n--- calibrated per-class metrics (mixed thresholds) ---")
    print(
        calibrated_per_class[
            ["class", "threshold", "dice", "iou", "precision", "recall", "false_alarm_rate_empty_gt"]
        ].to_string(index=False)
    )

    standard_summary_path = eval_dir / STANDARD_SUMMARY_CSV
    if standard_summary_path.is_file():
        standard_summary = pd.read_csv(standard_summary_path)
        reference = standard_summary.loc[
            (standard_summary["threshold"] - float(cfg.eval.threshold)).abs().idxmin()
        ]
        print(
            f"\nmacro Dice @ global {reference['threshold']:.2f} : "
            f"{reference['macro_dice_present']:.4f}  (standard, untouched)"
        )
    print(f"macro Dice @ per-class thresholds : {aggregated['macro_dice_present']:.4f}  (calibrated)")
    print(f"\n[calibration] {CALIBRATION_NOTE}")
    print(f"[calibration] artifacts written to {ensure_dir(eval_dir).resolve()}:")
    for name in (THRESHOLDS_JSON, CALIBRATED_PER_CLASS_CSV, CALIBRATED_SUMMARY_CSV):
        print(f"  + {name}")


if __name__ == "__main__":
    main()
