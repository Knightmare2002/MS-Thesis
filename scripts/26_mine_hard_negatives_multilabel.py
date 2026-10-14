#!/usr/bin/env python
"""Step 26 - per-channel hard-negative mining for P1ML (T11).

A hard negative for channel c is a train-split window where the frozen model
(e.g. T9) predicts c at its operating point (tau_c + min-area, scripts/20)
although c is NOT annotated, and the false positive lies on the GT of one of
the "source" classes that the confusion matrix (scripts/25) identified:

    surface   <- moisture, crack      (moisture->surface excess .40, crack->surface .20)
    moisture  <- surface,  crack      (surface->moisture .11, crack->moisture .18)
    corrosion <- spalling             (spalling->corrosion .09)

The window keeps all its GT: it stays positive for the source classes, so the
model is taught to separate c from the source, not to suppress damage.

Output: JSON pool consumed by Dacl10kMultilabelPatchDataset
(data.p1ml_patch.hard_negative_pool).

Usage
-----
    # timing check first
    python scripts/26_mine_hard_negatives_multilabel.py --run-dir outputs/runs/p1ml_t9_unetplusplus_effb4_512 \
        --operating-points outputs/runs/p1ml_t9_unetplusplus_effb4_512/eval_multilabel_sliding_postproc_objrec05/operating_points_frozen.json \
        --limit 20 --output outputs/hnm/debug_pool.json
    # full train split
    python scripts/26_mine_hard_negatives_multilabel.py --run-dir ... --operating-points ... \
        --output outputs/hnm/p1ml_t9_hnm_pool.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.data.class_mapping import UNIFIED_DAMAGE_CLASSES
from src.data.dacl10k import list_samples
from src.eval.postprocess import apply_operating_points
from src.utils import ensure_dir, get_device, load_config, multilabel_patch_cfg, seed_everything

DEFAULT_RULES = ("surface:moisture,crack", "moisture:surface,crack", "corrosion:spalling")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Per-channel hard-negative mining (P1ML).")
    parser.add_argument("--run-dir", required=True, help="run that mines (e.g. T9)")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--operating-points", required=True, help="operating_points_frozen.json (scripts/20)")
    parser.add_argument("--mode", default="threshold_min_area", choices=("threshold_min_area", "threshold_only"))
    parser.add_argument("--split", default=None, help="default: data.dacl10k.train_split")
    parser.add_argument("--rules", nargs="+", default=list(DEFAULT_RULES),
                        help="target:source1,source2 ; 'background' = no GT in any channel")
    parser.add_argument("--source-dilation", type=int, default=16,
                        help="px dilation of the source GT (thin cracks)")
    parser.add_argument("--min-fp-pixels", type=int, default=2048, help="min FP pixels per window")
    parser.add_argument("--stride", type=int, default=None, help="window stride (default patch/2)")
    parser.add_argument("--max-per-image", type=int, default=2, help="windows per image and channel")
    parser.add_argument("--max-per-channel", type=int, default=4000, help="pool cap per channel")
    parser.add_argument("--limit", type=int, default=None, help="debug: first N eligible images")
    parser.add_argument("--max-images", type=int, default=None, help="seeded random subset of eligible images")
    parser.add_argument("--output", default="outputs/hnm/p1ml_t9_hnm_pool.json")
    return parser.parse_args()


def load_step17():
    path = Path(__file__).with_name("17_calibrate_multilabel_thresholds.py")
    spec = importlib.util.spec_from_file_location("step17_calibration", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_rules(rules: list[str]) -> dict[int, list[str]]:
    parsed = {}
    names = set(UNIFIED_DAMAGE_CLASSES) | {"background"}
    for rule in rules:
        target, sources = rule.split(":")
        sources = [s.strip() for s in sources.split(",") if s.strip()]
        if target not in UNIFIED_DAMAGE_CLASSES or not sources or not set(sources) <= names or target in sources:
            raise ValueError(f"invalid rule '{rule}'")
        parsed[UNIFIED_DAMAGE_CLASSES.index(target)] = sources
    return parsed


def source_region(gt: np.ndarray, sources: list[str], dilation: int) -> np.ndarray:
    """gt [C,H,W] bool -> bool [H,W] region covered by the source classes."""
    region = np.zeros(gt.shape[1:], dtype=bool)
    for name in sources:
        region |= ~gt.any(0) if name == "background" else gt[UNIFIED_DAMAGE_CLASSES.index(name)]
    if dilation > 0 and region.any():
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilation + 1, 2 * dilation + 1))
        region = cv2.dilate(region.astype(np.uint8), kernel) > 0
    return region


def select_windows(fp: np.ndarray, patch: int, stride: int, min_pixels: int, k: int) -> list[tuple[int, int, int]]:
    """Greedy top-k windows (top, left, fp_pixels) by FP count, at most 50% overlap per axis."""
    height, width = fp.shape
    if height < patch or width < patch or not fp.any():
        return []
    integral = cv2.integral(fp.astype(np.uint8), sdepth=cv2.CV_32S)
    tops = sorted(set(range(0, height - patch + 1, stride)) | {height - patch})
    lefts = sorted(set(range(0, width - patch + 1, stride)) | {width - patch})
    candidates = []
    for top in tops:
        for left in lefts:
            n = int(integral[top + patch, left + patch] - integral[top, left + patch]
                    - integral[top + patch, left] + integral[top, left])
            if n >= min_pixels:
                candidates.append((n, top, left))
    candidates.sort(reverse=True)
    chosen: list[tuple[int, int, int]] = []
    for n, top, left in candidates:
        if all(abs(top - t) >= patch // 2 or abs(left - l) >= patch // 2 for t, l, _ in chosen):
            chosen.append((top, left, n))
        if len(chosen) == k:
            break
    return chosen


def mine_image(probability, gt, thresholds, areas, rules, patch, stride, dilation, min_pixels, k):
    """Return pool entries (without image name) for one image."""
    kept = apply_operating_points(probability, thresholds, areas)
    gt = gt > 0
    entries = []
    for channel, sources in rules.items():
        fp = kept[channel] & ~gt[channel] & source_region(gt, sources, dilation)
        for top, left, n in select_windows(fp, patch, stride, min_pixels, k):
            entries.append({"channel": UNIFIED_DAMAGE_CLASSES[channel], "sources": sources,
                            "top": int(top), "left": int(left), "fp_pixels": int(n)})
    return entries


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    cfg = load_config(run_dir / "config.yaml")
    seed_everything(cfg.project.seed)
    patch_cfg = multilabel_patch_cfg(cfg)
    patch = int(patch_cfg.patch_size)
    stride = int(args.stride or patch // 2)
    rules = parse_rules(args.rules)

    with open(args.operating_points, encoding="utf-8") as fh:
        points = json.load(fh)["frozen_fold0"][args.mode]
    thresholds = [float(points[n]["threshold"]) for n in UNIFIED_DAMAGE_CLASSES]
    areas = [float(points[n]["min_area_fraction"]) for n in UNIFIED_DAMAGE_CLASSES]

    step17 = load_step17()
    predict, checkpoint_path, checkpoint, family = step17.build_predictor(cfg, run_dir, args.checkpoint, get_device())
    split = args.split or str(cfg.data.dacl10k.train_split)
    if split == str(cfg.data.dacl10k.val_split):
        raise ValueError("Mining on the validation split would leak into model selection.")
    samples = list_samples(cfg.data.dacl10k.root, split)

    # Images without any source GT are skipped after loading the mask, before inference.
    needs_background = any("background" in s for s in rules.values())
    source_channels = sorted({UNIFIED_DAMAGE_CLASSES.index(s) for v in rules.values() for s in v if s != "background"})
    eligible = list(samples)
    if args.max_images:
        rng = np.random.default_rng(int(cfg.project.seed))
        eligible = [eligible[i] for i in sorted(rng.choice(len(eligible), min(args.max_images, len(eligible)), replace=False))]
    if args.limit:
        eligible = eligible[: args.limit]
    print(f"[hnm] {family} | {checkpoint_path.name} (epoch {checkpoint.get('epoch')}) | {split}: "
          f"{len(eligible)} images | rules {{{', '.join(f'{UNIFIED_DAMAGE_CLASSES[c]}<-{s}' for c, s in rules.items())}}}")

    items, n_skipped = [], 0
    for index, sample in enumerate(eligible, start=1):
        image, mask = step17.load_rgb_and_mask(sample)
        if not needs_background and not (mask[source_channels] > 0).any():
            n_skipped += 1
            continue
        probability = predict(image, patch_cfg)
        for entry in mine_image(probability, mask, thresholds, areas, rules, patch, stride,
                                args.source_dilation, args.min_fp_pixels, args.max_per_image):
            entry["image"] = Path(sample[0]).name
            items.append(entry)
        if index % 50 == 0 or index == len(eligible):
            counts = Counter(e["channel"] for e in items)
            print(f"[hnm] {index}/{len(eligible)} images | pool {dict(counts)} | skipped (no source GT) {n_skipped}")

    # Per-channel cap: keep the strongest false positives.
    pool = []
    for name in UNIFIED_DAMAGE_CLASSES:
        channel_items = sorted((e for e in items if e["channel"] == name), key=lambda e: -e["fp_pixels"])
        pool.extend(channel_items[: args.max_per_channel])

    output = Path(args.output)
    ensure_dir(output.parent)
    payload = {
        "meta": {
            "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "mined_by_run": run_dir.name, "checkpoint": checkpoint_path.name,
            "checkpoint_epoch": checkpoint.get("epoch"), "split": split,
            "operating_points": str(args.operating_points), "mode": args.mode,
            "thresholds": dict(zip(UNIFIED_DAMAGE_CLASSES, thresholds)),
            "min_area_fractions": dict(zip(UNIFIED_DAMAGE_CLASSES, areas)),
            "rules": {UNIFIED_DAMAGE_CLASSES[c]: s for c, s in rules.items()},
            "patch_size": patch, "stride": stride, "source_dilation": args.source_dilation,
            "min_fp_pixels": args.min_fp_pixels, "max_per_image": args.max_per_image,
            "max_per_channel": args.max_per_channel, "n_images_scanned": len(eligible) - n_skipped,
            "n_items": len(pool), "items_per_channel": dict(Counter(e["channel"] for e in pool)),
        },
        "items": pool,
    }
    with open(output, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=1)
    print(f"[hnm] pool saved to {output.resolve()} | {payload['meta']['items_per_channel']}")


if __name__ == "__main__":
    main()
