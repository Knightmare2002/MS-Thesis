#!/usr/bin/env python
"""DACL10K relative-resolution control: downsample RGB, rerun frozen detector, align to native GT.

Never resize predicted probabilities as a substitute for rerunning the model.
Output counts/recall/FPR by image and scale; scale-1 supplies optional anchor rates.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from damage3d.classes import DAMAGE_CLASSES
from damage3d.experiment_data import write_json
from damage3d.run_state import sha256_file


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--split", default="validation")
    ap.add_argument("--checkpoint", default="best.pt")
    ap.add_argument("--scales", nargs="+", type=int, default=list(range(1, 9)))
    ap.add_argument("--limit", type=int)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    if args.out.exists():
        raise FileExistsError(args.out)
    if 1 not in args.scales or any(s < 1 for s in args.scales) or len(set(args.scales)) != len(args.scales):
        raise ValueError("Scales must be distinct positive integers including 1.")
    import cv2
    import torch
    from src.data.class_mapping import UNIFIED_DAMAGE_CLASSES
    from src.data.dacl10k import list_samples, load_annotation, rasterize_unified_damage
    from src.data.transforms import IMAGENET_MEAN, IMAGENET_STD
    from src.engine import load_checkpoint
    from src.eval.sliding_window import predict_sliding_window_multilabel
    from src.eval.probability_calibration import load_platt, apply_platt
    from src.models.unet import build_model
    from src.utils import get_device, load_config, seed_everything
    seed_everything(0)
    cfg = load_config(args.run_dir / "config.yaml")
    cal_path = args.run_dir / "eval_curves/probability_calibration.json"
    a, b, cal = load_platt(cal_path, DAMAGE_CLASSES)
    checkpoint = args.run_dir / args.checkpoint
    if cal.get("checkpoint", {}).get("sha256") != sha256_file(checkpoint):
        raise ValueError("Checkpoint must match frozen Platt calibration.")
    device = get_device()
    mc = dict(cfg["model"])
    mc["encoder_weights"] = None
    model = build_model(type(cfg)(mc)).to(device)
    load_checkpoint(checkpoint, model, device=device)
    model.eval()
    patch = cfg.data.p1ml_patch
    channels = [list(UNIFIED_DAMAGE_CLASSES).index(c) for c in DAMAGE_CLASSES]
    thresholds = np.array([cal["thresholds_calibrated"][c] for c in DAMAGE_CLASSES])[:, None, None]
    samples = list_samples(args.data_root, args.split)
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be positive.")
    if args.limit:
        samples = samples[:args.limit]
    if not samples:
        raise ValueError("No DACL10K samples.")
    rows = []
    for image_path, annotation in samples:
        bgr = cv2.imread(str(image_path))
        if bgr is None:
            raise ValueError(f"Unreadable {image_path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        gt = rasterize_unified_damage(load_annotation(annotation), shape=rgb.shape[:2])[channels] > 0
        h, w = rgb.shape[:2]
        for scale in args.scales:
            small = rgb if scale == 1 else cv2.resize(rgb, (max(1, w//scale), max(1, h//scale)),
                                                     interpolation=cv2.INTER_AREA)
            with torch.inference_mode():
                pred = predict_sliding_window_multilabel(
                    model=model, image=small, device=device, patch_size=int(patch.patch_size),
                    stride=int(patch.eval_stride),
                    batch_size=int(patch.eval_batch_size), mean=IMAGENET_MEAN, std=IMAGENET_STD,
                    n_classes=len(UNIFIED_DAMAGE_CLASSES), blend_mode=str(patch.blend_mode),
                    tta=patch.get("tta"))
            raw = pred.detach().cpu().numpy()[channels]
            # Calibrate low-resolution predictions, then restore native grid for fixed GT evaluation.
            prob = apply_platt(raw, a, b)
            aligned = np.stack([cv2.resize(p, (w, h), interpolation=cv2.INTER_LINEAR) for p in prob])
            hit = aligned >= thresholds
            rows.append({"image": str(image_path), "annotation_sha256": sha256_file(annotation),
                         "image_sha256": sha256_file(image_path), "scale": scale,
                         "actual_xy_scale": [w/small.shape[1], h/small.shape[0]],
                         "tp": (hit & gt).sum(axis=(1, 2)).tolist(),
                         "fp": (hit & ~gt).sum(axis=(1, 2)).tolist(),
                         "positive": gt.sum(axis=(1, 2)).tolist(),
                         "negative": (~gt).sum(axis=(1, 2)).tolist()})
        print(f"{len(rows)//len(args.scales)}/{len(samples)} {image_path.name}", flush=True)
    aggregate = {}
    for scale in args.scales:
        sel = [r for r in rows if r["scale"] == scale]
        counts = {k: np.array([r[k] for r in sel]).sum(axis=0) for k in ("tp", "fp", "positive", "negative")}
        aggregate[str(scale)] = {
            c: {k: int(counts[k][j]) for k in counts} |
               {"recall": float(counts["tp"][j]/counts["positive"][j]) if counts["positive"][j] else None,
                "fpr": float(counts["fp"][j]/counts["negative"][j]) if counts["negative"][j] else None}
            for j, c in enumerate(DAMAGE_CLASSES)}
    write_json(args.out, {"classes": list(DAMAGE_CLASSES), "split": args.split,
                         "checkpoint_sha256": sha256_file(checkpoint), "calibration_sha256": sha256_file(cal_path),
                         "thresholds": thresholds[:, 0, 0].tolist(), "config_sha256": sha256_file(args.run_dir/"config.yaml"),
                         "aggregate": aggregate, "per_image": rows,
                         "note": "Relative image degradation control, NOT physical bridge GSD or latent POD ground truth."})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
