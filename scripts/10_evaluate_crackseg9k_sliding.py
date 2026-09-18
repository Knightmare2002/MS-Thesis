"""Evaluate a P0 checkpoint on CrackSeg9k through two geometric paths.

--mode resize : reproduces the training-time evaluation (A.Resize to 512).
--mode sliding: reproduces the path used by 08 (no global resize, reflect-pad,
                gaussian-blended sliding window at native resolution).

Comparing the two isolates a scale artefact from an evaluation-path bug.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.crackseg9k import list_pairs, split_pairs
from src.data.transforms import IMAGENET_MEAN, IMAGENET_STD
from src.engine import load_checkpoint
from src.eval.metrics import SegmentationMetrics
from src.eval.sliding_window import predict_sliding_window
from src.models.unet import build_model
from src.utils import ensure_dir, get_device, load_config, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--checkpoint", default="best.pt")
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--mode", default="sliding", choices=["sliding", "resize"])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--split-json", default=None, help="frozen split.json of the reference run")
    parser.add_argument("--set", nargs="*", default=[], help="config overrides key=value")
    return parser.parse_args()


def load_pair(image_path: Path, mask_path: Path, mask_threshold: int):
    """Load native-resolution RGB image and binary mask, matching the dataset class."""
    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise RuntimeError(f"Unreadable image: {image_path}")
    image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise RuntimeError(f"Unreadable mask: {mask_path}")

    return image, (mask > mask_threshold).astype(np.float32)


@torch.no_grad()
def predict_resize(model, image, device, image_size: int) -> torch.Tensor:
    """Single forward pass on the globally resized image, as during training."""
    resized = cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
    tensor = torch.from_numpy(resized.transpose(2, 0, 1)).float() / 255.0
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    tensor = ((tensor - mean) / std).unsqueeze(0).to(device)

    return torch.sigmoid(model(tensor))[0, 0].cpu()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    cfg = load_config(run_dir / "config.yaml")
    seed_everything(cfg.project.seed)
    device = get_device()

    model = build_model(cfg.model).to(device)
    checkpoint = load_checkpoint(run_dir / args.checkpoint, model, device=device)
    print(
        f"[eval] checkpoint {args.checkpoint} | epoch {checkpoint.get('epoch')} | "
        f"best_dice {checkpoint.get('best_dice')}"
    )

    pairs = list_pairs(cfg.data.crackseg9k.images_dir, cfg.data.crackseg9k.masks_dir)
    if args.split_json:
        splits = load_frozen_split(args.split_json, pairs)
    else:
        splits = split_pairs(
            pairs,
            val_fraction=cfg.data.crackseg9k.val_fraction,
            test_fraction=cfg.data.crackseg9k.test_fraction,
            seed=cfg.project.seed,
        )
    items = splits[args.split]

    mask_threshold = int(cfg.data.crackseg9k.mask_threshold)
    image_size = int(cfg.data.image_size)
    thresholds = list(cfg.eval.threshold_sweep)
    meters = {t: SegmentationMetrics(threshold=t) for t in thresholds}

    print(f"[eval] CrackSeg9k {args.split}: {len(items)} images | mode={args.mode}")

    for index, (image_path, mask_path) in enumerate(items, start=1):
        image, mask = load_pair(image_path, mask_path, mask_threshold)

        if args.mode == "sliding":
            probability = predict_sliding_window(
                model=model,
                image=image,
                device=device,
                patch_size=int(cfg.data.p1_patch.patch_size),
                stride=int(cfg.data.p1_patch.eval_stride),
                batch_size=int(cfg.data.p1_patch.eval_batch_size),
                mean=IMAGENET_MEAN,
                std=IMAGENET_STD,
                blend_mode=str(cfg.data.p1_patch.blend_mode),
            )
            target = torch.from_numpy(mask)
        else:
            probability = predict_resize(model, image, device, image_size)
            # Nearest interpolation on the mask, as in A.Resize.
            target = torch.from_numpy(
                cv2.resize(
                    mask, (image_size, image_size), interpolation=cv2.INTER_NEAREST
                )
            )

        logits = torch.logit(probability.clamp(1e-6, 1.0 - 1e-6))
        for meter in meters.values():
            meter.update(
                logits.unsqueeze(0).unsqueeze(0),
                target.unsqueeze(0).unsqueeze(0),
            )

        if index % 100 == 0 or index == len(items):
            print(f"[eval] processed {index}/{len(items)} images")

    results = pd.DataFrame(
        [{"threshold": t, **meter.compute()} for t, meter in meters.items()]
    ).sort_values("threshold").reset_index(drop=True)

    results.insert(0, "dataset", f"crackseg9k_{args.split}_{args.mode}")
    results.insert(1, "mode", args.mode)
    results.insert(2, "checkpoint_name", args.checkpoint)
    results.insert(3, "checkpoint_epoch", checkpoint.get("epoch", "unknown"))

    eval_dir = ensure_dir(run_dir / "eval_crackseg9k")
    out_path = eval_dir / f"metrics_crackseg9k_{args.split}_{args.mode}.csv"
    results.to_csv(out_path, index=False)

    print(results[["threshold", "iou", "dice", "precision", "recall"]].to_string(index=False))
    print(f"[eval] saved to {out_path}")


if __name__ == "__main__":
    main()