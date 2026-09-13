#!/usr/bin/env python
"""Mine static hard-negative 512x512 patches from DACL10K train images."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm.auto import tqdm

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.data.dacl10k import list_samples, load_annotation, rasterize_binary
from src.models.unet import build_model
from src.utils import ensure_dir, get_device, load_config, seed_everything
from src.data.transforms import IMAGENET_MEAN, IMAGENET_STD


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mine hard-negative training patches.")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--set", nargs="*", default=[], help="config overrides key=value")
    parser.add_argument("--limit", type=int, default=None, help="mine only the first N train images")
    return parser.parse_args()


def positions(length: int, patch_size: int, stride: int) -> list[int]:
    if length <= patch_size:
        return [0]
    values = list(range(0, length - patch_size + 1, stride))
    last = length - patch_size
    if values[-1] != last:
        values.append(last)
    return values


def deduplicate_windows(
    candidates: list[dict],
    patch_size: int,
    max_overlap: float,
) -> list[dict]:
    """Greedy suppression of overlapping windows.

    Two windows overlapping by 50% are essentially the same hard negative;
    keeping both would concentrate the pool on a single image region.
    """
    kept: list[dict] = []
    area = patch_size * patch_size

    for item in candidates:
        overlaps = any(
            max(0, patch_size - abs(item["x"] - other["x"]))
            * max(0, patch_size - abs(item["y"] - other["y"]))
            > max_overlap * area
            for other in kept
        )
        if not overlaps:
            kept.append(item)

    return kept


def normalize_patch(patch_rgb: np.ndarray) -> torch.Tensor:
    """ImageNet normalisation only: the patch is already native patch_size."""
    image = patch_rgb.astype(np.float32) / 255.0
    image = (image - np.array(IMAGENET_MEAN, dtype=np.float32)) / np.array(IMAGENET_STD, dtype=np.float32)
    return torch.from_numpy(image.transpose(2, 0, 1)).float()


@torch.no_grad()
def predict_patch_probabilities(
    model: torch.nn.Module,
    patches: list[np.ndarray],
    device: torch.device,
    batch_size: int,
) -> list[np.ndarray]:
    """Batched inference over all candidate windows of one image."""
    probabilities: list[np.ndarray] = []
    for start in range(0, len(patches), batch_size):
        block = patches[start:start + batch_size]
        tensor = torch.stack([normalize_patch(patch) for patch in block]).to(device)
        output = torch.sigmoid(model(tensor))[:, 0].float().cpu().numpy()
        probabilities.extend(output[index] for index in range(len(block)))
    return probabilities


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config, args.set)
    seed_everything(cfg.project.seed)

    hnm = cfg.data.hard_negative_mining
    patch_cfg = cfg.data.p1_patch
    device = get_device()

    checkpoint_path = Path(args.checkpoint or hnm.source_checkpoint)
    output_path = Path(args.output or hnm.pool_path)

    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    if not bool(hnm.enabled):
        raise ValueError("data.hard_negative_mining.enabled must be true to run this script.")

    patch_size = int(patch_cfg.patch_size)
    stride = int(hnm.candidate_stride)
    image_size = int(cfg.data.image_size)
    top_k = int(hnm.top_k_per_image)

    probability_threshold = float(hnm.probability_threshold)
    min_predicted_fraction = float(hnm.min_predicted_fraction)
    max_window_overlap = float(hnm.max_window_overlap)

    crack_labels = list(cfg.data.dacl10k.crack_labels)
    negative_images_only = bool(hnm.negative_images_only)

    if not 0.0 <= probability_threshold <= 1.0:
        raise ValueError("probability_threshold must be in [0, 1].")

    if not 0.0 <= min_predicted_fraction <= 1.0:
        raise ValueError("min_predicted_fraction must be in [0, 1].")

    if not 0.0 <= max_window_overlap < 1.0:
        raise ValueError("max_window_overlap must be in [0, 1).")

    model = build_model(cfg.model).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    samples = list_samples(cfg.data.dacl10k.root, cfg.data.dacl10k.train_split)
    if args.limit:
        samples = samples[: args.limit]
        print(f"[mine] DRY RUN on the first {len(samples)} train images")
    pool: list[dict] = []
    n_negative_images = 0
    n_candidates = 0

    for image_path, ann_path in tqdm(samples, desc="mining hard negatives"):
        image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise RuntimeError(f"Unreadable image: {image_path}")

        image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        annotation = load_annotation(ann_path)
        mask = rasterize_binary(annotation, labels=crack_labels, shape=image.shape[:2])

        if negative_images_only and int(mask.sum()) > 0:
            continue

        n_negative_images += 1
        height, width = image.shape[:2]
        candidates: list[dict] = []

        windows = []    
        for y in positions(height, patch_size, stride):
            for x in positions(width, patch_size, stride):
                patch_mask = mask[y:y + patch_size, x:x + patch_size]
                # Hard negatives must contain no target crack pixels at all.
                if int(patch_mask.sum()) > int(patch_cfg.max_negative_pixels):
                    continue
                windows.append((y, x, image[y:y + patch_size, x:x + patch_size], patch_mask))

        if not windows:
            continue

        probabilities = predict_patch_probabilities(
            model,
            [window[2] for window in windows],
            device,
            int(patch_cfg.eval_batch_size),
        )

        candidates: list[dict] = []
        for (y, x, patch_image, patch_mask), probability in zip(windows, probabilities):
            n_predicted = int((probability >= probability_threshold).sum())
            predicted_fraction = n_predicted / probability.size
            n_candidates += 1

            if predicted_fraction < min_predicted_fraction:
                continue

            candidates.append({
                "image_path": str(image_path),
                "annotation_path": str(ann_path),
                "x": int(x), "y": int(y),
                "width": int(patch_image.shape[1]), "height": int(patch_image.shape[0]),
                "predicted_fraction": predicted_fraction,
                "n_predicted_pixels": n_predicted,
                "mean_probability": float(probability.mean()),
                "gt_positive_pixels": int(patch_mask.sum()),
            })

        candidates.sort(
            key=lambda item: (item["predicted_fraction"], item["mean_probability"]),
            reverse=True,
        )
        pool.extend(deduplicate_windows(candidates, patch_size, max_window_overlap)[:top_k])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "metadata": {
            "dataset": "dacl10k",
            "split": str(cfg.data.dacl10k.train_split),
            "checkpoint": str(checkpoint_path),
            "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
            "patch_size": patch_size,
            "candidate_stride": stride,
            "top_k_per_image": top_k,
            "probability_threshold": probability_threshold,
            "min_predicted_fraction": min_predicted_fraction,
            "max_window_overlap": max_window_overlap,
            "negative_images_only": negative_images_only,
            "n_negative_images_considered": n_negative_images,
            "n_candidate_windows_evaluated": n_candidates,
            "n_hard_negative_patches": len(pool),
            "seed": int(hnm.seed),
        },
        "patches": pool,
    }

    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    summary_path = output_path.with_name(f"{output_path.stem}_summary.json")
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(payload["metadata"], handle, indent=2)

    print(f"[mine] checkpoint: {checkpoint_path}")
    print(f"[mine] negative train images considered: {n_negative_images}")
    print(f"[mine] candidate windows evaluated: {n_candidates}")
    print(f"[mine] hard-negative patches retained: {len(pool)}")
    print(f"[mine] pool saved: {output_path}")


if __name__ == "__main__":
    main()