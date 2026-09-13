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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mine hard-negative training patches.")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--set", nargs="*", default=[], help="config overrides key=value")
    return parser.parse_args()


def positions(length: int, patch_size: int, stride: int) -> list[int]:
    if length <= patch_size:
        return [0]
    values = list(range(0, length - patch_size + 1, stride))
    last = length - patch_size
    if values[-1] != last:
        values.append(last)
    return values


def preprocess(image_rgb: np.ndarray, image_size: int) -> torch.Tensor:
    image = cv2.resize(image_rgb, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
    image = image.astype(np.float32) / 255.0
    image = (image - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array(
        [0.229, 0.224, 0.225], dtype=np.float32
    )
    return torch.from_numpy(image.transpose(2, 0, 1)).float()


@torch.no_grad()
def predict_patch_probability(
    model: torch.nn.Module,
    patch_rgb: np.ndarray,
    image_size: int,
    device: torch.device,
) -> np.ndarray:
    tensor = preprocess(patch_rgb, image_size).unsqueeze(0).to(device)
    logits = model(tensor)
    probability = torch.sigmoid(logits)[0, 0].float().cpu().numpy()
    return cv2.resize(
        probability,
        (patch_rgb.shape[1], patch_rgb.shape[0]),
        interpolation=cv2.INTER_LINEAR,
    )


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
    min_mean_probability = float(hnm.min_mean_probability)
    probability_threshold = float(hnm.probability_threshold)
    crack_labels = list(cfg.data.dacl10k.crack_labels)
    negative_images_only = bool(hnm.negative_images_only)

    model = build_model(cfg.model).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    samples = list_samples(cfg.data.dacl10k.root, cfg.data.dacl10k.train_split)
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

        for y in positions(height, patch_size, stride):
            for x in positions(width, patch_size, stride):
                patch_mask = mask[y:y + patch_size, x:x + patch_size]

                # Safety check: hard negatives must contain no target crack pixels.
                if int(patch_mask.sum()) > int(patch_cfg.max_negative_pixels):
                    continue

                patch_image = image[y:y + patch_size, x:x + patch_size]
                probability = predict_patch_probability(model, patch_image, image_size, device)

                mean_probability = float(probability.mean())
                predicted_fraction = float((probability >= probability_threshold).mean())

                n_candidates += 1
                if mean_probability < min_mean_probability:
                    continue

                candidates.append(
                    {
                        "image_path": str(image_path),
                        "annotation_path": str(ann_path),
                        "x": int(x),
                        "y": int(y),
                        "width": int(patch_image.shape[1]),
                        "height": int(patch_image.shape[0]),
                        "mean_probability": mean_probability,
                        "predicted_fraction": predicted_fraction,
                        "gt_positive_pixels": int(patch_mask.sum()),
                    }
                )

        candidates.sort(
            key=lambda item: (item["mean_probability"], item["predicted_fraction"]),
            reverse=True,
        )
        pool.extend(candidates[:top_k])

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
            "min_mean_probability": min_mean_probability,
            "probability_threshold": probability_threshold,
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