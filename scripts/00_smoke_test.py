#!/usr/bin/env python
"""Step 0 - end-to-end smoke test on synthetic data (no real dataset needed).

Generates a tiny fake CrackSeg9k and a tiny fake dacl10k under a temp folder,
then runs EDA, one training epoch and the evaluation. Purpose: catch shape,
dtype and path bugs in seconds, on CPU, before spending Colab GPU time.

Usage
-----
    python scripts/00_smoke_test.py
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))


def make_fake_crackseg9k(root: Path, n: int = 12, size: int = 128) -> None:
    """Images with a bright random line + the matching binary mask."""
    images_dir, masks_dir = root / "images", root / "masks"
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)

    for i in range(n):
        image = rng.integers(60, 180, size=(size, size, 3), dtype=np.uint8)
        mask = np.zeros((size, size), dtype=np.uint8)
        p1 = tuple(rng.integers(0, size, 2).tolist())
        p2 = tuple(rng.integers(0, size, 2).tolist())
        cv2.line(image, p1, p2, (240, 240, 240), 2)
        cv2.line(mask, p1, p2, 255, 2)
        cv2.imwrite(str(images_dir / f"img_{i:03d}.jpg"), image)
        cv2.imwrite(str(masks_dir / f"img_{i:03d}.png"), mask)


def make_fake_dacl10k(root: Path, n: int = 6, size: int = 128) -> None:
    """Official layout with polygon annotations for two classes."""
    for split in ("train", "validation"):
        images_dir = root / "images" / split
        ann_dir = root / "annotations" / split
        images_dir.mkdir(parents=True, exist_ok=True)
        ann_dir.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(1)

        for i in range(n):
            name = f"dacl_{i:03d}"
            cv2.imwrite(
                str(images_dir / f"{name}.jpg"),
                rng.integers(50, 200, size=(size, size, 3), dtype=np.uint8),
            )
            annotation = {
                "imageName": f"{name}.jpg",
                "imageWidth": size,
                "imageHeight": size,
                "split": split,
                "shapes": [
                    {"label": "Crack", "shape_type": "polygon",
                     "points": [[10, 10], [60, 12], [62, 20], [12, 18]]},
                    {"label": "Rust", "shape_type": "polygon",
                     "points": [[70, 70], [110, 72], [108, 110], [72, 108]]},
                ],
            }
            (ann_dir / f"{name}.json").write_text(json.dumps(annotation), encoding="utf-8")


def build_config(tmp: Path) -> Path:
    """Copy the real config and point it at the synthetic data with tiny settings."""
    cfg = yaml.safe_load((ROOT / "configs" / "config.yaml").read_text(encoding="utf-8"))
    cfg["project"]["output_dir"] = str(tmp / "outputs")
    cfg["data"]["crackseg9k"]["images_dir"] = str(tmp / "crackseg9k" / "images")
    cfg["data"]["crackseg9k"]["masks_dir"] = str(tmp / "crackseg9k" / "masks")
    cfg["data"]["dacl10k"]["root"] = str(tmp / "dacl10k")
    cfg["data"]["image_size"] = 64
    cfg["data"]["num_workers"] = 0
    cfg["model"]["encoder_weights"] = None  # no download in the smoke test
    cfg["train"].update({"epochs": 1, "batch_size": 2, "amp": False, "resume": False})
    cfg["train"]["scheduler"]["t_0"] = 1
    cfg["eval"]["n_qualitative_samples"] = 2

    path = tmp / "smoke_config.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return path


def run(command: list[str]) -> None:
    """Run a step and fail loudly."""
    print(f"\n$ {' '.join(command)}")
    subprocess.run(command, check=True, cwd=ROOT)


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp_name:
        tmp = Path(tmp_name)
        make_fake_crackseg9k(tmp / "crackseg9k")
        make_fake_dacl10k(tmp / "dacl10k")
        config = build_config(tmp)
        run_dir = tmp / "outputs" / "runs" / "smoke"

        run([sys.executable, "scripts/01_eda.py", "--config", str(config)])
        run([sys.executable, "scripts/02_train_unet.py", "--config", str(config), "--run-name", "smoke"])
        run([sys.executable, "scripts/03_evaluate.py", "--run-dir", str(run_dir), "--cross-dataset"])

        print("\nSMOKE TEST PASSED - pipeline is consistent end to end.")


if __name__ == "__main__":
    main()
