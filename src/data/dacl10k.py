"""dacl10k: polygon annotations -> raster masks.

Official layout
---------------
    <root>/images/<split>/*.jpg
    <root>/annotations/<split>/*.json

Each JSON contains `imageName`, `imageWidth`, `imageHeight` and `shapes`, where
every shape has a `label` (one of the 19 classes) and `points`, a list of [x, y]
polygon vertices with the origin in the top-left corner.

Two products are exposed here:
* `rasterize_binary`  -> single crack channel (Crack + ACrack), i.e. the label
  space of CrackSeg9k. This is what makes the *cross-dataset* evaluation of the
  week-3 U-Net possible without training anything on dacl10k.
* `rasterize_multilabel` -> [C,H,W] stack of overlapping classes, ready for the
  multi-label head of the dual-branch network (weeks 5+).
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from torch.utils.data import Dataset

from .class_mapping import CRACK_LIKE_DACL10K, DACL10K_CLASS_TO_IDX, DACL10K_CLASSES


# --------------------------------------------------------------------------- #
# Indexing
# --------------------------------------------------------------------------- #
def list_samples(root: str | Path, split: str) -> list[tuple[Path, Path]]:
    """Return [(image_path, annotation_path)] for a dacl10k split."""
    root = Path(root)
    images_dir, ann_dir = root / "images" / split, root / "annotations" / split
    if not images_dir.is_dir() or not ann_dir.is_dir():
        raise FileNotFoundError(f"Expected {images_dir} and {ann_dir}")

    samples, missing = [], []
    for img in sorted(images_dir.glob("*.jpg")):
        ann = ann_dir / f"{img.stem}.json"
        (samples.append((img, ann)) if ann.exists() else missing.append(img.name))
    if missing:
        print(f"[dacl10k] WARNING: {len(missing)} images without annotation, e.g. {missing[:5]}")
    if not samples:
        raise RuntimeError(f"No sample found in {images_dir}")
    return samples


def load_annotation(path: str | Path) -> dict:
    """Read one dacl10k JSON annotation."""
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)

def list_images(root: str | Path, split: str) -> list[Path]:
    """List image files for any DACL10K split, including unlabeled testdev."""
    images_dir = Path(root) / "images" / split
    if not images_dir.is_dir():
        raise FileNotFoundError(
            f"Expected image directory: {images_dir}"
        )

    images = sorted(images_dir.glob("*.jpg"))
    if not images:
        raise RuntimeError(
            f"No .jpg images found in {images_dir}"
        )

    return images

# --------------------------------------------------------------------------- #
# Rasterisation
# --------------------------------------------------------------------------- #
def _fill(mask: np.ndarray, points: list[list[float]]) -> None:
    """Fill one polygon in-place (rounded to integer pixel coordinates)."""
    if len(points) < 3:  # degenerate annotation: nothing to fill
        return
    polygon = np.round(np.asarray(points, dtype=np.float64)).astype(np.int32)
    cv2.fillPoly(mask, [polygon.reshape(-1, 1, 2)], color=1)


def rasterize_binary(
    annotation: dict,
    labels: list[str] | None = None,
    shape: tuple[int, int] | None = None,
) -> np.ndarray:
    """Rasterise the union of `labels` into a single [H,W] uint8 mask.

    Defaults to the crack-like classes (Crack + ACrack).
    """
    labels = labels or CRACK_LIKE_DACL10K
    height, width = shape or (int(annotation["imageHeight"]), int(annotation["imageWidth"]))
    mask = np.zeros((height, width), dtype=np.uint8)
    wanted = set(labels)
    for shape_dict in annotation.get("shapes", []):
        if shape_dict.get("label") in wanted:
            _fill(mask, shape_dict.get("points", []))
    return mask


def rasterize_multilabel(
    annotation: dict,
    classes: list[str] | None = None,
    shape: tuple[int, int] | None = None,
) -> np.ndarray:
    """Rasterise all classes into a [C,H,W] uint8 stack (channels may overlap)."""
    classes = classes or DACL10K_CLASSES
    class_to_channel = {c: i for i, c in enumerate(classes)}
    height, width = shape or (int(annotation["imageHeight"]), int(annotation["imageWidth"]))
    masks = np.zeros((len(classes), height, width), dtype=np.uint8)
    for shape_dict in annotation.get("shapes", []):
        channel = class_to_channel.get(shape_dict.get("label"))
        if channel is not None:
            _fill(masks[channel], shape_dict.get("points", []))
    return masks


def present_labels(annotation: dict) -> set[str]:
    """Set of class names annotated in one image (for the EDA class histogram)."""
    return {
        s["label"] for s in annotation.get("shapes", [])
        if s.get("label") in DACL10K_CLASS_TO_IDX
    }


# --------------------------------------------------------------------------- #
# Dataset (binary crack view, used as external validation of the week-3 U-Net)
# --------------------------------------------------------------------------- #
class Dacl10kCrackDataset(Dataset):
    """Yields (image [3,H,W], crack mask [1,H,W]) from dacl10k polygons."""

    def __init__(self, samples: list[tuple[Path, Path]], transform, labels: list[str] | None = None) -> None:
        self.samples = samples
        self.transform = transform
        self.labels = labels or CRACK_LIKE_DACL10K

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        image_path, ann_path = self.samples[index]

        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Unreadable image: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        annotation = load_annotation(ann_path)
        # Rasterise at the *actual* image size: some JSON headers disagree with
        # the decoded image by a pixel or two.
        mask = rasterize_binary(annotation, self.labels, shape=image.shape[:2]).astype(np.float32)

        augmented = self.transform(image=image, mask=mask)
        return augmented["image"], augmented["mask"].unsqueeze(0).float()
