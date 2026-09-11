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
# P1 helpers: binary crack target and image-level balancing
# --------------------------------------------------------------------------- #
def sample_has_positive_mask(
    annotation_path: str | Path,
    labels: list[str] | None = None,
    ) -> bool:
    """Return True when at least one non-degenerate target polygon is present."""
    labels = set(labels or CRACK_LIKE_DACL10K)
    annotation = load_annotation(annotation_path)

    for shape_dict in annotation.get("shapes", []):
        if shape_dict.get("label") in labels and len(shape_dict.get("points", [])) >= 3:
            return True
    return False


def binary_sample_targets(
    samples: list[tuple[Path, Path]],
    labels: list[str] | None = None,
    ) -> list[int]:
    """Return image-level targets: 1 if Crack/ACrack is present, else 0."""
    return [int(sample_has_positive_mask(annotation_path, labels)) for _, annotation_path in samples]


def summarize_binary_targets(targets: list[int]) -> dict[str, float | int]:
    """Summarize the crack-positive/negative composition of a DACL10K split."""
    n_total = len(targets)
    n_positive = int(sum(targets))
    n_negative = n_total - n_positive

    if n_total == 0:
        raise ValueError("Cannot summarize an empty DACL10K split.")

    return {
        "n_images": n_total,
        "n_positive": n_positive,
        "n_negative": n_negative,
        "positive_fraction": n_positive / n_total,
        "negative_fraction": n_negative / n_total,
    }

# --------------------------------------------------------------------------- #
# P1-Patch Version
# --------------------------------------------------------------------------- #

def _pad_to_minimum_size(
    image: np.ndarray,
    mask: np.ndarray,
    patch_size: int,
    ) -> tuple[np.ndarray, np.ndarray]:
    """Pad aligned image/mask so a square native-resolution crop is always valid."""
    height, width = image.shape[:2]
    pad_bottom = max(patch_size - height, 0)
    pad_right = max(patch_size - width, 0)

    if pad_bottom == 0 and pad_right == 0:
        return image, mask

    image = cv2.copyMakeBorder(
        image,
        top=0,
        bottom=pad_bottom,
        left=0,
        right=pad_right,
        borderType=cv2.BORDER_REFLECT_101,
    )
    mask = cv2.copyMakeBorder(
        mask,
        top=0,
        bottom=pad_bottom,
        left=0,
        right=pad_right,
        borderType=cv2.BORDER_CONSTANT,
        value=0,
    )
    return image, mask


def _crop_at(
    image: np.ndarray,
    mask: np.ndarray,
    top: int,
    left: int,
    patch_size: int,
    ) -> tuple[np.ndarray, np.ndarray]:
    """Extract geometrically aligned RGB and binary-mask patches."""
    return (
        image[top : top + patch_size, left : left + patch_size],
        mask[top : top + patch_size, left : left + patch_size],
    )


class Dacl10kCrackPatchDataset(Dataset):
    """Native-resolution training patches with guaranteed positive/negative ratio.

    An index does not identify one fixed image. Instead, it identifies a desired
    patch type:

    - indices in the first positive_patch_count positions always return a patch
      with at least min_positive_pixels Crack/ACrack pixels;
    - remaining indices always return a patch with at most max_negative_pixels
      Crack/ACrack pixels.

    DataLoader(shuffle=True) randomizes the order of these pre-defined patch
    types. Random image selection, crop coordinates and augmentation still make
    samples different across accesses and epochs.
    """

    def __init__(
        self,
        samples: list[tuple[Path, Path]],
        transform,
        patch_size: int,
        positive_patch_fraction: float = 0.70,
        min_positive_pixels: int = 64,
        max_negative_pixels: int = 0,
        max_crop_attempts: int = 30,
        patches_per_image: int = 4,
        labels: list[str] | None = None,
    ) -> None:
        if patch_size <= 0:
            raise ValueError("patch_size must be positive.")
        if not 0.0 < positive_patch_fraction < 1.0:
            raise ValueError("positive_patch_fraction must be strictly in (0, 1).")
        if min_positive_pixels < 1:
            raise ValueError("min_positive_pixels must be >= 1.")
        if max_negative_pixels < 0:
            raise ValueError("max_negative_pixels must be >= 0.")
        if max_crop_attempts < 1:
            raise ValueError("max_crop_attempts must be >= 1.")
        if patches_per_image < 1:
            raise ValueError("patches_per_image must be >= 1.")

        self.samples = samples
        self.transform = transform
        self.patch_size = int(patch_size)
        self.positive_patch_fraction = float(positive_patch_fraction)
        self.min_positive_pixels = int(min_positive_pixels)
        self.max_negative_pixels = int(max_negative_pixels)
        self.max_crop_attempts = int(max_crop_attempts)
        self.patches_per_image = int(patches_per_image)
        self.labels = labels or CRACK_LIKE_DACL10K

        self.positive_sample_indices = self._find_positive_sample_indices()
        self.negative_sample_indices = [
            index for index in range(len(self.samples))
            if index not in set(self.positive_sample_indices)
        ]

        if not self.positive_sample_indices:
            raise RuntimeError(
                "No DACL10K training image can produce a positive patch with at least "
                f"{self.min_positive_pixels} Crack/ACrack pixels."
            )
        if not self.negative_sample_indices:
            raise RuntimeError("No DACL10K training image is Crack/ACrack-negative.")

        self.n_patches = len(self.samples) * self.patches_per_image
        self.n_positive_patches = round(self.n_patches * self.positive_patch_fraction)
        self.n_negative_patches = self.n_patches - self.n_positive_patches

    def _find_positive_sample_indices(self) -> list[int]:
        """Return images that can produce a valid positive patch.

        A source image is eligible only if its full-resolution binary mask contains
        at least min_positive_pixels Crack/ACrack pixels. This guarantees that a
        512 × 512 crop can satisfy the same lower bound.
        """
        positive_indices = []

        for index, (image_path, annotation_path) in enumerate(self.samples):
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                raise RuntimeError(f"Unreadable image while indexing patches: {image_path}")

            annotation = load_annotation(annotation_path)
            mask = rasterize_binary(
                annotation,
                labels=self.labels,
                shape=image.shape[:2],
            )

            if int(mask.sum()) >= self.min_positive_pixels:
                positive_indices.append(index)

        return positive_indices

    def __len__(self) -> int:
        return self.n_patches

    def _load_sample(self, sample_index: int) -> tuple[np.ndarray, np.ndarray]:
        image_path, annotation_path = self.samples[sample_index]

        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Unreadable image: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        annotation = load_annotation(annotation_path)
        mask = rasterize_binary(
            annotation,
            labels=self.labels,
            shape=image.shape[:2],
        ).astype(np.float32)

        return _pad_to_minimum_size(image, mask, self.patch_size)

    def _random_crop(
        self,
        image: np.ndarray,
        mask: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        height, width = image.shape[:2]
        top = np.random.randint(0, height - self.patch_size + 1)
        left = np.random.randint(0, width - self.patch_size + 1)
        return _crop_at(image, mask, top, left, self.patch_size)

    def _positive_crop(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        ) -> tuple[np.ndarray, np.ndarray]:
        """Return a crop guaranteed to contain sufficient crack pixels."""
        for _ in range(self.max_crop_attempts):
            image_patch, mask_patch = self._random_crop(image, mask)
            if int(mask_patch.sum()) >= self.min_positive_pixels:
                return image_patch, mask_patch

        crack_y, crack_x = np.where(mask > 0)
        if len(crack_y) == 0:
            raise RuntimeError("Positive source image unexpectedly has an empty crack mask.")

        anchor_index = np.random.randint(len(crack_y))
        center_y = int(crack_y[anchor_index])
        center_x = int(crack_x[anchor_index])

        height, width = image.shape[:2]
        top_min = max(0, center_y - self.patch_size + 1)
        top_max = min(center_y, height - self.patch_size)
        left_min = max(0, center_x - self.patch_size + 1)
        left_max = min(center_x, width - self.patch_size)

        top = np.random.randint(top_min, top_max + 1)
        left = np.random.randint(left_min, left_max + 1)
        image_patch, mask_patch = _crop_at(image, mask, top, left, self.patch_size)

        if int(mask_patch.sum()) < self.min_positive_pixels:
            raise RuntimeError(
                "Eligible positive source did not yield a valid positive patch. "
                "Increase max_crop_attempts or inspect the crop-anchor logic."
            )
        return image_patch, mask_patch

    def _negative_crop(
        self,
        image: np.ndarray,
        mask: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return a crop with no target crack pixels, whenever possible."""
        for _ in range(self.max_crop_attempts):
            image_patch, mask_patch = self._random_crop(image, mask)
            if int(mask_patch.sum()) <= self.max_negative_pixels:
                return image_patch, mask_patch

        if int(mask.sum()) <= self.max_negative_pixels:
            return self._random_crop(image, mask)

        raise RuntimeError(
            "Could not construct a negative patch from the selected image. "
            "Increase max_crop_attempts or use crack-negative source images."
        )

    def _get_positive_patch(self) -> tuple[np.ndarray, np.ndarray]:
        sample_index = int(np.random.choice(self.positive_sample_indices))
        image, mask = self._load_sample(sample_index)
        return self._positive_crop(image, mask)

    def _get_negative_patch(self) -> tuple[np.ndarray, np.ndarray]:
        sample_index = int(np.random.choice(self.negative_sample_indices))
        image, mask = self._load_sample(sample_index)
        return self._negative_crop(image, mask)

    def __getitem__(self, index: int):
        wants_positive = index < self.n_positive_patches

        if wants_positive:
            image_patch, mask_patch = self._get_positive_patch()
        else:
            image_patch, mask_patch = self._get_negative_patch()

        augmented = self.transform(image=image_patch, mask=mask_patch)
        return augmented["image"], augmented["mask"].unsqueeze(0).float()

class Dacl10kCrackCenterPatchDataset(Dataset):
    """One deterministic center patch per image for training-time monitoring."""

    def __init__(
        self,
        samples: list[tuple[Path, Path]],
        transform,
        patch_size: int,
        labels: list[str] | None = None,
    ) -> None:
        self.samples = samples
        self.transform = transform
        self.patch_size = int(patch_size)
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
        mask = rasterize_binary(
            annotation,
            labels=self.labels,
            shape=image.shape[:2],
        ).astype(np.float32)

        image, mask = _pad_to_minimum_size(image, mask, self.patch_size)
        height, width = image.shape[:2]
        top = (height - self.patch_size) // 2
        left = (width - self.patch_size) // 2

        image_patch, mask_patch = _crop_at(image, mask, top, left, self.patch_size)
        augmented = self.transform(image=image_patch, mask=mask_patch)
        return augmented["image"], augmented["mask"].unsqueeze(0).float()
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
