"""dacl10k: polygon annotations -> raster masks.

Official layout
---------------
    <root>/images/<split>/*.jpg
    <root>/annotations/<split>/*.json

Each JSON contains `imageName`, `imageWidth`, `imageHeight` and `shapes`, where every shape has a `label` (one of the 19 classes) and `points`, a list of [x, y] polygon vertices with the origin in the top-left corner.

Two products are exposed here:
* `rasterize_binary`  -> single crack channel (Crack + ACrack), i.e. the label space of CrackSeg9k. This is what makes the *cross-dataset* evaluation of the week-3 U-Net possible without training anything on dacl10k.
* `rasterize_multilabel` -> [C,H,W] stack of overlapping classes, ready for the multi-label head of the dual-branch network (weeks 5+).
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from torch.utils.data import Dataset

from .class_mapping import (
    CRACK_LIKE_DACL10K,
    DACL10K_CLASS_TO_IDX,
    DACL10K_CLASSES,
    UNIFIED_DAMAGE_CLASSES,
    unified_damage_label_to_channel,
)


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
            hard_negative_pool_path: str | Path | None = None,
            hard_negative_fraction: float = 0.0,
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
            if not 0.0 <= hard_negative_fraction <= 1.0:
                raise ValueError("hard_negative_fraction must be in [0, 1].")

            self.samples = samples
            self.transform = transform
            self.patch_size = int(patch_size)
            self.positive_patch_fraction = float(positive_patch_fraction)
            self.min_positive_pixels = int(min_positive_pixels)
            self.max_negative_pixels = int(max_negative_pixels)
            self.max_crop_attempts = int(max_crop_attempts)
            self.patches_per_image = int(patches_per_image)
            self.labels = labels or CRACK_LIKE_DACL10K
            self.hard_negative_fraction = float(hard_negative_fraction)
            self.hard_negative_pool = self._load_hard_negative_pool(
                hard_negative_pool_path
            )
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
            self.n_hard_negative_patches = round(
            self.n_negative_patches * self.hard_negative_fraction
            )
            
            self.n_random_negative_patches = (
                self.n_negative_patches - self.n_hard_negative_patches
            )

            if self.n_hard_negative_patches > 0 and not self.hard_negative_pool:
                raise RuntimeError(
                    "hard_negative_fraction > 0, but the hard-negative pool is empty. "
                    "Run scripts/08_mine_hard_negatives.py or check pool_path."
                )
            
            print(
                "[patch-dataset] composition | "
                f"total={self.n_patches} | "
                f"positive={self.n_positive_patches} | "
                f"hard_negative={self.n_hard_negative_patches} | "
                f"random_negative={self.n_random_negative_patches}"
            )

    def _load_hard_negative_pool(
    self,
    hard_negative_pool_path: str | Path | None,
    ) -> list[dict]:
        """Load offline-mined zero-GT-crack patches for hard-negative sampling."""
        if hard_negative_pool_path is None:
            return []

        path = Path(hard_negative_pool_path)
        if not path.is_file():
            raise FileNotFoundError(f"Hard-negative pool not found: {path}")

        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)

        patches = payload.get("patches", [])
        valid_patches = []

        for item in patches:
            required = {"image_path", "annotation_path", "x", "y"}
            missing = required.difference(item)
            if missing:
                raise ValueError(
                    f"Malformed hard-negative entry in {path}: missing {sorted(missing)}"
                )

            gt_pixels = int(item.get("gt_positive_pixels", -1))
            if gt_pixels < 0:
                raise ValueError(
                    "Hard-negative entry has no gt_positive_pixels field. "
                    "Regenerate the pool with scripts/09_mine_hard_negatives.py."
                )
            if gt_pixels > self.max_negative_pixels:
                continue

            valid_patches.append(item)

        print(
            f"[patch-dataset] loaded {len(valid_patches)} hard-negative patches "
            f"from {path}"
        )
        return valid_patches

    def _find_positive_sample_indices(self) -> list[int]:
        """Return images whose full-resolution mask holds enough Crack/ACrack pixels.

        Sizes are taken from the JSON header, so no JPEG is decoded here: this turns a ~7k-image scan from minutes into seconds. Note that this is a necessary, not a sufficient, condition for a single 512x512 crop to reach min_positive_pixels: sparse annotations are handled by _positive_crop.
        """
        positive_indices = []

        for index, (_, annotation_path) in enumerate(self.samples):
            annotation = load_annotation(annotation_path)
            mask = rasterize_binary(annotation, labels=self.labels)

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
        """Return a crack-anchored crop; falls back to the richest one found.

        Anchoring on a random crack pixel is far more sample-efficient than
        uniform cropping, because crack pixels cover well below 1% of a bridge
        photo. The quota is statistical, so a pathological image degrades the
        patch instead of aborting a multi-hour run.
        """
        crack_y, crack_x = np.where(mask > 0)
        if len(crack_y) == 0:
            raise RuntimeError("Positive source image unexpectedly has an empty crack mask.")

        height, width = image.shape[:2]
        best_patch, best_pixels = None, -1

        for _ in range(self.max_crop_attempts):
            anchor_index = np.random.randint(len(crack_y))
            center_y, center_x = int(crack_y[anchor_index]), int(crack_x[anchor_index])

            top = np.random.randint(
                max(0, center_y - self.patch_size + 1),
                min(center_y, height - self.patch_size) + 1,
            )
            left = np.random.randint(
                max(0, center_x - self.patch_size + 1),
                min(center_x, width - self.patch_size) + 1,
            )

            image_patch, mask_patch = _crop_at(image, mask, top, left, self.patch_size)
            n_positive = int(mask_patch.sum())
            if n_positive >= self.min_positive_pixels:
                return image_patch, mask_patch
            if n_positive > best_pixels:
                best_patch, best_pixels = (image_patch, mask_patch), n_positive

        return best_patch

    def _negative_crop(
        self,
        image: np.ndarray,
        mask: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return a crop with no target crack pixels, whenever possible.

        Streaming best effort: materialising all max_crop_attempts crops at once
        costs ~1.8 MB each and defeats the point of an early exit.
        """
        best_patch, best_pixels = None, None

        for _ in range(self.max_crop_attempts):
            image_patch, mask_patch = self._random_crop(image, mask)
            n_positive = int(mask_patch.sum())
            if n_positive <= self.max_negative_pixels:
                return image_patch, mask_patch
            if best_pixels is None or n_positive < best_pixels:
                best_patch, best_pixels = (image_patch, mask_patch), n_positive

        return best_patch

    def _get_positive_patch(self) -> tuple[np.ndarray, np.ndarray]:
        sample_index = int(np.random.choice(self.positive_sample_indices))
        image, mask = self._load_sample(sample_index)
        return self._positive_crop(image, mask)

    def _get_negative_patch(self) -> tuple[np.ndarray, np.ndarray]:
        sample_index = int(np.random.choice(self.negative_sample_indices))
        image, mask = self._load_sample(sample_index)
        return self._negative_crop(image, mask)

    def _get_hard_negative_patch(self) -> tuple[np.ndarray, np.ndarray]:
        """Load one offline-mined difficult zero-GT-crack patch."""
        if not self.hard_negative_pool:
            raise RuntimeError("Cannot sample from an empty hard-negative pool.")

        item = self.hard_negative_pool[
            int(np.random.randint(len(self.hard_negative_pool)))
        ]

        image_path = Path(item["image_path"])
        annotation_path = Path(item["annotation_path"])

        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(
                f"Unreadable hard-negative source image: {image_path}"
            )
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        annotation = load_annotation(annotation_path)
        mask = rasterize_binary(
            annotation,
            labels=self.labels,
            shape=image.shape[:2],
        ).astype(np.float32)

        image, mask = _pad_to_minimum_size(image, mask, self.patch_size)

        height, width = image.shape[:2]
        max_top = height - self.patch_size
        max_left = width - self.patch_size

        top = int(np.clip(int(item["y"]), 0, max_top))
        left = int(np.clip(int(item["x"]), 0, max_left))

        image_patch, mask_patch = _crop_at(
            image=image,
            mask=mask,
            top=top,
            left=left,
            patch_size=self.patch_size,
        )

        if int(mask_patch.sum()) > self.max_negative_pixels:
            print(
                f"[patch-dataset] WARNING: mined hard negative at ({left},{top}) of "
                f"{image_path.name} contains crack pixels, falling back to a random negative."
            )
            return self._get_negative_patch()

        return image_patch, mask_patch

    def __getitem__(self, index: int):
        """Return positive, mined-hard-negative, or random-negative patch."""
        if index < self.n_positive_patches:
            image_patch, mask_patch = self._get_positive_patch()

        elif index < self.n_positive_patches + self.n_hard_negative_patches:
            image_patch, mask_patch = self._get_hard_negative_patch()

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
        # Deterministic but informative crop: centred on the crack centroid when the image is annotated, on the geometric centre otherwise. A plain centre crop is crack-free on most bridge photos, which makes the monitored Dice almost pure noise and weakens best.pt selection.
        crack_y, crack_x = np.where(mask > 0)
        if len(crack_y) > 0:
            center_y, center_x = int(crack_y.mean()), int(crack_x.mean())
        else:
            center_y, center_x = height // 2, width // 2

        top = int(np.clip(center_y - self.patch_size // 2, 0, height - self.patch_size))
        left = int(np.clip(center_x - self.patch_size // 2, 0, width - self.patch_size))

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


# --------------------------------------------------------------------------- #
# P1ML: multilabel damage-only target (6 independent channels)
# --------------------------------------------------------------------------- #
def rasterize_unified_damage(
    annotation: dict,
    shape: tuple[int, int] | None = None,
    ) -> np.ndarray:
    """Rasterise the 6 unified damage channels into a [6,H,W] uint8 stack.

    Channels are independent and may overlap (a spalling area can be corroded), which is why the target is multilabel and not multiclass. Component labels (Bearing, EJoint, ...) are dropped by the mapping, never folded into a class.
    """
    label_to_channel = unified_damage_label_to_channel()
    n_channels = len(UNIFIED_DAMAGE_CLASSES)

    height, width = shape or (int(annotation["imageHeight"]), int(annotation["imageWidth"]))
    masks = np.zeros((n_channels, height, width), dtype=np.uint8)

    for shape_dict in annotation.get("shapes", []):
        channel = label_to_channel.get(shape_dict.get("label"))
        if channel is not None:
            _fill(masks[channel], shape_dict.get("points", []))

    return masks


def unified_damage_mask_hwc(
    annotation: dict,
    shape: tuple[int, int] | None = None,
    ) -> np.ndarray:
    """Return the multilabel target as float32 [H,W,6], the layout albumentations expects."""
    masks = rasterize_unified_damage(annotation, shape=shape)
    return np.ascontiguousarray(masks.transpose(1, 2, 0)).astype(np.float32)


def _pad_to_minimum_size_multilabel(
    image: np.ndarray,
    mask: np.ndarray,
    patch_size: int,
    ) -> tuple[np.ndarray, np.ndarray]:
    """Pad aligned image and [H,W,C] multilabel mask so a native crop is always valid.

    `cv2.copyMakeBorder` is limited to at most 4 channels, so the mask is padded with `np.pad` (zeros = "no damage annotated", which is the correct padding semantics for every channel).
    """
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
    mask = np.pad(
        mask,
        ((0, pad_bottom), (0, pad_right), (0, 0)),
        mode="constant",
        constant_values=0.0,
    )
    return image, mask


def _union_positive_pixels(mask_hwc: np.ndarray) -> int:
    """Number of pixels positive in at least one of the 6 damage channels."""
    return int((mask_hwc.max(axis=2) > 0).sum())


def multilabel_sample_targets(
    samples: list[tuple[Path, Path]],
    ) -> list[list[int]]:
    """Return per-image presence flags for the 6 unified damage classes."""
    label_to_channel = unified_damage_label_to_channel()
    n_channels = len(UNIFIED_DAMAGE_CLASSES)
    targets = []

    for _, annotation_path in samples:
        annotation = load_annotation(annotation_path)
        present = [0] * n_channels

        for shape_dict in annotation.get("shapes", []):
            channel = label_to_channel.get(shape_dict.get("label"))
            if channel is not None and len(shape_dict.get("points", [])) >= 3:
                present[channel] = 1

        targets.append(present)

    return targets


def summarize_multilabel_targets(targets: list[list[int]]) -> dict:
    """Image-level composition of a split for the 6 unified damage classes."""
    if not targets:
        raise ValueError("Cannot summarize an empty DACL10K split.")

    n_images = len(targets)
    per_class = {
        name: int(sum(row[channel] for row in targets))
        for channel, name in enumerate(UNIFIED_DAMAGE_CLASSES)
    }

    n_any = int(sum(1 for row in targets if any(row)))

    return {
        "n_images": n_images,
        "n_images_with_any_damage": n_any,
        "n_images_without_any_damage": n_images - n_any,
        "n_images_per_class": per_class,
        "image_fraction_per_class": {
            name: count / n_images for name, count in per_class.items()
        },
        "mean_classes_per_image": sum(sum(row) for row in targets) / n_images,
    }


class Dacl10kMultilabelPatchDataset(Dataset):
    """Native-resolution multilabel patches with a guaranteed positive quota.

    Identical sampling contract to `Dacl10kCrackPatchDataset`, with one
    difference that matters for P1ML: a patch is *positive* when it contains at least `min_positive_pixels` pixels positive in the **union** of the 6 damage channels. A crack-only criterion would starve the five non-crack channels.

    Hard-negative mining is deliberately not supported here: the P1-B3 pool was mined by a binary crack model and would bias the multilabel negatives.
    """

    def __init__(
        self,
        samples: list[tuple[Path, Path]],
        transform,
        patch_size: int,
        positive_patch_fraction: float = 0.70,
        min_positive_pixels: int = 256,
        max_negative_pixels: int = 0,
        max_crop_attempts: int = 100,
        patches_per_image: int = 4,
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
        self.n_channels = len(UNIFIED_DAMAGE_CLASSES)

        self.positive_sample_indices = self._find_positive_sample_indices()
        positive_set = set(self.positive_sample_indices)
        self.negative_sample_indices = [
            index for index in range(len(self.samples)) if index not in positive_set
        ]

        if not self.positive_sample_indices:
            raise RuntimeError(
                "No DACL10K image can produce a positive multilabel patch with at least "
                f"{self.min_positive_pixels} damage pixels."
            )

        # Almost every DACL10K image carries some damage (Weathering alone covers a large share of the split), so a pool of fully damage-free images can be empty. Negatives are then cropped from annotated images, where empty regions are abundant.
        self.negative_source_indices = self.negative_sample_indices or list(
            range(len(self.samples))
        )
        if not self.negative_sample_indices:
            print(
                "[patch-dataset] no damage-free image in this split: negative patches "
                "will be cropped from annotated images (union-empty regions)."
            )

        self.n_patches = len(self.samples) * self.patches_per_image
        self.n_positive_patches = round(self.n_patches * self.positive_patch_fraction)
        self.n_negative_patches = self.n_patches - self.n_positive_patches

        print(
            "[patch-dataset] multilabel composition | "
            f"total={self.n_patches} | positive={self.n_positive_patches} | "
            f"negative={self.n_negative_patches} | "
            f"positive source images={len(self.positive_sample_indices)} | "
            f"damage-free images={len(self.negative_sample_indices)}"
        )

    def _find_positive_sample_indices(self) -> list[int]:
        """Images whose full-resolution union mask holds enough damage pixels.

        Only the JSON annotations are decoded here, never the JPEGs. As in the binary dataset this is a necessary but not sufficient condition for a single crop, which `_positive_crop` handles statistically.
        """
        positive_indices = []

        for index, (_, annotation_path) in enumerate(self.samples):
            annotation = load_annotation(annotation_path)
            masks = rasterize_unified_damage(annotation)

            if int((masks.max(axis=0) > 0).sum()) >= self.min_positive_pixels:
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
        mask = unified_damage_mask_hwc(annotation, shape=image.shape[:2])

        return _pad_to_minimum_size_multilabel(image, mask, self.patch_size)

    def _random_crop(self, image: np.ndarray, mask: np.ndarray):
        height, width = image.shape[:2]
        top = np.random.randint(0, height - self.patch_size + 1)
        left = np.random.randint(0, width - self.patch_size + 1)
        return _crop_at(image, mask, top, left, self.patch_size)

    def _positive_crop(self, image: np.ndarray, mask: np.ndarray):
        """Return a damage-anchored crop, falling back to the richest one found."""
        union = mask.max(axis=2)
        damage_y, damage_x = np.where(union > 0)
        if len(damage_y) == 0:
            raise RuntimeError("Positive source image unexpectedly has an empty damage mask.")

        height, width = image.shape[:2]
        best_patch, best_pixels = None, -1

        for _ in range(self.max_crop_attempts):
            anchor_index = np.random.randint(len(damage_y))
            center_y, center_x = int(damage_y[anchor_index]), int(damage_x[anchor_index])

            top = np.random.randint(
                max(0, center_y - self.patch_size + 1),
                min(center_y, height - self.patch_size) + 1,
            )
            left = np.random.randint(
                max(0, center_x - self.patch_size + 1),
                min(center_x, width - self.patch_size) + 1,
            )

            image_patch, mask_patch = _crop_at(image, mask, top, left, self.patch_size)
            n_positive = _union_positive_pixels(mask_patch)

            if n_positive >= self.min_positive_pixels:
                return image_patch, mask_patch
            if n_positive > best_pixels:
                best_patch, best_pixels = (image_patch, mask_patch), n_positive

        return best_patch

    def _negative_crop(self, image: np.ndarray, mask: np.ndarray):
        """Return a crop with no damage pixels in any channel, whenever possible."""
        best_patch, best_pixels = None, None

        for _ in range(self.max_crop_attempts):
            image_patch, mask_patch = self._random_crop(image, mask)
            n_positive = _union_positive_pixels(mask_patch)

            if n_positive <= self.max_negative_pixels:
                return image_patch, mask_patch
            if best_pixels is None or n_positive < best_pixels:
                best_patch, best_pixels = (image_patch, mask_patch), n_positive

        return best_patch

    def __getitem__(self, index: int):
        if index < self.n_positive_patches:
            sample_index = int(np.random.choice(self.positive_sample_indices))
            image, mask = self._load_sample(sample_index)
            image_patch, mask_patch = self._positive_crop(image, mask)
        else:
            sample_index = int(np.random.choice(self.negative_source_indices))
            image, mask = self._load_sample(sample_index)
            image_patch, mask_patch = self._negative_crop(image, mask)

        augmented = self.transform(image=image_patch, mask=mask_patch)
        # transpose_mask=True in the multilabel transforms already yields [C,H,W].
        return augmented["image"], augmented["mask"].float()


class Dacl10kMultilabelCenterPatchDataset(Dataset):
    """One deterministic damage-centred patch per image, for training-time monitoring."""

    def __init__(
        self,
        samples: list[tuple[Path, Path]],
        transform,
        patch_size: int,
    ) -> None:
        self.samples = samples
        self.transform = transform
        self.patch_size = int(patch_size)
        self.n_channels = len(UNIFIED_DAMAGE_CLASSES)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        image_path, annotation_path = self.samples[index]

        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Unreadable image: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        annotation = load_annotation(annotation_path)
        mask = unified_damage_mask_hwc(annotation, shape=image.shape[:2])
        image, mask = _pad_to_minimum_size_multilabel(image, mask, self.patch_size)

        height, width = image.shape[:2]
        union = mask.max(axis=2)
        damage_y, damage_x = np.where(union > 0)

        if len(damage_y) > 0:
            center_y, center_x = int(damage_y.mean()), int(damage_x.mean())
        else:
            center_y, center_x = height // 2, width // 2

        top = int(np.clip(center_y - self.patch_size // 2, 0, height - self.patch_size))
        left = int(np.clip(center_x - self.patch_size // 2, 0, width - self.patch_size))

        image_patch, mask_patch = _crop_at(image, mask, top, left, self.patch_size)
        augmented = self.transform(image=image_patch, mask=mask_patch)
        return augmented["image"], augmented["mask"].float()
