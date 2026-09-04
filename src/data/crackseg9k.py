"""CrackSeg9k: binary crack segmentation dataset.

CrackSeg9k ships as two flat folders (images / masks) whose files share the same
stem. Because different mirrors use different extensions (.jpg vs .png) the
pairing is done on the stem, and any unpaired file is reported instead of being
silently skipped.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from torch.utils.data import Dataset

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


# --------------------------------------------------------------------------- #
# Pairing + splitting
# --------------------------------------------------------------------------- #
def list_pairs(images_dir: str | Path, masks_dir: str | Path) -> list[tuple[Path, Path]]:
    """Return [(image_path, mask_path)] pairs matched by filename stem."""
    images_dir, masks_dir = Path(images_dir), Path(masks_dir)
    if not images_dir.is_dir() or not masks_dir.is_dir():
        raise FileNotFoundError(f"Missing folder: {images_dir} or {masks_dir}")

    masks = {
        p.stem: p for p in sorted(masks_dir.rglob("*"))
        if p.suffix.lower() in IMAGE_EXTENSIONS
    }
    pairs, orphans = [], []
    for img in sorted(images_dir.rglob("*")):
        if img.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        mask = masks.get(img.stem)
        (pairs.append((img, mask)) if mask else orphans.append(img.name))

    if orphans:
        print(f"[CrackSeg9k] WARNING: {len(orphans)} images without mask, e.g. {orphans[:5]}")
    if not pairs:
        raise RuntimeError("No image/mask pair found: check the configured paths.")
    return pairs


def split_pairs(
    pairs: list[tuple[Path, Path]],
    val_fraction: float,
    test_fraction: float,
    seed: int = 42,
) -> dict[str, list[tuple[Path, Path]]]:
    """Deterministic random split into train / val / test (image-level, no leakage)."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(pairs))
    n_val = int(round(val_fraction * len(pairs)))
    n_test = int(round(test_fraction * len(pairs)))
    val_idx, test_idx, train_idx = idx[:n_val], idx[n_val:n_val + n_test], idx[n_val + n_test:]
    return {
        "train": [pairs[i] for i in train_idx],
        "val": [pairs[i] for i in val_idx],
        "test": [pairs[i] for i in test_idx],
    }


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
class CrackSeg9kDataset(Dataset):
    """Yields (image_tensor [3,H,W] float, mask_tensor [1,H,W] float in {0,1})."""

    def __init__(
        self,
        pairs: list[tuple[Path, Path]],
        transform,
        mask_threshold: int = 0,
    ) -> None:
        self.pairs = pairs
        self.transform = transform
        self.mask_threshold = mask_threshold

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int):
        image_path, mask_path = self.pairs[index]

        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Unreadable image: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise RuntimeError(f"Unreadable mask: {mask_path}")
        # Binarise: some mirrors store 0/255, others 0/1, others anti-aliased edges.
        mask = (mask > self.mask_threshold).astype(np.float32)

        augmented = self.transform(image=image, mask=mask)
        # [H,W] -> [1,H,W] so the tensor matches the single-logit model output.
        return augmented["image"], augmented["mask"].unsqueeze(0).float()
