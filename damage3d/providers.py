"""Probability providers: the ONLY component to replace when the model is ready.

Contract
--------
``provider.get(camera) -> ProbabilityMap`` with

* ``probs``: float32 array (6, Hm, Wm), channel order = ``DAMAGE_CLASSES``
  (crack, spalling, corrosion, moisture, delamination, surface), independent
  sigmoid probabilities in [0, 1]; channels may overlap (no softmax/argmax).
  NaN marks "no prediction for this class at this pixel".
* ``valid``: optional bool (Hm, Wm); False pixels are not a valid view.
* The map covers the full Metashape image frame (same orientation as the
  pixel array used by Metashape, i.e. WITHOUT EXIF rotation). (Hm, Wm) may be
  a downscaled version of (H, W) with the same aspect ratio (tolerance 1%);
  pixel coordinates are rescaled by Wm/W and Hm/H.

File format for ``--probability-source files`` (defined here; the model does
not exist yet): ``<probabilities-dir>/<camera stem>.npz`` with keys

* ``probs``   (6, Hm, Wm) float16/float32 in [0, 1], or uint8 (value / 255);
* ``classes`` array of 6 strings, must equal DAMAGE_CLASSES (checked);
* ``valid``   optional (Hm, Wm) bool/uint8.

Use ``save_probability_npz`` to write it from the future inference script.
"""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .classes import DAMAGE_CLASSES, NUM_CLASSES


@dataclass
class ProbabilityMap:
    probs: np.ndarray
    valid: np.ndarray | None = None

    def validate(self, width: int, height: int, label: str) -> None:
        if self.probs.ndim != 3 or self.probs.shape[0] != NUM_CLASSES:
            raise ValueError(f"{label}: probs must be ({NUM_CLASSES}, H, W), got {self.probs.shape}.")
        hm, wm = self.probs.shape[1:]
        if abs((wm / width) - (hm / height)) > 0.01 * max(wm / width, hm / height):
            raise ValueError(f"{label}: map {wm}x{hm} does not preserve the aspect ratio of {width}x{height}.")
        if self.valid is not None and self.valid.shape != (hm, wm):
            raise ValueError(f"{label}: valid mask shape {self.valid.shape} != {(hm, wm)}.")
        finite = self.probs[np.isfinite(self.probs)]
        if finite.size and (finite.min() < 0.0 or finite.max() > 1.0):
            raise ValueError(f"{label}: probabilities outside [0, 1].")


class ProbabilityProvider(ABC):
    classes: tuple[str, ...] = DAMAGE_CLASSES
    is_synthetic: bool = False

    @abstractmethod
    def get(self, stem: str, width: int, height: int) -> ProbabilityMap:
        ...

    @abstractmethod
    def describe(self) -> dict:
        """JSON-serialisable identity, part of the resume fingerprint."""

    def check_available(self, stems: list[str]) -> list[str]:
        """Return the stems for which no map can be produced."""
        return []


class SyntheticProvider(ProbabilityProvider):
    """Deterministic synthetic maps. NOT an AI prediction.

    ``pattern="regions"``: class c is a Gaussian bump centred at a fixed
    position of a 3x2 grid in normalised image coordinates, sigma = 0.22 of
    the image size, peak 0.95; neighbouring bumps overlap so that points with
    two or more classes above 0.5 exist. The pattern is identical for every
    camera (image-space pattern), so different views of the same 3D point
    usually disagree: this exercises multi-view fusion.

    ``pattern="mask"``: reads ``mask_pattern`` (e.g. "{stem}_mask_test.png",
    the prototype synthetic mask, grayscale, white >= 128 as in
    project_test_mask.py) at full resolution; white pixels get probability
    1.0 for ``mask_classes``, 0.0 elsewhere and for the other classes.
    """

    is_synthetic = True
    CENTERS = ((0.25, 0.30), (0.50, 0.30), (0.75, 0.30), (0.25, 0.70), (0.50, 0.70), (0.75, 0.70))

    def __init__(self, pattern: str = "regions", downsample: int = 8, mask_dir: Path | None = None,
                 mask_pattern: str = "{stem}_mask_test.png", mask_classes: tuple[str, ...] = ("crack",)):
        if pattern not in ("regions", "mask"):
            raise ValueError(f"Unknown synthetic pattern '{pattern}'.")
        unknown = [c for c in mask_classes if c not in DAMAGE_CLASSES]
        if unknown:
            raise ValueError(f"Unknown classes for the synthetic mask: {unknown}")
        self.pattern = pattern
        self.downsample = max(1, int(downsample))
        self.mask_dir = Path(mask_dir) if mask_dir is not None else None
        self.mask_pattern = mask_pattern
        self.mask_classes = tuple(mask_classes)

    def _mask_path(self, stem: str) -> Path:
        if self.mask_dir is None:
            raise RuntimeError("Synthetic mask pattern requires a mask directory.")
        return self.mask_dir / self.mask_pattern.format(stem=stem)

    def check_available(self, stems: list[str]) -> list[str]:
        if self.pattern != "mask":
            return []
        return [s for s in stems if not self._mask_path(s).is_file()]

    def get(self, stem: str, width: int, height: int) -> ProbabilityMap:
        if self.pattern == "mask":
            from PIL import Image
            path = self._mask_path(stem)
            with Image.open(path) as image:
                if image.size != (width, height):
                    raise ValueError(f"Mask size {image.size} does not match camera resolution {(width, height)}.")
                mask = np.asarray(image.convert("L")) >= 128
            probs = np.zeros((NUM_CLASSES, height, width), dtype=np.float32)
            for name in self.mask_classes:
                probs[DAMAGE_CLASSES.index(name)][mask] = 1.0
            return ProbabilityMap(probs)

        wm = max(1, int(round(width / self.downsample)))
        hm = max(1, int(round(height / self.downsample)))
        # Pixel centres of the downscaled map in normalised [0, 1] coordinates.
        xs = (np.arange(wm) + 0.5) / wm
        ys = (np.arange(hm) + 0.5) / hm
        gx, gy = np.meshgrid(xs, ys)
        probs = np.empty((NUM_CLASSES, hm, wm), dtype=np.float32)
        sigma = 0.22
        for c, (cx, cy) in enumerate(self.CENTERS):
            probs[c] = 0.95 * np.exp(-0.5 * (((gx - cx) ** 2 + (gy - cy) ** 2) / sigma**2))
        return ProbabilityMap(probs)

    def describe(self) -> dict:
        d = {"type": "synthetic", "pattern": self.pattern, "is_ai_prediction": False}
        if self.pattern == "regions":
            d.update({"downsample": self.downsample, "centers": self.CENTERS, "sigma": 0.22, "peak": 0.95})
        else:
            d.update({"mask_pattern": self.mask_pattern, "mask_classes": list(self.mask_classes),
                      "mask_dir": str(self.mask_dir)})
        return d


def _file_identity(path: Path, partial_bytes: int = 1 << 20) -> dict:
    stat = path.stat()
    h = hashlib.sha256()
    with path.open("rb") as f:
        h.update(f.read(partial_bytes))
        if stat.st_size > 2 * partial_bytes:
            f.seek(-partial_bytes, 2)
        h.update(f.read(partial_bytes))
    return {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns, "sha256_head_tail": h.hexdigest()}


class FileProvider(ProbabilityProvider):
    """Reads ``<dir>/<stem>.npz`` maps produced by the (future) model."""

    def __init__(self, probabilities_dir: Path, filename_pattern: str = "{stem}.npz"):
        self.dir = Path(probabilities_dir)
        self.pattern = filename_pattern
        self._identities: dict[str, dict] = {}

    def path_for(self, stem: str) -> Path:
        return self.dir / self.pattern.format(stem=stem)

    def check_available(self, stems: list[str]) -> list[str]:
        return [s for s in stems if not self.path_for(s).is_file()]

    def register(self, stems: list[str]) -> None:
        """Record size/mtime/hash of every selected map (resume fingerprint)."""
        self._identities = {s: _file_identity(self.path_for(s)) for s in stems}

    def get(self, stem: str, width: int, height: int) -> ProbabilityMap:
        path = self.path_for(stem)
        with np.load(path, allow_pickle=False) as data:
            if "probs" not in data or "classes" not in data:
                raise ValueError(f"{path}: required keys 'probs' and 'classes'.")
            classes = tuple(str(c) for c in data["classes"].tolist())
            if classes != DAMAGE_CLASSES:
                raise ValueError(f"{path}: class order {classes} != {DAMAGE_CLASSES}.")
            probs = data["probs"]
            probs = probs.astype(np.float32) / 255.0 if probs.dtype == np.uint8 else probs.astype(np.float32)
            valid = data["valid"].astype(bool) if "valid" in data else None
        pmap = ProbabilityMap(probs, valid)
        pmap.validate(width, height, stem)
        return pmap

    def describe(self) -> dict:
        return {"type": "files", "dir": str(self.dir), "pattern": self.pattern,
                "files": self._identities, "is_ai_prediction": True}


def save_probability_npz(path: Path, probs: np.ndarray, valid: np.ndarray | None = None,
                         dtype: str = "float16") -> None:
    """Helper for the future inference script (writes the FileProvider format)."""
    probs = np.asarray(probs)
    if probs.ndim != 3 or probs.shape[0] != NUM_CLASSES:
        raise ValueError(f"probs must be ({NUM_CLASSES}, H, W), got {probs.shape}.")
    if dtype == "uint8":
        stored = np.clip(np.rint(probs * 255.0), 0, 255).astype(np.uint8)
    else:
        stored = probs.astype(dtype)
    arrays = {"probs": stored, "classes": np.array(DAMAGE_CLASSES)}
    if valid is not None:
        arrays["valid"] = np.asarray(valid, dtype=bool)
    np.savez_compressed(path, **arrays)
