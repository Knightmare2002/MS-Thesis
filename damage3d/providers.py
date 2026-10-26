"""Probability providers: the ONLY component to replace when the model is ready.

Contract
--------
``provider.get(camera) -> ProbabilityMap`` with

* ``probs``: float32 array (5, Hm, Wm), channel order = ``DAMAGE_CLASSES``
  (crack, spalling, corrosion, moisture, surface; delamination closed), independent
  sigmoid probabilities in [0, 1]; channels may overlap (no softmax/argmax).
  NaN marks "no prediction for this class at this pixel".
* ``valid``: optional bool (Hm, Wm); False pixels are not a valid view.
* The map covers the full Metashape image frame (same orientation as the
  pixel array used by Metashape, i.e. WITHOUT EXIF rotation). (Hm, Wm) may be
  a downscaled version of (H, W) with the same aspect ratio (tolerance 1%);
  pixel coordinates are rescaled by Wm/W and Hm/H.

File format for ``--probability-source files`` (defined here; the model does
not exist yet): ``<probabilities-dir>/<camera stem>.npz`` with keys

* ``probs``   (K, Hm, Wm) float16/float32 in [0, 1], or uint8 (value / 255);
* ``classes`` array of K strings (scripts/23 writes the 6 MODEL_CLASSES); the
  DAMAGE_CLASSES channels are selected BY NAME (missing class -> error);
* ``valid``   optional (Hm, Wm) bool/uint8.

Use ``save_probability_npz`` to write it from the future inference script.
"""

from __future__ import annotations

import hashlib
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .classes import DAMAGE_CLASSES, MODEL_CLASSES, NUM_CLASSES

log = logging.getLogger("damage3d")


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
        for c, (cx, cy) in enumerate(self.CENTERS[:NUM_CLASSES]):
            probs[c] = 0.95 * np.exp(-0.5 * (((gx - cx) ** 2 + (gy - cy) ** 2) / sigma**2))
        return ProbabilityMap(probs)

    def describe(self) -> dict:
        d = {"type": "synthetic", "pattern": self.pattern, "is_ai_prediction": False}
        if self.pattern == "regions":
            d.update({"downsample": self.downsample, "centers": self.CENTERS[:NUM_CLASSES], "sigma": 0.22, "peak": 0.95})
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
            missing = [c for c in DAMAGE_CLASSES if c not in classes]
            if missing or data["probs"].shape[0] != len(classes):
                raise ValueError(f"{path}: classes {classes} (probs {data['probs'].shape}) "
                                 f"do not provide {DAMAGE_CLASSES}.")
            probs = data["probs"][[classes.index(c) for c in DAMAGE_CLASSES]]   # closed classes dropped
            probs = probs.astype(np.float32) / 255.0 if probs.dtype == np.uint8 else probs.astype(np.float32)
            valid = data["valid"].astype(bool) if "valid" in data else None
        pmap = ProbabilityMap(probs, valid)
        pmap.validate(width, height, stem)
        return pmap

    def describe(self) -> dict:
        return {"type": "files", "dir": str(self.dir), "pattern": self.pattern,
                "files": self._identities, "is_ai_prediction": True}


# --------------------------------------------------------------------------- #
# Platt-calibrated providers (single model and ensemble)
# --------------------------------------------------------------------------- #
MAPS_MANIFEST = "maps_manifest.json"


def check_maps_checkpoint(maps_dir: Path, expected_sha256: str | None, label: str) -> None:
    """Refuse maps written by a checkpoint different from the calibrated one.

    The check uses ``maps_manifest.json`` of scripts/23 and the checkpoint hash
    stored by scripts/27; it is skipped (with a warning) if either is missing.
    """
    manifest = Path(maps_dir) / MAPS_MANIFEST
    if expected_sha256 is None or not manifest.is_file():
        log.warning("%s: checkpoint/maps binding not verified (hash or %s missing).", label, MAPS_MANIFEST)
        return
    found = json.loads(manifest.read_text(encoding="utf-8")).get("settings", {}).get("checkpoint_sha256")
    if found != expected_sha256:
        raise ValueError(f"{label}: maps in {maps_dir} come from checkpoint {str(found)[:12]}, "
                         f"calibration from {expected_sha256[:12]}.")


class CalibratedProvider(ProbabilityProvider):
    """Raw maps of ONE model + per-class Platt map (scripts/27 JSON)."""

    def __init__(self, base: FileProvider, calibration_path: Path):
        from src.eval.probability_calibration import load_platt, sha256_file

        self.base = base
        self.calibration_path = Path(calibration_path)
        self.a, self.b, payload = load_platt(self.calibration_path, DAMAGE_CLASSES)
        self.calibration_sha256 = sha256_file(self.calibration_path)
        self.run_name = payload.get("run_name")
        check_maps_checkpoint(base.dir, payload.get("checkpoint", {}).get("sha256"), str(self.run_name))

    @property
    def dir(self) -> Path:
        return self.base.dir

    def check_available(self, stems: list[str]) -> list[str]:
        return self.base.check_available(stems)

    def register(self, stems: list[str]) -> None:
        self.base.register(stems)

    def get(self, stem: str, width: int, height: int) -> ProbabilityMap:
        from src.eval.probability_calibration import apply_platt

        pmap = self.base.get(stem, width, height)
        return ProbabilityMap(apply_platt(pmap.probs, self.a, self.b), pmap.valid)

    def describe(self) -> dict:
        return {"type": "files+platt", "base": self.base.describe(), "run_name": self.run_name,
                "calibration": {"path": str(self.calibration_path), "sha256": self.calibration_sha256,
                                "a": self.a.tolist(), "b": self.b.tolist()},
                "is_ai_prediction": True}


class EnsembleProvider(ProbabilityProvider):
    """Weighted mean of Platt-calibrated maps of several models (scripts/28 JSON).

    p_ens(x, c) = sum_m w_m * sigmoid(a_mc * logit(p_m(x, c)) + b_mc),  sum_m w_m = 1.
    Fusion then averages p_ens over the views of each 3D point, i.e. the fused
    score is the weighted mean of the calibrated probabilities over models
    and views. All members must store maps of the same size (same
    ``--downsample`` in scripts/23); ``valid`` is the AND of the members.
    """

    def __init__(self, ensemble_path: Path, project_root: Path, member_maps: dict[str, Path] | None = None):
        from src.eval.probability_calibration import FORMAT_ENSEMBLE, params_arrays, read_json, sha256_file

        self.ensemble_path = Path(ensemble_path)
        payload = read_json(self.ensemble_path)
        if payload.get("format") != FORMAT_ENSEMBLE:
            raise ValueError(f"{ensemble_path}: format '{payload.get('format')}' != '{FORMAT_ENSEMBLE}'.")
        missing = [c for c in DAMAGE_CLASSES if c not in payload.get("classes", [])]
        if missing:
            raise ValueError(f"{ensemble_path}: classes {missing} not in {payload.get('classes')}.")
        self.ensemble_sha256 = sha256_file(self.ensemble_path)
        self.name = payload.get("name")
        member_maps = dict(member_maps or {})
        unknown = set(member_maps) - {m["name"] for m in payload["members"]}
        if unknown:
            raise ValueError(f"--member-maps refers to unknown members {sorted(unknown)}.")
        self.members = []
        for m in payload["members"]:
            maps_dir = Path(member_maps.get(m["name"]) or Path(project_root) / "damage3d_maps" / Path(m["run_dir"]).name)
            if not maps_dir.is_dir():
                raise FileNotFoundError(f"Maps of member '{m['name']}' not found: {maps_dir} (use --member-maps).")
            check_maps_checkpoint(maps_dir, m.get("checkpoint_sha256"), m["name"])
            a, b = params_arrays(m["params"], DAMAGE_CLASSES)
            self.members.append({"name": m["name"], "provider": FileProvider(maps_dir), "a": a, "b": b,
                                 "weight": float(m["weight"])})
        total = sum(m["weight"] for m in self.members)
        if not self.members or total <= 0:
            raise ValueError(f"{ensemble_path}: no members or non-positive weights.")
        for m in self.members:
            m["weight"] /= total

    @property
    def dir(self) -> list[str]:
        return [str(m["provider"].dir) for m in self.members]

    def check_available(self, stems: list[str]) -> list[str]:
        return sorted({s for m in self.members for s in m["provider"].check_available(stems)})

    def register(self, stems: list[str]) -> None:
        for m in self.members:
            m["provider"].register(stems)

    def get(self, stem: str, width: int, height: int) -> ProbabilityMap:
        from src.eval.probability_calibration import apply_platt

        probs, valid, shape = None, None, None
        for m in self.members:
            pmap = m["provider"].get(stem, width, height)
            if shape is not None and pmap.probs.shape != shape:
                raise ValueError(f"{stem}: member '{m['name']}' map {pmap.probs.shape} != {shape}; "
                                 "write all members with the same --downsample.")
            shape = pmap.probs.shape
            term = m["weight"] * apply_platt(pmap.probs, m["a"], m["b"])
            probs = term if probs is None else probs + term   # NaN in any member -> NaN (no prediction)
            if pmap.valid is not None:
                valid = pmap.valid if valid is None else (valid & pmap.valid)
        return ProbabilityMap(np.clip(probs, 0.0, 1.0).astype(np.float32), valid)

    def describe(self) -> dict:
        return {"type": "ensemble+platt", "name": self.name,
                "ensemble": {"path": str(self.ensemble_path), "sha256": self.ensemble_sha256},
                "members": [{"name": m["name"], "weight": m["weight"], "a": m["a"].tolist(), "b": m["b"].tolist(),
                             "maps": m["provider"].describe()} for m in self.members],
                "is_ai_prediction": True}


def save_probability_npz(path: Path, probs: np.ndarray, valid: np.ndarray | None = None,
                         dtype: str = "float16", classes: tuple[str, ...] = MODEL_CLASSES) -> None:
    """Writes the FileProvider format; default = all model channels (scripts/23)."""
    probs = np.asarray(probs)
    if probs.ndim != 3 or probs.shape[0] != len(classes):
        raise ValueError(f"probs must be ({len(classes)}, H, W), got {probs.shape}.")
    if dtype == "uint8":
        stored = np.clip(np.rint(probs * 255.0), 0, 255).astype(np.uint8)
    else:
        stored = probs.astype(dtype)
    arrays = {"probs": stored, "classes": np.array(classes)}
    if valid is not None:
        arrays["valid"] = np.asarray(valid, dtype=bool)
    np.savez_compressed(path, **arrays)
