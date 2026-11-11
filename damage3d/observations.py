"""Per point-view observation log (``--observation-log``).

One record per VALID observation (point in front, inside the image, radial
guard passed, not occluded, provider pixel valid), i.e. exactly the views added
to the accumulator. It lets every fusion rule (mean, max, GSD culling, cos/g
weighting, llr, censored llr) and every detectability estimate be recomputed
offline without projecting again.

Record (``OBS_DTYPE``, 54 B; probabilities stored exactly as accumulated, float32):
    source_index  index in the source PLY
    camera        order of the camera in ``selected_cameras.csv``
    u, v          full-image pixel coordinates (Metashape corner convention)
    depth         camera Z (chunk units)
    range_source  Euclidean camera-to-point distance, SOURCE units (not camera Z)
    gsd_frontal   depth * s / f, SOURCE units per image pixel (s = chunk->source scale)
    cos_theta     |cos| between source normal and viewing ray (NaN without normals)
    map_scale     sigma_v = W / W_m times explicitly declared additional inference_scale
                  (g_iv = gsd_frontal * map_scale / sqrt(cos_theta), SOURCE units per map px)
    p_<class>     probability sampled by the provider (NaN = invalid channel)

Files: ``<run>/observations/chunk_XXXXXX_batch_YYYYYY.npy``, one per (point chunk,
camera batch), written atomically BEFORE the accumulator commit of that chunk:
a batch replayed after an interruption rewrites the same file, so records are
never duplicated. ``observations/meta.json`` stores the camera table and units.

Size: ~54 B x (points x mean valid views). Restrict it with
``--observation-subset`` (``.npy`` of source indices, e.g. ``roi_points_index.npy``
written by ``damage3d.roi_views``).
"""

from __future__ import annotations

import os
import json
from itertools import groupby
from pathlib import Path

import numpy as np

from .classes import DAMAGE_CLASSES
from .run_state import sha256_file, write_json_atomic

OBS_DTYPE = np.dtype([("source_index", "<u4"), ("camera", "<u2"), ("u", "<f4"), ("v", "<f4"),
                      ("depth", "<f4"), ("gsd_frontal", "<f4"), ("cos_theta", "<f4"),
                      ("map_scale", "<f4"), ("range_source", "<f4")]
                     + [(f"p_{c}", "<f4") for c in DAMAGE_CLASSES])


def cos_incidence(xyz_src: np.ndarray, normals_src: np.ndarray, center_src: np.ndarray) -> np.ndarray:
    """|cos| between the (unoriented) normal and the ray point -> camera centre."""
    ray = np.asarray(center_src, np.float64)[None, :] - np.asarray(xyz_src, np.float64)
    n = np.asarray(normals_src, np.float64)
    den = np.linalg.norm(ray, axis=1) * np.linalg.norm(n, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.abs(np.einsum("ij,ij->i", ray, n)) / den


def make_records(source_index, camera: int, u, v, depth, gsd_frontal, cos_theta, map_scale: float,
                 probs, range_source=None) -> np.ndarray:
    rec = np.empty(len(source_index), dtype=OBS_DTYPE)
    rec["source_index"] = source_index
    rec["camera"] = camera
    rec["u"], rec["v"], rec["depth"], rec["gsd_frontal"] = u, v, depth, gsd_frontal
    rec["cos_theta"] = np.nan if cos_theta is None else cos_theta
    rec["map_scale"] = map_scale
    rec["range_source"] = np.nan if range_source is None else range_source
    for c, name in enumerate(DAMAGE_CLASSES):
        rec[f"p_{name}"] = probs[:, c]
    return rec


class ObservationLog:
    def __init__(self, run_path: Path, subset: np.ndarray | None = None):
        self.dir = Path(run_path) / "observations"
        self.subset = None if subset is None else np.unique(np.asarray(subset, dtype=np.int64))

    def path(self, chunk: int, batch: int) -> Path:
        return self.dir / f"chunk_{chunk:06d}_batch_{batch:06d}.npy"

    def seal(self, verify_only=False):
        """Freeze/check shard inventory after all camera batches are committed.

        Culling and empty ROI intersections legitimately produce no shard. An inventory
        distinguishes that case from deletion/corruption after the run completed.
        """
        inventory = {p.name: sha256_file(p) for p in sorted(self.dir.glob("chunk_*_batch_*.npy"))}
        path = self.dir / "inventory.json"
        if path.exists():
            if json.loads(path.read_text(encoding="utf-8")) != inventory:
                raise ValueError("Observation inventory changed: missing, extra or corrupted shard.")
        elif verify_only:
            raise ValueError("Unsealed observation log: complete/resume projection first.")
        else:
            write_json_atomic(path, inventory)
        return inventory

    def keep_mask(self, source_index: np.ndarray) -> np.ndarray | None:
        """Points of the chunk that are logged (None = all)."""
        if self.subset is None:
            return None
        return np.isin(source_index, self.subset)

    def write(self, chunk: int, batch: int, records: list[np.ndarray]) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        data = np.concatenate(records) if records else np.zeros(0, dtype=OBS_DTYPE)
        path = self.path(chunk, batch)
        tmp = path.with_name(path.name + ".tmp")
        with tmp.open("wb") as f:
            np.save(f, data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)


def load_observations(run_path: Path) -> np.ndarray:
    """All records of a run, sorted by (source_index, camera)."""
    data = np.concatenate(list(iter_observations(run_path)))
    return data[np.lexsort((data["camera"], data["source_index"]))]


def iter_observations(run_path: Path, source_ids=None):
    """Read one point chunk across batches; filter ROI before concatenation."""
    files = sorted((Path(run_path) / "observations").glob("chunk_*_batch_*.npy"))
    if not files:
        raise FileNotFoundError(f"No observation log in {run_path} (run with --observation-log).")
    for _, batch_files in groupby(files, key=lambda p: p.name.split("_batch_")[0]):
        parts = []
        for f in batch_files:
            a = np.load(f, mmap_mode="r", allow_pickle=False)
            if a.dtype != OBS_DTYPE:
                raise ValueError(f"{f}: incompatible observation schema; regenerate this run.")
            parts.append(a if source_ids is None else a[np.isin(a["source_index"], source_ids)])
        a = np.concatenate(parts)
        a = a[np.lexsort((a["camera"], a["source_index"]))]
        if len(a) > 1 and np.any((a["source_index"][1:] == a["source_index"][:-1]) &
                                (a["camera"][1:] == a["camera"][:-1])):
            raise ValueError("Duplicate point-camera observations.")
        yield a


def probs_of(records: np.ndarray) -> np.ndarray:
    """(N, C) float32 probabilities of the records."""
    return np.column_stack([records[f"p_{c}"].astype(np.float32) for c in DAMAGE_CLASSES])
