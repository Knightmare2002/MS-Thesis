"""Multi-view accumulation and decision rules.

Per point i and class c (all counts over VALID views only):

* ``sum[i, c]``    = sum of p_v(i, c) over views v where the class value is finite;
* ``count[i, c]``  = number of such views;
* ``votes[i, c]``  = number of such views with p_v(i, c) >= vote_threshold;
* ``n_views[i]``   = number of geometrically valid views (in front, inside the
  image, radial guard passed, not occluded, provider pixel valid).

Fused score: mean probability  s(i, c) = sum / count  (NaN if count = 0).
Label: l(i, c) = [count >= min_views] and [s(i, c) >= threshold_c]; classes
are independent, several labels can be true on the same point.

State (uint8): see ``damage3d.classes``; a point with n_views = 0 is
NOT_OBSERVED and is never reported as healthy. OBSERVED_NO_DAMAGE requires
count >= min_views for EVERY class and no positive label.

Views are added one camera at a time in the deterministic camera order, so
the float32 sums are bitwise independent of the point-chunk size and of the
camera-batch size.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .classes import (CLASS_COLORS, DAMAGE_CLASSES, NUM_CLASSES, STATE_COLORS, STATE_DAMAGE,
                      STATE_INSUFFICIENT_VIEWS, STATE_NOT_OBSERVED, STATE_OBSERVED_NO_DAMAGE)

COUNT_DTYPE = np.uint16
MAX_VIEWS = np.iinfo(COUNT_DTYPE).max
CAMERA_STAT_FIELDS = ("candidates", "in_front", "in_image", "occluded", "valid_views")


@dataclass
class ChunkAccumulator:
    sum: np.ndarray          # (n, C) float32
    count: np.ndarray        # (n, C) uint16
    votes: np.ndarray        # (n, C) uint16
    n_views: np.ndarray      # (n,) uint16
    camera_stats: np.ndarray  # (n_cameras, len(CAMERA_STAT_FIELDS) + C) int64, per selected camera
    last_batch: int          # index of the last camera batch applied (-1 = none)

    @classmethod
    def zeros(cls, n: int, n_cameras: int) -> "ChunkAccumulator":
        return cls(
            sum=np.zeros((n, NUM_CLASSES), dtype=np.float32),
            count=np.zeros((n, NUM_CLASSES), dtype=COUNT_DTYPE),
            votes=np.zeros((n, NUM_CLASSES), dtype=COUNT_DTYPE),
            n_views=np.zeros(n, dtype=COUNT_DTYPE),
            camera_stats=np.zeros((n_cameras, len(CAMERA_STAT_FIELDS) + NUM_CLASSES), dtype=np.int64),
            last_batch=-1,
        )

    def add_view(self, local_idx: np.ndarray, probs: np.ndarray, vote_threshold: float) -> None:
        """Add one camera's observations. ``local_idx`` must be unique."""
        if local_idx.size == 0:
            return
        if int(self.n_views[local_idx].max()) >= MAX_VIEWS:
            raise OverflowError("More than 65535 views for a point.")
        self.n_views[local_idx] += 1
        finite = np.isfinite(probs)
        for c in range(NUM_CLASSES):
            ok = finite[:, c]
            idx = local_idx[ok]
            p = probs[ok, c]
            self.sum[idx, c] += p
            self.count[idx, c] += 1
            self.votes[idx, c] += (p >= vote_threshold).astype(COUNT_DTYPE)

    # ---- atomic persistence -------------------------------------------------
    def save_atomic(self, path: Path) -> None:
        path = Path(path)
        tmp = path.with_name(path.name + ".tmp")
        with tmp.open("wb") as f:
            np.savez(f, sum=self.sum, count=self.count, votes=self.votes, n_views=self.n_views,
                     camera_stats=self.camera_stats, last_batch=np.int64(self.last_batch))
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)

    @classmethod
    def load(cls, path: Path) -> "ChunkAccumulator":
        with np.load(path, allow_pickle=False) as d:
            return cls(sum=d["sum"], count=d["count"], votes=d["votes"], n_views=d["n_views"],
                       camera_stats=d["camera_stats"], last_batch=int(d["last_batch"]))


def parse_thresholds(text: str | None, default: float = 0.5) -> np.ndarray:
    """'0.5' or 'crack=0.6,spalling=0.7' (unspecified classes keep ``default``)."""
    thr = np.full(NUM_CLASSES, default, dtype=np.float32)
    if not text:
        return thr
    text = text.strip()
    if "=" not in text:
        return np.full(NUM_CLASSES, float(text), dtype=np.float32)
    for item in text.split(","):
        name, value = item.split("=", 1)
        name = name.strip()
        if name not in DAMAGE_CLASSES:
            raise ValueError(f"Unknown class in thresholds: '{name}'.")
        thr[DAMAGE_CLASSES.index(name)] = float(value)
    if np.any((thr <= 0) | (thr > 1)):
        raise ValueError("Thresholds must be in (0, 1].")
    return thr


@dataclass
class FusedChunk:
    score: np.ndarray        # (n, C) float32, NaN where count == 0
    labels: np.ndarray       # (n, C) bool
    label_mask: np.ndarray   # (n,) uint8 bitmask, bit c = class c
    state: np.ndarray        # (n,) uint8
    dominant: np.ndarray     # (n,) int8, class used for the color (-1 if none)
    rgb: np.ndarray          # (n, 3) uint8


def fuse(acc: ChunkAccumulator, thresholds: np.ndarray, min_views: int) -> FusedChunk:
    count = acc.count.astype(np.int64)
    with np.errstate(invalid="ignore", divide="ignore"):
        score = np.where(count > 0, acc.sum / np.maximum(count, 1), np.nan).astype(np.float32)
    enough = count >= min_views
    labels = enough & (np.nan_to_num(score, nan=-1.0) >= thresholds[None, :])
    label_mask = (labels.astype(np.uint8) << np.arange(NUM_CLASSES, dtype=np.uint8)[None, :]).sum(axis=1).astype(np.uint8)

    state = np.full(len(score), STATE_INSUFFICIENT_VIEWS, dtype=np.uint8)
    state[acc.n_views == 0] = STATE_NOT_OBSERVED
    state[enough.all(axis=1) & ~labels.any(axis=1)] = STATE_OBSERVED_NO_DAMAGE
    state[labels.any(axis=1)] = STATE_DAMAGE

    # Color only: dominant positive class = largest score / threshold ratio.
    ratio = np.where(labels, np.nan_to_num(score, nan=0.0) / thresholds[None, :], -np.inf)
    dominant = np.where(labels.any(axis=1), ratio.argmax(axis=1), -1).astype(np.int8)

    palette = np.array([CLASS_COLORS[c] for c in DAMAGE_CLASSES], dtype=np.uint8)
    rgb = np.empty((len(score), 3), dtype=np.uint8)
    for s, color in STATE_COLORS.items():
        rgb[state == s] = color
    dmg = state == STATE_DAMAGE
    rgb[dmg] = palette[dominant[dmg]]
    return FusedChunk(score=score, labels=labels, label_mask=label_mask, state=state, dominant=dominant, rgb=rgb)
