"""Reading the six probabilities at projected pixel positions.

Coordinates: (u, v) in the Metashape corner convention of the full image
(W x H). For a map of size (Wm, Hm) the continuous map coordinates are
xm = u * Wm / W, ym = v * Hm / H; the centre of map pixel (i, j) is (i + 0.5, j + 0.5).

* ``nearest``: pixel (floor(xm), floor(ym)). The prototype used rint(u); with
  the Metashape convention floor() is the pixel containing the point.
* ``bilinear``: interpolation between the 4 surrounding pixel centres
  (xm - 0.5, ym - 0.5). Borders: in the outer half-pixel band the indices are
  clamped (edge replication); points outside [0, W) x [0, H) never reach this
  function.
* NaN in a channel propagates (bilinear) and marks the class invalid for this view.
* ``valid`` mask: the view is valid only if every tap used is valid.
"""

from __future__ import annotations

import numpy as np

from .providers import ProbabilityMap


def sample_probabilities(pmap: ProbabilityMap, u: np.ndarray, v: np.ndarray, width: int, height: int,
                         mode: str = "bilinear") -> tuple[np.ndarray, np.ndarray]:
    """Return (probs (N, C) float32, view_valid (N,) bool)."""
    probs = pmap.probs
    c, hm, wm = probs.shape
    xm = np.asarray(u, dtype=np.float64) * (wm / width)
    ym = np.asarray(v, dtype=np.float64) * (hm / height)
    n = xm.size
    if n == 0:
        return np.empty((0, c), dtype=np.float32), np.empty(0, dtype=bool)

    if mode == "nearest":
        ix = np.clip(np.floor(xm).astype(np.int64), 0, wm - 1)
        iy = np.clip(np.floor(ym).astype(np.int64), 0, hm - 1)
        out = probs[:, iy, ix].T.astype(np.float32)
        valid = np.ones(n, dtype=bool) if pmap.valid is None else pmap.valid[iy, ix]
        return out, valid

    if mode != "bilinear":
        raise ValueError(f"Unknown interpolation '{mode}'.")
    fx = xm - 0.5
    fy = ym - 0.5
    x0 = np.floor(fx).astype(np.int64)
    y0 = np.floor(fy).astype(np.int64)
    wx = (fx - x0).astype(np.float32)
    wy = (fy - y0).astype(np.float32)
    x0c = np.clip(x0, 0, wm - 1)
    x1c = np.clip(x0 + 1, 0, wm - 1)
    y0c = np.clip(y0, 0, hm - 1)
    y1c = np.clip(y0 + 1, 0, hm - 1)
    p00 = probs[:, y0c, x0c]
    p01 = probs[:, y0c, x1c]
    p10 = probs[:, y1c, x0c]
    p11 = probs[:, y1c, x1c]
    out = ((1 - wy) * ((1 - wx) * p00 + wx * p01) + wy * ((1 - wx) * p10 + wx * p11)).T.astype(np.float32)
    if pmap.valid is None:
        valid = np.ones(n, dtype=bool)
    else:
        vm = pmap.valid
        valid = vm[y0c, x0c] & vm[y0c, x1c] & vm[y1c, x0c] & vm[y1c, x1c]
    return out, valid
