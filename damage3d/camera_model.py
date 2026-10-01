"""Metashape frame-camera projection.

The equations are the ones of ``project_chunk_point`` in the verified script
``verify_marker_projection.py`` (marker on DSC02150, reported difference
0.000000 px against Metashape). ``project_point_reference`` is a verbatim
copy used by the regression tests; ``CameraProjector`` is the vectorised
equivalent used by the pipeline.

Frames and conventions
----------------------
* Input points are in the chunk (internal) frame of Metashape.
* ``camera_to_chunk`` is ``<camera><transform>``; camera frame: X right,
  Y down, Z along the viewing direction (Metashape manual, Appendix C).
* Normalised coordinates x = X/Z, y = Y/Z; Brown radial (k1..k3) and
  tangential (p1, p2, Metashape ordering) distortion; then
  u = W/2 + cx + f*x', v = H/2 + cy + f*y'.
* Pixel coordinates use the Metashape corner convention: the image spans
  [0, W) x [0, H) and the centre of pixel (i, j) is at (i + 0.5, j + 0.5).

Added guard (not in the prototype)
----------------------------------
The polynomial radial model is not monotonic for large radii: points far
outside the field of view can "fold back" into the image. Points whose
undistorted radius exceeds ``r_limit`` are rejected, where ``r_limit`` is the
smaller of (a) the first radius where d(r*(1+k1 r^2+k2 r^4+k3 r^6))/dr <= 0 and
(b) 1.05 x the undistorted radius of the farthest image corner.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .metashape_xml import Calibration


def project_point_reference(point_xyz, camera_to_chunk, c):
    """Verbatim copy of the verified prototype ``project_chunk_point``."""
    point_h = np.array([*point_xyz, 1.0], dtype=np.float64)
    camera_xyz = np.linalg.solve(camera_to_chunk, point_h)[:3]

    if camera_xyz[2] <= 0:
        raise ValueError("Point is behind the camera.")

    x = camera_xyz[0] / camera_xyz[2]
    y = camera_xyz[1] / camera_xyz[2]
    r2 = x * x + y * y

    radial = (
        1.0
        + c["k1"] * r2
        + c["k2"] * r2**2
        + c["k3"] * r2**3
    )

    x_distorted = (
        x * radial
        + c["p1"] * (r2 + 2 * x * x)
        + 2 * c["p2"] * x * y
    )
    y_distorted = (
        y * radial
        + c["p2"] * (r2 + 2 * y * y)
        + 2 * c["p1"] * x * y
    )

    u = c["width"] / 2 + c["cx"] + c["f"] * x_distorted
    v = c["height"] / 2 + c["cy"] + c["f"] * y_distorted
    return np.array([u, v]), camera_xyz


def radial_limit(calib: Calibration, margin: float = 0.05, r_scan_max: float = 4.0, steps: int = 40001) -> float:
    """Largest undistorted normalised radius accepted by the projector (see module doc)."""
    r = np.linspace(0.0, r_scan_max, steps)
    r2 = r * r
    g = r * (1.0 + calib.k1 * r2 + calib.k2 * r2**2 + calib.k3 * r2**3)
    dg = 1.0 + 3 * calib.k1 * r2 + 5 * calib.k2 * r2**2 + 7 * calib.k3 * r2**3
    non_monotonic = np.nonzero(dg <= 0)[0]
    last_mono = int(non_monotonic[0]) - 1 if non_monotonic.size else steps - 1
    r_mono = float(r[max(last_mono, 1)])

    corners_u = np.array([0.0, calib.width, 0.0, calib.width])
    corners_v = np.array([0.0, 0.0, calib.height, calib.height])
    xd = (corners_u - calib.width / 2 - calib.cx) / calib.f
    yd = (corners_v - calib.height / 2 - calib.cy) / calib.f
    rd_corner = float(np.max(np.hypot(xd, yd)))
    reach = np.nonzero(g[: last_mono + 1] >= rd_corner)[0]
    if reach.size == 0:
        return r_mono
    return float(min(r_mono, r[int(reach[0])] * (1.0 + margin)))


@dataclass
class Projection:
    u: np.ndarray            # (N,) float64 pixel column, corner convention
    v: np.ndarray            # (N,) float64 pixel row
    depth: np.ndarray        # (N,) camera Z
    in_front: np.ndarray     # (N,) bool, Z > min_depth
    in_fov: np.ndarray       # (N,) bool, radial guard passed
    in_image: np.ndarray     # (N,) bool, in_front & in_fov & inside [margin, W-margin) x [margin, H-margin)


class CameraProjector:
    """Vectorised projection for one camera (chunk-frame points)."""

    def __init__(self, camera_to_chunk: np.ndarray, calib: Calibration, min_depth: float = 0.0,
                 border_margin_px: float = 0.0):
        if calib.unsupported_nonzero:
            raise RuntimeError(
                f"Additional non-zero calibration parameters: {list(calib.unsupported_nonzero)}. "
                "Extend the projection model before using it."
            )
        if calib.sensor_type not in ("frame", ""):
            raise RuntimeError(f"Unsupported sensor type '{calib.sensor_type}' (only 'frame').")
        self.camera_to_chunk = np.asarray(camera_to_chunk, dtype=np.float64)
        self.chunk_to_camera = np.linalg.inv(self.camera_to_chunk)
        self.calib = calib
        self.min_depth = float(min_depth)
        self.border_margin_px = float(border_margin_px)
        self.r_limit = radial_limit(calib)
        self.center_chunk = self.camera_to_chunk[:3, 3].copy()

    def to_camera(self, points_chunk: np.ndarray) -> np.ndarray:
        m = self.chunk_to_camera
        return points_chunk @ m[:3, :3].T + m[:3, 3]

    def project(self, points_chunk: np.ndarray) -> Projection:
        c = self.calib
        cam = self.to_camera(np.asarray(points_chunk, dtype=np.float64))
        z = cam[:, 2]
        in_front = z > self.min_depth
        with np.errstate(divide="ignore", invalid="ignore"):
            safe_z = np.where(in_front, z, 1.0)
            x = cam[:, 0] / safe_z
            y = cam[:, 1] / safe_z
        r2 = x * x + y * y
        radial = 1.0 + c.k1 * r2 + c.k2 * r2**2 + c.k3 * r2**3
        x_d = x * radial + c.p1 * (r2 + 2 * x * x) + 2 * c.p2 * x * y
        y_d = y * radial + c.p2 * (r2 + 2 * y * y) + 2 * c.p1 * x * y
        u = c.width / 2 + c.cx + c.f * x_d
        v = c.height / 2 + c.cy + c.f * y_d
        in_fov = in_front & (r2 <= self.r_limit**2)
        m = self.border_margin_px
        in_image = in_fov & (u >= m) & (u < c.width - m) & (v >= m) & (v < c.height - m)
        return Projection(u=u, v=v, depth=z, in_front=in_front, in_fov=in_fov, in_image=in_image)

    def sphere_may_be_visible(self, center_chunk: np.ndarray, radius: float) -> bool:
        """Conservative frustum test of a bounding sphere against the FOV cone.

        Returns False only if the sphere is certainly behind the camera or
        entirely outside the cone of half-angle atan(r_limit).
        """
        cc = self.to_camera(np.asarray(center_chunk, dtype=np.float64)[None, :])[0]
        if cc[2] + radius <= self.min_depth:
            return False
        alpha = np.arctan(self.r_limit)
        lateral = float(np.hypot(cc[0], cc[1]))
        signed_distance = lateral * np.cos(alpha) - cc[2] * np.sin(alpha)
        return not (signed_distance > radius)
