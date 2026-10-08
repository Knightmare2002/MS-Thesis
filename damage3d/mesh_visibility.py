"""Occlusion test against the Metashape OBJ mesh (Open3D ray casting).

Test
----
For a camera centre C and a candidate point P (both in the source frame of
PLY/OBJ), a ray is cast from C towards P. With t_hit the distance to the first
mesh intersection and d = |P - C|, the point is visible iff

    t_hit >= d - tol,      tol = abs_tol + rel_tol * d.

A miss (t_hit = inf) counts as visible.

Units: source-frame units (the PLY/OBJ units; metres only if the export is
metrically scaled - this is not established by the XML alone).

Numerical precision: Open3D ray casting is float32. The mesh and the rays are
translated by the mesh bounding-box centre before building the scene, so the
float32 error depends on the structure size, not on the absolute coordinates.

When the test is not reliable
-----------------------------
* holes or missing parts in the mesh -> hidden points counted as visible;
* PLY points farther than ``tol`` behind the mesh surface (mesh simplification,
  noise) -> visible points dropped (conservative: fewer observations);
* thin elements thinner than ``tol`` -> back-side points counted as visible;
* grazing rays: the along-ray deviation grows like dev / cos(angle);
* the medium-quality mesh does not coincide exactly with every PLY point
  (see check_surface_alignment.py statistics).

Cost: one ray per (point inside the image) x camera; Embree-based BVH,
roughly O(log T) per ray for T triangles. Rays are processed in batches of
``ray_batch`` to bound memory (~(24 + 4) bytes per ray plus Open3D outputs).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def _o3d():
    try:
        import open3d as o3d  # noqa: WPS433
    except ImportError as exc:  # pragma: no cover
        raise ImportError("open3d is required for mesh occlusion (pip install open3d).") from exc
    return o3d


class MeshOcclusion:
    def __init__(self, mesh_path: Path, ray_batch: int = 2_000_000):
        o3d = _o3d()
        self.mesh_path = Path(mesh_path)
        if not self.mesh_path.is_file():
            raise FileNotFoundError(f"Mesh not found: {self.mesh_path}")
        mesh = o3d.io.read_triangle_mesh(str(self.mesh_path))
        if mesh.is_empty() or not mesh.has_triangles():
            raise RuntimeError("The OBJ doesn't contain a readable triangular mesh.")
        vertices = np.asarray(mesh.vertices, dtype=np.float64)
        triangles = np.asarray(mesh.triangles, dtype=np.uint32)
        self.bbox_min = vertices.min(axis=0)
        self.bbox_max = vertices.max(axis=0)
        self.offset = (self.bbox_min + self.bbox_max) / 2.0
        self.diagonal = float(np.linalg.norm(self.bbox_max - self.bbox_min))
        self.n_vertices = int(len(vertices))
        self.n_triangles = int(len(triangles))
        self.scene = o3d.t.geometry.RaycastingScene()
        self.scene.add_triangles(
            o3d.core.Tensor((vertices - self.offset).astype(np.float32)),
            o3d.core.Tensor(triangles),
        )
        del mesh, vertices, triangles
        self.ray_batch = int(ray_batch)

    def distance(self, points_source: np.ndarray) -> np.ndarray:
        """Unsigned distance of points (source frame) to the mesh surface."""
        o3d = _o3d()
        pts = (np.asarray(points_source, dtype=np.float64) - self.offset).astype(np.float32)
        return self.scene.compute_distance(o3d.core.Tensor(pts)).numpy().astype(np.float64)

    def surface_deviation_stats(self, points_source: np.ndarray) -> dict:
        d = self.distance(points_source)
        d = d[np.isfinite(d)]
        if d.size == 0:
            return {"n": 0}
        return {"n": int(d.size), "median": float(np.median(d)), "p90": float(np.percentile(d, 90)),
                "p95": float(np.percentile(d, 95)), "p99": float(np.percentile(d, 99)),
                "max": float(d.max())}

    def visible(self, points_source: np.ndarray, camera_center_source: np.ndarray,
                abs_tol: float, rel_tol: float = 0.0) -> np.ndarray:
        """Boolean visibility for each point (see module docstring)."""
        o3d = _o3d()
        pts = np.asarray(points_source, dtype=np.float64)
        n = len(pts)
        out = np.zeros(n, dtype=bool)
        if n == 0:
            return out
        center = np.asarray(camera_center_source, dtype=np.float64)
        
        for s in range(0, n, self.ray_batch):
            e = min(s + self.ray_batch, n)
            direction = pts[s:e] - center
            dist = np.linalg.norm(direction, axis=1)
            ok = dist > 0
            unit = np.zeros_like(direction)
            unit[ok] = direction[ok] / dist[ok, None]
            rays = np.empty((e - s, 6), dtype=np.float32)
            rays[:, :3] = (center - self.offset).astype(np.float32)
            rays[:, 3:] = unit.astype(np.float32)
            t_hit = self.scene.cast_rays(o3d.core.Tensor(rays))["t_hit"].numpy().astype(np.float64)
            tol = abs_tol + rel_tol * dist
            out[s:e] = ok & (t_hit >= dist - tol)
        return out
