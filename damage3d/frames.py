"""Relation between the frame of the exported PLY/OBJ ("source frame") and the
Metashape chunk (internal) frame used by the verified projection.

Facts established by the existing scripts
-----------------------------------------
* The projection is verified for points in the chunk frame (marker_reference.txt
  stores chunk_x/chunk_y/chunk_z; 0.000000 px vs Metashape on DSC02150).
* PLY and OBJ share the same frame (check_obj_ply_alignment.py,
  check_surface_alignment.py).

NOT established
---------------
* Whether the PLY/OBJ are in the chunk frame or in the frame obtained with the
  chunk transform (``<chunk><transform>``, internal -> world). The prototype
  project_test_mask.py assumed the chunk frame without checking it, and
  inspect_camera_geometry.py searched ``<transform><matrix>``, whereas Metashape
  XML usually stores rotation/translation/scale, so a present transform may
  have been reported as absent.

Resolution policy (``--point-frame``)
-------------------------------------
* ``chunk``: source = chunk frame.
* ``transformed``: source = chunk transform applied (world = M @ chunk).
* ``auto`` (default): if the chunk transform is absent/identity both are the
  same frame. Otherwise the marker of marker_reference.txt is placed in both
  candidate frames and its distance to the OBJ surface is measured; the
  frame is accepted only if one distance is at least 10x smaller than the
  other. Otherwise the run stops and asks for an explicit --point-frame.
  Export shifts or CRS conversions applied in the Metashape export dialog are
  not represented in the XML and cannot be detected except by this check.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .metashape_xml import ChunkTransform


@dataclass(frozen=True)
class SourceFrame:
    name: str                 # "chunk" or "transformed"
    source_to_chunk: np.ndarray
    chunk_to_source: np.ndarray

    @property
    def is_identity(self) -> bool:
        return bool(np.allclose(self.source_to_chunk, np.eye(4), rtol=0.0, atol=1e-12))

    def to_chunk(self, points_source: np.ndarray) -> np.ndarray:
        m = self.source_to_chunk
        pts = np.asarray(points_source, dtype=np.float64)
        return pts if self.is_identity else pts @ m[:3, :3].T + m[:3, 3]

    def to_source(self, points_chunk: np.ndarray) -> np.ndarray:
        m = self.chunk_to_source
        pts = np.asarray(points_chunk, dtype=np.float64)
        return pts if self.is_identity else pts @ m[:3, :3].T + m[:3, 3]

    def radius_to_chunk(self, radius_source: float) -> float:
        """Conservative radius mapping (largest singular value of the linear part)."""
        return float(radius_source * np.linalg.svd(self.source_to_chunk[:3, :3], compute_uv=False).max())

    def length_to_source(self, length_chunk: float) -> float:
        return float(length_chunk * np.linalg.svd(self.chunk_to_source[:3, :3], compute_uv=False).max())


def make_frame(name: str, chunk_transform: ChunkTransform) -> SourceFrame:
    if name == "chunk":
        return SourceFrame("chunk", np.eye(4), np.eye(4))
    if name == "transformed":
        m = chunk_transform.matrix
        return SourceFrame("transformed", np.linalg.inv(m), m.copy())
    raise ValueError(f"Unknown frame '{name}'.")


def read_marker_reference(path: Path) -> dict[str, str]:
    """Same parser as verify_marker_projection.py (key = value lines)."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    result = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            result[key.strip()] = value.strip()
    return result


def marker_chunk_point(reference: dict[str, str]) -> np.ndarray:
    return np.array([float(reference["chunk_x"]), float(reference["chunk_y"]), float(reference["chunk_z"])])


def resolve_frame(requested: str, chunk_transform: ChunkTransform, marker_path: Path | None,
                  mesh_distance_fn=None, ratio_required: float = 10.0) -> tuple[SourceFrame, dict]:
    """Return the source frame and a JSON-serialisable record of how it was chosen."""
    record: dict = {"requested": requested, "chunk_transform_source": chunk_transform.source,
                    "chunk_transform_identity": chunk_transform.is_identity,
                    "chunk_transform_scale": chunk_transform.scale}
    if requested in ("chunk", "transformed"):
        record["resolved"] = requested
        record["method"] = "explicit CLI choice"
        return make_frame(requested, chunk_transform), record
    if requested != "auto":
        raise ValueError(f"Unknown --point-frame '{requested}'.")
    if chunk_transform.is_identity:
        record["resolved"] = "chunk"
        record["method"] = "chunk transform absent or identity: both frames coincide"
        return make_frame("chunk", chunk_transform), record
    if marker_path is None or not Path(marker_path).is_file():
        raise RuntimeError(
            "The XML contains a non-identity chunk transform, so the PLY frame is ambiguous. "
            f"Missing input: marker reference file ({marker_path}) with chunk_x/chunk_y/chunk_z. "
            "Provide it (--marker-reference) or pass --point-frame chunk|transformed explicitly."
        )
    if mesh_distance_fn is None:
        raise RuntimeError("Automatic frame check needs the OBJ mesh (--occlusion mesh).")
    marker = marker_chunk_point(read_marker_reference(marker_path))
    d_chunk = float(mesh_distance_fn(marker[None, :])[0])
    transformed = (chunk_transform.matrix[:3, :3] @ marker) + chunk_transform.matrix[:3, 3]
    d_transformed = float(mesh_distance_fn(transformed[None, :])[0])
    record.update({"marker_chunk_xyz": marker.tolist(), "marker_to_mesh_if_chunk": d_chunk,
                   "marker_to_mesh_if_transformed": d_transformed, "ratio_required": ratio_required})
    if d_chunk * ratio_required <= d_transformed:
        record["resolved"], record["method"] = "chunk", "marker-to-mesh distance test"
        return make_frame("chunk", chunk_transform), record
    if d_transformed * ratio_required <= d_chunk:
        record["resolved"], record["method"] = "transformed", "marker-to-mesh distance test"
        return make_frame("transformed", chunk_transform), record
    raise RuntimeError(
        f"Frame check inconclusive: marker-to-mesh distance {d_chunk:.6g} (chunk) vs "
        f"{d_transformed:.6g} (transformed). Pass --point-frame explicitly after checking in Metashape."
    )
