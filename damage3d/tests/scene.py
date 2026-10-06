"""Synthetic Metashape-like project used by the tests (no real data needed).

Chunk (internal) frame: a deck plane z = 0 (x in [-5, 5], y in [-2, 2]) and a
thin occluding plate 0.95 <= z <= 1.05 over |x|, |y| <= 1. Cameras look
straight down from z = 6. PLY and OBJ are written in the TRANSFORMED frame
(world = s * R * chunk + t) when ``chunk_transform=True``, as a Metashape
export with a chunk transform would be. The PLY header reproduces the
layout of the real export (x, y, z, nx, ny, nz float + class uchar).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from damage3d.camera_model import project_point_reference

WIDTH, HEIGHT = 640, 480
CALIB = {"f": 500.0, "cx": 3.5, "cy": -2.25, "k1": -0.05, "k2": 0.01, "k3": 0.0, "p1": 1e-4, "p2": -2e-4}
CHUNK_ROT_DEG = 30.0
CHUNK_SCALE = 2.0
CHUNK_T = np.array([100.0, 200.0, 50.0])
CAMERA_X = [-3.0, -1.5, 0.0, 1.5, 3.0]
MARKER_CHUNK = np.array([0.3, 1.4, 0.0])


def chunk_matrix() -> np.ndarray:
    a = np.deg2rad(CHUNK_ROT_DEG)
    r = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    m = np.eye(4)
    m[:3, :3] = CHUNK_SCALE * r
    m[:3, 3] = CHUNK_T
    return m, r


def camera_to_chunk(x: float, tilt_deg: float = 0.0) -> np.ndarray:
    X, Y, Z = np.array([1.0, 0, 0]), np.array([0, -1.0, 0]), np.array([0, 0, -1.0])
    t = np.deg2rad(tilt_deg)  # rotation about the camera X axis
    rot = np.column_stack((X, Y, Z))
    tilt = np.array([[1, 0, 0], [0, np.cos(t), -np.sin(t)], [0, np.sin(t), np.cos(t)]])
    rot = rot @ tilt
    m = np.eye(4)
    m[:3, :3] = rot
    m[:3, 3] = [x, 0.2, 6.0]
    return m


def calib_dict() -> dict:
    return {"width": WIDTH, "height": HEIGHT, **CALIB}


def _box(lo, hi):
    x0, y0, z0 = lo
    x1, y1, z1 = hi
    v = np.array([[x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
                  [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]])
    f = np.array([[0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7], [0, 1, 5], [0, 5, 4],
                  [1, 2, 6], [1, 6, 5], [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7]])
    return v, f


def mesh_chunk():
    deck_v = np.array([[-5, -2, 0], [5, -2, 0], [5, 2, 0], [-5, 2, 0]], dtype=float)
    deck_f = np.array([[0, 1, 2], [0, 2, 3]])
    box_v, box_f = _box((-1, -1, 0.95), (1, 1, 1.05))
    return np.vstack((deck_v, box_v)), np.vstack((deck_f, box_f + 4))


def points_chunk(n_deck: int = 16000, n_top: int = 4000, seed: int = 1):
    rng = np.random.default_rng(seed)
    deck = np.column_stack((rng.uniform(-5, 5, n_deck), rng.uniform(-2, 2, n_deck), np.zeros(n_deck)))
    top = np.column_stack((rng.uniform(-1, 1, n_top), rng.uniform(-1, 1, n_top), np.full(n_top, 1.05)))
    cls = np.concatenate((np.zeros(n_deck, np.uint8), np.ones(n_top, np.uint8)))
    return np.vstack((deck, top)), cls


def write_ply(path: Path, xyz: np.ndarray, cls: np.ndarray) -> None:
    dtype = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("nx", "<f4"), ("ny", "<f4"),
                      ("nz", "<f4"), ("class", "u1")])
    rec = np.zeros(len(xyz), dtype=dtype)
    rec["x"], rec["y"], rec["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    rec["nz"] = 1.0
    rec["class"] = cls
    header = ("ply\nformat binary_little_endian 1.0\ncomment synthetic test cloud\n"
              f"element vertex {len(xyz)}\nproperty float x\nproperty float y\nproperty float z\n"
              "property float nx\nproperty float ny\nproperty float nz\nproperty uchar class\nend_header\n")
    with Path(path).open("wb") as f:
        f.write(header.encode("ascii"))
        f.write(rec.tobytes())


def write_obj(path: Path, v: np.ndarray, f: np.ndarray) -> None:
    lines = [f"v {a:.9f} {b:.9f} {c:.9f}" for a, b, c in v]
    lines += [f"f {a + 1} {b + 1} {c + 1}" for a, b, c in f]
    Path(path).write_text("\n".join(lines) + "\n", encoding="ascii")


def _matrix_text(m: np.ndarray) -> str:
    return " ".join(f"{v:.17e}" for v in m.reshape(-1))


def write_xml(path: Path, chunk_transform: bool, extra_coefficients: dict | None = None) -> None:
    coeff = {**CALIB, **(extra_coefficients or {})}
    calib_xml = "".join(f"<{k}>{float(v)!r}</{k}>" for k, v in coeff.items())
    cams = []
    for i, x in enumerate(CAMERA_X):
        tilt = 5.0 if i % 2 else 0.0
        cam = f'<camera id="{i}" sensor_id="0" label="DSC{1000 + 10 * i:05d}.JPG"><transform>{_matrix_text(camera_to_chunk(x, tilt))}</transform></camera>'
        cams.append(cam)
    group = f'<group id="0" label="g0">{cams.pop()}</group>'  # last camera inside a group
    unaligned = '<camera id="10" sensor_id="0" label="DSC09990.JPG"/>'
    disabled = (f'<camera id="11" sensor_id="0" label="DSC09995.JPG" enabled="false">'
                f'<transform>{_matrix_text(camera_to_chunk(0.0))}</transform></camera>')
    transform = ""
    if chunk_transform:
        _, r = chunk_matrix()
        transform = (f'<transform><rotation locked="false">{" ".join(repr(float(v)) for v in r.reshape(-1))}</rotation>'
                     f'<translation locked="false">{" ".join(repr(float(v)) for v in CHUNK_T)}</translation>'
                     f'<scale locked="true">{CHUNK_SCALE!r}</scale></transform>')
    xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<document version="1.5.0">
  <chunk label="Chunk 1" enabled="true">
    <sensors next_id="1">
      <sensor id="0" label="synthetic" type="frame">
        <resolution width="{WIDTH}" height="{HEIGHT}"/>
        <calibration type="frame" class="adjusted"><resolution width="{WIDTH}" height="{HEIGHT}"/>{calib_xml}</calibration>
      </sensor>
    </sensors>
    <cameras next_id="12" next_group_id="1">
      {''.join(cams)}{group}{unaligned}{disabled}
    </cameras>
    {transform}
  </chunk>
</document>
"""
    Path(path).write_text(xml, encoding="utf-8")


def build_project(root: Path, chunk_transform: bool = True, n_deck: int = 16000, n_top: int = 4000) -> dict:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    m, _ = chunk_matrix()
    to_source = m if chunk_transform else np.eye(4)
    xyz_c, cls = points_chunk(n_deck, n_top)
    xyz_s = xyz_c @ to_source[:3, :3].T + to_source[:3, 3]
    v, f = mesh_chunk()
    write_ply(root / "bridge_3dpc_point_cloud.ply", xyz_s, cls)
    write_obj(root / "model_medium_quality.obj", v @ to_source[:3, :3].T + to_source[:3, 3], f)
    write_xml(root / "cameras.xml", chunk_transform)
    # Marker reference in the verified format (chunk coordinates + Metashape pixel).
    pixel, _ = project_point_reference(MARKER_CHUNK, camera_to_chunk(CAMERA_X[2]), calib_dict())
    (root / "marker_reference.txt").write_text(
        "camera = DSC01020\n"
        f"chunk_x = {float(MARKER_CHUNK[0])!r}\nchunk_y = {float(MARKER_CHUNK[1])!r}\nchunk_z = {float(MARKER_CHUNK[2])!r}\n"
        f"pixel_u = {float(pixel[0])!r}\npixel_v = {float(pixel[1])!r}\n", encoding="utf-8")
    return {"root": root, "n_points": len(xyz_c), "xyz_chunk": xyz_c, "to_source": to_source}
