"""Parsing of the Metashape cameras XML export.

The camera/sensor lookup and the adjusted-calibration reading reproduce
``read_camera_and_calibration`` from the verified scripts
(verify_marker_projection.py / project_test_mask.py):

* camera matched by ``Path(label).stem``;
* ``<camera><transform>`` = 16 values, row-major, camera -> chunk (internal) frame;
* calibration = ``<sensor><calibration class="adjusted">``;
* missing coefficients default to 0.0;
* non-zero k4, b1, b2, p3, p4 are rejected (the verified model does not use them).

Additions (not present in the prototype, documented here):

* cameras inside ``<group>`` elements are also found;
* cameras without ``<transform>`` (not aligned) or with enabled="false"/"0"
  are listed but not selectable;
* the chunk transform (internal -> world similarity) is parsed. It is NOT used
  by the verified marker test (the marker was given in chunk coordinates); it
  is needed only if the PLY/OBJ were exported in the transformed frame. See
  ``damage3d.frames``.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

SUPPORTED_COEFFICIENTS = ("f", "cx", "cy", "k1", "k2", "k3", "p1", "p2")
UNSUPPORTED_COEFFICIENTS = ("k4", "b1", "b2", "p3", "p4")


@dataclass(frozen=True)
class Calibration:
    sensor_id: str
    sensor_type: str
    width: int
    height: int
    f: float
    cx: float
    cy: float
    k1: float
    k2: float
    k3: float
    p1: float
    p2: float
    unsupported_nonzero: tuple[str, ...] = ()

    def as_dict(self) -> dict:
        """Dictionary with the keys used by the verified prototype functions."""
        return {
            "width": self.width, "height": self.height, "f": self.f,
            "cx": self.cx, "cy": self.cy, "k1": self.k1, "k2": self.k2,
            "k3": self.k3, "p1": self.p1, "p2": self.p2,
        }


@dataclass(frozen=True)
class CameraEntry:
    camera_id: str
    label: str
    stem: str
    sensor_id: str
    enabled: bool
    camera_to_chunk: np.ndarray | None  # 4x4, None if not aligned

    @property
    def aligned(self) -> bool:
        return self.camera_to_chunk is not None


@dataclass(frozen=True)
class ChunkTransform:
    matrix: np.ndarray            # 4x4 internal(chunk) -> world
    source: str                   # "rotation/translation/scale", "matrix" or "absent"
    scale: float

    @property
    def is_identity(self) -> bool:
        return bool(np.allclose(self.matrix, np.eye(4), rtol=0.0, atol=1e-12))


@dataclass
class MetashapeCameras:
    path: Path
    cameras: list[CameraEntry]
    calibrations: dict[str, Calibration]
    chunk_transform: ChunkTransform
    reference_crs: str | None
    n_chunks: int
    warnings: list[str] = field(default_factory=list)

    def calibration_for(self, camera: CameraEntry) -> Calibration:
        if camera.sensor_id not in self.calibrations:
            raise RuntimeError(
                f"Sensor {camera.sensor_id} (camera {camera.label}) has no adjusted calibration."
            )
        return self.calibrations[camera.sensor_id]


def _floats(text: str | None) -> np.ndarray:
    if text is None:
        return np.empty(0)
    return np.array([float(v) for v in text.split()], dtype=np.float64)


def _parse_chunk_transform(chunk: ET.Element) -> ChunkTransform:
    element = chunk.find("./transform")
    if element is None:
        return ChunkTransform(np.eye(4), "absent", 1.0)

    matrix_el = element.find("matrix")
    if matrix_el is not None and matrix_el.text:
        values = _floats(matrix_el.text)
        if values.size != 16:
            raise ValueError(f"Chunk transform matrix: expected 16 values, got {values.size}.")
        matrix = values.reshape(4, 4)
        scale = float(np.cbrt(abs(np.linalg.det(matrix[:3, :3]))))
        return ChunkTransform(matrix, "matrix", scale)

    rotation = _floats(element.findtext("rotation"))
    translation = _floats(element.findtext("translation"))
    scale_values = _floats(element.findtext("scale"))
    if rotation.size == 0 and translation.size == 0 and scale_values.size == 0:
        return ChunkTransform(np.eye(4), "absent", 1.0)

    rot = rotation.reshape(3, 3) if rotation.size == 9 else np.eye(3)
    if rotation.size not in (0, 9):
        raise ValueError(f"Chunk rotation: expected 9 values, got {rotation.size}.")
    if translation.size not in (0, 3):
        raise ValueError(f"Chunk translation: expected 3 values, got {translation.size}.")
    trans = translation if translation.size == 3 else np.zeros(3)
    scale = float(scale_values[0]) if scale_values.size else 1.0

    matrix = np.eye(4)
    matrix[:3, :3] = scale * rot
    matrix[:3, 3] = trans
    return ChunkTransform(matrix, "rotation/translation/scale", scale)


def _parse_calibration(sensor: ET.Element) -> Calibration | None:
    calibration = next(
        (item for item in sensor.findall("calibration") if item.get("class") == "adjusted"),
        None,
    )
    if calibration is None:
        return None
    resolution = calibration.find("resolution")
    if resolution is None:
        raise RuntimeError(f"Sensor {sensor.get('id')}: calibration resolution not found.")

    def coefficient(name: str, default: float = 0.0) -> float:
        element = calibration.find(name)
        return float(element.text) if element is not None and element.text else default

    unsupported = tuple(n for n in UNSUPPORTED_COEFFICIENTS if abs(coefficient(n)) > 1e-12)
    return Calibration(
        sensor_id=str(sensor.get("id")),
        sensor_type=str(sensor.get("type", calibration.get("type", "frame"))),
        width=int(resolution.get("width")),
        height=int(resolution.get("height")),
        f=coefficient("f"), cx=coefficient("cx"), cy=coefficient("cy"),
        k1=coefficient("k1"), k2=coefficient("k2"), k3=coefficient("k3"),
        p1=coefficient("p1"), p2=coefficient("p2"),
        unsupported_nonzero=unsupported,
    )


def _is_enabled(element: ET.Element) -> bool:
    value = element.get("enabled")
    return value is None or value.strip().lower() not in ("false", "0")


def parse_cameras_xml(path: Path) -> MetashapeCameras:
    """Parse the Metashape XML export (read only)."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Cameras XML not found: {path}")

    root = ET.parse(path).getroot()
    chunks = root.findall(".//chunk")
    if not chunks:
        raise RuntimeError("No <chunk> element found in the camera XML.")
    chunk = chunks[0]
    warnings: list[str] = []
    if len(chunks) > 1:
        warnings.append(f"{len(chunks)} chunks found; only the first one is used (as in the prototype).")

    calibrations: dict[str, Calibration] = {}
    for sensor in chunk.findall(".//sensors/sensor"):
        calib = _parse_calibration(sensor)
        if calib is not None:
            calibrations[calib.sensor_id] = calib

    cameras: list[CameraEntry] = []
    cameras_el = chunk.find("cameras")
    if cameras_el is None:
        raise RuntimeError("No <cameras> element found in the chunk.")
    for item in cameras_el.iter("camera"):
        label = item.get("label", "")
        transform_el = item.find("transform")
        matrix = None
        if transform_el is not None and transform_el.text and transform_el.text.strip():
            values = np.fromstring(transform_el.text, sep=" ", dtype=np.float64)
            if values.size != 16:
                raise ValueError(f"Camera {label}: expected 16 transform values, got {values.size}.")
            matrix = values.reshape(4, 4)
        cameras.append(CameraEntry(
            camera_id=str(item.get("id")),
            label=label,
            stem=Path(label).stem,
            sensor_id=str(item.get("sensor_id")),
            enabled=_is_enabled(item),
            camera_to_chunk=matrix,
        ))
        if item.find("rolling_shutter") is not None:
            warnings.append(f"Camera {label} has rolling-shutter data, ignored by the frame model.")

    reference = chunk.find("./reference")
    reference_crs = reference.text.strip() if reference is not None and reference.text else None

    return MetashapeCameras(
        path=path,
        cameras=cameras,
        calibrations=calibrations,
        chunk_transform=_parse_chunk_transform(chunk),
        reference_crs=reference_crs,
        n_chunks=len(chunks),
        warnings=warnings,
    )


# --------------------------------------------------------------------------- #
# Deterministic ordering and camera selection
# --------------------------------------------------------------------------- #
def natural_key(text: str) -> tuple:
    """Natural sort key: DSC2 < DSC10; ties broken by the raw string."""
    parts = re.split(r"(\d+)", text)
    return tuple((0, int(p)) if p.isdigit() else (1, p.lower()) for p in parts if p != "")


def selectable_cameras(project: MetashapeCameras) -> list[CameraEntry]:
    """Aligned and enabled cameras in deterministic order (index space of --start/--end)."""
    usable = [c for c in project.cameras if c.aligned and c.enabled]
    return sorted(usable, key=lambda c: (natural_key(c.stem), c.label, natural_key(c.camera_id)))


class SelectionError(ValueError):
    pass


def read_camera_list(path: Path) -> list[str]:
    names = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            names.append(line)
    return names


def select_cameras(
    ordered: list[CameraEntry],
    all_cameras: list[CameraEntry],
    camera: list[str] | None = None,
    camera_list: list[str] | None = None,
    start_image: int | None = None,
    end_image: int | None = None,
    limit_images: int | None = None,
) -> list[tuple[int, CameraEntry]]:
    """Apply the selection rules; return (position in ``ordered``, camera) pairs.

    Rules
    -----
    1. Name selectors (--camera, --camera-list) and range selectors
       (--start-image/--end-image) are mutually exclusive; --camera and
       --camera-list are mutually exclusive too. Ambiguous combinations raise.
    2. Range: positions in the deterministic ordered list, 0-based,
       ``start`` inclusive, ``end`` EXCLUSIVE (Python slice semantics).
       Out-of-range values raise instead of being clipped silently.
    3. --limit-images is applied last and keeps the first N selected cameras.
    4. The result is always returned in the deterministic global order.
    """
    by_name = camera is not None or camera_list is not None
    by_range = start_image is not None or end_image is not None
    if camera is not None and camera_list is not None:
        raise SelectionError("--camera and --camera-list cannot be combined.")
    if by_name and by_range:
        raise SelectionError("--camera/--camera-list cannot be combined with --start-image/--end-image.")

    n = len(ordered)
    if n == 0:
        raise SelectionError("No aligned and enabled cameras in the XML.")
    positions: list[int]

    if by_name:
        requested = list(camera or []) + list(camera_list or [])
        seen: set[str] = set()
        positions_set: set[int] = set()
        missing, unusable = [], []
        for name in requested:
            key = Path(name).stem if Path(name).suffix.lower() in (".jpg", ".jpeg", ".png", ".tif", ".tiff") else name
            if key in seen:
                raise SelectionError(f"Camera '{name}' requested more than once.")
            seen.add(key)
            matches = [i for i, c in enumerate(ordered) if key in (c.stem, c.label)]
            if len(matches) > 1:
                raise SelectionError(f"Camera '{name}' is ambiguous: {[ordered[i].label for i in matches]}")
            if not matches:
                if any(key in (c.stem, c.label) for c in all_cameras):
                    unusable.append(name)
                else:
                    missing.append(name)
                continue
            positions_set.add(matches[0])
        if missing or unusable:
            raise SelectionError(
                f"Cameras not in the XML: {missing}; present but not aligned/disabled: {unusable}"
            )
        positions = sorted(positions_set)
    else:
        start = 0 if start_image is None else int(start_image)
        end = n if end_image is None else int(end_image)
        if start < 0 or end < 0:
            raise SelectionError("--start-image/--end-image must be >= 0.")
        if start >= n:
            raise SelectionError(f"--start-image {start} >= number of selectable cameras ({n}).")
        if end > n:
            raise SelectionError(f"--end-image {end} > number of selectable cameras ({n}); end is exclusive.")
        if end <= start:
            raise SelectionError(f"Empty range: --start-image {start} --end-image {end} (end is exclusive).")
        positions = list(range(start, end))

    if limit_images is not None:
        if limit_images <= 0:
            raise SelectionError("--limit-images must be > 0.")
        positions = positions[:limit_images]
    return [(p, ordered[p]) for p in positions]
