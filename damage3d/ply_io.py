"""Streaming PLY input/output.

Input
-----
The header is parsed at runtime (the prototype hard-coded the layout
``x,y,z,nx,ny,nz float32 + class uint8``; here it is read from the file and
must contain x, y, z). Only ``binary_little_endian 1.0`` with a first
``vertex`` element of scalar properties is supported, like the prototype.
Records are accessed through a read-only ``numpy.memmap``: the file is never
loaded entirely and never modified; only the pages of the requested records
are read.

Point selection
---------------
``PointSelection`` is either the full range [0, N) (no index array is ever
materialised) or a sorted array of source indices obtained with a documented
rule (``stride`` or seeded ``random``), saved in the run directory with its
SHA-256 so that each output point is traceable to its source record.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

PLY_TYPES = {
    "char": "i1", "int8": "i1", "uchar": "u1", "uint8": "u1",
    "short": "<i2", "int16": "<i2", "ushort": "<u2", "uint16": "<u2",
    "int": "<i4", "int32": "<i4", "uint": "<u4", "uint32": "<u4",
    "float": "<f4", "float32": "<f4", "double": "<f8", "float64": "<f8",
}


@dataclass(frozen=True)
class PlyHeader:
    format: str
    vertex_count: int
    properties: tuple[tuple[str, str], ...]
    header_size: int
    other_elements: tuple[str, ...]
    lines: tuple[str, ...]

    @property
    def dtype(self) -> np.dtype:
        return np.dtype([(name, PLY_TYPES[t]) for name, t in self.properties])


def read_ply_header(path: Path) -> PlyHeader:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"PLY not found: {path}")
    lines: list[str] = []
    with path.open("rb") as f:
        first = f.readline()
        if first.strip() != b"ply":
            raise RuntimeError(f"{path} is not a PLY file.")
        lines.append("ply")
        while True:
            line = f.readline()
            if not line:
                raise RuntimeError("Incomplete PLY header.")
            text = line.decode("ascii", errors="replace").strip()
            lines.append(text)
            if text == "end_header":
                break
        header_size = f.tell()

    fmt = next((l for l in lines if l.startswith("format ")), "")
    if fmt != "format binary_little_endian 1.0":
        raise RuntimeError(f"Only binary_little_endian PLY is supported, found '{fmt}'.")

    elements: list[tuple[str, int, list[tuple[str, str]]]] = []
    for line in lines:
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "element":
            elements.append((parts[1], int(parts[2]), []))
        elif parts[0] == "property":
            if not elements:
                raise RuntimeError("PLY property before any element.")
            if parts[1] == "list":
                if elements[-1][0] == "vertex":
                    raise RuntimeError("List properties in the vertex element are not supported.")
                continue
            if parts[1] not in PLY_TYPES:
                raise RuntimeError(f"Unsupported PLY type '{parts[1]}'.")
            elements[-1][2].append((parts[2], parts[1]))
    if not elements or elements[0][0] != "vertex":
        raise RuntimeError("The first PLY element must be 'vertex'.")
    props = tuple(elements[0][2])
    names = [p[0] for p in props]
    for axis in ("x", "y", "z"):
        if axis not in names:
            raise RuntimeError(f"PLY vertex element has no '{axis}' property.")
    return PlyHeader(
        format=fmt, vertex_count=elements[0][1], properties=props, header_size=header_size,
        other_elements=tuple(e[0] for e in elements[1:]), lines=tuple(lines),
    )


class PlyPointSource:
    """Read-only random/sequential access to the vertex records of a PLY."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.header = read_ply_header(self.path)
        itemsize = self.header.dtype.itemsize
        expected_min = self.header.header_size + self.header.vertex_count * itemsize
        size = self.path.stat().st_size
        if size < expected_min:
            raise RuntimeError(
                f"PLY shorter than declared: {size} bytes < {expected_min} "
                f"({self.header.vertex_count} x {itemsize} B + header)."
            )
        self.trailing_bytes = size - expected_min
        self._mm = np.memmap(self.path, dtype=self.header.dtype, mode="r",
                             offset=self.header.header_size, shape=(self.header.vertex_count,))

    @property
    def n_points(self) -> int:
        return self.header.vertex_count

    def xyz(self, index) -> np.ndarray:
        """float64 (n,3) coordinates for a slice or a sorted index array."""
        rec = self._mm[index]
        return np.column_stack((rec["x"], rec["y"], rec["z"])).astype(np.float64)

    def close(self) -> None:
        mm = getattr(self, "_mm", None)
        if mm is not None and getattr(mm, "_mmap", None) is not None:
            mm._mmap.close()
        self._mm = None


class PointSelection:
    """Full range or explicit sorted source indices."""

    def __init__(self, n_source: int, indices: np.ndarray | None, mode: str, seed: int | None):
        self.n_source = int(n_source)
        self.indices = None if indices is None else np.asarray(indices, dtype=np.int64)
        self.mode = mode
        self.seed = seed

    @property
    def size(self) -> int:
        return self.n_source if self.indices is None else int(self.indices.size)

    def source_index(self, start: int, stop: int):
        """Index object usable on the memmap for selection positions [start, stop)."""
        if self.indices is None:
            return slice(start, stop)
        return self.indices[start:stop]

    def source_index_array(self, start: int, stop: int) -> np.ndarray:
        if self.indices is None:
            return np.arange(start, stop, dtype=np.int64)
        return self.indices[start:stop]

    def sha256(self) -> str:
        if self.indices is None:
            return hashlib.sha256(f"full:{self.n_source}".encode()).hexdigest()
        return hashlib.sha256(self.indices.tobytes()).hexdigest()

    def describe(self) -> dict:
        return {"mode": self.mode, "seed": self.seed, "n_source": self.n_source,
                "n_selected": self.size, "indices_sha256": self.sha256()}


def make_selection(n_source: int, max_points: int | None, mode: str = "random", seed: int = 0) -> PointSelection:
    """Reproducible point subset.

    * ``max_points`` None or >= n_source: every point, in file order.
    * ``stride``: indices floor(k * n_source / max_points), k = 0..max_points-1.
    * ``random``: ``np.random.default_rng(seed).choice(n_source, max_points,
      replace=False)`` sorted ascending (numpy Generator, stable across runs
      for the same numpy major version; the SHA-256 in the manifest detects
      any difference).
    """
    if max_points is None or max_points >= n_source:
        return PointSelection(n_source, None, "all", None)
    if max_points <= 0:
        raise ValueError("--max-points must be > 0.")
    if mode == "stride":
        idx = (np.arange(max_points, dtype=np.int64) * n_source) // max_points
        return PointSelection(n_source, idx, "stride", None)
    if mode == "random":
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(n_source, size=max_points, replace=False).astype(np.int64))
        return PointSelection(n_source, idx, "random", seed)
    raise ValueError(f"Unknown sampling mode '{mode}'.")


# --------------------------------------------------------------------------- #
# Output
# --------------------------------------------------------------------------- #
NUMPY_TO_PLY = {
    np.dtype("i1"): "char", np.dtype("u1"): "uchar", np.dtype("<i2"): "short",
    np.dtype("<u2"): "ushort", np.dtype("<i4"): "int", np.dtype("<u4"): "uint",
    np.dtype("<f4"): "float", np.dtype("<f8"): "double",
}


class PlyStreamWriter:
    """Write a binary little-endian PLY whose vertex count is known in advance."""

    def __init__(self, path: Path, dtype: np.dtype, vertex_count: int, comments: list[str] = ()):
        self.path = Path(path)
        self.dtype = np.dtype(dtype)
        self.vertex_count = int(vertex_count)
        self.written = 0
        self._f = self.path.open("wb")
        header = ["ply", "format binary_little_endian 1.0"]
        header += [f"comment {c}" for c in comments]
        header.append(f"element vertex {self.vertex_count}")
        for name in self.dtype.names:
            header.append(f"property {NUMPY_TO_PLY[self.dtype[name].newbyteorder('<')]} {name}")
        header.append("end_header")
        self._f.write(("\n".join(header) + "\n").encode("ascii"))

    def write(self, records: np.ndarray) -> None:
        records = np.asarray(records, dtype=self.dtype)
        self._f.write(records.tobytes())
        self.written += len(records)

    def close(self) -> None:
        self._f.close()
        if self.written != self.vertex_count:
            raise RuntimeError(f"PLY writer: declared {self.vertex_count} vertices, wrote {self.written}.")
