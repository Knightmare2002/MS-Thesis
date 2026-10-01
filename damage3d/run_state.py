"""Run directory, fingerprint and resume bookkeeping.

Resume guarantees (process interruption: Ctrl+C, crash, kill)
-------------------------------------------------------------
* Cameras are processed in batches (``--camera-batch-size``) in the
  deterministic camera order.
* Each point chunk has its own accumulator file ``acc/chunk_XXXXXX.npz``
  written with write-to-temp + fsync + ``os.replace`` (atomic on NTFS and
  POSIX), and stores ``last_batch``, the index of the last batch applied.
* When batch b is (re)processed, a chunk with last_batch >= b is skipped,
  so a batch interrupted half-way is completed without adding any camera twice.
* ``progress.json`` (atomic) records the completed batches.
* Before resuming, the stored fingerprint (inputs, selected cameras, point
  selection, accumulation parameters, layout, provider) must match exactly.

Not guaranteed: consistency after power loss/OS crash beyond what the file
system provides for fsync'ed files.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

FINGERPRINT_SECTIONS = ("inputs", "cameras", "points", "accumulation", "layout", "provider")


def sha256_file(path: Path, block: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        while True:
            data = f.read(block)
            if not data:
                break
            h.update(data)
    return h.hexdigest()


def file_identity(path: Path, full_hash: bool = False, partial_bytes: int = 16 << 20) -> dict:
    """size + mtime + SHA-256 (full for small files, head/tail otherwise)."""
    path = Path(path)
    stat = path.stat()
    ident = {"path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    if full_hash or stat.st_size <= 2 * partial_bytes:
        ident["sha256"] = sha256_file(path)
    else:
        h = hashlib.sha256()
        with path.open("rb") as f:
            h.update(f.read(partial_bytes))
            f.seek(-partial_bytes, 2)
            h.update(f.read(partial_bytes))
        ident["sha256_head_tail_16MiB"] = h.hexdigest()
    return ident


def write_json_atomic(path: Path, data) -> None:
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True, default=_json_default)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _json_default(obj):
    import numpy as np
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Not JSON serialisable: {type(obj)}")


def normalise(data):
    """Round-trip through JSON so that comparisons are type-stable."""
    return json.loads(json.dumps(data, sort_keys=True, default=_json_default))


def short_hash(data) -> str:
    return hashlib.sha256(json.dumps(normalise(data), sort_keys=True).encode()).hexdigest()[:8]


def diff_fingerprints(old: dict, new: dict) -> list[str]:
    diffs = []
    for section in FINGERPRINT_SECTIONS:
        a, b = old.get(section), new.get(section)
        if a == b:
            continue
        if isinstance(a, dict) and isinstance(b, dict):
            for key in sorted(set(a) | set(b)):
                if a.get(key) != b.get(key):
                    diffs.append(f"{section}.{key}")
        else:
            diffs.append(section)
    return diffs


class RunDirectory:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.acc_dir = self.path / "acc"
        self.out_dir = self.path / "outputs"
        self.manifest_path = self.path / "manifest.json"
        self.progress_path = self.path / "progress.json"
        self.chunks_path = self.path / "chunks.npz"
        self.selection_path = self.path / "selection_indices.npy"

    def exists(self) -> bool:
        return self.manifest_path.is_file()

    def prepare(self, fingerprint: dict, extra: dict, resume: bool) -> None:
        fingerprint = normalise(fingerprint)
        if self.exists():
            if not resume:
                raise RuntimeError(
                    f"Run directory already contains a run: {self.path}. "
                    "Use --resume to continue it or choose another --output-dir."
                )
            old = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            diffs = diff_fingerprints(old.get("fingerprint", {}), fingerprint)
            if diffs:
                raise RuntimeError(
                    "Cannot resume: the run parameters differ from the previous run in "
                    f"{diffs}. Restore the original inputs/parameters or use a new --output-dir."
                )
            for tmp in list(self.acc_dir.glob("*.tmp")) + list(self.path.glob("*.tmp")):
                tmp.unlink()
            return
        if self.path.exists() and any(self.path.iterdir()):
            raise RuntimeError(f"Output directory is not empty and has no manifest: {self.path}")
        self.path.mkdir(parents=True, exist_ok=True)
        self.acc_dir.mkdir(exist_ok=True)
        write_json_atomic(self.manifest_path, {"fingerprint": fingerprint, **extra})
        write_json_atomic(self.progress_path, {"completed_batches": 0})

    def completed_batches(self) -> int:
        return int(json.loads(self.progress_path.read_text(encoding="utf-8"))["completed_batches"])

    def mark_batch_done(self, batch_index: int, n_batches: int) -> None:
        write_json_atomic(self.progress_path, {"completed_batches": batch_index + 1, "n_batches": n_batches})

    def acc_path(self, chunk_index: int) -> Path:
        return self.acc_dir / f"chunk_{chunk_index:06d}.npz"
