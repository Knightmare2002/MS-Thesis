"""Add source normals to an already exported fused_points.ply (no re-accumulation).

For runs exported before ``--normals`` existed. Normals are looked up through the
``source_index`` field and copied unchanged from the source PLY. The input files are
read through memory maps in chunks and never modified; a NEW file is written.

    python -m damage3d.add_normals --fused <run>/outputs/fused_points.ply \
        --source <project>/bridge_3dpc_point_cloud.ply
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import numpy as np

from .ply_io import NORMAL_FIELDS, PlyPointSource, PlyStreamWriter

log = logging.getLogger("damage3d")


def add_normals(fused_path: Path, source_path: Path, out_path: Path, chunk: int = 2_000_000) -> int:
    """Write ``out_path`` = fused PLY with nx, ny, nz inserted after z. Returns the point count."""
    fused_path, source_path, out_path = Path(fused_path), Path(source_path), Path(out_path)
    if out_path.resolve() in (fused_path.resolve(), source_path.resolve()):
        raise RuntimeError("--out must differ from the input files (originals are never overwritten).")
    fused, source = PlyPointSource(fused_path), PlyPointSource(source_path)
    try:
        names = list(fused.header.dtype.names)
        if fused.has_normals:
            raise RuntimeError(f"{fused_path} already has nx/ny/nz.")
        if "source_index" not in names:
            raise RuntimeError(f"{fused_path} has no source_index field; cannot match the source points.")
        if not source.has_normals:
            raise RuntimeError(f"{source_path} has no nx/ny/nz vertex properties.")
        fields = [(n, fused.header.dtype[n]) for n in names]
        k = names.index("z") + 1
        dtype = np.dtype(fields[:k] + [(n, "<f4") for n in NORMAL_FIELDS] + fields[k:])
        comments = [line[len("comment "):] for line in fused.header.lines if line.startswith("comment ")]
        comments.append(f"normals copied from {source_path.name} via source_index")
        tmp = out_path.with_name(out_path.name + ".tmp")
        writer = PlyStreamWriter(tmp, dtype, fused.n_points, comments=comments)
        for start in range(0, fused.n_points, chunk):
            rec_in = np.asarray(fused._mm[start:start + chunk])
            idx = rec_in["source_index"].astype(np.int64)
            if idx.min() < 0 or idx.max() >= source.n_points:
                raise RuntimeError("source_index outside the source PLY: wrong --source file?")
            order = np.argsort(idx, kind="stable")          # memmap gathers are fastest on sorted indices
            normals = np.empty((len(idx), 3), dtype=np.float32)
            normals[order] = source.normals(idx[order])
            src_xyz = source.xyz(idx[order[:: max(1, len(order) // 1000)]]).astype(np.float32)
            got_xyz = np.column_stack([rec_in[a][order[:: max(1, len(order) // 1000)]] for a in "xyz"])
            if not np.array_equal(src_xyz, got_xyz):
                raise RuntimeError("Coordinates differ from the source PLY at the same source_index: "
                                   "wrong --source file?")
            rec = np.zeros(len(idx), dtype=dtype)
            for n in names:
                rec[n] = rec_in[n]
            for i, n in enumerate(NORMAL_FIELDS):
                rec[n] = normals[:, i]
            writer.write(rec)
        writer.close()
        os.replace(tmp, out_path)
        return fused.n_points
    finally:
        fused.close()
        source.close()


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python -m damage3d.add_normals", description=__doc__.split("\n\n")[0])
    p.add_argument("--fused", required=True, type=Path, help="fused_points.ply exported by damage3d")
    p.add_argument("--source", required=True, type=Path, help="source PLY used by the run (with nx, ny, nz)")
    p.add_argument("--out", type=Path, help="output PLY (default: <fused stem>_normals.ply next to --fused)")
    p.add_argument("--chunk", type=int, default=2_000_000, help="points per streamed chunk")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    out = args.out or args.fused.with_name(args.fused.stem + "_normals.ply")
    n = add_normals(args.fused, args.source, out, args.chunk)
    log.info("Wrote %s (%d points, normals from %s)", out, n, args.source)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
