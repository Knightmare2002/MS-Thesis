"""Recolor the observation states of an exported fused_points.ply (no inference, no projection).

Only ``red, green, blue`` of points whose ``state`` is not ``damage`` are rewritten;
damage points keep their dominant-class color and every other field (coordinates,
normals, scores, views, labels, source_index) is copied byte for byte. The input is
read through a memory map in chunks and never modified; a NEW file is written.

Default: the current ``STATE_COLORS`` of damage3d/classes.py, so older exports get
the current palette. Override single states with ``--state-color NAME=R,G,B``.

    python -m damage3d.recolor --ply <run>/outputs/fused_points.ply
    python -m damage3d.recolor --ply ... --state-color insufficient_views=255,0,255
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import numpy as np

from .classes import STATE_COLORS, STATE_DAMAGE, STATE_NAMES
from .ply_io import PlyPointSource, PlyStreamWriter

log = logging.getLogger("damage3d")

_STATE_BY_NAME = {name: value for value, name in STATE_NAMES.items()}


def parse_state_color(text: str) -> tuple[int, tuple[int, int, int]]:
    """'insufficient_views=255,0,255' -> (1, (255, 0, 255))."""
    name, _, rgb = text.partition("=")
    if name not in _STATE_BY_NAME or _STATE_BY_NAME[name] == STATE_DAMAGE:
        allowed = [n for n, v in _STATE_BY_NAME.items() if v != STATE_DAMAGE]
        raise argparse.ArgumentTypeError(f"unknown/unsupported state {name!r}; use one of {allowed}")
    try:
        values = tuple(int(v) for v in rgb.split(","))
    except ValueError:
        values = ()
    if len(values) != 3 or not all(0 <= v <= 255 for v in values):
        raise argparse.ArgumentTypeError(f"color must be R,G,B with 0..255 values, got {rgb!r}")
    return _STATE_BY_NAME[name], values


def recolor(ply_path: Path, out_path: Path, colors: dict[int, tuple[int, int, int]],
            chunk: int = 2_000_000) -> dict[str, int]:
    """Write ``out_path`` with new state colors. Returns the number of recolored points per state."""
    ply_path, out_path = Path(ply_path), Path(out_path)
    if out_path.resolve() == ply_path.resolve():
        raise RuntimeError("--out must differ from --ply (the original export is never overwritten).")
    src = PlyPointSource(ply_path)
    try:
        names = src.header.dtype.names
        missing = [n for n in ("state", "red", "green", "blue") if n not in names]
        if missing:
            raise RuntimeError(f"{ply_path} has no {missing} field(s): not a damage3d export?")
        comments = [line[len("comment "):] for line in src.header.lines if line.startswith("comment ")]
        comments.append("state colors: " + ", ".join(f"{STATE_NAMES[s]}={c}" for s, c in sorted(colors.items())))
        tmp = out_path.with_name(out_path.name + ".tmp")
        writer = PlyStreamWriter(tmp, src.header.dtype, src.n_points, comments=comments)
        counts = {STATE_NAMES[s]: 0 for s in colors}
        for start in range(0, src.n_points, chunk):
            rec = np.array(src._mm[start:start + chunk])          # writable copy of this chunk only
            for s, (r, g, b) in colors.items():
                m = rec["state"] == s
                rec["red"][m], rec["green"][m], rec["blue"][m] = r, g, b
                counts[STATE_NAMES[s]] += int(m.sum())
            writer.write(rec)
        writer.close()
        os.replace(tmp, out_path)
        return counts
    finally:
        src.close()


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python -m damage3d.recolor", description=__doc__.split("\n\n")[0])
    p.add_argument("--ply", required=True, type=Path, help="fused_points.ply (or *_normals.ply) from damage3d")
    p.add_argument("--state-color", action="append", type=parse_state_color, default=[],
                   metavar="STATE=R,G,B", help="override one state color (repeatable)")
    p.add_argument("--out", type=Path, help="output PLY (default: <ply stem>_recolored.ply next to --ply)")
    p.add_argument("--chunk", type=int, default=2_000_000, help="points per streamed chunk")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    colors = {**STATE_COLORS, **dict(args.state_color)}
    out = args.out or args.ply.with_name(args.ply.stem + "_recolored.ply")
    counts = recolor(args.ply, out, colors, args.chunk)
    log.info("Wrote %s; recolored points per state: %s", out, counts)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
