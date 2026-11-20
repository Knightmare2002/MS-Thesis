#!/usr/bin/env python3
"""Labelme JSON -> multilabel ground truth + validity mask (Lincoln Street test set).

Annotation convention
---------------------
* ``roi``     inclusion polygon(s). Required: an image without ``roi`` is an error,
              never "whole image valid". Several roi shapes are merged (union).
* ``ignore``  exclusion inside the roi (truth not decidable by the annotator:
              vegetation, motion blur, unreadable texture). Not for shadows or
              overexposure: those are part of the test.
* damage      crack, spalling, corrosion, moisture, surface (damage3d/labels.txt).
              Shapes may overlap: channels are independent (multilabel).

valid = roi AND NOT ignore, shared by all channels. Pixels outside ``valid`` are
neither TP, FP, FN nor TN. Damage drawn outside the roi is dropped (and counted).

Output ``<stem>_gt.npz``: target uint8 [6,H,W] in MODEL_CLASSES order
(delamination channel always 0, closed class), valid bool [H,W], classes.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

MODEL_CLASSES = ("crack", "spalling", "corrosion", "moisture", "delamination", "surface")
DAMAGE_LABELS = ("crack", "spalling", "corrosion", "moisture", "surface")
ROI, IGNORE = "roi", "ignore"


def _draw(mask: np.ndarray, shape: dict, line_px: int) -> None:
    pts = np.asarray(shape["points"], dtype=np.float64)
    kind = shape.get("shape_type") or "polygon"
    ipts = np.rint(pts).astype(np.int32)
    if kind == "polygon":
        if len(ipts) < 3:
            raise ValueError(f"polygon with {len(ipts)} points")
        cv2.fillPoly(mask, [ipts], 1)
    elif kind == "rectangle":
        (x0, y0), (x1, y1) = ipts[:2]
        cv2.rectangle(mask, (min(x0, x1), min(y0, y1)), (max(x0, x1), max(y0, y1)), 1, -1)
    elif kind == "circle":
        c, e = pts[:2]
        cv2.circle(mask, tuple(np.rint(c).astype(int)), int(round(np.linalg.norm(e - c))), 1, -1)
    elif kind in ("linestrip", "line"):
        cv2.polylines(mask, [ipts], False, 1, thickness=line_px)
    else:  # point, mask, ...
        raise ValueError(f"unsupported shape_type '{kind}'")


def convert(json_path: Path, expected_wh: tuple[int, int] | None, line_px: int) -> dict:
    data = json.loads(json_path.read_text(encoding="utf-8"))
    w, h = int(data["imageWidth"]), int(data["imageHeight"])
    if expected_wh is not None and (w, h) != tuple(expected_wh):
        # Labelme applies EXIF orientation; damage3d maps use the raw sensor frame.
        raise ValueError(f"{json_path.name}: Labelme size {w}x{h} != camera {expected_wh} "
                         "(EXIF rotation or wrong image)")
    roi = np.zeros((h, w), np.uint8)
    ign = np.zeros((h, w), np.uint8)
    target = np.zeros((len(MODEL_CLASSES), h, w), np.uint8)
    for s in data["shapes"]:
        label = s["label"].strip().lower()
        if label == ROI:
            _draw(roi, s, line_px)
        elif label == IGNORE:
            _draw(ign, s, line_px)
        elif label in DAMAGE_LABELS:
            _draw(target[MODEL_CLASSES.index(label)], s, line_px)
        else:
            raise ValueError(f"{json_path.name}: unknown label '{s['label']}'")
    if not roi.any():
        raise ValueError(f"{json_path.name}: no '{ROI}' shape")
    valid = roi.astype(bool) & ~ign.astype(bool)
    dropped = {c: int((target[i].astype(bool) & ~roi.astype(bool)).sum()) for i, c in enumerate(MODEL_CLASSES)}
    target &= valid[None].astype(np.uint8)       # outside valid: undefined, stored as 0
    return {"target": target, "valid": valid, "size": (w, h), "dropped_outside_roi_px": dropped,
            "valid_fraction": float(valid.mean())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("jsons", nargs="+", type=Path,
                    help="Labelme JSON files or folders containing JSON files (non-recursive)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--expected-size", nargs=2, type=int, metavar=("W", "H"), default=(9504, 6336))
    ap.add_argument("--line-px", type=int, default=3, help="thickness for linestrip crack annotations")
    a = ap.parse_args()
    json_files = []
    seen = set()
    for source in a.jsons:
        if source.is_dir():
            candidates = sorted(
                (p for p in source.iterdir() if p.is_file() and p.suffix.lower() == ".json"),
                key=lambda p: p.name.lower(),
            )
            if not candidates:
                ap.error(f"No JSON files found in folder: {source}")
        elif source.is_file() and source.suffix.lower() == ".json":
            candidates = [source]
        else:
            ap.error(f"Input is not a JSON file or existing folder: {source}")
        for jp in candidates:
            resolved = jp.resolve()
            if resolved not in seen:
                seen.add(resolved)
                json_files.append(jp)
    output_names = {}
    for jp in json_files:
        key = jp.stem.lower()
        if key in output_names:
            ap.error(f"Duplicate output name: {output_names[key]} and {jp}")
        output_names[key] = jp
    if a.line_px < 1:
        ap.error("--line-px must be at least 1")
    a.out.mkdir(parents=True, exist_ok=True)
    print(f"Converting {len(json_files)} JSON file(s)")
    report = {}
    for jp in json_files:
        r = convert(jp, tuple(a.expected_size), a.line_px)
        np.savez_compressed(a.out / f"{jp.stem}_gt.npz", target=r["target"], valid=r["valid"],
                            classes=np.array(MODEL_CLASSES))
        report[jp.stem] = {k: r[k] for k in ("size", "valid_fraction", "dropped_outside_roi_px")}
        print(jp.stem, report[jp.stem])
    (a.out / "gt_manifest.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
