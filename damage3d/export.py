"""Outputs: CloudCompare-friendly PLY, machine-readable shards/CSV, summary.

``outputs/fused_points.ply`` (binary little endian), one vertex per exported point:
    x, y, z (float32, SAME frame and values as the source PLY), red, green, blue,
    state, label_mask, n_classes, n_views, score_<class> x6 (NaN = no valid view),
    views_<class> x6, source_index.
CloudCompare loads red/green/blue as colors and every other property as a
scalar field (filter e.g. ``score_crack`` or ``label_mask``).

``outputs/fused/chunk_XXXXXX.npz``: source_index, xyz, score (n,6), views (n,6),
votes (n,6), n_views, label_mask, state (+ ``outputs/classes.json``).
``outputs/fused_points.csv`` only when the exported points <= --csv-max-points.
"""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import numpy as np

from .classes import CLASS_COLORS, DAMAGE_CLASSES, NUM_CLASSES, STATE_COLORS, STATE_NAMES, class_bit
from .fusion import CAMERA_STAT_FIELDS, ChunkAccumulator, fuse
from .ply_io import PlyStreamWriter
from .run_state import write_json_atomic


def ply_dtype(index_dtype: str) -> np.dtype:
    fields = [("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1"),
              ("state", "u1"), ("label_mask", "u1"), ("n_classes", "u1"), ("n_views", "<u2")]
    fields += [(f"score_{c}", "<f4") for c in DAMAGE_CLASSES]
    fields += [(f"views_{c}", "<u2") for c in DAMAGE_CLASSES]
    fields += [("source_index", index_dtype)]
    return np.dtype(fields)


def scope_mask(state: np.ndarray, scope: str) -> np.ndarray:
    if scope == "all":
        return np.ones(state.shape, dtype=bool)
    if scope == "observed":
        return state != 0
    if scope == "damage":
        return state == 3
    raise ValueError(f"Unknown export scope '{scope}'.")


def legend() -> dict:
    return {
        "states": {STATE_NAMES[k]: {"value": k, "rgb": STATE_COLORS.get(k, "class color")} for k in STATE_NAMES},
        "classes": {c: {"bit": class_bit(c), "channel": i, "rgb": CLASS_COLORS[c]} for i, c in enumerate(DAMAGE_CLASSES)},
        "damage_color_rule": "points with >=1 positive label take the color of the positive class with the "
                             "largest score/threshold ratio; label_mask and n_classes keep ALL positive classes",
    }


def export_run(run, chunks: np.ndarray, selection, ply_source, thresholds: np.ndarray, min_views: int,
               scope: str, csv_max_points: int, n_cameras: int, camera_rows: list[dict], meta: dict) -> dict:
    """Stream every chunk twice (count, then write) without holding the cloud in RAM."""
    out = run.out_dir
    shard_dir = out / "fused"
    shard_dir.mkdir(parents=True, exist_ok=True)
    for old in shard_dir.glob("*.npz"):
        old.unlink()

    index_dtype = "<u4" if selection.n_source <= np.iinfo(np.uint32).max else "<f8"
    dtype = ply_dtype(index_dtype)
    state_counts = np.zeros(4, dtype=np.int64)
    label_counts = np.zeros(NUM_CLASSES, dtype=np.int64)
    multi_label_points = 0
    camera_stats = np.zeros((n_cameras, len(CAMERA_STAT_FIELDS) + NUM_CLASSES), dtype=np.int64)
    n_export = 0

    def load(j, n):
        path = run.acc_path(j)
        return ChunkAccumulator.load(path) if path.is_file() else ChunkAccumulator.zeros(n, n_cameras)

    # Pass 1: fuse, write shards, count.
    for j, (start, stop) in enumerate(chunks):
        n = int(stop - start)
        acc = load(j, n)
        fused = fuse(acc, thresholds, min_views)
        camera_stats += acc.camera_stats
        state_counts += np.bincount(fused.state, minlength=4)[:4]
        label_counts += fused.labels.sum(axis=0)
        multi_label_points += int((fused.labels.sum(axis=1) >= 2).sum())
        keep = scope_mask(fused.state, scope)
        n_export += int(keep.sum())
        if not keep.any():
            continue
        src_idx = selection.source_index_array(int(start), int(stop))[keep]
        xyz = ply_source.xyz(selection.source_index(int(start), int(stop)))[keep].astype(np.float32)
        tmp = shard_dir / f"chunk_{j:06d}.npz.tmp"
        with tmp.open("wb") as f:
            np.savez(f, source_index=src_idx, xyz=xyz, score=fused.score[keep], views=acc.count[keep],
                     votes=acc.votes[keep], n_views=acc.n_views[keep], label_mask=fused.label_mask[keep],
                     state=fused.state[keep], rgb=fused.rgb[keep])
        os.replace(tmp, shard_dir / f"chunk_{j:06d}.npz")

    # Pass 2: stream the PLY (and optional CSV) from the shards.
    ply_tmp = out / "fused_points.ply.tmp"
    writer = PlyStreamWriter(ply_tmp, dtype, n_export, comments=[
        "damage3d fused multi-view damage map",
        f"classes {' '.join(DAMAGE_CLASSES)}",
        f"provider {meta.get('provider_type')} synthetic={meta.get('synthetic')}",
    ])
    write_csv = n_export <= csv_max_points
    csv_file = (out / "fused_points.csv.tmp").open("w", newline="", encoding="utf-8") if write_csv else None
    csv_writer = csv.writer(csv_file) if csv_file else None
    if csv_writer:
        csv_writer.writerow(["source_index", "x", "y", "z", "state", "label_mask", "n_views"]
                            + [f"score_{c}" for c in DAMAGE_CLASSES] + [f"views_{c}" for c in DAMAGE_CLASSES]
                            + [f"label_{c}" for c in DAMAGE_CLASSES])
    for shard in sorted(shard_dir.glob("chunk_*.npz")):
        with np.load(shard) as d:
            rec = np.zeros(len(d["state"]), dtype=dtype)
            rec["x"], rec["y"], rec["z"] = d["xyz"][:, 0], d["xyz"][:, 1], d["xyz"][:, 2]
            rec["red"], rec["green"], rec["blue"] = d["rgb"][:, 0], d["rgb"][:, 1], d["rgb"][:, 2]
            rec["state"] = d["state"]
            rec["label_mask"] = d["label_mask"]
            rec["n_classes"] = np.unpackbits(d["label_mask"][:, None], axis=1).sum(axis=1)
            rec["n_views"] = d["n_views"]
            for i, c in enumerate(DAMAGE_CLASSES):
                rec[f"score_{c}"] = d["score"][:, i]
                rec[f"views_{c}"] = d["views"][:, i]
            rec["source_index"] = d["source_index"]
            writer.write(rec)
            if csv_writer:
                labels = (d["label_mask"][:, None] >> np.arange(NUM_CLASSES, dtype=np.uint8)) & 1
                for k in range(len(rec)):
                    csv_writer.writerow(
                        [int(d["source_index"][k]), *(f"{v:.6f}" for v in d["xyz"][k]), int(d["state"][k]),
                         int(d["label_mask"][k]), int(d["n_views"][k])]
                        + ["" if np.isnan(s) else f"{s:.6f}" for s in d["score"][k]]
                        + [int(v) for v in d["views"][k]] + [int(v) for v in labels[k]])
    writer.close()
    os.replace(ply_tmp, out / "fused_points.ply")
    if csv_file:
        csv_file.close()
        os.replace(out / "fused_points.csv.tmp", out / "fused_points.csv")
    elif (out / "fused_points.csv").exists():
        (out / "fused_points.csv").unlink()

    (out / "classes.json").write_text(json.dumps(list(DAMAGE_CLASSES)), encoding="utf-8")
    with (out / "per_camera_stats.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["order", "position", "label", "camera_id", *CAMERA_STAT_FIELDS,
                    *[f"positive_views_{c}" for c in DAMAGE_CLASSES]])
        for k, row in enumerate(camera_rows):
            w.writerow([k, row["position"], row["label"], row["camera_id"], *camera_stats[k].tolist()])

    summary = {
        **meta,
        "n_points_selected": int(selection.size),
        "n_points_exported": n_export,
        "export_scope": scope,
        "thresholds": dict(zip(DAMAGE_CLASSES, thresholds.tolist())),
        "min_views": min_views,
        "state_counts": {STATE_NAMES[k]: int(state_counts[k]) for k in range(4)},
        "label_counts": dict(zip(DAMAGE_CLASSES, label_counts.tolist())),
        "points_with_two_or_more_labels": multi_label_points,
        "legend": legend(),
        "csv_written": bool(write_csv),
    }
    write_json_atomic(out / "summary.json", summary)
    return summary
