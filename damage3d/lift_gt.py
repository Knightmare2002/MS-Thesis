"""T8: lift the 2D Labelme GT of the two annotated views of each block to the ROI points.

For a point of block b the labels come ONLY from the two annotated views of b (splits.csv
view1/view2), sampled at the nearest GT pixel of its logged (u, v). Per view and class:
1/0 if the pixel is valid, -1 if the point is not observed by that view or the pixel is
outside ``valid``. Two-view combination (``--combine``):
  agree (primary): both known and equal -> value; disagree -> -1; one known -> value; none -> -1
  or    (sensitivity): any 1 -> 1, else any 0 -> 0, else -1
Output: regions NPZ accepted by experiment_data.regions (region = group, split from splits.csv,
area = block ROI area / n points used) + per-block report JSON (view agreement, -1 fraction).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from damage3d.classes import DAMAGE_CLASSES, NUM_CLASSES  # noqa: E402
from damage3d.experiment_data import regions as check_regions  # noqa: E402
from damage3d.observations import iter_observations  # noqa: E402
from damage3d.ply_io import PlyPointSource  # noqa: E402


def combine(v1, v2, mode):
    if mode == "agree":
        out = np.where(v1 < 0, v2, np.where(v2 < 0, v1, np.where(v1 == v2, v1, -1)))
    else:
        out = np.where((v1 == 1) | (v2 == 1), 1, np.where((v1 == 0) | (v2 == 0), 0, -1))
    return out.astype(np.int8)


def agreement(v1, v2):
    """Per class on points known in both views: Dice of positives, Cohen kappa, n."""
    res = {}
    for c, name in enumerate(DAMAGE_CLASSES):
        ok = (v1[:, c] >= 0) & (v2[:, c] >= 0)
        a, b = v1[ok, c] == 1, v2[ok, c] == 1
        n = int(ok.sum())
        if not n:
            res[name] = {"n_both": 0}
            continue
        po = float(np.mean(a == b))
        pe = float(a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean()))
        s = int(a.sum() + b.sum())
        res[name] = {"n_both": n, "pos_view1": float(a.mean()), "pos_view2": float(b.mean()),
                     "dice": float(2 * np.sum(a & b) / s) if s else None,
                     "kappa": float((po - pe) / (1 - pe)) if pe < 1 else None}
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, required=True, help="T7 damage3d run (observation log)")
    ap.add_argument("--roi-blocks", type=Path, required=True, help="roi_blocks.npz from t7_roi_log prepare")
    ap.add_argument("--splits", type=Path, required=True)
    ap.add_argument("--summary", type=Path, required=True, help="roi_views summary_all.json (ROI areas)")
    ap.add_argument("--gt-dir", type=Path, required=True, help="<stem>_gt.npz from labelme_to_gt.py")
    ap.add_argument("--ply", type=Path, required=True, help="source PLY used by the run (xyz)")
    ap.add_argument("--buffer", type=float, required=True, help="split buffer, source units (1 m = 1.478)")
    ap.add_argument("--mm-per-unit", type=float, default=676.7, help="report only")
    ap.add_argument("--combine", choices=("agree", "or"), default="agree")
    ap.add_argument("--out", type=Path, required=True, help="regions .npz (report: same name .json)")
    a = ap.parse_args(argv)
    if a.out.exists():
        raise FileExistsError(f"{a.out}: choose a new output.")

    with open(a.splits, newline="", encoding="utf-8-sig") as f:
        rows = {int(r["block_id"]): r for r in csv.DictReader(f)}
    summary = json.loads(a.summary.read_text(encoding="utf-8"))
    meta = json.loads((a.run / "observations/meta.json").read_text(encoding="utf-8"))
    cams = {int(c["camera"]): c for c in meta["cameras"]}
    cam_of = {c["stem"]: k for k, c in cams.items()}
    rb = np.load(a.roi_blocks)
    ids, blocks = rb["source_index"].astype(np.int64), rb["block"].astype(int)
    n = len(ids)
    if set(np.unique(blocks)) - rows.keys():
        raise ValueError("Blocks in roi_blocks.npz missing from splits.csv.")

    # Pass 1: keep only (point, annotated view of its own block) records.
    view_cam = np.full((n, 2), -1, np.int64)
    for b in np.unique(blocks):
        for j, key in enumerate(("view1", "view2")):
            s = rows[b][key].strip()
            if s not in cam_of:
                raise ValueError(f"Block {b}: annotated view {s} not in run cameras.")
            view_cam[blocks == b, j] = cam_of[s]
    hits = {0: [], 1: []}
    for rec in iter_observations(a.run, ids):
        pos = np.searchsorted(ids, rec["source_index"])
        for j in (0, 1):
            m = rec["camera"] == view_cam[pos, j]
            hits[j].append((pos[m], rec["camera"][m].astype(int), rec["u"][m], rec["v"][m]))

    # Pass 2: one GT image in memory at a time.
    lab = np.full((2, n, NUM_CLASSES), -1, np.int8)
    for j in (0, 1):
        if not hits[j]:
            continue
        pos, cam, u, v = (np.concatenate(x) for x in zip(*hits[j]))
        for k in np.unique(cam):
            stem = cams[k]["stem"]
            with np.load(a.gt_dir / f"{stem}_gt.npz", allow_pickle=False) as g:
                target, valid, names = g["target"], g["valid"], g["classes"].tolist()
            h, w = valid.shape
            sel = cam == k
            # Metashape corner convention: pixel (i, j) covers [j, j+1) x [i, i+1); rescale if GT size differs.
            col = np.floor(u[sel] * w / cams[k]["width"]).astype(np.int64)
            row = np.floor(v[sel] * h / cams[k]["height"]).astype(np.int64)
            inside = (col >= 0) & (col < w) & (row >= 0) & (row < h)
            p, col, row = pos[sel][inside], col[inside], row[inside]
            ok = valid[row, col]
            for c, name in enumerate(DAMAGE_CLASSES):
                val = target[names.index(name), row, col].astype(np.int8)
                lab[j, p[ok], c] = val[ok]
    labels = combine(lab[0], lab[1], a.combine)

    area = np.empty(n)
    for b in np.unique(blocks):
        m = blocks == b
        area[m] = summary[f"block{b}"]["roi"]["area_source_units2"] / m.sum()
    xyz = PlyPointSource(a.ply).xyz(ids)
    out = {"source_index": ids, "region": np.array([rows[b]["group"] for b in blocks]),
           "split": np.array([rows[b]["split"] for b in blocks]), "xyz": xyz, "area": area,
           "labels": labels, "classes": np.array(DAMAGE_CLASSES), "block": blocks.astype(np.int16),
           "view_labels": np.moveaxis(lab, 0, 1)}  # (N,2,C) for audits
    a.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.out, **out)
    check_regions(a.out, a.buffer)  # contract: unique ids, no region across splits, buffer

    report = {"combine": a.combine, "buffer": a.buffer, "n_points": n, "blocks": {}}
    for b in np.unique(blocks):
        m = blocks == b
        report["blocks"][int(b)] = {
            "group": rows[b]["group"], "split": rows[b]["split"], "n_points": int(m.sum()),
            "observed_view1": float(np.mean(lab[0, m, 0] >= 0)), "observed_view2": float(np.mean(lab[1, m, 0] >= 0)),
            "unknown_fraction": {nm: float(np.mean(labels[m, c] < 0)) for c, nm in enumerate(DAMAGE_CLASSES)},
            "positive_area_fraction": {nm: float(np.mean(labels[m, c] == 1)) for c, nm in enumerate(DAMAGE_CLASSES)},
            "view_agreement": agreement(lab[0, m], lab[1, m]),
            "point_spacing_mm_est": float(np.sqrt(area[m][0]) * a.mm_per_unit)}
    a.out.with_suffix(".json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({b: {k: r[k] for k in ("split", "n_points", "observed_view1", "observed_view2")}
                      for b, r in report["blocks"].items()}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
