"""T7: ROI observation-log preparation (``prepare``) and post-run check (``verify``).

prepare  roi_union.npy  (sorted union of the per-block ROI points, optional per-block subsample)
         roi_blocks.npz (source_index, block): point -> block, used by lift_gt
         cams_roi.txt   (cameras with n_visible>0 for at least one block + all annotated views)
         and prints the damage3d command.
verify   on the finished run: sealed inventory, schema 3, files+platt provider, annotated
         cameras present, ROI points without records, per annotated view observed fraction
         vs frac_roi_visible of roi_views (sampled estimate, so only approximately equal).
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
from damage3d.observations import ObservationLog, OBS_DTYPE  # noqa: E402



def iter_observations(run_path, source_ids=None, block_records=250_000):
    """Verify-only bounded reader; exact per-chunk duplicate check on disk."""
    import tempfile
    from itertools import groupby

    if block_records <= 0:
        raise ValueError("block_records must be positive")
    files = sorted((Path(run_path) / "observations").glob("chunk_*_batch_*.npy"))
    if not files:
        raise FileNotFoundError(f"No observation log in {run_path}")
    ids = None if source_ids is None else np.unique(np.asarray(source_ids))

    def read_parts(paths):
        for path in paths:
            src = np.load(path, mmap_mode="r", allow_pickle=False)
            try:
                if src.dtype != OBS_DTYPE or src.ndim != 1:
                    raise ValueError(f"{path}: incompatible observation schema")
                for start in range(0, len(src), block_records):
                    part = src[start:start + block_records]
                    if ids is not None:
                        part = part[np.isin(part["source_index"], ids)]
                    if len(part):
                        yield part
                    del part
            finally:
                src._mmap.close()

    for chunk, paths in groupby(files, key=lambda p: p.name.split("_batch_")[0]):
        paths = list(paths)
        total = 0
        for path in paths:
            src = np.load(path, mmap_mode="r", allow_pickle=False)
            try:
                if src.dtype != OBS_DTYPE or src.ndim != 1:
                    raise ValueError(f"{path}: incompatible observation schema")
                total += len(src)
            finally:
                src._mmap.close()
        if not total:
            continue
        print(f"Verify {chunk}: checking duplicates on disk ({total:,} records maximum)", flush=True)
        with tempfile.TemporaryDirectory(prefix="t7_verify_keys_") as tmp:
            keys = np.memmap(Path(tmp) / "keys.dat", mode="w+", dtype=np.uint64, shape=(total,))
            try:
                used = 0
                for part in read_parts(paths):
                    count = len(part)
                    keys[used:used + count] = (
                        (part["source_index"].astype(np.uint64) << np.uint64(16))
                        | part["camera"].astype(np.uint64)
                    )
                    used += count
                if used:
                    view = keys[:used]
                    view.sort(kind="quicksort")
                    keys.flush()
                    previous = None
                    for start in range(0, used, block_records):
                        block = view[start:start + block_records]
                        if (previous is not None and previous == int(block[0])) or np.any(block[1:] == block[:-1]):
                            raise ValueError("Duplicate point-camera observations.")
                        previous = int(block[-1])
                        del block
                    del view
            finally:
                keys._mmap.close()
        print(f"Verify {chunk}: counting observations", flush=True)
        for part in read_parts(paths):
            yield np.array(part, copy=True)


def read_csv(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def annotated_views(splits):
    return {int(r["block_id"]): (r["view1"].strip(), r["view2"].strip()) for r in read_csv(splits)}


def prepare(a):
    rng = np.random.default_rng(a.seed)
    rows = read_csv(a.roi_csv)
    folders = {int(r["block_id"]): r["block_folder"] for r in rows}
    views = annotated_views(a.splits)
    if set(views) != set(folders):
        raise ValueError(f"Blocks differ: splits {sorted(views)} vs roi csv {sorted(folders)}")
    ids, blocks, report = [], [], {}
    for b in sorted(folders):
        idx = np.unique(np.load(a.roi_root / folders[b] / "roi_points_index.npy").astype(np.int64))
        n_all = len(idx)
        if a.max_points_per_block and n_all > a.max_points_per_block:
            idx = np.sort(rng.choice(idx, a.max_points_per_block, replace=False))
        ids.append(idx)
        blocks.append(np.full(len(idx), b, np.int16))
        report[b] = {"n_in_polygon": n_all, "n_used": len(idx), "sampling_fraction": len(idx) / n_all}
    ids, blocks = np.concatenate(ids), np.concatenate(blocks)
    order = np.argsort(ids, kind="stable")
    ids, blocks = ids[order], blocks[order]
    dup = ids[1:] == ids[:-1]
    if dup.any():
        raise ValueError(f"{int(dup.sum())} points belong to two blocks: polygons overlap.")
    a.out.mkdir(parents=True, exist_ok=True)
    np.save(a.out / "roi_union.npy", ids)
    np.savez(a.out / "roi_blocks.npz", source_index=ids, block=blocks)
    stems = {r["stem"] for r in rows if int(r["n_visible"]) > 0}
    ann = {s for v in views.values() for s in v}
    missing_ann = sorted(ann - stems)
    stems = sorted(stems | ann)
    (a.out / "cams_roi.txt").write_text(
        "# cameras with n_visible>0 in roi_views_all.csv + annotated views\n" + "\n".join(stems) + "\n",
        encoding="utf-8")
    no_map = []
    if a.probabilities_dir:
        no_map = [s for s in stems if not (a.probabilities_dir / f"{s}.npz").is_file()]
    summary = {"n_points": int(len(ids)), "n_cameras": len(stems), "seed": a.seed,
               "max_points_per_block": a.max_points_per_block, "blocks": report,
               "annotated_not_visible_in_roi_csv": missing_ann, "cameras_without_probability_map": no_map}
    (a.out / "t7_prepare.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "blocks"}, indent=2))
    if no_map:
        print(f"WARNING: {len(no_map)} cameras without a probability map: run inference first.")
    print("\nCommand (one line):\n"
          f'python -m damage3d --probability-source files --probabilities-dir "<maps>" '
          f'--calibration "<T13 run>\\eval_curves\\probability_calibration.json" '
          f'--camera-list "{a.out / "cams_roi.txt"}" --observation-log '
          f'--observation-subset "{a.out / "roi_union.npy"}" --output-dir damage3d_runs\\t7_roi_obslog')
    return 0


def verify(a):
    run = a.run
    meta = json.loads((run / "observations/meta.json").read_text(encoding="utf-8"))
    problems = []
    if meta.get("schema") != 3:
        problems.append("meta schema != 3")
    provider = meta.get("provider", {})
    if provider.get("type") != "files+platt":
        problems.append(f"provider type {provider.get('type')!r} != 'files+platt'")
    ObservationLog(run).seal(verify_only=True)  # raises if shards changed after the run
    if not (run / "outputs/summary.json").is_file():
        problems.append("outputs/summary.json missing (run not completed)")
    cam = {c["stem"]: int(c["camera"]) for c in meta["cameras"]}
    views = annotated_views(a.splits)
    absent = sorted({s for v in views.values() for s in v} - cam.keys())
    if absent:
        problems.append(f"annotated cameras not in run: {absent}")
    rb = np.load(a.roi_blocks)
    ids, blocks = rb["source_index"], rb["block"]
    n_rec = np.zeros(len(ids), np.int64)
    seen = {(b, s): np.zeros(len(ids), bool) for b, v in views.items() for s in v}
    for rec in iter_observations(run, ids):
        pos = np.searchsorted(ids, rec["source_index"])
        np.add.at(n_rec, pos, 1)
        for (b, s), mask in seen.items():
            if s in cam:
                hit = pos[(rec["camera"] == cam[s]) & (blocks[pos] == b)]
                mask[hit] = True
    frac_csv = {(int(r["block_id"]), r["stem"]): float(r["frac_roi_visible"]) for r in read_csv(a.roi_csv)}
    per_view = []
    for (b, s), mask in seen.items():
        nb = int((blocks == b).sum())
        per_view.append({"block": b, "stem": s, "frac_observed_log": float(mask.sum() / nb),
                         "frac_roi_visible_roi_views": frac_csv.get((b, s))})
    out = {"problems": problems, "n_points": int(len(ids)),
           "points_without_records": int((n_rec == 0).sum()),
           "records_per_point_median": float(np.median(n_rec)),
           "per_block_without_records": {int(b): int(((blocks == b) & (n_rec == 0)).sum())
                                         for b in np.unique(blocks)},
           "annotated_views": per_view}
    (run / "t7_verify.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))
    return 1 if problems else 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--roi-root", type=Path, required=True, help="roi_views output dir containing blockN/")
    p.add_argument("--roi-csv", type=Path, required=True, help="roi_views_all.csv")
    p.add_argument("--splits", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--max-points-per-block", type=int, default=300000, help="0 = all points")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--probabilities-dir", type=Path)
    v = sub.add_parser("verify")
    v.add_argument("--run", type=Path, required=True)
    v.add_argument("--roi-blocks", type=Path, required=True)
    v.add_argument("--roi-csv", type=Path, required=True)
    v.add_argument("--splits", type=Path, required=True)
    a = ap.parse_args(argv)
    return prepare(a) if a.cmd == "prepare" else verify(a)


if __name__ == "__main__":
    raise SystemExit(main())
