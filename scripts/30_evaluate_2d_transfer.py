#!/usr/bin/env python
"""Step 30 (T6 / H1) - 2D transfer of frozen P1ML maps to the annotated Lincoln Street views.

Inputs (nothing is fitted here):
  * maps     <maps-dir>/<stem>.npz written by scripts/23 (raw probs, 6 channels, downsampled);
  * GT       <gt-dir>/<stem>_gt.npz written by labelme_to_gt.py (target [6,H,W], valid [H,W]);
  * splits   splits.csv (block_id, group, split, view1, view2): defines images, groups, splits.

Protocol:
  * every annotated image is evaluated ONCE at full GT resolution (probabilities are
    bilinearly upsampled); an image shared by several blocks must belong to one group;
  * pixels outside ``valid`` (roi AND NOT ignore) are excluded;
  * thresholds are FROZEN on DACL10K validation: from --calibration (calibrated scale, Platt
    applied) or from --run-dir / --thresholds (raw scale). Bridge data never choose them;
  * metrics are pooled over pixels (micro) per split; 95% CIs by bootstrap over GROUPS;
  * threshold-free AP and ECE from 1000-bin score histograms; false-alarm rate on
    damage-free images; optional crack tolerance (--crack-tol-px) for thin structures.

Run from the repository root (imports damage3d and src).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from damage3d.classes import DAMAGE_CLASSES, MODEL_CLASSES
from damage3d.fusion import parse_thresholds
from damage3d.providers import FileProvider
from src.eval.probability_calibration import apply_platt, load_platt, sha256_file, thresholds_from_json

BINS = 1000
RUN_THRESHOLDS = Path("eval_multilabel_sliding_calibrated") / "thresholds_per_class_validation_calibrated.json"


def read_splits(path: Path) -> dict[str, dict]:
    """stem -> {group, split, blocks}; error if an image would span two groups."""
    images: dict[str, dict] = {}
    with open(path, newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            for stem in (r["view1"], r["view2"]):
                e = images.setdefault(stem, {"group": r["group"], "split": r["split"], "blocks": []})
                if (e["group"], e["split"]) != (r["group"], r["split"]):
                    raise ValueError(f"{stem} is annotated for different groups/splits: leakage.")
                e["blocks"].append(int(r["block_id"]))
    return images


def frozen_thresholds(args) -> tuple[np.ndarray, str]:
    if args.calibration:
        return thresholds_from_json(args.calibration, DAMAGE_CLASSES), f"calibrated:{args.calibration}"
    if args.thresholds:
        return np.asarray(parse_thresholds(args.thresholds), np.float32), "cli"
    path = args.run_dir / RUN_THRESHOLDS
    thr = json.loads(path.read_text(encoding="utf-8"))["thresholds"]
    return np.array([thr[f"threshold_{c}"] for c in DAMAGE_CLASSES], np.float32), f"raw:{path}"


def image_counts(prob: np.ndarray, gt: np.ndarray, valid: np.ndarray, thr: float, tol: int) -> dict:
    pred, y = prob > thr, gt.astype(bool)
    pv, yv = pred[valid], y[valid]
    out = {"tp": int((pv & yv).sum()), "fp": int((pv & ~yv).sum()),
           "fn": int((~pv & yv).sum()), "tn": int((~pv & ~yv).sum()),
           "hpos": np.histogram(prob[valid][yv], BINS, (0, 1))[0],
           "hneg": np.histogram(prob[valid][~yv], BINS, (0, 1))[0]}
    if tol > 0:  # tolerant precision/recall (thin cracks, annotation width)
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * tol + 1, 2 * tol + 1))
        y_d = cv2.dilate(y.astype(np.uint8), k).astype(bool)
        p_d = cv2.dilate(pred.astype(np.uint8), k).astype(bool)
        out.update(tol_pred=int(pv.sum()), tol_pred_hit=int((pv & y_d[valid]).sum()),
                   tol_gt=int(yv.sum()), tol_gt_hit=int((yv & p_d[valid]).sum()))
    return out


def summarize(c: dict) -> dict:
    tp, fp, fn, tn = (float(c[k]) for k in ("tp", "fp", "fn", "tn"))
    s = {"dice": 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else None,
         "iou": tp / (tp + fp + fn) if tp + fp + fn else None,
         "precision": tp / (tp + fp) if tp + fp else None,
         "recall": tp / (tp + fn) if tp + fn else None,
         "fpr": fp / (fp + tn) if fp + tn else None,
         "prevalence": (tp + fn) / (tp + fp + fn + tn) if tp + fp + fn + tn else None}
    hp, hn = c["hpos"][::-1].cumsum(), c["hneg"][::-1].cumsum()   # descending thresholds
    if hp[-1] > 0:
        prec = hp / np.maximum(hp + hn, 1)
        s["ap"] = float(np.sum(np.diff(np.r_[0, hp / hp[-1]]) * prec))
        centers = (np.arange(BINS) + .5) / BINS
        n = c["hpos"] + c["hneg"]
        edges = np.linspace(0, BINS, 16).astype(int)   # 15 equal-width bins
        err = [abs((c["hpos"][a:b].sum() - (centers[a:b] * n[a:b]).sum())) for a, b in zip(edges[:-1], edges[1:])]
        s["ece"] = float(np.sum(err) / max(n.sum(), 1))
    else:
        s["ap"] = s["ece"] = None
    if "tol_pred" in c:
        p = c["tol_pred_hit"] / c["tol_pred"] if c["tol_pred"] else None
        r = c["tol_gt_hit"] / c["tol_gt"] if c["tol_gt"] else None
        s.update(tol_precision=p, tol_recall=r, tol_f1=2 * p * r / (p + r) if p and r else None)
    return s


def add(acc: dict, c: dict) -> None:
    for k, v in c.items():
        acc[k] = acc[k] + v if k in acc else (v.copy() if isinstance(v, np.ndarray) else v)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--maps-dir", type=Path, required=True)
    ap.add_argument("--gt-dir", type=Path, required=True)
    ap.add_argument("--splits", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    t = ap.add_mutually_exclusive_group(required=True)
    t.add_argument("--calibration", type=Path, help="scripts/27 probability_calibration.json (Platt + frozen thr)")
    t.add_argument("--run-dir", type=Path, help="P1ML run: raw frozen validation thresholds (scripts/17)")
    t.add_argument("--thresholds", help="explicit frozen thresholds, e.g. 'crack=0.6,...'")
    ap.add_argument("--crack-tol-px", type=int, default=0, help="tolerance radius at full resolution (0 = off)")
    ap.add_argument("--exclude-blocks", default="", help="sensitivity: comma list, e.g. '15'")
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)

    images = read_splits(a.splits)
    excl = {int(b) for b in a.exclude_blocks.split(",") if b}
    images = {s: e for s, e in images.items() if not set(e["blocks"]) <= excl}
    thr, thr_source = frozen_thresholds(a)
    platt = load_platt(a.calibration, DAMAGE_CLASSES)[:2] if a.calibration else None
    missing = [s for s in images if not (a.gt_dir / f"{s}_gt.npz").is_file()
               or not (a.maps_dir / f"{s}.npz").is_file()]
    if missing:
        raise SystemExit(f"Missing GT or map for: {missing}")

    provider, rows = FileProvider(a.maps_dir), []
    per_img: dict[str, dict] = {}
    for i, (stem, e) in enumerate(sorted(images.items()), 1):
        with np.load(a.gt_dir / f"{stem}_gt.npz", allow_pickle=False) as g:
            target, valid = g["target"], g["valid"].astype(bool)
        h, w = valid.shape
        probs = provider.get(stem, w, h).probs            # (5, h', w'), DAMAGE_CLASSES order
        if platt is not None:
            probs = apply_platt(probs, *platt)
        per_img[stem] = {}
        for c, name in enumerate(DAMAGE_CLASSES):
            p = cv2.resize(probs[c].astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
            gt = target[MODEL_CLASSES.index(name)]
            cnt = image_counts(p, gt, valid, float(thr[c]), a.crack_tol_px if name == "crack" else 0)
            per_img[stem][name] = cnt
            rows.append({"stem": stem, "blocks": "+".join(map(str, e["blocks"])), "group": e["group"],
                         "split": e["split"], "class": name, "threshold": float(thr[c]),
                         **{k: cnt[k] for k in ("tp", "fp", "fn", "tn")}})
        print(f"[{i}/{len(images)}] {stem} ({e['split']}, {e['group']}) valid={valid.mean():.3f}")

    rng = np.random.default_rng(a.seed)
    report = {"thresholds": dict(zip(DAMAGE_CLASSES, map(float, thr))), "threshold_source": thr_source,
              "splits_sha256": sha256_file(a.splits), "excluded_blocks": sorted(excl),
              "crack_tol_px": a.crack_tol_px, "n_images": len(images), "results": {}}
    for split in ("C_fit", "C_cal", "T", "all"):
        stems = [s for s, e in images.items() if split == "all" or e["split"] == split]
        groups = sorted({images[s]["group"] for s in stems})
        res = {"n_images": len(stems), "groups": groups, "classes": {}}
        for name in DAMAGE_CLASSES:
            by_group = defaultdict(dict)
            for s in stems:
                add(by_group[images[s]["group"]], per_img[s][name])
            pooled: dict = {}
            for gacc in by_group.values():
                add(pooled, gacc)
            m = summarize(pooled)
            boot = defaultdict(list)
            for _ in range(a.bootstrap if len(groups) > 1 else 0):
                acc: dict = {}
                for gname in rng.choice(groups, len(groups), replace=True):
                    add(acc, by_group[gname])
                for k, v in summarize(acc).items():
                    if v is not None:
                        boot[k].append(v)
            m["ci95"] = {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
                         for k, v in boot.items() if len(v) >= .9 * a.bootstrap}
            m["n_positive_groups"] = sum(1 for g in by_group.values() if g["tp"] + g["fn"] > 0)
            clean = [s for s in stems if all(per_img[s][c]["tp"] + per_img[s][c]["fn"] == 0
                                             for c in DAMAGE_CLASSES)]
            fp_c = sum(per_img[s][name]["fp"] for s in clean)
            n_c = sum(per_img[s][name]["fp"] + per_img[s][name]["tn"] for s in clean)
            m["false_alarm_rate_damage_free_images"] = fp_c / n_c if n_c else None
            res["classes"][name] = m
        report["results"][split] = res

    a.out.mkdir(parents=True, exist_ok=True)
    with open(a.out / "per_image_counts.csv", "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    (a.out / "h1_2d_transfer.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    for split, res in report["results"].items():
        print(f"\n{split}: {res['n_images']} images, groups {res['groups']}")
        for name, m in res["classes"].items():
            f = lambda v: "  n/a" if v is None else f"{v:.3f}"
            print(f"  {name:10s} dice {f(m['dice'])} P {f(m['precision'])} R {f(m['recall'])} "
                  f"AP {f(m['ap'])} FPR {f(m['fpr'])} pos.groups {m['n_positive_groups']}")
    print(f"\nWritten {a.out / 'h1_2d_transfer.json'} and per_image_counts.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
