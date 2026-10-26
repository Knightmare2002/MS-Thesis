#!/usr/bin/env python
"""Step 28 - Platt-calibrated ensemble of P1ML runs: thresholds and validation metrics.

Combination rule (identical in damage3d, see damage3d/providers.py::EnsembleProvider)
--------------------------------------------------------------------------------------
    p_ens(x, c) = sum_m w_m * sigmoid(a_mc * logit(p_m(x, c)) + b_mc),   sum_m w_m = 1

1. every member m is run with its OWN config (patch size, stride, blending) and
   checkpoint, exactly as in scripts/12-17;
2. its raw probabilities are mapped with its OWN Platt parameters (scripts/27
   JSON, nothing hard-coded);
3. the calibrated probabilities are averaged with normalised weights;
4. per class, the ensemble threshold is argmax_t Dice_c(t) on the validation
   split (dataset-level pixel counts, ties -> highest t), i.e. the scripts/17
   policy; these are the FROZEN thresholds written to ``ensemble.json`` and
   used by damage3d (``--probability-source ensemble --ensemble ensemble.json``).

Honest estimate
---------------
The thresholds above are selected and evaluated on the same images
(optimistic, like scripts/17). The script also reports the 2-fold protocol of
scripts/20 (same seed, same halves): select on one half, report on the other.
The same protocol is applied to each calibrated member alone, so members and
ensemble are compared on identical images, thresholds grid and rule. (The
Platt maps themselves were fitted on the whole validation split: 2
parameters per class, monotone, negligible leakage.)

Members are fully parametric:
    --member name=T9  run_dir=outputs/runs/<t9>  weight=1 [checkpoint=best.pt] [calibration=<json>]
    --member name=T12 run_dir=outputs/runs/<t12> weight=1
or  --members-file members.json   ({"members": [{"name": ..., "run_dir": ..., "weight": ...}, ...]})

Outputs (default outputs/ensembles/<name>/)
-------
    ensemble.json                    members + Platt params + weights + frozen thresholds + summary
    ensemble_metrics_per_class.csv   source x protocol x class
    ensemble_summary.csv             source x protocol (macro metrics)
    ensemble_cache.npz               histograms (re-run selection with --from-cache, no inference)

Changing weights or members requires a new inference pass; changing the
threshold grid does not (--from-cache).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.data.class_mapping import CLOSED_DAMAGE_CLASSES, REPORTED_DAMAGE_CLASSES, UNIFIED_DAMAGE_CLASSES  # noqa: F401
from src.eval.probability_calibration import FORMAT_ENSEMBLE, apply_platt, load_platt, sha256_file

CLASSES = list(UNIFIED_DAMAGE_CLASSES)
PLATT_JSON = Path("eval_curves") / "probability_calibration.json"
MEMBER_KEYS = {"name", "run_dir", "weight", "checkpoint", "calibration"}


# --------------------------------------------------------------------------- #
# Arguments and members
# --------------------------------------------------------------------------- #
def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Platt-calibrated ensemble of P1ML runs.")
    p.add_argument("--member", nargs="+", action="append", default=[], metavar="KEY=VALUE",
                   help="name=.. run_dir=.. [weight=1] [checkpoint=best.pt] [calibration=<scripts/27 json>]")
    p.add_argument("--members-file", help="JSON file with {'members': [{...}, ...]} (same keys as --member)")
    p.add_argument("--name", help="ensemble name (default: member names joined by '+')")
    p.add_argument("--output-dir", help="default: outputs/ensembles/<name>")
    p.add_argument("--split", help="default: data.dacl10k.val_split of the first member")
    p.add_argument("--seed", type=int, help="split seed of the 2-fold protocol (default: project.seed, as scripts/20)")
    p.add_argument("--limit", type=int, help="debug: first N images (the fold halves then differ from scripts/20)")
    p.add_argument("--bins", type=int, default=1000)
    p.add_argument("--thresholds", type=float, nargs="+",
                   help="candidate thresholds (default 0.02..0.98 step 0.01; must be multiples of 1/bins)")
    p.add_argument("--from-cache", action="store_true", help="reuse ensemble_cache.npz, skip inference")
    return p.parse_args(argv)


def parse_members(args) -> list[dict]:
    raw = []
    if args.members_file:
        raw += json.loads(Path(args.members_file).read_text(encoding="utf-8"))["members"]
    for tokens in args.member:
        item = {}
        for token in tokens:
            if "=" not in token:
                raise SystemExit(f"--member expects KEY=VALUE tokens, got '{token}'.")
            key, value = token.split("=", 1)
            item[key.strip()] = value.strip()
        raw.append(item)
    if len(raw) < 1:
        raise SystemExit("Pass at least one --member (or --members-file).")
    members = []
    for item in raw:
        unknown = set(item) - MEMBER_KEYS
        if unknown or "name" not in item or "run_dir" not in item:
            raise SystemExit(f"Member {item}: required name, run_dir; allowed keys {sorted(MEMBER_KEYS)}.")
        run_dir = Path(item["run_dir"])
        members.append({
            "name": str(item["name"]),
            "run_dir": run_dir,
            "weight": float(item.get("weight", 1.0)),
            "checkpoint": str(item.get("checkpoint", "best.pt")),
            "calibration": Path(item["calibration"]) if item.get("calibration") else run_dir / PLATT_JSON,
        })
    names = [m["name"] for m in members]
    if len(set(names)) != len(names):
        raise SystemExit(f"Member names must be unique: {names}.")
    if any(m["weight"] <= 0 for m in members):
        raise SystemExit("Member weights must be > 0.")
    total = sum(m["weight"] for m in members)
    for m in members:
        m["weight_normalized"] = m["weight"] / total
        if not m["calibration"].is_file():
            raise SystemExit(f"{m['name']}: {m['calibration']} not found (run scripts/27 on {m['run_dir']}).")
        m["a"], m["b"], m["platt_payload"] = load_platt(m["calibration"], CLASSES)
    return members


def load_step17():
    path = Path(__file__).with_name("17_calibrate_multilabel_thresholds.py")
    spec = importlib.util.spec_from_file_location("step17_calibration", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --------------------------------------------------------------------------- #
# Metrics from histograms (pure numpy, testable)
# --------------------------------------------------------------------------- #
def counts_at(hist_pos: np.ndarray, hist_neg: np.ndarray, thresholds: np.ndarray) -> dict:
    """hist_* [C, bins] -> TP, FP, FN [C, T] at p >= t (exact for t multiple of 1/bins)."""
    bins = hist_pos.shape[-1]
    k = np.rint(np.asarray(thresholds) * bins).astype(int)
    tail_pos = np.concatenate([np.cumsum(hist_pos[:, ::-1], axis=1)[:, ::-1], np.zeros((hist_pos.shape[0], 1))], 1)
    tail_neg = np.concatenate([np.cumsum(hist_neg[:, ::-1], axis=1)[:, ::-1], np.zeros((hist_neg.shape[0], 1))], 1)
    tp, fp = tail_pos[:, k], tail_neg[:, k]
    fn = hist_pos.sum(1, keepdims=True) - tp
    return {"tp": tp, "fp": fp, "fn": fn}


def far_at(max_prob: np.ndarray, gt_present: np.ndarray, thresholds: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """max_prob, gt_present [N, C] -> (false alarms [C, T], empty-GT images [C])."""
    empty = ~gt_present
    alarms = (max_prob[:, :, None] >= np.asarray(thresholds)[None, None, :]) & empty[:, :, None]
    return alarms.sum(0), empty.sum(0)


def metrics_table(c: dict, alarms: np.ndarray, n_empty: np.ndarray) -> dict:
    tp, fp, fn = c["tp"], c["fp"], c["fn"]
    with np.errstate(invalid="ignore", divide="ignore"):
        return {
            "dice": np.where(2 * tp + fp + fn > 0, 2 * tp / (2 * tp + fp + fn), 0.0),
            "iou": np.where(tp + fp + fn > 0, tp / (tp + fp + fn), 0.0),
            "precision": np.where(tp + fp > 0, tp / (tp + fp), 0.0),
            "recall": np.where(tp + fn > 0, tp / (tp + fn), 0.0),
            "far": np.where(n_empty[:, None] > 0, alarms / np.maximum(n_empty[:, None], 1), 0.0),
            "n_false_alarms": alarms, "n_empty": np.repeat(n_empty[:, None], tp.shape[1], 1),
            "support": tp + fn,
        }


def select(dice: np.ndarray) -> np.ndarray:
    """Per class argmax over thresholds; ties -> highest threshold (scripts/17 rule)."""
    rounded = np.round(dice, 6)
    return rounded.shape[1] - 1 - np.argmax(rounded[:, ::-1], axis=1)


def evaluate_source(hist_pos, hist_neg, max_prob, gt_present, halves, thresholds) -> tuple[list, dict]:
    """hist_* [2, C, bins] per half; returns per-class rows and the frozen full-validation indices."""
    rows = []

    def table(sel_halves):
        hp, hn = hist_pos[sel_halves].sum(0), hist_neg[sel_halves].sum(0)
        idx = np.concatenate([halves[h] for h in sel_halves])
        alarms, n_empty = far_at(max_prob[idx], gt_present[idx], thresholds)
        return metrics_table(counts_at(hp, hn, thresholds), alarms, n_empty)

    full = table([0, 1])
    frozen = select(full["dice"])
    plans = [("full_validation_optimistic", full, frozen)]
    for fold, (cal, test) in enumerate(((0, 1), (1, 0))):
        chosen = select(table([cal])["dice"])
        plans.append((f"fold{fold}_test", table([test]), chosen))
    for protocol, m, chosen in plans:
        for c, name in enumerate(CLASSES):
            j = chosen[c]
            rows.append({"protocol": protocol, "class": name, "threshold": float(thresholds[j]),
                         **{k: float(m[k][c, j]) for k in ("dice", "iou", "precision", "recall", "far",
                                                          "n_false_alarms", "n_empty", "support")}})
    return rows, frozen


def summarize(per_class: pd.DataFrame) -> pd.DataFrame:
    out = []
    for (source, protocol), g in per_class.groupby(["source", "protocol"], sort=False):
        g = g[~g["class"].isin(CLOSED_DAMAGE_CLASSES)]          # closed classes: per-class rows only
        present = g[g["support"] > 0]
        out.append({"source": source, "protocol": protocol,
                    **{f"macro_{k}_present": present[k].mean() for k in ("dice", "iou", "precision", "recall")},
                    "false_alarm_rate_empty_gt": g["n_false_alarms"].sum() / max(g["n_empty"].sum(), 1)})
    summary = pd.DataFrame(out)
    folds = summary[summary["protocol"].str.startswith("fold")]
    mean = folds.groupby("source", sort=False).mean(numeric_only=True).reset_index()
    mean.insert(1, "protocol", "mean_of_folds_test")
    return pd.concat([summary, mean], ignore_index=True)


# --------------------------------------------------------------------------- #
# Inference
# --------------------------------------------------------------------------- #
def run_inference(args, members, cache_path: Path) -> dict:
    from src.data.dacl10k import list_samples
    from src.utils import get_device, load_config, multilabel_patch_cfg, seed_everything

    step17 = load_step17()
    device = get_device()
    cfgs = [load_config(m["run_dir"] / "config.yaml") for m in members]
    roots = {str(cfg.data.dacl10k.root) for cfg in cfgs}
    if len(roots) != 1:
        raise SystemExit(f"Members use different DACL10K roots: {sorted(roots)}.")
    seed = int(args.seed if args.seed is not None else cfgs[0].project.seed)
    seed_everything(seed)
    split = args.split or str(cfgs[0].data.dacl10k.val_split)

    for m, cfg in zip(members, cfgs):
        m["patch_cfg"] = multilabel_patch_cfg(cfg)
        m["predict"], ckpt_path, ckpt, family = step17.build_predictor(cfg, m["run_dir"], m["checkpoint"], device)
        m["checkpoint_sha256"] = sha256_file(ckpt_path)
        m["checkpoint_epoch"] = str(ckpt.get("epoch", "unknown"))
        expected = m["platt_payload"].get("checkpoint", {})
        if expected.get("sha256") and expected["sha256"] != m["checkpoint_sha256"]:
            raise SystemExit(f"{m['name']}: {m['calibration']} was fitted on another checkpoint.")
        if str(expected.get("epoch")) != m["checkpoint_epoch"]:
            raise SystemExit(f"{m['name']}: calibration epoch {expected.get('epoch')} != checkpoint {m['checkpoint_epoch']}.")
        print(f"[ensemble] {m['name']}: {family} | {ckpt_path.name} epoch {m['checkpoint_epoch']} "
              f"| weight {m['weight_normalized']:.3f} | stride {m['patch_cfg'].eval_stride}")

    samples = list_samples(cfgs[0].data.dacl10k.root, split)
    if args.limit:
        samples = samples[: args.limit]
    n, n_cls, bins = len(samples), len(CLASSES), int(args.bins)
    permutation = np.random.default_rng(seed).permutation(n)          # same halves as scripts/20
    half_of = np.zeros(n, dtype=np.int64)
    half_of[np.sort(permutation[n // 2:])] = 1
    sources = [m["name"] for m in members] + ["ensemble"]
    hist_pos = np.zeros((len(sources), 2, n_cls, bins), dtype=np.int64)
    hist_neg = np.zeros_like(hist_pos)
    max_prob = np.zeros((len(sources), n, n_cls), dtype=np.float32)
    gt_present = np.zeros((n, n_cls), dtype=bool)
    names = []

    for i, sample in enumerate(samples):
        image, mask = step17.load_rgb_and_mask(sample)
        gt = mask > 0
        gt_present[i] = gt.reshape(n_cls, -1).any(1)
        ensemble = None
        calibrated = []
        for m in members:
            p = apply_platt(m["predict"](image, m["patch_cfg"]), m["a"], m["b"])
            if p.shape != gt.shape:
                raise RuntimeError(f"{m['name']}: shape {p.shape} != mask {gt.shape} for {sample[0]}")
            calibrated.append(p)
            ensemble = m["weight_normalized"] * p if ensemble is None else ensemble + m["weight_normalized"] * p
        for s, p in enumerate([*calibrated, np.clip(ensemble, 0.0, 1.0)]):
            index = np.minimum((p * bins).astype(np.int64), bins - 1)
            for c in range(n_cls):
                hist_pos[s, half_of[i], c] += np.bincount(index[c][gt[c]], minlength=bins)
                hist_neg[s, half_of[i], c] += np.bincount(index[c][~gt[c]], minlength=bins)
            max_prob[s, i] = p.reshape(n_cls, -1).max(1)
        names.append(Path(sample[0]).name)
        if (i + 1) % 25 == 0 or i + 1 == n:
            print(f"[ensemble] processed {i + 1}/{n} images")

    payload = {"hist_pos": hist_pos, "hist_neg": hist_neg, "max_prob": max_prob, "gt_present": gt_present,
               "half_of": half_of, "sources": np.asarray(sources), "image_names": np.asarray(names),
               "split": np.asarray(split), "seed": np.asarray(seed),
               "member_checkpoint_sha256": np.asarray([m["checkpoint_sha256"] for m in members]),
               "member_checkpoint_epoch": np.asarray([m["checkpoint_epoch"] for m in members]),
               "member_weights": np.asarray([m["weight_normalized"] for m in members])}
    np.savez_compressed(cache_path, **payload)
    print(f"[ensemble] cache written to {cache_path}")
    return payload


# --------------------------------------------------------------------------- #
def main(argv=None) -> None:
    args = parse_args(argv)
    members = parse_members(args)
    name = args.name or "+".join(m["name"] for m in members)
    output_dir = Path(args.output_dir) if args.output_dir else Path("outputs") / "ensembles" / name.replace("+", "_")
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = output_dir / "ensemble_cache.npz"

    if args.from_cache:
        data = dict(np.load(cache_path, allow_pickle=False))
        cached = [str(s) for s in data["sources"].tolist()][:-1]
        if cached != [m["name"] for m in members] or not np.allclose(
                data["member_weights"], [m["weight_normalized"] for m in members]):
            raise SystemExit(f"{cache_path} was built with members {cached} / weights {data['member_weights']}: "
                             "members and weights must match (re-run without --from-cache).")
        for m, sha, ep in zip(members, data["member_checkpoint_sha256"], data["member_checkpoint_epoch"]):
            m["checkpoint_sha256"], m["checkpoint_epoch"] = str(sha), str(ep)
        print(f"[ensemble] reusing {cache_path}")
    else:
        data = run_inference(args, members, cache_path)

    bins = data["hist_pos"].shape[-1]
    thresholds = np.asarray(args.thresholds or np.round(np.arange(0.02, 0.981, 0.01), 2), dtype=np.float64)
    if not np.allclose(thresholds * bins, np.rint(thresholds * bins)):
        raise SystemExit(f"Thresholds must be multiples of 1/{bins} (exact histogram counts).")
    half_of = data["half_of"]
    halves = (np.flatnonzero(half_of == 0), np.flatnonzero(half_of == 1))

    rows, frozen = [], {}
    for s, source in enumerate([str(x) for x in data["sources"].tolist()]):
        src_rows, frozen_idx = evaluate_source(data["hist_pos"][s], data["hist_neg"][s], data["max_prob"][s],
                                               data["gt_present"], halves, thresholds)
        rows += [{"source": source, **r} for r in src_rows]
        frozen[source] = {c: float(thresholds[frozen_idx[k]]) for k, c in enumerate(CLASSES)}
    per_class = pd.DataFrame(rows)
    summary = summarize(per_class)
    per_class.to_csv(output_dir / "ensemble_metrics_per_class.csv", index=False)
    summary.to_csv(output_dir / "ensemble_summary.csv", index=False)

    ens = per_class[(per_class["source"] == "ensemble") & (per_class["protocol"] == "full_validation_optimistic")]
    payload = {
        "format": FORMAT_ENSEMBLE,
        "name": name,
        "classes": CLASSES,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "combination": "p_ens = sum_m w_m * sigmoid(a_mc * logit(p_m) + b_mc), weights normalised to 1",
        "members": [{
            "name": m["name"], "run_dir": str(m["run_dir"].resolve()), "checkpoint": m["checkpoint"],
            "checkpoint_epoch": m["checkpoint_epoch"], "checkpoint_sha256": m["checkpoint_sha256"],
            "calibration_file": str(m["calibration"].resolve()), "calibration_sha256": sha256_file(m["calibration"]),
            "weight_input": m["weight"], "weight": m["weight_normalized"],
            "params": m["platt_payload"]["params"],
        } for m in members],
        "split": str(data["split"]), "n_images": int(len(half_of)), "seed": int(data["seed"]),
        "threshold_grid": thresholds.tolist(),
        "selection": "per class argmax Dice (dataset-level pixel counts) on the whole split; ties -> highest threshold",
        "thresholds": frozen["ensemble"],
        "thresholds_members_calibrated_scale": {k: v for k, v in frozen.items() if k != "ensemble"},
        "validation_metrics_optimistic": {r["class"]: {k: r[k] for k in ("dice", "iou", "precision", "recall", "far")}
                                          for r in ens.to_dict("records")},
        "summary": summary.to_dict("records"),
        "note": ("thresholds are selected on the same split they are evaluated on (OPTIMISTIC, as scripts/17); "
                 "the honest estimate is protocol 'mean_of_folds_test'. Held-out use (Lincoln Street) must "
                 "reuse these frozen thresholds without re-selection."),
    }
    (output_dir / "ensemble.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    pd.set_option("display.width", 200)
    print("\n--- macro metrics (calibrated members vs ensemble) ---")
    print(summary.round(4).to_string(index=False))
    test = per_class[per_class["protocol"].str.startswith("fold")]
    print("\n--- per-class Dice, mean of fold test halves ---")
    print(test.pivot_table(index="class", columns="source", values="dice", sort=False).loc[CLASSES].round(4).to_string())
    print(f"\n[ensemble] frozen thresholds: {frozen['ensemble']}")
    print(f"[ensemble] artifacts in {output_dir.resolve()}")


if __name__ == "__main__":
    main()
