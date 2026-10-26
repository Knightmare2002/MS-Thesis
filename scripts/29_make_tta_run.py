"""Create a test-time-augmentation (TTA) view of an existing P1ML run.

Every evaluation script (12, 17, 20, 25, 27, 28, 23) reads `<run>/config.yaml`
and `<run>/best.pt` and writes its outputs inside the run directory. To compare
TTA with plain inference without overwriting anything, this script creates a
sibling run directory

    outputs/runs/<run>_tta_<tag>/
        config.yaml          copy of the source config + data.p1ml_patch.tta
        best.pt              hard link (same volume) or copy of the source checkpoint
        class_weights.json   copied when present (informational)
        dataset_summary.json copied when present (informational)
        tta_source.json      provenance: source run, checkpoint SHA-256, TTA spec

The weights are unchanged (same SHA-256): TTA only changes inference, i.e. the
probability maps, hence thresholds (17), curves (25) and Platt parameters (27)
must be recomputed on the new directory.

    python scripts/29_make_tta_run.py --run-dir outputs/runs/p1ml_t12_upernet_convnext_base_512_ade20k --hflip --scales 0.75 1.0 1.25
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.eval.sliding_window import tta_variants  # noqa: E402


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", required=True, type=Path)
    p.add_argument("--checkpoint", default="best.pt")
    p.add_argument("--hflip", action="store_true", help="add the horizontally flipped view")
    p.add_argument("--vflip", action="store_true", help="add the vertically flipped view (not recommended: "
                                                        "bridge photos have a gravity direction)")
    p.add_argument("--scales", type=float, nargs="+", default=[1.0], help="patch rescaling factors, e.g. 0.75 1.0 1.25")
    p.add_argument("--tag", default=None, help="suffix of the new run (default derived from the TTA spec)")
    p.add_argument("--copy", action="store_true", help="copy best.pt instead of hard-linking it")
    return p.parse_args(argv)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main(argv=None) -> int:
    args = parse_args(argv)
    src = args.run_dir.resolve()
    ckpt = src / args.checkpoint
    if not (src / "config.yaml").is_file() or not ckpt.is_file():
        raise SystemExit(f"{src} must contain config.yaml and {args.checkpoint}.")

    tta = {"hflip": bool(args.hflip), "vflip": bool(args.vflip), "scales": [float(s) for s in args.scales]}
    scales, flips = tta_variants(tta)
    n_views = len(scales) * len(flips)
    if n_views == 1:
        raise SystemExit("The TTA spec has a single view: add --hflip and/or more --scales.")
    tag = args.tag or ("h" if args.hflip else "") + ("v" if args.vflip else "") + (
        "s" + "-".join(f"{s:g}" for s in scales) if scales != (1.0,) else "")
    dst = src.parent / f"{src.name}_tta_{tag}"
    if dst.exists():
        raise SystemExit(f"{dst} already exists: delete it or pass another --tag.")

    with open(src / "config.yaml", "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    if cfg["data"].get("p1ml_patch") is None:
        raise SystemExit("TTA is implemented for single-head P1ML runs (data.p1ml_patch) only.")
    cfg["data"]["p1ml_patch"]["tta"] = tta
    cfg.setdefault("model", {})["tta_source_run"] = src.name

    dst.mkdir(parents=True)
    with open(dst / "config.yaml", "w", encoding="utf-8") as fh:
        yaml.safe_dump(cfg, fh, sort_keys=False, allow_unicode=True)
    target = dst / args.checkpoint
    linked = False
    if not args.copy:
        try:
            os.link(ckpt, target)
            linked = True
        except OSError:
            pass
    if not linked:
        shutil.copy2(ckpt, target)
    for name in ("class_weights.json", "dataset_summary.json"):
        if (src / name).is_file():
            shutil.copy2(src / name, dst / name)

    provenance = {"source_run": src.name, "source_checkpoint": str(ckpt), "checkpoint": args.checkpoint,
                  "checkpoint_sha256": sha256_file(target), "checkpoint_link": "hardlink" if linked else "copy",
                  "tta": tta, "n_views_per_patch": n_views,
                  "note": "same weights as the source run; probabilities, thresholds and Platt "
                          "parameters must be recomputed (scripts 12, 17, 25, 27)."}
    with open(dst / "tta_source.json", "w", encoding="utf-8") as fh:
        json.dump(provenance, fh, indent=2)

    print(f"[tta] {dst} created ({n_views} views/patch, checkpoint {'hard-linked' if linked else 'copied'})")
    print(f"[tta] next: scripts 12 -> 17 -> 25 -> 27 with --run-dir {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
