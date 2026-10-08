#!/usr/bin/env python
"""Step 23 - write per-photo multilabel probability maps for damage3d.

Runs a trained P1ML checkpoint on the bridge photos with the same
sliding-window protocol as 12_evaluate_dacl10k_multilabel_sliding.py
(native resolution, Gaussian blending, independent sigmoid per channel) and
writes one ``<stem>.npz`` per photo in the format read by the damage3d
``files`` provider (keys ``probs`` (6, H', W') and ``classes``).

Camera selection reuses damage3d (same XML, same ordering, same flags), so the
same ``--camera`` / ``--start-image`` / ``--end-image`` arguments select the
same photos in both steps. Pixels are read in the raw sensor frame (EXIF
orientation ignored), which is the frame of the Metashape calibration.

Usage
-----
    python scripts/23_infer_probability_maps.py \
        --run-dir outputs/runs/<p1ml_run> --start-image 76 --end-image 86

Then:
    python -m damage3d --probability-source files \
        --probabilities-dir <maps dir printed by this script> \
        --start-image 76 --end-image 86 --thresholds "<frozen thresholds>"
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from damage3d.classes import DAMAGE_CLASSES  # noqa: E402
from damage3d.metashape_xml import parse_cameras_xml, read_camera_list, select_cameras, selectable_cameras  # noqa: E402
from damage3d.paths import resolve_path, resolve_project_root  # noqa: E402
from damage3d.providers import save_probability_npz  # noqa: E402
from src.data.class_mapping import UNIFIED_DAMAGE_CLASSES  # noqa: E402
from src.data.transforms import IMAGENET_MEAN, IMAGENET_STD  # noqa: E402
from src.engine import load_checkpoint  # noqa: E402
from src.eval.sliding_window import _sliding_positions, predict_sliding_window_multilabel  # noqa: E402
from src.models.unet import build_model  # noqa: E402
from src.utils import get_device, load_config  # noqa: E402

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
CALIBRATED_THRESHOLDS = Path("eval_multilabel_sliding_calibrated") / "thresholds_per_class_validation_calibrated.json"
MANIFEST_NAME = "maps_manifest.json"


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Write damage3d probability maps with a trained P1ML checkpoint.")
    p.add_argument("--run-dir", required=True, help="P1ML run folder with config.yaml and the checkpoint")
    p.add_argument("--checkpoint", default="best.pt")
    p.add_argument("--project-root", help="bridge_model folder (same default as damage3d)")
    p.add_argument("--cameras-xml", help="default: damage3d default (cameras.xml)")
    p.add_argument("--images-dir", help="default: damage3d default images folder")
    p.add_argument("--output-dir", help="maps folder; default <project-root>/damage3d_maps/<run name>")
    g = p.add_argument_group("camera selection (same rules as damage3d; end is EXCLUSIVE)")
    g.add_argument("--camera", action="append")
    g.add_argument("--camera-list")
    g.add_argument("--start-image", type=int)
    g.add_argument("--end-image", type=int)
    g.add_argument("--limit-images", type=int)
    g = p.add_argument_group("inference and storage")
    g.add_argument("--stride", type=int, help="default: data.p1ml_patch.eval_stride of the run")
    g.add_argument("--batch-size", type=int, help="default: data.p1ml_patch.eval_batch_size of the run")
    g.add_argument("--downsample", type=int, default=2,
                   help="store maps at 1/N resolution (area average); 1 = full resolution")
    g.add_argument("--dtype", choices=("float16", "uint8"), default="float16")
    g.add_argument("--overwrite", action="store_true", help="recompute maps that already exist")
    g.add_argument("--dry-run", action="store_true", help="check inputs and print the plan; write nothing")
    return p.parse_args(argv)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def find_images(images_dir: Path, stems: list[str]) -> dict[str, Path | None]:
    index: dict[str, Path] = {}
    if images_dir is not None and images_dir.is_dir():
        for entry in images_dir.iterdir():
            if entry.is_file() and entry.suffix.lower() in IMAGE_EXTS:
                index.setdefault(entry.stem, entry)
    return {s: index.get(s) for s in stems}


def read_raw_rgb(path: Path) -> np.ndarray:
    """Read pixels in the stored (sensor) order: EXIF orientation is NOT applied."""
    data = np.fromfile(str(path), dtype=np.uint8)  # works with non-ASCII Windows paths
    bgr = cv2.imdecode(data, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if bgr is None:
        raise RuntimeError(f"Unreadable image: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def downsample(probs: torch.Tensor, factor: int) -> np.ndarray:
    """Area-average a (C, H, W) map to round(H/f) x round(W/f)."""
    if factor == 1:
        return probs.numpy()
    _, h, w = probs.shape
    size = (max(1, round(h / factor)), max(1, round(w / factor)))
    return F.interpolate(probs[None], size=size, mode="area")[0].numpy()


def frozen_thresholds(run_dir: Path) -> tuple[str | None, Path]:
    """Return the damage3d --thresholds string from the run's validation calibration, if any."""
    path = run_dir / CALIBRATED_THRESHOLDS
    if not path.is_file():
        return None, path
    thr = json.loads(path.read_text(encoding="utf-8"))["thresholds"]
    return ",".join(f"{c}={thr[f'threshold_{c}']}" for c in DAMAGE_CLASSES), path


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.downsample < 1:
        raise SystemExit("--downsample must be >= 1.")
    if tuple(UNIFIED_DAMAGE_CLASSES) != tuple(DAMAGE_CLASSES):
        raise SystemExit(f"Class order mismatch: {UNIFIED_DAMAGE_CLASSES} vs {DAMAGE_CLASSES}.")

    run_dir = Path(args.run_dir).resolve()
    cfg = load_config(run_dir / "config.yaml")
    if int(cfg.model.classes) != len(DAMAGE_CLASSES):
        raise SystemExit(f"{run_dir.name} emits {cfg.model.classes} channels: not a P1ML multilabel run.")
    checkpoint_path = run_dir / args.checkpoint
    if not checkpoint_path.is_file():
        raise SystemExit(f"Checkpoint not found: {checkpoint_path}")

    root = resolve_project_root(args.project_root)
    xml_path = resolve_path(root, args.cameras_xml, "cameras_xml")
    images_dir = resolve_path(root, args.images_dir, "images_dir")
    out_dir = resolve_path(root, args.output_dir, None) or (root / "damage3d_maps" / run_dir.name)

    project = parse_cameras_xml(xml_path)
    selected = select_cameras(
        selectable_cameras(project), project.cameras,
        camera=args.camera,
        camera_list=read_camera_list(Path(args.camera_list)) if args.camera_list else None,
        start_image=args.start_image, end_image=args.end_image, limit_images=args.limit_images,
    )
    found = find_images(images_dir, [c.stem for _, c in selected])
    missing = [s for s, p in found.items() if p is None]
    if missing:
        raise SystemExit(f"{len(missing)} selected photos not found in {images_dir}: {missing[:10]}")

    patch = cfg.data.p1ml_patch
    stride = int(args.stride or patch.eval_stride)
    batch_size = int(args.batch_size or patch.eval_batch_size)
    thr_string, thr_path = frozen_thresholds(run_dir)

    print(f"Run: {run_dir.name} | checkpoint {checkpoint_path.name}")
    print(f"Cameras: {len(selected)} ({selected[0][1].label} .. {selected[-1][1].label}) from {xml_path}")
    print(f"Images: {images_dir}")
    print(f"Maps -> {out_dir} (downsample {args.downsample}, {args.dtype})")
    calib0 = project.calibration_for(selected[0][1])
    n_patches = (len(_sliding_positions(calib0.height, int(patch.patch_size), stride))
                 * len(_sliding_positions(calib0.width, int(patch.patch_size), stride)))
    print(f"Sliding window: patch {patch.patch_size}, stride {stride}, batch {batch_size}, blend {patch.blend_mode} "
          f"| photo {calib0.width}x{calib0.height} -> {n_patches} patches per photo")
    print(f"Frozen validation thresholds: {thr_string or f'not found ({thr_path})'}")
    if args.dry_run:
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / MANIFEST_NAME
    checkpoint_sha = sha256_file(checkpoint_path)
    settings = {
        "run_dir": str(run_dir), "checkpoint": checkpoint_path.name, "checkpoint_sha256": checkpoint_sha,
        "model": {k: cfg.model.get(k) for k in ("arch", "encoder", "classes", "decoder_attention_type")},
        "patch_size": int(patch.patch_size), "stride": stride, "blend_mode": str(patch.blend_mode),
        "normalization": {"mean": list(IMAGENET_MEAN), "std": list(IMAGENET_STD)},
        "downsample": args.downsample, "dtype": args.dtype, "classes": list(DAMAGE_CLASSES),
        "exif_orientation": "ignored (raw sensor frame)",
    }
    manifest = {"settings": settings, "maps": {}}
    if manifest_path.is_file():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous.get("settings") != settings and not args.overwrite:
            raise SystemExit(f"{manifest_path} was written with different settings; "
                             f"use another --output-dir or --overwrite.")
        if previous.get("settings") == settings:
            manifest["maps"] = previous.get("maps", {})

    device = get_device()
    model_cfg = dict(cfg["model"])
    model_cfg["encoder_weights"] = None  # weights come from the checkpoint; avoid any download
    model = build_model(type(cfg)(model_cfg)).to(device)
    checkpoint = load_checkpoint(checkpoint_path, model, device=device)
    model.eval()
    print(f"Checkpoint epoch {checkpoint.get('epoch', 'unknown')}, sha256 {checkpoint_sha[:12]}")

    for i, (position, cam) in enumerate(selected, 1):
        target = out_dir / f"{cam.stem}.npz"
        if target.is_file() and cam.stem in manifest["maps"] and not args.overwrite:
            print(f"[{i}/{len(selected)}] {cam.stem}: exists, skipped")
            continue
        calib = project.calibration_for(cam)
        image = read_raw_rgb(found[cam.stem])
        if (image.shape[1], image.shape[0]) != (calib.width, calib.height):
            raise SystemExit(f"{found[cam.stem].name}: size {image.shape[1]}x{image.shape[0]} "
                             f"!= calibration {calib.width}x{calib.height}.")
        t0 = time.perf_counter()
        probs = predict_sliding_window_multilabel(
            model=model, image=image, device=device, patch_size=int(patch.patch_size),
            stride=stride, batch_size=batch_size, mean=IMAGENET_MEAN, std=IMAGENET_STD,
            n_classes=len(DAMAGE_CLASSES), blend_mode=str(patch.blend_mode),
        )
        stored = downsample(probs, args.downsample)
        tmp = out_dir / f"{cam.stem}.tmp.npz"
        save_probability_npz(tmp, stored, dtype=args.dtype)
        os.replace(tmp, target)
        seconds = time.perf_counter() - t0
        manifest["maps"][cam.stem] = {
            "position": position, "label": cam.label, "image": found[cam.stem].name,
            "image_size": [calib.width, calib.height], "map_size": [stored.shape[2], stored.shape[1]],
            "max_prob": [round(float(m), 4) for m in stored.reshape(len(DAMAGE_CLASSES), -1).max(1)],
            "seconds": round(seconds, 1),
        }
        manifest["updated_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        tmp_manifest = manifest_path.with_suffix(".json.tmp")
        tmp_manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        os.replace(tmp_manifest, manifest_path)
        print(f"[{i}/{len(selected)}] {cam.stem}: {stored.shape[2]}x{stored.shape[1]} map in {seconds:.1f} s")

    print(f"\nMaps folder: {out_dir}")
    if thr_string:
        print(f'damage3d thresholds (frozen, from {thr_path.name}): --thresholds "{thr_string}"')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
