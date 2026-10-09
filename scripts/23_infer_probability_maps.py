#!/usr/bin/env python
"""Step 23 - write per-photo multilabel probability maps for damage3d.

Runs a trained P1ML checkpoint on the bridge photos with the same
sliding-window protocol as 12_evaluate_dacl10k_multilabel_sliding.py
(native resolution, Gaussian blending, independent sigmoid per channel) and
writes one ``<stem>.npz`` per photo in the format read by the damage3d
``files`` provider (keys ``probs`` (6, H', W') and ``classes``).

With ``--save-overlays`` it also writes ``<maps dir>/overlays/<stem>_overlay.jpg``
for visual inspection: the photo, all classes together, and one panel per
class thresholded with the frozen validation thresholds. Overlays are rendered
from the STORED map (the exact input of damage3d), so they can also be
produced later for maps that already exist, without re-running the model.

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

from damage3d.classes import CLASS_COLORS, DAMAGE_CLASSES  
from damage3d.fusion import parse_thresholds  
from damage3d.metashape_xml import parse_cameras_xml, read_camera_list, select_cameras, selectable_cameras  
from damage3d.paths import resolve_path, resolve_project_root  
from damage3d.providers import FileProvider, save_probability_npz  
from src.data.class_mapping import UNIFIED_DAMAGE_CLASSES  
from src.data.transforms import IMAGENET_MEAN, IMAGENET_STD  
from src.engine import load_checkpoint  
from src.eval.sliding_window import _sliding_positions, predict_sliding_window_multilabel  
from src.models.unet import build_model 
from src.utils import get_device, load_config  

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
    g = p.add_argument_group("visual inspection")
    g.add_argument("--save-overlays", action="store_true",
                   help="write overlays/<stem>_overlay.jpg (also for maps that already exist)")
    g.add_argument("--overlay-thresholds",
                   help="'0.5' or 'crack=0.7,...'; default: frozen validation thresholds of the run")
    g.add_argument("--overlay-width", type=int, default=1200, help="width of each panel in pixels")
    g.add_argument("--overlay-alpha", type=float, default=0.5, help="opacity of the class colors")
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


def read_checked(path: Path, calib) -> np.ndarray:
    image = read_raw_rgb(path)
    if (image.shape[1], image.shape[0]) != (calib.width, calib.height):
        raise SystemExit(f"{path.name}: size {image.shape[1]}x{image.shape[0]} "
                         f"!= calibration {calib.width}x{calib.height}.")
    return image


def downsample(probs: torch.Tensor, factor: int) -> np.ndarray:
    """Area-average a (C, H, W) map to round(H/f) x round(W/f)."""
    if factor == 1:
        return probs.numpy()
    _, h, w = probs.shape
    size = (max(1, round(h / factor)), max(1, round(w / factor)))
    return F.interpolate(probs[None], size=size, mode="area")[0].numpy()


def render_overlay(image_rgb: np.ndarray, probs: np.ndarray, thresholds: np.ndarray, title: str,
                   panel_width: int = 1200, alpha: float = 0.5) -> tuple[np.ndarray, dict]:
    """Return a BGR grid [photo | all classes | one panel per class] and the panel boxes.

    ``probs`` is the (C, h, w) stored map (any resolution with the photo aspect
    ratio); it is resized to the panel size with bilinear interpolation and
    thresholded per class. Classes are independent: in the "all classes"
    panel a pixel with several labels gets the mean of their colors and every
    class keeps its own outline, so overlaps stay visible.
    """
    h, w = image_rgb.shape[:2]
    pw = int(panel_width)
    ph = max(1, round(h * pw / w))
    base = cv2.resize(image_rgb, (pw, ph), interpolation=cv2.INTER_AREA).astype(np.float32)
    p = np.stack([cv2.resize(np.nan_to_num(c, nan=0.0).astype(np.float32), (pw, ph),
                             interpolation=cv2.INTER_LINEAR) for c in probs])
    labels = p >= thresholds[:, None, None]
    colors = np.array([CLASS_COLORS[c] for c in DAMAGE_CLASSES], dtype=np.float32)

    def blend(mask: np.ndarray, color: np.ndarray) -> np.ndarray:
        """Alpha-blend ``color`` (RGB triple or per-pixel (ph, pw, 3)) where ``mask`` is True."""
        out = base.copy()
        tint = color[mask] if color.ndim == 3 else color
        out[mask] = (1.0 - alpha) * out[mask] + alpha * tint
        return out

    def outline(panel: np.ndarray, mask: np.ndarray, color) -> np.ndarray:
        """Draw the mask boundary in the full class color (readable on similar backgrounds)."""
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(panel, contours, -1, tuple(float(v) for v in color), 1, cv2.LINE_AA)
        return panel

    n_labels = labels.sum(0)
    any_label = n_labels > 0
    mean_color = np.tensordot(labels.transpose(1, 2, 0).astype(np.float32), colors, axes=1)
    mean_color /= np.maximum(n_labels, 1)[..., None]
    combined = blend(any_label, mean_color)
    for c in range(len(DAMAGE_CLASSES)):
        outline(combined, labels[c], colors[c])
    panels = [(base, "photo (raw sensor frame)"),
              (combined, f"all classes  {100 * any_label.mean():.2f}% px")]
    for c, name in enumerate(DAMAGE_CLASSES):
        panels.append((outline(blend(labels[c], colors[c]), labels[c], colors[c]),
                       f"{name}  thr {thresholds[c]:.2f}  {100 * labels[c].mean():.2f}% px"))

    bar = 34
    cols, rows = 4, 2
    grid = np.full((bar + rows * (ph + bar), cols * pw, 3), 255, dtype=np.uint8)
    cv2.putText(grid, title, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
    boxes = {}
    for k, (panel, text) in enumerate(panels):
        r, c = divmod(k, cols)
        y0, x0 = bar + r * (ph + bar) + bar, c * pw
        grid[y0:y0 + ph, x0:x0 + pw] = cv2.cvtColor(np.clip(panel, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        if k >= 2:
            sw = colors[k - 2][::-1].tolist()
            cv2.rectangle(grid, (x0 + 6, y0 - 26), (x0 + 24, y0 - 8), sw, -1)
        cv2.putText(grid, text, (x0 + (30 if k >= 2 else 8), y0 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 0, 0), 1, cv2.LINE_AA)
        boxes["photo" if k == 0 else "all" if k == 1 else DAMAGE_CLASSES[k - 2]] = (y0, x0, ph, pw)
    return grid, boxes


def write_jpeg(path: Path, image_bgr: np.ndarray, quality: int = 90) -> None:
    ok, buf = cv2.imencode(".jpg", image_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError(f"JPEG encoding failed for {path}")
    tmp = path.with_name(path.stem + ".tmp.jpg")
    buf.tofile(str(tmp))  # works with non-ASCII Windows paths
    os.replace(tmp, path)


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
    overlay_thr = None
    if args.save_overlays:
        source = args.overlay_thresholds or thr_string
        if source is None:
            raise SystemExit(f"--save-overlays needs thresholds: {thr_path} not found, "
                             f"pass --overlay-thresholds explicitly.")
        overlay_thr = parse_thresholds(source)
        print(f"Overlays -> {out_dir / 'overlays'} (thresholds "
              f"{', '.join(f'{c}={t:.2f}' for c, t in zip(DAMAGE_CLASSES, overlay_thr))})")
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

    overlay_dir = out_dir / "overlays"
    if args.save_overlays:
        overlay_dir.mkdir(exist_ok=True)
    stored_maps = FileProvider(out_dir)

    def save_overlay(cam, image: np.ndarray, calib) -> None:
        pmap = stored_maps.get(cam.stem, calib.width, calib.height)  # same reader and checks as damage3d
        grid, _ = render_overlay(image, pmap.probs, overlay_thr,
                                 title=f"{cam.label} | {run_dir.name} | map {pmap.probs.shape[2]}x{pmap.probs.shape[1]}",
                                 panel_width=args.overlay_width, alpha=args.overlay_alpha)
        write_jpeg(overlay_dir / f"{cam.stem}_overlay.jpg", grid)

    device = get_device()
    model_cfg = dict(cfg["model"])
    model_cfg["encoder_weights"] = None  # weights come from the checkpoint; avoid any download
    model = build_model(type(cfg)(model_cfg)).to(device)
    checkpoint = load_checkpoint(checkpoint_path, model, device=device)
    model.eval()
    print(f"Checkpoint epoch {checkpoint.get('epoch', 'unknown')}, sha256 {checkpoint_sha[:12]}")

    for i, (position, cam) in enumerate(selected, 1):
        target = out_dir / f"{cam.stem}.npz"
        calib = project.calibration_for(cam)
        if target.is_file() and cam.stem in manifest["maps"] and not args.overwrite:
            if args.save_overlays:
                save_overlay(cam, read_checked(found[cam.stem], calib), calib)
                print(f"[{i}/{len(selected)}] {cam.stem}: map exists, overlay written")
            else:
                print(f"[{i}/{len(selected)}] {cam.stem}: exists, skipped")
            continue
        image = read_checked(found[cam.stem], calib)
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
        if args.save_overlays:
            save_overlay(cam, image, calib)
        print(f"[{i}/{len(selected)}] {cam.stem}: {stored.shape[2]}x{stored.shape[1]} map in {seconds:.1f} s"
              f"{' + overlay' if args.save_overlays else ''}")

    print(f"\nMaps folder: {out_dir}")
    if args.save_overlays:
        print(f"Overlays folder: {overlay_dir}")
    if thr_string:
        print(f'damage3d thresholds (frozen, from {thr_path.name}): --thresholds "{thr_string}"')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
