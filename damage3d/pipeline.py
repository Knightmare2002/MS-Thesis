"""Orchestration: selection -> projection -> visibility -> sampling -> fusion -> export."""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .camera_model import CameraProjector
from .classes import DAMAGE_CLASSES, NUM_CLASSES
from .export import export_run, ply_dtype
from .frames import resolve_frame
from .fusion import CAMERA_STAT_FIELDS, ChunkAccumulator, parse_thresholds
from .metashape_xml import parse_cameras_xml, read_camera_list, select_cameras, selectable_cameras
from .ply_io import PlyPointSource, make_selection
from .providers import FileProvider, SyntheticProvider
from .run_state import RunDirectory, file_identity, normalise, short_hash, write_json_atomic
from .sampling import sample_probabilities

log = logging.getLogger("damage3d")


class SimulatedInterruption(RuntimeError):
    """Raised by the test hook to emulate a crash between two chunk commits."""


@dataclass
class RunConfig:
    project_root: Path
    images_dir: Path | None
    cameras_xml: Path
    point_cloud: Path
    mesh: Path | None
    marker_reference: Path | None
    output_dir: Path | None
    runs_dir: Path
    probability_source: str = "synthetic"
    probabilities_dir: Path | None = None
    synthetic_pattern: str = "regions"
    synthetic_downsample: int = 8
    synthetic_mask_pattern: str = "{stem}_mask_test.png"
    synthetic_mask_classes: tuple[str, ...] = ("crack",)
    camera: list[str] | None = None
    camera_list: Path | None = None
    start_image: int | None = None
    end_image: int | None = None
    limit_images: int | None = None
    max_points: int | None = None
    sample_mode: str = "random"
    seed: int = 0
    point_chunk_size: int = 2_000_000
    camera_batch_size: int = 4
    point_frame: str = "auto"
    occlusion: str = "mesh"
    occlusion_tolerance: float | None = None
    occlusion_rel_tolerance: float = 0.0
    tolerance_sample_size: int = 20_000
    interpolation: str = "bilinear"
    border_margin_px: float = 0.0
    min_depth: float = 0.0
    vote_threshold: float = 0.5
    thresholds: str | None = None
    min_views: int = 1
    export_scope: str = "all"
    csv_max_points: int = 200_000
    dry_run: bool = False
    resume: bool = False
    test_interrupt_after_writes: int | None = None
    warnings: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- #
def _build_provider(cfg: RunConfig):
    if cfg.probability_source == "synthetic":
        return SyntheticProvider(pattern=cfg.synthetic_pattern, downsample=cfg.synthetic_downsample,
                                 mask_dir=cfg.project_root, mask_pattern=cfg.synthetic_mask_pattern,
                                 mask_classes=tuple(cfg.synthetic_mask_classes))
    if cfg.probability_source == "files":
        if cfg.probabilities_dir is None:
            raise RuntimeError("--probability-source files requires --probabilities-dir.")
        if not cfg.probabilities_dir.is_dir():
            raise FileNotFoundError(f"Probabilities directory not found: {cfg.probabilities_dir}")
        return FileProvider(cfg.probabilities_dir)
    raise ValueError(f"Unknown probability source '{cfg.probability_source}'.")


def _find_images(images_dir: Path | None, stems: list[str]) -> dict[str, Path | None]:
    if images_dir is None or not images_dir.is_dir():
        return {s: None for s in stems}
    exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
    index: dict[str, Path] = {}
    for entry in images_dir.iterdir():
        if entry.is_file() and entry.suffix.lower() in exts:
            index.setdefault(entry.stem, entry)
    return {s: index.get(s) for s in stems}


def _check_images(cfg: RunConfig, selected, project) -> list[dict]:
    """Size/EXIF check of the photos (not needed by synthetic/file providers)."""
    from PIL import Image
    rows = []
    found = _find_images(cfg.images_dir, [c.stem for _, c in selected])
    for _, cam in selected:
        calib = project.calibration_for(cam)
        path = found[cam.stem]
        row = {"stem": cam.stem, "image": str(path) if path else None}
        if path is not None:
            with Image.open(path) as im:
                row["size"] = im.size
                orientation = im.getexif().get(274, 1)
            row["exif_orientation"] = orientation
            if im.size != (calib.width, calib.height):
                raise RuntimeError(f"{path.name}: size {im.size} != calibration {(calib.width, calib.height)}.")
            if orientation not in (None, 1):
                cfg.warnings.append(f"{path.name}: EXIF orientation {orientation}; maps must use the raw pixel frame.")
        rows.append(row)
    missing = [r["stem"] for r in rows if r["image"] is None]
    if missing:
        cfg.warnings.append(f"{len(missing)} selected photos not found in {cfg.images_dir} "
                            f"(not needed by this provider): {missing[:5]}{' ...' if len(missing) > 5 else ''}")
    return rows


def _chunk_ranges(n: int, size: int) -> np.ndarray:
    starts = np.arange(0, n, size, dtype=np.int64)
    return np.column_stack((starts, np.minimum(starts + size, n)))


def _compute_chunk_bounds(ply, selection, ranges, frame) -> tuple[np.ndarray, np.ndarray]:
    centers = np.zeros((len(ranges), 3))
    radii = np.zeros(len(ranges))
    t0 = time.time()
    for j, (s, e) in enumerate(ranges):
        xyz = ply.xyz(selection.source_index(int(s), int(e)))
        lo, hi = xyz.min(axis=0), xyz.max(axis=0)
        centers[j] = frame.to_chunk(((lo + hi) / 2)[None, :])[0]
        radii[j] = frame.radius_to_chunk(float(np.linalg.norm(hi - lo) / 2)) * (1 + 1e-6) + 1e-9
        if (j + 1) % 50 == 0:
            log.info("Bounds pass: %d/%d chunks (%.0f s)", j + 1, len(ranges), time.time() - t0)
    return centers, radii


def _estimate_tolerance(cfg, occ, ply, selection) -> dict:
    n = min(cfg.tolerance_sample_size, selection.size)
    positions = (np.arange(n, dtype=np.int64) * selection.size) // n
    if selection.indices is None:
        src = positions
    else:
        src = selection.indices[positions]
    stats = occ.surface_deviation_stats(ply.xyz(src))
    floor = 1e-4 * occ.diagonal
    stats["floor_1e-4_mesh_diagonal"] = floor
    stats["auto_tolerance"] = max(stats.get("p95", 0.0), floor)
    return stats


# --------------------------------------------------------------------------- #
def plan(cfg: RunConfig) -> dict:
    """Everything that can be checked without the mesh and without writing."""
    for name, path in (("cameras XML", cfg.cameras_xml), ("point cloud", cfg.point_cloud)):
        if not path.is_file():
            raise FileNotFoundError(f"Missing input - {name}: {path}")
    if cfg.occlusion == "mesh" and (cfg.mesh is None or not cfg.mesh.is_file()):
        raise FileNotFoundError(f"Missing input - mesh for occlusion: {cfg.mesh} (or pass --occlusion none).")

    project = parse_cameras_xml(cfg.cameras_xml)
    cfg.warnings.extend(project.warnings)
    ordered = selectable_cameras(project)
    listed = read_camera_list(cfg.camera_list) if cfg.camera_list else None
    selected = select_cameras(ordered, project.cameras, camera=cfg.camera, camera_list=listed,
                              start_image=cfg.start_image, end_image=cfg.end_image,
                              limit_images=cfg.limit_images)
    for _, cam in selected:
        calib = project.calibration_for(cam)
        if calib.unsupported_nonzero:
            raise RuntimeError(f"Camera {cam.label}: non-zero {list(calib.unsupported_nonzero)} not supported.")

    ply = PlyPointSource(cfg.point_cloud)
    selection = make_selection(ply.n_points, cfg.max_points, cfg.sample_mode, cfg.seed)
    provider = _build_provider(cfg)
    missing_maps = provider.check_available([c.stem for _, c in selected])
    if missing_maps:
        raise FileNotFoundError(
            f"Missing probability maps for {len(missing_maps)} cameras, e.g. {missing_maps[:5]} "
            f"(expected {getattr(provider, 'dir', cfg.project_root)})."
        )
    images = _check_images(cfg, selected, project)
    ranges = _chunk_ranges(selection.size, cfg.point_chunk_size)
    n_sel = selection.size
    acc_bytes = n_sel * (NUM_CLASSES * (4 + 2 + 2) + 2)
    ply_bytes = n_sel * ply_dtype("<u4").itemsize
    chunk_ram = cfg.point_chunk_size * (24 * 3 + 8 * 6 + NUM_CLASSES * 8 * 2 + 64)
    return {
        "project": project, "ordered": ordered, "selected": selected, "ply": ply,
        "selection": selection, "provider": provider, "ranges": ranges, "images": images,
        "estimates": {
            "accumulator_disk_GB": acc_bytes / 1e9,
            "output_ply_GB_if_scope_all": ply_bytes / 1e9,
            "peak_RAM_per_chunk_GB_approx": chunk_ram / 1e9,
            "n_chunks": int(len(ranges)),
            "n_camera_batches": int(math.ceil(len(selected) / cfg.camera_batch_size)),
        },
    }


def _print_plan(cfg: RunConfig, p: dict) -> None:
    project, ply, selection, selected = p["project"], p["ply"], p["selection"], p["selected"]
    ct = project.chunk_transform
    log.info("DRY RUN - nothing is written.")
    log.info("Project root: %s", cfg.project_root)
    log.info("Cameras XML: %s (%d cameras, %d aligned+enabled, %d calibrated sensors)",
             cfg.cameras_xml, len(project.cameras), len(p["ordered"]), len(project.calibrations))
    log.info("Chunk transform: %s, identity=%s, scale=%.9g", ct.source, ct.is_identity, ct.scale)
    if project.reference_crs:
        log.info("Chunk reference CRS: %s", project.reference_crs[:120])
    log.info("Point cloud: %s (%d points, %s, record %d B, trailing bytes %d)", cfg.point_cloud,
             ply.n_points, [n for n, _ in ply.header.properties], ply.header.dtype.itemsize, ply.trailing_bytes)
    log.info("Point selection: %s", selection.describe())
    log.info("Mesh: %s (occlusion=%s)", cfg.mesh, cfg.occlusion)
    log.info("Selected cameras: %d -> first %s, last %s", len(selected), selected[0][1].label, selected[-1][1].label)
    log.info("Provider: %s", p["provider"].describe() if cfg.probability_source == "synthetic" else "files")
    log.info("Estimates: %s", p["estimates"])
    if not ct.is_identity and cfg.point_frame == "auto":
        log.info("Frame: will be decided with the marker-to-mesh test (%s).", cfg.marker_reference)
    for w in cfg.warnings:
        log.warning(w)


# --------------------------------------------------------------------------- #
def run(cfg: RunConfig) -> dict:
    p = plan(cfg)
    if cfg.dry_run:
        _print_plan(cfg, p)
        p["ply"].close()
        return {"dry_run": True, "n_selected_cameras": len(p["selected"]),
                "n_selected_points": p["selection"].size, "estimates": p["estimates"]}

    project, selected, ply, selection, provider, ranges = (
        p["project"], p["selected"], p["ply"], p["selection"], p["provider"], p["ranges"])
    if isinstance(provider, FileProvider):
        provider.register([c.stem for _, c in selected])

    occ = None
    if cfg.occlusion == "mesh":
        from .mesh_visibility import MeshOcclusion
        t0 = time.time()
        occ = MeshOcclusion(cfg.mesh)
        log.info("Mesh loaded: %d vertices, %d triangles (%.1f s)", occ.n_vertices, occ.n_triangles, time.time() - t0)
    elif cfg.occlusion == "none":
        cfg.warnings.append("Occlusion test DISABLED: hidden points can receive probabilities.")
    else:
        raise ValueError(f"Unknown occlusion mode '{cfg.occlusion}'.")

    frame, frame_record = resolve_frame(cfg.point_frame, project.chunk_transform, cfg.marker_reference,
                                        occ.distance if occ else None)
    log.info("Source frame: %s (%s)", frame_record["resolved"], frame_record["method"])

    tolerance_record = {"mode": cfg.occlusion}
    abs_tol = 0.0
    if occ is not None:
        if cfg.occlusion_tolerance is None:
            tolerance_record.update(_estimate_tolerance(cfg, occ, ply, selection))
            abs_tol = float(tolerance_record["auto_tolerance"])
            tolerance_record["source"] = "auto: max(P95 point-to-mesh distance, 1e-4 * mesh diagonal)"
        else:
            abs_tol = float(cfg.occlusion_tolerance)
            tolerance_record["source"] = "CLI"
        tolerance_record.update({"abs_tolerance": abs_tol, "rel_tolerance": cfg.occlusion_rel_tolerance,
                                 "units": "source-frame units (PLY/OBJ)"})
        log.info("Occlusion tolerance: %.6g source units (%s)", abs_tol, tolerance_record["source"])

    camera_rows = [{"position": pos, "label": cam.label, "camera_id": cam.camera_id, "sensor_id": cam.sensor_id}
                   for pos, cam in selected]
    inputs = {"cameras_xml": file_identity(cfg.cameras_xml, full_hash=True),
              "point_cloud": {**file_identity(cfg.point_cloud), "header": list(ply.header.lines)}}
    if occ is not None:
        inputs["mesh"] = file_identity(cfg.mesh)
    if frame_record.get("method") == "marker-to-mesh distance test":
        inputs["marker_reference"] = file_identity(cfg.marker_reference, full_hash=True)
    fingerprint = {
        "inputs": inputs,
        "cameras": {"selected": camera_rows},
        "points": selection.describe(),
        "accumulation": {"frame": frame_record, "source_to_chunk": frame.source_to_chunk,
                         "occlusion": tolerance_record, "interpolation": cfg.interpolation,
                         "border_margin_px": cfg.border_margin_px, "min_depth": cfg.min_depth,
                         "vote_threshold": cfg.vote_threshold, "classes": list(DAMAGE_CLASSES)},
        "layout": {"point_chunk_size": cfg.point_chunk_size, "camera_batch_size": cfg.camera_batch_size},
        "provider": provider.describe(),
    }
    run_path = cfg.output_dir
    if run_path is None:
        tag = selected[0][1].stem if len(selected) == 1 else f"{selected[0][1].stem}-{selected[-1][1].stem}_n{len(selected)}"
        tag += f"_pts{selection.size}"
        run_path = cfg.runs_dir / f"{tag}_{short_hash({k: fingerprint[k] for k in ('cameras', 'points', 'accumulation', 'layout', 'provider')})}"
    rd = RunDirectory(run_path)
    for name, path in (("cameras XML", cfg.cameras_xml), ("point cloud", cfg.point_cloud), ("mesh", cfg.mesh)):
        if path is not None and rd.path.resolve() in Path(path).resolve().parents:
            raise RuntimeError(f"The {name} ({path}) is inside the run directory; choose another --output-dir "
                               "so that original exports can never be touched.")
    rd.prepare(fingerprint, {"project_root": str(cfg.project_root), "created": time.strftime("%Y-%m-%dT%H:%M:%S")},
               cfg.resume)
    _attach_file_log(rd.path / "run.log")
    log.info("Run directory: %s", rd.path)
    if selection.indices is not None and not rd.selection_path.is_file():
        np.save(rd.selection_path, selection.indices)
    _write_selected_cameras(rd.path / "selected_cameras.csv", camera_rows)

    if rd.chunks_path.is_file():
        with np.load(rd.chunks_path) as d:
            centers, radii = d["centers"], d["radii"]
    else:
        centers, radii = _compute_chunk_bounds(ply, selection, ranges, frame)
        tmp = rd.chunks_path.with_name("chunks.npz.tmp")
        with tmp.open("wb") as f:
            np.savez(f, ranges=ranges, centers=centers, radii=radii)
        import os
        os.replace(tmp, rd.chunks_path)

    _accumulate(cfg, rd, project, selected, ply, selection, provider, ranges, centers, radii, frame, occ, abs_tol)

    thresholds = parse_thresholds(cfg.thresholds)
    meta = {"provider_type": provider.describe()["type"], "synthetic": provider.is_synthetic,
            "note": ("SYNTHETIC probabilities - NOT an AI prediction; thresholds are test values."
                     if provider.is_synthetic else "Probabilities read from files."),
            "frame": frame_record, "occlusion": tolerance_record, "warnings": cfg.warnings,
            "interpolation": cfg.interpolation, "n_cameras": len(selected), "run_dir": str(rd.path)}
    summary = export_run(rd, ranges, selection, ply, thresholds, cfg.min_views, cfg.export_scope,
                         cfg.csv_max_points, len(selected), camera_rows, meta)
    ply.close()
    log.info("States: %s", summary["state_counts"])
    log.info("Labels: %s", summary["label_counts"])
    log.info("Outputs: %s", rd.out_dir)
    for w in cfg.warnings:
        log.warning(w)
    return summary


def _accumulate(cfg, rd, project, selected, ply, selection, provider, ranges, centers, radii, frame, occ, abs_tol):
    n_cams = len(selected)
    B = cfg.camera_batch_size
    n_batches = int(math.ceil(n_cams / B))
    done = rd.completed_batches()
    if done:
        log.info("Resuming: %d/%d camera batches already committed.", done, n_batches)
    projectors = []
    for _, cam in selected:
        calib = project.calibration_for(cam)
        projectors.append(CameraProjector(cam.camera_to_chunk, calib, cfg.min_depth, cfg.border_margin_px))
    centers_src = [frame.to_source(pr.center_chunk[None, :])[0] for pr in projectors]
    writes = 0
    for b in range(done, n_batches):
        t0 = time.time()
        cams = list(range(b * B, min((b + 1) * B, n_cams)))
        maps: dict[int, object] = {}
        touched = 0
        for j, (s, e) in enumerate(ranges):
            touching = [k for k in cams if projectors[k].sphere_may_be_visible(centers[j], radii[j])]
            if not touching:
                continue
            path = rd.acc_path(j)
            n = int(e - s)
            acc = ChunkAccumulator.load(path) if path.is_file() else ChunkAccumulator.zeros(n, n_cams)
            if acc.last_batch >= b:
                continue  # committed before the interruption
            xyz_src = ply.xyz(selection.source_index(int(s), int(e)))
            xyz_chunk = frame.to_chunk(xyz_src)
            for k in touching:
                _observe(k, selected[k][1], projectors[k], centers_src[k], xyz_src, xyz_chunk, acc, maps,
                         provider, occ, abs_tol, cfg)
            acc.last_batch = b
            acc.save_atomic(path)
            touched += 1
            writes += 1
            if cfg.test_interrupt_after_writes is not None and writes >= cfg.test_interrupt_after_writes:
                raise SimulatedInterruption(f"Simulated interruption after {writes} chunk commits.")
        rd.mark_batch_done(b, n_batches)
        log.info("Batch %d/%d committed: cameras %s, %d chunks updated (%.1f s)", b + 1, n_batches,
                 [selected[k][1].stem for k in cams], touched, time.time() - t0)


def _observe(k, cam, projector, center_src, xyz_src, xyz_chunk, acc, maps, provider, occ, abs_tol, cfg):
    calib = projector.calib
    proj = projector.project(xyz_chunk)
    inside = np.nonzero(proj.in_image)[0]
    stats = acc.camera_stats[k]
    stats[0] += len(xyz_chunk)
    stats[1] += int(proj.in_front.sum())
    stats[2] += len(inside)
    if occ is not None and inside.size:
        vis = occ.visible(xyz_src[inside], center_src, abs_tol, cfg.occlusion_rel_tolerance)
        stats[3] += int((~vis).sum())
        inside = inside[vis]
    if inside.size == 0:
        return
    if k not in maps:
        maps[k] = provider.get(cam.stem, calib.width, calib.height)
        maps[k].validate(calib.width, calib.height, cam.stem)
    probs, valid = sample_probabilities(maps[k], proj.u[inside], proj.v[inside], calib.width, calib.height,
                                        cfg.interpolation)
    idx = inside[valid]
    probs = probs[valid]
    acc.add_view(idx, probs, cfg.vote_threshold)
    stats[4] += len(idx)
    stats[len(CAMERA_STAT_FIELDS):] += (np.nan_to_num(probs, nan=-1.0) >= cfg.vote_threshold).sum(axis=0)


def _write_selected_cameras(path: Path, rows: list[dict]) -> None:
    import csv
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["order", "position", "label", "camera_id", "sensor_id"])
        w.writeheader()
        for k, row in enumerate(rows):
            w.writerow({"order": k, **row})


def _attach_file_log(path: Path) -> None:
    for h in log.handlers:
        if isinstance(h, logging.FileHandler) and Path(h.baseFilename) == path.resolve():
            return
    for h in [h for h in log.handlers if isinstance(h, logging.FileHandler)]:
        log.removeHandler(h)
        h.close()
    handler = logging.FileHandler(path, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    log.addHandler(handler)
