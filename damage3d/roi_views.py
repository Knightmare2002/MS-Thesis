"""ROI coverage, ground sampling distance and Labelme ROI pre-fill, per camera.

For a 3D region of interest (ROI) exported from the dense cloud, every aligned
camera is tested with the same geometry as the fusion pipeline
(``CameraProjector`` + ``MeshOcclusion``, same frame resolution, same automatic
occlusion tolerance). Nothing in the fusion inputs is changed: the ROI is an
evaluation-only object.

    python -m damage3d.roi_views --project-root <bridge_model> \
        --roi roi_points.ply --out roi_views/ \
        --labelme-dir labelme_json/ --sample-per-stratum 12

Inputs
------
* ``--roi``: binary little-endian PLY with x, y, z in the SAME frame as the
  dense cloud / OBJ (e.g. the target portion segmented in CloudCompare and saved
  without global shift). nx, ny, nz are used when present, otherwise normals are
  estimated by local PCA (k nearest neighbours, needs SciPy). The script stops if
  the ROI points do not lie on the mesh (typical cause: CloudCompare global shift).

Per camera (``roi_views.csv``)
------------------------------
* ``n_visible``: ROI points inside the image and not occluded by the mesh;
  ``frac_roi_visible`` = n_visible / ROI points used.
* ``image_coverage``: fraction of the image covered by the ROI footprint
  (visible points rasterised on a grid of ``--cell-px`` pixels, morphological
  closing sized on the projected point spacing; holes are NOT filled because
  they are mostly occluders in front of the ROI).
* GSD, in source-frame units per pixel (metres only if the export is metric):
  ``gsd_frontal = Z * s / f`` with Z the camera depth (chunk frame), s the
  chunk-to-source scale, f the focal length in pixels; along the surface
  ``gsd_eff = gsd_frontal / max(|cos theta|, cos(--max-incidence-deg))``, theta
  the angle between the surface normal and the viewing ray. Medians and p10/p90
  over the visible ROI points; ``gsd_eff_map_median`` multiplies by
  ``--map-downsample`` (resolution of the stored probability maps).
* ``eligible`` = image_coverage >= --min-image-coverage and n_visible >=
  --min-visible-points. Fix both BEFORE looking at images or damage, and report
  them; ``summary.json`` lists the coverage quantiles to choose them.
* ``stratum``: GSD tercile (near / mid / far) of gsd_eff_median among the
  eligible cameras; ``sampled``: seeded random draw of --sample-per-stratum
  cameras per stratum (written to ``test_views_sample.txt``).

Labelme pre-fill (``--labelme-dir``)
------------------------------------
For the chosen cameras a Labelme JSON is written with one ``roi`` polygon per
footprint component and one ``ignore`` polygon per hole (occluder or missing
surface; check them while annotating). The shapes carry the description
``auto:roi_views`` so they can be replaced later. Existing JSON files are never
overwritten; with ``--replace-roi`` only the ``roi`` shapes and the ``auto:``
shapes are replaced and every damage annotation is kept.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .camera_model import CameraProjector
from .frames import resolve_frame
from .metashape_xml import parse_cameras_xml, read_camera_list, select_cameras, selectable_cameras
from .paths import resolve_path, resolve_project_root
from .ply_io import PlyPointSource

log = logging.getLogger("damage3d")

AUTO_TAG = "auto:roi_views"
LABELME_VERSION = "5.5.0"
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".tif", ".tiff")


# --------------------------------------------------------------------------- #
# ROI loading and local geometry
# --------------------------------------------------------------------------- #
def load_roi(path: Path, max_points: int | None, seed: int) -> tuple[np.ndarray, np.ndarray | None, int]:
    """Return (xyz source frame, normals or None, total ROI points) with a seeded subsample."""
    ply = PlyPointSource(path)
    n = ply.n_points
    if n == 0:
        raise RuntimeError(f"ROI PLY {path} has no points.")
    if max_points is not None and n > max_points:
        idx = np.sort(np.random.default_rng(seed).choice(n, size=max_points, replace=False))
    else:
        idx = slice(0, n)
    xyz = ply.xyz(idx)
    normals = ply.normals(idx).astype(np.float64) if ply.has_normals else None
    ply.close()
    if normals is not None:
        norm = np.linalg.norm(normals, axis=1)
        if not np.all(np.isfinite(norm)) or np.median(norm) < 0.5:
            log.warning("ROI normals look invalid (median norm %.3g): they will be estimated.", np.median(norm))
            normals = None
        else:
            normals = normals / np.maximum(norm, 1e-12)[:, None]
    return xyz, normals, n


def _kdtree(xyz: np.ndarray):
    try:
        from scipy.spatial import cKDTree
    except ImportError as exc:  # pragma: no cover
        raise ImportError("SciPy is required for ROI spacing/normal estimation (pip install scipy).") from exc
    return cKDTree(xyz)


def nn_spacing(xyz: np.ndarray, sample: int = 20_000, seed: int = 0) -> float:
    """Median nearest-neighbour distance (source units) on a seeded sample of the ROI."""
    tree = _kdtree(xyz)
    rng = np.random.default_rng(seed)
    q = xyz if len(xyz) <= sample else xyz[rng.choice(len(xyz), sample, replace=False)]
    d, _ = tree.query(q, k=2)
    return float(np.median(d[:, 1]))


def estimate_normals(xyz: np.ndarray, k: int = 16) -> np.ndarray:
    """Unoriented PCA normals (smallest eigenvector of the k-NN covariance)."""
    tree = _kdtree(xyz)
    k = min(k, len(xyz))
    _, nbr = tree.query(xyz, k=k)
    out = np.empty_like(xyz)
    for s in range(0, len(xyz), 200_000):
        e = min(s + 200_000, len(xyz))
        p = xyz[nbr[s:e]]                               # (m, k, 3)
        p = p - p.mean(axis=1, keepdims=True)
        cov = np.einsum("mki,mkj->mij", p, p)
        _, vec = np.linalg.eigh(cov)                    # ascending eigenvalues
        out[s:e] = vec[:, :, 0]
    return out


# --------------------------------------------------------------------------- #
# Footprint rasterisation and polygons
# --------------------------------------------------------------------------- #
def footprint_mask(u: np.ndarray, v: np.ndarray, width: int, height: int, cell_px: int,
                   spacing_px: float) -> np.ndarray:
    """Binary grid (cell_px pixels per cell) of the projected visible points, closed on the point spacing."""
    gw, gh = int(np.ceil(width / cell_px)), int(np.ceil(height / cell_px))
    grid = np.zeros((gh, gw), np.uint8)
    if u.size == 0:
        return grid
    cu = np.clip((u // cell_px).astype(np.int64), 0, gw - 1)
    cv_ = np.clip((v // cell_px).astype(np.int64), 0, gh - 1)
    grid[cv_, cu] = 1
    k = int(np.clip(2 * np.ceil(1.5 * spacing_px / cell_px) + 1, 3, 51))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    return cv2.morphologyEx(grid, cv2.MORPH_CLOSE, kernel)


def mask_to_polygons(grid: np.ndarray, cell_px: int, width: int, height: int, min_area_frac: float,
                     eps_cells: float = 1.0) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Outer contours -> roi polygons, holes -> ignore polygons (pixel coordinates, Labelme convention)."""
    contours, hierarchy = cv2.findContours(grid, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    rois, holes = [], []
    if hierarchy is None:
        return rois, holes
    min_cells = min_area_frac * grid.size
    for c, h in zip(contours, hierarchy[0]):
        if cv2.contourArea(c) < max(min_cells, 1.0):
            continue
        poly = cv2.approxPolyDP(c, eps_cells, True).reshape(-1, 2).astype(np.float64)
        if len(poly) < 3:
            continue
        # Cell (i, j) covers pixels [j*cell, (j+1)*cell): scale the contour to pixel units.
        poly = (poly + 0.5) * cell_px
        poly[:, 0] = np.clip(poly[:, 0], 0, width - 1)
        poly[:, 1] = np.clip(poly[:, 1], 0, height - 1)
        (holes if h[3] >= 0 else rois).append(poly)
    return rois, holes


# --------------------------------------------------------------------------- #
# Per-camera analysis
# --------------------------------------------------------------------------- #
@dataclass
class ViewResult:
    row: dict
    rois: list[np.ndarray]
    holes: list[np.ndarray]
    grid: np.ndarray | None


def analyse_camera(cam, calib, projector: CameraProjector, center_src: np.ndarray, xyz_src: np.ndarray,
                   xyz_chunk: np.ndarray, normals_src: np.ndarray, occ, abs_tol: float, rel_tol: float,
                   chunk_to_source_scale: float, spacing_src: float, opts) -> ViewResult:
    w, h = calib.width, calib.height
    row = {"label": cam.label, "stem": cam.stem, "image_w": w, "image_h": h, "n_roi_points_used": len(xyz_src)}
    proj = projector.project(xyz_chunk)
    inside = np.nonzero(proj.in_image)[0]
    row["n_in_image"] = int(inside.size)
    if occ is not None and inside.size:
        inside = inside[occ.visible(xyz_src[inside], center_src, abs_tol, rel_tol)]
    n_vis = int(inside.size)
    row["n_visible"] = n_vis
    row["frac_roi_visible"] = n_vis / max(len(xyz_src), 1)
    empty = {k: np.nan for k in ("gsd_frontal_median", "gsd_eff_median", "gsd_eff_p10", "gsd_eff_p90",
                                 "gsd_eff_map_median", "incidence_median_deg", "incidence_p90_deg",
                                 "depth_median")}
    if n_vis == 0:
        row.update(empty, image_coverage=0.0, n_roi_polygons=0, n_ignore_polygons=0)
        return ViewResult(row, [], [], None)

    depth_src = proj.depth[inside] * chunk_to_source_scale
    gsd_frontal = depth_src / calib.f
    ray = center_src[None, :] - xyz_src[inside]
    ray /= np.maximum(np.linalg.norm(ray, axis=1, keepdims=True), 1e-12)
    cos = np.abs(np.einsum("ij,ij->i", normals_src[inside], ray))
    cos_floor = np.cos(np.deg2rad(opts.max_incidence_deg))
    gsd_eff = gsd_frontal / np.maximum(cos, cos_floor)
    incidence = np.rad2deg(np.arccos(np.clip(cos, 0.0, 1.0)))
    g_med = float(np.median(gsd_frontal))
    row.update({
        "gsd_frontal_median": g_med,
        "gsd_eff_median": float(np.median(gsd_eff)),
        "gsd_eff_p10": float(np.percentile(gsd_eff, 10)),
        "gsd_eff_p90": float(np.percentile(gsd_eff, 90)),
        "gsd_eff_map_median": float(np.median(gsd_eff)) * opts.map_downsample,
        "incidence_median_deg": float(np.median(incidence)),
        "incidence_p90_deg": float(np.percentile(incidence, 90)),
        "depth_median": float(np.median(depth_src)),
    })
    if opts.mm_per_unit:
        for k in ("gsd_frontal_median", "gsd_eff_median", "gsd_eff_p90", "gsd_eff_map_median"):
            row[f"{k}_mm"] = row[k] * opts.mm_per_unit

    # Largest projected spacing (frontal direction): spacing / gsd_frontal pixels.
    spacing_px = spacing_src / g_med
    grid = footprint_mask(proj.u[inside], proj.v[inside], w, h, opts.cell_px, spacing_px)
    row["image_coverage"] = float(grid.sum()) * opts.cell_px ** 2 / (w * h)
    rois, holes = mask_to_polygons(grid, opts.cell_px, w, h, opts.min_component_frac)
    row["n_roi_polygons"], row["n_ignore_polygons"] = len(rois), len(holes)
    return ViewResult(row, rois, holes, grid)


# --------------------------------------------------------------------------- #
# Eligibility and stratified sampling
# --------------------------------------------------------------------------- #
def assign_strata(rows: list[dict], min_cov: float, min_pts: int, per_stratum: int, seed: int) -> dict:
    for r in rows:
        r["eligible"] = bool(r["image_coverage"] >= min_cov and r["n_visible"] >= min_pts)
        r["stratum"], r["sampled"] = "", False
    elig = [r for r in rows if r["eligible"]]
    info = {"n_cameras": len(rows), "n_eligible": len(elig)}
    if len(elig) < 3:
        return info
    g = np.array([r["gsd_eff_median"] for r in elig])
    q1, q2 = np.quantile(g, [1 / 3, 2 / 3])
    info["gsd_eff_tercile_edges"] = [float(q1), float(q2)]
    for r in elig:
        r["stratum"] = "near" if r["gsd_eff_median"] <= q1 else ("mid" if r["gsd_eff_median"] <= q2 else "far")
    if per_stratum > 0:
        rng = np.random.default_rng(seed)
        for s in ("near", "mid", "far"):
            members = sorted((r for r in elig if r["stratum"] == s), key=lambda r: r["stem"])
            take = rng.choice(len(members), size=min(per_stratum, len(members)), replace=False)
            for i in take:
                members[i]["sampled"] = True
        info["n_sampled"] = sum(r["sampled"] for r in rows)
    return info


# --------------------------------------------------------------------------- #
# Labelme output
# --------------------------------------------------------------------------- #
def _shape(label: str, poly: np.ndarray) -> dict:
    return {"label": label, "points": [[round(float(x), 1), round(float(y), 1)] for x, y in poly],
            "group_id": None, "description": AUTO_TAG, "shape_type": "polygon", "flags": {}, "mask": None}


def _image_path(stem: str, label: str, images_dir: Path | None, json_dir: Path) -> str:
    if images_dir is not None and images_dir.is_dir():
        for ext in IMAGE_EXTS:
            for cand in (images_dir / f"{stem}{ext}", images_dir / f"{stem}{ext.upper()}"):
                if cand.is_file():
                    return os.path.relpath(cand, json_dir)
    return label


def write_labelme(json_dir: Path, res: ViewResult, images_dir: Path | None, replace_roi: bool) -> str:
    """Return 'written', 'updated' or 'skipped_existing'."""
    json_dir.mkdir(parents=True, exist_ok=True)
    path = json_dir / f"{res.row['stem']}.json"
    auto = [_shape("roi", p) for p in res.rois] + [_shape("ignore", p) for p in res.holes]
    if path.is_file():
        if not replace_roi:
            return "skipped_existing"
        data = json.loads(path.read_text(encoding="utf-8"))
        kept = [s for s in data.get("shapes", [])
                if s.get("label") != "roi" and not str(s.get("description") or "").startswith("auto:")]
        data["shapes"] = auto + kept
        status = "updated"
    else:
        data = {"version": LABELME_VERSION, "flags": {}, "shapes": auto,
                "imagePath": _image_path(res.row["stem"], res.row["label"], images_dir, json_dir),
                "imageData": None, "imageHeight": int(res.row["image_h"]), "imageWidth": int(res.row["image_w"])}
        status = "written"
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    os.replace(tmp, path)
    return status


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
CSV_FIELDS = ["position", "label", "stem", "eligible", "stratum", "sampled", "image_coverage", "n_visible",
              "frac_roi_visible", "n_in_image", "n_roi_points_used", "gsd_frontal_median", "gsd_eff_median",
              "gsd_eff_p10", "gsd_eff_p90", "gsd_eff_map_median", "incidence_median_deg", "incidence_p90_deg",
              "depth_median", "n_roi_polygons", "n_ignore_polygons", "image_w", "image_h"]


def run(args, occ=None) -> dict:
    root = resolve_project_root(args.project_root)
    cameras_xml = resolve_path(root, args.cameras_xml, "cameras_xml")
    mesh_path = resolve_path(root, args.mesh, "mesh")
    marker = resolve_path(root, args.marker_reference, "marker_reference")
    roi_path = resolve_path(root, args.roi, None)
    out_dir = resolve_path(root, args.out, None)
    labelme_dir = resolve_path(root, args.labelme_dir, None)
    images_dir = resolve_path(root, args.images_dir, None)
    out_dir.mkdir(parents=True, exist_ok=True)

    project = parse_cameras_xml(cameras_xml)
    ordered = selectable_cameras(project)
    listed = read_camera_list(resolve_path(root, args.camera_list, None)) if args.camera_list else None
    selected = select_cameras(ordered, project.cameras, camera=args.camera, camera_list=listed)

    if occ is None and args.occlusion == "mesh":
        from .mesh_visibility import MeshOcclusion
        occ = MeshOcclusion(mesh_path)
    frame, frame_record = resolve_frame(args.point_frame, project.chunk_transform, marker,
                                        occ.distance if occ is not None else None)
    scale = frame.length_to_source(1.0)

    xyz_src, normals, n_total = load_roi(roi_path, args.max_roi_points, args.seed)
    normals_source = "roi_ply"
    if normals is None:
        normals, normals_source = estimate_normals(xyz_src, args.normal_k), f"pca_k{args.normal_k}"
    spacing = nn_spacing(xyz_src, seed=args.seed)
    xyz_chunk = frame.to_chunk(xyz_src)

    tol = {"mode": "none"}
    abs_tol = 0.0
    if occ is not None:
        dev = occ.surface_deviation_stats(xyz_src)
        floor = 1e-4 * occ.diagonal
        abs_tol = float(args.occlusion_tolerance) if args.occlusion_tolerance is not None else max(dev.get("p95", 0.0), floor)
        tol = {"mode": "mesh", "abs_tolerance": abs_tol, "rel_tolerance": args.occlusion_rel_tolerance,
               "roi_to_mesh_deviation": dev,
               "source": "CLI" if args.occlusion_tolerance is not None else "auto: max(P95 ROI-to-mesh, 1e-4 diag)"}
        if dev.get("median", 0.0) > 50 * floor:  # > 0.5% of the mesh diagonal
            raise RuntimeError(
                f"ROI points are far from the mesh (median {dev['median']:.4g} source units). The ROI PLY is "
                "probably not in the dense-cloud frame (CloudCompare global shift?). Re-export it without shift.")

    lo, hi = xyz_chunk.min(0), xyz_chunk.max(0)
    center, radius = (lo + hi) / 2, float(np.linalg.norm(hi - lo) / 2) * (1 + 1e-6) + 1e-9

    rows, results = [], {}
    t0 = time.time()
    for i, (pos, cam) in enumerate(selected, 1):
        calib = project.calibration_for(cam)
        pr = CameraProjector(cam.camera_to_chunk, calib, args.min_depth, 0.0)
        if not pr.sphere_may_be_visible(center, radius):
            row = {"label": cam.label, "stem": cam.stem, "image_w": calib.width, "image_h": calib.height,
                   "n_roi_points_used": len(xyz_src), "n_in_image": 0, "n_visible": 0, "frac_roi_visible": 0.0,
                   "image_coverage": 0.0, "n_roi_polygons": 0, "n_ignore_polygons": 0}
            res = ViewResult(row, [], [], None)
        else:
            center_src = frame.to_source(pr.center_chunk[None, :])[0]
            res = analyse_camera(cam, calib, pr, center_src, xyz_src, xyz_chunk, normals, occ, abs_tol,
                                 args.occlusion_rel_tolerance, scale, spacing, args)
        res.row["position"] = pos
        rows.append(res.row)
        results[cam.stem] = res
        if i % 25 == 0:
            log.info("%d/%d cameras (%.0f s)", i, len(selected), time.time() - t0)

    strata = assign_strata(rows, args.min_image_coverage, args.min_visible_points, args.sample_per_stratum, args.seed)

    extra = sorted({k for r in rows for k in r} - set(CSV_FIELDS))
    with (out_dir / "roi_views.csv").open("w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=CSV_FIELDS + extra, extrasaction="ignore")
        wr.writeheader()
        for r in rows:
            wr.writerow({k: r.get(k, "") for k in CSV_FIELDS + extra})
    sampled = [r["stem"] for r in rows if r.get("sampled")]
    if args.sample_per_stratum > 0:
        (out_dir / "test_views_sample.txt").write_text(
            f"# seed={args.seed} per_stratum={args.sample_per_stratum} min_image_coverage={args.min_image_coverage} "
            f"min_visible_points={args.min_visible_points}\n" + "\n".join(sampled) + "\n", encoding="utf-8")

    if args.save_masks:
        (out_dir / "masks").mkdir(exist_ok=True)
        for stem, res in results.items():
            if res.grid is not None and res.row.get("eligible"):
                cv2.imwrite(str(out_dir / "masks" / f"{stem}_roi_cells.png"), res.grid * 255)

    labelme = {}
    if labelme_dir is not None:
        which = args.labelme_cameras
        if which == "auto":
            which = "sampled" if args.sample_per_stratum > 0 else "eligible"
        for r in rows:
            if (which == "all" and r["n_visible"] > 0) or (which == "eligible" and r["eligible"]) or \
               (which == "sampled" and r.get("sampled")):
                res = results[r["stem"]]
                if res.rois:
                    labelme[r["stem"]] = write_labelme(labelme_dir, res, images_dir, args.replace_roi)

    cov = np.array([r["image_coverage"] for r in rows if r["n_visible"] > 0])
    summary = {
        "roi": {"path": str(roi_path), "n_points_total": n_total, "n_points_used": len(xyz_src),
                "normals": normals_source, "median_nn_spacing_source_units": spacing},
        "frame": frame_record, "chunk_to_source_scale": scale, "occlusion": tol,
        "units": "source-frame units per pixel" + (f" (x {args.mm_per_unit} mm/unit in *_mm columns)"
                                                   if args.mm_per_unit else " (metric scale not applied)"),
        "eligibility": {"min_image_coverage": args.min_image_coverage, "min_visible_points": args.min_visible_points},
        "strata": strata,
        "image_coverage_quantiles_of_cameras_seeing_roi": (
            {f"q{int(q * 100)}": float(np.quantile(cov, q)) for q in (0.1, 0.25, 0.5, 0.75, 0.9)} if cov.size else {}),
        "n_cameras_seeing_roi": int(cov.size),
        "labelme": {"dir": str(labelme_dir) if labelme_dir else None,
                    "counts": {s: list(labelme.values()).count(s) for s in set(labelme.values())}},
        "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log.info("Cameras seeing the ROI: %d, eligible: %d, sampled: %d", cov.size, strata["n_eligible"], len(sampled))
    return summary


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="damage3d.roi_views", description=__doc__.split("\n")[0],
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    g = p.add_argument_group("paths (relative to --project-root)")
    g.add_argument("--project-root")
    g.add_argument("--roi", required=True, help="ROI PLY (binary little endian, dense-cloud frame)")
    g.add_argument("--out", required=True, help="output folder (CSV, summary, sample list, masks)")
    g.add_argument("--cameras-xml")
    g.add_argument("--mesh")
    g.add_argument("--marker-reference")
    g.add_argument("--images-dir", help="photos folder, only to write a relative imagePath in Labelme JSON")
    g.add_argument("--labelme-dir", help="write/merge Labelme JSON with the projected roi polygons here")
    g = p.add_argument_group("cameras")
    g.add_argument("--camera", action="append")
    g.add_argument("--camera-list")
    g = p.add_argument_group("geometry (same defaults as python -m damage3d)")
    g.add_argument("--point-frame", choices=("auto", "chunk", "transformed"), default="auto")
    g.add_argument("--occlusion", choices=("mesh", "none"), default="mesh")
    g.add_argument("--occlusion-tolerance", type=float)
    g.add_argument("--occlusion-rel-tolerance", type=float, default=0.0)
    g.add_argument("--min-depth", type=float, default=0.0)
    g = p.add_argument_group("ROI sampling and GSD")
    g.add_argument("--max-roi-points", type=int, default=300_000, help="seeded subsample of the ROI")
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--normal-k", type=int, default=16, help="k-NN for PCA normals when the ROI PLY has none")
    g.add_argument("--max-incidence-deg", type=float, default=85.0, help="cap of theta in gsd_eff")
    g.add_argument("--map-downsample", type=int, default=2, help="scripts/23 --downsample of the stored maps")
    g.add_argument("--mm-per-unit", type=float, help="metric scale, if established: adds *_mm columns")
    g = p.add_argument_group("footprint, eligibility, sampling")
    g.add_argument("--cell-px", type=int, default=8)
    g.add_argument("--min-component-frac", type=float, default=0.001, help="drop footprint parts/holes below this image fraction")
    g.add_argument("--min-image-coverage", type=float, default=0.05)
    g.add_argument("--min-visible-points", type=int, default=500)
    g.add_argument("--sample-per-stratum", type=int, default=0)
    g.add_argument("--labelme-cameras", choices=("auto", "sampled", "eligible", "all"), default="auto")
    g.add_argument("--replace-roi", action="store_true", help="in existing JSON replace roi/auto shapes, keep the rest")
    g.add_argument("--save-masks", action="store_true")
    g.add_argument("--verbose", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s %(message)s")
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
