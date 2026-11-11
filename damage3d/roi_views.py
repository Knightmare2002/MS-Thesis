"""ROI coverage, ground sampling distance and Labelme ROI pre-fill, per camera.

The ROI is a planar polygon drawn in Metashape (vertex coordinates exported as
text). Its 3D points are the dense-cloud points inside the polygon prism; every
camera is then tested with the same geometry as the fusion pipeline
(``CameraProjector`` + ``MeshOcclusion``, same frame resolution, same automatic
occlusion tolerance). Nothing in the fusion inputs is changed: the ROI is an
evaluation-only object.

    python -m damage3d.roi_views --project-root <bridge_model> \
        --polygons poligoni_coordinate.txt --polygon-id 3 --name face \
        --camera-list bridge_face.txt --out roi_views/face --labelme-dir labelme_json

Inputs
------
* ``--polygons``: blocks ``=== <id> ===`` followed by ``V<k>: x, y, z`` lines
  (a closing vertex equal to V0 is dropped). The polygon must be planar: the
  max vertex-to-plane distance must stay below half the slab.
* ``--polygon-frame auto``: the vertices are tried as chunk coordinates and as
  chunk-transformed (Metashape local) coordinates; the hypothesis whose vertices
  lie on the mesh (10x closer) is kept, else the run stops.
* ROI points: dense-cloud points within +-slab of the polygon plane whose
  projection lies inside the polygon (``--slab``, default 1% of the polygon
  bounding-box diagonal), seeded subsample of ``--max-roi-points``.
* Normal for the incidence angle: the polygon plane (default) or the
  dense-cloud normals (``--normals cloud``).
* ``--camera-list``: the views already saved in Metashape for this ROI; the
  summary reports listed cameras that do not see the polygon (wrong id or list).

Per camera (``roi_views.csv``)
------------------------------
* ``n_visible``: ROI points inside the image and not occluded by the mesh;
  ``frac_roi_visible`` = n_visible / ROI points used.
* ``image_coverage``: fraction of the image covered by the ROI footprint
  (visible points rasterised on a grid of ``--cell-px`` pixels, morphological
  closing sized on the projected mean point spacing sqrt(area / points); holes are NOT filled because
  they are mostly occluders in front of the ROI).
* GSD, in source-frame units per pixel (metres only if the export is metric):
  ``gsd_frontal = Z * s / f`` with Z the camera depth (chunk frame), s the
  chunk-to-source scale, f the focal length in pixels; along the surface
  ``gsd_eff = gsd_frontal / sqrt(max(|cos theta|, cos(--max-incidence-deg)))``, theta
  the angle between the surface normal and the viewing ray (geometric mean of the
  anisotropic footprint (Z/f, Z/(f cos theta)), as g_iv in damage3d.detectability). Medians and p10/p90
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
``auto:roi_views:<name>``. Shapes of other ROIs and manual annotations are never
touched: a ROI is added to an existing JSON that does not contain it yet; if it
is already there the file is skipped unless ``--replace-roi`` (which replaces
only the shapes of that ROI).
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import re
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
# ROI from Metashape polygon vertices
# --------------------------------------------------------------------------- #
def read_polygons(path: Path) -> dict[str, np.ndarray]:
    """Parse blocks ``=== <id> ===`` followed by ``V<k>: x, y, z`` lines; a repeated closing vertex is dropped."""
    polys: dict[str, list] = {}
    current = None
    for line in Path(path).read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line:
            continue
        m = re.fullmatch(r"=+\s*(.+?)\s*=+", line)
        if m:
            current = m.group(1)
            polys[current] = []
            continue
        m = re.fullmatch(r"V\d+\s*:\s*(.+)", line)
        if m is None or current is None:
            raise RuntimeError(f"{path}: unexpected line '{line}'")
        polys[current].append([float(x) for x in m.group(1).replace(";", ",").split(",")])
    out = {}
    for k, v in polys.items():
        v = np.asarray(v, np.float64)
        if len(v) > 3 and np.allclose(v[0], v[-1]):
            v = v[:-1]
        if v.ndim != 2 or v.shape[1] != 3 or len(v) < 3:
            raise RuntimeError(f"{path}: polygon {k} needs >= 3 vertices with x, y, z")
        out[k] = v
    return out


@dataclass(frozen=True)
class PlanarPolygon:
    vertices: np.ndarray      # (n, 3) source frame
    origin: np.ndarray
    axes: np.ndarray          # (2, 3) in-plane orthonormal basis
    normal: np.ndarray
    uv: np.ndarray            # (n, 2) vertices in the plane basis
    residual_max: float       # max |vertex-to-plane| distance
    area: float

    @staticmethod
    def fit(vertices: np.ndarray) -> "PlanarPolygon":
        c = vertices.mean(axis=0)
        _, _, vt = np.linalg.svd(vertices - c)
        n = vt[2] / np.linalg.norm(vt[2])
        uv = (vertices - c) @ vt[:2].T
        x, y = uv[:, 0], uv[:, 1]
        area = 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))
        return PlanarPolygon(vertices, c, vt[:2], n, uv, float(np.abs((vertices - c) @ n).max()), area)

    def contains(self, xyz: np.ndarray, slab: float) -> np.ndarray:
        """Points within +-slab of the plane whose projection lies inside the polygon (even-odd rule)."""
        d = xyz - self.origin
        keep = np.abs(d @ self.normal) <= slab
        q = d[keep] @ self.axes.T
        inside = np.zeros(len(q), bool)
        a, b = self.uv, np.roll(self.uv, -1, axis=0)
        for (x1, y1), (x2, y2) in zip(a, b):
            crosses = (y1 > q[:, 1]) != (y2 > q[:, 1])
            with np.errstate(divide="ignore", invalid="ignore"):
                xc = x1 + (q[:, 1] - y1) * (x2 - x1) / (y2 - y1)
            inside ^= crosses & (q[:, 0] < xc)
        keep[keep] = inside
        return keep


def polygon_to_source(vertices: np.ndarray, polygon_frame: str, frame, chunk_transform, distance_fn,
                      ratio_required: float = 10.0) -> tuple[np.ndarray, dict]:
    """Bring the Metashape vertices into the dense-cloud (source) frame.

    ``chunk``: internal chunk coordinates; ``transformed``: chunk transform applied (Metashape local/world
    coordinates of the shapes). ``auto`` keeps the hypothesis whose vertices lie on the mesh (10x rule)."""
    m = chunk_transform.matrix
    cands = {"chunk": vertices,
             "transformed": (vertices - m[:3, 3]) @ np.linalg.inv(m[:3, :3]).T}   # -> chunk frame
    cands = {k: frame.to_source(v) for k, v in cands.items()}
    if polygon_frame != "auto":
        return cands[polygon_frame], {"requested": polygon_frame, "resolved": polygon_frame, "method": "CLI"}
    if np.allclose(cands["chunk"], cands["transformed"], atol=1e-9):
        return cands["chunk"], {"requested": "auto", "resolved": "chunk=transformed", "method": "identity transform"}
    if distance_fn is None:
        raise RuntimeError("Polygon frame check needs the mesh: pass --polygon-frame chunk|transformed.")
    d = {k: float(np.median(distance_fn(v))) for k, v in cands.items()}
    rec = {"requested": "auto", "median_vertex_to_mesh": d, "ratio_required": ratio_required}
    for a, b in (("chunk", "transformed"), ("transformed", "chunk")):
        if d[a] * ratio_required <= d[b]:
            rec.update(resolved=a, method="vertex-to-mesh distance test")
            return cands[a], rec
    raise RuntimeError(f"Polygon frame check inconclusive {d}: pass --polygon-frame chunk|transformed.")


def select_roi_points(ply: PlyPointSource, poly: PlanarPolygon, slab: float, block: int = 2_000_000):
    """Stream the dense cloud and return the indices of the points inside the polygon prism."""
    lo = poly.vertices.min(0) - slab
    hi = poly.vertices.max(0) + slab
    out = []
    for s in range(0, ply.n_points, block):
        e = min(s + block, ply.n_points)
        xyz = ply.xyz(slice(s, e)).astype(np.float64)
        box = np.nonzero(np.all((xyz >= lo) & (xyz <= hi), axis=1))[0]
        if box.size:
            out.append(s + box[poly.contains(xyz[box], slab)])
    return np.concatenate(out) if out else np.zeros(0, np.int64)


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
    gsd_eff = gsd_frontal / np.sqrt(np.maximum(cos, cos_floor))
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
def _shape(label: str, poly: np.ndarray, tag: str) -> dict:
    return {"label": label, "points": [[round(float(x), 1), round(float(y), 1)] for x, y in poly],
            "group_id": None, "description": tag, "shape_type": "polygon", "flags": {}, "mask": None}


def _image_path(stem: str, label: str, images_dir: Path | None, json_dir: Path) -> str:
    if images_dir is not None and images_dir.is_dir():
        for ext in IMAGE_EXTS:
            for cand in (images_dir / f"{stem}{ext}", images_dir / f"{stem}{ext.upper()}"):
                if cand.is_file():
                    return os.path.relpath(cand, json_dir)
    return label


def write_labelme(json_dir: Path, res: ViewResult, images_dir: Path | None, replace_roi: bool, tag: str) -> str:
    """Return 'written', 'merged', 'updated' or 'skipped_existing'.

    Only shapes whose description equals ``tag`` belong to this ROI: other ROIs and every manual
    annotation are always kept."""
    json_dir.mkdir(parents=True, exist_ok=True)
    path = json_dir / f"{res.row['stem']}.json"
    auto = [_shape("roi", p, tag) for p in res.rois] + [_shape("ignore", p, tag) for p in res.holes]
    if path.is_file():
        data = json.loads(path.read_text(encoding="utf-8"))
        shapes = data.get("shapes", [])
        own = [s_ for s_ in shapes if s_.get("description") == tag]
        if own and not replace_roi:
            return "skipped_existing"
        data["shapes"] = auto + [s_ for s_ in shapes if s_.get("description") != tag]
        status = "updated" if own else "merged"
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
    polygons_path = resolve_path(root, args.polygons, None)
    cloud_path = resolve_path(root, args.point_cloud, "point_cloud")
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

    polys = read_polygons(polygons_path)
    if args.polygon_id not in polys:
        raise RuntimeError(f"Polygon '{args.polygon_id}' not in {polygons_path} (found: {', '.join(polys)}).")
    verts_src, poly_frame = polygon_to_source(polys[args.polygon_id], args.polygon_frame, frame,
                                              project.chunk_transform, occ.distance if occ is not None else None)
    poly = PlanarPolygon.fit(verts_src)
    extent = float(np.linalg.norm(verts_src.max(0) - verts_src.min(0)))
    slab = args.slab if args.slab is not None else args.slab_frac * extent
    if poly.residual_max > 0.5 * slab:
        raise RuntimeError(f"Polygon {args.polygon_id} is not planar enough: max vertex-to-plane distance "
                           f"{poly.residual_max:.4g} vs slab {slab:.4g}. Split it into planar parts or raise --slab.")

    ply = PlyPointSource(cloud_path)
    idx = select_roi_points(ply, poly, slab)
    n_total = int(idx.size)
    np.save(out_dir / "roi_points_index.npy", idx.astype(np.int64))  # all ROI points: --observation-subset
    if n_total == 0:
        raise RuntimeError(f"No dense-cloud point inside polygon {args.polygon_id} (slab {slab:.4g}): check the frame.")
    if args.max_roi_points is not None and n_total > args.max_roi_points:
        idx = np.sort(np.random.default_rng(args.seed).choice(idx, size=args.max_roi_points, replace=False))
    xyz_src = ply.xyz(idx)
    plane_dist = (xyz_src - poly.origin) @ poly.normal
    if args.normals == "cloud" and ply.has_normals:
        normals, normals_source = ply.normals(idx).astype(np.float64), "dense_cloud"
        normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
    else:
        normals, normals_source = np.repeat(poly.normal[None, :], len(idx), axis=0), "polygon_plane"
    ply.close()
    spacing = float(np.sqrt(poly.area / len(xyz_src)))  # mean spacing of the points actually projected
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
                f"ROI points are far from the mesh (median {dev['median']:.4g} source units): dense cloud and "
                "mesh are not in the same frame.")

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

    no_view = [r["stem"] for r in rows if r["n_in_image"] == 0]
    camera_check = {"n_listed": len(rows), "n_without_roi_in_image": len(no_view), "stems": no_view[:50]}
    if listed is not None and len(no_view) > 0.5 * len(rows):
        log.warning("%d/%d listed cameras do not see polygon %s: wrong polygon id or camera list?",
                    len(no_view), len(rows), args.polygon_id)
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
                    labelme[r["stem"]] = write_labelme(labelme_dir, res, images_dir, args.replace_roi, f"{AUTO_TAG}:{args.name}")

    cov = np.array([r["image_coverage"] for r in rows if r["n_visible"] > 0])
    summary = {
        "roi": {"name": args.name, "polygons": str(polygons_path), "polygon_id": args.polygon_id,
                "polygon_frame": poly_frame, "vertices_source": verts_src.tolist(),
                "plane_normal": poly.normal.tolist(), "area_source_units2": poly.area,
                "vertex_plane_residual_max": poly.residual_max, "slab": slab, "point_cloud": str(cloud_path),
                "n_points_in_polygon": n_total, "n_points_used": len(xyz_src),
                "point_to_plane_abs_p50_p95": [float(np.percentile(np.abs(plane_dist), q)) for q in (50, 95)],
                "normals": normals_source, "mean_spacing_used_source_units": spacing},
        "camera_list_check": camera_check,
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
    g.add_argument("--polygons", required=True, help="Metashape polygon vertices (=== id === / V0: x, y, z)")
    g.add_argument("--polygon-id", required=True, help="block id in --polygons, e.g. 2")
    g.add_argument("--name", required=True, help="ROI name, e.g. face or roof (Labelme tag, logs)")
    g.add_argument("--point-cloud", help="dense cloud PLY (default bridge_3dpc_point_cloud.ply)")
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
    g.add_argument("--polygon-frame", choices=("auto", "chunk", "transformed"), default="auto",
                   help="frame of the polygon vertices (auto: vertex-to-mesh test)")
    g.add_argument("--slab", type=float, help="half thickness of the prism around the polygon plane (source units)")
    g.add_argument("--slab-frac", type=float, default=0.01, help="default slab = this x polygon bbox diagonal")
    g.add_argument("--max-roi-points", type=int, default=300_000, help="seeded subsample of the ROI points")
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--normals", choices=("plane", "cloud"), default="plane",
                   help="normal for the incidence angle: polygon plane or dense-cloud nx,ny,nz")
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
