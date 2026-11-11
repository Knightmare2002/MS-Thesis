"""roi_views on the synthetic project, with an analytic stand-in for the mesh occlusion (no Open3D)."""

from __future__ import annotations

import csv
import json

import numpy as np

from damage3d.roi_views import build_parser, footprint_mask, mask_to_polygons, run
from damage3d.tests.scene import CHUNK_SCALE, chunk_matrix


class PlateOcclusion:
    """Exact visibility for the synthetic scene: deck z=0 hidden by the plate |x|,|y|<=1, z in [0.95, 1.05]."""

    def __init__(self, to_source: np.ndarray):
        self.to_chunk = np.linalg.inv(to_source)
        self.diagonal = 20.0

    def _c(self, p):
        return np.asarray(p, float) @ self.to_chunk[:3, :3].T + self.to_chunk[:3, 3]

    def distance(self, pts):
        """Distance to the deck plane z=0 (source units, chunk scale 2); far away outside the deck."""
        c = self._c(pts)
        on_deck = (np.abs(c[:, 0]) <= 5) & (np.abs(c[:, 1]) <= 2)
        return np.where(on_deck, np.abs(c[:, 2]) * CHUNK_SCALE, 1e3)

    def surface_deviation_stats(self, pts):
        return {"n": len(pts), "median": 0.0, "p95": 0.0}

    def visible(self, pts, center, abs_tol, rel_tol=0.0):
        p, c = self._c(pts), self._c(np.asarray(center)[None, :])[0]
        out = np.ones(len(p), bool)
        below = p[:, 2] < 0.95                       # deck points can be hidden by the plate top face
        t = (c[2] - 1.05) / (c[2] - p[below, 2])
        hit = c[None, :2] + t[:, None] * (p[below, :2] - c[None, :2])
        out[below] = ~((np.abs(hit[:, 0]) <= 1) & (np.abs(hit[:, 1]) <= 1))
        return out


DECK_QUAD = np.array([[-1.6, -1.4, 0.0], [1.6, -1.4, 0.0], [1.6, 1.4, 0.0], [-1.6, 1.4, 0.0]])


def _polygons(tmp_path, to_source, frame="transformed"):
    """Metashape-like export: one quad on the deck, partly under the plate, closed by repeating V0."""
    v = DECK_QUAD if frame == "chunk" else DECK_QUAD @ to_source[:3, :3].T + to_source[:3, 3]
    v = np.vstack((v, v[:1]))
    text = "=== 7 ===\r\n" + "".join(f"V{i}: {x:.6f}, {y:.6f}, {z:.6f}\r\n" for i, (x, y, z) in enumerate(v))
    (tmp_path / "poly.txt").write_text(text + "\r\n", encoding="utf-8")


def _args(project, tmp_path, *extra):
    return build_parser().parse_args([
        "--project-root", str(project["root"]), "--polygons", str(tmp_path / "poly.txt"), "--polygon-id", "7",
        "--name", "deck", "--out", str(tmp_path / "out"), "--labelme-dir", str(tmp_path / "lm"),
        "--point-frame", "transformed", "--cell-px", "4", "--min-image-coverage", "0.02",
        "--min-visible-points", "50", "--map-downsample", "2", *extra])


def _roi(project, tmp_path):
    _polygons(tmp_path, project["to_source"])


def test_coverage_gsd_and_labelme(project_tf, tmp_path):
    _roi(project_tf, tmp_path)
    summary = run(_args(project_tf, tmp_path), occ=PlateOcclusion(project_tf["to_source"]))
    rows = {r["stem"]: r for r in csv.DictReader(open(tmp_path / "out" / "roi_views.csv"))}
    nadir = rows["DSC01020"]                                  # camera above x = 0, no tilt
    assert int(nadir["n_visible"]) < int(nadir["n_in_image"])  # the plate hides part of the ROI
    # GSD = Z * s / f = 6 * 2 / 500 source units per pixel, normals estimated by PCA (vertical).
    assert abs(float(nadir["gsd_frontal_median"]) - 6 * CHUNK_SCALE / 500) < 1e-3
    assert float(nadir["incidence_median_deg"]) < 15
    assert abs(float(nadir["gsd_eff_map_median"]) - 2 * float(nadir["gsd_eff_median"])) < 1e-12
    assert summary["roi"]["normals"] == "polygon_plane"
    assert summary["roi"]["polygon_frame"]["resolved"] == "transformed"
    in_quad = (np.abs(project_tf["xyz_chunk"][:, 0]) <= 1.6) & (np.abs(project_tf["xyz_chunk"][:, 1]) <= 1.4) \
        & (project_tf["xyz_chunk"][:, 2] == 0)
    assert summary["roi"]["n_points_in_polygon"] == int(in_quad.sum())   # plate top (z=1.05) excluded by the slab
    data = json.loads((tmp_path / "lm" / "DSC01020.json").read_text())
    labels = [s["label"] for s in data["shapes"]]
    assert "roi" in labels and "ignore" in labels             # plate footprint -> ignore hole
    assert data["imageWidth"] == 640 and data["imageHeight"] == 480 and data["imageData"] is None
    pts = np.array(data["shapes"][labels.index("roi")]["points"])
    assert pts[:, 0].min() >= 0 and pts[:, 0].max() <= 639 and pts[:, 1].max() <= 479


def test_existing_annotations_are_kept(project_tf, tmp_path):
    _roi(project_tf, tmp_path)
    occ = PlateOcclusion(project_tf["to_source"])
    run(_args(project_tf, tmp_path), occ=occ)
    path = tmp_path / "lm" / "DSC01020.json"
    data = json.loads(path.read_text())
    data["shapes"].append({"label": "crack", "points": [[1, 1], [5, 5]], "shape_type": "linestrip",
                           "group_id": None, "description": "", "flags": {}})
    path.write_text(json.dumps(data))
    s = run(_args(project_tf, tmp_path), occ=occ)                 # default: never overwrite
    assert s["labelme"]["counts"].get("skipped_existing", 0) >= 1
    assert "crack" in [x["label"] for x in json.loads(path.read_text())["shapes"]]
    run(_args(project_tf, tmp_path, "--replace-roi"), occ=occ)   # replace only roi/auto shapes
    shapes = json.loads(path.read_text())["shapes"]
    assert [x["label"] for x in shapes].count("crack") == 1 and "roi" in [x["label"] for x in shapes]


def test_polygon_frame_chunk_and_merge(project_tf, tmp_path):
    occ = PlateOcclusion(project_tf["to_source"])
    _polygons(tmp_path, project_tf["to_source"], frame="chunk")
    s = run(_args(project_tf, tmp_path), occ=occ)
    assert s["roi"]["polygon_frame"]["resolved"] == "chunk"
    # A second ROI written in the same JSON is merged, the first one is kept.
    s2 = run(_args(project_tf, tmp_path, "--name", "other"), occ=occ)
    assert s2["labelme"]["counts"].get("merged", 0) >= 1
    tags = {x["description"] for x in json.loads((tmp_path / "lm" / "DSC01020.json").read_text())["shapes"]}
    assert tags == {"auto:roi_views:deck", "auto:roi_views:other"}


def test_stratified_sample_is_reproducible(project_tf, tmp_path):
    _roi(project_tf, tmp_path)
    occ = PlateOcclusion(project_tf["to_source"])
    a = run(_args(project_tf, tmp_path, "--sample-per-stratum", "1"), occ=occ)
    first = (tmp_path / "out" / "test_views_sample.txt").read_text()
    run(_args(project_tf, tmp_path, "--sample-per-stratum", "1"), occ=occ)
    assert (tmp_path / "out" / "test_views_sample.txt").read_text() == first
    assert a["strata"]["n_eligible"] >= 3 and a["strata"]["n_sampled"] == 3


def test_footprint_polygon_with_hole():
    u, v = np.meshgrid(np.arange(0, 400, 2.0), np.arange(0, 300, 2.0))
    keep = ~((u > 150) & (u < 250) & (v > 100) & (v < 200))
    grid = footprint_mask(u[keep], v[keep], 400, 300, cell_px=4, spacing_px=2.0)
    rois, holes = mask_to_polygons(grid, 4, 400, 300, min_area_frac=0.001)
    assert len(rois) == 1 and len(holes) == 1
    hx, hy = holes[0][:, 0], holes[0][:, 1]
    assert 140 <= hx.min() <= 160 and 240 <= hx.max() <= 260 and 90 <= hy.min() <= 110
