"""Unit tests: XML parsing, selection, frames, projection, image bounds, radial guard."""

from __future__ import annotations

import numpy as np
import pytest

from damage3d.camera_model import CameraProjector, project_point_reference, radial_limit
from damage3d.frames import make_frame, read_marker_reference, resolve_frame
from damage3d.metashape_xml import (Calibration, SelectionError, parse_cameras_xml, select_cameras,
                                    selectable_cameras)
from damage3d.tests import scene


# ---------------------------------------------------------------- XML parsing
def test_parse_cameras_and_calibration(project_tf):
    proj = parse_cameras_xml(project_tf["root"] / "cameras.xml")
    assert len(proj.cameras) == 7
    labels = {c.label: c for c in proj.cameras}
    assert labels["DSC01040.JPG"].aligned            # camera inside <group>
    assert not labels["DSC09990.JPG"].aligned        # no transform
    assert not labels["DSC09995.JPG"].enabled        # enabled="false"
    calib = proj.calibrations["0"]
    assert (calib.width, calib.height) == (scene.WIDTH, scene.HEIGHT)
    for k, v in scene.CALIB.items():
        assert getattr(calib, k) == pytest.approx(v, abs=0)
    np.testing.assert_allclose(labels["DSC01020.JPG"].camera_to_chunk, scene.camera_to_chunk(0.0), atol=1e-15)


def test_chunk_transform_parsed(project_tf, project_id):
    tf = parse_cameras_xml(project_tf["root"] / "cameras.xml").chunk_transform
    m, _ = scene.chunk_matrix()
    assert tf.source == "rotation/translation/scale"
    assert tf.scale == pytest.approx(scene.CHUNK_SCALE)
    np.testing.assert_allclose(tf.matrix, m, atol=1e-12)
    ident = parse_cameras_xml(project_id["root"] / "cameras.xml").chunk_transform
    assert ident.source == "absent" and ident.is_identity


def test_unsupported_coefficients_rejected(tmp_path):
    scene.write_xml(tmp_path / "cameras.xml", False, extra_coefficients={"b1": 0.5})
    proj = parse_cameras_xml(tmp_path / "cameras.xml")
    calib = proj.calibrations["0"]
    assert calib.unsupported_nonzero == ("b1",)
    with pytest.raises(RuntimeError, match="b1"):
        CameraProjector(np.eye(4), calib)


# ---------------------------------------------------------------- selection
def _ordered(project_tf):
    proj = parse_cameras_xml(project_tf["root"] / "cameras.xml")
    return proj, selectable_cameras(proj)


def test_deterministic_order_and_exclusive_end(project_tf):
    proj, ordered = _ordered(project_tf)
    assert [c.stem for c in ordered] == ["DSC01000", "DSC01010", "DSC01020", "DSC01030", "DSC01040"]
    sel = select_cameras(ordered, proj.cameras, start_image=1, end_image=3)
    assert [p for p, _ in sel] == [1, 2]              # end exclusive
    sel = select_cameras(ordered, proj.cameras, start_image=1, limit_images=2)
    assert [c.stem for _, c in sel] == ["DSC01010", "DSC01020"]


def test_selection_by_name(project_tf):
    proj, ordered = _ordered(project_tf)
    sel = select_cameras(ordered, proj.cameras, camera=["DSC01030", "DSC01000.JPG"])
    assert [c.stem for _, c in sel] == ["DSC01000", "DSC01030"]  # global order, not CLI order


@pytest.mark.parametrize("kwargs, message", [
    ({"camera": ["DSC01000"], "camera_list": ["DSC01010"]}, "cannot be combined"),
    ({"camera": ["DSC01000"], "start_image": 0}, "cannot be combined"),
    ({"start_image": 5}, ">="),
    ({"end_image": 6}, "exclusive"),
    ({"start_image": 2, "end_image": 2}, "Empty range"),
    ({"camera": ["DSC77777"]}, "not in the XML"),
    ({"camera": ["DSC09990"]}, "not aligned"),
    ({"camera": ["DSC01000", "DSC01000"]}, "more than once"),
    ({"limit_images": 0}, "> 0"),
])
def test_selection_errors(project_tf, kwargs, message):
    proj, ordered = _ordered(project_tf)
    with pytest.raises(SelectionError, match=message):
        select_cameras(ordered, proj.cameras, **kwargs)


# ---------------------------------------------------------------- frames
def test_frame_roundtrip_and_auto_resolution(project_tf, project_id):
    proj = parse_cameras_xml(project_tf["root"] / "cameras.xml")
    frame = make_frame("transformed", proj.chunk_transform)
    pts = np.random.default_rng(0).normal(size=(100, 3))
    np.testing.assert_allclose(frame.to_chunk(frame.to_source(pts)), pts, atol=1e-12)
    np.testing.assert_allclose(frame.to_source(pts), pts @ project_tf["to_source"][:3, :3].T + project_tf["to_source"][:3, 3])

    from damage3d.mesh_visibility import MeshOcclusion
    occ = MeshOcclusion(project_tf["root"] / "model_medium_quality.obj")
    resolved, record = resolve_frame("auto", proj.chunk_transform, project_tf["root"] / "marker_reference.txt", occ.distance)
    assert resolved.name == "transformed" and record["method"] == "marker-to-mesh distance test"

    with pytest.raises(RuntimeError, match="marker reference"):
        resolve_frame("auto", proj.chunk_transform, project_tf["root"] / "missing.txt", occ.distance)

    proj_id = parse_cameras_xml(project_id["root"] / "cameras.xml")
    resolved, record = resolve_frame("auto", proj_id.chunk_transform, None, None)
    assert resolved.name == "chunk" and resolved.is_identity


# ---------------------------------------------------------------- projection
def _calib(**over) -> Calibration:
    base = dict(sensor_id="0", sensor_type="frame", width=scene.WIDTH, height=scene.HEIGHT, **scene.CALIB)
    base.update(over)
    return Calibration(**base)


def test_vectorised_projection_matches_verified_reference():
    """Regression: the pipeline projector equals the verified prototype formula."""
    cam = scene.camera_to_chunk(1.5, tilt_deg=5.0)
    proj = CameraProjector(cam, _calib())
    rng = np.random.default_rng(3)
    pts = np.column_stack((rng.uniform(-3, 3, 500), rng.uniform(-2, 2, 500), rng.uniform(-0.5, 1.5, 500)))
    out = proj.project(pts)
    for i in range(len(pts)):
        ref, cam_xyz = project_point_reference(pts[i], cam, scene.calib_dict())
        assert abs(out.u[i] - ref[0]) < 1e-9 and abs(out.v[i] - ref[1]) < 1e-9
        assert abs(out.depth[i] - cam_xyz[2]) < 1e-12


def test_synthetic_marker_regression(project_tf):
    """Same procedure as verify_marker_projection.py on the synthetic marker."""
    ref = read_marker_reference(project_tf["root"] / "marker_reference.txt")
    proj = parse_cameras_xml(project_tf["root"] / "cameras.xml")
    cam = next(c for c in proj.cameras if c.stem == "DSC01020")
    projector = CameraProjector(cam.camera_to_chunk, proj.calibrations[cam.sensor_id])
    pt = np.array([[float(ref["chunk_x"]), float(ref["chunk_y"]), float(ref["chunk_z"])]])
    out = projector.project(pt)
    err = np.hypot(out.u[0] - float(ref["pixel_u"]), out.v[0] - float(ref["pixel_v"]))
    assert err < 1e-6 and out.in_image[0]


def test_behind_camera_rejected():
    proj = CameraProjector(scene.camera_to_chunk(0.0), _calib())
    out = proj.project(np.array([[0.0, 0.2, 7.0], [0.0, 0.2, 6.0], [0.0, 0.2, 0.0]]))  # above, at, below the camera
    assert list(out.in_front) == [False, False, True]
    assert list(out.in_image) == [False, False, True]
    with pytest.raises(ValueError):
        project_point_reference([0.0, 0.2, 7.0], scene.camera_to_chunk(0.0), scene.calib_dict())


def test_image_bounds_corner_convention():
    """u in [0, W) and v in [0, H): points mapping to u = W or v = -eps are rejected."""
    c = _calib(k1=0.0, k2=0.0, k3=0.0, p1=0.0, p2=0.0, cx=0.0, cy=0.0)
    proj = CameraProjector(np.eye(4), c)
    f, w, h = c.f, c.width, c.height

    def point(u, v, z=10.0):
        return [(u - w / 2) / f * z, (v - h / 2) / f * z, z]

    pts = np.array([point(0.0, 0.0), point(w - 1e-6, h - 1e-6), point(w, 10), point(10, h),
                    point(-1e-6, 10), point(w / 2, -1e-6)])
    out = proj.project(pts)
    assert list(out.in_image) == [True, True, False, False, False, False]
    margin = CameraProjector(np.eye(4), c, border_margin_px=2.0).project(pts[:2])
    assert not margin.in_image.any()


def test_radial_guard_rejects_fold_over():
    """A strongly negative k1 folds far-away points back into the image; the guard removes them."""
    c = _calib(k1=-0.3, k2=0.0, k3=0.0, p1=0.0, p2=0.0, cx=0.0, cy=0.0)
    r_lim = radial_limit(c)
    x_far = 1.6  # r = 1.6 -> r_d = 1.6 * (1 - 0.3 * 2.56) = 0.371 -> inside the image, but non-monotonic
    assert 1 + 3 * c.k1 * x_far**2 < 0
    proj = CameraProjector(np.eye(4), c)
    out = proj.project(np.array([[x_far * 10, 0.0, 10.0], [0.1, 0.0, 10.0]]))
    u_fold = c.width / 2 + c.f * x_far * (1 + c.k1 * x_far**2)
    assert 0 <= u_fold < c.width                  # without the guard it would be accepted
    assert list(out.in_image) == [False, True]
    assert r_lim < x_far


def test_sphere_culling_is_conservative():
    proj = CameraProjector(scene.camera_to_chunk(0.0, 5.0), _calib())
    rng = np.random.default_rng(7)
    for _ in range(300):
        center = rng.uniform([-10, -10, -3], [10, 10, 9])
        radius = rng.uniform(0.05, 3.0)
        d = rng.normal(size=(400, 3))
        d /= np.linalg.norm(d, axis=1, keepdims=True)
        pts = center + d * radius * rng.uniform(0, 1, (400, 1)) ** (1 / 3)
        if proj.project(pts).in_image.any():
            assert proj.sphere_may_be_visible(center, radius)
