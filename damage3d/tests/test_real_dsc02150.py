"""Regression on the REAL verified case (DSC02150 marker), run on the thesis workstation.

Skipped unless $DAMAGE3D_PROJECT_ROOT points to bridge_model with cameras.xml
and marker_reference.txt (the inputs of verify_marker_projection.py).
The verified script reported 0.000000 px; the test accepts < 1e-3 px because
marker_reference.txt stores decimal values of limited precision.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from damage3d.camera_model import CameraProjector, project_point_reference
from damage3d.frames import read_marker_reference
from damage3d.metashape_xml import parse_cameras_xml

ROOT = Path(os.environ.get("DAMAGE3D_PROJECT_ROOT", "__missing__"))
XML = ROOT / "cameras.xml"
REF = ROOT / "marker_reference.txt"
pytestmark = pytest.mark.skipif(not (XML.is_file() and REF.is_file()),
                                reason="real bridge_model data not available ($DAMAGE3D_PROJECT_ROOT)")


def test_dsc02150_marker_projection_matches_metashape():
    ref = read_marker_reference(REF)
    proj = parse_cameras_xml(XML)
    matches = [c for c in proj.cameras if c.stem == "DSC02150"]
    assert len(matches) == 1 and matches[0].aligned
    cam = matches[0]
    calib = proj.calibration_for(cam)
    point = np.array([float(ref["chunk_x"]), float(ref["chunk_y"]), float(ref["chunk_z"])])
    expected = np.array([float(ref["pixel_u"]), float(ref["pixel_v"])])

    reference_px, _ = project_point_reference(point, cam.camera_to_chunk, calib.as_dict())
    out = CameraProjector(cam.camera_to_chunk, calib).project(point[None, :])
    vectorised_px = np.array([out.u[0], out.v[0]])

    assert np.linalg.norm(reference_px - expected) < 1e-3
    assert np.linalg.norm(vectorised_px - reference_px) < 1e-9
    assert out.in_image[0]
