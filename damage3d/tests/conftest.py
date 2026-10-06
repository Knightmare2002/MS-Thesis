from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from damage3d.tests.scene import build_project  # noqa: E402


@pytest.fixture(scope="session")
def project_tf(tmp_path_factory):
    """Synthetic project whose PLY/OBJ are in the chunk-TRANSFORMED frame."""
    return build_project(tmp_path_factory.mktemp("proj_tf"), chunk_transform=True)


@pytest.fixture(scope="session")
def project_id(tmp_path_factory):
    """Synthetic project without chunk transform (PLY/OBJ in the chunk frame)."""
    return build_project(tmp_path_factory.mktemp("proj_id"), chunk_transform=False)
