"""Project-root resolution and default input names.

Default file names are the ones used by the existing, verified scripts in
``bridge_model/scripts`` (check_obj_ply_alignment.py, verify_marker_projection.py,
project_test_mask.py). Relative paths are resolved against the project root,
never against the current working directory.
"""

from __future__ import annotations

import os
from pathlib import Path

ENV_PROJECT_ROOT = "DAMAGE3D_PROJECT_ROOT"
DEFAULT_PROJECT_ROOT = Path(r"C:\\Users\Samuele_Caruso\\OneDrive - UMass Lowell\\Desktop\\bridge_model")

DEFAULT_NAMES = {
    "cameras_xml": "cameras.xml",
    "point_cloud": "bridge_3dpc_point_cloud.ply",
    "mesh": "model_medium_quality.obj",
    "marker_reference": "marker_reference.txt",
    "images_dir": "C:\\Users\\Samuele_Caruso\\OneDrive - UMass Lowell\\Desktop\\150_Lincoln_St_22_09_2026",
    "runs_dir": "damage3d_runs",
}


def resolve_project_root(cli_value: str | None) -> Path:
    """CLI value > $DAMAGE3D_PROJECT_ROOT > DEFAULT_PROJECT_ROOT."""
    if cli_value:
        root = Path(cli_value)
    elif os.environ.get(ENV_PROJECT_ROOT):
        root = Path(os.environ[ENV_PROJECT_ROOT])
    else:
        root = DEFAULT_PROJECT_ROOT
    return root.expanduser().resolve()


def resolve_path(project_root: Path, value: str | os.PathLike | None, default_key: str | None) -> Path | None:
    """Resolve a CLI path; relative values (and defaults) are relative to the project root."""
    if value is None:
        if default_key is None:
            return None
        value = DEFAULT_NAMES[default_key]
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = project_root / path
    return path.resolve()
