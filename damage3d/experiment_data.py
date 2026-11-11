"""Strict ROI contract and provenance for Direction A experiments (no inference dependencies)."""
from __future__ import annotations

import json
import hashlib
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree

from .classes import DAMAGE_CLASSES, NUM_CLASSES
from .run_state import sha256_file, write_json_atomic


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, data):
    """Refuse accidental replacement of frozen experiment results."""
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"{path}: choose a new experiment output.")
    path.parent.mkdir(parents=True, exist_ok=True)
    # Do not export non-standard JSON NaN as an apparent measurement.
    def clean(x):
        if isinstance(x, dict):
            return {str(k): clean(v) for k, v in x.items()}
        if isinstance(x, (list, tuple, np.ndarray)):
            return [clean(v) for v in x]
        if isinstance(x, np.generic):
            x = x.item()
        return None if isinstance(x, float) and not np.isfinite(x) else x
    write_json_atomic(path, clean(data))


def regions(path, buffer):
    """NPZ: source_index, region, split, xyz, area, labels, classes.

    source_index is the original PLY row, never a sampled/reordered row number.
    splits are C_fit / C_cal / T. labels is N,C with -1 unknown, 0 absent, 1 present.
    Buffer applies between ALL pairs of different splits in source units.
    """
    with np.load(path, allow_pickle=False) as f:
        d = {k: f[k] for k in f.files}
    required = {"source_index", "region", "split", "xyz", "area", "labels", "classes"}
    if not required <= d.keys():
        raise ValueError(f"Region NPZ missing {sorted(required - d.keys())}")
    n = len(d["source_index"])
    if (n == 0 or d["source_index"].ndim != 1 or d["source_index"].dtype.kind not in "ui"
            or np.any(d["source_index"] < 0) or len(np.unique(d["source_index"])) != n):
        raise ValueError("source_index must be a nonempty unique nonnegative integer vector.")
    if d["classes"].tolist() != list(DAMAGE_CLASSES):
        raise ValueError("Wrong class order.")
    for k in ("region", "split", "area"):
        if d[k].shape != (n,):
            raise ValueError(f"Wrong shape for {k}.")
    if d["xyz"].shape != (n, 3) or not np.isfinite(d["xyz"]).all():
        raise ValueError("xyz must be finite N,3 source-frame coordinates.")
    if d["labels"].shape != (n, NUM_CLASSES) or not np.isin(d["labels"], [-1, 0, 1]).all():
        raise ValueError("labels must be N,C in {-1,0,1}.")
    if not np.isfinite(d["area"]).all() or np.any(d["area"] <= 0):
        raise ValueError("area must be finite, positive surface support per point.")
    if not np.isin(d["split"], ["C_fit", "C_cal", "T"]).all():
        raise ValueError("split must be C_fit, C_cal or T.")
    for r in np.unique(d["region"]):
        if len(np.unique(d["split"][d["region"] == r])) != 1:
            raise ValueError(f"Region {r} crosses splits.")
    if not np.isfinite(buffer) or buffer <= 0:
        raise ValueError("Declare a positive spatial buffer in source units.")
    for a, b in (("C_fit", "C_cal"), ("C_fit", "T"), ("C_cal", "T")):
        x, y = d["xyz"][d["split"] == a], d["xyz"][d["split"] == b]
        if len(x) and len(y) and cKDTree(x).query(y)[0].min() < buffer:
            raise ValueError(f"Spatial buffer violation: {a} / {b}.")
    order = np.argsort(d["source_index"])
    return {k: (v[order] if k != "classes" else v) for k, v in d.items() if k in required}


def provenance(run, region_path, config_path=None):
    run = Path(run)
    manifest = run / "manifest.json"
    meta = read_json(run / "observations/meta.json")
    if meta.get("schema") != 3:
        raise ValueError("Observation schema 3 required; regenerate the log.")
    if not (run / "outputs/summary.json").is_file():
        raise ValueError("Run has no completed export; resume projection before fitting.")
    # Culling legitimately leaves absent chunks/old per-chunk last_batch values.
    # Completion is the global commit frontier plus the sealed observation inventory.
    m = read_json(manifest)
    fp = m["fingerprint"]
    nc = len(fp["cameras"]["selected"])
    last = (nc - 1) // fp["layout"]["camera_batch_size"]
    progress = read_json(run / "progress.json")
    if progress["completed_batches"] != last+1:
        raise ValueError("Incomplete camera batches: resume projection.")
    chunks = sorted((run / "acc").glob("chunk_*.npz"))
    for p in chunks:
        with np.load(p, allow_pickle=False) as a:
            if not -1 <= int(a["last_batch"]) <= last:
                raise ValueError(f"Invalid accumulator commit {p}.")
    from .observations import ObservationLog
    inventory = ObservationLog(run).seal(verify_only=True)
    digest = hashlib.sha256(json.dumps(inventory, sort_keys=True).encode())
    return {"run_manifest_sha256": sha256_file(manifest), "regions_sha256": sha256_file(region_path),
            "config_sha256": sha256_file(config_path) if config_path else None,
            "observation_meta_sha256": sha256_file(run / "observations/meta.json"),
            "observation_records_sha256": digest.hexdigest()}


def validate_fit_provenance(params, prov):
    fit = params.get("fit_provenance")
    if fit:
        for key in ("run_manifest_sha256", "regions_sha256", "observation_meta_sha256", "observation_records_sha256"):
            if fit[key] != prov[key]:
                raise ValueError(f"Fitted params use different {key}. Explicit transfer experiment required.")


def validate_detector(run, params):
    meta = read_json(Path(run) / "observations/meta.json")
    provider = meta["provider"]
    if provider["type"] == "synthetic":
        return
    if provider["type"] != "files+platt":
        raise ValueError("Likelihood experiments require single-model Platt-calibrated observations.")
    if params.get("calibration_sha256") != provider["calibration"]["sha256"]:
        raise ValueError("Likelihood parameters must reference the observation provider's calibration hash.")
    rho = meta.get("calibration_prevalence")
    if not rho or not np.allclose([params["classes"][c]["rho"] for c in DAMAGE_CLASSES],
                                  [rho[c] for c in DAMAGE_CLASSES], rtol=1e-8, atol=1e-10):
        raise ValueError("rho must match the stored calibration prevalence; regenerate legacy logs.")


def match_rows(source_ids, d):
    pos = np.searchsorted(d["source_index"], source_ids)
    if np.any(pos >= len(d["source_index"])) or not np.array_equal(d["source_index"][pos], source_ids):
        raise ValueError("Observations are not in the declared ROI.")
    return pos
