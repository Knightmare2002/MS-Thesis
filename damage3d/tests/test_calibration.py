"""Platt calibration (scripts/27), calibrated/ensemble providers and threshold resolution."""
from __future__ import annotations

import json

import numpy as np
import pytest

from damage3d.classes import DAMAGE_CLASSES
from damage3d.providers import CalibratedProvider, EnsembleProvider, FileProvider, save_probability_npz
from src.eval.probability_calibration import (FORMAT_ENSEMBLE, FORMAT_SINGLE, apply_platt, bin_centers,
                                              fit_platt_from_histograms, logit, map_thresholds, sigmoid,
                                              thresholds_from_json)

C = len(DAMAGE_CLASSES)


def _params(a, b):
    return {c: {"a": float(a[i]), "b": float(b[i])} for i, c in enumerate(DAMAGE_CLASSES)}


def _single_json(path, a, b, thr):
    path.write_text(json.dumps({"format": FORMAT_SINGLE, "classes": list(DAMAGE_CLASSES), "run_name": "r",
                                "params": _params(a, b),
                                "thresholds_calibrated": dict(zip(DAMAGE_CLASSES, map(float, thr)))}))
    return path


def test_identity_and_nan():
    p = np.random.default_rng(0).uniform(0.01, 0.99, (C, 5, 7)).astype(np.float32)
    p[0, 0, 0] = np.nan
    out = apply_platt(p, np.ones(C), np.zeros(C))
    assert np.isnan(out[0, 0, 0])
    np.testing.assert_allclose(out[np.isfinite(p)], p[np.isfinite(p)], atol=1e-6)


def test_threshold_mapping_preserves_masks():
    rng = np.random.default_rng(1)
    p = rng.uniform(0, 1, (C, 64, 64)).astype(np.float32)
    a, b = rng.uniform(0.5, 1.5, C), rng.uniform(-2.5, 0.5, C)
    tau = np.array([0.9, 0.8, 0.8, 0.8, 0.8, 0.6])
    raw = p > tau[:, None, None]
    cal = apply_platt(p, a, b) > map_thresholds(tau, a, b)[:, None, None]
    assert (raw == cal).mean() > 0.9999        # monotone map: same decision (up to float32 ties)


def test_fit_recovers_known_parameters():
    a_true, b_true, bins = 0.7, -2.0, 1000
    z = logit(bin_centers(bins))
    n = np.full(bins, 1e5)
    q = sigmoid(a_true * z + b_true)
    a, b, info = fit_platt_from_histograms(n * q, n * (1 - q))
    assert info["converged"] and abs(a - a_true) < 1e-3 and abs(b - b_true) < 1e-3


def test_calibrated_provider_and_thresholds(tmp_path):
    maps = tmp_path / "maps"
    maps.mkdir()
    p = np.full((C, 8, 12), 0.3, np.float32)
    save_probability_npz(maps / "IMG.npz", p)
    a, b = np.full(C, 0.8), np.full(C, -1.0)
    js = _single_json(tmp_path / "cal.json", a, b, np.full(C, 0.25))
    prov = CalibratedProvider(FileProvider(maps), js)
    got = prov.get("IMG", 120, 80).probs
    np.testing.assert_allclose(got, apply_platt(p.astype(np.float16).astype(np.float32), a, b), atol=1e-6)
    np.testing.assert_allclose(thresholds_from_json(js, DAMAGE_CLASSES), 0.25)


def test_ensemble_weighted_mean_and_shape_check(tmp_path):
    p1, p2 = np.full((C, 8, 12), 0.2, np.float32), np.full((C, 8, 12), 0.7, np.float32)
    for name, p in (("m1", p1), ("m2", p2)):
        (tmp_path / name).mkdir()
        save_probability_npz(tmp_path / name / "IMG.npz", p)
    a1, b1, a2, b2 = np.full(C, 1.1), np.full(C, -1.5), np.full(C, 0.6), np.full(C, -2.0)
    ens = {"format": FORMAT_ENSEMBLE, "name": "e", "classes": list(DAMAGE_CLASSES),
           "thresholds": dict(zip(DAMAGE_CLASSES, [0.3] * C)),
           "members": [{"name": "m1", "run_dir": "x/m1", "weight": 3, "params": _params(a1, b1)},
                       {"name": "m2", "run_dir": "x/m2", "weight": 1, "params": _params(a2, b2)}]}
    js = tmp_path / "ensemble.json"
    js.write_text(json.dumps(ens))
    prov = EnsembleProvider(js, tmp_path, {"m1": tmp_path / "m1", "m2": tmp_path / "m2"})
    got = prov.get("IMG", 120, 80).probs
    f16 = lambda p: p.astype(np.float16).astype(np.float32)  # noqa: E731
    exp = 0.75 * apply_platt(f16(p1), a1, b1) + 0.25 * apply_platt(f16(p2), a2, b2)
    np.testing.assert_allclose(got, exp, atol=1e-6)
    np.testing.assert_allclose(thresholds_from_json(js, DAMAGE_CLASSES), 0.3)

    save_probability_npz(tmp_path / "m2" / "IMG.npz", np.full((C, 4, 6), 0.7, np.float32))
    with pytest.raises(ValueError, match="same --downsample"):
        prov.get("IMG", 120, 80)
    with pytest.raises(ValueError, match="unknown members"):
        EnsembleProvider(js, tmp_path, {"zz": tmp_path})


def test_damage3d_cli_calibration_and_ensemble(project_id, tmp_path):
    from damage3d.cli import main as damage3d_main
    from damage3d.tests.scene import HEIGHT, WIDTH

    cams = ["DSC01010", "DSC01020", "DSC01030"]
    p = np.full((C, HEIGHT // 4, WIDTH // 4), 0.05, np.float32)
    p[1] = 0.6                                           # spalling only
    for name in ("m1", "m2"):
        (tmp_path / name).mkdir()
        for s in cams:
            save_probability_npz(tmp_path / name / f"{s}.npz", p)
    a, b = np.ones(C), np.full(C, -1.0)                  # 0.6 -> 0.355, 0.05 -> 0.019
    cal = _single_json(tmp_path / "cal.json", a, b, np.full(C, 0.3))
    ens = tmp_path / "ensemble.json"
    ens.write_text(json.dumps({"format": FORMAT_ENSEMBLE, "name": "e", "classes": list(DAMAGE_CLASSES),
                               "thresholds": dict(zip(DAMAGE_CLASSES, [0.3] * C)),
                               "members": [{"name": n, "run_dir": f"x/{n}", "weight": 1, "params": _params(a, b)}
                                           for n in ("m1", "m2")]}))
    common = ["--project-root", str(project_id["root"]), "--start-image", "1", "--end-image", "4",
              "--max-points", "4000"]
    runs = {"cal": ["--probability-source", "files", "--probabilities-dir", str(tmp_path / "m1"),
                    "--calibration", str(cal)],
            "ens": ["--probability-source", "ensemble", "--ensemble", str(ens),
                    "--member-maps", f"m1={tmp_path / 'm1'}", "--member-maps", f"m2={tmp_path / 'm2'}"]}
    for key, extra in runs.items():
        out = tmp_path / f"out_{key}"
        assert damage3d_main([*common, "--output-dir", str(out), *extra]) == 0
        arrays = {}
        for shard in sorted((out / "outputs" / "fused").glob("*.npz")):
            with np.load(shard) as d:
                for k in d.files:
                    arrays.setdefault(k, []).append(d[k])
        d = {k: np.concatenate(v) for k, v in arrays.items()}
        seen = d["n_views"] > 0
        assert seen.any() and (d["label_mask"][seen] == 2).all(), key   # 0.355 > 0.3, 0.019 < 0.3
