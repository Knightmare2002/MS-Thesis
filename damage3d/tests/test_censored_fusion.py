"""Censored llr fusion and observation log (unit + end to end on the synthetic project)."""

from __future__ import annotations

import json

import numpy as np
import pytest

from damage3d.classes import DAMAGE_CLASSES, NUM_CLASSES
from damage3d.cli import main
from damage3d.detectability import FusionModel
from damage3d.fusion import ChunkAccumulator, fuse
from damage3d.observations import OBS_DTYPE, load_observations, probs_of
from damage3d.pipeline import SimulatedInterruption
from damage3d.tests.scene import CHUNK_SCALE, CALIB, WIDTH, build_project

BETA = 1.0 / 3.0   # logistic scale in log g
G50 = 0.1


def _params(mode_censored=True, alpha=1.0, **over):
    cls = {c: {"rho": 0.1, "alpha": alpha, **({"g50": G50, "beta": BETA} if mode_censored else {})}
           for c in DAMAGE_CLASSES}
    return {"gsd": "effective", "classes": cls, **over}


def _model(mode="censored_llr", **over):
    return FusionModel.from_dict(mode, _params(mode == "censored_llr", **over))


# --------------------------------------------------------------------------- #
# Unit tests
# --------------------------------------------------------------------------- #
def test_censored_evidence_limits_and_formula():
    m = _model()
    p = np.tile(np.array([[0.02], [0.3], [0.7], [0.97]]), (1, NUM_CLASSES))
    lam = _model("llr").view_llr(p, None)
    near = m.view_llr(p, np.full(4, 1e-6))               # pi -> 1
    far = m.view_llr(p, np.full(4, 1e6))                 # pi -> 0
    np.testing.assert_allclose(near, lam, atol=1e-9)
    np.testing.assert_allclose(far, 0.0, atol=1e-9)
    g = np.full(4, G50)                                  # pi = 0.5
    np.testing.assert_allclose(m.view_llr(p, g), np.log(1 + 0.5 * (np.exp(lam) - 1)), rtol=1e-12)
    assert np.all(np.diff(m.view_llr(p, g)[:, 0]) > 0)   # monotone in p
    extreme = m.view_llr(np.array([[0.0] * NUM_CLASSES, [1.0] * NUM_CLASSES]), np.array([1e-3, 1e-3]))
    assert np.isfinite(extreme).all()                    # clipped
    assert np.isnan(m.view_llr(np.full((1, NUM_CLASSES), np.nan), np.array([0.05]))).all()
    # Report parametrization: pi(g) = sigmoid(-(log g - log g50) / beta).
    g2 = np.array([G50 * np.e])
    np.testing.assert_allclose(m.pod(g2)[0], 1.0 / (1.0 + np.exp(1.0 / BETA)), rtol=1e-12)


def test_gsd_uses_sqrt_cos_and_map_scale():
    m = _model()
    g = m.view_gsd(np.array([0.002, 0.002]), np.array([1.0, 0.25]), 2.0)
    np.testing.assert_allclose(g, [0.004, 0.008])                     # sigma_v = 2, 1/sqrt(0.25) = 2
    np.testing.assert_allclose(m.view_gsd(np.array([0.002]), np.array([0.0]), 1.0),
                               0.002 / np.sqrt(m.min_cos))             # grazing floor
    fr = _model(gsd="frontal")
    np.testing.assert_allclose(fr.view_gsd(np.array([0.002]), np.array([0.25]), 2.0), 0.004)


def test_tempering_alpha_scales_the_evidence():
    m, t = _model("llr"), _model("llr", alpha=0.5)
    s = np.array([[2.0] * NUM_CLASSES])
    one = np.ones((1, NUM_CLASSES))
    lo = np.log(0.1 / 0.9)
    np.testing.assert_allclose(m.posterior(s, one), 1 / (1 + np.exp(-(lo + 2.0))), rtol=1e-6)
    np.testing.assert_allclose(t.posterior(s, one), 1 / (1 + np.exp(-(lo + 1.0))), rtol=1e-6)
    with pytest.raises(ValueError, match="alpha"):
        _model("llr", alpha=1.5)


def test_far_negative_views_do_not_dilute_a_near_positive():
    """1 near view p=0.9 (g = 0.01) + 10 far views p=0.05 (g = 1): only the censored rule keeps the damage."""
    probs = np.vstack([np.full((1, NUM_CLASSES), 0.9), np.full((10, NUM_CLASSES), 0.05)])
    g = np.array([0.01] + [1.0] * 10)
    acc = ChunkAccumulator.zeros(1, 11, llr=True)
    acc_llr = ChunkAccumulator.zeros(1, 11, llr=True)
    cens, plain = _model(), _model("llr")
    for v in range(11):
        acc.add_view(np.array([0]), probs[v:v + 1], 0.5, cens.view_llr(probs[v:v + 1], g[v:v + 1]))
        acc_llr.add_view(np.array([0]), probs[v:v + 1], 0.5, plain.view_llr(probs[v:v + 1], None))
    thr = np.full(NUM_CLASSES, 0.5, np.float32)
    f_cens, f_llr = fuse(acc, thr, 1, cens), fuse(acc_llr, thr, 1, plain)
    f_mean = fuse(ChunkAccumulator(acc.sum, acc.count, acc.votes, acc.n_views, acc.camera_stats, -1), thr, 1)
    assert f_mean.score[0, 0] == pytest.approx((0.9 + 0.5) / 11, rel=1e-6) and not f_mean.labels.any()
    assert f_llr.score[0, 0] < 0.5 and not f_llr.labels.any()
    assert f_cens.score[0, 0] > 0.5 and f_cens.labels.all()
    np.testing.assert_allclose(f_cens.score_mean, f_mean.score)


def test_fuse_posterior_nan_without_views_and_param_checks():
    m = _model()
    acc = ChunkAccumulator.zeros(3, 1, llr=True)
    acc.add_view(np.array([1]), np.full((1, NUM_CLASSES), 0.1), 0.5, m.view_llr(np.full((1, NUM_CLASSES), 0.1),
                                                                                 np.array([1e-6])))
    f = fuse(acc, np.full(NUM_CLASSES, 0.5, np.float32), 1, m)
    assert np.isnan(f.score[[0, 2]]).all()
    np.testing.assert_allclose(f.score[1], 0.1, rtol=1e-5)  # p = rho, one resolving view -> rho
    with pytest.raises(ValueError, match="rho"):
        FusionModel.from_dict("llr", {"classes": {"crack": {"rho": 0.1}}})
    bad = _params()
    bad["classes"]["crack"]["beta"] = -1.0
    with pytest.raises(ValueError, match="beta"):
        FusionModel.from_dict("censored_llr", bad)
    with pytest.raises(ValueError, match="llr evidence"):
        fuse(ChunkAccumulator.zeros(2, 1), np.full(NUM_CLASSES, 0.5, np.float32), 1, m)


# --------------------------------------------------------------------------- #
# End to end
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def project_n(tmp_path_factory):
    """Transformed-frame project with true surface normals (+z), for the incidence angle."""
    return build_project(tmp_path_factory.mktemp("proj_normals"), chunk_transform=True, true_normals=True)


def _run(project, out, *extra):
    assert main(["--project-root", str(project["root"]), "--output-dir", str(out), "--max-points", "6000",
                 "--start-image", "0", "--end-image", "5", *extra]) == 0
    return json.loads((out / "outputs" / "summary.json").read_text())


def _shards(out):
    arrays = {}
    for shard in sorted((out / "outputs" / "fused").glob("*.npz")):
        with np.load(shard) as d:
            for k in d.files:
                arrays.setdefault(k, []).append(d[k])
    return {k: np.concatenate(v) for k, v in arrays.items()}


def _write_params(tmp_path, **over):
    path = tmp_path / "fusion_params.json"
    path.write_text(json.dumps(_params(**over)))
    return path


def test_censored_run_log_and_offline_recomputation(project_n, tmp_path):
    params = _write_params(tmp_path)
    out = tmp_path / "cens"
    summary = _run(project_n, out, "--fusion", "censored_llr", "--fusion-params", str(params), "--observation-log",
                   "--point-chunk-size", "1500", "--camera-batch-size", "2")
    assert summary["fusion"]["mode"] == "censored_llr"
    d = _shards(out)
    assert {"score_mean", "llr"} <= set(d)
    obs = load_observations(out)
    assert obs.dtype == OBS_DTYPE
    assert len(obs) == int(d["n_views"].sum())          # one record per valid point-view

    # Geometry: GSD = Z * s / f and, for the deck under a nadir camera, cos ~ 1.
    np.testing.assert_allclose(obs["gsd_frontal"], obs["depth"] * CHUNK_SCALE / CALIB["f"], rtol=1e-6)
    deck = obs[(obs["camera"] == 2) & (obs["depth"] > 5.9)]
    assert len(deck) and abs(np.median(deck["gsd_frontal"]) - 6 * CHUNK_SCALE / CALIB["f"]) < 1e-3
    xyz = project_n["xyz_chunk"][deck["source_index"]]
    expected = 6.0 / np.linalg.norm(xyz - np.array([0.0, 0.2, 6.0]), axis=1)   # nadir camera at (0, 0.2, 6)
    np.testing.assert_allclose(deck["cos_theta"], expected, atol=1e-5)

    # Offline recomputation from the log = online accumulation (mean and censored evidence).
    m = FusionModel.load("censored_llr", params)
    p = probs_of(obs)
    # sigma_v = W / W_m: the synthetic provider writes maps downsampled by 8.
    np.testing.assert_allclose(obs["map_scale"], WIDTH / round(WIDTH / 8), rtol=1e-6)
    lam = m.view_llr(p, m.view_gsd(obs["gsd_frontal"], obs["cos_theta"], obs["map_scale"]))
    pts, inv = np.unique(obs["source_index"], return_inverse=True)
    llr = np.zeros((len(pts), NUM_CLASSES))
    psum = np.zeros((len(pts), NUM_CLASSES))
    np.add.at(llr, inv, np.nan_to_num(lam))
    np.add.at(psum, inv, np.nan_to_num(p))
    order = np.searchsorted(d["source_index"], pts)
    assert (d["source_index"][order] == pts).all()
    np.testing.assert_allclose(d["llr"][order], llr, atol=1e-4)
    np.testing.assert_allclose(d["score_mean"][order], psum / d["views"][order], atol=1e-6)
    np.testing.assert_allclose(d["score"][order], m.posterior(llr, d["views"][order]), atol=1e-4)


def test_mean_outputs_unchanged_by_the_observation_log(project_n, tmp_path):
    _run(project_n, tmp_path / "plain")
    _run(project_n, tmp_path / "logged", "--observation-log")
    a, b = _shards(tmp_path / "plain"), _shards(tmp_path / "logged")
    assert set(a) == set(b) and "llr" not in a
    for k in a:
        np.testing.assert_array_equal(a[k], b[k])
    ply = lambda p: (p / "outputs" / "fused_points.ply").read_bytes()  # noqa: E731
    assert ply(tmp_path / "plain") == ply(tmp_path / "logged")
    fp = json.loads((tmp_path / "plain" / "manifest.json").read_text())["fingerprint"]["accumulation"]
    assert "fusion" not in fp and "observation_log" not in fp    # old mean runs keep their fingerprint


def test_resume_does_not_duplicate_observations(project_n, tmp_path):
    params = _write_params(tmp_path)
    common = ["--fusion", "censored_llr", "--fusion-params", str(params), "--observation-log",
              "--point-chunk-size", "1000", "--camera-batch-size", "2"]
    _run(project_n, tmp_path / "ref", *common)
    with pytest.raises(SimulatedInterruption):
        _run(project_n, tmp_path / "cut", *common, "--test-interrupt-after-writes", "5")
    _run(project_n, tmp_path / "cut", *common, "--resume")
    np.testing.assert_array_equal(load_observations(tmp_path / "ref"), load_observations(tmp_path / "cut"))
    np.testing.assert_array_equal(_shards(tmp_path / "ref")["llr"], _shards(tmp_path / "cut")["llr"])

    (tmp_path / "other").mkdir()
    changed = [str(_write_params(tmp_path / "other", alpha=0.5)) if a == str(params) else a for a in common]
    with pytest.raises(RuntimeError, match="accumulation"):  # evidence is accumulated: params are frozen
        _run(project_n, tmp_path / "cut", *changed, "--resume")


def test_censored_chunk_invariance_and_sealed_provenance(project_n, tmp_path):
    from damage3d.experiment_data import provenance
    from damage3d.observations import ObservationLog
    params = _write_params(tmp_path)
    common = ["--fusion", "censored_llr", "--fusion-params", str(params), "--observation-log"]
    a, b = tmp_path/"small_chunks", tmp_path/"large_chunks"
    _run(project_n, a, *common, "--point-chunk-size", "1000", "--camera-batch-size", "2")
    _run(project_n, b, *common, "--point-chunk-size", "3000", "--camera-batch-size", "5")
    for key in ("score", "score_mean", "llr", "views"):
        np.testing.assert_array_equal(_shards(a)[key], _shards(b)[key])
    # Hash/progress contract must accept legitimate culled chunks/missing empty shards.
    region_file = tmp_path/"region_stub.npz"
    np.savez(region_file, source_index=[1])
    prov = provenance(a, region_file)
    assert len(prov["observation_records_sha256"]) == 64
    shard = next((a/"observations").glob("chunk_*_batch_*.npy"))
    shard.unlink()
    with pytest.raises(ValueError, match="inventory"):
        ObservationLog(a).seal(verify_only=True)


def test_observation_subset(project_n, tmp_path):
    subset = np.arange(0, project_n["n_points"], 7)
    np.save(tmp_path / "subset.npy", subset)
    _run(project_n, tmp_path / "sub", "--observation-log", "--observation-subset", str(tmp_path / "subset.npy"))
    obs = load_observations(tmp_path / "sub")
    assert len(obs) and np.isin(obs["source_index"], subset).all()
    meta = json.loads((tmp_path / "sub" / "observations" / "meta.json").read_text())
    assert meta["subset"] == len(subset) and len(meta["cameras"]) == 5
    with pytest.raises(RuntimeError, match="observation-log"):
        _run(project_n, tmp_path / "bad", "--observation-subset", str(tmp_path / "subset.npy"))


def test_fusion_params_from_calibration(tmp_path):
    from damage3d.fusion_params import main as fp_main
    from damage3d.classes import REPO_ROOT
    cal = REPO_ROOT / "outputs/runs/p1ml_t12_upernet_convnext_base_512_ade20k/eval_curves/probability_calibration.json"
    if not cal.is_file():
        pytest.skip("T12 calibration JSON not in this checkout")
    out = tmp_path / "fp.json"
    assert fp_main(["--calibration", str(cal), "--out", str(out)]) == 0
    diag = json.loads(cal.read_text())["diagnostics"]["crack"]
    rho = json.loads(out.read_text())["classes"]["crack"]["rho"]
    assert rho == diag["n_positive_pixels"] / (diag["n_positive_pixels"] + diag["n_negative_pixels"])
    m = FusionModel.load("llr", out)
    # base_rate moves only the offset of the posterior, not the evidence.
    d = json.loads(out.read_text())
    d["classes"]["crack"]["base_rate"] = 0.5
    m2 = FusionModel.from_dict("llr", d)
    p = np.full((1, NUM_CLASSES), 0.3)
    np.testing.assert_allclose(m.view_llr(p, None), m2.view_llr(p, None))
    assert m2.posterior(np.zeros((1, NUM_CLASSES)), np.ones((1, NUM_CLASSES)))[0, 0] == pytest.approx(0.5)
