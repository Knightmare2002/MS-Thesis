"""Methodological invariants, leakage checks, identifiability, and experiment CLI integration."""
import json
from dataclasses import replace
import numpy as np
import pytest

from damage3d.classes import DAMAGE_CLASSES, NUM_CLASSES
from damage3d.detectability import FusionModel, sigmoid
from damage3d.observations import OBS_DTYPE
from damage3d.resolvability_fit import held_out_samples, fit_curve, balanced_weights
from damage3d.tempering_fit import fit_alpha
from damage3d.experiment_metrics import binary_metrics, paired_bootstrap, axis_iou
from damage3d.experiment_data import regions
from damage3d.fusion_benchmark import point_scores, camera_subset


def params():
    return {"classes": {c: {"rho": .1, "g50": .1, "beta": .4, "alpha": 1.}
                        for c in DAMAGE_CLASSES}}


def observations(g=(.01, .02, .2, .5), p=(.95, .95, .05, .01)):
    a = np.zeros(len(g), dtype=OBS_DTYPE)
    a["camera"] = np.arange(len(g))
    a["gsd_frontal"], a["cos_theta"], a["map_scale"], a["range_source"] = g, 1, 1, np.arange(len(g))+1
    for c in DAMAGE_CLASSES:
        a[f"p_{c}"] = p
    return a


@pytest.mark.parametrize("key,value", [("rho", np.nan), ("alpha", np.inf), ("beta", np.nan),
                                      ("g50", -1), ("prior", .1)])
def test_reject_invalid_params(key, value):
    d = params()
    d["classes"]["crack"][key] = value
    with pytest.raises(ValueError):
        FusionModel.from_dict("censored_llr", d)


def test_no_silent_uncensored_fallback_and_invalid_geometry():
    d = params()
    del d["classes"]["crack"]["g50"]
    with pytest.raises(ValueError, match="g50"):
        FusionModel.from_dict("censored_llr", d)
    m = FusionModel.from_dict("censored_llr", params())
    with pytest.raises(ValueError, match="normal"):
        m.view_gsd(np.array([1.]), np.array([np.nan]))
    np.testing.assert_allclose(sigmoid([-1e9, 0, 1e9]), [0, .5, 1])


def test_mixture_is_not_generic_weighting():
    m = FusionModel.from_dict("censored_llr", params())
    p = np.repeat(np.array([.001, .05, .1, .7])[:, None], NUM_CLASSES, axis=1)
    lam = replace(m, mode="llr").view_llr(p, None)
    mixed = m.view_llr(p, np.full(4, .1))
    assert np.all(mixed >= .5*lam-1e-12)  # Jensen
    assert np.all(mixed[lam < 0] >= np.log(.5))
    assert np.all(np.sign(mixed) == np.sign(lam))
    assert np.max(abs(mixed-.5*lam)) > .1


def test_leave_view_out_and_group_leakage():
    m = FusionModel.from_dict("censored_llr", params())
    a = observations()
    cfg = {"top_k": 2, "min_resolution_ratio": 1.25, "reference_threshold": .9}
    ix, _ = held_out_samples(a, m, cfg, 0)
    np.testing.assert_array_equal(ix, [2, 3])
    a["p_crack"][2:] = .99  # changing held-out outcomes cannot change these candidates' eligibility
    np.testing.assert_array_equal(held_out_samples(a, m, cfg, 0)[0], [2, 3])
    a["p_crack"][2:] = [.05, .01]
    cfg["camera_groups"] = {"0": "same", "1": "same", "2": "other", "3": "another"}
    assert not len(held_out_samples(a, m, cfg, 0)[0])


def test_curve_recovers_known_parameters_with_nonperfect_detector():
    rng = np.random.default_rng(37)
    g = np.exp(rng.uniform(-4, 2, 30000))
    q = .08 + (.85-.08)*sigmoid((np.log(.3)-np.log(g))/.55)
    hits = rng.random(len(g)) < q
    fit, diag = fit_curve(g, hits, np.ones(len(g)), .85, .08)
    assert fit["g50"] == pytest.approx(.3, rel=.12)
    assert fit["beta"] == pytest.approx(.55, rel=.12)
    assert diag["nll"] < diag["constant_nll"]
    # Perfect-detector fit is generally biased when treating q as pi.
    wrong, _ = fit_curve(g, hits, np.ones(len(g)), 1., 0.)
    assert abs(wrong["beta"]-.55) > abs(fit["beta"]-.55)


def test_unidentifiable_curve_fails_closed():
    with pytest.raises(ValueError, match="Insufficient"):
        fit_curve(np.linspace(.01, 1, 100), np.ones(100), np.ones(100), .9, .05)
    with pytest.raises(ValueError, match="anchors"):
        fit_curve(np.ones(100), np.zeros(100), np.ones(100), .1, .2)


def test_balancing_is_per_region_then_point():
    p, r = np.array([1, 1, 2, 3, 3]), np.array([0, 0, 0, 1, 1])
    w = balanced_weights(p, r)
    assert w[r == 0].sum() == pytest.approx(.5)
    assert w[p == 1].sum() == pytest.approx(.25)
    assert w[p == 2].sum() == pytest.approx(.25)


def test_tempering_convex_fit_and_label_requirement():
    x = np.tile([-20., 20.], 100)
    y = np.tile([0., 1.], 100)
    y[::10] = 1
    a, diag = fit_alpha(x, y, np.ones(200), .1)
    assert 0 < a < 1
    assert diag["nll_after"] < diag["nll_before"]
    with pytest.raises(ValueError):
        fit_alpha(x, np.ones(200), np.ones(200), .1)


def test_weighted_ap_ties_and_empty_support():
    y, p, w = np.array([1, 0, 1]), np.array([.5, .5, .5]), np.array([1., 3., 2.])
    assert binary_metrics(y, p, w)["ap"] == pytest.approx(.5)
    assert binary_metrics(y[::-1], p, w[::-1]) == binary_metrics(y, p, w)
    assert np.isnan(binary_metrics([], [], [])["ap"])
    with pytest.raises(ValueError):
        binary_metrics([-1], [.3], [1])


def test_region_bootstrap_paired_identity_and_axis_counts():
    y = np.array([0, 1]*10)
    p = np.linspace(.01, .99, 20)
    out = paired_bootstrap(y, p, p, np.ones(20), np.repeat([0, 1, 2, 3], 5), repeats=50)
    assert out["brier"]["ci95"] == [0., 0.]
    t = np.array([[0, 0, 0], [1, 0, 0]])
    result = axis_iou(t, t+np.array([0, .01, 0]), [.005, .02])
    assert result[0]["iou"] == 0 and result[1]["iou"] == 1


def test_baselines_and_nested_view_cardinality():
    m = FusionModel.from_dict("censored_llr", params())
    a = observations()
    s, *_ = point_scores(a, m)
    assert s["mean"][0] == pytest.approx((.95+.95+.05+.01)/4)
    assert s["max"][0] == pytest.approx(.95)
    assert s["distance_range"][0] == pytest.approx(.95)
    a["range_source"] = 1
    s, *_ = point_scores(a, m)
    assert np.isnan(s["distance_range"]).all()
    np.testing.assert_allclose(s["distance_weighted"], s["mean"])
    small = camera_subset(np.arange(20), .25, 4)
    big = camera_subset(np.arange(20), .5, 4)
    assert set(small) <= set(big)


def test_spatial_split_guard(tmp_path):
    file = tmp_path/"regions.npz"
    data = dict(source_index=np.arange(3), region=np.arange(3),
                split=np.array(["C_fit", "C_cal", "T"]), xyz=np.array([[0, 0, 0], [2, 0, 0], [4, 0, 0]]),
                area=np.ones(3), labels=np.full((3, NUM_CLASSES), -1), classes=np.array(DAMAGE_CLASSES))
    np.savez(file, **data)
    regions(file, 1.)
    with pytest.raises(ValueError, match="buffer"):
        regions(file, 3.)
    data["region"][:] = 0
    np.savez(file, **data)
    with pytest.raises(ValueError, match="crosses"):
        regions(file, 1.)


def test_complete_experiment_cli_and_test_label_isolation(tmp_path):
    from damage3d.resolvability_fit import main as fit_main
    from damage3d.fusion_benchmark import main as bench_main
    from damage3d.tempering_fit import main as temper_main
    from damage3d.freeze_direction_a import main as freeze_main
    from damage3d.evaluate_direction_a import main as eval_main
    from damage3d.observations import ObservationLog
    rng = np.random.default_rng(42)
    run = tmp_path/"run"
    (run/"outputs").mkdir(parents=True)
    (run/"acc").mkdir()
    (run/"outputs/summary.json").write_text("{}")
    (run/"progress.json").write_text(json.dumps({"completed_batches": 1, "n_batches": 1}))
    (run/"manifest.json").write_text(json.dumps({"fingerprint": {
        "cameras": {"selected": list(range(6))}, "layout": {"camera_batch_size": 6}}}))
    np.savez(run/"acc/chunk_000000.npz", last_batch=0)
    n = 900
    y = np.arange(n) % 2
    obs = np.zeros(n*6, dtype=OBS_DTYPE)
    obs["source_index"] = np.repeat(np.arange(n), 6)
    obs["camera"] = np.tile(np.arange(6), n)
    g = np.column_stack([np.full(n, .0001), np.full(n, .0002), np.exp(rng.uniform(-4, 0, (n, 4)))])
    q = .05 + .85*sigmoid((np.log(.1)-np.log(g))/.4)
    p = np.where(rng.random((n, 6)) < np.where(y[:, None], q, .05), .95, .001)
    p[:, :2] = np.where(y[:, None], .99, .001)
    obs["gsd_frontal"], obs["cos_theta"], obs["map_scale"] = g.ravel(), 1, 1
    obs["range_source"] = np.tile(np.arange(1, 7), n)
    for c in DAMAGE_CLASSES:
        obs[f"p_{c}"] = p.ravel()
    ObservationLog(run).write(0, 0, [obs])
    ObservationLog(run).seal()
    (run/"observations/meta.json").write_text(json.dumps({
        "schema": 3, "provider": {"type": "synthetic"}, "cameras": [{"camera": k} for k in range(6)]}))
    rp = tmp_path/"regions.npz"
    data = dict(source_index=np.arange(n), region=np.arange(n)//100,
                split=np.repeat(["C_fit", "C_cal", "T"], 300),
                xyz=np.column_stack([np.arange(n)//100*10., np.arange(n)%100*.001, np.zeros(n)]),
                area=np.ones(n), labels=np.repeat(y[:, None], NUM_CLASSES, axis=1),
                classes=np.array(DAMAGE_CLASSES))
    np.savez(rp, **data)
    base, config, fit = tmp_path/"base.json", tmp_path/"config.json", tmp_path/"fit.json"
    base.write_text(json.dumps(params()))
    config.write_text(json.dumps({"common": {"top_k": 2, "min_resolution_ratio": 1.25,
                                            "reference_threshold": .9},
                                  "classes": {c: {"threshold": .5, "resolved_tpr": .9, "background_fpr": .05,
                                                   "anchor_source": "unit-test generative model"}
                                              for c in DAMAGE_CLASSES}}))
    common = ["--run", str(run), "--regions", str(rp), "--buffer", "1"]
    assert fit_main([*common, "--params", str(base), "--config", str(config),
                     "--bootstrap", "2", "--out", str(fit)]) == 0
    assert json.loads(fit.read_text())["fit_diagnostics"]["crack"]["region_bootstrap"]["requested"] == 2
    # Flip all T labels: the numerical fitting result MUST remain unchanged.
    data["labels"][data["split"] == "T"] = 1-data["labels"][data["split"] == "T"]
    rp2 = tmp_path/"regions_changed_test.npz"
    np.savez(rp2, **data)
    fit2 = tmp_path/"fit2.json"
    assert fit_main(["--run", str(run), "--regions", str(rp2), "--buffer", "1", "--params", str(base),
                     "--config", str(config), "--out", str(fit2)]) == 0
    assert json.loads(fit.read_text())["classes"] == json.loads(fit2.read_text())["classes"]
    scores = tmp_path/"scores.npz"
    bench_main([*common, "--params", str(fit), "--out", str(scores)])
    final = tmp_path/"final.json"
    temper_main(["--scores", str(scores), "--regions", str(rp), "--buffer", "1",
                 "--params", str(fit), "--out", str(final)])
    changed_scores, changed_final = tmp_path/"changed_scores.npz", tmp_path/"changed_final.json"
    bench_main(["--run", str(run), "--regions", str(rp2), "--buffer", "1",
                "--params", str(fit2), "--out", str(changed_scores)])
    temper_main(["--scores", str(changed_scores), "--regions", str(rp2), "--buffer", "1",
                 "--params", str(fit2), "--out", str(changed_final)])
    assert json.loads(final.read_text())["classes"] == json.loads(changed_final.read_text())["classes"]
    scored = tmp_path/"final_scores.npz"
    bench_main([*common, "--params", str(final), "--out", str(scored)])
    thresholds = tmp_path/"thresholds.json"
    freeze_main(["--scores", str(scored), "--regions", str(rp), "--buffer", "1", "--out", str(thresholds)])
    bins = tmp_path/"bins.json"
    bins.write_text(json.dumps({"g_edges": [0, .001, .1, 1], "incidence_edges": [0, 30, 60, 90]}))
    output = tmp_path/"evaluation.json"
    eval_main(["--scores", str(scored), "--regions", str(rp), "--buffer", "1", "--thresholds", str(thresholds),
               "--bins", str(bins), "--bootstrap", "10", "--out", str(output)])
    evaluation = json.loads(output.read_text())
    assert evaluation["split"] == "T"
    assert evaluation["classes"]["crack"]["n_regions"] == 3
    assert evaluation["classes"]["crack"]["methods"]["censored"]["metrics"]["nll"] is not None
    oracle = tmp_path/"oracle.npz"
    bench_main([*common, "--params", str(final), "--oracle", "--out", str(oracle)])
    with np.load(oracle) as x, np.load(scored) as normal:
        np.testing.assert_array_equal(x["censored"], normal["censored"])
        assert np.all(x["oracle_best_view"][y == 1] >= normal["mean"][y == 1])
        assert np.all(x["oracle_best_view"][y == 0] <= normal["mean"][y == 0])
    with pytest.raises(ValueError, match="non-oracle"):
        temper_main(["--scores", str(oracle), "--regions", str(rp), "--buffer", "1",
                     "--params", str(final), "--out", str(tmp_path/"forbidden.json")])


def test_preparation_and_axis_cli(tmp_path):
    from damage3d.prepare_direction_a import main as prepare
    from damage3d.crack_axis_metrics import main as axes
    degradation = tmp_path/"degradation.json"
    degradation.write_text(json.dumps({"aggregate": {"1": {c: {"recall": .8, "fpr": .1}
                                                              for c in DAMAGE_CLASSES}},
                                       "thresholds": [.5]*NUM_CLASSES, "calibration_sha256": "test",
                                       "classes": list(DAMAGE_CLASSES)}))
    cfg = tmp_path/"config.json"
    prepare(["--degradation", str(degradation), "--out", str(cfg)])
    assert json.loads(cfg.read_text())["classes"]["crack"]["resolved_tpr"] == .8
    with pytest.raises(FileExistsError):
        prepare(["--degradation", str(degradation), "--out", str(cfg)])
    np.save(tmp_path/"axis.npy", np.array([[0., 0, 0], [.001, 0, 0]]))
    output = tmp_path/"axis.json"
    axes(["--truth", str(tmp_path/"axis.npy"), "--prediction", str(tmp_path/"axis.npy"),
          "--metres-per-unit", "1", "--sampling-mm", "1", "--out", str(output)])
    assert all(row["iou"] == 1 for row in json.loads(output.read_text())["results"])
    with pytest.raises(ValueError):
        axis_iou(np.zeros((1, 3)), np.zeros((1, 3)), [np.nan])


def test_detector_provenance_guard(tmp_path):
    from damage3d.experiment_data import validate_detector
    (tmp_path/"observations").mkdir()
    meta = {"provider": {"type": "files+platt", "calibration": {"sha256": "cal"}},
            "calibration_prevalence": {c: .1 for c in DAMAGE_CLASSES}}
    path = tmp_path/"observations/meta.json"
    path.write_text(json.dumps(meta))
    p = params()
    p["calibration_sha256"] = "cal"
    validate_detector(tmp_path, p)
    p["classes"]["crack"]["rho"] = .2
    with pytest.raises(ValueError, match="rho"):
        validate_detector(tmp_path, p)
    p["classes"]["crack"]["rho"] = .1
    p["calibration_sha256"] = "wrong"
    with pytest.raises(ValueError, match="hash"):
        validate_detector(tmp_path, p)
    meta["provider"]["type"] = "ensemble"
    path.write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="single-model"):
        validate_detector(tmp_path, p)
