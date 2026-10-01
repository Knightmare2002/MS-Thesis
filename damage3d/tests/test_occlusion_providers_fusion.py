"""Unit tests: mesh occlusion, probability providers, sampling, six channels, fusion."""

from __future__ import annotations

import numpy as np
import pytest

from damage3d.classes import (DAMAGE_CLASSES, STATE_DAMAGE, STATE_INSUFFICIENT_VIEWS, STATE_NOT_OBSERVED,
                              STATE_OBSERVED_NO_DAMAGE, class_bit)
from damage3d.fusion import ChunkAccumulator, fuse, parse_thresholds
from damage3d.providers import FileProvider, ProbabilityMap, SyntheticProvider, save_probability_npz
from damage3d.sampling import sample_probabilities
from damage3d.tests import scene


# ---------------------------------------------------------------- occlusion
@pytest.fixture(scope="module")
def occ_chunk(project_id):
    from damage3d.mesh_visibility import MeshOcclusion
    return MeshOcclusion(project_id["root"] / "model_medium_quality.obj")


def test_occluder_hides_deck_points(occ_chunk):
    center = np.array([0.0, 0.2, 6.0])
    under = np.array([[0.0, 0.0, 0.0], [0.5, -0.5, 0.0]])        # below the plate
    open_deck = np.array([[3.0, 1.5, 0.0], [-4.0, -1.0, 0.0]])  # free deck
    top = np.array([[0.2, 0.3, 1.05], [-0.7, 0.6, 1.05]])       # on top of the plate
    vis = occ_chunk.visible(np.vstack((under, open_deck, top)), center, abs_tol=1e-3)
    assert list(vis) == [False, False, True, True, True, True]


def test_occlusion_tolerance_and_miss(occ_chunk):
    center = np.array([3.0, 1.5, 6.0])
    slightly_behind = np.array([[3.0, 1.5, -0.004]])  # 4 mm behind the deck surface
    assert not occ_chunk.visible(slightly_behind, center, abs_tol=0.001)[0]
    assert occ_chunk.visible(slightly_behind, center, abs_tol=0.01)[0]
    outside_mesh = np.array([[20.0, 20.0, 0.0]])       # no triangle along the ray -> counted visible
    assert occ_chunk.visible(outside_mesh, np.array([20.0, 20.0, 6.0]), abs_tol=1e-3)[0]


def test_surface_deviation_stats(occ_chunk):
    pts = np.array([[0.0, 1.5, 0.01], [2.0, 1.0, -0.02]])
    stats = occ_chunk.surface_deviation_stats(pts)
    assert stats["n"] == 2 and stats["max"] == pytest.approx(0.02, rel=1e-3)


# ---------------------------------------------------------------- providers
def test_synthetic_provider_six_overlapping_channels():
    pmap = SyntheticProvider(downsample=4).get("DSC00001", 640, 480)
    assert pmap.probs.shape == (6, 120, 160) and pmap.probs.dtype == np.float32
    assert 0.0 <= pmap.probs.min() and pmap.probs.max() <= 1.0
    positive = pmap.probs >= 0.5
    assert positive.any(axis=(1, 2)).all()                  # every class present
    assert (positive.sum(axis=0) >= 2).any()                # overlapping classes exist
    pmap.validate(640, 480, "DSC00001")


def test_synthetic_mask_provider(tmp_path):
    from PIL import Image
    mask = np.zeros((480, 640), np.uint8)
    mask[100:200, 300:400] = 255
    Image.fromarray(mask).save(tmp_path / "DSC00001_mask_test.png")
    prov = SyntheticProvider(pattern="mask", mask_dir=tmp_path, mask_classes=("crack", "moisture"))
    assert prov.check_available(["DSC00001", "DSC00002"]) == ["DSC00002"]
    pmap = prov.get("DSC00001", 640, 480)
    assert pmap.probs[0, 150, 350] == 1.0 and pmap.probs[3, 150, 350] == 1.0 and pmap.probs[1, 150, 350] == 0.0
    with pytest.raises(ValueError, match="does not match"):
        prov.get("DSC00001", 641, 480)


def test_file_provider_roundtrip_and_class_order(tmp_path):
    probs = np.random.default_rng(0).uniform(size=(6, 60, 80)).astype(np.float32)
    save_probability_npz(tmp_path / "DSC00001.npz", probs, dtype="float32")
    prov = FileProvider(tmp_path)
    assert prov.check_available(["DSC00001", "DSC00002"]) == ["DSC00002"]
    np.testing.assert_array_equal(prov.get("DSC00001", 640, 480).probs, probs)
    np.savez(tmp_path / "DSC00003.npz", probs=probs, classes=np.array(DAMAGE_CLASSES[::-1]))
    with pytest.raises(ValueError, match="class order"):
        prov.get("DSC00003", 640, 480)
    with pytest.raises(ValueError, match="aspect ratio"):
        prov.get("DSC00001", 640, 640)
    save_probability_npz(tmp_path / "DSC00004.npz", probs, dtype="uint8")
    np.testing.assert_allclose(prov.get("DSC00004", 640, 480).probs, probs, atol=0.5 / 255 + 1e-7)


# ---------------------------------------------------------------- sampling
def _ramp_map(w=8, h=4):
    probs = np.zeros((6, h, w), np.float32)
    xs = np.arange(w, dtype=np.float32)
    for c in range(6):
        probs[c] = (xs[None, :] + c) / 20.0
    return ProbabilityMap(probs)


def test_nearest_uses_floor_and_bilinear_is_exact_at_centres():
    pmap = _ramp_map()
    u = np.array([0.0, 0.99, 1.0, 7.99])
    v = np.full(4, 2.0)
    near, _ = sample_probabilities(pmap, u, v, 8, 4, "nearest")
    np.testing.assert_allclose(near[:, 0], np.array([0, 0, 1, 7]) / 20.0)
    centres = np.array([0.5, 3.5, 7.5])
    bil, _ = sample_probabilities(pmap, centres, np.full(3, 1.5), 8, 4, "bilinear")
    np.testing.assert_allclose(bil[:, 2], (np.array([0, 3, 7]) + 2) / 20.0, atol=1e-7)
    mid, _ = sample_probabilities(pmap, np.array([1.0]), np.array([1.5]), 8, 4, "bilinear")
    assert mid[0, 0] == pytest.approx(0.5 / 20.0)       # halfway between centres 0.5 and 1.5


def test_bilinear_border_clamp_and_downscaled_map():
    pmap = _ramp_map()
    edge, _ = sample_probabilities(pmap, np.array([0.1, 7.9]), np.array([0.1, 3.9]), 8, 4, "bilinear")
    np.testing.assert_allclose(edge[:, 0], [0.0, 7 / 20.0], atol=1e-7)  # edge replication
    # Same map interpreted as a /10 downscale of an 80x40 image.
    down, _ = sample_probabilities(pmap, np.array([35.0]), np.array([15.0]), 80, 40, "bilinear")
    assert down[0, 0] == pytest.approx(3.0 / 20.0)


def test_nan_channel_and_valid_mask():
    pmap = _ramp_map()
    pmap.probs[2, :, 4:] = np.nan
    pmap.valid = np.ones((4, 8), bool)
    pmap.valid[:, 7] = False
    probs, valid = sample_probabilities(pmap, np.array([1.5, 5.5, 7.8]), np.array([1.5, 1.5, 1.5]), 8, 4, "bilinear")
    assert np.isfinite(probs[0]).all() and np.isnan(probs[1, 2]) and np.isfinite(probs[1, [0, 1, 3, 4, 5]]).all()
    assert list(valid) == [True, True, False]


# ---------------------------------------------------------------- fusion
def test_accumulation_mean_counts_votes():
    acc = ChunkAccumulator.zeros(4, n_cameras=2)
    p1 = np.array([[0.9, 0.8, 0.1, 0.0, 0.0, 0.0], [0.2] * 6], np.float32)
    p2 = np.array([[0.7, 0.2, np.nan, 0.0, 0.0, 0.0]], np.float32)
    acc.add_view(np.array([0, 1]), p1, vote_threshold=0.5)
    acc.add_view(np.array([0]), p2, vote_threshold=0.5)
    assert list(acc.n_views) == [2, 1, 0, 0]
    assert list(acc.count[0]) == [2, 2, 1, 2, 2, 2]
    assert list(acc.votes[0]) == [2, 1, 0, 0, 0, 0]
    fused = fuse(acc, parse_thresholds("0.5"), min_views=1)
    assert fused.score[0, 0] == pytest.approx(0.8) and fused.score[0, 1] == pytest.approx(0.5)
    assert np.isnan(fused.score[2]).all()


def test_overlapping_classes_and_states():
    acc = ChunkAccumulator.zeros(5, n_cameras=1)
    views = np.array([
        [0.9, 0.0, 0.0, 0.8, 0.0, 0.0],   # crack + moisture
        [0.1] * 6,                        # observed, no damage
        [0.1, 0.1, np.nan, 0.1, 0.1, 0.1],  # class 2 never valid -> insufficient
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.95],  # surface
    ], np.float32)
    acc.add_view(np.array([0, 1, 2, 3]), views, 0.5)
    fused = fuse(acc, parse_thresholds("0.5"), min_views=1)
    assert fused.labels[0].tolist() == [True, False, False, True, False, False]
    assert fused.label_mask[0] == class_bit("crack") | class_bit("moisture")
    assert list(fused.state) == [STATE_DAMAGE, STATE_OBSERVED_NO_DAMAGE, STATE_INSUFFICIENT_VIEWS,
                                 STATE_DAMAGE, STATE_NOT_OBSERVED]
    assert tuple(fused.rgb[4]) == (40, 70, 160)          # unobserved is never grey/healthy
    assert tuple(fused.rgb[1]) == (200, 200, 200)
    # min_views = 2: one view is not enough for any decision.
    fused2 = fuse(acc, parse_thresholds("0.5"), min_views=2)
    assert not fused2.labels.any() and set(fused2.state[:4]) == {STATE_INSUFFICIENT_VIEWS}


def test_dominant_color_uses_threshold_ratio():
    acc = ChunkAccumulator.zeros(1, 1)
    acc.add_view(np.array([0]), np.array([[0.6, 0.0, 0.0, 0.0, 0.0, 0.9]], np.float32), 0.5)
    thr = parse_thresholds("crack=0.3,surface=0.85")
    fused = fuse(acc, thr, 1)
    assert fused.labels[0, 0] and fused.labels[0, 5]
    assert fused.dominant[0] == 0          # 0.6/0.3 = 2.0 > 0.9/0.85


def test_parse_thresholds():
    assert parse_thresholds("0.4").tolist() == pytest.approx([0.4] * 6)
    thr = parse_thresholds("crack=0.6,surface=0.7")
    assert thr[0] == pytest.approx(0.6) and thr[5] == pytest.approx(0.7) and thr[1] == pytest.approx(0.5)
    with pytest.raises(ValueError):
        parse_thresholds("rust=0.5")


def test_accumulator_atomic_save(tmp_path):
    acc = ChunkAccumulator.zeros(3, 2)
    acc.add_view(np.array([1]), np.full((1, 6), 0.7, np.float32), 0.5)
    acc.last_batch = 4
    acc.save_atomic(tmp_path / "chunk_000000.npz")
    assert not list(tmp_path.glob("*.tmp"))
    back = ChunkAccumulator.load(tmp_path / "chunk_000000.npz")
    assert back.last_batch == 4 and np.array_equal(back.sum, acc.sum)
