"""End-to-end tests on the synthetic project (single camera, range, chunk sizes, resume)."""

from __future__ import annotations

import json

import numpy as np
import pytest

from damage3d.cli import main
from damage3d.classes import STATE_DAMAGE, STATE_NOT_OBSERVED
from damage3d.pipeline import SimulatedInterruption
from damage3d.ply_io import PlyPointSource, make_selection


def _run(project, out, *extra):
    argv = ["--project-root", str(project["root"]), "--output-dir", str(out), *extra]
    assert main(argv) == 0
    return json.loads((out / "outputs" / "summary.json").read_text())


def _load_shards(out):
    arrays = {}
    for shard in sorted((out / "outputs" / "fused").glob("*.npz")):
        with np.load(shard) as d:
            for k in d.files:
                arrays.setdefault(k, []).append(d[k])
    return {k: np.concatenate(v) for k, v in arrays.items()}


def test_single_camera_few_points_produces_visible_ply(project_tf, tmp_path):
    out = tmp_path / "one"
    summary = _run(project_tf, out, "--camera", "DSC01020", "--max-points", "3000")
    ply = PlyPointSource(out / "outputs" / "fused_points.ply")
    assert ply.n_points == 3000
    names = [n for n, _ in ply.header.properties]
    assert {"red", "green", "blue", "state", "label_mask", "n_views", "score_crack", "views_surface"} <= set(names)
    ply.close()
    import open3d as o3d
    cloud = o3d.io.read_point_cloud(str(out / "outputs" / "fused_points.ply"))
    assert len(cloud.points) == 3000 and cloud.has_colors()
    assert summary["synthetic"] is True and summary["frame"]["resolved"] == "transformed"
    assert summary["state_counts"]["damage"] > 0 and summary["state_counts"]["not_observed"] > 0
    assert summary["points_with_two_or_more_labels"] > 0
    assert (out / "outputs" / "fused_points.csv").is_file()
    stats = (out / "outputs" / "per_camera_stats.csv").read_text().splitlines()
    assert len(stats) == 2 and ",DSC01020.JPG," in stats[1]


def test_output_coordinates_equal_source_and_occluded_points_unobserved(project_tf, tmp_path):
    out = tmp_path / "coords"
    _run(project_tf, out, "--camera", "DSC01020", "--max-points", "4000", "--sample-mode", "stride")
    d = _load_shards(out)
    src = PlyPointSource(project_tf["root"] / "bridge_3dpc_point_cloud.ply")
    np.testing.assert_array_equal(d["xyz"], src.xyz(d["source_index"]).astype(np.float32))
    src.close()
    chunk = project_tf["xyz_chunk"][d["source_index"]]
    under_plate = (np.abs(chunk[:, 0]) < 0.8) & (np.abs(chunk[:, 1]) < 0.8) & (chunk[:, 2] == 0.0)
    assert under_plate.sum() > 10
    assert (d["state"][under_plate] == STATE_NOT_OBSERVED).all()      # occluded by the plate
    top = chunk[:, 2] > 1.0
    assert (d["n_views"][top] == 1).all()                              # plate top seen


def test_camera_range_and_multiview_counts(project_tf, tmp_path):
    out = tmp_path / "range"
    summary = _run(project_tf, out, "--start-image", "1", "--end-image", "4", "--max-points", "5000")
    rows = (out / "selected_cameras.csv").read_text().splitlines()[1:]
    assert [r.split(",")[2] for r in rows] == ["DSC01010.JPG", "DSC01020.JPG", "DSC01030.JPG"]
    d = _load_shards(out)
    assert d["n_views"].max() == 3 and summary["n_cameras"] == 3
    assert (d["views"] <= d["n_views"][:, None]).all()
    assert (d["votes"] <= d["views"]).all()


def test_chunk_size_and_batch_size_do_not_change_results(project_tf, tmp_path):
    common = ["--start-image", "0", "--end-image", "5", "--max-points", "6000"]
    _run(project_tf, tmp_path / "a", *common, "--point-chunk-size", "6000", "--camera-batch-size", "5")
    _run(project_tf, tmp_path / "b", *common, "--point-chunk-size", "777", "--camera-batch-size", "2")
    a, b = _load_shards(tmp_path / "a"), _load_shards(tmp_path / "b")
    for key in ("source_index", "xyz", "views", "votes", "n_views", "label_mask", "state"):
        np.testing.assert_array_equal(a[key], b[key])
    np.testing.assert_array_equal(np.nan_to_num(a["score"], nan=-1), np.nan_to_num(b["score"], nan=-1))  # bitwise


def test_resume_after_interruption_does_not_double_count(project_tf, tmp_path):
    common = ["--start-image", "0", "--end-image", "5", "--max-points", "6000",
              "--point-chunk-size", "500", "--camera-batch-size", "2"]
    _run(project_tf, tmp_path / "ref", *common)
    out = tmp_path / "interrupted"
    with pytest.raises(SimulatedInterruption):
        main(["--project-root", str(project_tf["root"]), "--output-dir", str(out), *common,
              "--test-interrupt-after-writes", "17"])
    progress = json.loads((out / "progress.json").read_text())
    assert progress["completed_batches"] < 3                          # stopped inside a batch
    with pytest.raises(RuntimeError, match="--resume"):
        _run(project_tf, out, *common)                                # refuses to overwrite
    _run(project_tf, out, *common, "--resume")
    ref, res = _load_shards(tmp_path / "ref"), _load_shards(out)
    for key in ("views", "votes", "n_views", "state", "label_mask"):
        np.testing.assert_array_equal(ref[key], res[key])
    np.testing.assert_array_equal(np.nan_to_num(ref["score"], nan=-1), np.nan_to_num(res["score"], nan=-1))
    # A second resume of a finished run adds nothing.
    _run(project_tf, out, *common, "--resume")
    np.testing.assert_array_equal(_load_shards(out)["n_views"], ref["n_views"])


def test_resume_refuses_changed_parameters(project_tf, tmp_path):
    out = tmp_path / "r"
    _run(project_tf, out, "--camera", "DSC01020", "--max-points", "1000")
    with pytest.raises(RuntimeError, match="differ"):
        _run(project_tf, out, "--camera", "DSC01030", "--max-points", "1000", "--resume")
    with pytest.raises(RuntimeError, match="differ"):
        _run(project_tf, out, "--camera", "DSC01020", "--max-points", "1000", "--seed", "1", "--resume")
    # Export-only parameters may change on resume (re-export, no re-accumulation).
    s = _run(project_tf, out, "--camera", "DSC01020", "--max-points", "1000", "--resume", "--thresholds", "0.9")
    assert s["thresholds"]["crack"] == pytest.approx(0.9)


def test_dry_run_writes_nothing(project_tf, tmp_path):
    out = tmp_path / "dry"
    assert main(["--project-root", str(project_tf["root"]), "--output-dir", str(out), "--dry-run",
                 "--camera", "DSC01020", "--max-points", "100"]) == 0
    assert not out.exists()


def test_missing_inputs_are_reported(project_tf, tmp_path):
    with pytest.raises(FileNotFoundError, match="point cloud"):
        main(["--project-root", str(project_tf["root"]), "--point-cloud", "nope.ply", "--dry-run"])
    with pytest.raises(FileNotFoundError, match="probability maps"):
        main(["--project-root", str(project_tf["root"]), "--probability-source", "files",
              "--probabilities-dir", str(tmp_path), "--camera", "DSC01020", "--dry-run"])


def test_files_provider_end_to_end(project_id, tmp_path):
    """Same pipeline with maps read from disk (the path the real model will use)."""
    from damage3d.providers import save_probability_npz
    probs = np.zeros((6, 48, 64), np.float32)
    probs[1] = 0.9          # spalling everywhere
    probs[4, :, :32] = 0.8  # delamination on the left half, overlapping spalling
    maps = tmp_path / "maps"
    maps.mkdir()
    save_probability_npz(maps / "DSC01020.npz", probs)
    out = tmp_path / "files"
    summary = _run(project_id, out, "--camera", "DSC01020", "--max-points", "4000",
                   "--probability-source", "files", "--probabilities-dir", str(maps))
    assert summary["synthetic"] is False and summary["frame"]["resolved"] == "chunk"
    d = _load_shards(out)
    seen = d["n_views"] > 0
    assert (d["label_mask"][seen] & 2).all()                           # spalling bit on every seen point
    assert ((d["label_mask"][seen] & 16) > 0).any() and ((d["label_mask"][seen] & 16) == 0).any()
    assert (d["state"][seen] == STATE_DAMAGE).all()


def test_point_sampling_is_reproducible():
    a = make_selection(1_000_000, 1000, "random", 5)
    b = make_selection(1_000_000, 1000, "random", 5)
    c = make_selection(1_000_000, 1000, "random", 6)
    assert a.sha256() == b.sha256() != c.sha256()
    assert np.all(np.diff(a.indices) > 0)
    s = make_selection(10, 4, "stride", 0)
    assert s.indices.tolist() == [0, 2, 5, 7]
    assert make_selection(10, None).indices is None
