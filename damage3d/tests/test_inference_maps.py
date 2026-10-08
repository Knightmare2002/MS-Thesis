"""End-to-end check of scripts/23_infer_probability_maps.py -> damage3d files provider.

A real smp model is built from a tiny config; its segmentation head is forced
to a constant output (spalling ~1, every other channel ~0) so the expected
labels are known exactly. Skipped when torch / segmentation_models_pytorch
are not installed.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("segmentation_models_pytorch")
cv2 = pytest.importorskip("cv2")
yaml = pytest.importorskip("yaml")

from damage3d.cli import main as damage3d_main  # noqa: E402
from damage3d.classes import STATE_DAMAGE  # noqa: E402
from damage3d.tests.scene import HEIGHT, WIDTH  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
CAMERAS = ["DSC01010", "DSC01020", "DSC01030"]


def _load_script():
    spec = importlib.util.spec_from_file_location("infer23", REPO / "scripts" / "23_infer_probability_maps.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _make_run(run_dir: Path) -> None:
    from src.models.unet import build_model
    from src.utils import Config

    cfg = {
        "model": {"arch": "unet", "encoder": "resnet18", "encoder_weights": None,
                  "in_channels": 3, "classes": 6},
        "data": {"p1ml_patch": {"patch_size": 256, "eval_stride": 192, "eval_batch_size": 4,
                                "blend_mode": "gaussian"}},
    }
    run_dir.mkdir(parents=True)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")
    model = build_model(Config(cfg["model"]))
    head = model.segmentation_head[0]
    with torch.no_grad():
        head.weight.zero_()
        head.bias.copy_(torch.tensor([-12.0, 12.0, -12.0, -12.0, -12.0, -12.0]))
    torch.save({"model": model.state_dict(), "epoch": 0}, run_dir / "best.pt")
    calib = run_dir / "eval_multilabel_sliding_calibrated"
    calib.mkdir()
    thresholds = {f"threshold_{c}": 0.5 for c in
                  ("crack", "spalling", "corrosion", "moisture", "delamination", "surface")}
    (calib / "thresholds_per_class_validation_calibrated.json").write_text(
        json.dumps({"thresholds": thresholds}), encoding="utf-8")


def test_inference_maps_feed_damage3d(project_id, tmp_path):
    run_dir = tmp_path / "run"
    _make_run(run_dir)
    images = tmp_path / "images"
    images.mkdir()
    rng = np.random.default_rng(0)
    for stem in CAMERAS:
        cv2.imwrite(str(images / f"{stem}.JPG"), rng.integers(0, 255, (HEIGHT, WIDTH, 3), dtype=np.uint8))
    maps = tmp_path / "maps"
    common = ["--run-dir", str(run_dir), "--project-root", str(project_id["root"]),
              "--images-dir", str(images), "--output-dir", str(maps), "--start-image", "1", "--end-image", "4"]

    script = _load_script()
    assert script.main([*common, "--downsample", "2"]) == 0
    for stem in CAMERAS:
        with np.load(maps / f"{stem}.npz") as d:
            assert d["probs"].shape == (6, HEIGHT // 2, WIDTH // 2)
            assert d["probs"][1].min() > 0.99 and d["probs"][[0, 2, 3, 4, 5]].max() < 0.01
    manifest = json.loads((maps / "maps_manifest.json").read_text())
    assert sorted(manifest["maps"]) == CAMERAS

    # Second call skips existing maps; changed settings are refused.
    mtime = (maps / "DSC01020.npz").stat().st_mtime_ns
    assert script.main([*common, "--downsample", "2"]) == 0
    assert (maps / "DSC01020.npz").stat().st_mtime_ns == mtime
    with pytest.raises(SystemExit, match="different settings"):
        script.main([*common, "--downsample", "4"])

    out = tmp_path / "fused"
    assert damage3d_main(["--project-root", str(project_id["root"]), "--output-dir", str(out),
                          "--probability-source", "files", "--probabilities-dir", str(maps),
                          "--start-image", "1", "--end-image", "4", "--max-points", "4000",
                          "--thresholds", "0.5"]) == 0
    shards = sorted((out / "outputs" / "fused").glob("*.npz"))
    arrays = {}
    for shard in shards:
        with np.load(shard) as d:
            for k in d.files:
                arrays.setdefault(k, []).append(d[k])
    d = {k: np.concatenate(v) for k, v in arrays.items()}
    seen = d["n_views"] > 0
    assert seen.any() and (d["n_views"] >= 2).any()
    assert (d["label_mask"][seen] == 2).all()          # spalling only, no other class
    assert (d["state"][seen] == STATE_DAMAGE).all()
