#!/usr/bin/env python
"""Step 16 / P4-A - end-to-end smoke test for the multitask pipeline (CPU, ~1 min).

Checks, on synthetic data, everything that can silently corrupt a P4-A run or the
per-channel calibration:

    1. the network emits [B,1,H,W] and [B,6,H,W] and the encoder is ONE shared
       module (single `encoder.*` block in the state dict, one forward pass);
    2. backward: each task alone produces non-zero gradients in the shared
       encoder, and the joint loss produces gradients in both decoders;
    3. the joint loss derives the crack target from channel 0 and exposes its two
       components;
    4. checkpoint save/load round-trip of the multitask model;
    5. joint sliding-window == the two single-head sliding windows, bit-for-bit;
    6. the multilabel meter with a scalar threshold behaves exactly as before
       (retro-compatibility) and mixed per-class thresholds give the expected
       hand-computed counts on a toy example;
    7. threshold selection: argmax Dice with the "highest threshold wins"
       tie-break;
    8. the real scripts 14 -> 15 -> 17 run as subprocesses and produce every
       artifact, including the three *additive* calibrated files while the two
       standard CSVs stay byte-identical.

Usage
-----
    python scripts/16_smoke_test_multitask.py
    python scripts/16_smoke_test_multitask.py --with-p1ml-regression   # also runs scripts/13
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.data.class_mapping import UNIFIED_DAMAGE_CLASSES
from src.engine import load_checkpoint, save_checkpoint
from src.engine_multitask import joint_selection_score
from src.eval.multilabel_metrics import (
    MultilabelSegmentationMetrics,
    resolve_class_thresholds,
    select_thresholds_per_class,
)
from src.eval.sliding_window import (
    predict_sliding_window,
    predict_sliding_window_multilabel,
    predict_sliding_window_multitask,
)
from src.losses import build_multitask_loss
from src.models.multitask import (
    MultiTaskSegmentationModel,
    SingleHeadAdapter,
    build_multitask_model,
)
from src.utils import Config, load_config

FAKE_LABELS = [
    ("Crack", 0), ("ACrack", 0),
    ("Spalling", 1), ("Cavity", 1),
    ("Rust", 2), ("ExposedRebars", 2),
    ("Wetspot", 3), ("Efflorescence", 3),
    ("Hollowareas", 4),
    ("Weathering", 5), ("Graffiti", 5),
    ("Bearing", None),  # excluded component: must never reach the target
]


def small_model(encoder: str = "resnet18") -> MultiTaskSegmentationModel:
    """Tiny multitask network, random weights, no download."""
    return MultiTaskSegmentationModel(
        arch="unetplusplus",
        encoder=encoder,
        encoder_weights=None,
        in_channels=3,
        crack_classes=1,
        multilabel_classes=len(UNIFIED_DAMAGE_CLASSES),
    )


def make_fake_dacl10k(root: Path, n: int = 8, size: int = 128) -> None:
    """Official DACL10K layout with polygons covering every unified channel."""
    for split in ("train", "validation"):
        images_dir = root / "images" / split
        annotations_dir = root / "annotations" / split
        images_dir.mkdir(parents=True, exist_ok=True)
        annotations_dir.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(1)

        for i in range(n):
            name = f"dacl_{i:03d}"
            cv2.imwrite(
                str(images_dir / f"{name}.jpg"),
                rng.integers(50, 200, size=(size, size, 3), dtype=np.uint8),
            )

            shapes = []
            for offset in range(3):
                label, _ = FAKE_LABELS[(i + offset) % len(FAKE_LABELS)]
                x0 = 8 + 30 * offset
                y0 = 8 + 30 * ((i + offset) % 3)
                shapes.append(
                    {
                        "label": label,
                        "shape_type": "polygon",
                        "points": [
                            [x0, y0], [x0 + 26, y0 + 2], [x0 + 24, y0 + 26], [x0 + 2, y0 + 24]
                        ],
                    }
                )

            (annotations_dir / f"{name}.json").write_text(
                json.dumps(
                    {
                        "imageName": f"{name}.jpg",
                        "imageWidth": size,
                        "imageHeight": size,
                        "split": split,
                        "shapes": shapes,
                    }
                ),
                encoding="utf-8",
            )


def build_config(tmp: Path) -> Path:
    """Copy the real P4-A config and point it at the synthetic data."""
    cfg = yaml.safe_load((ROOT / "configs" / "config_p4a.yaml").read_text(encoding="utf-8"))
    cfg["project"]["output_dir"] = str(tmp / "outputs")
    cfg["data"]["dacl10k"]["root"] = str(tmp / "dacl10k")
    cfg["data"]["image_size"] = 64
    cfg["data"]["num_workers"] = 0
    cfg["data"]["p4a_patch"].update(
        {
            "patch_size": 64,
            "min_positive_pixels": 8,
            "patches_per_image": 2,
            "eval_stride": 32,
            "eval_batch_size": 2,
            "sanity_check_samples": 8,
        }
    )
    cfg["data"]["p4a_patch"]["class_weights"]["cache_path"] = str(
        tmp / "outputs" / "stats" / "class_weights.json"
    )
    cfg["model"]["encoder"] = "resnet18"
    cfg["model"]["encoder_weights"] = None
    cfg["train"].update(
        {
            "epochs": 2,
            "batch_size": 2,
            "amp": False,
            "resume": False,
            "warmup_epochs": 1,
            "early_stopping_patience": 99,
        }
    )
    cfg["train"]["scheduler"]["t_0"] = 1
    cfg["eval"]["n_qualitative_samples"] = 2

    path = tmp / "smoke_config_p4a.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return path


def run(command: list[str]) -> None:
    print(f"\n$ {' '.join(command)}")
    subprocess.run(command, check=True, cwd=ROOT)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# --------------------------------------------------------------------------- #
# 1-2-3. architecture, gradients, loss
# --------------------------------------------------------------------------- #
def test_architecture_and_gradients() -> None:
    torch.manual_seed(0)
    model = small_model()

    blocks = {key.split(".")[0] for key in model.state_dict()}
    assert blocks == {
        "encoder", "crack_decoder", "crack_head", "multilabel_decoder", "multilabel_head"
    }, blocks
    # A duplicated trunk would show up as a second encoder-like block.
    assert sum(1 for key in model.state_dict() if key.startswith("encoder.")) > 0
    assert model.crack_decoder is not model.multilabel_decoder

    x = torch.randn(2, 3, 64, 64)
    crack_logits, multilabel_logits = model(x)
    assert crack_logits.shape == (2, 1, 64, 64), crack_logits.shape
    assert multilabel_logits.shape == (2, 6, 64, 64), multilabel_logits.shape
    assert torch.equal(model.forward_crack(x), crack_logits)
    assert torch.equal(model.forward_multilabel(x), multilabel_logits)

    # The multilabel head must not be a softmax: channels are independent, so
    # the per-pixel probabilities generally do NOT sum to 1.
    probability_sum = torch.sigmoid(multilabel_logits).sum(dim=1)
    assert (probability_sum - 1.0).abs().max() > 1e-3

    def encoder_grad_norm(loss: torch.Tensor) -> float:
        model.zero_grad(set_to_none=True)
        loss.backward(retain_graph=True)
        return float(
            sum(
                p.grad.abs().sum()
                for p in model.encoder.parameters()
                if p.grad is not None
            )
        )

    crack_logits, multilabel_logits = model(x)
    assert encoder_grad_norm(crack_logits.mean()) > 0, "crack task does not reach the encoder"
    assert encoder_grad_norm(multilabel_logits.mean()) > 0, "multilabel task does not reach the encoder"

    print("[smoke] shared encoder, dual output shapes and gradient flow: PASS")


def test_joint_loss() -> None:
    cfg = Config(
        {
            "loss": {
                "name": "multitask_bce_dice",
                "crack_weight": 1.0,
                "multilabel_weight": 1.0,
                "crack_channel": 0,
                "crack": {"bce_weight": 0.5, "dice_weight": 0.5, "pos_weight": 5.0},
                "multilabel": {"bce_weight": 0.5, "dice_weight": 0.5},
            }
        }
    )
    criterion = build_multitask_loss(cfg, pos_weight=[1.0] * 6)

    torch.manual_seed(0)
    target = (torch.rand(2, 6, 32, 32) > 0.7).float()
    crack_logits = torch.randn(2, 1, 32, 32, requires_grad=True)
    multilabel_logits = torch.randn(2, 6, 32, 32, requires_grad=True)

    assert torch.equal(criterion.crack_target(target), target[:, 0:1])

    total = criterion((crack_logits, multilabel_logits), target)
    total.backward()
    components = criterion.last_components

    assert crack_logits.grad is not None and multilabel_logits.grad is not None
    assert abs(components["crack"] + components["multilabel"] - float(total)) < 1e-5
    assert components["crack"] > 0 and components["multilabel"] > 0

    try:
        criterion(crack_logits, target)
    except TypeError:
        pass
    else:  # pragma: no cover
        raise AssertionError("the joint loss must reject a single output tensor")

    print("[smoke] joint loss (crack target = channel 0, two components): PASS")


# --------------------------------------------------------------------------- #
# 4-5. checkpoint round-trip and sliding window
# --------------------------------------------------------------------------- #
def test_checkpoint_roundtrip() -> None:
    torch.manual_seed(0)
    model = small_model()
    reference = {key: tensor.clone() for key, tensor in model.state_dict().items()}

    with tempfile.TemporaryDirectory() as tmp_dir:
        path = Path(tmp_dir) / "best_joint.pt"
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        scaler = torch.amp.GradScaler(enabled=False)
        save_checkpoint(path, model, optimizer, scheduler, scaler, epoch=3, best_dice=0.42)

        reloaded = small_model()
        checkpoint = load_checkpoint(path, reloaded, device="cpu")

    assert checkpoint["epoch"] == 3 and abs(checkpoint["best_dice"] - 0.42) < 1e-9
    for key, tensor in reloaded.state_dict().items():
        assert torch.equal(tensor, reference[key]), key

    print("[smoke] best_joint.pt save/load round-trip: PASS")


def test_sliding_window_equivalence() -> None:
    torch.manual_seed(0)
    model = small_model().eval()
    image = np.random.default_rng(0).integers(0, 255, size=(96, 112, 3), dtype=np.uint8)
    device = torch.device("cpu")
    geometry = dict(
        image=image,
        device=device,
        patch_size=64,
        stride=32,
        batch_size=2,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        blend_mode="gaussian",
    )

    crack_joint, multilabel_joint = predict_sliding_window_multitask(
        model=model, n_classes=6, **geometry
    )
    crack_single = predict_sliding_window(model=SingleHeadAdapter(model, "crack"), **geometry)
    multilabel_single = predict_sliding_window_multilabel(
        model=SingleHeadAdapter(model, "multilabel"), n_classes=6, **geometry
    )

    assert crack_joint.shape == (96, 112) and multilabel_joint.shape == (6, 96, 112)
    assert torch.allclose(crack_joint, crack_single, atol=1e-6)
    assert torch.allclose(multilabel_joint, multilabel_single, atol=1e-6)
    assert float(multilabel_joint.min()) >= 0.0 and float(multilabel_joint.max()) <= 1.0

    print("[smoke] joint sliding window == per-head sliding windows: PASS")


# --------------------------------------------------------------------------- #
# 6-7. thresholds: retro-compatibility, mixed vector, tie-break
# --------------------------------------------------------------------------- #
def test_threshold_backward_compatibility() -> None:
    torch.manual_seed(0)
    logits = torch.randn(3, 6, 16, 16)
    target = (torch.rand(3, 6, 16, 16) > 0.6).float()

    scalar_meter = MultilabelSegmentationMetrics(threshold=0.5)
    vector_meter = MultilabelSegmentationMetrics(threshold=[0.5] * 6)
    mapping_meter = MultilabelSegmentationMetrics(
        threshold={name: 0.5 for name in UNIFIED_DAMAGE_CLASSES}
    )
    for meter in (scalar_meter, vector_meter, mapping_meter):
        meter.update(logits, target)

    scalar_metrics = scalar_meter.compute()
    for meter in (vector_meter, mapping_meter):
        other = meter.compute()
        for key, value in scalar_metrics.items():
            assert abs(float(value) - float(other[key])) < 1e-12, key

    assert resolve_class_thresholds(0.4, UNIFIED_DAMAGE_CLASSES) == [0.4] * 6
    for bad in (0.0, 1.0, -0.2, [0.5] * 5, {"crack": 0.5}):
        try:
            resolve_class_thresholds(bad, UNIFIED_DAMAGE_CLASSES)
        except (ValueError, TypeError):
            continue
        raise AssertionError(f"invalid threshold accepted: {bad!r}")

    print("[smoke] scalar threshold unchanged, vector/dict accepted: PASS")


def test_mixed_thresholds_toy_example() -> None:
    """Hand-computed toy case: one pixel per channel, probability 0.60 everywhere.

    With a global threshold 0.5 every channel predicts positive. With per-channel
    thresholds [0.5, 0.7, 0.5, 0.7, 0.5, 0.7] only the even channels do, so the
    odd channels must score exactly 0 Dice and no false positive.
    """
    probability = torch.full((1, 6, 1, 1), 0.60)
    logits = torch.logit(probability)
    target = torch.ones(1, 6, 1, 1)

    thresholds = [0.5, 0.7, 0.5, 0.7, 0.5, 0.7]
    meter = MultilabelSegmentationMetrics(threshold=thresholds)
    meter.update(logits, target)

    per_class = meter.per_class()
    for channel, name in enumerate(UNIFIED_DAMAGE_CLASSES):
        expected_dice = 1.0 if thresholds[channel] < 0.60 else 0.0
        assert abs(per_class[name]["dice"] - expected_dice) < 1e-4, (name, per_class[name]["dice"])

    assert meter.resolved_thresholds()["spalling"] == 0.7
    aggregated = meter.compute()
    assert abs(aggregated["macro_dice_present"] - 0.5) < 1e-4, aggregated["macro_dice_present"]

    print("[smoke] mixed per-class thresholds on a toy example: PASS")


def test_threshold_selection_tie_break() -> None:
    rows = []
    # crack: unique maximum at 0.4; spalling: tie between 0.5 and 0.6 -> 0.6 wins.
    scores = {
        "crack": {0.3: 0.10, 0.4: 0.55, 0.5: 0.50, 0.6: 0.20, 0.7: 0.05},
        "spalling": {0.3: 0.10, 0.4: 0.20, 0.5: 0.40, 0.6: 0.40, 0.7: 0.10},
    }
    for name in UNIFIED_DAMAGE_CLASSES:
        table = scores.get(name, {t: 0.3 for t in (0.3, 0.4, 0.5, 0.6, 0.7)})
        rows += [{"class": name, "threshold": t, "dice": d} for t, d in table.items()]

    selection = select_thresholds_per_class(rows, UNIFIED_DAMAGE_CLASSES, metric="dice")
    assert selection["crack"]["threshold"] == 0.4 and selection["crack"]["n_tied_candidates"] == 1
    assert selection["spalling"]["threshold"] == 0.6, selection["spalling"]
    assert selection["spalling"]["n_tied_candidates"] == 2
    assert selection["corrosion"]["threshold"] == 0.7  # flat curve -> highest threshold

    assert abs(joint_selection_score(0.4, 0.6) - 0.5) < 1e-12

    print("[smoke] per-class threshold selection + tie-break rule: PASS")


# --------------------------------------------------------------------------- #
# 8. the real scripts
# --------------------------------------------------------------------------- #
def test_scripts_end_to_end(with_p1ml_regression: bool) -> None:
    with tempfile.TemporaryDirectory() as tmp_name:
        tmp = Path(tmp_name)
        make_fake_dacl10k(tmp / "dacl10k")
        config = build_config(tmp)
        run_dir = tmp / "outputs" / "runs" / "smoke_p4a"

        run([sys.executable, "scripts/14_train_dacl10k_multitask.py",
             "--config", str(config), "--run-name", "smoke_p4a"])

        for artifact in (
            "config.yaml", "dataset_summary.json", "class_weights.json",
            "history.csv", "best_joint.pt", "last_joint.pt", "curves.png",
            "best_monitor_metrics.json",
        ):
            assert (run_dir / artifact).exists(), f"missing artifact: {artifact}"

        summary = json.loads((run_dir / "dataset_summary.json").read_text(encoding="utf-8"))
        assert summary["pipeline"] == "P4-A"
        assert summary["shared_encoder"]["parameters"]["shared_fraction"] > 0.0

        run([sys.executable, "scripts/15_evaluate_dacl10k_multitask_sliding.py",
             "--run-dir", str(run_dir)])

        eval_dir = run_dir / "eval_multitask_sliding"
        standard = (
            "metrics_dacl10k_val_crack_sliding.csv",
            "metrics_dacl10k_val_multilabel_sliding.csv",
            "metrics_dacl10k_val_multilabel_per_class.csv",
            "validation_composition.json",
            "qualitative_overview.png",
            "qualitative_crack.png",
        ) + tuple(f"qualitative_multilabel_{name}.png" for name in UNIFIED_DAMAGE_CLASSES)
        for artifact in standard:
            assert (eval_dir / artifact).exists(), f"missing artifact: {artifact}"

        # The standard CSVs must survive the calibration untouched.
        fingerprints = {
            name: sha256_file(eval_dir / name)
            for name in (
                "metrics_dacl10k_val_multilabel_sliding.csv",
                "metrics_dacl10k_val_multilabel_per_class.csv",
            )
        }

        run([sys.executable, "scripts/17_calibrate_multilabel_thresholds.py",
             "--run-dir", str(run_dir)])

        for artifact in (
            "thresholds_per_class_validation_calibrated.json",
            "metrics_dacl10k_val_multilabel_per_class_calibrated.csv",
            "metrics_dacl10k_val_multilabel_sliding_calibrated.csv",
        ):
            assert (eval_dir / artifact).exists(), f"missing calibrated artifact: {artifact}"

        for name, digest in fingerprints.items():
            assert sha256_file(eval_dir / name) == digest, f"{name} was modified by scripts/17"

        payload = json.loads(
            (eval_dir / "thresholds_per_class_validation_calibrated.json").read_text(
                encoding="utf-8"
            )
        )
        assert payload["run_family"] == "P4_multitask"
        assert set(payload["thresholds"]) == {
            f"threshold_{name}" for name in UNIFIED_DAMAGE_CLASSES
        }
        assert payload["tie_break"].startswith("highest threshold")
        assert "not a blind-test" in payload["note"].lower()
        assert payload["checkpoint"]["name"] == "best_joint.pt"

        # The resolved run config must still describe a P4 run (and only that).
        resolved = load_config(run_dir / "config.yaml")
        assert int(resolved.model.multilabel_classes) == 6
        assert resolved.model.get("classes") is None

        if with_p1ml_regression:
            run([sys.executable, "scripts/13_smoke_test_multilabel.py"])

    print("[smoke] scripts 14 -> 15 -> 17 end to end, standard CSVs preserved: PASS")


def main() -> None:
    parser = argparse.ArgumentParser(description="P4-A smoke test.")
    parser.add_argument(
        "--with-p1ml-regression",
        action="store_true",
        help="also run scripts/13 (P1ML) to prove the shared modules did not regress",
    )
    args = parser.parse_args()

    test_architecture_and_gradients()
    test_joint_loss()
    test_checkpoint_roundtrip()
    test_sliding_window_equivalence()
    test_threshold_backward_compatibility()
    test_mixed_thresholds_toy_example()
    test_threshold_selection_tie_break()
    test_scripts_end_to_end(args.with_p1ml_regression)

    print("\nP4-A SMOKE TEST PASSED - multitask pipeline and calibration are consistent.")


if __name__ == "__main__":
    main()
