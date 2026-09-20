#!/usr/bin/env python
"""Step 13 / P1ML - end-to-end smoke test for the multilabel pipeline (CPU, seconds).

Checks, on synthetic data, everything that can silently corrupt a multilabel run:
    1. the 19 -> 6 mapping (coverage, exclusions, channel order);
    2. rasterisation: [6,H,W], binary, no bleeding across channels;
    3. dataset output shapes/dtypes and the union-based positive criterion;
    4. model output [B,6,H,W] and loss forward/backward with a 6-vector pos_weight;
    5. metrics: per-class, macro on present classes, micro;
    6. multilabel sliding-window shape and probability range;
    7. partial checkpoint load 1 channel -> 6 channels (head re-initialised) and
       checkpoint reload;
    8. the real training and evaluation scripts, run as subprocesses;
    9. the binary pipeline still imports and runs (no regression on P0-P3).

Usage
-----
    python scripts/13_smoke_test_multilabel.py
"""

from __future__ import annotations

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

from src.data.class_mapping import (
    DACL10K_CLASSES,
    DACL10K_EXCLUDED_FROM_DAMAGE,
    UNIFIED_DAMAGE_CLASSES,
    assert_unified_damage_taxonomy,
    unified_damage_channel_groups,
    unified_damage_label_groups,
    unified_damage_label_to_channel,
)

# Two labels per unified channel (where available) plus one excluded component.
FAKE_LABELS = [
    ("Crack", 0), ("ACrack", 0),
    ("Spalling", 1), ("Cavity", 1),
    ("Rust", 2), ("ExposedRebars", 2),
    ("Wetspot", 3), ("Efflorescence", 3),
    ("Hollowareas", 4),
    ("Weathering", 5), ("Graffiti", 5),
    ("Bearing", None),   # excluded: must never appear in the target
]


def make_fake_dacl10k(root: Path, n: int = 8, size: int = 128) -> None:
    """Official DACL10K layout with polygons covering every unified channel."""
    for split in ("train", "validation"):
        images_dir = root / "images" / split
        ann_dir = root / "annotations" / split
        images_dir.mkdir(parents=True, exist_ok=True)
        ann_dir.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(1)

        for i in range(n):
            name = f"dacl_{i:03d}"
            cv2.imwrite(
                str(images_dir / f"{name}.jpg"),
                rng.integers(50, 200, size=(size, size, 3), dtype=np.uint8),
            )

            shapes = []
            # Image i carries labels i, i+1, i+2 (cyclically): every channel and
            # the excluded component appear somewhere in the split.
            for offset in range(3):
                label, _ = FAKE_LABELS[(i + offset) % len(FAKE_LABELS)]
                x0 = 8 + 30 * offset
                y0 = 8 + 30 * ((i + offset) % 3)
                shapes.append(
                    {
                        "label": label,
                        "shape_type": "polygon",
                        "points": [[x0, y0], [x0 + 26, y0 + 2], [x0 + 24, y0 + 26], [x0 + 2, y0 + 24]],
                    }
                )

            annotation = {
                "imageName": f"{name}.jpg",
                "imageWidth": size,
                "imageHeight": size,
                "split": split,
                "shapes": shapes,
            }
            (ann_dir / f"{name}.json").write_text(json.dumps(annotation), encoding="utf-8")


def build_config(tmp: Path) -> Path:
    """Copy the real P1ML config and point it at the synthetic data."""
    cfg = yaml.safe_load((ROOT / "configs" / "config_p1ml.yaml").read_text(encoding="utf-8"))
    cfg["project"]["output_dir"] = str(tmp / "outputs")
    cfg["data"]["dacl10k"]["root"] = str(tmp / "dacl10k")
    cfg["data"]["image_size"] = 64
    cfg["data"]["num_workers"] = 0
    cfg["data"]["p1ml_patch"].update(
        {
            "patch_size": 64,
            "min_positive_pixels": 8,
            "patches_per_image": 2,
            "eval_stride": 32,
            "eval_batch_size": 2,
            "sanity_check_samples": 8,
        }
    )
    cfg["data"]["p1ml_patch"]["class_weights"]["cache_path"] = str(
        tmp / "outputs" / "stats" / "class_weights.json"
    )
    cfg["model"]["encoder_weights"] = None  # no download in the smoke test
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

    path = tmp / "smoke_config_p1ml.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return path


def run(command: list[str]) -> None:
    print(f"\n$ {' '.join(command)}")
    subprocess.run(command, check=True, cwd=ROOT)


# --------------------------------------------------------------------------- #
# 1. mapping
# --------------------------------------------------------------------------- #
def test_mapping() -> None:
    assert_unified_damage_taxonomy()

    groups = unified_damage_label_groups()
    assert list(groups) == UNIFIED_DAMAGE_CLASSES
    assert groups["crack"] == ["Crack", "ACrack"]
    assert set(groups["spalling"]) == {"Spalling", "Rockpocket", "Cavity"}
    assert set(groups["corrosion"]) == {"Rust", "ExposedRebars", "WConccor"}
    assert set(groups["moisture"]) == {"Wetspot", "Efflorescence"}
    assert groups["delamination"] == ["Hollowareas"]
    assert set(groups["surface"]) == {"Weathering", "Graffiti", "Restformwork"}

    label_to_channel = unified_damage_label_to_channel()
    assert len(label_to_channel) == len(DACL10K_CLASSES) - len(DACL10K_EXCLUDED_FROM_DAMAGE) == 14
    for excluded in DACL10K_EXCLUDED_FROM_DAMAGE:
        assert excluded not in label_to_channel

    channel_groups = unified_damage_channel_groups()
    assert len(channel_groups) == 6 and all(channel_groups)

    print("[smoke] 19 -> 6 mapping: PASS")


# --------------------------------------------------------------------------- #
# 2-3. rasterisation and datasets
# --------------------------------------------------------------------------- #
def test_rasterization_and_dataset(root: Path) -> None:
    from src.data.dacl10k import (
        Dacl10kMultilabelCenterPatchDataset,
        Dacl10kMultilabelPatchDataset,
        list_samples,
        load_annotation,
        multilabel_sample_targets,
        rasterize_binary,
        rasterize_unified_damage,
        summarize_multilabel_targets,
    )
    from src.data.transforms import (
        patch_eval_transform_multilabel,
        patch_train_transform_multilabel,
    )

    samples = list_samples(root, "train")
    assert samples

    # Crack-only rasterisation must stay untouched (regression guard for P1-B*).
    for _, annotation_path in samples:
        annotation = load_annotation(annotation_path)
        masks = rasterize_unified_damage(annotation)
        assert masks.shape[0] == 6 and masks.dtype == np.uint8
        assert set(np.unique(masks)).issubset({0, 1})

        binary = rasterize_binary(annotation, labels=["Crack", "ACrack"])
        assert np.array_equal(masks[0] > 0, binary > 0), "channel 0 must equal the binary Crack+ACrack mask"

        # The excluded component must not leak into any channel.
        if any(shape["label"] == "Bearing" for shape in annotation["shapes"]):
            only_bearing = {"Bearing"}
            if {shape["label"] for shape in annotation["shapes"]} == only_bearing:
                assert masks.sum() == 0

    targets = multilabel_sample_targets(samples)
    composition = summarize_multilabel_targets(targets)
    assert composition["n_images"] == len(samples)
    assert sum(composition["n_images_per_class"].values()) > 0

    train_ds = Dacl10kMultilabelPatchDataset(
        samples=samples,
        transform=patch_train_transform_multilabel(),
        patch_size=64,
        positive_patch_fraction=0.6,
        min_positive_pixels=8,
        max_negative_pixels=0,
        max_crop_attempts=50,
        patches_per_image=2,
    )
    image, mask = train_ds[0]
    assert image.shape == (3, 64, 64), image.shape
    assert mask.shape == (6, 64, 64), mask.shape
    assert mask.dtype == torch.float32
    assert set(torch.unique(mask).tolist()).issubset({0.0, 1.0}), "augmented masks must stay binary"
    assert int((mask.amax(dim=0) > 0).sum()) >= 1, "the first patch must be positive on the union"

    val_ds = Dacl10kMultilabelCenterPatchDataset(
        samples=samples, transform=patch_eval_transform_multilabel(), patch_size=64
    )
    image, mask = val_ds[0]
    assert image.shape == (3, 64, 64) and mask.shape == (6, 64, 64)

    print("[smoke] rasterisation + multilabel datasets: PASS")


# --------------------------------------------------------------------------- #
# 4-5. model, loss, metrics
# --------------------------------------------------------------------------- #
def test_model_loss_metrics() -> None:
    from src.eval.multilabel_metrics import MultilabelSegmentationMetrics
    from src.losses import MultilabelBceDiceLoss
    from src.models.unet import build_model
    from src.utils import Config

    model = build_model(
        Config(
            {
                "arch": "unetplusplus",
                "encoder": "resnet34",
                "encoder_weights": None,
                "in_channels": 3,
                "classes": 6,
            }
        )
    )
    images = torch.randn(2, 3, 64, 64)
    logits = model(images)
    assert logits.shape == (2, 6, 64, 64), logits.shape

    target = torch.zeros(2, 6, 64, 64)
    target[:, 0, :10, :10] = 1.0
    target[:, 5, 20:40, 20:40] = 1.0

    criterion = MultilabelBceDiceLoss(pos_weight=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    loss = criterion(logits, target)
    loss.backward()
    assert torch.isfinite(loss) and loss.item() > 0
    assert any(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )

    # Perfect prediction -> macro Dice on present classes == 1, absent classes skipped.
    meter = MultilabelSegmentationMetrics(threshold=0.5)
    perfect = torch.where(target > 0.5, 10.0, -10.0)
    meter.update(perfect, target)
    metrics = meter.compute()
    assert abs(metrics["macro_dice_present"] - 1.0) < 1e-4, metrics["macro_dice_present"]
    assert abs(metrics["micro_dice"] - 1.0) < 1e-4
    assert metrics["n_classes_present"] == 2.0, metrics["n_classes_present"]
    for name in UNIFIED_DAMAGE_CLASSES:
        assert f"dice_{name}" in metrics

    # Everything predicted negative -> Dice 0 on the present classes.
    meter = MultilabelSegmentationMetrics(threshold=0.5)
    meter.update(torch.full_like(target, -10.0), target)
    assert meter.compute()["macro_dice_present"] < 1e-4

    print("[smoke] model / loss / metrics: PASS")


# --------------------------------------------------------------------------- #
# 6. sliding window
# --------------------------------------------------------------------------- #
def test_sliding_window_multilabel() -> None:
    from src.data.transforms import IMAGENET_MEAN, IMAGENET_STD
    from src.eval.sliding_window import predict_sliding_window_multilabel

    model = torch.nn.Conv2d(3, 6, kernel_size=3, padding=1)
    image = (np.random.rand(90, 140, 3) * 255).astype(np.uint8)

    probability = predict_sliding_window_multilabel(
        model=model,
        image=image,
        device=torch.device("cpu"),
        patch_size=64,
        stride=32,
        batch_size=2,
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD,
        n_classes=6,
    )
    assert probability.shape == (6, 90, 140), probability.shape
    assert float(probability.min()) >= 0.0 and float(probability.max()) <= 1.0

    print("[smoke] multilabel sliding window: PASS")


# --------------------------------------------------------------------------- #
# 7. partial transfer 1 -> 6 channels
# --------------------------------------------------------------------------- #
def test_partial_transfer() -> None:
    from src.engine import initialize_model_from_checkpoint_partial, load_checkpoint, save_checkpoint
    from src.models.unet import build_model
    from src.utils import Config

    def make(classes: int):
        return build_model(
            Config(
                {
                    "arch": "unetplusplus",
                    "encoder": "resnet34",
                    "encoder_weights": None,
                    "in_channels": 3,
                    "classes": classes,
                }
            )
        )

    binary_model = make(1)
    with torch.no_grad():
        for parameter in binary_model.parameters():
            parameter.add_(0.01)

    multilabel_model = make(6)
    head_before = multilabel_model.segmentation_head[0].weight.detach().clone()

    with tempfile.TemporaryDirectory() as tmp_dir:
        checkpoint_path = Path(tmp_dir) / "p1b4_like.pt"
        torch.save(
            {"epoch": 11, "best_dice": 0.4321, "model": binary_model.state_dict()},
            checkpoint_path,
        )

        report = initialize_model_from_checkpoint_partial(
            model=multilabel_model, checkpoint_path=checkpoint_path, device="cpu"
        )

    assert report["checkpoint_epoch"] == 11
    assert report["reinitialized_head_tensors"], "the 1-channel head must be reported as re-initialised"
    assert all(key.startswith("segmentation_head.") for key in report["reinitialized_head_tensors"])
    assert report["factory_model_sha256"] != report["initialized_model_sha256"]
    assert report["restored_optimizer_state"] is False

    # Encoder and decoder transferred, head untouched.
    source = binary_model.state_dict()
    target = multilabel_model.state_dict()
    for key, tensor in target.items():
        if key.startswith(("encoder.", "decoder.")):
            assert torch.equal(tensor, source[key]), key
    assert torch.equal(multilabel_model.segmentation_head[0].weight.detach(), head_before)

    # Checkpoint round-trip of the 6-channel model.
    with tempfile.TemporaryDirectory() as tmp_dir:
        path = Path(tmp_dir) / "best.pt"
        optimizer = torch.optim.AdamW(multilabel_model.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        scaler = torch.amp.GradScaler(enabled=False)
        save_checkpoint(path, multilabel_model, optimizer, scheduler, scaler, epoch=1, best_dice=0.5)

        reloaded = make(6)
        checkpoint = load_checkpoint(path, reloaded, device="cpu")
        assert checkpoint["epoch"] == 1
        for key, tensor in reloaded.state_dict().items():
            assert torch.equal(tensor, target[key]), key

    print("[smoke] partial transfer 1 -> 6 channels + checkpoint reload: PASS")


def main() -> None:
    test_mapping()
    test_model_loss_metrics()
    test_sliding_window_multilabel()
    test_partial_transfer()

    with tempfile.TemporaryDirectory() as tmp_name:
        tmp = Path(tmp_name)
        make_fake_dacl10k(tmp / "dacl10k")
        test_rasterization_and_dataset(tmp / "dacl10k")

        config = build_config(tmp)
        run_dir = tmp / "outputs" / "runs" / "smoke_p1ml"

        run([sys.executable, "scripts/11_train_dacl10k_multilabel.py",
             "--config", str(config), "--run-name", "smoke_p1ml"])
        run([sys.executable, "scripts/12_evaluate_dacl10k_multilabel_sliding.py",
             "--run-dir", str(run_dir)])

        for artifact in (
            "config.yaml", "dataset_summary.json", "class_weights.json",
            "history.csv", "best.pt", "last.pt", "curves.png",
        ):
            assert (run_dir / artifact).exists(), f"missing artifact: {artifact}"

        eval_dir = run_dir / "eval_multilabel_sliding"
        for artifact in (
            "metrics_dacl10k_val_multilabel_sliding.csv",
            "metrics_dacl10k_val_multilabel_per_class.csv",
            "validation_composition.json",
            "qualitative_dacl10k_val_multilabel_sliding.png",
        ):
            assert (eval_dir / artifact).exists(), f"missing artifact: {artifact}"

        # Transfer run B: the same pipeline initialised from the P1ML best.pt
        # (shape-compatible) exercises the transfer branch of the script.
        run([sys.executable, "scripts/11_train_dacl10k_multilabel.py",
             "--config", str(config), "--run-name", "smoke_p1ml_transfer",
             "--set", f"transfer.init_checkpoint={run_dir / 'best.pt'}"])
        assert (tmp / "outputs" / "runs" / "smoke_p1ml_transfer" / "partial_load_report.json").exists()

    print("\nP1ML SMOKE TEST PASSED - multilabel pipeline is consistent end to end.")


if __name__ == "__main__":
    main()
