"""Experiment provenance shared by every training entry point.

Sequential-transfer runs must record which weights the optimization actually
started from, independently of the dataset involved: CrackSeg9K -> DACL10K (P2)
and DACL10K -> CrackSeg9K (P3) use the same records.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
import yaml


def to_jsonable(value: Any) -> Any:
    """
    Convert metadata recursively to JSON-native types.

    In particular, pathlib.Path objects are converted to strings so that experiment provenance remains portable across Windows and Linux.
    """
    if isinstance(value, Path):
        return str(value)

    if isinstance(value, dict):
        return {
            str(key): to_jsonable(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple, set)):
        return [
            to_jsonable(item)
            for item in value
        ]

    # Optional but robust: PyTorch device and tensor metadata.
    if isinstance(value, torch.device):
        return str(value)

    # Converts NumPy scalar values, if any, to Python scalar values.
    if hasattr(value, "item") and callable(value.item):
        try:
            return value.item()
        except (ValueError, RuntimeError):
            pass

    return value


def save_transfer_metadata(
    run_dir: Path,
    cfg,
    metadata: dict | None,
) -> None:
    """
    Save dataset-agnostic provenance for standard training or sequential transfer.

    `source_dataset` identifies the dataset used to train `init_checkpoint`;

    `target_dataset` identifies the dataset used by the current training run.
    """
    transfer_cfg = cfg.get("transfer", {})

    init_checkpoint = transfer_cfg.get("init_checkpoint")
    source_dataset = transfer_cfg.get("source_dataset")
    target_dataset = transfer_cfg.get("target_dataset")

    is_transfer = bool(init_checkpoint)

    if is_transfer:
        required_fields = {
            "transfer.source_name": transfer_cfg.get("source_name"),
            "transfer.source_dataset": source_dataset,
            "transfer.target_dataset": target_dataset,
        }

        missing = [
            field_name
            for field_name, value in required_fields.items()
            if not value
        ]

        if missing:
            raise ValueError(
                "[transfer] incomplete transfer configuration. "
                f"Required fields: {', '.join(missing)}"
            )

        if source_dataset == target_dataset:
            raise ValueError(
                "[transfer] source_dataset and target_dataset must differ "
                f"for sequential transfer; both are '{source_dataset}'."
            )

        direction = f"{source_dataset}_to_{target_dataset}"
        initialization = "external_checkpoint_model_weights_only"

    else:
        direction = None
        initialization = "model_factory_encoder_weights"

    record = {
        "experiment_type": (
            "sequential_transfer"
            if is_transfer
            else "standard_target_training"
        ),
        "source_name": (
            transfer_cfg.get("source_name")
            if is_transfer
            else None
        ),
        "source_dataset": (
            source_dataset
            if is_transfer
            else None
        ),
        "target_dataset": target_dataset,
        "direction": direction,
        "initialization": initialization,
        "optimizer_reinitialized": (
            bool(transfer_cfg.get("reset_optimizer", True))
            if is_transfer
            else True
        ),
        "scheduler_reinitialized": True,
        "amp_scaler_reinitialized": True,
        "transfer_config": dict(transfer_cfg),
        "checkpoint_load": metadata,
    }

    metadata_path = run_dir / "transfer_metadata.json"

    
    if metadata_path.exists():
        print(
            "[transfer] transfer_metadata.json already exists; "
            "keeping original initialization provenance."
        )
        return

    with open(metadata_path, "w", encoding="utf-8") as fh:
        json.dump(
            to_jsonable(record),
            fh,
            indent=2,
            ensure_ascii=False,
            sort_keys=False,
        )

    print(f"[transfer] metadata saved to {metadata_path}")


def save_resolved_run_config(
    run_dir: Path,
    cfg,
    transfer_metadata: dict | None,
    resumed: bool,
) -> None:
    """
    Save the effective configuration of a run.

    `model.encoder_weights` remains the model-factory bootstrap setting,
    while `effective_initialization` records the actual weights used when
    training begins.
    """
    run_cfg = dict(cfg)

    transfer_cfg = run_cfg.get("transfer", {})
    is_transfer = bool(transfer_cfg.get("init_checkpoint"))

    run_cfg["model"]["model_factory_encoder_weights"] = (
        run_cfg["model"].get("encoder_weights")
    )

    source_name = transfer_cfg.get("source_name", "external checkpoint")
    source_dataset = transfer_cfg.get("source_dataset", "unknown")
    target_dataset = transfer_cfg.get("target_dataset", "unknown")

    if is_transfer:
        run_cfg["model"]["effective_initialization"] = (
            f"full_model_checkpoint: {source_name} "
            f"({source_dataset}_to_{target_dataset})"
        )
        run_cfg["model"]["effective_checkpoint"] = str(
            transfer_cfg["init_checkpoint"]
        )
        run_cfg["model"]["effective_source_dataset"] = str(
            transfer_cfg["source_dataset"]
        )
        run_cfg["model"]["effective_target_dataset"] = str(
            transfer_cfg["target_dataset"]
        )
        run_cfg["model"]["effective_load_mode"] = str(
            transfer_cfg.get("load_mode", "full_model")
        )
        run_cfg["model"]["effective_strict_loading"] = bool(
            transfer_cfg.get("strict", True)
        )

        if transfer_metadata is not None:
            run_cfg["model"]["effective_source_model_sha256"] = (
                transfer_metadata.get("source_model_sha256")
            )
            run_cfg["model"]["effective_initialized_model_sha256"] = (
                transfer_metadata.get("initialized_model_sha256")
            )

    else:
        run_cfg["model"]["effective_initialization"] = (
            f"model_factory_encoder_weights={run_cfg['model'].get('encoder_weights')}"
        )

    run_cfg["run_provenance"] = {
        "resumed_from_local_last_checkpoint": bool(resumed),
        "optimizer_state_initialization": (
            "fresh"
            if not resumed
            else "restored_from_local_last_pt"
        ),
        "scheduler_state_initialization": (
            "fresh"
            if not resumed
            else "restored_from_local_last_pt"
        ),
        "amp_scaler_state_initialization": (
            "fresh"
            if not resumed
            else "restored_from_local_last_pt"
        ),
    }

    config_path = run_dir / "config.yaml"

    if config_path.exists():
        print(
            "[config] config.yaml already exists; "
            "keeping original run configuration."
        )
        return

    with open(config_path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(
            to_jsonable(run_cfg),
            fh,
            sort_keys=False,
            allow_unicode=True,
        )

    print(f"[config] resolved run configuration saved to {config_path}")
