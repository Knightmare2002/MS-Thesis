"""Training / validation loop.

Colab-oriented choices:
* mixed precision (AMP) to fit 512x512 batches on a T4;
* gradient accumulation to emulate a larger batch when VRAM is the limit;
* `last.pt` written every epoch so a killed session resumes instead of restarting;
* `best.pt` selected on validation Dice, plus early stopping;
* a per-epoch CSV history that feeds the learning curves in the slides.
"""

from __future__ import annotations

import csv
import time
from pathlib import Path

import torch
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, LambdaLR, LinearLR, SequentialLR
from torch.optim.swa_utils import AveragedModel
from tqdm.auto import tqdm

import hashlib
from typing import Any

from .eval.metrics import SegmentationMetrics
from .utils import ensure_dir


# --------------------------------------------------------------------------- #
# Single epoch
# --------------------------------------------------------------------------- #
def train_one_epoch(
    model, loader, criterion, optimizer, scaler, device, accumulation_steps: int = 1, ema=None
) -> float:
    """Run one training epoch and return the mean loss."""
    model.train()
    total_loss, n_batches = 0.0, 0
    optimizer.zero_grad(set_to_none=True)

    for step, (images, masks) in enumerate(tqdm(loader, desc="train", leave=False)):
        images, masks = images.to(device, non_blocking=True), masks.to(device, non_blocking=True)

        with torch.autocast(device_type=device.type, enabled=scaler.is_enabled()):
            logits = model(images)
            # Scale the loss so accumulated gradients match a single large batch.
            loss = criterion(logits, masks) / accumulation_steps

        scaler.scale(loss).backward()

        if (step + 1) % accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            if ema is not None:
                ema.update_parameters(model)

        total_loss += loss.item() * accumulation_steps
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(model, loader, criterion, device, threshold: float = 0.5) -> dict[str, float]:
    """Evaluate on a loader and return {loss, iou, dice, precision, recall, ...}."""
    model.eval()
    meter = SegmentationMetrics(threshold=threshold)
    total_loss, n_batches = 0.0, 0

    for images, masks in tqdm(loader, desc="eval", leave=False):
        images, masks = images.to(device, non_blocking=True), masks.to(device, non_blocking=True)
        logits = model(images)
        total_loss += criterion(logits, masks).item()
        n_batches += 1
        meter.update(logits, masks)

    metrics = meter.compute()
    metrics["loss"] = total_loss / max(n_batches, 1)
    return metrics


# --------------------------------------------------------------------------- #
# Checkpointing
# --------------------------------------------------------------------------- #
def save_checkpoint(path: Path, model, optimizer, scheduler, scaler, epoch: int, best_dice: float, epochs_without_improvement: int = 0, ema=None) -> None:
    """Persist everything needed to resume training bit-for-bit.

    With EMA, "model" holds the EMA weights (the ones validated and used by every
    downstream script) and "model_raw" the optimised weights needed to resume.
    """
    extra = {}
    if ema is not None:
        extra = {"model_raw": model.state_dict(), "ema_n_averaged": int(ema.n_averaged.item()),
                 "weights": "ema"}
    torch.save(
        {
            **extra,
            "epoch": epoch,
            "best_dice": best_dice,
            "model": (ema.module if ema is not None else model).state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "epochs_without_improvement": epochs_without_improvement,
        },
        path,
    )


def load_checkpoint(path: Path, model, optimizer=None, scheduler=None, scaler=None, device="cpu") -> dict:
    """Restore a checkpoint; optimizer/scheduler/scaler are optional (inference)."""
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler is not None:
        scheduler.load_state_dict(checkpoint["scheduler"])
    if scaler is not None:
        scaler.load_state_dict(checkpoint["scaler"])
    return checkpoint


def state_dict_sha256(state_dict: dict[str, torch.Tensor]) -> str:
    """Return a deterministic SHA-256 fingerprint of model parameters."""
    digest = hashlib.sha256()

    for key in sorted(state_dict):
        tensor = state_dict[key].detach().cpu().contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("utf-8"))
        digest.update(tensor.numpy().tobytes())

    return digest.hexdigest()


def initialize_model_from_checkpoint(
    model,
    checkpoint_path: str | Path,
    device: torch.device | str = "cpu",
    strict: bool = True,
) -> dict[str, Any]:
    """
    Initialize only model weights from an external checkpoint.

    This is intentionally different from `load_checkpoint()`:
    it does not restore epoch, optimizer, scheduler or AMP scaler state.
    Use it for a new sequential-transfer experiment, e.g.
    CrackSeg9K -> DACL10K.
    """
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"[transfer] source checkpoint not found: {checkpoint_path.resolve()}"
        )

    checkpoint = torch.load(checkpoint_path, map_location=device)

    if not isinstance(checkpoint, dict):
        raise TypeError(
            "[transfer] unsupported checkpoint format: expected a dictionary."
        )

    if "model" not in checkpoint:
        available = ", ".join(sorted(checkpoint.keys()))
        raise KeyError(
            "[transfer] checkpoint does not contain key 'model'. "
            f"Available keys: {available}"
        )

    source_state_dict = checkpoint["model"]

    if not isinstance(source_state_dict, dict):
        raise TypeError(
            "[transfer] checkpoint['model'] must be a PyTorch state_dict."
        )

    # Model fingerprint, BEFORE the transfer.
    factory_sha256 = state_dict_sha256(model.state_dict())

    # Carica una sola volta i pesi P0 nel modello target.
    incompatible = model.load_state_dict(source_state_dict, strict=strict)

    if strict and (
        incompatible.missing_keys or incompatible.unexpected_keys
    ):
        raise RuntimeError(
            "[transfer] strict model initialization failed. "
            f"Missing keys: {incompatible.missing_keys}; "
            f"Unexpected keys: {incompatible.unexpected_keys}"
        )

    initialized_model_sha256 = state_dict_sha256(model.state_dict())
    source_model_sha256 = state_dict_sha256(source_state_dict)

    if initialized_model_sha256 == factory_sha256:
        raise RuntimeError(
            "[transfer] model weights are unchanged after loading: "
            "the checkpoint did not overwrite the factory initialization."
        )

    if strict and source_model_sha256 != initialized_model_sha256:
        raise RuntimeError(
            "[transfer] loaded model fingerprint differs from the source state_dict "
            "under strict loading."
        )

    metadata = {
        "checkpoint_path": str(checkpoint_path.resolve()),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_best_dice": checkpoint.get("best_dice"),
        "checkpoint_keys": sorted(checkpoint.keys()),
        "strict": bool(strict),
        "missing_keys": list(incompatible.missing_keys),
        "unexpected_keys": list(incompatible.unexpected_keys),
        "factory_model_sha256": factory_sha256,
        "source_model_sha256": source_model_sha256,
        "initialized_model_sha256": initialized_model_sha256,
    }

    if metadata["initialized_model_sha256"] == factory_sha256:
        raise RuntimeError(
            "[transfer] model weights are unchanged after loading: "
            "the checkpoint did not overwrite the factory initialization."
        )

    if strict and metadata["source_model_sha256"] != metadata["initialized_model_sha256"]:
        raise RuntimeError(
            "[transfer] loaded model fingerprint differs from the source state_dict "
            "under strict loading."
        )

    print(
        "[transfer] initialized model weights from "
        f"{metadata['checkpoint_path']}"
    )
    print(
        "[transfer] source epoch="
        f"{metadata['checkpoint_epoch']} | "
        f"source best_dice={metadata['checkpoint_best_dice']}"
    )
    print(
        "[transfer] SHA-256="
        f"{metadata['initialized_model_sha256']}"
    )

    return metadata

# --------------------------------------------------------------------------- #
# Full training run
# --------------------------------------------------------------------------- #

# --------------------------------------------------------------------------- #
# Optimizer / scheduler / EMA factories (T12 follow-ups; defaults = original)
# --------------------------------------------------------------------------- #
def convnext_layer_id(name: str, n_layers: int = 12) -> int:
    """Layer index of a timm ConvNeXt parameter (official ConvNeXt layer-decay rule).

    stem -> 0; stage 0/1 -> 1/2; stage 2 -> 3 + block // 3; stage 3 -> n_layers;
    downsample of stage k (inside `stages_k` in timm) -> like the official
    `downsample_layers.k`. Non-encoder parameters (decoder, head) -> n_layers + 1.
    """
    if not name.startswith("encoder."):
        return n_layers + 1
    parts = name.split(".")
    stem = next((p for p in parts if p.startswith("stem")), None)
    stage = next((p for p in parts if p.startswith("stages_") or p == "stages"), None)
    if stem is not None:
        return 0
    if stage is None:
        return n_layers  # e.g. final norm
    idx = parts.index(stage)
    stage_id = int(stage.split("_")[1]) if "_" in stage else int(parts[idx + 1])
    tail = parts[idx + 1:] if "_" in stage else parts[idx + 2:]
    if tail and tail[0] == "downsample":
        return {0: 0, 1: 2, 2: 3}.get(stage_id, n_layers)
    block_id = int(tail[1]) if len(tail) > 1 and tail[0] == "blocks" else 0
    if stage_id in (0, 1):
        return stage_id + 1
    if stage_id == 2:
        return 3 + block_id // 3
    return n_layers


def build_optimizer(model, cfg) -> torch.optim.Optimizer:
    """AdamW with optional layer-wise LR decay, head LR multiplier and no-WD on 1-D params.

    `train.optimizer` (all optional; absent -> one group, identical to the
    original `AdamW(model.parameters(), lr, weight_decay)`):
      layer_decay   : float, e.g. 0.9 (ConvNeXt encoders only)
      head_lr_mult  : float, LR multiplier of `segmentation_head.*`
      no_weight_decay_1d : bool, no weight decay on biases / norms / layer-scale
    """
    lr, wd = float(cfg.train.lr), float(cfg.train.weight_decay)
    ocfg = dict(cfg.train.get("optimizer") or {})
    layer_decay = float(ocfg.get("layer_decay", 1.0))
    head_mult = float(ocfg.get("head_lr_mult", 1.0))
    no_wd_1d = bool(ocfg.get("no_weight_decay_1d", False))
    if layer_decay == 1.0 and head_mult == 1.0 and not no_wd_1d:
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    if layer_decay != 1.0 and "convnext" not in str(cfg.model.encoder).lower():
        raise ValueError("train.optimizer.layer_decay is implemented for ConvNeXt encoders only.")

    n_layers = 12
    groups: dict[tuple[float, float], list] = {}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        scale = layer_decay ** (n_layers + 1 - convnext_layer_id(name, n_layers))
        if name.startswith("segmentation_head."):
            scale *= head_mult
        decay = 0.0 if (no_wd_1d and param.ndim <= 1) else wd
        groups.setdefault((round(scale, 8), decay), []).append(param)
    param_groups = [{"params": ps, "lr": lr * sc, "weight_decay": d, "lr_scale": sc}
                    for (sc, d), ps in sorted(groups.items())]
    scales = sorted({g["lr_scale"] for g in param_groups})
    print(f"[optimizer] AdamW, {len(param_groups)} groups, lr scale {scales[0]:.3f}..{scales[-1]:.3f}, "
          f"wd {wd} (1-D params {'0' if no_wd_1d else wd})")
    return torch.optim.AdamW(param_groups, lr=lr, weight_decay=wd)


def build_scheduler(optimizer, cfg):
    """`train.scheduler.name`: "cosine_restarts" (default, original SGDR) or "constant"."""
    sch = cfg.train.scheduler
    name = str(sch.get("name", "cosine_restarts"))
    if name == "constant":
        main = LambdaLR(optimizer, lr_lambda=lambda epoch: 1.0)
    elif name == "cosine_restarts":
        main = CosineAnnealingWarmRestarts(optimizer, T_0=sch.t_0, T_mult=sch.t_mult, eta_min=sch.eta_min)
    else:
        raise ValueError(f"Unknown train.scheduler.name '{name}'.")

    warmup_epochs = int(cfg.train.get("warmup_epochs", 0))
    warmup_start_factor = float(cfg.train.get("warmup_start_factor", 1.0))
    if warmup_epochs <= 0:
        return main
    if not 0.0 < warmup_start_factor <= 1.0:
        raise ValueError("train.warmup_start_factor must be in (0, 1].")
    warmup = LinearLR(optimizer, start_factor=warmup_start_factor, end_factor=1.0, total_iters=warmup_epochs)
    print(f"[scheduler] linear warm-up for {warmup_epochs} epochs: "
          f"{warmup_start_factor:.3f} × lr -> 1.000 × lr, then {name}")
    return SequentialLR(optimizer, schedulers=[warmup, main], milestones=[warmup_epochs])


def build_ema(model, cfg):
    """`train.ema: {enabled, decay, warmup}` -> AveragedModel (EMA of params and buffers) or None.

    Effective decay min(decay, (1 + n) / (warmup + n)), n = optimizer steps, so the
    average is not dominated by the initial weights in short fine-tunings.
    """
    ecfg = dict(cfg.train.get("ema") or {})
    if not bool(ecfg.get("enabled", False)):
        return None
    decay, warmup = float(ecfg.get("decay", 0.999)), float(ecfg.get("warmup", 10.0))
    if not 0.0 < decay < 1.0:
        raise ValueError("train.ema.decay must be in (0, 1).")

    def ema_avg(averaged, current, num_averaged):
        n = float(num_averaged)
        d = min(decay, (1.0 + n) / (warmup + n))
        return averaged * d + current * (1.0 - d)

    print(f"[ema] enabled: decay {decay}, warm-up {warmup}; validation and best.pt use EMA weights")
    return AveragedModel(model, avg_fn=ema_avg, use_buffers=True)


def fit(
    model,
    train_loader,
    val_loader,
    criterion,
    cfg,
    device,
    output_dir: Path,
    evaluate_fn=None,
    selection_metric: str = "dice",
    extra_history_keys: tuple[str, ...] = (),
) -> dict:
    """Train with SGDR + early stopping. Returns the best validation metrics.

    `evaluate_fn`, `selection_metric` and `extra_history_keys` default to the
    binary behaviour used by P0-P3. The multilabel pipeline (P1ML) passes
    `evaluate_multilabel` and selects on `macro_dice_present`.
    """
    evaluate_fn = evaluate_fn or evaluate
    output_dir = ensure_dir(output_dir)
    last_path, best_path = output_dir / "last.pt", output_dir / "best.pt"
    history_path = output_dir / "history.csv"

    # The model must be on the target device before constructing/loading AdamW.
    # Otherwise Adam moment tensors restored from a checkpoint can remain on CPU
    model.to(device)

    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)
    ema = build_ema(model, cfg)

    scaler = torch.amp.GradScaler(enabled=bool(cfg.train.amp) and device.type == "cuda")

    start_epoch, best_dice, epochs_without_improvement = 0, -1.0, 0
    if cfg.train.get("resume") and last_path.exists():
        checkpoint = load_checkpoint(last_path, model, optimizer, scheduler, scaler, device)
        if ema is not None:
            if "model_raw" not in checkpoint:
                raise RuntimeError(f"{last_path} has no EMA state: cannot resume with train.ema.enabled=true.")
            ema.module.load_state_dict(checkpoint["model"])
            model.load_state_dict(checkpoint["model_raw"])
            ema.n_averaged.fill_(int(checkpoint.get("ema_n_averaged", 0)))
        start_epoch, best_dice = checkpoint["epoch"] + 1, checkpoint["best_dice"]
        epochs_without_improvement = checkpoint.get("epochs_without_improvement", 0)
        print(f"[fit] resumed from {last_path} at epoch {start_epoch} (best dice {best_dice:.4f})")

    best_metrics: dict[str, float] = {}

    for epoch in range(start_epoch, cfg.train.epochs):
        started = time.time()
        current_lr = optimizer.param_groups[0]["lr"]
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device,
            accumulation_steps=cfg.train.accumulation_steps, ema=ema,
        )
        eval_model = ema.module if ema is not None else model
        val_metrics = evaluate_fn(eval_model, val_loader, criterion, device, threshold=cfg.eval.threshold)
        scheduler.step()  # per-epoch stepping matches T_0 expressed in epochs

        row = {
            "epoch": epoch,
            "lr": current_lr,
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "val_iou": val_metrics["iou"],
            "val_dice": val_metrics["dice"],
            "val_precision": val_metrics["precision"],
            "val_recall": val_metrics["recall"],
        }
        for key in extra_history_keys:
            row[f"val_{key}"] = val_metrics[key]
        row["seconds"] = round(time.time() - started, 1)
        _append_csv(history_path, row)
        print(
            f"epoch {epoch:03d} | train {train_loss:.4f} | val {val_metrics['loss']:.4f} "
            f"| IoU {val_metrics['iou']:.4f} | Dice {val_metrics['dice']:.4f} "
            f"| P {val_metrics['precision']:.3f} R {val_metrics['recall']:.3f}"
        )
       
        if selection_metric not in val_metrics:
            raise KeyError(
                f"selection_metric '{selection_metric}' is not returned by the "
                f"evaluation function. Available: {sorted(val_metrics)}"
            )

        improved = val_metrics[selection_metric] > best_dice
        if improved:
            best_dice, best_metrics, epochs_without_improvement = (
                val_metrics[selection_metric],
                val_metrics,
                0,
            )
        else:
            epochs_without_improvement += 1

        save_checkpoint(last_path, model, optimizer, scheduler, scaler, epoch, best_dice,
                        epochs_without_improvement, ema=ema)
        if improved:
            save_checkpoint(best_path, model, optimizer, scheduler, scaler, epoch, best_dice,
                            epochs_without_improvement, ema=ema)
            print(f"  -> new best {selection_metric} {best_dice:.4f}, saved {best_path.name}")
        elif epochs_without_improvement >= cfg.train.early_stopping_patience:
            print(f"[fit] early stopping after {epochs_without_improvement} epochs without improvement")
            break

    return best_metrics


def _append_csv(path: Path, row: dict) -> None:
    """Append one row to a CSV, writing the header on first use.

    If the columns changed (e.g. a run resumed after new metrics were added),
    the file is rewritten with the union of the columns (missing cells empty).
    """
    if path.exists():
        with open(path, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            old_fields, old_rows = list(reader.fieldnames or []), list(reader)
        if old_fields != list(row):
            fields = old_fields + [k for k in row if k not in old_fields]
            with open(path, "w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=fields)
                writer.writeheader()
                writer.writerows(old_rows + [row])
            return
    write_header = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(row))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


# --------------------------------------------------------------------------- #
# P1ML: task transfer with an incompatible segmentation head
# --------------------------------------------------------------------------- #
def initialize_model_from_checkpoint_partial(
    model,
    checkpoint_path: str | Path,
    device: torch.device | str = "cpu",
    allowed_missing_prefixes: tuple[str, ...] = ("segmentation_head.",),
    required_prefixes: tuple[str, ...] = ("encoder.", "decoder."),
) -> dict[str, Any]:
    """Initialize encoder + decoder from a checkpoint whose head has a different shape.

    Used for P1-B4 (binary, 1 output channel) -> P1ML (multilabel, 6 channels):
    the shared representation is transferred, the head is re-initialised from
    scratch. Contract, deliberately strict:

    * only tensors under `allowed_missing_prefixes` may be skipped, and only when
      their shape is incompatible; any other omission raises;
    * every parameter under `required_prefixes` must be transferred, otherwise a
      silent partial transfer would masquerade as a transfer experiment;
    * no optimizer / scheduler / scaler / epoch / early-stopping state is read.

    Returns a JSON-serialisable report (also written to the run dir by the
    training script).
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"[transfer] source checkpoint not found: {checkpoint_path.resolve()}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    if not isinstance(checkpoint, dict) or "model" not in checkpoint:
        raise KeyError("[transfer] checkpoint must be a dict containing key 'model'.")

    source_state_dict = checkpoint["model"]
    target_state_dict = model.state_dict()
    factory_sha256 = state_dict_sha256(target_state_dict)

    transferred, skipped_shape_mismatch, skipped_absent_in_target = [], [], []

    filtered: dict[str, torch.Tensor] = {}
    for key, tensor in source_state_dict.items():
        if key not in target_state_dict:
            skipped_absent_in_target.append(key)
            continue
        if tuple(tensor.shape) != tuple(target_state_dict[key].shape):
            skipped_shape_mismatch.append(
                {
                    "key": key,
                    "source_shape": list(tensor.shape),
                    "target_shape": list(target_state_dict[key].shape),
                }
            )
            continue
        filtered[key] = tensor
        transferred.append(key)

    def _is_allowed(key: str) -> bool:
        return any(key.startswith(prefix) for prefix in allowed_missing_prefixes)

    unexpected_mismatch = [
        entry["key"] for entry in skipped_shape_mismatch if not _is_allowed(entry["key"])
    ]
    if unexpected_mismatch:
        raise RuntimeError(
            "[transfer] shape mismatch outside the replaceable head: "
            f"{unexpected_mismatch}"
        )

    unexpected_absent = [key for key in skipped_absent_in_target if not _is_allowed(key)]
    if unexpected_absent:
        raise RuntimeError(
            f"[transfer] checkpoint keys absent from the target model: {unexpected_absent}"
        )

    incompatible = model.load_state_dict(filtered, strict=False)

    not_initialized = [
        key for key in incompatible.missing_keys if not _is_allowed(key)
    ]
    if not_initialized:
        raise RuntimeError(
            f"[transfer] parameters left at factory initialization: {not_initialized}"
        )
    if incompatible.unexpected_keys:
        raise RuntimeError(
            f"[transfer] unexpected keys after filtering: {incompatible.unexpected_keys}"
        )

    # Every encoder/decoder parameter must have been overwritten.
    for prefix in required_prefixes:
        required = [key for key in target_state_dict if key.startswith(prefix)]
        missing = sorted(set(required) - set(transferred))
        if not required:
            raise RuntimeError(f"[transfer] target model has no parameter under '{prefix}'.")
        if missing:
            raise RuntimeError(
                f"[transfer] incomplete transfer for '{prefix}': {len(missing)} tensors "
                f"not loaded, e.g. {missing[:5]}"
            )

    initialized_sha256 = state_dict_sha256(model.state_dict())
    if initialized_sha256 == factory_sha256:
        raise RuntimeError(
            "[transfer] model weights unchanged after the partial load: nothing was transferred."
        )

    report = {
        "mode": "partial_encoder_decoder_head_reinit",
        "checkpoint_path": str(checkpoint_path.resolve()),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_best_metric": checkpoint.get("best_dice"),
        "n_source_tensors": len(source_state_dict),
        "n_target_tensors": len(target_state_dict),
        "n_transferred_tensors": len(transferred),
        "reinitialized_head_tensors": [entry["key"] for entry in skipped_shape_mismatch],
        "head_shape_changes": skipped_shape_mismatch,
        "skipped_absent_in_target": skipped_absent_in_target,
        "allowed_missing_prefixes": list(allowed_missing_prefixes),
        "required_prefixes": list(required_prefixes),
        "factory_model_sha256": factory_sha256,
        "initialized_model_sha256": initialized_sha256,
        "restored_optimizer_state": False,
        "restored_scheduler_state": False,
        "restored_scaler_state": False,
        "restored_epoch": False,
        "restored_early_stopping_state": False,
    }

    print(
        f"[transfer] partial load from {checkpoint_path.name}: "
        f"{len(transferred)}/{len(target_state_dict)} tensors transferred | "
        f"head re-initialised: {report['reinitialized_head_tensors']}"
    )
    print(f"[transfer] SHA-256={initialized_sha256}")

    return report
