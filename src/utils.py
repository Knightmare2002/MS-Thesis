"""Small shared helpers: config loading, reproducibility, device selection."""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import yaml


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
class Config(dict):
    """Dict with attribute access, so cfg.train.epochs works like cfg['train']['epochs']."""

    def __getattr__(self, name: str) -> Any:
        try:
            value = self[name]
        except KeyError as exc:  # pragma: no cover - defensive
            raise AttributeError(name) from exc
        return Config(value) if isinstance(value, dict) else value


def load_config(path: str | Path, overrides: list[str] | None = None) -> Config:
    """Load a YAML config and apply `key.subkey=value` CLI overrides.

    Values are parsed as JSON when possible (so `true`, `3`, `0.1`, `[1,2]` keep
    their type) and fall back to plain strings otherwise.
    """
    with open(path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    for override in overrides or []:
        key, _, raw = override.partition("=")
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            value = raw
        node = cfg
        *parents, leaf = key.split(".")
        for part in parents:
            node = node.setdefault(part, {})
        node[leaf] = value

    return Config(cfg)


# --------------------------------------------------------------------------- #
# Reproducibility / device
# --------------------------------------------------------------------------- #
def seed_everything(seed: int = 42) -> None:
    """Seed python, numpy and torch so runs are comparable across experiments."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:  # EDA scripts do not need torch
        pass


def get_device(verbose: bool = True) -> "torch.device":  # type: ignore[name-defined]
    """Return CUDA when available (with Ampere-friendly flags), else CPU.

    On the RTX A2000 (Ampere, GA106) two backend switches are free speed-ups:
    * TF32 matmul/conv kernels: ~same accuracy, noticeably faster than strict FP32;
    * `cudnn.benchmark`: the input size is fixed by `data.image_size`, so cuDNN can
      autotune once and cache the best convolution algorithms.
    """
    import torch

    if not torch.cuda.is_available():
        if verbose:
            print("[device] CUDA not available - falling back to CPU")
        return torch.device("cpu")

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    if verbose:
        properties = torch.cuda.get_device_properties(0)
        print(f"[device] {properties.name} | {properties.total_memory / 1024**3:.1f} GB VRAM")
    return torch.device("cuda")

def _seed_worker(worker_id: int) -> None:
    """Seed NumPy and Python random inside each DataLoader worker."""
    import torch

    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def loader_kwargs(data_cfg, device) -> dict:
    """DataLoader options shared by training and evaluation.

    `persistent_workers` and `prefetch_factor` are only valid when workers > 0,
    so they are added conditionally (passing them with num_workers=0 raises).
    """
    num_workers = int(data_cfg.get("num_workers", 0))
    kwargs = {"num_workers": num_workers, "pin_memory": device.type == "cuda"}
    if num_workers > 0:
        kwargs["persistent_workers"] = bool(data_cfg.get("persistent_workers", False))
        kwargs["prefetch_factor"] = int(data_cfg.get("prefetch_factor", 2))
        kwargs["worker_init_fn"] = _seed_worker
    return kwargs


def ensure_dir(path: str | Path) -> Path:
    """Create a directory (and parents) and return it as a Path."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path
