"""Sliding-window inference for native-resolution segmentation images."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def _sliding_positions(length: int, patch_size: int, stride: int) -> list[int]:
    """Return patch origins covering [0, length] with no uncovered border."""
    if length <= patch_size:
        return [0]

    positions = list(range(0, length - patch_size + 1, stride))
    last = length - patch_size
    if positions[-1] != last:
        positions.append(last)
    return positions


def _blend_window(
    patch_size: int,
    mode: str = "gaussian",
    minimum_weight: float = 1e-3,
) -> torch.Tensor:
    """Create [1, 1, P, P] overlap weights with non-zero border values."""
    if mode == "uniform":
        return torch.ones((1, 1, patch_size, patch_size), dtype=torch.float32)

    if mode != "gaussian":
        raise ValueError(f"Unknown blend mode '{mode}'. Expected 'uniform' or 'gaussian'.")

    coordinates = torch.arange(patch_size, dtype=torch.float32)
    center = (patch_size - 1) / 2.0
    sigma = patch_size / 4.0

    weights_1d = torch.exp(-0.5 * ((coordinates - center) / sigma) ** 2)
    weights_2d = torch.outer(weights_1d, weights_1d)
    weights_2d = weights_2d / weights_2d.max()
    weights_2d = weights_2d.clamp_min(minimum_weight)

    return weights_2d.unsqueeze(0).unsqueeze(0)


def _normalize_patch(
    patch: np.ndarray,
    mean: tuple[float, float, float],
    std: tuple[float, float, float],
) -> torch.Tensor:
    """Convert RGB uint8 [H,W,3] patch into normalized float [3,H,W]."""
    tensor = torch.from_numpy(patch.transpose(2, 0, 1)).float() / 255.0
    mean_tensor = torch.tensor(mean, dtype=torch.float32).view(3, 1, 1)
    std_tensor = torch.tensor(std, dtype=torch.float32).view(3, 1, 1)
    return (tensor - mean_tensor) / std_tensor


@torch.no_grad()
def predict_sliding_window(
    model,
    image: np.ndarray,
    device: torch.device,
    patch_size: int,
    stride: int,
    batch_size: int,
    mean: tuple[float, float, float],
    std: tuple[float, float, float],
    blend_mode: str = "gaussian",
) -> torch.Tensor:
    """Return a full-resolution crack-probability map [H,W].

    The input image is never globally resized. Patches are padded only when an image side is smaller than patch_size, inferred in batches, then fused using
    weighted averaging in all overlapping regions.
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected RGB image [H,W,3], got shape {image.shape}.")
    if patch_size <= 0 or stride <= 0 or batch_size <= 0:
        raise ValueError("patch_size, stride and batch_size must be positive.")

    original_height, original_width = image.shape[:2]
    pad_bottom = max(patch_size - original_height, 0)
    pad_right = max(patch_size - original_width, 0)

    if pad_bottom or pad_right:
        image = np.pad(
            image,
            ((0, pad_bottom), (0, pad_right), (0, 0)),
            mode="reflect",
        )

    height, width = image.shape[:2]
    top_positions = _sliding_positions(height, patch_size, stride)
    left_positions = _sliding_positions(width, patch_size, stride)

    coordinates = [(top, left) for top in top_positions for left in left_positions]
    blend = _blend_window(patch_size, blend_mode).to(device)

    probability_sum = torch.zeros((1, 1, height, width), dtype=torch.float32, device=device)
    weight_sum = torch.zeros((1, 1, height, width), dtype=torch.float32, device=device)

    model.eval()
    for start in range(0, len(coordinates), batch_size):
        batch_coordinates = coordinates[start : start + batch_size]
        patches = [
            _normalize_patch(
                image[top : top + patch_size, left : left + patch_size],
                mean=mean,
                std=std,
            )
            for top, left in batch_coordinates
        ]

        batch = torch.stack(patches, dim=0).to(device, non_blocking=True)
        probabilities = torch.sigmoid(model(batch))

        for probability, (top, left) in zip(probabilities, batch_coordinates):
            probability_sum[:, :, top : top + patch_size, left : left + patch_size] += (
                probability.unsqueeze(0) * blend
            )
            weight_sum[:, :, top : top + patch_size, left : left + patch_size] += blend

    probability_map = probability_sum / weight_sum.clamp_min(1e-8)
    return probability_map[0, 0, :original_height, :original_width].cpu()

@torch.no_grad()
def predict_sliding_window_multilabel(
    model,
    image: np.ndarray,
    device: torch.device,
    patch_size: int,
    stride: int,
    batch_size: int,
    mean: tuple[float, float, float],
    std: tuple[float, float, float],
    n_classes: int,
    blend_mode: str = "gaussian",
) -> torch.Tensor:
    """Return a full-resolution multilabel probability map [C,H,W].

    Same geometry, padding and Gaussian fusion as `predict_sliding_window`: only
    the accumulators carry C channels, and the sigmoid is applied per channel
    (independent classes, never softmax across them).
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected RGB image [H,W,3], got shape {image.shape}.")
    if patch_size <= 0 or stride <= 0 or batch_size <= 0:
        raise ValueError("patch_size, stride and batch_size must be positive.")
    if n_classes <= 0:
        raise ValueError("n_classes must be positive.")

    original_height, original_width = image.shape[:2]
    pad_bottom = max(patch_size - original_height, 0)
    pad_right = max(patch_size - original_width, 0)

    if pad_bottom or pad_right:
        image = np.pad(image, ((0, pad_bottom), (0, pad_right), (0, 0)), mode="reflect")

    height, width = image.shape[:2]
    top_positions = _sliding_positions(height, patch_size, stride)
    left_positions = _sliding_positions(width, patch_size, stride)

    coordinates = [(top, left) for top in top_positions for left in left_positions]
    blend = _blend_window(patch_size, blend_mode).to(device)

    probability_sum = torch.zeros(
        (1, n_classes, height, width), dtype=torch.float32, device=device
    )
    weight_sum = torch.zeros((1, 1, height, width), dtype=torch.float32, device=device)

    model.eval()
    for start in range(0, len(coordinates), batch_size):
        batch_coordinates = coordinates[start : start + batch_size]
        patches = [
            _normalize_patch(
                image[top : top + patch_size, left : left + patch_size],
                mean=mean,
                std=std,
            )
            for top, left in batch_coordinates
        ]

        batch = torch.stack(patches, dim=0).to(device, non_blocking=True)
        logits = model(batch)

        if logits.shape[1] != n_classes:
            raise ValueError(
                f"Model emits {logits.shape[1]} channels but n_classes={n_classes}."
            )

        probabilities = torch.sigmoid(logits.float())

        for probability, (top, left) in zip(probabilities, batch_coordinates):
            probability_sum[:, :, top : top + patch_size, left : left + patch_size] += (
                probability.unsqueeze(0) * blend
            )
            weight_sum[:, :, top : top + patch_size, left : left + patch_size] += blend

    probability_map = probability_sum / weight_sum.clamp_min(1e-8)
    return probability_map[0, :, :original_height, :original_width].cpu()


@torch.no_grad()
def predict_sliding_window_multitask(
    model,
    image: np.ndarray,
    device: torch.device,
    patch_size: int,
    stride: int,
    batch_size: int,
    mean: tuple[float, float, float],
    std: tuple[float, float, float],
    n_classes: int,
    blend_mode: str = "gaussian",
) -> tuple[torch.Tensor, torch.Tensor]:
    """P4-A: full-resolution maps of both heads with a single encoder pass per patch.

    Returns `(crack_probability [H,W], multilabel_probability [C,H,W])`.

    Geometry, padding, Gaussian weights and normalization are byte-identical to
    `predict_sliding_window` / `predict_sliding_window_multilabel`; the only
    difference is that the shared encoder is evaluated once per patch and the two
    decoders consume the same features, which is what makes the joint evaluation
    cost one forward pass instead of two. Running the two single-head functions
    separately on a multitask model (via `SingleHeadAdapter`) yields the same
    numbers, only slower.
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected RGB image [H,W,3], got shape {image.shape}.")
    if patch_size <= 0 or stride <= 0 or batch_size <= 0:
        raise ValueError("patch_size, stride and batch_size must be positive.")
    if n_classes <= 0:
        raise ValueError("n_classes must be positive.")

    original_height, original_width = image.shape[:2]
    pad_bottom = max(patch_size - original_height, 0)
    pad_right = max(patch_size - original_width, 0)

    if pad_bottom or pad_right:
        image = np.pad(image, ((0, pad_bottom), (0, pad_right), (0, 0)), mode="reflect")

    height, width = image.shape[:2]
    coordinates = [
        (top, left)
        for top in _sliding_positions(height, patch_size, stride)
        for left in _sliding_positions(width, patch_size, stride)
    ]
    blend = _blend_window(patch_size, blend_mode).to(device)

    crack_sum = torch.zeros((1, 1, height, width), dtype=torch.float32, device=device)
    multilabel_sum = torch.zeros((1, n_classes, height, width), dtype=torch.float32, device=device)
    weight_sum = torch.zeros((1, 1, height, width), dtype=torch.float32, device=device)

    model.eval()
    for start in range(0, len(coordinates), batch_size):
        batch_coordinates = coordinates[start : start + batch_size]
        patches = [
            _normalize_patch(
                image[top : top + patch_size, left : left + patch_size], mean=mean, std=std
            )
            for top, left in batch_coordinates
        ]

        batch = torch.stack(patches, dim=0).to(device, non_blocking=True)
        crack_logits, multilabel_logits = model(batch)

        if crack_logits.shape[1] != 1:
            raise ValueError(f"The crack head emits {crack_logits.shape[1]} channels, expected 1.")
        if multilabel_logits.shape[1] != n_classes:
            raise ValueError(
                f"The multilabel head emits {multilabel_logits.shape[1]} channels "
                f"but n_classes={n_classes}."
            )

        crack_probabilities = torch.sigmoid(crack_logits.float())
        multilabel_probabilities = torch.sigmoid(multilabel_logits.float())

        for index, (top, left) in enumerate(batch_coordinates):
            window = (slice(None), slice(None), slice(top, top + patch_size), slice(left, left + patch_size))
            crack_sum[window] += crack_probabilities[index].unsqueeze(0) * blend
            multilabel_sum[window] += multilabel_probabilities[index].unsqueeze(0) * blend
            weight_sum[window] += blend

    weights = weight_sum.clamp_min(1e-8)
    crack_map = (crack_sum / weights)[0, 0, :original_height, :original_width].cpu()
    multilabel_map = (multilabel_sum / weights)[0, :, :original_height, :original_width].cpu()
    return crack_map, multilabel_map
