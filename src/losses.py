"""Losses for binary crack segmentation.

Supported objectives:
- Weighted BCEWithLogits + Soft Dice.
- Focal BCE + Soft Dice.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftDiceLoss(nn.Module):
    """1 - Dice, computed per sample on sigmoid probabilities and then averaged."""

    def __init__(self, smooth: float = 1.0) -> None:
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        probability = torch.sigmoid(logits).flatten(1)
        target = target.flatten(1).float()

        intersection = (probability * target).sum(1)
        cardinality = probability.sum(1) + target.sum(1)

        dice = (2 * intersection + self.smooth) / (cardinality + self.smooth)

        return 1 - dice.mean()


class BceDiceLoss(nn.Module):
    """`bce_weight * BCEWithLogits + dice_weight * SoftDice`."""

    def __init__(
        self,
        bce_weight: float = 0.5,
        dice_weight: float = 0.5,
        pos_weight: float | None = None,
    ) -> None:
        super().__init__()

        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.dice = SoftDiceLoss()

        # Registered as a buffer so it follows the module to the GPU.
        self.register_buffer(
            "pos_weight",
            torch.tensor([pos_weight]) if pos_weight is not None else None,
        )

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        
        bce = F.binary_cross_entropy_with_logits(
            logits, target.float(), pos_weight=self.pos_weight
        )

        return self.bce_weight * bce + self.dice_weight * self.dice(logits, target)


class FocalBceLoss(nn.Module):
    """Numerically stable binary focal loss computed from logits.

    alpha is the positive-class weight:
      - alpha = 0.25: conventional RetinaNet starting point.
      - alpha = 0.50: equal positive/negative base weighting.

    gamma controls focal strength:
      - gamma = 0.0 gives alpha-balanced BCE.
      - gamma = 2.0 is the standard focal setting.
    """

    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        reduction: str = "mean",
    ) -> None:
        super().__init__()

        if not 0.0 < alpha < 1.0:
            raise ValueError("focal_alpha must be strictly between 0 and 1.")

        if gamma < 0.0:
            raise ValueError("focal_gamma must be non-negative.")

        if reduction not in {"mean", "sum", "none"}:
            raise ValueError("reduction must be 'mean', 'sum', or 'none'.")

        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:

        logits = logits.float()
        target = target.float()

        bce_per_pixel = F.binary_cross_entropy_with_logits(
            logits,
            target,
            reduction="none",
        )

        probability = torch.sigmoid(logits)
        p_t = probability * target + (1.0 - probability) * (1.0 - target)

        alpha_t = self.alpha * target + (1.0 - self.alpha) * (1.0 - target)
        focal_factor = (1.0 - p_t).pow(self.gamma)

        loss = alpha_t * focal_factor * bce_per_pixel

        if self.reduction == "mean":
            return loss.mean()

        if self.reduction == "sum":
            return loss.sum()

        return loss


class FocalDiceLoss(nn.Module):
    """focal_weight * Focal BCE + dice_weight * Soft Dice."""

    def __init__(
        self,
        focal_weight: float = 0.5,
        dice_weight: float = 0.5,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
    ) -> None:
        super().__init__()

        if focal_weight < 0.0 or dice_weight < 0.0:
            raise ValueError("Loss weights must be non-negative.")
        if focal_weight + dice_weight <= 0.0:
            raise ValueError("At least one loss weight must be positive.")

        self.focal_weight = float(focal_weight)
        self.dice_weight = float(dice_weight)

        self.focal = FocalBceLoss(
            alpha=focal_alpha,
            gamma=focal_gamma,
            reduction="mean",
        )
        self.dice = SoftDiceLoss()

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        focal = self.focal(logits, target)
        dice = self.dice(logits, target)

        return self.focal_weight * focal + self.dice_weight * dice

def build_loss(cfg) -> nn.Module:
    """Build the configured binary segmentation loss."""

    name = str(cfg.get("name", "bce_dice")).lower()
    print(f"[loss] selected: {name}")

    if name == "bce_dice":

        bce_weight = float(cfg.get("bce_weight", 0.5))
        dice_weight = float(cfg.get("dice_weight", 0.5))

        pos_weight = cfg.get("pos_weight")
        pos_weight = float(pos_weight) if pos_weight is not None else None

        print(
            f"[loss] bce_dice | bce_weight={bce_weight} | "
            f"dice_weight={dice_weight} | "
            f"pos_weight={pos_weight}"
        )

        return BceDiceLoss(
            bce_weight=bce_weight,
            dice_weight=dice_weight,
            pos_weight=pos_weight,
        )

    if name == "focal_dice":

        focal_weight = float(cfg.get("focal_weight", 0.5))
        dice_weight = float(cfg.get("dice_weight", 0.5))
        focal_alpha = float(cfg.get("focal_alpha", 0.25))
        focal_gamma = float(cfg.get("focal_gamma", 2.0))
        
        print(
            f"[loss] focal_dice | focal_weight={focal_weight} | "
            f"dice_weight={dice_weight} | "
            f"alpha={focal_alpha} | "
            f"gamma={focal_gamma}"
        )

        return FocalDiceLoss(
            focal_weight=focal_weight,
            dice_weight=dice_weight,
            focal_alpha=focal_alpha,
            focal_gamma=focal_gamma,
        )

    raise ValueError(
        f"Unknown loss.name='{name}'. Supported: 'bce_dice', 'focal_dice'."
    )


# --------------------------------------------------------------------------- #
# P1ML: multilabel (independent channels)
# --------------------------------------------------------------------------- #
class MultilabelSoftDiceLoss(nn.Module):
    """Soft Dice computed per (sample, channel) and averaged.

    The binary `SoftDiceLoss` flattens from dim 1 onwards, so with C>1 it would collapse the six channels into a single region and let `surface` (very frequent) dominate `delamination` (rare). Here the overlap is measured channel-wise, which is the multilabel formulation:

        L = 1 - (1/BC) * sum_{b,c} (2*|p_bc ∩ y_bc| + eps) / (|p_bc| + |y_bc| + eps)
    """

    def __init__(self, eps: float = 1.0) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probability = torch.sigmoid(logits.float())
        targets = targets.float()

        batch, channels = probability.shape[0], probability.shape[1]
        probability = probability.reshape(batch, channels, -1)
        targets = targets.reshape(batch, channels, -1)

        intersection = (probability * targets).sum(dim=2)
        cardinality = probability.sum(dim=2) + targets.sum(dim=2)
        dice = (2.0 * intersection + self.eps) / (cardinality + self.eps)

        return 1.0 - dice.mean()


class MultilabelBceDiceLoss(nn.Module):
    """Per-channel weighted BCEWithLogits + per-channel Soft Dice.

    `pos_weight` is a vector of length C (one value per unified damage class), estimated on the official DACL10K train split only. Sigmoid + BCE keeps the channels independent: no softmax, no CrossEntropy, so overlapping damages (e.g. corrosion inside a spalling area) remain representable.
    """

    def __init__(
        self,
        pos_weight: Sequence[float] | torch.Tensor,
        bce_weight: float = 0.5,
        dice_weight: float = 0.5,
    ) -> None:
        super().__init__()
        weight = torch.as_tensor(pos_weight, dtype=torch.float32).flatten()
        if weight.numel() < 1:
            raise ValueError("pos_weight must contain one value per channel.")

        # Shape [C,1,1] broadcasts against logits [B,C,H,W].
        self.register_buffer("pos_weight", weight.view(-1, 1, 1))
        self.bce_weight = float(bce_weight)
        self.dice_weight = float(dice_weight)
        self.dice = MultilabelSoftDiceLoss()

    @property
    def n_channels(self) -> int:
        return int(self.pos_weight.numel())

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if logits.shape[1] != self.n_channels:
            raise ValueError(
                f"Model emits {logits.shape[1]} channels but pos_weight has "
                f"{self.n_channels} entries."
            )
        if logits.shape != targets.shape:
            raise ValueError(f"Shape mismatch: logits {tuple(logits.shape)} vs targets {tuple(targets.shape)}.")

        bce = F.binary_cross_entropy_with_logits(
            logits.float(), targets.float(), pos_weight=self.pos_weight
        )
        return self.bce_weight * bce + self.dice_weight * self.dice(logits, targets)


def build_multilabel_loss(cfg, pos_weight: Sequence[float]) -> nn.Module:
    """Factory for the P1ML loss. `build_loss` is left untouched for P0-P3."""
    name = str(cfg.loss.name).lower()
    if name not in {"multilabel_bce_dice", "bce_dice"}:
        raise ValueError(
            f"Unsupported multilabel loss '{cfg.loss.name}'. P1ML supports "
            "'multilabel_bce_dice' only (focal variants are out of scope)."
        )

    return MultilabelBceDiceLoss(
        pos_weight=pos_weight,
        bce_weight=float(cfg.loss.bce_weight),
        dice_weight=float(cfg.loss.dice_weight),
    )
