"""Losses for binary crack segmentation.

Supported objectives:
- Weighted BCEWithLogits + Soft Dice.
- Focal BCE + Soft Dice.
"""

from __future__ import annotations

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
