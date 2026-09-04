"""Loss for the binary crack task: BCE + soft Dice.

Why a combination
-----------------
Crack pixels are typically 1-5% of an image. Pure BCE converges to an almost
empty prediction (high pixel accuracy, useless recall). Pure Dice is directly
tied to the evaluation metric but its gradients are noisy when the ground truth
is nearly empty. The standard remedy in crack/defect literature is the weighted
sum used here, with an optional `pos_weight` inside the BCE term.
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
            torch.tensor([pos_weight]) if pos_weight else None,
        )

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(
            logits, target.float(), pos_weight=self.pos_weight
        )
        return self.bce_weight * bce + self.dice_weight * self.dice(logits, target)


def build_loss(cfg) -> nn.Module:
    """Instantiate the loss from the `loss` section of the config."""
    return BceDiceLoss(
        bce_weight=cfg.bce_weight,
        dice_weight=cfg.dice_weight,
        pos_weight=cfg.get("pos_weight"),
    )
