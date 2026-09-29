"""Model factory built on segmentation_models_pytorch (smp).

Only three architectures are exposed on purpose: the week-3 baseline must be a
*reference point*, not an architecture search. The encoder is ImageNet
pretrained and is the same component that will later be shared by the two heads
of the dual-branch network, so the encoder name is a config field, not a
hard-coded string.
"""

from __future__ import annotations

import segmentation_models_pytorch as smp
import torch.nn as nn

ARCHITECTURES = {
    "unet": smp.Unet,
    "unetplusplus": smp.UnetPlusPlus,
    "deeplabv3plus": smp.DeepLabV3Plus,
}


def build_model(cfg) -> nn.Module:
    """Instantiate a segmentation model from the `model` section of the config."""
    arch = str(cfg.arch).lower()
    if arch not in ARCHITECTURES:
        raise KeyError(f"Unknown arch '{arch}'. Available: {sorted(ARCHITECTURES)}")

    kwargs = dict(
        encoder_name=cfg.encoder,
        encoder_weights=cfg.get("encoder_weights"),  # None -> random init
        in_channels=cfg.in_channels,
        classes=cfg.classes,
    )

    # T5: optional scSE attention in the decoder blocks (smp Unet / UnetPlusPlus only).
    attention = cfg.get("decoder_attention_type")
    if attention:
        if arch not in ("unet", "unetplusplus"):
            raise ValueError(f"decoder_attention_type is not supported by arch '{arch}'.")
        kwargs["decoder_attention_type"] = str(attention)

    return ARCHITECTURES[arch](**kwargs)


def count_parameters(model: nn.Module) -> tuple[int, int]:
    """Return (total, trainable) parameter counts."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
