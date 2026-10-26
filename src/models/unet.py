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
    # T12: fully pretrained checkpoint from smp-hub (e.g. UPerNet-ConvNeXt on ADE20K);
    # only the segmentation head is re-initialised to cfg.classes channels.
    hub = cfg.get("hub_checkpoint")
    if hub:
        model = smp.from_pretrained(str(hub), classes=int(cfg.classes), strict=False)
        return set_drop_path(model, float(cfg.get("drop_path_rate", 0.0) or 0.0))
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

    return set_drop_path(ARCHITECTURES[arch](**kwargs), float(cfg.get("drop_path_rate", 0.0) or 0.0))


def set_drop_path(model: nn.Module, rate: float) -> nn.Module:
    """Stochastic depth on the residual blocks of a timm encoder (e.g. ConvNeXt).

    Every module owning a `drop_path` attribute (timm ConvNeXt/Swin/ViT blocks) gets
    `DropPath(rate * i / (N - 1))`, i = block index in forward order: the linear
    schedule of the official recipes. Active only in train() mode, so inference and
    checkpoints are unaffected. rate = 0 leaves the model untouched.
    """
    if rate <= 0.0:
        return model
    if rate >= 1.0:
        raise ValueError("model.drop_path_rate must be in [0, 1).")
    from timm.layers import DropPath

    blocks = [m for m in model.encoder.modules() if hasattr(m, "drop_path") and isinstance(m.drop_path, nn.Module)]
    if not blocks:
        raise ValueError(f"model.drop_path_rate={rate}: the encoder has no timm residual blocks with drop_path.")
    for i, block in enumerate(blocks):
        block.drop_path = DropPath(rate * i / max(len(blocks) - 1, 1))
    print(f"[model] drop_path linear 0 -> {rate} on {len(blocks)} encoder blocks")
    return model


def count_parameters(model: nn.Module) -> tuple[int, int]:
    """Return (total, trainable) parameter counts."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
