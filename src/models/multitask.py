"""P4-A: dual-head segmentation network with a genuinely shared encoder.

Scientific contract
-------------------
P4-A answers one question only: *does a shared representation help the binary
crack task and the 6-channel multilabel damage task at the same time?* To make
the answer attributable to the architecture and to nothing else, everything that
is not the architecture is frozen to the P1ML-A / P1-B4 protocol (official
DACL10K split, native 512 patches, identical transforms, optimizer, schedule and
sliding-window evaluation).

Architecture
------------

    x --> encoder (ResNet-34, ImageNet, SHARED, one forward pass)
             |--> crack_decoder      --> crack_head      --> [B,1,H,W]
             |--> multilabel_decoder --> multilabel_head --> [B,6,H,W]

* the encoder is a *single* module instance: its parameters receive gradients
  from both losses, which is the object under study;
* the two decoders are separate (allowed by the P4-A specification): a single
  decoder shared by a 1-channel and a 6-channel head would confound "shared
  encoder" with "shared decoder", and the crack channel would then be an
  implicit sub-problem of the multilabel decoder;
* the multilabel head emits independent logits: the sigmoid lives in the loss and
  in the metrics, never a softmax/argmax across channels, because damages
  overlap (corrosion inside a spalling area is not a labelling error);
* `forward` runs the encoder **once** and feeds both decoders, so the shared
  trunk costs exactly one forward/backward pass per batch. Two independent
  U-Net++ networks would cost two, and would not be P4-A.

Why not `aux_params` or a 7-channel head
----------------------------------------
`smp`'s `aux_params` adds a *classification* head, not a second dense decoder. A
single 7-channel head would tie crack and multilabel through the same 1x1
convolution and the same decoder, making the crack task numerically identical to
channel 0 of the multilabel task: there would be no second task to speak of.

Backward compatibility
----------------------
This module is additive. `src/models/unet.py:build_model` is untouched, so every
P0/P1/P2/P3/P1ML run keeps building exactly the same single-head network.
"""

from __future__ import annotations

import segmentation_models_pytorch as smp
import torch
import torch.nn as nn

from ..data.class_mapping import UNIFIED_DAMAGE_CLASSES

# Only the architectures whose smp implementation exposes the canonical
# encoder / decoder / segmentation_head triplet are accepted here.
MULTITASK_ARCHITECTURES = {
    "unetplusplus": smp.UnetPlusPlus,
    "unet": smp.Unet,
}

CRACK_HEAD = "crack"
MULTILABEL_HEAD = "multilabel"


class MultiTaskSegmentationModel(nn.Module):
    """Shared-encoder, two-decoder segmentation network for P4-A.

    Parameters
    ----------
    arch, encoder, encoder_weights, in_channels
        Same meaning as in `build_model`; the encoder is instantiated once.
    crack_classes
        Output channels of the binary crack head (1).
    multilabel_classes
        Output channels of the multilabel damage head (6).

    Notes
    -----
    The two decoders are taken from two temporary `smp` models built with the
    *same* encoder name, hence with identical `encoder.out_channels`; only the
    first model's encoder is retained and the second one is dropped immediately,
    so no duplicated trunk survives in the state dict. The resulting parameter
    namespace is explicit and stable:

        encoder.*  crack_decoder.*  crack_head.*  multilabel_decoder.*  multilabel_head.*
    """

    def __init__(
        self,
        arch: str = "unetplusplus",
        encoder: str = "resnet34",
        encoder_weights: str | None = "imagenet",
        in_channels: int = 3,
        crack_classes: int = 1,
        multilabel_classes: int = len(UNIFIED_DAMAGE_CLASSES),
    ) -> None:
        super().__init__()

        arch_key = str(arch).lower()
        if arch_key not in MULTITASK_ARCHITECTURES:
            raise KeyError(
                f"Unknown multitask arch '{arch}'. Available: {sorted(MULTITASK_ARCHITECTURES)}"
            )
        if crack_classes != 1:
            raise ValueError(f"The crack head must emit 1 logit per pixel, got {crack_classes}.")
        if multilabel_classes < 2:
            raise ValueError("The multilabel head must emit at least 2 channels.")

        factory = MULTITASK_ARCHITECTURES[arch_key]

        # Branch 1: keeps its encoder (the shared trunk, ImageNet initialised).
        crack_branch = factory(
            encoder_name=encoder,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=crack_classes,
        )
        # Branch 2: built only to obtain a decoder + head with the right widths.
        # `encoder_weights=None` avoids downloading/initialising a trunk we drop.
        multilabel_branch = factory(
            encoder_name=encoder,
            encoder_weights=None,
            in_channels=in_channels,
            classes=multilabel_classes,
        )

        self.arch = arch_key
        self.encoder_name = str(encoder)
        self.encoder = crack_branch.encoder

        self.crack_decoder = crack_branch.decoder
        self.crack_head = crack_branch.segmentation_head

        self.multilabel_decoder = multilabel_branch.decoder
        self.multilabel_head = multilabel_branch.segmentation_head

        # Drop every reference to the throw-away trunk before it can be
        # registered as a submodule of `self`.
        del multilabel_branch.encoder
        del multilabel_branch, crack_branch

        self.crack_classes = int(crack_classes)
        self.multilabel_classes = int(multilabel_classes)
        self.output_stride = int(getattr(self.encoder, "output_stride", 32))

    # ------------------------------------------------------------------ #
    # Forward paths
    # ------------------------------------------------------------------ #
    def _check_input(self, x: torch.Tensor) -> None:
        if x.ndim != 4:
            raise ValueError(f"Expected a [B,C,H,W] batch, got shape {tuple(x.shape)}.")
        height, width = x.shape[-2:]
        stride = self.output_stride
        if height % stride or width % stride:
            raise ValueError(
                f"Input {height}x{width} is not divisible by the encoder output "
                f"stride {stride}: pad the input (the 512 patch geometry of "
                "P1-B4 / P1ML satisfies this by construction)."
            )

    def encode(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Run the shared encoder once and return its feature pyramid."""
        self._check_input(x)
        return self.encoder(x)

    def decode_crack(self, features: list[torch.Tensor]) -> torch.Tensor:
        return self.crack_head(self.crack_decoder(features))

    def decode_multilabel(self, features: list[torch.Tensor]) -> torch.Tensor:
        return self.multilabel_head(self.multilabel_decoder(features))

    #==== This 2 methods are the ones used by the SingleHeadAdapter to expose one head only. ====#
    def forward_crack(self, x: torch.Tensor) -> torch.Tensor:
        """Binary crack logits [B,1,H,W] (no sigmoid)."""
        return self.decode_crack(self.encode(x))

    def forward_multilabel(self, x: torch.Tensor) -> torch.Tensor:
        """Multilabel damage logits [B,6,H,W] (no sigmoid, never softmax)."""
        return self.decode_multilabel(self.encode(x))
    #==========================================================================#

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return `(crack_logits, multilabel_logits)` with one encoder pass."""
        features = self.encode(x)
        return self.decode_crack(features), self.decode_multilabel(features)

    # ------------------------------------------------------------------ #
    # Introspection
    # ------------------------------------------------------------------ #
    def parameter_report(self) -> dict[str, int]:
        """Parameter counts per block, used in the run summary and in the thesis."""

        def count(module: nn.Module) -> int:
            return sum(p.numel() for p in module.parameters())

        report = {
            "encoder_shared": count(self.encoder),
            "crack_decoder": count(self.crack_decoder),
            "crack_head": count(self.crack_head),
            "multilabel_decoder": count(self.multilabel_decoder),
            "multilabel_head": count(self.multilabel_head),
        }
        report["total"] = sum(report.values())
        report["trainable"] = sum(p.numel() for p in self.parameters() if p.requires_grad)
        report["shared_fraction"] = report["encoder_shared"] / max(report["total"], 1)
        return report


class SingleHeadAdapter(nn.Module):
    """Expose one head of a multitask model as a plain single-output module.

    `predict_sliding_window` and `predict_sliding_window_multilabel` call
    `model(batch)` and expect a single tensor. Wrapping instead of editing them
    keeps the P0-P3 and P1ML inference code paths bit-for-bit identical.
    """

    def __init__(self, model: MultiTaskSegmentationModel, head: str) -> None:
        super().__init__()
        if head not in {CRACK_HEAD, MULTILABEL_HEAD}:
            raise ValueError(f"head must be '{CRACK_HEAD}' or '{MULTILABEL_HEAD}', got '{head}'.")
        self.model = model
        self.head = head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.head == CRACK_HEAD:
            return self.model.forward_crack(x)
        return self.model.forward_multilabel(x)


def build_multitask_model(cfg) -> MultiTaskSegmentationModel:
    """Instantiate the P4-A network from the `model` section of the config."""
    multilabel_classes = int(cfg.get("multilabel_classes", len(UNIFIED_DAMAGE_CLASSES)))
    if multilabel_classes != len(UNIFIED_DAMAGE_CLASSES):
        raise ValueError(
            f"model.multilabel_classes must be {len(UNIFIED_DAMAGE_CLASSES)} "
            f"(the unified DACL10K taxonomy), got {multilabel_classes}."
        )

    return MultiTaskSegmentationModel(
        arch=cfg.arch,
        encoder=cfg.encoder,
        encoder_weights=cfg.get("encoder_weights"),
        in_channels=int(cfg.in_channels),
        crack_classes=int(cfg.get("crack_classes", 1)),
        multilabel_classes=multilabel_classes,
    )
