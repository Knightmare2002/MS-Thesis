"""Albumentations pipelines.

Rationale for the augmentations
-------------------------------
* Cracks are thin, low-contrast, orientation-agnostic structures: flips and
  90-degree rotations are safe and effective; heavy elastic warping is avoided
  because it can destroy 1-2 px wide crack topology.

* Real inspection photos vary a lot in exposure and white balance, hence the
  brightness/contrast and gamma jitter (cheap domain randomisation, which is
  what should help on the external UAV/RGB blind test set).
  
* Normalisation uses ImageNet statistics because the smp encoders are
  ImageNet-pretrained.
"""

from __future__ import annotations

import albumentations as A
import cv2
from albumentations.pytorch import ToTensorV2

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def train_transform(image_size: int) -> A.Compose:
    """Augmentations + resize + normalise for training."""
    return A.Compose(
        [
            # Keep aspect-ratio-free square resize: simple and consistent at eval.
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.2),
            A.RandomRotate90(p=0.5),
            # Small geometric jitter; constant border avoids mirrored fake cracks.
            A.Affine(
                translate_percent=(-0.05, 0.05),
                scale=(0.85, 1.15),
                rotate=(-15, 15),
                border_mode=cv2.BORDER_CONSTANT,
                p=0.5,
            ),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.RandomGamma(gamma_limit=(80, 120), p=0.3),
            A.GaussNoise(p=0.2),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]
    )


def eval_transform(image_size: int) -> A.Compose:
    """Deterministic pipeline for validation / test / external benchmark."""
    return A.Compose(
        [
            A.Resize(image_size, image_size),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]
    )

def patch_train_transform() -> A.Compose:
    """Augmentations for an already extracted native-resolution training patch."""
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.2),
            A.RandomRotate90(p=0.5),
            A.Affine(
                translate_percent=(-0.05, 0.05),
                scale=(0.85, 1.15),
                rotate=(-15, 15),
                border_mode=cv2.BORDER_CONSTANT,
                p=0.5,
            ),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.RandomGamma(gamma_limit=(80, 120), p=0.3),
            A.GaussNoise(p=0.2),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]
    )


def patch_eval_transform() -> A.Compose:
    """Deterministic normalization for native-resolution sliding-window patches."""
    return A.Compose(
        [
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]
    )

# --------------------------------------------------------------------------- #
# P1ML: multilabel patches
# --------------------------------------------------------------------------- #
# Two differences w.r.t. the binary patch pipelines:
#   * masks are [H,W,6] stacks, so ToTensorV2(transpose_mask=True) is required to obtain the [6,H,W] layout the loss and the metrics expect;
#   * every geometric op resamples the mask with nearest neighbour
#     (mask_interpolation=cv2.INTER_NEAREST), otherwise bilinear interpolation would produce non-binary targets and silently corrupt per-class Dice.
def patch_train_transform_multilabel(profile: str = "base") -> A.Compose:
    """Augmentations for an extracted native-resolution multilabel patch.

    profile="base": pipeline used by P1ML-A ... T3b (unchanged).
    profile="t4":   adds robustness to the Lincoln Street acquisition (illumination,
                    shadows, blur/compression, scale and mild perspective), with limits
                    chosen so that thin cracks survive (blur <= 5 px, downscale >= 0.5).
    """
    if profile not in ("base", "t4"):
        raise ValueError(f"Unknown augmentation profile: {profile!r}")
    t4 = profile == "t4"

    geometric = [
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.2),
        A.RandomRotate90(p=0.5),
        A.Affine(
            translate_percent=(-0.05, 0.05),
            scale=(0.75, 1.30) if t4 else (0.85, 1.15),
            rotate=(-15, 15),
            border_mode=cv2.BORDER_CONSTANT,
            interpolation=cv2.INTER_LINEAR,
            mask_interpolation=cv2.INTER_NEAREST,
            p=0.5,
        ),
    ]
    if t4:
        geometric.append(
            A.Perspective(
                scale=(0.02, 0.05),
                keep_size=True,
                border_mode=cv2.BORDER_CONSTANT,
                interpolation=cv2.INTER_LINEAR,
                mask_interpolation=cv2.INTER_NEAREST,
                p=0.2,
            )
        )

    photometric = [
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.RandomGamma(gamma_limit=(80, 120), p=0.3),
        A.GaussNoise(p=0.2),
    ]
    if t4:
        photometric += [
            A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=15, val_shift_limit=0, p=0.3),
            A.RandomShadow(shadow_roi=(0, 0, 1, 1), num_shadows_limit=(1, 2), p=0.15),
            A.OneOf(
                [
                    A.MotionBlur(blur_limit=(3, 5)),
                    A.GaussianBlur(blur_limit=(3, 5)),
                    A.Downscale(scale_range=(0.5, 0.9)),
                ],
                p=0.2,
            ),
            A.ImageCompression(quality_range=(60, 95), p=0.3),
        ]

    return A.Compose(
        geometric
        + photometric
        + [
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(transpose_mask=True),
        ]
    )


def patch_eval_transform_multilabel() -> A.Compose:
    """Deterministic normalization for multilabel patches ([H,W,6] -> [6,H,W])."""
    return A.Compose(
        [
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(transpose_mask=True),
        ]
    )
