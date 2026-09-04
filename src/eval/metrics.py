"""Segmentation metrics for the binary crack task.

Definitions (TP/FP/FN counted over pixels of the positive class):

    IoU  = TP / (TP + FP + FN)
    Dice = 2TP / (2TP + FP + FN)          == F1 on pixels

Both are reported in two flavours, because they answer different questions and
the gap between them is a known reporting pitfall in crack segmentation:

* **dataset-level (micro)**: counts are accumulated over the whole split, then
  the ratio is computed once. Dominated by large cracks, stable, comparable with
  most published tables.
* **image-level (macro)**: the metric is computed per image and then averaged.
  Penalises failures on images with few crack pixels, which is the realistic
  inspection scenario.

Empty ground-truth images (no crack at all) are excluded from the image-level
average and tracked separately as a false-positive rate, since Dice is undefined
when TP = FN = 0.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

EPS = 1e-7


def _counts(pred: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """Per-sample TP, FP, FN, TN over binary tensors of shape [B,1,H,W]."""
    pred = pred.flatten(1).float()
    target = target.flatten(1).float()
    tp = (pred * target).sum(1)
    fp = (pred * (1 - target)).sum(1)
    fn = ((1 - pred) * target).sum(1)
    tn = ((1 - pred) * (1 - target)).sum(1)
    return tp, fp, fn, tn


@dataclass
class SegmentationMetrics:
    """Streaming accumulator: call `update` per batch, `compute` at the end."""

    threshold: float = 0.5
    tp: float = 0.0
    fp: float = 0.0
    fn: float = 0.0
    tn: float = 0.0
    _image_iou: list[float] = field(default_factory=list)
    _image_dice: list[float] = field(default_factory=list)
    n_empty_gt: int = 0
    n_empty_gt_with_prediction: int = 0

    @torch.no_grad()
    def update(self, logits: torch.Tensor, target: torch.Tensor) -> None:
        """Accumulate one batch. `logits` are raw model outputs (pre-sigmoid)."""
        pred = (torch.sigmoid(logits) > self.threshold).float()
        tp, fp, fn, tn = _counts(pred, target)

        # Dataset-level (micro) counts.
        self.tp += tp.sum().item()
        self.fp += fp.sum().item()
        self.fn += fn.sum().item()
        self.tn += tn.sum().item()

        # Image-level (macro) values, only where the ground truth is non-empty.
        positive_gt = (tp + fn) > 0
        for i in range(pred.shape[0]):
            if positive_gt[i]:
                self._image_iou.append((tp[i] / (tp[i] + fp[i] + fn[i] + EPS)).item())
                self._image_dice.append((2 * tp[i] / (2 * tp[i] + fp[i] + fn[i] + EPS)).item())
            else:
                self.n_empty_gt += 1
                self.n_empty_gt_with_prediction += int(fp[i].item() > 0)

    def compute(self) -> dict[str, float]:
        """Return the full metric dictionary."""
        tp, fp, fn = self.tp, self.fp, self.fn
        n_img = max(len(self._image_iou), 1)
        return {
            "iou": tp / (tp + fp + fn + EPS),
            "dice": 2 * tp / (2 * tp + fp + fn + EPS),
            "precision": tp / (tp + fp + EPS),
            "recall": tp / (tp + fn + EPS),
            "pixel_accuracy": (tp + self.tn) / (tp + fp + fn + self.tn + EPS),
            "iou_image_mean": sum(self._image_iou) / n_img,
            "dice_image_mean": sum(self._image_dice) / n_img,
            "n_images_with_crack": len(self._image_iou),
            "n_images_empty_gt": self.n_empty_gt,
            # Share of crack-free images where the model still predicts crack.
            "false_alarm_rate_empty_gt": (
                self.n_empty_gt_with_prediction / max(self.n_empty_gt, 1)
            ),
        }


@torch.no_grad()
def sweep_threshold(
    logits: torch.Tensor, target: torch.Tensor, thresholds: list[float]
) -> dict[float, dict[str, float]]:
    """Evaluate one batch of stored logits at several thresholds (calibration)."""
    results = {}
    for threshold in thresholds:
        meter = SegmentationMetrics(threshold=threshold)
        meter.update(logits, target)
        results[threshold] = meter.compute()
    return results
