"""Metrics for the P1ML multilabel damage task (6 independent channels).

The binary `SegmentationMetrics` is intentionally left untouched: P0-P3 keep their exact numbers. Here every quantity is accumulated **per channel**, because the six unified damage classes differ by more than two orders of magnitude in pixel frequency and any channel-collapsed statistic would be a proxy for `surface` alone.

Reported quantities
-------------------
* per class: IoU, Dice, precision, recall (dataset-level, i.e. micro over the pixels of that channel), plus its pixel support and how many images contain it;

* `macro_dice_present` / `macro_iou_present`: unweighted mean over the classes that actually occur in the evaluated split (support > 0). Classes absent from the split are skipped rather than scored 0, which would confound "not present" with "not detected". This is the model-selection and early-stopping metric;

* `micro_dice` / `micro_iou` / `micro_precision` / `micro_recall`: counts pooled over all channels, dominated by frequent classes, reported for comparability;

* `false_alarm_rate_empty_gt`: share of (image, channel) pairs with empty ground truth where the model still predicts that damage.

Aliases `dice`, `iou`, `precision`, `recall` map to the micro values so the existing `fit` history writer and console logging keep working unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from tqdm.auto import tqdm

from ..data.class_mapping import UNIFIED_DAMAGE_CLASSES

EPS = 1e-7


@dataclass
class MultilabelSegmentationMetrics:
    """Streaming per-channel accumulator: `update` per batch, `compute` at the end."""

    threshold: float = 0.5
    class_names: list[str] = field(default_factory=lambda: list(UNIFIED_DAMAGE_CLASSES))
    tp: list[float] = field(default_factory=list)
    fp: list[float] = field(default_factory=list)
    fn: list[float] = field(default_factory=list)
    tn: list[float] = field(default_factory=list)
    n_images_present: list[int] = field(default_factory=list)
    n_images_empty: list[int] = field(default_factory=list)
    n_images_empty_with_prediction: list[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        n = len(self.class_names)
        if not self.tp:
            self.tp, self.fp, self.fn, self.tn = ([0.0] * n for _ in range(4))
            self.n_images_present = [0] * n
            self.n_images_empty = [0] * n
            self.n_images_empty_with_prediction = [0] * n

    @property
    def n_classes(self) -> int:
        return len(self.class_names)

    @torch.no_grad()
    def update(self, logits: torch.Tensor, target: torch.Tensor) -> None:
        """Accumulate one batch of raw logits [B,C,H,W] against a binary target."""
        if logits.shape != target.shape:
            raise ValueError(
                f"Shape mismatch: logits {tuple(logits.shape)} vs target {tuple(target.shape)}."
            )
        if logits.shape[1] != self.n_classes:
            raise ValueError(
                f"Expected {self.n_classes} channels, got {logits.shape[1]}."
            )

        pred = (torch.sigmoid(logits.float()) > self.threshold).float()
        target = target.float()

        # Flatten the spatial dimensions only: shape [B,C,HW] -> counts [B,C].
        pred_flat = pred.flatten(2)
        target_flat = target.flatten(2)

        tp = (pred_flat * target_flat).sum(2)
        fp = (pred_flat * (1.0 - target_flat)).sum(2)
        fn = ((1.0 - pred_flat) * target_flat).sum(2)
        tn = ((1.0 - pred_flat) * (1.0 - target_flat)).sum(2)

        tp_c, fp_c, fn_c, tn_c = (t.sum(0).tolist() for t in (tp, fp, fn, tn))
        present = ((tp + fn) > 0)
        has_fp = (fp > 0)

        for channel in range(self.n_classes):
            self.tp[channel] += tp_c[channel]
            self.fp[channel] += fp_c[channel]
            self.fn[channel] += fn_c[channel]
            self.tn[channel] += tn_c[channel]

            present_column = present[:, channel]
            self.n_images_present[channel] += int(present_column.sum().item())

            empty_column = ~present_column
            self.n_images_empty[channel] += int(empty_column.sum().item())
            self.n_images_empty_with_prediction[channel] += int(
                (empty_column & has_fp[:, channel]).sum().item()
            )

    def per_class(self) -> dict[str, dict[str, float]]:
        """Dataset-level metrics for each unified damage class."""
        out = {}
        for channel, name in enumerate(self.class_names):
            tp, fp, fn = self.tp[channel], self.fp[channel], self.fn[channel]
            support = tp + fn
            out[name] = {
                "iou": tp / (tp + fp + fn + EPS),
                "dice": 2 * tp / (2 * tp + fp + fn + EPS),
                "precision": tp / (tp + fp + EPS),
                "recall": tp / (tp + fn + EPS),
                "support_pixels": support,
                "predicted_pixels": tp + fp,
                "n_images_present": self.n_images_present[channel],
                "n_images_empty_gt": self.n_images_empty[channel],
                "false_alarm_rate_empty_gt": (
                    self.n_images_empty_with_prediction[channel]
                    / max(self.n_images_empty[channel], 1)
                ),
                "is_present": float(support > 0),
            }
        return out

    def compute(self) -> dict[str, float]:
        """Return a flat, CSV-friendly dictionary of all metrics."""
        per_class = self.per_class()
        present = [name for name, values in per_class.items() if values["support_pixels"] > 0]
        n_present = max(len(present), 1)

        tp, fp, fn, tn = (sum(values) for values in (self.tp, self.fp, self.fn, self.tn))

        metrics: dict[str, float] = {
            "macro_dice_present": sum(per_class[n]["dice"] for n in present) / n_present,
            "macro_iou_present": sum(per_class[n]["iou"] for n in present) / n_present,
            "macro_precision_present": sum(per_class[n]["precision"] for n in present) / n_present,
            "macro_recall_present": sum(per_class[n]["recall"] for n in present) / n_present,
            "micro_dice": 2 * tp / (2 * tp + fp + fn + EPS),
            "micro_iou": tp / (tp + fp + fn + EPS),
            "micro_precision": tp / (tp + fp + EPS),
            "micro_recall": tp / (tp + fn + EPS),
            "pixel_accuracy": (tp + tn) / (tp + fp + fn + tn + EPS),
            "n_classes_present": float(len(present)),
            "false_alarm_rate_empty_gt": (
                sum(self.n_images_empty_with_prediction) / max(sum(self.n_images_empty), 1)
            ),
        }

        # Aliases so engine.fit's history writer / logging work unchanged.
        metrics["dice"] = metrics["micro_dice"]
        metrics["iou"] = metrics["micro_iou"]
        metrics["precision"] = metrics["micro_precision"]
        metrics["recall"] = metrics["micro_recall"]

        for name, values in per_class.items():
            for key, value in values.items():
                metrics[f"{key}_{name}"] = value

        return metrics


@torch.no_grad()
def evaluate_multilabel(model, loader, criterion, device, threshold: float = 0.5) -> dict[str, float]:
    """Patch-level multilabel evaluation. Signature mirrors `engine.evaluate`."""
    model.eval()
    meter = MultilabelSegmentationMetrics(threshold=threshold)
    total_loss, n_batches = 0.0, 0

    for images, masks in tqdm(loader, desc="eval-ml", leave=False):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        logits = model(images)
        total_loss += criterion(logits, masks).item()
        n_batches += 1
        meter.update(logits, masks)

    metrics = meter.compute()
    metrics["loss"] = total_loss / max(n_batches, 1)
    return metrics


@torch.no_grad()
def sweep_threshold_multilabel(
    probability: torch.Tensor,
    target: torch.Tensor,
    thresholds: list[float],
    class_names: list[str] | None = None,
) -> dict[float, dict[str, float]]:
    """Re-score stored *probabilities* at several thresholds (calibration).

    Unlike the binary `sweep_threshold`, the input is already sigmoid-activated:
    the sliding-window predictor returns blended probabilities, so logits are no longer available. `logit(p)` is applied before delegating to the meter, which keeps a single thresholding code path.
    """
    names = list(class_names or UNIFIED_DAMAGE_CLASSES)
    probability = probability.clamp(1e-6, 1.0 - 1e-6)
    logits = torch.log(probability / (1.0 - probability))

    results = {}
    for threshold in thresholds:
        meter = MultilabelSegmentationMetrics(threshold=threshold, class_names=names)
        meter.update(logits, target)
        results[threshold] = meter.compute()
    return results
