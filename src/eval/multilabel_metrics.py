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

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import torch
from tqdm.auto import tqdm

from ..data.class_mapping import UNIFIED_DAMAGE_CLASSES

EPS = 1e-7


@dataclass
class MultilabelSegmentationMetrics:
    """Streaming per-channel accumulator: `update` per batch, `compute` at the end."""

    threshold: float | Sequence[float] | dict[str, float] = 0.5
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

        # P4-A: `threshold` may be a single float (global threshold, the historical
        # behaviour used by every P1ML CSV) or one value per channel (calibrated
        # thresholds). The scalar path below is kept byte-identical on purpose.
        self._threshold_is_scalar = isinstance(self.threshold, (int, float)) and not isinstance(
            self.threshold, bool
        )
        self._threshold_vector = resolve_class_thresholds(self.threshold, self.class_names)
        self._threshold_tensor: torch.Tensor | None = None

    def resolved_thresholds(self) -> dict[str, float]:
        """Threshold actually applied to each channel, scalar or calibrated."""
        return dict(zip(self.class_names, self._threshold_vector))

    def _threshold_as_tensor(self, reference: torch.Tensor) -> torch.Tensor:
        """Cache the per-channel threshold as a [1,C,1,1] tensor for broadcasting."""
        if (
            self._threshold_tensor is None
            or self._threshold_tensor.device != reference.device
            or self._threshold_tensor.dtype != reference.dtype
        ):
            self._threshold_tensor = torch.tensor(
                self._threshold_vector, dtype=reference.dtype, device=reference.device
            ).view(1, -1, 1, 1)
        return self._threshold_tensor

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

        probability = torch.sigmoid(logits.float())
        if self._threshold_is_scalar:
            pred = (probability > float(self.threshold)).float()
        else:
            pred = (probability > self._threshold_as_tensor(probability)).float()
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


# --------------------------------------------------------------------------- #
# P4-A: per-channel threshold calibration helpers
# --------------------------------------------------------------------------- #
def resolve_class_thresholds(
    threshold: float | Sequence[float] | Mapping[str, float],
    class_names: Sequence[str],
) -> list[float]:
    """Normalize a threshold specification into one float per channel.

    Accepted forms:

    * `float` -> the same global threshold on every channel (legacy behaviour);
    * `Sequence[float]` of length C -> already in channel order;
    * `Mapping[str, float]` -> keyed by unified class name; every class must be
      present, because a silently defaulted channel would make a "calibrated"
      table partly uncalibrated without any trace in the artifacts.
    """
    names = list(class_names)

    if isinstance(threshold, Mapping):
        missing = [name for name in names if name not in threshold]
        if missing:
            raise ValueError(f"Missing threshold for classes: {missing}.")
        unknown = [key for key in threshold if key not in names]
        if unknown:
            raise ValueError(f"Unknown classes in the threshold mapping: {unknown}.")
        values = [float(threshold[name]) for name in names]
    elif isinstance(threshold, (int, float)) and not isinstance(threshold, bool):
        values = [float(threshold)] * len(names)
    elif isinstance(threshold, Sequence) and not isinstance(threshold, (str, bytes)):
        values = [float(value) for value in threshold]
        if len(values) != len(names):
            raise ValueError(
                f"Expected {len(names)} thresholds (one per channel), got {len(values)}."
            )
    else:
        raise TypeError(f"Unsupported threshold specification of type {type(threshold)!r}.")

    for name, value in zip(names, values):
        if not 0.0 < value < 1.0:
            raise ValueError(f"Threshold for '{name}' must lie in (0, 1), got {value}.")

    return values


def select_thresholds_per_class(
    rows,
    class_names: Sequence[str] | None = None,
    metric: str = "dice",
) -> dict[str, dict]:
    """Pick, per class, the sweep threshold maximising `metric`.

    `rows` is any iterable of mappings holding at least `class`, `threshold` and
    `metric` (typically the rows of `metrics_*_per_class.csv`).

    Tie-break: **the highest threshold wins**. Two thresholds with the same Dice
    describe the same operating quality at different operating points; the
    stricter one predicts fewer pixels, hence fewer false positives, which is the
    conservative choice for an inspection pipeline and makes the selection
    deterministic and reproducible instead of dependent on the row order.

    Returns, per class: selected threshold, its score, all candidates, and the
    number of tied candidates, so the JSON artifact documents the decision.
    """
    names = list(class_names or UNIFIED_DAMAGE_CLASSES)
    candidates: dict[str, list[tuple[float, float]]] = {name: [] for name in names}

    for row in rows:
        name = str(row["class"])
        if name not in candidates:
            raise ValueError(f"Unexpected class '{name}' in the sweep rows.")
        candidates[name].append((float(row["threshold"]), float(row[metric])))

    selection: dict[str, dict] = {}
    for name in names:
        values = sorted(candidates[name])
        if not values:
            raise ValueError(f"No sweep row available for class '{name}'.")

        best_score = max(score for _, score in values)
        tied = [threshold for threshold, score in values if score == best_score]

        selection[name] = {
            "threshold": float(max(tied)),  # tie-break: highest threshold
            f"{metric}_at_selected_threshold": best_score,
            "metric": metric,
            "candidate_thresholds": [threshold for threshold, _ in values],
            f"candidate_{metric}": [score for _, score in values],
            "n_tied_candidates": len(tied),
            "tie_break_rule": "highest threshold among the tied maxima",
        }

    return selection


@torch.no_grad()
def score_multilabel_probabilities(
    probability: torch.Tensor,
    target: torch.Tensor,
    threshold: float | Sequence[float] | Mapping[str, float],
    class_names: Sequence[str] | None = None,
    meter: "MultilabelSegmentationMetrics | None" = None,
) -> "MultilabelSegmentationMetrics":
    """Accumulate one full-resolution prediction into a (possibly mixed-threshold) meter.

    The sliding-window predictor returns blended *probabilities*, so the inverse
    sigmoid is applied before delegating to the meter: a single thresholding code
    path is kept for global and calibrated thresholds alike, which is what makes
    the calibrated CSVs comparable in kind with the standard ones.
    """
    names = list(class_names or UNIFIED_DAMAGE_CLASSES)
    active = meter or MultilabelSegmentationMetrics(threshold=threshold, class_names=names)

    logits = torch.logit(probability.clamp(1e-6, 1.0 - 1e-6))
    if logits.ndim == 3:
        logits = logits.unsqueeze(0)
    if target.ndim == 3:
        target = target.unsqueeze(0)

    active.update(logits, target)
    return active
