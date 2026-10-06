"""Connected-component post-processing for multilabel damage masks.

A predicted component smaller than a per-class minimum area is discarded. The
area is expressed as a FRACTION of the image area, so that operating points
calibrated on DACL10K transfer to images of a different resolution (Lincoln
Street) without re-tuning a pixel count.

`image_counts` evaluates every (threshold, min-area) pair of a grid for one
image in a single pass: per threshold, one connected-component labelling plus
per-component overlap statistics; each min-area value is then a mask over the
component list, not a new labelling.
"""

from __future__ import annotations

import cv2
import numpy as np

COUNT_FIELDS = (
    "tp",                    # predicted & GT pixels (after filtering)
    "fp",                    # predicted & not-GT pixels
    "fn",                    # GT pixels not predicted
    "gt_empty",              # 1 if the channel has no GT pixel in this image
    "false_alarm",           # 1 if gt_empty and at least one component survives
    "n_pred_components",     # surviving predicted components
    "n_pred_components_tp",  # surviving components overlapping the GT
    "n_gt_components",       # GT connected components
    "n_gt_detected",         # GT components touched by a surviving component
    "n_gt_components_large",  # GT components with area >= min area
    "n_gt_detected_large",    # of those, touched by a surviving component
)


def min_area_pixels(area_fraction: float, image_shape) -> int:
    """Convert an area fraction into a pixel count (>= 1, i.e. no filtering at 0)."""
    height, width = image_shape[:2]
    return max(1, int(np.ceil(float(area_fraction) * height * width)))


def remove_small_components(binary: np.ndarray, min_area: int) -> np.ndarray:
    """Drop 8-connected components with fewer than `min_area` pixels."""
    binary = binary.astype(bool)
    if min_area <= 1 or not binary.any():
        return binary
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary.astype(np.uint8), connectivity=8)
    keep = np.zeros(n, dtype=bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_area
    return keep[labels]


def apply_operating_points(probability: np.ndarray, thresholds, min_area_fractions) -> np.ndarray:
    """[C,H,W] probabilities -> [C,H,W] boolean masks with frozen per-class (tau_c, A_c)."""
    out = np.zeros(probability.shape, dtype=bool)
    for channel, (threshold, fraction) in enumerate(zip(thresholds, min_area_fractions)):
        out[channel] = remove_small_components(
            probability[channel] > float(threshold),
            min_area_pixels(fraction, probability.shape[1:]),
        )
    return out


def image_counts(probability: np.ndarray, target: np.ndarray, thresholds, min_area_fractions) -> np.ndarray:
    """Counts [C, T, A, len(COUNT_FIELDS)] for one image and every grid point."""
    n_classes = probability.shape[0]
    area_px = np.array([min_area_pixels(a, probability.shape[1:]) for a in min_area_fractions])
    counts = np.zeros((n_classes, len(thresholds), len(area_px), len(COUNT_FIELDS)), dtype=np.int64)

    for channel in range(n_classes):
        gt = target[channel] > 0
        gt_total = int(gt.sum())
        gt_empty = int(gt_total == 0)
        n_gt, gt_labels, gt_stats, _ = cv2.connectedComponentsWithStats(gt.astype(np.uint8), connectivity=8)
        n_gt -= 1  # background
        gt_areas = gt_stats[1:, cv2.CC_STAT_AREA].astype(np.int64)

        for t_index, threshold in enumerate(thresholds):
            pred = probability[channel] > float(threshold)
            n, labels, stats, _ = cv2.connectedComponentsWithStats(pred.astype(np.uint8), connectivity=8)
            areas = stats[1:, cv2.CC_STAT_AREA].astype(np.int64)
            tp_per_component = np.bincount(labels[gt], minlength=n)[1:].astype(np.int64)

            # Largest predicted component touching each GT component -> detected iff >= A.
            max_overlap_area = np.zeros(n_gt, dtype=np.int64)
            both = gt & pred
            if n_gt > 0 and both.any():
                pairs = np.unique((gt_labels[both].astype(np.int64) - 1) * n + (labels[both] - 1))
                np.maximum.at(max_overlap_area, pairs // n, areas[pairs % n])

            for a_index, min_area in enumerate(area_px):
                keep = areas >= min_area
                tp = int(tp_per_component[keep].sum())
                predicted = int(areas[keep].sum())
                counts[channel, t_index, a_index] = (
                    tp,
                    predicted - tp,
                    gt_total - tp,
                    gt_empty,
                    int(gt_empty and keep.any()),
                    int(keep.sum()),
                    int((keep & (tp_per_component > 0)).sum()),
                    n_gt,
                    int((max_overlap_area >= min_area).sum()),
                    int((gt_areas >= min_area).sum()),
                    int(((max_overlap_area >= min_area) & (gt_areas >= min_area)).sum()),
                )

    return counts


def metrics_from_counts(counts: np.ndarray) -> dict[str, np.ndarray]:
    """Metrics from summed counts [..., len(COUNT_FIELDS)] (any leading shape)."""
    tp, fp, fn, empty, false_alarm, n_pred, n_pred_tp, n_gt, n_gt_det, n_gt_large, n_gt_det_large = np.moveaxis(
        counts.astype(np.float64), -1, 0
    )
    eps = 1e-12
    return {
        "dice": 2 * tp / (2 * tp + fp + fn + eps),
        "iou": tp / (tp + fp + fn + eps),
        "precision": tp / (tp + fp + eps),
        "recall": tp / (tp + fn + eps),
        "false_alarm_rate_empty_gt": false_alarm / np.maximum(empty, 1),
        "object_precision": n_pred_tp / np.maximum(n_pred, 1),
        "object_recall": n_gt_det / np.maximum(n_gt, 1),
        "object_recall_large": n_gt_det_large / np.maximum(n_gt_large, 1),
        "support_pixels": tp + fn,
        "n_images_empty_gt": empty,
        "n_false_alarms": false_alarm,
    }
