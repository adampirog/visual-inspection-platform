import numpy as np

from visual_inspection_platform.types import AnomalyMaps, BinaryMasks


def _binary_counts(y_pred: BinaryMasks, y_true: BinaryMasks) -> tuple[int, int, int]:
    """Return counts in (true positives, false positives, false negatives) order."""
    if y_pred.shape != y_true.shape:
        raise ValueError("Prediction and ground-truth shapes must match")

    tp = int(np.count_nonzero(y_pred & y_true))
    fp = int(np.count_nonzero(y_pred & ~y_true))
    fn = int(np.count_nonzero(~y_pred & y_true))

    return tp, fp, fn


def _binary_metrics(tp: int, fp: int, fn: int) -> dict[str, float]:
    """Compute binary segmentation metrics, returning zero for undefined ratios."""
    return {
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "dice": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
    }


def precision(y_pred: BinaryMasks, y_true: BinaryMasks) -> float:
    """Return pixel precision, or zero when no pixels are predicted anomalous."""
    tp, fp, _ = _binary_counts(y_pred, y_true)
    return tp / (tp + fp) if tp + fp else 0.0


def recall(y_pred: BinaryMasks, y_true: BinaryMasks) -> float:
    """Return pixel recall, or zero when there are no anomalous ground-truth pixels."""
    tp, _, fn = _binary_counts(y_pred, y_true)
    return tp / (tp + fn) if tp + fn else 0.0


def dice(y_pred: BinaryMasks, y_true: BinaryMasks) -> float:
    """Return pixel Dice, or zero when both masks contain no anomalous pixels."""
    return _binary_metrics(*_binary_counts(y_pred, y_true))["dice"]


def _ranking_metrics(y_pred: np.ndarray, y_true: np.ndarray) -> dict[str, float]:
    """Compute AUROC and average precision over pooled pixels.

    Scores must be finite. Ties share a ranking threshold. AUROC is NaN
    for single-class ground truth; AP is zero without positives and one
    without negatives.
    """

    if y_pred.shape != y_true.shape:
        raise ValueError("Prediction and ground-truth shapes must match")

    scores = y_pred.ravel()
    labels = y_true.ravel()

    if scores.size == 0:
        raise ValueError("Cannot evaluate empty arrays")
    if not np.isfinite(scores).all():
        raise ValueError("Anomaly scores must be finite")

    positives = int(np.count_nonzero(labels))
    negatives = scores.size - positives

    if positives == 0:
        return {"auroc": float("nan"), "ap": 0.0}
    if negatives == 0:
        return {"auroc": float("nan"), "ap": 1.0}

    order = np.argsort(-scores, kind="stable")
    sorted_scores = scores[order]
    sorted_labels = labels[order]

    # Evaluate each distinct score once to handle ties correctly.
    ends = np.r_[
        np.flatnonzero(sorted_scores[1:] != sorted_scores[:-1]),
        scores.size - 1,
    ]

    tp = np.cumsum(sorted_labels, dtype=np.int64)[ends]
    fp = ends + 1 - tp

    tpr = tp / positives
    fpr = fp / negatives
    precision_values = tp / (tp + fp)

    auroc = float(np.trapezoid(np.r_[0.0, tpr], np.r_[0.0, fpr]))
    ap = float(np.sum(np.diff(np.r_[0.0, tpr]) * precision_values))

    return {"auroc": auroc, "ap": ap}


def pixel_auroc(y_pred: AnomalyMaps, y_true: BinaryMasks) -> float:
    """Area under the pixel-level ROC curve; NaN for single-class ground truth."""
    return _ranking_metrics(y_pred, y_true)["auroc"]


def pixel_ap(y_pred: AnomalyMaps, y_true: BinaryMasks) -> float:
    """Compute average precision, zero without positives and one without negatives."""
    return _ranking_metrics(y_pred, y_true)["ap"]
