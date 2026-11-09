import numpy as np


def roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Exact ROC AUC with correct handling of tied scores."""
    labels = np.asarray(labels, dtype=bool).ravel()
    scores = np.asarray(scores).ravel()
    if labels.size != scores.size or labels.size == 0:
        raise ValueError("Labels and scores must have the same nonzero size")

    positives = int(labels.sum())
    negatives = labels.size - positives
    if not positives or not negatives:
        return float("nan")  # ROC AUC is undefined for a single class

    order = np.argsort(scores)[::-1]
    ranked_labels = labels[order]
    ranked_scores = scores[order]
    # Calculate TPR/FPR only at the end of each group of equal scores.
    ends = np.r_[np.flatnonzero(np.diff(ranked_scores)), labels.size - 1]
    tp = np.cumsum(ranked_labels)[ends]
    fp = ends + 1 - tp
    tpr = np.r_[0, tp / positives]
    fpr = np.r_[0, fp / negatives]
    return float(np.trapezoid(tpr, fpr))
