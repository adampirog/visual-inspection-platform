import numpy as np


def safe_div(num: np.intp, denom: np.intp) -> float:
    return float(num / denom) if denom else 0.0


def _precision_recall_f1_support(true: np.ndarray, pred: np.ndarray) -> dict[str, float | int]:
    tp = np.count_nonzero(true & pred)
    fp = np.count_nonzero(~true & pred)
    fn = np.count_nonzero(true & ~pred)

    return {
        "precision": safe_div(tp, tp + fp),
        "recall": safe_div(tp, tp + fn),
        "f1 score": safe_div(2 * tp, 2 * tp + fp + fn),
        "support": int(np.count_nonzero(true)),
    }


def evaluate(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    names: tuple[str, str] = ("0", "1"),
) -> dict[str, dict[str, float | int]]:
    if y_true.shape != y_pred.shape or y_true.size == 0:
        raise ValueError("Expected nonempty arrays with matching shapes")

    if not np.isin(y_true, (0, 1)).all() or not np.isin(y_pred, (0, 1)).all():
        raise ValueError("Expected binary labels (0 or 1)")

    result = {
        name: _precision_recall_f1_support(y_true == label, y_pred == label)
        for label, name in zip((0, 1), names, strict=True)
    }

    metrics = ("precision", "recall", "f1 score")

    result["macro avg"] = {
        metric: float(np.mean([result[name][metric] for name in names])) for metric in metrics
    }
    result["macro avg"]["support"] = y_true.size

    result["weighted avg"] = {
        metric: float(
            sum(result[name][metric] * result[name]["support"] for name in names) / y_true.size
        )
        for metric in metrics
    }
    result["weighted avg"]["support"] = y_true.size

    return result
