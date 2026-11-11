"""Evaluate anomaly segmentation predictions by object category."""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import TypedDict

import numpy as np

from visual_inspection_platform.postprocessors import Postprocessor
from visual_inspection_platform.types import AnomalyMaps, BinaryMasks

from .metrics import _binary_counts, _binary_metrics, _ranking_metrics


class EvaluationBatch(TypedDict):
    y_true: BinaryMasks
    y_pred: AnomalyMaps
    categories: Sequence[str]


@dataclass
class Accumulator:
    scores: list[np.ndarray] = field(default_factory=list)
    labels: list[np.ndarray] = field(default_factory=list)
    tp: int = 0
    fp: int = 0
    fn: int = 0

    def add(self, scores: AnomalyMaps, labels: BinaryMasks, masks: BinaryMasks | None) -> None:
        self.scores.append(scores.ravel())
        self.labels.append(labels.ravel())

        if masks is not None:
            tp, fp, fn = _binary_counts(masks, labels)
            self.tp += tp
            self.fp += fp
            self.fn += fn

    def result(self, include_binary: bool) -> dict[str, dict[str, float]]:
        cont = _ranking_metrics(np.concatenate(self.scores), np.concatenate(self.labels))
        cont = {"pixel_" + name: value for name, value in cont.items()}

        result = {"continuous": cont}

        if include_binary:
            result["binary"] = _binary_metrics(self.tp, self.fp, self.fn)

        return result


def _prepare_batch(
    batch: EvaluationBatch, postprocessor: Postprocessor | None
) -> tuple[AnomalyMaps, BinaryMasks, np.ndarray, BinaryMasks | None]:
    scores, labels = batch["y_pred"], batch["y_true"]
    categories = batch["categories"]

    if not isinstance(scores, np.ndarray) or not isinstance(labels, np.ndarray):
        raise TypeError("y_pred and y_true must be NumPy arrays")

    if scores.ndim != 3 or scores.shape != labels.shape or 0 in scores.shape:
        raise ValueError("Expected nonempty matching [N, H, W] arrays")

    if scores.dtype != np.float32 or labels.dtype != np.bool_:
        raise ValueError("y_pred must be float32 and y_true must be boolean")

    if isinstance(categories, str) or len(categories) != len(scores):
        raise ValueError("Expected one category per image")

    if any(not isinstance(name, str) or not name or name == "macro_avg" for name in categories):
        raise ValueError("Categories must be nonempty strings other than 'macro_avg'")

    masks = None
    if postprocessor is not None:
        masks = postprocessor(scores)

        if (
            not isinstance(masks, np.ndarray)
            or masks.shape != scores.shape
            or masks.dtype != np.bool_
        ):
            raise ValueError("Postprocessor must return matching boolean masks")

    return scores, labels, np.asarray(categories), masks


def _macro_average(rows: dict[str, dict[str, float]]) -> dict[str, float]:
    averages = {}

    for metric in next(iter(rows.values())):
        values = [row[metric] for row in rows.values()]
        valid = [value for value in values if np.isfinite(value)]

        averages[metric] = float(np.mean(valid)) if valid else float("nan")

    return averages


def evaluate(
    batches: Iterable[EvaluationBatch],
    *,
    postprocessor: Postprocessor | None = None,
) -> dict[str, dict[str, dict[str, float]]]:
    """Return per-category and macro-averaged continuous/binary metrics."""
    accumulators: dict[str, Accumulator] = {}

    for batch in batches:
        scores, labels, categories, masks = _prepare_batch(batch, postprocessor)

        for category in np.unique(categories):
            if category not in accumulators:
                accumulators[category] = Accumulator()

            selected = categories == category
            accumulators[category].add(
                scores[selected],
                labels[selected],
                masks[selected] if masks is not None else None,
            )

    if not accumulators:
        raise ValueError("Evaluation requires at least one batch")

    results = {
        category: accumulator.result(postprocessor is not None)
        for category, accumulator in accumulators.items()
    }

    report = {}

    for section in ("continuous", "binary"):
        if section == "binary" and postprocessor is None:
            continue

        rows = {category: metrics[section] for category, metrics in results.items()}
        rows["macro_avg"] = _macro_average(rows)
        report[section] = rows

    return report
