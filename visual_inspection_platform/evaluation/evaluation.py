"""Evaluate anomaly segmentation predictions by object category."""

from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import TypedDict

import numpy as np
from tqdm.auto import tqdm

from visual_inspection_platform.postprocessors import Postprocessor
from visual_inspection_platform.types import AnomalyMaps, BinaryMasks

from .metrics import _binary_counts, _binary_metrics, _ranking_metrics
from .result import EvaluationResult


MetricsDict = dict[str, float]


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

    def result(self, include_binary: bool) -> dict[str, MetricsDict]:
        cont = _ranking_metrics(np.concatenate(self.scores), np.concatenate(self.labels))
        cont = {"pixel_" + name: value for name, value in cont.items()}

        result = {"continuous": cont}

        if include_binary:
            result["binary"] = _binary_metrics(self.tp, self.fp, self.fn)

        return result


def _prepare_batch(
    batch: EvaluationBatch,
    postprocessor: Postprocessor | None,
) -> tuple[AnomalyMaps, BinaryMasks, np.ndarray, BinaryMasks | None]:
    scores, labels = batch["y_pred"], batch["y_true"]
    categories = batch["categories"]

    if scores.shape != labels.shape:
        raise ValueError("Prediction and ground-truth shapes must match")

    if labels.dtype != np.bool_:
        raise ValueError("Ground-truth masks must be boolean")

    if isinstance(categories, str) or len(categories) != len(scores):
        raise ValueError("Expected one category per image")

    if "macro_avg" in categories:
        raise ValueError("'macro_avg' is a reserved category name")

    masks = None if postprocessor is None else postprocessor(scores)

    return scores, labels, np.asarray(categories), masks


def _macro_average(rows: dict[str, MetricsDict]) -> MetricsDict:
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
    verbose: bool = False,
) -> EvaluationResult:
    """Return per-category and macro-averaged continuous/binary metrics."""
    accumulators = defaultdict(Accumulator)

    for batch in tqdm(batches, desc="Evaluating", disable=not verbose):
        scores, labels, categories, masks = _prepare_batch(batch, postprocessor)

        for category in np.unique(categories):
            selected = categories == category
            accumulators[category].add(
                scores[selected],
                labels[selected],
                masks[selected] if masks is not None else None,
            )

    results = {
        category: accumulator.result(include_binary=postprocessor is not None)
        for category, accumulator in accumulators.items()
    }

    report = {}

    for section in ("continuous", "binary"):
        if section == "binary" and postprocessor is None:
            continue

        rows = {category: metrics[section] for category, metrics in results.items()}
        rows["macro_avg"] = _macro_average(rows)
        report[section] = rows

    return EvaluationResult(report)
