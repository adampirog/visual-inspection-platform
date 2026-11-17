from collections.abc import Iterator

from visual_inspection_platform.data import ImageRecordsDataset
from visual_inspection_platform.models import AnomalyDetector

from .evaluation import EvaluationBatch, evaluate
from .result import EvaluationResult


def _batches(
    model: AnomalyDetector, data: ImageRecordsDataset, batch_size: int
) -> Iterator[EvaluationBatch]:
    for start in range(0, len(data), batch_size):
        batch = data[start : start + batch_size]

        yield EvaluationBatch(
            y_pred=model.predict(batch.load_images(model.image_size)),
            y_true=batch.load_masks(model.image_size),
            categories=[record.category for record in batch],
        )


def evaluate_model(
    model: AnomalyDetector,
    data: ImageRecordsDataset,
    *,
    batch_size: int = 32,
    verbose: bool = False,
) -> EvaluationResult:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    return evaluate(
        _batches(model, data, batch_size), postprocessor=model.postprocessor, verbose=verbose
    )
