import numpy as np
import pytest

from visual_inspection_platform.evaluation.evaluation import EvaluationBatch, evaluate
from visual_inspection_platform.postprocessors import Postprocessor


class ThresholdPostprocessor(Postprocessor):
    KIND = "threshold"
    VERSION = "1"

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self.calls = 0

    def __call__(self, maps):
        self.calls += 1
        return maps >= self.threshold

    def get_params(self) -> dict:
        return {"threshold": self.threshold}


@pytest.fixture
def batches():
    return [
        {
            "y_pred": np.array([[[0.9, 0.1]], [[0.2, 0.8]]], dtype=np.float32),
            "y_true": np.array([[[True, False]], [[True, False]]]),
            "categories": ["cashew", "pcb1"],
        },
        {
            "y_pred": np.array([[[0.8, 0.2]]], dtype=np.float32),
            "y_true": np.array([[[True, False]]]),
            "categories": ["cashew"],
        },
    ]


def test_continuous_report(batches):
    result = evaluate(batches)

    assert set(result) == {"continuous"}

    rows = result["continuous"]
    assert set(rows) == {"cashew", "pcb1", "macro_avg"}

    assert rows["cashew"] == pytest.approx({"pixel_auroc": 1.0, "pixel_ap": 1.0})
    assert rows["pcb1"] == pytest.approx({"pixel_auroc": 0.0, "pixel_ap": 0.5})
    assert rows["macro_avg"] == pytest.approx({"pixel_auroc": 0.5, "pixel_ap": 0.75})


def test_binary_report(batches):
    processor = ThresholdPostprocessor()
    result = evaluate(batches, postprocessor=processor)

    assert set(result) == {"continuous", "binary"}

    rows = result["binary"]

    assert rows["cashew"] == pytest.approx({"precision": 1.0, "recall": 1.0, "dice": 1.0})
    assert rows["pcb1"] == pytest.approx({"precision": 0.0, "recall": 0.0, "dice": 0.0})
    assert rows["macro_avg"] == pytest.approx({"precision": 0.5, "recall": 0.5, "dice": 0.5})

    assert processor.calls == len(batches)


def test_global_ranking_across_batches():
    batches: list[EvaluationBatch] = [
        {
            "y_pred": np.array([[[0.9, 0.8]]], dtype=np.float32),
            "y_true": np.array([[[True, False]]]),
            "categories": ["cashew"],
        },
        {
            "y_pred": np.array([[[0.2, 0.1]]], dtype=np.float32),
            "y_true": np.array([[[True, False]]]),
            "categories": ["cashew"],
        },
    ]

    metrics = evaluate(batches)["continuous"]["cashew"]

    assert metrics == pytest.approx({"pixel_auroc": 0.75, "pixel_ap": 5 / 6})


def test_binary_aggregation_across_batches():
    batches: list[EvaluationBatch] = [
        {
            "y_pred": np.array([[[0.9, 0.8]]], dtype=np.float32),
            "y_true": np.array([[[True, False]]]),
            "categories": ["cashew"],
        },
        {
            "y_pred": np.array([[[0.1, 0.8]]], dtype=np.float32),
            "y_true": np.array([[[True, False]]]),
            "categories": ["cashew"],
        },
    ]

    result = evaluate(batches, postprocessor=ThresholdPostprocessor())
    metrics = result["binary"]["cashew"]

    assert metrics == pytest.approx(
        {
            "precision": 1 / 3,
            "recall": 0.5,
            "dice": 0.4,
        }
    )


def test_macro_average_ignores_undefined_auc():
    batches: list[EvaluationBatch] = [
        {
            "y_pred": np.array([[[0.9, 0.1]], [[0.2, 0.3]]], dtype=np.float32),
            "y_true": np.array([[[True, False]], [[False, False]]]),
            "categories": ["cashew", "pcb1"],
        }
    ]

    rows = evaluate(batches)["continuous"]

    assert np.isnan(rows["pcb1"]["pixel_auroc"])
    assert rows["macro_avg"] == pytest.approx(
        {
            "pixel_auroc": 1.0,
            "pixel_ap": 0.5,
        }
    )


@pytest.mark.parametrize(
    ("shape", "dtype", "categories"),
    [
        ((1, 2), np.float32, ["cashew"]),
        ((1, 1, 2), np.float64, ["cashew"]),
        ((1, 1, 2), np.float32, ["cashew", "pcb1"]),
        ((1, 1, 2), np.float32, [""]),
        ((1, 1, 2), np.float32, ["macro_avg"]),
        ((1, 1, 2), np.float32, "cashew"),
    ],
)
def test_invalid_batch(shape, dtype, categories):
    batch = {
        "y_pred": np.zeros(shape, dtype=dtype),
        "y_true": np.zeros(shape, dtype=bool),
        "categories": categories,
    }

    with pytest.raises(ValueError):
        evaluate([batch])  # type:ignore


def test_mismatched_ground_truth():
    batch: EvaluationBatch = {
        "y_pred": np.zeros((2, 2, 2), dtype=np.float32),
        "y_true": np.zeros((1, 2, 2), dtype=bool),
        "categories": ["cashew", "pcb1"],
    }

    with pytest.raises(ValueError):
        evaluate([batch])


def test_invalid_ground_truth_dtype():
    batch = {
        "y_pred": np.zeros((1, 2, 2), dtype=np.float32),
        "y_true": np.zeros((1, 2, 2), dtype=np.uint8),
        "categories": ["cashew"],
    }

    with pytest.raises(ValueError):
        evaluate([batch])  # type:ignore


def test_missing_categories():
    batch = {
        "y_pred": np.zeros((1, 2, 2), dtype=np.float32),
        "y_true": np.zeros((1, 2, 2), dtype=bool),
    }

    with pytest.raises(KeyError, match="categories"):
        evaluate([batch])  # type:ignore
