import numpy as np
import pytest

from visual_inspection_platform.evaluation.metrics import (
    dice,
    pixel_ap,
    pixel_auroc,
    precision,
    recall,
)


@pytest.mark.parametrize(
    ("pred", "true", "expected"),
    [
        ([1, 1, 0, 0], [1, 1, 0, 0], (1.0, 1.0, 1.0)),
        ([1, 0, 1, 0], [1, 1, 0, 0], (0.5, 0.5, 0.5)),
        ([0, 0, 0, 0], [1, 1, 0, 0], (0.0, 0.0, 0.0)),
        ([0, 0, 0, 0], [0, 0, 0, 0], (0.0, 0.0, 0.0)),
    ],
)
def test_binary_metrics(pred, true, expected):
    y_pred = np.array(pred, dtype=bool).reshape(1, 2, 2)
    y_true = np.array(true, dtype=bool).reshape(1, 2, 2)

    assert precision(y_pred, y_true) == pytest.approx(expected[0])
    assert recall(y_pred, y_true) == pytest.approx(expected[1])
    assert dice(y_pred, y_true) == pytest.approx(expected[2])


@pytest.mark.parametrize(
    ("scores", "labels", "expected_auc", "expected_ap"),
    [
        ([0.9, 0.8, 0.2, 0.1], [1, 1, 0, 0], 1.0, 1.0),
        ([0.1, 0.2, 0.8, 0.9], [1, 1, 0, 0], 0.0, 5 / 12),
        ([0.9, 0.8, 0.3, 0.1], [1, 0, 1, 0], 0.75, 5 / 6),
        ([0.5, 0.5, 0.5, 0.5], [1, 0, 1, 0], 0.5, 0.5),
    ],
)
def test_ranking_metrics(scores, labels, expected_auc, expected_ap):
    y_pred = np.array(scores, dtype=np.float32).reshape(1, 2, 2)
    y_true = np.array(labels, dtype=bool).reshape(1, 2, 2)

    assert pixel_auroc(y_pred, y_true) == pytest.approx(expected_auc)
    assert pixel_ap(y_pred, y_true) == pytest.approx(expected_ap)


@pytest.mark.parametrize(
    ("labels", "expected_ap"),
    [
        ([0, 0], 0.0),
        ([1, 1], 1.0),
    ],
)
def test_single_class_ranking(labels, expected_ap):
    y_pred = np.array([[[0.2, 0.8]]], dtype=np.float32)
    y_true = np.array(labels, dtype=bool).reshape(1, 1, 2)

    assert np.isnan(pixel_auroc(y_pred, y_true))
    assert pixel_ap(y_pred, y_true) == pytest.approx(expected_ap)


def test_binary_shape_mismatch():
    y_pred = np.zeros((1, 2, 2), dtype=bool)
    y_true = np.zeros((1, 3, 3), dtype=bool)

    with pytest.raises(ValueError, match="shapes must match"):
        dice(y_pred, y_true)


def test_ranking_shape_mismatch():
    y_pred = np.zeros((1, 2, 2), dtype=np.float32)
    y_true = np.zeros((1, 3, 3), dtype=bool)

    with pytest.raises(ValueError, match="shapes must match"):
        pixel_auroc(y_pred, y_true)


@pytest.mark.parametrize("invalid_score", [np.nan, np.inf, -np.inf])
def test_nonfinite_scores(invalid_score):
    y_pred = np.array([[[0.1, invalid_score]]], dtype=np.float32)
    y_true = np.array([[[False, True]]])

    with pytest.raises(ValueError, match="finite"):
        pixel_auroc(y_pred, y_true)


def test_empty_ranking():
    y_pred = np.empty((0, 2, 2), dtype=np.float32)
    y_true = np.empty((0, 2, 2), dtype=bool)

    with pytest.raises(ValueError, match="empty"):
        pixel_ap(y_pred, y_true)
