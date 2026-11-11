import json
import math

import pytest

from visual_inspection_platform.evaluation.result import EvaluationResult


@pytest.fixture
def result():
    return EvaluationResult(
        {
            "continuous": {
                "cashew": {"pixel_auroc": 0.95, "pixel_ap": 0.8},
                "pcb1": {"pixel_auroc": 0.85, "pixel_ap": 0.6},
                "macro_avg": {"pixel_auroc": 0.9, "pixel_ap": 0.7},
            },
            "binary": {
                "cashew": {"precision": 0.8, "recall": 0.6, "dice": 0.6857},
                "macro_avg": {"precision": 0.8, "recall": 0.6, "dice": 0.6857},
            },
        }
    )


def test_mapping(result):
    assert result["continuous"]["cashew"]["pixel_auroc"] == 0.95
    assert result["binary"]["cashew"]["dice"] == 0.6857


def test_printable_report(result):
    output = str(result)

    assert "Continuous segmentation" in output
    assert "Binary segmentation" in output
    assert "Pixel AUROC" in output
    assert "Precision" in output
    assert "cashew" in output
    assert "Macro avg" in output
    assert "0.6857" in output


def test_save_json(result, tmp_path):
    path = tmp_path / "evaluation.json"
    result.save(path)

    assert json.loads(path.read_text(encoding="utf-8")) == result.data


def test_save_text(result, tmp_path):
    path = tmp_path / "evaluation.txt"
    result.save(path)

    assert path.read_text(encoding="utf-8") == str(result)


def test_undefined_metrics(tmp_path):
    result = EvaluationResult(
        {
            "continuous": {
                "cashew": {"pixel_auroc": float("nan"), "pixel_ap": 0.0},
            }
        }
    )

    assert "N/A" in str(result)

    path = tmp_path / "evaluation.json"
    result.save(path)

    auroc = json.loads(path.read_text(encoding="utf-8"))["continuous"]["cashew"]["pixel_auroc"]
    assert math.isnan(auroc)


def test_unsupported_format(result, tmp_path):
    with pytest.raises(ValueError, match="not supported"):
        result.save(tmp_path / "evaluation.csv")


def test_empty_result():
    result = EvaluationResult()

    assert str(result) == "{}"
