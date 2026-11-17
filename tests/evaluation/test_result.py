import json

import pytest

from visual_inspection_platform.evaluation.result import EvaluationResult


def test_report(tmp_path):
    result = EvaluationResult(
        {
            "continuous": {
                "cashew": {"pixel_auroc": 0.95, "pixel_ap": 0.8},
                "macro_avg": {"pixel_auroc": 0.95, "pixel_ap": 0.8},
            }
        }
    )

    output = str(result)

    assert "Pixel AUROC" in output
    assert "Macro avg" in output
    assert "0.9500" in output

    text_path = tmp_path / "evaluation.txt"
    json_path = tmp_path / "evaluation.json"

    result.save(text_path)
    result.save(json_path)

    assert text_path.read_text(encoding="utf-8") == output
    assert json.loads(json_path.read_text(encoding="utf-8")) == result.data


def test_unsupported_format(tmp_path):
    with pytest.raises(ValueError, match="not supported"):
        EvaluationResult().save(tmp_path / "evaluation.csv")


def test_undefined_metrics(tmp_path):
    result = EvaluationResult(
        {"continuous": {"cashew": {"pixel_auroc": float("nan"), "pixel_ap": 0.0}}}
    )

    assert "N/A" in str(result)

    path = tmp_path / "evaluation.json"
    result.save(path)

    metrics = json.loads(path.read_text(encoding="utf-8"))
    assert metrics["continuous"]["cashew"]["pixel_auroc"] is None
