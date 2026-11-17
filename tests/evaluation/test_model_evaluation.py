import pytest

from visual_inspection_platform.evaluation.evaluate_model import evaluate_model
from visual_inspection_platform.models.pca.trainer import PCAConfig, PCATrainer
from visual_inspection_platform.postprocessors import ThresholdPostprocessor


def test_evaluate_model(dataset_factory):
    train = dataset_factory(split="train", objects="cashew")
    test = dataset_factory(split="test", objects="cashew")

    trainer = PCATrainer(PCAConfig(n_components=1, image_size=(4, 6)))
    model = trainer.fit(train)

    result = evaluate_model(model, test, batch_size=1)

    expected = {"pixel_auroc": 1.0, "pixel_ap": 1.0}

    assert set(result) == {"continuous"}
    assert result["continuous"]["cashew"] == pytest.approx(expected)

    model.postprocessor = ThresholdPostprocessor(0.25)

    result = evaluate_model(model, test, batch_size=2)

    assert result["continuous"]["cashew"] == pytest.approx(expected)
    assert result["binary"]["cashew"] == pytest.approx(
        {"precision": 1.0, "recall": 1.0, "dice": 1.0}
    )
