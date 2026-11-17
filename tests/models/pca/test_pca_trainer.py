import numpy as np
import pytest

from visual_inspection_platform.models.pca.trainer import PCAConfig, PCATrainer


@pytest.fixture
def trainer():
    return PCATrainer(PCAConfig(n_components=1, image_size=(4, 6)))


def test_fit(trainer, dataset_factory):
    train = dataset_factory(split="train", objects="cashew")
    model = trainer.fit(train)

    assert model.image_size == (4, 6)
    assert model.components.shape == (1, 72)
    assert model.training_metadata.config == trainer.config.model_dump(mode="json")

    maps = model.predict(train.load_images(model.image_size))

    np.testing.assert_allclose(maps, 0.0, atol=1e-6)


def test_anomalous_training_data(trainer, dataset_factory):
    dataset = dataset_factory(split="test")

    with pytest.raises(ValueError, match="anomaly-free"):
        trainer.fit(dataset)


def test_config_roundtrip(trainer, tmp_path):
    path = tmp_path / "config.json"

    trainer.config.save(path)

    assert PCAConfig.load(path) == trainer.config
