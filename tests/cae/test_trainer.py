import numpy as np
import pytest
import torch

from visual_inspection_platform.models.cae import CAEDetector
from visual_inspection_platform.models.cae.lightning_module import CAELightningModule
from visual_inspection_platform.models.cae.trainer import CAEConfig


def test_fit_network(trainer, network, dataset_factory, tmp_path):
    data = dataset_factory(objects="cashew")
    run_dir = tmp_path / "run"

    trained = trainer.fit_network(data[:-1], run_dir=run_dir, validation_data=data[-1:])

    assert not trained.training

    parameters = list(trained.parameters())

    assert all(parameter.device.type == "cpu" for parameter in parameters)
    assert all(parameter.dtype == torch.float32 for parameter in parameters)
    assert all(torch.isfinite(parameter).all() for parameter in parameters)

    assert any(
        not torch.equal(before, after)
        for before, after in zip(network.parameters(), parameters, strict=True)
    )

    assert CAEConfig.load(run_dir / "config.json") == trainer.config

    (checkpoint,) = (run_dir / "checkpoints").glob("*.ckpt")

    restored = CAELightningModule.load_from_checkpoint(checkpoint, map_location="cpu")

    torch.testing.assert_close(trained.state_dict(), restored.network.state_dict())


def test_fit(trainer, dataset_factory):
    data = dataset_factory(objects="cashew")

    model = trainer.fit(data)

    assert isinstance(model, CAEDetector)
    assert model.image_size == trainer.config.image_size
    assert model.training_metadata == trainer.get_metadata()
    assert model.postprocessor is None

    images = data.load_images(model.image_size)
    maps = model.predict(images)

    assert maps.shape == (len(data), *model.image_size)
    assert maps.dtype == np.float32
    assert np.isfinite(maps).all()
    assert (maps >= 0).all()


@pytest.mark.parametrize("anomalous_validation", [False, True])
def test_anomalous_data(trainer, dataset_factory, tmp_path, anomalous_validation):
    normal = dataset_factory(split="train")
    anomalous = dataset_factory(split="test")

    train, val = (normal, anomalous) if anomalous_validation else (anomalous, normal)

    with pytest.raises(ValueError, match="anomaly-free"):
        trainer.fit_network(train, run_dir=tmp_path, validation_data=val)
