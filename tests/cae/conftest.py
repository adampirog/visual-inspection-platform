import numpy as np
import pytest
import torch

from visual_inspection_platform.models import TrainingMetadata
from visual_inspection_platform.models.cae import CAEDetector
from visual_inspection_platform.models.cae.network import (
    CAEAnomalyModule,
    ConvolutionalAutoencoder,
)
from visual_inspection_platform.models.cae.trainer import CAEConfig, CAETrainer
from visual_inspection_platform.onnx import export_anomaly_detector
from visual_inspection_platform.postprocessors import ThresholdPostprocessor


@pytest.fixture(scope="session")
def cae_config():
    return CAEConfig(
        image_size=(32, 48),
        latent_channels=8,
        batch_size=2,
        max_epochs=1,
        random_state=42,
        accelerator="cpu",
    )


@pytest.fixture
def trainer(cae_config):
    return CAETrainer(cae_config)


@pytest.fixture(scope="session")
def network(cae_config):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)

        return ConvolutionalAutoencoder(latent_channels=cae_config.latent_channels).eval()


@pytest.fixture(scope="session")
def image_batch(cae_config):
    return np.random.default_rng(42).random((8, 3, *cae_config.image_size), dtype=np.float32)


@pytest.fixture(scope="session")
def onnx_model(network, cae_config):
    return export_anomaly_detector(
        CAEAnomalyModule(network).eval(), image_size=cae_config.image_size
    )


@pytest.fixture(scope="session")
def detector(onnx_model, cae_config):
    return CAEDetector(
        onnx_model=onnx_model,
        image_size=cae_config.image_size,
        training_metadata=TrainingMetadata(
            trainer="test_trainer",
            config=cae_config.model_dump(mode="json"),
        ),
        postprocessor=ThresholdPostprocessor(0.1),
    )
