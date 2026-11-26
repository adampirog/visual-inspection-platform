from typing import override

import torch
from lightning.pytorch import LightningModule
from lightning.pytorch.utilities.types import OptimizerLRSchedulerConfig
from torch.nn import functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from .network import ConvolutionalAutoencoder


class AutoencoderLightningModule(LightningModule):
    """Train a convolutional autoencoder using reconstruction loss.

    Training and validation batches contain only input images.
    """

    def __init__(
        self,
        latent_channels: int,
        learning_rate: float,
        weight_decay: float,
        scheduler_factor: float,
        scheduler_patience: int,
    ) -> None:
        super().__init__()

        self.save_hyperparameters()

        self.network = ConvolutionalAutoencoder(latent_channels=latent_channels)

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.scheduler_factor = scheduler_factor
        self.scheduler_patience = scheduler_patience

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.network(images)

    def _reconstruction_loss(self, images: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(self(images), images)

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        loss = self._reconstruction_loss(batch)

        self.log(
            "train_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch.size(0),
        )

        return loss

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> None:
        loss = self._reconstruction_loss(batch)

        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch.size(0),
        )

    @override
    def configure_optimizers(self) -> OptimizerLRSchedulerConfig:
        optimizer = AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=self.scheduler_factor,
            patience=self.scheduler_patience,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"},
        }
