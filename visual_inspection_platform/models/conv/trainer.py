from collections.abc import Sequence
from pathlib import Path
from typing import Literal, override

import torch
from lightning.pytorch import Trainer as LightningTrainer
from lightning.pytorch import seed_everything
from lightning.pytorch.callbacks import (
    Callback,
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from lightning.pytorch.loggers import CSVLogger
from pydantic import (
    Field,
    NonNegativeFloat,
    NonNegativeInt,
    PositiveFloat,
    PositiveInt,
    field_validator,
)
from torch.utils.data import DataLoader, Dataset

from visual_inspection_platform.data import ImageRecordsDataset
from visual_inspection_platform.data.pytorch_data import ImageTensorDataset
from visual_inspection_platform.onnx import export_anomaly_detector
from visual_inspection_platform.training import Trainer, TrainerConfig
from visual_inspection_platform.utils import generate_model_id

from .lightning_module import CAELightningModule
from .model import CAEDetector
from .network import CAEAnomalyModule, ConvolutionalAutoencoder


class CAEConfig(TrainerConfig):
    # Architecture
    latent_channels: PositiveInt = 64

    # Optimization
    learning_rate: PositiveFloat = 1e-3
    weight_decay: NonNegativeFloat = 1e-4
    scheduler_factor: float = Field(default=0.5, gt=0, lt=1)
    scheduler_patience: NonNegativeInt = 5

    # Training
    image_size: tuple[PositiveInt, PositiveInt] = (128, 128)
    batch_size: PositiveInt = 32
    max_epochs: PositiveInt = 100
    validation_fraction: float = Field(default=0.15, gt=0, lt=1)
    early_stopping_patience: PositiveInt = 15
    early_stopping_min_delta: NonNegativeFloat = 1e-5
    gradient_clip_val: PositiveFloat | None = None
    accumulate_grad_batches: PositiveInt = 1
    random_state: NonNegativeInt | None = None

    # Execution
    accelerator: Literal["auto", "cpu", "cuda", "mps"] = "auto"
    precision: Literal["32-true", "16-mixed", "bf16-mixed"] = "32-true"
    num_workers: NonNegativeInt = 0
    deterministic: bool = False

    @field_validator("image_size")
    @classmethod
    def validate_image_size(cls, size: tuple[int, int]) -> tuple[int, int]:
        if any(dimension < 32 or dimension % 16 for dimension in size):
            raise ValueError("Image dimensions must be at least 32 and divisible by 16")

        return size


class AutoencoderTrainer(Trainer[CAEDetector, CAEConfig]):
    @override
    def fit(
        self,
        data: ImageRecordsDataset,
        *,
        validation_data: ImageRecordsDataset | None = None,
        run_dir: str | Path | None = None,
        callbacks: Sequence[Callback] = (),
    ) -> CAEDetector:
        """Train and export a convolutional autoencoder detector.

        Training and validation data must be anomaly-free and disjoint.
        If validation_data is omitted, a seeded random split is created.
        The returned detector has no configured postprocessor.
        """
        model_id = generate_model_id(model_family=CAEDetector.MODEL_FAMILY)

        if run_dir is None:
            run_dir = Path("runs") / model_id

        network = self.fit_network(
            data, validation_data=validation_data, run_dir=run_dir, callbacks=callbacks
        )

        return CAEDetector(
            onnx_model=export_anomaly_detector(CAEAnomalyModule(network), self.config.image_size),
            image_size=self.config.image_size,
            model_id=model_id,
            training_metadata=self.get_metadata(),
        )

    def fit_network(
        self,
        data: ImageRecordsDataset,
        *,
        validation_data: ImageRecordsDataset | None = None,
        run_dir: str | Path,
        callbacks: Sequence[Callback] = (),
    ) -> ConvolutionalAutoencoder:
        """Train and restore the best network in FP32 evaluation mode on CPU.

        Training and validation data must be anomaly-free and disjoint.
        If validation_data is omitted, a random split is created.
        Checkpoint selection uses validation reconstruction loss.
        The run directory must not already exist.
        """
        config = self.config

        datasets = (data,) if validation_data is None else (data, validation_data)

        for dataset in datasets:
            if not dataset:
                raise ValueError("Training and validation datasets must not be empty")

            if any(record.is_anomaly for record in dataset):
                raise ValueError("Autoencoder training requires anomaly-free data")

        if config.random_state is not None:
            seed_everything(config.random_state, workers=True)

        run_dir = Path(run_dir)
        run_dir.mkdir(parents=True, exist_ok=False)
        config.save(run_dir / "config.json")

        train_loader, val_loader = self._create_dataloaders(data, validation_data)

        module = CAELightningModule(
            latent_channels=config.latent_channels,
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
            scheduler_factor=config.scheduler_factor,
            scheduler_patience=config.scheduler_patience,
        )

        checkpoint = ModelCheckpoint(
            dirpath=run_dir / "checkpoints",
            filename="{epoch:03d}-{val_loss:.6f}",
            monitor="val_loss",
            mode="min",
            save_top_k=1,
        )

        trainer = LightningTrainer(
            accelerator=config.accelerator,
            devices=1,
            precision=config.precision,
            max_epochs=config.max_epochs,
            deterministic=config.deterministic,
            gradient_clip_val=config.gradient_clip_val,
            accumulate_grad_batches=config.accumulate_grad_batches,
            default_root_dir=str(run_dir),
            logger=CSVLogger(save_dir=str(run_dir), name="metrics"),
            callbacks=[
                checkpoint,
                EarlyStopping(
                    monitor="val_loss",
                    mode="min",
                    patience=config.early_stopping_patience,
                    min_delta=config.early_stopping_min_delta,
                ),
                LearningRateMonitor(logging_interval="epoch"),
                *callbacks,
            ],
        )

        trainer.fit(module, train_dataloaders=train_loader, val_dataloaders=val_loader)

        if not checkpoint.best_model_path:
            raise RuntimeError("Training did not produce a valid checkpoint")

        best_module = CAELightningModule.load_from_checkpoint(
            checkpoint.best_model_path, map_location="cpu"
        )

        return best_module.network.cpu().float().eval()

    def _create_dataloaders(
        self,
        data: ImageRecordsDataset,
        validation_data: ImageRecordsDataset | None,
    ) -> tuple[DataLoader[torch.Tensor], DataLoader[torch.Tensor]]:
        config = self.config
        dataset = ImageTensorDataset(data, config.image_size)

        train: Dataset[torch.Tensor]
        val: Dataset[torch.Tensor]

        if validation_data is None:
            train, val = dataset.split(config.validation_fraction, random_state=config.random_state)
        else:
            train = dataset
            val = ImageTensorDataset(validation_data, config.image_size)

        return (
            self._create_dataloader(train, shuffle=True),
            self._create_dataloader(val, shuffle=False),
        )

    def _create_dataloader(
        self,
        dataset: Dataset[torch.Tensor],
        *,
        shuffle: bool,
    ) -> DataLoader[torch.Tensor]:
        config = self.config

        pin_memory = config.accelerator in ("auto", "cuda") and torch.cuda.is_available()

        rng = (
            torch.Generator().manual_seed(config.random_state)
            if config.random_state is not None
            else torch.default_generator
        )

        return DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=shuffle,
            num_workers=config.num_workers,
            pin_memory=pin_memory,
            persistent_workers=config.num_workers > 0,
            generator=rng,
        )
