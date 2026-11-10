from abc import ABC, abstractmethod
from pathlib import Path
from typing import Self

from pydantic import BaseModel, ConfigDict

from visual_inspection_platform.data import ImageRecordsDataset
from visual_inspection_platform.models import AnomalyDetector, TrainingMetadata


class TrainerConfig(BaseModel):
    """Configuration used to train a detector."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    @classmethod
    def load(cls, path: str | Path) -> Self:
        with open(path, encoding="utf-8") as handle:
            return cls.model_validate_json(handle.read())

    def save(self, path: str | Path) -> None:
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(self.model_dump_json(indent=2))


class Trainer[DetectorType: AnomalyDetector, ConfigType: TrainerConfig](ABC):
    """Train a model-specific detector."""

    def __init__(self, config: ConfigType) -> None:
        self.config = config

    @abstractmethod
    def fit(self, data: ImageRecordsDataset) -> DetectorType:
        """Fit a detector from the training dataset."""

    @abstractmethod
    def evaluate(
        self,
        model: DetectorType,
        data: ImageRecordsDataset,
        *,
        batch_size: int = 32,
        verbose: bool = False,
    ) -> dict:
        """Evaluate a detector on a given dataset"""

    def get_metadata(self) -> TrainingMetadata:
        return TrainingMetadata(
            trainer=f"{type(self).__module__}.{type(self).__qualname__}",
            config=self.config.model_dump(mode="json"),
        )
