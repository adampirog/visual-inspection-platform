from abc import ABC, abstractmethod
from pathlib import Path
from typing import Self

from pydantic import BaseModel, ConfigDict

from visual_inspection_platform.data import ImageRecordsDataset
from visual_inspection_platform.models import AnomalyDetector, TrainingMetadata


class TrainerConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    @classmethod
    def load(cls, path: str | Path) -> Self:
        with open(path, encoding="utf-8") as handle:
            return cls.model_validate_json(handle.read())

    def save(self, path: str | Path) -> None:
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(self.model_dump_json(indent=2))


class Trainer[DetectorType: AnomalyDetector, ConfigType: TrainerConfig](ABC):
    def __init__(self, config: ConfigType) -> None:
        self.config = config

    @abstractmethod
    def fit(self, data: ImageRecordsDataset) -> DetectorType:
        """Return a fitted detector; training data must contain only normal images."""

    def get_metadata(self) -> TrainingMetadata:
        """Describe the trainer by its fully qualified class name and JSON configuration."""
        return TrainingMetadata(
            trainer=f"{type(self).__module__}.{type(self).__qualname__}",
            config=self.config.model_dump(mode="json"),
        )
