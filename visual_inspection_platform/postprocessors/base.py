from abc import ABC, abstractmethod
from pathlib import Path
from typing import ClassVar, Self

from visual_inspection_platform.types import AnomalyMaps, BinaryMasks
from visual_inspection_platform.utils import generate_model_id

from .manifest import PostprocessorManifest


class Postprocessor(ABC):
    """Base interface for serializable anomaly prediction models."""

    MODEL_FAMILY: ClassVar[str]
    ARTIFACT_VERSION: ClassVar[str]

    def __init__(self, *, model_id: str | None = None) -> None:
        if model_id is None:
            self.model_id = generate_model_id(model_family=type(self).MODEL_FAMILY)
        else:
            self.model_id = self.model_id

    @abstractmethod
    def __call__(self, maps: AnomalyMaps) -> BinaryMasks:
        pass

    @abstractmethod
    def _save(self, path: Path) -> None: ...

    def save(self, path: str | Path) -> None:
        """Serialize the fitted model to disk."""

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        self._save(path)

        PostprocessorManifest(
            model_id=self.model_id,
            model_family=self.MODEL_FAMILY,
            artifact_version=self.ARTIFACT_VERSION,
        ).save(path / "MANIFEST.json")

    @classmethod
    @abstractmethod
    def _load(cls, path: Path, manifest: PostprocessorManifest) -> Self: ...

    @classmethod
    def load(cls, path: str | Path) -> Self:
        """Load a serialized Inspector from disk."""

        path = Path(path)
        manifest = PostprocessorManifest.load(path / "MANIFEST.json")

        if manifest.model_family != cls.MODEL_FAMILY:
            raise ValueError(
                "Cannot load saved postprocessor with "
                f"Manifest(model_family={manifest.model_family}) "
                f"with class {cls.__name__}(model_family={cls.MODEL_FAMILY})."
            )

        model = cls._load(path, manifest=manifest)

        if model.model_id != manifest.model_id:
            raise RuntimeError("'model_id' was overridden during loading.")

        return model
