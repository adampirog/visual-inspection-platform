from dataclasses import dataclass
from logging import getLogger
from pathlib import Path
from threading import Lock

from pydantic import ValidationError

from visual_inspection_platform.models import AnomalyDetector, ModelManifest

from .registry import ModelRegistry


logger = getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ModelInfo:
    """Serving metadata for a discovered model artifact."""

    model_id: str
    model_family: str
    path: str
    loaded: bool
    image_size: tuple[int, int]
    has_postprocessor: bool


class ServableModel:
    """
    Model artifact with eagerly loaded metadata and lazily loaded parameters.

    Loading and unloading cached state are synchronized.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).resolve()
        self.manifest = ModelManifest.load(self.path / "MANIFEST.json")

        self._model: AnomalyDetector | None = None
        self._lock = Lock()

    @property
    def model_id(self) -> str:
        return self.manifest.model_id

    @property
    def is_loaded(self) -> bool:
        with self._lock:
            return self._model is not None

    @property
    def info(self) -> ModelInfo:
        with self._lock:
            loaded = self._model is not None

        return ModelInfo(
            model_id=self.manifest.model_id,
            model_family=self.manifest.model_family,
            path=str(self.path),
            loaded=loaded,
            has_postprocessor=self.manifest.postprocessor is not None,
            image_size=self.manifest.image_size,
        )

    def load(self, model_class: type[AnomalyDetector]) -> AnomalyDetector:
        """Load the model on first access and return the cached instance thereafter."""
        with self._lock:
            if self._model is None:
                self._model = model_class.load(self.path)

            return self._model

    def unload(self) -> None:
        """Release the cached model while keeping the artifact discoverable."""
        with self._lock:
            self._model = None


class ModelManager:
    """Manage discovered model artifacts and their in-memory instances."""

    def __init__(self, registry: ModelRegistry) -> None:
        self.registry = registry

        self._models: dict[str, ServableModel] = {}
        self._lock = Lock()

    def discover(self, path: str | Path) -> tuple[str, ...]:
        """
        Recursively register new artifacts identified by MANIFEST.json.

        Unreadable or invalid manifests are skipped and logged.
        Previously registered IDs remain unchanged.
        Returns the sorted IDs of newly registered models.
        """

        new = set()
        for manifest_path in Path(path).rglob("MANIFEST.json"):
            try:
                model = ServableModel(manifest_path.parent)
            except (OSError, ValidationError):
                logger.exception(f"Could not read manifest: {manifest_path}")
                continue

            with self._lock:
                if model.model_id in self._models:
                    continue

                self._models[model.model_id] = model

            new.add(model.model_id)

        return tuple(sorted(new))

    def load(self, model_id: str) -> AnomalyDetector:
        """
        Load or retrieve a cached model by ID.

        Raises ModelNotFoundError for unknown IDs.
        Artifact deserialization errors propagate to the caller.
        """
        with self._lock:
            try:
                servable = self._models[model_id]
            except KeyError as exc:
                raise ModelNotFoundError(model_id) from exc

        try:
            model_class = self.registry[servable.manifest.model_family]
        except KeyError as exc:
            raise RuntimeError(
                f"Model family {servable.manifest.model_family!r} is not registered."
            ) from exc

        return servable.load(model_class)

    def unload(self, model_id: str) -> None:
        """
        Release the cached model while retaining its registration.

        Unknown model IDs are ignored.
        """
        with self._lock:
            model = self._models.get(model_id)

        if model is not None:
            model.unload()

    def is_loaded(self, model_id: str) -> bool:
        """Return whether the requested model is currently loaded."""
        with self._lock:
            model = self._models.get(model_id)

        return model.is_loaded if model is not None else False

    def available_models(self) -> tuple[ModelInfo, ...]:
        """Return metadata for all discovered models."""
        with self._lock:
            models = tuple(self._models.values())

        return tuple(model.info for model in models)

    @property
    def n_models(self) -> int:
        with self._lock:
            return len(self._models)

    @property
    def n_loaded(self) -> int:
        with self._lock:
            models = tuple(self._models.values())

        return sum(model.is_loaded for model in models)


class ModelNotFoundError(KeyError):
    pass
