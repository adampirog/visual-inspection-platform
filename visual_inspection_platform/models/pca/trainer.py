from collections.abc import Iterator
from typing import Literal

from pydantic import PositiveInt
from sklearn.decomposition import PCA

from visual_inspection_platform.data import ImageRecordsDataset
from visual_inspection_platform.evaluation import EvaluationBatch
from visual_inspection_platform.evaluation import evaluate as evaluate_predictions
from visual_inspection_platform.training import Trainer, TrainerConfig

from .model import PCADetector


class PCAConfig(TrainerConfig):
    n_components: PositiveInt = 32
    image_size: tuple[PositiveInt, PositiveInt] = (64, 64)
    svd_solver: Literal["auto", "full", "arpack", "randomized"] = "auto"
    random_seed: int = 42


class PCATrainer(Trainer[PCADetector, PCAConfig]):
    """Train and evaluate a PCA reconstruction baseline."""

    def fit(self, data: ImageRecordsDataset) -> PCADetector:
        """Fit PCA on normal images from one object category."""
        if not data:
            raise ValueError("Training dataset is empty")

        if any(record.label != "normal" for record in data):
            raise ValueError("PCA training requires normal images only")

        if len({record.category for record in data}) != 1:
            raise ValueError("PCA training requires a single object category")

        images = data.load_images(self.config.image_size)
        features = images.reshape(len(data), -1)

        pca = PCA(
            n_components=self.config.n_components,
            svd_solver=self.config.svd_solver,
            random_state=self.config.random_seed,
        )
        pca.fit(features)

        return PCADetector(
            mean=pca.mean_,
            components=pca.components_,
            image_size=self.config.image_size,
            training_metadata=self.get_metadata(),
        )

    def _batches(
        self, model: PCADetector, data: ImageRecordsDataset, batch_size: int
    ) -> Iterator[EvaluationBatch]:
        """Generate precomputed prediction batches for evaluation."""
        for start in range(0, len(data), batch_size):
            batch = data[start : start + batch_size]

            if any(record.label != "normal" and record.mask is None for record in batch):
                raise ValueError("Anomalous image is missing a segmentation mask")

            images = batch.load_images(model.image_size)

            yield EvaluationBatch(
                y_pred=model.predict(images),
                y_true=batch.load_masks(model.image_size),
                categories=[record.category for record in batch],
            )

    def evaluate(
        self,
        model: PCADetector,
        data: ImageRecordsDataset,
        *,
        batch_size: int = 32,
        verbose: bool = False,
    ) -> dict:
        """Evaluate PCA anomaly maps using segmentation metrics."""
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        if not data:
            raise ValueError("Evaluation dataset is empty")

        return evaluate_predictions(
            self._batches(model, data, batch_size),
            postprocessor=model.postprocessor,
            verbose=verbose,
        )
