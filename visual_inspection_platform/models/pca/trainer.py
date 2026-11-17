from pydantic import PositiveInt
from sklearn.decomposition import PCA

from visual_inspection_platform.data import ImageRecordsDataset
from visual_inspection_platform.training import Trainer, TrainerConfig

from .model import PCADetector


class PCAConfig(TrainerConfig):
    n_components: PositiveInt = 32
    image_size: tuple[PositiveInt, PositiveInt] = (64, 64)
    svd_solver: str = "auto"
    random_seed: int = 42


class PCATrainer(Trainer[PCADetector, PCAConfig]):
    """Train and evaluate a PCA reconstruction baseline."""

    def fit(self, data: ImageRecordsDataset) -> PCADetector:
        """Fit PCA on normal images from one object category."""

        if any(record.is_anomaly for record in data):
            raise ValueError("PCA training requires anomaly-free data")

        images = data.load_images(self.config.image_size)
        features = images.reshape(len(data), -1)

        pca = PCA(
            n_components=self.config.n_components,
            svd_solver=self.config.svd_solver,  # type:ignore specific scikit solvers
            random_state=self.config.random_seed,
        )
        pca.fit(features)

        return PCADetector(
            mean=pca.mean_,
            components=pca.components_,
            image_size=self.config.image_size,
            training_metadata=self.get_metadata(),
        )
