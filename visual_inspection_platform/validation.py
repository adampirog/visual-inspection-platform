import numpy as np

from .types import AnomalyMaps, AnomalyScores, ImageBatch, RGBImage


def validate_rgb_image(image: RGBImage) -> None:
    """Validate a decoded RGB image in HWC format."""
    if not isinstance(image, np.ndarray):
        raise TypeError("Image must be a NumPy array")

    if image.dtype != np.uint8:
        raise ValueError(f"Expected uint8 image, got {image.dtype}")

    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"Expected HWC RGB image, got {image.shape}")

    if min(image.shape[:2]) == 0:
        raise ValueError("Image dimensions must be positive")


def validate_image_batch(images: ImageBatch) -> None:
    """Validate model input in NCHW float32 format."""
    if not isinstance(images, np.ndarray):
        raise TypeError("Images must be a NumPy array")

    if images.dtype != np.float32:
        raise ValueError(f"Expected float32, got {images.dtype}")

    if images.ndim != 4:
        raise ValueError(f"Expected NCHW batch, got {images.shape}")

    n, c, h, w = images.shape

    if n == 0 or h == 0 or w == 0:
        raise ValueError("Batch and spatial dimensions must be positive")

    if c != 3:
        raise ValueError(f"Expected 3 channels, got {c}")


def validate_prediction(
    scores: AnomalyScores,
    anomaly_maps: AnomalyMaps | None,
    *,
    batch_shape: tuple[int, int, int, int],
) -> None:
    """Validate standardized model output against the input batch."""
    n, _, h, w = batch_shape

    if not isinstance(scores, np.ndarray):
        raise TypeError("Scores must be a NumPy array")

    if scores.dtype != np.float32 or scores.shape != (n,):
        raise ValueError(
            f"Expected float32 scores with shape {(n,)}, got {scores.shape}, {scores.dtype}"
        )

    if anomaly_maps is not None:
        if not isinstance(anomaly_maps, np.ndarray):
            raise TypeError("Anomaly maps must be a NumPy array")

        if anomaly_maps.dtype != np.float32 or anomaly_maps.shape != (n, h, w):
            raise ValueError(
                f"Expected float32 anomaly maps with shape {(n, h, w)}, "
                f"got {anomaly_maps.shape}, {anomaly_maps.dtype}"
            )
