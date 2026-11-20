import numpy as np

from .types import AnomalyMaps, ImageBatch, RGBImage


def validate_rgb_image(image: RGBImage) -> None:
    """Require a uint8 RGB array of shape [H, W, 3] with positive spatial dimensions."""
    if not isinstance(image, np.ndarray):
        raise TypeError("Image must be a NumPy array")

    if image.dtype != np.uint8:
        raise ValueError(f"Expected uint8 image, got {image.dtype}")

    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"Expected HWC RGB image, got {image.shape}")

    if min(image.shape[:2]) == 0:
        raise ValueError("Image dimensions must be positive")


def validate_image_batch(images: ImageBatch) -> None:
    """Require a nonempty float32 array of shape [N, 3, H, W].

    Pixel values are not checked.
    """
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


def validate_prediction(maps: AnomalyMaps, *, batch_shape: tuple[int, int, int, int]) -> None:
    """Require float32 anomaly maps of shape [N, H, W] matching the input batch.

    Score values are not checked.
    """
    if not isinstance(maps, np.ndarray):
        raise TypeError("Anomaly maps must be a NumPy array")

    n, _, h, w = batch_shape
    if maps.dtype != np.float32 or maps.shape != (n, h, w):
        raise ValueError(
            f"Expected float32 anomaly maps with shape {(n, h, w)}, got {maps.shape}, {maps.dtype}"
        )
