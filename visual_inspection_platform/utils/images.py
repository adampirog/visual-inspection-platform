from pathlib import Path

import numpy as np
from PIL import Image

from visual_inspection_platform.types import BinaryMask, RGBImage


def load_rgb_image(path: str | Path, size: tuple[int, int]) -> RGBImage:
    """Load and resize an RGB image to (height, width)."""
    height, width = size

    with Image.open(path) as image:
        image = image.convert("RGB")
        image = image.resize((width, height), Image.Resampling.BILINEAR)
        return np.asarray(image, dtype=np.uint8)


def load_binary_mask(path: str | Path | None, size: tuple[int, int]) -> BinaryMask:
    """Load a binary mask; missing masks represent normal images."""
    height, width = size

    if path is None:
        return np.zeros((height, width), dtype=np.bool_)

    with Image.open(path) as image:
        image = image.convert("L")
        image = image.resize((width, height), Image.Resampling.NEAREST)
        return np.asarray(image) != 0
