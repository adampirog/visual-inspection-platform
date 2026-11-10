import numpy as np


# Images

# Decoded RGB image: [H, W, 3], uint8 (0 - 255)
type RGBImage = np.ndarray[tuple[int, int, int], np.dtype[np.uint8]]

# Preprocessed model inputs: float32, channel-first
type ModelImage = np.ndarray[tuple[int, int, int], np.dtype[np.float32]]
type ImageBatch = np.ndarray[tuple[int, int, int, int], np.dtype[np.float32]]


# Anomaly predictions

# Continuous anomaly scores, larger means more anomalous
type AnomalyMap = np.ndarray[tuple[int, int], np.dtype[np.float32]]
type AnomalyMaps = np.ndarray[tuple[int, int, int], np.dtype[np.float32]]


# Ground-truth annotations

# Original categorical segmentation mask: [H, W]
type SegmentationMask = np.ndarray[tuple[int, int], np.dtype[np.uint8]]

# Binary segmentation masks: False=normal, True=anomaly
type BinaryMask = np.ndarray[tuple[int, int], np.dtype[np.bool_]]
type BinaryMasks = np.ndarray[tuple[int, int, int], np.dtype[np.bool_]]
