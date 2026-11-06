import numpy as np


# Convention - [batch_size, channels, height, width]


# Decoded image: [H, W, C], RGB, values 0-255
type RGBImage = np.ndarray[tuple[int, int, int], np.dtype[np.uint8]]

# Preprocessed model inputs: float32, channel-first
type ModelImage = np.ndarray[tuple[int, int, int], np.dtype[np.float32]]
type ImageBatch = np.ndarray[tuple[int, int, int, int], np.dtype[np.float32]]

# Model outputs
type AnomalyScores = np.ndarray[tuple[int], np.dtype[np.float32]]
type AnomalyMaps = np.ndarray[tuple[int, int, int], np.dtype[np.float32]]

# Ground-truth annotations
type SegmentationMask = np.ndarray[tuple[int, int], np.dtype[np.uint8]]
type BinaryMask = np.ndarray[tuple[int, int], np.dtype[np.bool_]]
