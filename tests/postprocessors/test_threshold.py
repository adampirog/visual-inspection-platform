import numpy as np
import pytest

from visual_inspection_platform.postprocessors import (
    SpatialThresholdPostprocessor,
    ThresholdPostprocessor,
    get_postprocessor_class,
)


def test_global_threshold():
    maps = np.array([[[0.1, 0.5, 0.9]]], dtype=np.float32)
    processor = ThresholdPostprocessor(0.5)

    np.testing.assert_array_equal(
        processor(maps),
        [[[False, False, True]]],
    )


def test_global_calibration():
    normal_maps = np.array([[[0.1, 0.2]], [[0.3, 0.4]]], dtype=np.float32)
    processor = ThresholdPostprocessor.calibrate(normal_maps, quantile=1.0)

    assert processor.threshold == pytest.approx(0.4)


def test_spatial_calibration():
    normal_maps = np.array([[[0.1, 0.5]], [[0.2, 0.8]]], dtype=np.float32)
    processor = SpatialThresholdPostprocessor.calibrate(normal_maps, quantile=1.0)

    maps = np.array([[[0.3, 0.6]]], dtype=np.float32)

    np.testing.assert_array_equal(
        processor(maps),
        [[[True, False]]],
    )


@pytest.mark.parametrize(
    "processor",
    [
        ThresholdPostprocessor(0.4),
        SpatialThresholdPostprocessor(
            thresholds=np.array([[0.2, 0.5]], dtype=np.float32),
            offset=0.1,
        ),
    ],
)
def test_roundtrip(processor, tmp_path):
    manifest = processor.save(tmp_path)
    loaded = get_postprocessor_class(manifest.kind).load(tmp_path, manifest)

    maps = np.array([[[0.35, 0.7]]], dtype=np.float32)
    np.testing.assert_array_equal(loaded(maps), processor(maps))
