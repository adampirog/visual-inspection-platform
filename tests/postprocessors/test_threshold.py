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

    np.testing.assert_array_equal(processor(maps), [[[False, False, True]]])


def test_global_calibration():
    maps = np.array([[[0.0, 0.25]], [[0.5, 0.75]]], dtype=np.float32)
    processor = ThresholdPostprocessor.calibrate(maps, quantile=0.5)

    assert processor.threshold == pytest.approx(0.375)


def test_spatial_calibration_and_roundtrip(tmp_path):
    maps = np.array([[[0.0, 0.5]], [[0.25, 0.75]]], dtype=np.float32)

    processor = SpatialThresholdPostprocessor.calibrate(maps, quantile=1.0, offset=0.125)

    np.testing.assert_array_equal(processor.thresholds, [[0.25, 0.75]])

    predictions = np.array([[[0.5, 0.875]]], dtype=np.float32)
    expected = [[[True, False]]]

    np.testing.assert_array_equal(processor(predictions), expected)

    manifest = processor.save(tmp_path)
    loaded = get_postprocessor_class(manifest.kind).load(tmp_path, manifest)

    assert loaded.offset == processor.offset  # type:ignore
    np.testing.assert_array_equal(loaded.thresholds, processor.thresholds)  # type:ignore
    np.testing.assert_array_equal(loaded(predictions), expected)


def test_postprocessor_kind_mismatch(tmp_path):
    processor = ThresholdPostprocessor(0.5)
    manifest = processor.save(tmp_path)

    with pytest.raises(ValueError, match="Postprocessor kind mismatch"):
        SpatialThresholdPostprocessor.load(tmp_path, manifest)
