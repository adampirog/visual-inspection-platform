from dataclasses import replace

import numpy as np
import pytest

from visual_inspection_platform.data import ImageRecordsDataset


def test_manifest_selection(dataset_factory):
    train = dataset_factory(split="train")
    test = dataset_factory(split="test", objects="cashew")

    assert len(train) == 4
    assert len(test) == 2
    assert [record.is_anomaly for record in test] == [False, True]
    assert test[1].mask.name == "mask.png"

    assert len(dataset_factory(objects={"pcb1"})) == 1
    assert len(dataset_factory(objects="unknown")) == 0


def test_batch_loading(dataset_factory):
    dataset = dataset_factory(split="test")
    batch = dataset[:2]

    assert isinstance(batch, ImageRecordsDataset)
    assert batch[0] is dataset[0]

    images = batch.load_images((4, 6))
    masks = batch.load_masks((4, 6))

    assert images.shape == (2, 3, 4, 6)
    assert images.dtype == np.float32
    assert images.flags.c_contiguous

    assert masks.shape == (2, 4, 6)
    assert masks.dtype == np.bool_

    assert not images[0].any()
    assert not masks[0].any()

    np.testing.assert_array_equal(images[1, :, 1, 2], 1.0)
    assert masks[1].sum() == 1
    assert masks[1, 1, 2]


def test_missing_anomaly_mask(dataset_factory):
    dataset = dataset_factory(split="test")
    record = replace(dataset[1], mask=None)

    with pytest.raises(ValueError, match="Missing anomaly mask"):
        record.load_mask((4, 6))
