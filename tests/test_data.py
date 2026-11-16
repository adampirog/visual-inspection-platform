import numpy as np
import pytest
from PIL import Image

from visual_inspection_platform.data import ImageRecord, ImageRecordsDataset


@pytest.fixture
def image_dataset(tmp_path):
    image = np.zeros((2, 3, 3), dtype=np.uint8)
    image[:] = [255, 128, 0]

    mask = np.zeros((2, 3), dtype=np.uint8)
    mask[0, 1] = 255

    Image.fromarray(image).save(tmp_path / "normal.png")
    Image.fromarray(image).save(tmp_path / "anomaly.png")
    Image.fromarray(mask).save(tmp_path / "mask.png")

    manifest = tmp_path / "manifest.csv"
    manifest.write_text(
        "object,split,label,image,mask\n"
        "cashew,train,normal,normal.png,\n"
        "cashew,train,anomaly,anomaly.png,mask.png\n",
        encoding="utf-8",
    )

    label_map = {"normal": False, "anomaly": True}
    return ImageRecordsDataset(manifest, tmp_path, label_map=label_map)


@pytest.fixture
def manifest(tmp_path):
    path = tmp_path / "1cls.csv"
    path.write_text(
        "object,split,label,image,mask\n"
        "cashew,train,normal,cashew/normal/001.JPG,\n"
        "cashew,test,anomaly,cashew/anomaly/002.JPG,cashew/masks/002.png\n"
        "pcb1,train,normal,pcb1/normal/003.JPG,\n",
        encoding="utf-8",
    )
    return path


def test_manifest_loading(manifest, tmp_path):
    data_dir = tmp_path / "raw"
    label_map = {"normal": False, "anomaly": True}
    dataset = ImageRecordsDataset(manifest, data_dir, label_map=label_map)

    assert len(dataset) == 2
    assert dataset[0] == ImageRecord(
        path=data_dir / "cashew/normal/001.JPG", mask=None, is_anomaly=False, category="cashew"
    )

    subset = dataset[:1]
    assert isinstance(subset, ImageRecordsDataset)
    assert len(subset) == 1
    assert subset[0] is dataset[0]


def test_split_and_category_filtering(manifest, tmp_path):
    label_map = {"normal": False, "anomaly": True}
    dataset = ImageRecordsDataset(
        manifest, tmp_path, split="test", objects="cashew", label_map=label_map
    )

    assert len(dataset) == 1
    assert dataset[0].is_anomaly
    assert dataset[0].mask == tmp_path / "cashew/masks/002.png"

    assert len(ImageRecordsDataset(manifest, tmp_path, objects={"pcb1"}, label_map=label_map)) == 1
    assert len(ImageRecordsDataset(manifest, tmp_path, objects=[], label_map=label_map)) == 2
    assert len(ImageRecordsDataset(manifest, tmp_path, objects=None, label_map=label_map)) == 2


def test_empty_selection(manifest, tmp_path):
    dataset = ImageRecordsDataset(manifest, tmp_path, objects="unknown", label_map={})
    assert len(dataset) == 0


def test_image_loading(image_dataset):
    image = image_dataset[0].load_image((2, 3))

    assert image.shape == (3, 2, 3)
    assert image.dtype == np.float32
    np.testing.assert_allclose(image[:, 0, 0], [1.0, 128 / 255, 0.0])


def test_mask_loading(image_dataset):
    normal = image_dataset[0].load_mask((2, 3))
    anomaly = image_dataset[1].load_mask((2, 3))

    assert normal.dtype == np.bool_
    assert not normal.any()
    assert anomaly.sum() == 1
    assert anomaly[0, 1]

    resized = image_dataset[1].load_mask((4, 6))
    assert resized.shape == (4, 6)
    assert resized.sum() == 4


def test_sliced_batch_loading(image_dataset):
    batch = image_dataset[:2]
    images = batch.load_images((2, 3))
    masks = batch.load_masks((2, 3))

    assert isinstance(batch, ImageRecordsDataset)
    assert images.shape == (2, 3, 2, 3)
    assert masks.shape == (2, 2, 3)
    assert images.flags.c_contiguous
    assert masks.flags.c_contiguous
    assert masks[1].sum() == 1

    assert image_dataset[:0].load_images((2, 3)).shape == (0, 3, 2, 3)


def test_custom_anomaly_label(tmp_path):
    manifest = tmp_path / "manifest.csv"
    manifest.write_text(
        "object,split,label,image,mask\n"
        "cashew,train,good,normal.png,\n"
        "cashew,train,bad,anomaly.png,mask.png\n",
        encoding="utf-8",
    )

    label_map = {"good": False, "bad": True}
    dataset = ImageRecordsDataset(manifest, tmp_path, label_map=label_map)

    assert [record.is_anomaly for record in dataset] == [False, True]


def test_missing_anomaly_mask(image_dataset):
    from dataclasses import replace

    record = replace(image_dataset[1], mask=None)

    with pytest.raises(ValueError, match="Missing anomaly mask"):
        record.load_mask((2, 3))
