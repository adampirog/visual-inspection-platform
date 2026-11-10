import pytest

from visual_inspection_platform.data import ImageRecord, ImageRecordsDataset


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
    dataset = ImageRecordsDataset(manifest, data_dir)

    assert len(dataset) == 2
    assert dataset[0] == ImageRecord(
        path=data_dir / "cashew/normal/001.JPG", mask=None, label="normal", category="cashew"
    )
    assert dataset[:1] == (dataset[0],)


def test_split_and_category_filtering(manifest, tmp_path):
    dataset = ImageRecordsDataset(manifest, tmp_path, split="test", objects="cashew")

    assert len(dataset) == 1
    assert dataset[0].label == "anomaly"
    assert dataset[0].mask == tmp_path / "cashew/masks/002.png"

    assert len(ImageRecordsDataset(manifest, tmp_path, objects={"pcb1"})) == 1
    assert len(ImageRecordsDataset(manifest, tmp_path, objects=[])) == 2
    assert len(ImageRecordsDataset(manifest, tmp_path, objects=None)) == 2


def test_empty_selection(manifest, tmp_path):
    dataset = ImageRecordsDataset(manifest, tmp_path, objects="unknown")
    assert len(dataset) == 0
