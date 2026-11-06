import pytest

from visual_inspection_platform.data import ImageRecord, ImageRecords


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
    records = ImageRecords(manifest, data_dir)

    assert len(records) == 2
    assert records[0] == ImageRecord(
        path=data_dir / "cashew/normal/001.JPG", mask=None, label="normal", category="cashew"
    )
    assert records[:1] == (records[0],)


def test_split_and_category_filtering(manifest, tmp_path):
    records = ImageRecords(manifest, tmp_path, split="test", objects="cashew")

    assert len(records) == 1
    assert records[0].label == "anomaly"
    assert records[0].mask == tmp_path / "cashew/masks/002.png"

    assert len(ImageRecords(manifest, tmp_path, objects={"pcb1"})) == 1
    assert len(ImageRecords(manifest, tmp_path, objects=[])) == 2
    assert len(ImageRecords(manifest, tmp_path, objects=None)) == 2
