# Visual Inspection Platform

[![CI](https://github.com/adampirog/visual-inspection-platform/actions/workflows/ci.yml/badge.svg)](https://github.com/adampirog/visual-inspection-platform/actions/workflows/ci.yml)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
![Python](https://img.shields.io/badge/python-3.13%2B-blue)
![License](https://img.shields.io/github/license/adampirog/visual-inspection-platform)

An end-to-end computer vision platform for industrial anomaly detection and defect segmentation, designed around reproducible experimentation, modular model development and efficient inference.

The platform covers the complete model lifecycle: data preparation, training, evaluation, serialization and deployment through a unified serving API.

## Features

- Unified interfaces for anomaly detection models and trainers.
- Shared image preprocessing for training and inference.
- Dataset-independent, manifest-based data loading.
- Pixel-level anomaly maps and binary defect segmentation.
- Calibrated global and spatial thresholding.
- Evaluation with pixel-level AUROC, Average Precision, Precision, Recall and Dice, including per-category results and macro averages.
- Self-contained model artifacts with training metadata and postprocessor state.
- FastAPI serving with model discovery, lazy loading and batch inference.
- Separation of training and inference dependencies.

## Supported models

| Model | Approach                    |
| ----- | --------------------------- |
| PCA   | Global image reconstruction |

## Installation

Requires Python 3.13 or newer.

```bash
git clone https://github.com/adampirog/visual-inspection-platform.git
cd visual-inspection-platform

pip install -e ".[pca,serving]"
```

For inference and serving only:

```bash
pip install -e ".[serving]"
```

## Dataset

The primary benchmark is [Visual Anomaly (VisA)](https://github.com/amazon-science/spot-diff), an industrial anomaly detection dataset containing 10,821 images across 12 object categories, with image-level and pixel-level annotations.

Download the dataset and the official one-class split manifest:

```bash
mkdir -p data/visa

curl -L \
    https://amazon-visual-anomaly.s3.us-west-2.amazonaws.com/VisA_20220922.tar \
    -o data/VisA.tar

tar -xf data/VisA.tar -C data/visa

curl -L \
    https://raw.githubusercontent.com/amazon-science/spot-diff/main/split_csv/1cls.csv \
    -o data/visa/1cls.csv
```

The platform operates directly on the original dataset structure using the provided CSV manifest. No image duplication or directory reorganization is required.

The dataset root should contain the object directories, such as `cashew`, `candle` and `pcb1`.

Other datasets can be integrated using a CSV manifest with the following columns:

```text
object,split,label,image,mask
```

Image and mask paths are relative to the dataset root. Dataset-specific labels are mapped to a common boolean representation.

## Usage

### Training and evaluation

Train a PCA detector on normal images from the `cashew` category, reserving a subset for postprocessor calibration.

```python
from visual_inspection_platform.data import ImageRecordsDataset
from visual_inspection_platform.evaluation.evaluate_model import evaluate_model
from visual_inspection_platform.models.pca.trainer import PCAConfig, PCATrainer
from visual_inspection_platform.postprocessors import ThresholdPostprocessor


dataset_config = {
    "manifest": "data/visa/1cls.csv",
    "data_dir": "data/visa",
    "label_map": {"normal": False, "anomaly": True},
    "objects": {"cashew"},
}

train = ImageRecordsDataset(**dataset_config, split="train")
test = ImageRecordsDataset(**dataset_config, split="test")

calibration, train = train[:64], train[64:]

trainer = PCATrainer(
    PCAConfig(n_components=32, image_size=(64, 64))
)

model = trainer.fit(train)

normal_maps = model.predict(
    calibration.load_images(model.image_size)
)

model.postprocessor = ThresholdPostprocessor.calibrate(
    normal_maps, quantile=0.995
)

result = evaluate_model(model, test, batch_size=32, verbose=True)
print(result)

model.save(f"artifacts/{model.model_id}")
```

### Inference

Serialized models can be loaded independently of their training environment.

```python
from visual_inspection_platform.models.pca import PCADetector
from visual_inspection_platform.utils.images import load_images


model = PCADetector.load("artifacts/MODEL_ID")

images = load_images(["image.png"], size=model.image_size)

anomaly_maps = model.predict(images)
masks = model.predict_mask(anomaly_maps)
```

### Serving

Start the inference server:

```bash
python -m visual_inspection_platform.serving.app artifacts
```

List available models:

```bash
curl http://localhost:8000/v1/models
```

Generate predictions:

```bash
curl -X POST \
    "http://localhost:8000/v1/models/MODEL_ID/predictions?include_masks=true" \
    -F "files=@image.png"
```

Interactive API documentation is available at `http://localhost:8000/docs`.

### Docker

Build and run the inference container:

```bash
docker build -t visual-inspection-platform .

docker run --rm \
    -p 127.0.0.1:8000:8000 \
    -v "$(pwd)/artifacts:/models:ro" \
    visual-inspection-platform
```

The container includes inference and serving dependencies without the model training stack.

## Roadmap

The platform is under active development, with successive milestones focused on expanding model capabilities, establishing comparable benchmarks and optimizing deployment.

### Models

- [x] PCA reconstruction
- [ ] Convolutional Autoencoder
- [ ] PaDiM
- [ ] PatchCore
- [ ] SimpleNet
- [ ] EfficientAD
- [ ] DINO-based anomaly detection
- [ ] Dinomaly

### Platform

- [x] Common model and training interfaces
- [x] Manifest-based datasets and shared preprocessing
- [x] Model serialization and postprocessor calibration
- [x] Segmentation evaluation framework
- [x] Model management and HTTP serving
- [ ] Reproducible multi-model benchmarks on VisA
- [ ] Additional dataset integration (MVTec AD)
- [ ] ONNX export and ONNX Runtime inference
- [ ] FP16 and INT8 inference optimization
- [ ] CPU and GPU performance benchmarking

## Development

```bash
pip install -e ".[pca,serving]" --group dev

ruff check .
pytest
```

## License

This project is licensed under the MIT License. External datasets are subject to their respective licenses.
