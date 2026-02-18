# Cross-Bench

A benchmarking tool for cross-image segmentation/detection with SAM3 using extracted concepts.

## Overview

Cross-Bench benchmarks three key capabilities of SAM3's cross-image segmentation:

1. **Segmentation from Concept**: Segment objects in a reference image using various prompt types (text, point, box, mask)
2. **Concept Transfer**: Extract a concept from a reference image and transfer it to find similar objects in a target image
3. **Object Detection**: Benchmark detection on classic datasets (COCO). Segmentation is trivial with SAM1/SAM2 once the bbox is acquired—the hard part is finding the boxes

## Installation

### Prerequisites

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) (recommended) or pip
- The `sam3` repository cloned alongside this repo

### Directory Structure

```
sam-project/
├── sam3/           # SAM3 fork (feat/cross-image branch)
└── cross-bench/    # This repo
```

### Setup with uv (Recommended)

```bash
# Clone both repos
git clone https://github.com/keejkrej/sam3.git -b feat/cross-image
git clone https://github.com/keejkrej/cross-bench.git -b feat/sam

cd cross-bench

# Install cross-bench dependencies
uv sync

# Install SAM3 as editable from local directory (with all extras)
uv pip install -e "../sam3[notebooks,dev]"
```

### Setup with pip

```bash
# Install cross-bench
pip install -e .

# Install SAM3 from the fork
pip install git+https://github.com/keejkrej/sam3.git@feat/cross-image
```

## Dataset Format

Datasets consist of triplets:
- **Reference image**: Contains the object of interest
- **Reference mask**: Binary mask indicating the object (white=object)
- **Target image**: Contains similar objects to find

### Directory Structure

```
dataset/
├── manifest.json      # Optional: explicit sample definitions
├── reference/         # Reference images
│   ├── sample_001.jpg
│   └── sample_002.jpg
├── masks/            # Binary masks (white=object)
│   ├── sample_001.png
│   └── sample_002.png
└── target/           # Target images
    ├── sample_001.jpg
    └── sample_002.jpg
```

### COCO Format (Detection Benchmark)

**Hugging Face** (recommended, uses cache at `~/.cache/huggingface/hub/`):
```bash
cross-bench detection --from-hf
```
Uses `detection-datasets/coco` from the Hub. If already downloaded, loads from cache.

**Local COCO** structure:
```
coco_root/
├── images/
│   ├── train2017/
│   └── val2017/
└── annotations/
    ├── instances_train2017.json
    └── instances_val2017.json
```
Download from https://cocodataset.org or https://huggingface.co/datasets/detection-datasets/coco

### Manifest Format (Optional)

```json
{
  "name": "my_dataset",
  "samples": [
    {
      "id": "sample_001",
      "reference": "reference/sample_001.jpg",
      "mask": "masks/sample_001.png",
      "target": "target/sample_001.jpg",
      "category": "dog"
    }
  ]
}
```

## Usage

### Command Line

```bash
# Run segmentation benchmark
cross-bench segmentation --dataset ./data/my_dataset --visualize

# Run concept transfer benchmark
cross-bench transfer --dataset ./data/my_dataset --prompts mask,box,point

# Run object detection benchmark (Hugging Face - uses cache)
cross-bench detection --from-hf --max-samples 100
cross-bench detection --from-hf -n 500 -c "person,dog,cat"

# Or local COCO
cross-bench detection -d /path/to/coco -s val -n 100

# Quick single-image test
cross-bench single \
  --reference img1.jpg \
  --target img2.jpg \
  --prompt-type mask \
  --mask mask.png \
  --output result.png
```

### Python API

```python
from cross_bench import (
    CrossImageDataset,
    CrossImagePredictor,
    SegmentationBenchmark,
    ConceptTransferBenchmark,
    DetectionBenchmark,
    COCODetectionDataset,
)
from cross_bench.visualization import plot_transfer_comparison

# Load dataset
dataset = CrossImageDataset.from_directory("./data/my_dataset")

# Or from manifest
dataset = CrossImageDataset.from_manifest("./data/my_dataset/manifest.json")

# Create predictor
predictor = CrossImagePredictor(confidence_threshold=0.5)

# Run segmentation benchmark
seg_benchmark = SegmentationBenchmark(predictor=predictor)
seg_run = seg_benchmark.run(dataset, prompt_types=["mask", "box", "point"])

# Run concept transfer benchmark
transfer_benchmark = ConceptTransferBenchmark(predictor=predictor)
transfer_run = transfer_benchmark.run(dataset, prompt_types=["mask", "box"])

# Run detection benchmark (Hugging Face)
coco_ds = COCODetectionDataset.from_huggingface(
    name="detection-datasets/coco",
    split="val",
    max_samples=100,
    categories=["person", "dog", "cat"],
)

# Or from local COCO
# coco_ds = COCODetectionDataset.from_coco(root="/path/to/coco", split="val", ...)
det_benchmark = DetectionBenchmark(predictor=predictor)
det_run = det_benchmark.run(coco_ds, prompt_types=["box", "point", "mask"])
scores = det_benchmark.calculate_scores(det_run)
print(f"mAP@50: {scores['mAP50']:.3f}, mAP@75: {scores['mAP75']:.3f}")

# Quick single-sample transfer
sample = dataset[0]
ref_result, tgt_result, prompt = transfer_benchmark.run_single(sample, "mask")

# Visualize
fig = plot_transfer_comparison(sample, ref_result, tgt_result, prompt)
fig.savefig("transfer_result.png")
```

### Direct Predictor Usage

```python
from PIL import Image
from cross_bench import CrossImagePredictor
from cross_bench.predictor import Prompt

predictor = CrossImagePredictor(confidence_threshold=0.5)

# Load images
ref_img = Image.open("reference.jpg")
tgt_img = Image.open("target.jpg")

# Create prompt from mask
import numpy as np
mask = (np.array(Image.open("mask.png").convert("L")) > 128).astype(np.float32)
prompt = Prompt.from_mask(mask)

# Or other prompt types
prompt = Prompt.from_point(100, 200)  # x, y
prompt = Prompt.from_box(50, 50, 100, 100)  # x, y, w, h
prompt = Prompt.from_text("dog")

# Segment reference image
ref_result = predictor.segment(ref_img, prompt)

# Extract concept and transfer
concept = predictor.extract_concept(ref_img, prompt)
tgt_result = predictor.transfer_concept(tgt_img, concept)

# Or do both in one call
ref_result, tgt_result = predictor.segment_and_transfer(ref_img, tgt_img, prompt)

print(f"Reference: {ref_result.num_detections} objects")
print(f"Target: {tgt_result.num_detections} objects")
```

## Architecture

```
cross_bench/
├── __init__.py           # Package exports
├── datasets/             # Dataset loading
│   ├── __init__.py
│   ├── base.py          # CrossImageDataset, DatasetSample
│   └── coco.py          # COCODetectionDataset for detection benchmark
├── predictor.py          # CrossImagePredictor wrapper
├── benchmarks/           # Benchmark implementations
│   ├── __init__.py
│   ├── base.py          # BaseBenchmark, BenchmarkResult
│   ├── segmentation.py  # SegmentationBenchmark
│   ├── transfer.py      # ConceptTransferBenchmark
│   └── detection.py     # DetectionBenchmark (mAP, AP@50, AP@75)
├── visualization/        # Plotting utilities
│   ├── __init__.py
│   └── plotting.py
├── config.py            # Configuration classes
└── cli.py               # Command-line interface
```

## SAM3 Dependency

This benchmark requires the SAM3 model with cross-image capabilities from the fork:
https://github.com/keejkrej/sam3/tree/feat/cross-image

The fork contains modifications to enable:
- Prompt embedding extraction
- Cross-image concept transfer
- Enhanced processor API
- `NaiveMaskEncoder` for mask prompts without pretrained weights

See [Installation](#installation) for setup instructions.

## License

MIT
