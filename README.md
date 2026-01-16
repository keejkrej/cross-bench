# Cross-Bench

A benchmarking tool for cross-image segmentation/detection with SAM3 using extracted concepts.

## Overview

Cross-Bench benchmarks two key capabilities of SAM3's cross-image segmentation:

1. **Segmentation from Concept**: Segment objects in a reference image using various prompt types (text, point, box, mask)
2. **Concept Transfer**: Extract a concept from a reference image and transfer it to find similar objects in a target image

## Installation

```bash
# Install cross-bench
pip install -e .

# Install SAM3 from the fork (required for running benchmarks)
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
│   └── base.py          # CrossImageDataset, DatasetSample
├── predictor.py          # CrossImagePredictor wrapper
├── benchmarks/           # Benchmark implementations
│   ├── __init__.py
│   ├── base.py          # BaseBenchmark, BenchmarkResult
│   ├── segmentation.py  # SegmentationBenchmark
│   └── transfer.py      # ConceptTransferBenchmark
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

Install with:
```bash
pip install git+https://github.com/keejkrej/sam3.git@feat/cross-image
```

## License

MIT
