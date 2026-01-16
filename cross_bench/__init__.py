"""
Cross-Bench: A benchmarking tool for cross-image segmentation/detection with SAM3.

This package provides tools to benchmark:
1. Segmentation from concept by text/point/box/mask prompts
2. Concept transfer from reference images to target images
"""

__version__ = "0.1.0"

from cross_bench.datasets import CrossImageDataset, DatasetSample
from cross_bench.predictor import CrossImagePredictor
from cross_bench.benchmarks import SegmentationBenchmark, ConceptTransferBenchmark

__all__ = [
    "CrossImageDataset",
    "DatasetSample",
    "CrossImagePredictor",
    "SegmentationBenchmark",
    "ConceptTransferBenchmark",
]
