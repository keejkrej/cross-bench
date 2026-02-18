"""Dataset module for cross-image segmentation benchmarking."""

from cross_bench.datasets.base import CrossImageDataset, DatasetSample
from cross_bench.datasets.coco import COCODetectionDataset, COCODetectionSample

__all__ = [
    "CrossImageDataset",
    "DatasetSample",
    "COCODetectionDataset",
    "COCODetectionSample",
]
