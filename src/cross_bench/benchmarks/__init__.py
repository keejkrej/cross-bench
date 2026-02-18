"""Benchmark modules for cross-image segmentation evaluation."""

from cross_bench.benchmarks.segmentation import SegmentationBenchmark
from cross_bench.benchmarks.transfer import ConceptTransferBenchmark
from cross_bench.benchmarks.detection import DetectionBenchmark

__all__ = [
    "SegmentationBenchmark",
    "ConceptTransferBenchmark",
    "DetectionBenchmark",
]
