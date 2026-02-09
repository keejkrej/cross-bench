"""
Cross-Bench: A benchmarking tool for cross-image segmentation/detection with SAM3.

This package provides tools to benchmark:
1. Segmentation from concept by text/point/box/mask prompts
2. Concept transfer from reference images to target images
"""

from importlib.resources import files
from pathlib import Path

__version__ = "0.1.0"

from cross_bench.datasets import CrossImageDataset, DatasetSample
from cross_bench.predictor import CrossImagePredictor
from cross_bench.benchmarks import SegmentationBenchmark, ConceptTransferBenchmark


def get_example_path(name: str) -> Path:
    """Get path to a bundled example asset.

    Args:
        name: Name of the asset file (e.g., "cat.jpg", "cat-mask.jpg", "cats.jpg")

    Returns:
        Path to the asset file.

    Example:
        >>> from cross_bench import get_example_path
        >>> cat_path = get_example_path("cat.jpg")
        >>> mask_path = get_example_path("cat-mask.jpg")
    """
    return Path(str(files("cross_bench").joinpath(f"assets/examples/{name}")))


__all__ = [
    "CrossImageDataset",
    "DatasetSample",
    "CrossImagePredictor",
    "SegmentationBenchmark",
    "ConceptTransferBenchmark",
    "get_example_path",
]
