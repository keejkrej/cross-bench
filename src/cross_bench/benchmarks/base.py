"""Base benchmark class with common functionality."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Iterator
from datetime import datetime

from cross_bench.datasets import CrossImageDataset, DatasetSample
from cross_bench.predictor import CrossImagePredictor, SegmentationResult


@dataclass
class BenchmarkResult:
    """Result from running a benchmark on a single sample.

    Attributes:
        sample_id: ID of the dataset sample
        prompt_type: Type of prompt used (text/point/box/mask)
        results: Dictionary of SegmentationResults keyed by stage name
        metadata: Additional metadata about the run
    """
    sample_id: str
    prompt_type: str
    results: dict[str, SegmentationResult] = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)

    def calculate_iou(self, stage: str = "target") -> Optional[float]:
        """Calculate IoU between prediction and ground truth mask.
        
        Args:
            stage: Which stage to evaluate ("reference" or "target")
            
        Returns:
            IoU score (0-1) or None if no ground truth available
        """
        from cross_bench.datasets import DatasetSample
        
        # Get the result for this stage
        result = self.results.get(stage)
        if not result or not result.masks:
            return None
            
        # Get ground truth mask from metadata if available
        gt_mask_path = self.metadata.get(f"{stage}_mask_path")
        if not gt_mask_path:
            return None
            
        # Load ground truth
        import numpy as np
        from PIL import Image
        gt_img = Image.open(gt_mask_path).convert("L")
        gt_mask = (np.array(gt_img) > 128).astype(np.float32)
        
        # Combine predicted masks
        pred_mask = np.zeros_like(gt_mask)
        for mask in result.masks:
            pred_mask = np.maximum(pred_mask, mask)
            
        # Calculate IoU
        intersection = np.sum(pred_mask * gt_mask)
        union = np.sum(pred_mask) + np.sum(gt_mask) - intersection
        
        if union == 0:
            return 0.0
            
        return float(intersection / union)


@dataclass
class BenchmarkRun:
    """Collection of results from a complete benchmark run.

    Attributes:
        name: Name of the benchmark run
        timestamp: When the run was executed
        dataset_name: Name of the dataset used
        results: List of BenchmarkResult objects
        config: Configuration used for the run
    """
    name: str
    timestamp: datetime
    dataset_name: str
    results: list[BenchmarkResult] = field(default_factory=list)
    config: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.results)

    def __iter__(self) -> Iterator[BenchmarkResult]:
        return iter(self.results)

    def add_result(self, result: BenchmarkResult) -> None:
        """Add a result to the run."""
        self.results.append(result)

    def filter_by_prompt_type(self, prompt_type: str) -> list[BenchmarkResult]:
        """Filter results by prompt type."""
        return [r for r in self.results if r.prompt_type == prompt_type]


class BaseBenchmark(ABC):
    """Abstract base class for benchmarks.

    Subclasses implement specific benchmark types (segmentation, transfer, etc.)
    """

    def __init__(
        self,
        predictor: Optional[CrossImagePredictor] = None,
        output_dir: Optional[Path] = None,
        confidence_threshold: float = 0.5,
    ):
        """Initialize the benchmark.

        Args:
            predictor: CrossImagePredictor instance (created if None)
            output_dir: Directory to save results and visualizations
            confidence_threshold: Confidence threshold for predictions
        """
        self.predictor = predictor or CrossImagePredictor(
            confidence_threshold=confidence_threshold
        )
        self.output_dir = Path(output_dir) if output_dir else None
        self.confidence_threshold = confidence_threshold

        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the benchmark."""
        pass

    @abstractmethod
    def run_sample(
        self,
        sample: DatasetSample,
        prompt_types: list[str],
    ) -> list[BenchmarkResult]:
        """Run benchmark on a single sample.

        Args:
            sample: Dataset sample to benchmark
            prompt_types: List of prompt types to test

        Returns:
            List of BenchmarkResult objects (one per prompt type)
        """
        pass

    def run(
        self,
        dataset: CrossImageDataset,
        prompt_types: Optional[list[str]] = None,
        max_samples: Optional[int] = None,
        verbose: bool = True,
    ) -> BenchmarkRun:
        """Run the benchmark on a dataset.

        Args:
            dataset: Dataset to benchmark
            prompt_types: List of prompt types to test (default: all)
            max_samples: Maximum number of samples to process
            verbose: Whether to print progress

        Returns:
            BenchmarkRun with all results
        """
        prompt_types = prompt_types or ["mask", "box", "point"]

        run = BenchmarkRun(
            name=self.name,
            timestamp=datetime.now(),
            dataset_name=dataset.name,
            config={
                "confidence_threshold": self.confidence_threshold,
                "prompt_types": prompt_types,
            },
        )

        samples = list(dataset)
        if max_samples is not None:
            samples = samples[:max_samples]

        for i, sample in enumerate(samples):
            if verbose:
                print(f"[{i+1}/{len(samples)}] Processing {sample.sample_id}...")

            try:
                results = self.run_sample(sample, prompt_types)
                for result in results:
                    run.add_result(result)
            except Exception as e:
                if verbose:
                    print(f"  Error: {e}")
                continue

            # Clear cache to free memory
            sample.clear_cache()

        return run

    def calculate_scores(self, run: BenchmarkRun) -> dict[str, float]:
        """Calculate evaluation scores for a benchmark run.
        
        Args:
            run: BenchmarkRun with results
            
        Returns:
            Dictionary with scores (IoU, detection rate, etc.)
        """
        scores = {
            "total_samples": len(run.results),
            "reference_detections": 0,
            "target_detections": 0,
            "reference_iou_sum": 0.0,
            "target_iou_sum": 0.0,
            "reference_iou_count": 0,
            "target_iou_count": 0,
        }
        
        for result in run.results:
            if "reference" in result.results:
                scores["reference_detections"] += result.results["reference"].num_detections
                ref_iou = result.calculate_iou("reference")
                if ref_iou is not None:
                    scores["reference_iou_sum"] += ref_iou
                    scores["reference_iou_count"] += 1
                    
            if "target" in result.results:
                scores["target_detections"] += result.results["target"].num_detections
                tgt_iou = result.calculate_iou("target")
                if tgt_iou is not None:
                    scores["target_iou_sum"] += tgt_iou
                    scores["target_iou_count"] += 1
        
        # Calculate averages
        if scores["reference_iou_count"] > 0:
            scores["reference_iou_avg"] = scores["reference_iou_sum"] / scores["reference_iou_count"]
        else:
            scores["reference_iou_avg"] = 0.0
            
        if scores["target_iou_count"] > 0:
            scores["target_iou_avg"] = scores["target_iou_sum"] / scores["target_iou_count"]
        else:
            scores["target_iou_avg"] = 0.0
        
        return scores
