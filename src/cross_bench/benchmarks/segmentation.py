"""Segmentation benchmark for evaluating prompt-based segmentation.

This benchmark evaluates SAM3's ability to segment objects using different
prompt types (text, point, box, mask) on reference images.
"""

from pathlib import Path
from typing import Optional

from cross_bench.datasets import DatasetSample
from cross_bench.predictor import CrossImagePredictor, Prompt, SegmentationResult
from cross_bench.benchmarks.base import BaseBenchmark, BenchmarkResult


class SegmentationBenchmark(BaseBenchmark):
    """Benchmark for single-image segmentation with various prompt types.

    This evaluates how well SAM3 segments objects in the reference image
    when given different types of prompts derived from the ground truth mask.

    Prompt types:
    - mask: Use the ground truth mask directly as prompt
    - box: Use the bounding box of the mask as prompt
    - point: Use the centroid of the mask as prompt
    - text: Use the category name as text prompt (if available)
    """

    @property
    def name(self) -> str:
        return "segmentation"

    def _create_prompts(
        self,
        sample: DatasetSample,
        prompt_types: list[str],
        mask_encoding_method: str = "default",
        mask_encoding_params: dict = None,
    ) -> dict[str, Prompt]:
        """Create prompts from a dataset sample.

        Args:
            sample: Dataset sample with mask
            prompt_types: List of prompt types to create
            mask_encoding_method: Encoding method for mask prompts
            mask_encoding_params: Additional encoding parameters

        Returns:
            Dictionary mapping prompt type to Prompt object
        """
        prompts = {}
        mask_encoding_params = mask_encoding_params or {}

        for ptype in prompt_types:
            if ptype == "mask":
                prompts["mask"] = Prompt.from_mask(
                    sample.reference_mask,
                    encoding_method=mask_encoding_method,
                    **mask_encoding_params
                )

            elif ptype == "box":
                x, y, w, h = sample.get_mask_bbox()
                prompts["box"] = Prompt.from_box(x, y, w, h)

            elif ptype == "point":
                x, y = sample.get_mask_centroid()
                prompts["point"] = Prompt.from_point(x, y, label=True)

            elif ptype == "text":
                if sample.category:
                    prompts["text"] = Prompt.from_text(sample.category)
                # Skip if no category available

        return prompts

    def run_sample(
        self,
        sample: DatasetSample,
        prompt_types: list[str],
    ) -> list[BenchmarkResult]:
        """Run segmentation benchmark on a single sample.

        Args:
            sample: Dataset sample to benchmark
            prompt_types: List of prompt types to test

        Returns:
            List of BenchmarkResult objects
        """
        results = []

        # Create prompts from ground truth
        prompts = self._create_prompts(
            sample, 
            prompt_types,
            mask_encoding_method=self.mask_encoding_method,
            mask_encoding_params=self.mask_encoding_params,
        )

        for ptype, prompt in prompts.items():
            try:
                # Run segmentation
                seg_result = self.predictor.segment(
                    sample.reference_image,
                    prompt,
                    threshold=self.confidence_threshold,
                )

                result = BenchmarkResult(
                    sample_id=sample.sample_id,
                    prompt_type=ptype,
                    results={"reference": seg_result},
                    metadata={
                        "category": sample.category,
                        "num_detections": seg_result.num_detections,
                    },
                )
                results.append(result)

            except Exception as e:
                # Record failed attempt
                result = BenchmarkResult(
                    sample_id=sample.sample_id,
                    prompt_type=ptype,
                    metadata={"error": str(e)},
                )
                results.append(result)

        return results

    def run_single(
        self,
        sample: DatasetSample,
        prompt_type: str = "mask",
    ) -> tuple[SegmentationResult, Prompt]:
        """Run segmentation on a single sample with one prompt type.

        Convenience method for quick testing.

        Args:
            sample: Dataset sample
            prompt_type: Type of prompt to use

        Returns:
            Tuple of (SegmentationResult, Prompt used)
        """
        prompts = self._create_prompts(sample, [prompt_type])

        if prompt_type not in prompts:
            raise ValueError(
                f"Cannot create {prompt_type} prompt for sample {sample.sample_id}"
            )

        prompt = prompts[prompt_type]
        result = self.predictor.segment(
            sample.reference_image,
            prompt,
            threshold=self.confidence_threshold,
        )

        return result, prompt
