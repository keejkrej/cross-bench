"""Concept transfer benchmark for cross-image segmentation.

This benchmark evaluates SAM3's ability to transfer object concepts from
a reference image to find similar objects in a target image.
"""

from pathlib import Path
from typing import Optional

from cross_bench.datasets import DatasetSample
from cross_bench.predictor import (
    CrossImagePredictor,
    Prompt,
    SegmentationResult,
    ConceptEmbedding,
)
from cross_bench.benchmarks.base import BaseBenchmark, BenchmarkResult


class ConceptTransferBenchmark(BaseBenchmark):
    """Benchmark for cross-image concept transfer.

    This evaluates how well SAM3 can:
    1. Extract a concept from a prompted object in the reference image
    2. Transfer that concept to find similar objects in the target image

    The benchmark tests this with different prompt types on the reference image.
    """

    @property
    def name(self) -> str:
        return "concept_transfer"

    def _create_prompts(
        self,
        sample: DatasetSample,
        prompt_types: list[str],
    ) -> dict[str, Prompt]:
        """Create prompts from a dataset sample."""
        prompts = {}

        for ptype in prompt_types:
            if ptype == "mask":
                prompts["mask"] = Prompt.from_mask(sample.reference_mask)

            elif ptype == "box":
                x, y, w, h = sample.get_mask_bbox()
                prompts["box"] = Prompt.from_box(x, y, w, h)

            elif ptype == "point":
                x, y = sample.get_mask_centroid()
                prompts["point"] = Prompt.from_point(x, y, label=True)

            elif ptype == "text":
                if sample.category:
                    prompts["text"] = Prompt.from_text(sample.category)

        return prompts

    def run_sample(
        self,
        sample: DatasetSample,
        prompt_types: list[str],
    ) -> list[BenchmarkResult]:
        """Run concept transfer benchmark on a single sample.

        For each prompt type:
        1. Segment the reference image with the prompt
        2. Extract the concept embedding
        3. Transfer to the target image
        4. Record both results

        Args:
            sample: Dataset sample with reference, mask, and target
            prompt_types: List of prompt types to test

        Returns:
            List of BenchmarkResult objects
        """
        results = []

        prompts = self._create_prompts(sample, prompt_types)

        for ptype, prompt in prompts.items():
            try:
                # Wrap single prompt in PromptMap
                prompt_map = {prompt.prompt_type: prompt}
                
                # Run segmentation and transfer
                ref_result, tgt_result = self.predictor.segment_and_transfer(
                    reference_image=sample.reference_image,
                    target_image=sample.target_image,
                    prompts=prompt_map,
                    threshold=self.confidence_threshold,
                )

                result = BenchmarkResult(
                    sample_id=sample.sample_id,
                    prompt_type=ptype,
                    results={
                        "reference": ref_result,
                        "target": tgt_result,
                    },
                    metadata={
                        "category": sample.category,
                        "reference_detections": ref_result.num_detections,
                        "target_detections": tgt_result.num_detections,
                    },
                )
                results.append(result)

            except Exception as e:
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
    ) -> tuple[SegmentationResult, SegmentationResult, Prompt]:
        """Run transfer on a single sample with one prompt type.

        Convenience method for quick testing.

        Args:
            sample: Dataset sample
            prompt_type: Type of prompt to use

        Returns:
            Tuple of (reference_result, target_result, prompt)
        """
        prompts = self._create_prompts(sample, [prompt_type])

        if prompt_type not in prompts:
            raise ValueError(
                f"Cannot create {prompt_type} prompt for sample {sample.sample_id}"
            )

        prompt = prompts[prompt_type]
        ref_result, tgt_result = self.predictor.segment_and_transfer(
            reference_image=sample.reference_image,
            target_image=sample.target_image,
            prompt=prompt,
            threshold=self.confidence_threshold,
        )

        return ref_result, tgt_result, prompt

    def transfer_with_concept(
        self,
        sample: DatasetSample,
        concept: ConceptEmbedding,
    ) -> SegmentationResult:
        """Transfer a pre-extracted concept to a sample's target image.

        Useful for testing one concept against multiple targets.

        Args:
            sample: Dataset sample (uses target_image)
            concept: Pre-extracted concept embedding

        Returns:
            SegmentationResult from the target image
        """
        return self.predictor.transfer_concept(
            sample.target_image,
            concept,
            threshold=self.confidence_threshold,
        )
