#!/usr/bin/env python3
"""
Example usage of cross-bench for cross-image segmentation benchmarking.

This script demonstrates:
1. Loading a dataset
2. Running segmentation benchmark
3. Running concept transfer benchmark
4. Visualizing results
"""

from pathlib import Path

# Import cross-bench components
from cross_bench import (
    CrossImageDataset,
    CrossImagePredictor,
    SegmentationBenchmark,
    ConceptTransferBenchmark,
)
from cross_bench.predictor import Prompt
from cross_bench.visualization import (
    plot_segmentation,
    plot_transfer_comparison,
    create_benchmark_figure,
    save_figure,
)


def example_dataset_loading():
    """Demonstrate dataset loading methods."""
    print("=== Dataset Loading ===\n")

    # Method 1: From directory structure
    # dataset = CrossImageDataset.from_directory("./data/my_dataset")

    # Method 2: From manifest file
    # dataset = CrossImageDataset.from_manifest("./data/my_dataset/manifest.json")

    # Method 3: Programmatic construction
    from cross_bench.datasets import DatasetSample

    dataset = CrossImageDataset(name="example_dataset")

    # Add samples programmatically
    # sample = DatasetSample(
    #     sample_id="sample_001",
    #     reference_image_path=Path("./reference/img1.jpg"),
    #     reference_mask_path=Path("./masks/img1.png"),
    #     target_image_path=Path("./target/img1.jpg"),
    #     category="dog",
    # )
    # dataset.add_sample(sample)

    print(f"Dataset: {dataset.name}")
    print(f"Samples: {len(dataset)}")
    print(f"Categories: {dataset.categories}")
    print()


def example_predictor_usage():
    """Demonstrate direct predictor usage."""
    print("=== Predictor Usage ===\n")

    # Note: This requires SAM3 to be installed
    # predictor = CrossImagePredictor(confidence_threshold=0.5)

    # Example prompt creation
    print("Creating prompts:")

    # Text prompt
    text_prompt = Prompt.from_text("dog")
    print(f"  Text: {text_prompt}")

    # Point prompt
    point_prompt = Prompt.from_point(100, 200)
    print(f"  Point: {point_prompt}")

    # Box prompt (x, y, width, height)
    box_prompt = Prompt.from_box(50, 50, 100, 150)
    print(f"  Box: {box_prompt}")

    # Mask prompt (would need actual mask array)
    # import numpy as np
    # mask = np.zeros((256, 256), dtype=np.float32)
    # mask[50:150, 50:150] = 1.0
    # mask_prompt = Prompt.from_mask(mask)

    print()

    # Example segmentation (requires SAM3)
    # ref_img = Image.open("reference.jpg")
    # result = predictor.segment(ref_img, text_prompt)
    # print(f"Detected {result.num_detections} objects")

    # Example concept transfer (requires SAM3)
    # tgt_img = Image.open("target.jpg")
    # concept = predictor.extract_concept(ref_img, text_prompt)
    # tgt_result = predictor.transfer_concept(tgt_img, concept)


def example_benchmark_run():
    """Demonstrate running benchmarks."""
    print("=== Benchmark Run ===\n")

    # This requires SAM3 and a real dataset
    # predictor = CrossImagePredictor(confidence_threshold=0.5)
    # dataset = CrossImageDataset.from_directory("./data/my_dataset")

    # Segmentation benchmark
    print("Segmentation Benchmark:")
    print("  - Tests different prompt types on reference images")
    print("  - Prompt types: mask, box, point, text")
    print()

    # seg_benchmark = SegmentationBenchmark(predictor=predictor)
    # seg_run = seg_benchmark.run(
    #     dataset,
    #     prompt_types=["mask", "box", "point"],
    #     max_samples=10,
    # )

    # Concept transfer benchmark
    print("Concept Transfer Benchmark:")
    print("  - Extracts concept from reference with prompt")
    print("  - Transfers to find similar objects in target")
    print()

    # transfer_benchmark = ConceptTransferBenchmark(predictor=predictor)
    # transfer_run = transfer_benchmark.run(
    #     dataset,
    #     prompt_types=["mask", "box"],
    #     max_samples=10,
    # )


def example_visualization():
    """Demonstrate visualization capabilities."""
    print("=== Visualization ===\n")

    print("Available visualization functions:")
    print("  - plot_segmentation(): Single image with masks and prompt")
    print("  - plot_transfer_comparison(): Side-by-side reference/target")
    print("  - plot_benchmark_grid(): Multiple prompt types in grid")
    print("  - create_benchmark_figure(): Full benchmark result figure")
    print()

    # Example (requires actual images and results):
    # fig = plot_transfer_comparison(sample, ref_result, tgt_result, prompt)
    # save_figure(fig, "transfer_result.png")


def main():
    """Run all examples."""
    print("Cross-Bench Usage Examples")
    print("=" * 50)
    print()

    example_dataset_loading()
    example_predictor_usage()
    example_benchmark_run()
    example_visualization()

    print("=" * 50)
    print("For full functionality, install SAM3 from the fork:")
    print("  pip install git+https://github.com/keejkrej/sam3.git@feat/cross-image")
    print()


if __name__ == "__main__":
    main()
