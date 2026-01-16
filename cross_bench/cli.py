"""Command-line interface for cross-bench."""

import argparse
from pathlib import Path
from datetime import datetime

from cross_bench.datasets import CrossImageDataset
from cross_bench.predictor import CrossImagePredictor
from cross_bench.benchmarks import SegmentationBenchmark, ConceptTransferBenchmark
from cross_bench.visualization import (
    plot_transfer_comparison,
    create_benchmark_figure,
    save_figure,
)
from cross_bench.config import BenchmarkConfig


def run_segmentation_benchmark(args: argparse.Namespace) -> None:
    """Run the segmentation benchmark."""
    # Load dataset
    dataset_path = Path(args.dataset)
    if (dataset_path / "manifest.json").exists():
        dataset = CrossImageDataset.from_manifest(dataset_path / "manifest.json")
    else:
        dataset = CrossImageDataset.from_directory(dataset_path)

    print(f"Loaded dataset: {dataset.name} ({len(dataset)} samples)")

    # Setup output directory
    output_dir = Path(args.output) if args.output else Path("results") / "segmentation"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create predictor and benchmark
    predictor = CrossImagePredictor(confidence_threshold=args.threshold)
    benchmark = SegmentationBenchmark(
        predictor=predictor,
        output_dir=output_dir,
        confidence_threshold=args.threshold,
    )

    # Parse prompt types
    prompt_types = args.prompts.split(",") if args.prompts else ["mask", "box", "point"]

    # Run benchmark
    print(f"\nRunning segmentation benchmark with prompts: {prompt_types}")
    run = benchmark.run(
        dataset,
        prompt_types=prompt_types,
        max_samples=args.max_samples,
        verbose=True,
    )

    print(f"\nCompleted {len(run)} benchmark results")

    # Save visualizations
    if args.visualize:
        print("\nGenerating visualizations...")
        vis_dir = output_dir / "visualizations"
        vis_dir.mkdir(exist_ok=True)

        for result in run:
            sample = dataset[0]  # Find matching sample
            for s in dataset:
                if s.sample_id == result.sample_id:
                    sample = s
                    break

            fig = create_benchmark_figure(sample, result, show_ground_truth=True)
            save_figure(fig, vis_dir / f"{result.sample_id}_{result.prompt_type}.png")

        print(f"Saved visualizations to {vis_dir}")


def run_transfer_benchmark(args: argparse.Namespace) -> None:
    """Run the concept transfer benchmark."""
    # Load dataset
    dataset_path = Path(args.dataset)
    if (dataset_path / "manifest.json").exists():
        dataset = CrossImageDataset.from_manifest(dataset_path / "manifest.json")
    else:
        dataset = CrossImageDataset.from_directory(dataset_path)

    print(f"Loaded dataset: {dataset.name} ({len(dataset)} samples)")

    # Setup output directory
    output_dir = Path(args.output) if args.output else Path("results") / "transfer"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create predictor and benchmark
    predictor = CrossImagePredictor(confidence_threshold=args.threshold)
    benchmark = ConceptTransferBenchmark(
        predictor=predictor,
        output_dir=output_dir,
        confidence_threshold=args.threshold,
    )

    # Parse prompt types
    prompt_types = args.prompts.split(",") if args.prompts else ["mask", "box", "point"]

    # Run benchmark
    print(f"\nRunning concept transfer benchmark with prompts: {prompt_types}")
    run = benchmark.run(
        dataset,
        prompt_types=prompt_types,
        max_samples=args.max_samples,
        verbose=True,
    )

    print(f"\nCompleted {len(run)} benchmark results")

    # Print summary
    for ptype in prompt_types:
        results = run.filter_by_prompt_type(ptype)
        if results:
            total_ref = sum(r.metadata.get("reference_detections", 0) for r in results)
            total_tgt = sum(r.metadata.get("target_detections", 0) for r in results)
            print(f"  {ptype}: {len(results)} samples, "
                  f"{total_ref} ref detections, {total_tgt} target detections")

    # Save visualizations
    if args.visualize:
        print("\nGenerating visualizations...")
        vis_dir = output_dir / "visualizations"
        vis_dir.mkdir(exist_ok=True)

        for result in run:
            # Find matching sample
            sample = None
            for s in dataset:
                if s.sample_id == result.sample_id:
                    sample = s
                    break

            if sample is None:
                continue

            ref_result = result.results.get("reference")
            tgt_result = result.results.get("target")

            if ref_result and tgt_result:
                fig = plot_transfer_comparison(
                    sample,
                    ref_result,
                    tgt_result,
                    prompt=ref_result.prompt,
                    title=f"Concept Transfer - {result.prompt_type.upper()}",
                )
                save_figure(
                    fig,
                    vis_dir / f"{result.sample_id}_{result.prompt_type}_transfer.png"
                )

        print(f"Saved visualizations to {vis_dir}")


def run_single(args: argparse.Namespace) -> None:
    """Run benchmark on a single sample for quick testing."""
    from PIL import Image
    from cross_bench.predictor import Prompt
    import matplotlib.pyplot as plt

    # Load images
    ref_img = Image.open(args.reference).convert("RGB")
    tgt_img = Image.open(args.target).convert("RGB") if args.target else None

    predictor = CrossImagePredictor(confidence_threshold=args.threshold)

    # Create prompt based on type
    if args.prompt_type == "text":
        if not args.text:
            raise ValueError("--text required for text prompt")
        prompt = Prompt.from_text(args.text)
    elif args.prompt_type == "point":
        if not args.point:
            raise ValueError("--point required for point prompt")
        x, y = map(float, args.point.split(","))
        prompt = Prompt.from_point(x, y)
    elif args.prompt_type == "box":
        if not args.box:
            raise ValueError("--box required for box prompt")
        x, y, w, h = map(float, args.box.split(","))
        prompt = Prompt.from_box(x, y, w, h)
    elif args.prompt_type == "mask":
        if not args.mask:
            raise ValueError("--mask required for mask prompt")
        import numpy as np
        mask_img = Image.open(args.mask).convert("L")
        mask = (np.array(mask_img) > 128).astype(np.float32)
        prompt = Prompt.from_mask(mask)
    else:
        raise ValueError(f"Unknown prompt type: {args.prompt_type}")

    print(f"Running with {args.prompt_type} prompt...")

    if tgt_img:
        # Run transfer
        ref_result, tgt_result = predictor.segment_and_transfer(
            ref_img, tgt_img, prompt, threshold=args.threshold
        )
        print(f"Reference: {ref_result.num_detections} detections")
        print(f"Target: {tgt_result.num_detections} detections")

        if args.output:
            from cross_bench.visualization import plot_segmentation
            fig, axes = plt.subplots(1, 2, figsize=(16, 8))
            plot_segmentation(ref_img, ref_result, prompt=prompt, ax=axes[0],
                            title="Reference")
            plot_segmentation(tgt_img, tgt_result, ax=axes[1],
                            title="Target (Transfer)", show_prompt=False)
            plt.tight_layout()
            plt.savefig(args.output, dpi=150, bbox_inches="tight")
            print(f"Saved to {args.output}")
        else:
            plt.show()
    else:
        # Run segmentation only
        result = predictor.segment(ref_img, prompt, threshold=args.threshold)
        print(f"Detections: {result.num_detections}")

        if args.output:
            from cross_bench.visualization import plot_segmentation
            fig, ax = plt.subplots(1, 1, figsize=(10, 10))
            plot_segmentation(ref_img, result, prompt=prompt, ax=ax)
            plt.savefig(args.output, dpi=150, bbox_inches="tight")
            print(f"Saved to {args.output}")
        else:
            plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Cross-Bench: Cross-image segmentation benchmarking tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run segmentation benchmark
  cross-bench segmentation --dataset ./data/my_dataset --visualize

  # Run concept transfer benchmark
  cross-bench transfer --dataset ./data/my_dataset --prompts mask,box

  # Quick single-image test
  cross-bench single --reference img1.jpg --target img2.jpg --mask mask.png
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Benchmark command")

    # Segmentation benchmark
    seg_parser = subparsers.add_parser(
        "segmentation", help="Run segmentation benchmark"
    )
    seg_parser.add_argument(
        "--dataset", "-d", required=True,
        help="Path to dataset directory"
    )
    seg_parser.add_argument(
        "--output", "-o",
        help="Output directory for results"
    )
    seg_parser.add_argument(
        "--prompts", "-p", default="mask,box,point",
        help="Comma-separated prompt types (default: mask,box,point)"
    )
    seg_parser.add_argument(
        "--threshold", "-t", type=float, default=0.5,
        help="Confidence threshold (default: 0.5)"
    )
    seg_parser.add_argument(
        "--max-samples", "-n", type=int,
        help="Maximum samples to process"
    )
    seg_parser.add_argument(
        "--visualize", "-v", action="store_true",
        help="Generate and save visualizations"
    )

    # Transfer benchmark
    transfer_parser = subparsers.add_parser(
        "transfer", help="Run concept transfer benchmark"
    )
    transfer_parser.add_argument(
        "--dataset", "-d", required=True,
        help="Path to dataset directory"
    )
    transfer_parser.add_argument(
        "--output", "-o",
        help="Output directory for results"
    )
    transfer_parser.add_argument(
        "--prompts", "-p", default="mask,box,point",
        help="Comma-separated prompt types (default: mask,box,point)"
    )
    transfer_parser.add_argument(
        "--threshold", "-t", type=float, default=0.5,
        help="Confidence threshold (default: 0.5)"
    )
    transfer_parser.add_argument(
        "--max-samples", "-n", type=int,
        help="Maximum samples to process"
    )
    transfer_parser.add_argument(
        "--visualize", "-v", action="store_true",
        help="Generate and save visualizations"
    )

    # Single image test
    single_parser = subparsers.add_parser(
        "single", help="Run on single image(s) for quick testing"
    )
    single_parser.add_argument(
        "--reference", "-r", required=True,
        help="Path to reference image"
    )
    single_parser.add_argument(
        "--target", "-t",
        help="Path to target image (optional, for transfer)"
    )
    single_parser.add_argument(
        "--prompt-type", "-p", required=True,
        choices=["text", "point", "box", "mask"],
        help="Type of prompt to use"
    )
    single_parser.add_argument(
        "--text", help="Text prompt (for text prompt type)"
    )
    single_parser.add_argument(
        "--point", help="Point coordinates as 'x,y' (for point prompt type)"
    )
    single_parser.add_argument(
        "--box", help="Box as 'x,y,w,h' (for box prompt type)"
    )
    single_parser.add_argument(
        "--mask", help="Path to mask image (for mask prompt type)"
    )
    single_parser.add_argument(
        "--threshold", type=float, default=0.5,
        help="Confidence threshold (default: 0.5)"
    )
    single_parser.add_argument(
        "--output", "-o",
        help="Output image path (displays if not specified)"
    )

    args = parser.parse_args()

    if args.command == "segmentation":
        run_segmentation_benchmark(args)
    elif args.command == "transfer":
        run_transfer_benchmark(args)
    elif args.command == "single":
        run_single(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
