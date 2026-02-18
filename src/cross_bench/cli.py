"""Command-line interface for cross-bench using Typer."""

from pathlib import Path
from typing import Annotated, Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from cross_bench.datasets import CrossImageDataset, COCODetectionDataset
from cross_bench.predictor import CrossImagePredictor, PromptType, Prompt, PromptMap, format_prompt_display
from cross_bench.benchmarks import SegmentationBenchmark, ConceptTransferBenchmark, DetectionBenchmark
from cross_bench.visualization import (
    plot_transfer_comparison,
    create_benchmark_figure,
    save_figure,
)

console = Console()

app = typer.Typer(
    name="cross-bench",
    help="Cross-Bench: Cross-image segmentation benchmarking tool for SAM3",
    add_completion=False,
)


def info(msg: str) -> None:
    """Print info message in cyan."""
    console.print(f"[cyan]{msg}[/cyan]")


def success(msg: str) -> None:
    """Print success message in green."""
    console.print(f"[green]✓[/green] {msg}")


def warning(msg: str) -> None:
    """Print warning message in yellow."""
    console.print(f"[yellow]⚠[/yellow] {msg}")


def error(msg: str) -> None:
    """Print error message in red."""
    console.print(f"[red]✗[/red] {msg}")


def highlight(label: str, value: str) -> None:
    """Print label: value with highlighted value."""
    console.print(f"[dim]{label}:[/dim] [bold]{value}[/bold]")


@app.command()
def segmentation(
    dataset: Annotated[Path, typer.Option("--dataset", "-d", help="Path to dataset directory")],
    output: Annotated[Optional[Path], typer.Option("--output", "-o", help="Output directory for results")] = None,
    prompts: Annotated[str, typer.Option("--prompts", "-p", help="Comma-separated prompt types")] = "mask,box,point",
    threshold: Annotated[float, typer.Option("--threshold", "-t", help="Confidence threshold")] = 0.5,
    max_samples: Annotated[Optional[int], typer.Option("--max-samples", "-n", help="Maximum samples to process")] = None,
    visualize: Annotated[bool, typer.Option("--visualize", "-v", help="Generate and save visualizations")] = False,
    # Mask encoding options
    mask_encoding: Annotated[str, typer.Option("--mask-encoding", "-m", help="Mask encoding method: default, box, grid_simple, grid_hybrid")] = "default",
    grid_spacing: Annotated[int, typer.Option("--grid-spacing", help="Grid spacing for grid_simple")] = 16,
    grid_max_points: Annotated[int, typer.Option("--grid-max-points", help="Max points for grid methods")] = 50,
    grid_boundary_ratio: Annotated[float, typer.Option("--grid-boundary-ratio", help="Boundary ratio for grid_hybrid")] = 0.3,
    # Geometry encoder flags
    points_direct: Annotated[bool, typer.Option("--points-direct/--no-points-direct")] = True,
    points_pool: Annotated[bool, typer.Option("--points-pool/--no-points-pool")] = True,
    points_pos: Annotated[bool, typer.Option("--points-pos/--no-points-pos")] = True,
    boxes_direct: Annotated[bool, typer.Option("--boxes-direct/--no-boxes-direct")] = True,
    boxes_pool: Annotated[bool, typer.Option("--boxes-pool/--no-boxes-pool")] = True,
    boxes_pos: Annotated[bool, typer.Option("--boxes-pos/--no-boxes-pos")] = True,
    masks_direct: Annotated[bool, typer.Option("--masks-direct/--no-masks-direct")] = True,
    masks_pool: Annotated[bool, typer.Option("--masks-pool/--no-masks-pool")] = True,
    masks_pos: Annotated[bool, typer.Option("--masks-pos/--no-masks-pos")] = True,
) -> None:
    """Run the segmentation benchmark on a dataset."""
    # Load dataset
    if (dataset / "manifest.json").exists():
        ds = CrossImageDataset.from_manifest(dataset / "manifest.json")
    else:
        ds = CrossImageDataset.from_directory(dataset)

    success(f"Loaded dataset: [bold]{ds.name}[/bold] ({len(ds)} samples)")

    # Setup output directory
    output_dir = output if output else Path("results") / "segmentation"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build geometry encoder config
    geo_config = {
        'points_direct_project': points_direct,
        'points_pool': points_pool,
        'points_pos_enc': points_pos,
        'boxes_direct_project': boxes_direct,
        'boxes_pool': boxes_pool,
        'boxes_pos_enc': boxes_pos,
        'masks_direct_project': masks_direct,
        'masks_pool': masks_pool,
        'masks_pos_enc': masks_pos,
    }
    
    # Build mask encoding params
    mask_encoding_params = {
        'spacing': grid_spacing,
        'max_points': grid_max_points,
        'boundary_ratio': grid_boundary_ratio,
    }
    
    # Create predictor and benchmark
    predictor = CrossImagePredictor(confidence_threshold=threshold, geo_config=geo_config)
    benchmark = SegmentationBenchmark(
        predictor=predictor,
        output_dir=output_dir,
        confidence_threshold=threshold,
        mask_encoding_method=mask_encoding,
        mask_encoding_params=mask_encoding_params,
    )

    # Parse prompt types
    prompt_types = prompts.split(",")

    # Run benchmark
    info(f"Running segmentation benchmark with prompts: [bold]{prompt_types}[/bold]")
    run = benchmark.run(
        ds,
        prompt_types=prompt_types,
        max_samples=max_samples,
        verbose=True,
    )

    success(f"Completed [bold]{len(run)}[/bold] benchmark results")

    # Calculate and print scores
    scores = benchmark.calculate_scores(run)
    
    console.print("\n[bold]Summary:[/bold]")
    for ptype in prompt_types:
        results = run.filter_by_prompt_type(ptype)
        if results:
            total_ref = sum(r.metadata.get("reference_detections", 0) for r in results)
            total_tgt = sum(r.metadata.get("target_detections", 0) for r in results)
            console.print(f"  [magenta]{ptype}[/magenta]: {len(results)} samples, "
                         f"[cyan]{total_ref}[/cyan] ref detections, "
                         f"[green]{total_tgt}[/green] target detections")
    
    console.print("\n[bold]Scores:[/bold]")
    console.print(f"  Reference IoU: [cyan]{scores['reference_iou_avg']:.3f}[/cyan] "
                 f"({scores['reference_iou_count']} samples)")
    console.print(f"  Target IoU: [green]{scores['target_iou_avg']:.3f}[/green] "
                 f"({scores['target_iou_count']} samples)")

    # Save visualizations
    if visualize:
        info("Generating visualizations...")
        vis_dir = output_dir / "visualizations"
        vis_dir.mkdir(exist_ok=True)

        for result in run:
            sample = ds[0]
            for s in ds:
                if s.sample_id == result.sample_id:
                    sample = s
                    break

            fig = create_benchmark_figure(sample, result, show_ground_truth=True)
            save_figure(fig, vis_dir / f"{result.sample_id}_{result.prompt_type}.png")

        success(f"Saved visualizations to [bold]{vis_dir}[/bold]")


@app.command()
def transfer(
    dataset: Annotated[Path, typer.Option("--dataset", "-d", help="Path to dataset directory")],
    output: Annotated[Optional[Path], typer.Option("--output", "-o", help="Output directory for results")] = None,
    prompts: Annotated[str, typer.Option("--prompts", "-p", help="Comma-separated prompt types")] = "mask,box,point",
    threshold: Annotated[float, typer.Option("--threshold", "-t", help="Confidence threshold")] = 0.5,
    max_samples: Annotated[Optional[int], typer.Option("--max-samples", "-n", help="Maximum samples to process")] = None,
    visualize: Annotated[bool, typer.Option("--visualize", "-v", help="Generate and save visualizations")] = False,
    # Mask encoding options
    mask_encoding: Annotated[str, typer.Option("--mask-encoding", "-m", help="Mask encoding method: default, box, grid_simple, grid_hybrid")] = "default",
    grid_spacing: Annotated[int, typer.Option("--grid-spacing", help="Grid spacing for grid_simple")] = 16,
    grid_max_points: Annotated[int, typer.Option("--grid-max-points", help="Max points for grid methods")] = 50,
    grid_boundary_ratio: Annotated[float, typer.Option("--grid-boundary-ratio", help="Boundary ratio for grid_hybrid")] = 0.3,
    # Geometry encoder flags
    points_direct: Annotated[bool, typer.Option("--points-direct/--no-points-direct")] = True,
    points_pool: Annotated[bool, typer.Option("--points-pool/--no-points-pool")] = True,
    points_pos: Annotated[bool, typer.Option("--points-pos/--no-points-pos")] = True,
    boxes_direct: Annotated[bool, typer.Option("--boxes-direct/--no-boxes-direct")] = True,
    boxes_pool: Annotated[bool, typer.Option("--boxes-pool/--no-boxes-pool")] = True,
    boxes_pos: Annotated[bool, typer.Option("--boxes-pos/--no-boxes-pos")] = True,
    masks_direct: Annotated[bool, typer.Option("--masks-direct/--no-masks-direct")] = True,
    masks_pool: Annotated[bool, typer.Option("--masks-pool/--no-masks-pool")] = True,
    masks_pos: Annotated[bool, typer.Option("--masks-pos/--no-masks-pos")] = True,
) -> None:
    """Run the concept transfer benchmark on a dataset."""
    # Load dataset
    if (dataset / "manifest.json").exists():
        ds = CrossImageDataset.from_manifest(dataset / "manifest.json")
    else:
        ds = CrossImageDataset.from_directory(dataset)

    success(f"Loaded dataset: [bold]{ds.name}[/bold] ({len(ds)} samples)")

    # Setup output directory
    output_dir = output if output else Path("results") / "transfer"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build geometry encoder config
    geo_config = {
        'points_direct_project': points_direct,
        'points_pool': points_pool,
        'points_pos_enc': points_pos,
        'boxes_direct_project': boxes_direct,
        'boxes_pool': boxes_pool,
        'boxes_pos_enc': boxes_pos,
        'masks_direct_project': masks_direct,
        'masks_pool': masks_pool,
        'masks_pos_enc': masks_pos,
    }

    # Build mask encoding params
    mask_encoding_params = {
        'spacing': grid_spacing,
        'max_points': grid_max_points,
        'boundary_ratio': grid_boundary_ratio,
    }

    # Create predictor and benchmark
    predictor = CrossImagePredictor(confidence_threshold=threshold, geo_config=geo_config)
    benchmark = ConceptTransferBenchmark(
        predictor=predictor,
        output_dir=output_dir,
        confidence_threshold=threshold,
        mask_encoding_method=mask_encoding,
        mask_encoding_params=mask_encoding_params,
    )

    # Parse prompt types
    prompt_types = prompts.split(",")

    # Run benchmark
    info(f"Running concept transfer benchmark with prompts: [bold]{prompt_types}[/bold]")
    run = benchmark.run(
        ds,
        prompt_types=prompt_types,
        max_samples=max_samples,
        verbose=True,
    )

    success(f"Completed [bold]{len(run)}[/bold] benchmark results")

    # Calculate and print scores
    scores = benchmark.calculate_scores(run)
    
    console.print("\n[bold]Summary:[/bold]")
    for ptype in prompt_types:
        results = run.filter_by_prompt_type(ptype)
        if results:
            total_ref = sum(r.metadata.get("reference_detections", 0) for r in results)
            total_tgt = sum(r.metadata.get("target_detections", 0) for r in results)
            console.print(f"  [magenta]{ptype}[/magenta]: {len(results)} samples, "
                         f"[cyan]{total_ref}[/cyan] ref detections, "
                         f"[green]{total_tgt}[/green] target detections")
    
    console.print("\n[bold]Scores:[/bold]")
    console.print(f"  Reference IoU: [cyan]{scores['reference_iou_avg']:.3f}[/cyan] "
                 f"({scores['reference_iou_count']} samples)")
    console.print(f"  Target IoU: [green]{scores['target_iou_avg']:.3f}[/green] "
                 f"({scores['target_iou_count']} samples)")

    # Save visualizations
    if visualize:
        info("Generating visualizations...")
        vis_dir = output_dir / "visualizations"
        vis_dir.mkdir(exist_ok=True)

        for result in run:
            sample = None
            for s in ds:
                if s.sample_id == result.sample_id:
                    sample = s
                    break

            if sample is None:
                continue

            ref_result = result.results.get("reference")
            tgt_result = result.results.get("target")

            if ref_result and tgt_result:
                # Calculate IoU scores
                ref_iou = result.calculate_iou("reference")
                tgt_iou = result.calculate_iou("target")
                
                fig = plot_transfer_comparison(
                    sample,
                    ref_result,
                    tgt_result,
                    prompts=ref_result.prompts,
                    title=f"Concept Transfer - {result.prompt_type.upper()}",
                    ref_iou=ref_iou,
                    tgt_iou=tgt_iou,
                )
                save_figure(
                    fig,
                    vis_dir / f"{result.sample_id}_{result.prompt_type}_transfer.png"
                )

        success(f"Saved visualizations to [bold]{vis_dir}[/bold]")


@app.command()
def detection(
    dataset: Annotated[
        str,
        typer.Option("--dataset", "-d", help="COCO root path or HF repo (detection-datasets/coco) with --from-hf"),
    ] = "detection-datasets/coco",
    split: Annotated[str, typer.Option("--split", "-s", help="Split: val/validation (local) or train/validation (HF)")] = "val",
    from_hf: Annotated[bool, typer.Option("--from-hf/--no-from-hf", help="Load from Hugging Face Hub (uses cache)")] = False,
    output: Annotated[Optional[Path], typer.Option("--output", "-o", help="Output directory for results")] = None,
    prompts: Annotated[str, typer.Option("--prompts", "-p", help="Comma-separated prompt types")] = "box,point,mask",
    threshold: Annotated[float, typer.Option("--threshold", "-t", help="Confidence threshold")] = 0.5,
    max_samples: Annotated[Optional[int], typer.Option("--max-samples", "-n", help="Maximum samples to process")] = 100,
    categories: Annotated[Optional[str], typer.Option("--categories", "-c", help="Comma-separated category names to filter")] = None,
    seed: Annotated[int, typer.Option("--seed", help="Random seed for sampling")] = 42,
    # Geometry encoder flags
    points_direct: Annotated[bool, typer.Option("--points-direct/--no-points-direct")] = True,
    points_pool: Annotated[bool, typer.Option("--points-pool/--no-points-pool")] = True,
    points_pos: Annotated[bool, typer.Option("--points-pos/--no-points-pos")] = True,
    boxes_direct: Annotated[bool, typer.Option("--boxes-direct/--no-boxes-direct")] = True,
    boxes_pool: Annotated[bool, typer.Option("--boxes-pool/--no-boxes-pool")] = True,
    boxes_pos: Annotated[bool, typer.Option("--boxes-pos/--no-boxes-pos")] = True,
    masks_direct: Annotated[bool, typer.Option("--masks-direct/--no-masks-direct")] = True,
    masks_pool: Annotated[bool, typer.Option("--masks-pool/--no-masks-pool")] = True,
    masks_pos: Annotated[bool, typer.Option("--masks-pos/--no-masks-pos")] = True,
) -> None:
    """Run object detection benchmark on COCO-format dataset.

    Evaluates cross-image concept transfer for detection: given a reference object
    (box/point/mask prompt), transfer to target images and measure mAP, AP@50, AP@75.

    Datasets:
    - Hugging Face: --from-hf (uses detection-datasets/coco from cache)
    - Local COCO: path with images/ and annotations/instances_val2017.json
    """
    if from_hf:
        hf_name = str(dataset) if dataset else "detection-datasets/coco"
        info(f"Loading from Hugging Face: [bold]{hf_name}[/bold]")
        hf_split = "val" if split in ("val", "validation") else split
        cat_list = None
        if categories:
            cat_list = [c.strip() for c in categories.split(",") if c.strip()]
        ds = COCODetectionDataset.from_huggingface(
            name=hf_name,
            split=hf_split,
            max_samples=max_samples,
            categories=cat_list,
            seed=seed,
        )
    else:
        # Check for local COCO format
        dataset_path = Path(dataset)
        annot_candidates = [
            dataset_path / "annotations" / f"instances_{split}2017.json",
            dataset_path / f"annotations/instances_{split}.json",
        ]
        annot_path = None
        for p in annot_candidates:
            if p.exists():
                annot_path = p
                break

        if annot_path is None:
            # Suggest trying HF
            error(
                "COCO annotations not found. Try: cross-bench detection --from-hf\n"
                "Or ensure dataset/annotations/instances_val2017.json exists."
            )
            raise typer.Exit(1)

        cat_list = None
        if categories:
            cat_list = [c.strip() for c in categories.split(",") if c.strip()]
        ds = COCODetectionDataset.from_coco(
            root=dataset_path,
            split=split,
            max_samples=max_samples,
            categories=cat_list,
            seed=seed,
        )

    success(f"Loaded COCO detection dataset: [bold]{ds.name}[/bold] ({len(ds)} samples)")

    output_dir = output if output else Path("results") / "detection"
    output_dir.mkdir(parents=True, exist_ok=True)

    geo_config = {
        "points_direct_project": points_direct,
        "points_pool": points_pool,
        "points_pos_enc": points_pos,
        "boxes_direct_project": boxes_direct,
        "boxes_pool": boxes_pool,
        "boxes_pos_enc": boxes_pos,
        "masks_direct_project": masks_direct,
        "masks_pool": masks_pool,
        "masks_pos_enc": masks_pos,
    }

    predictor = CrossImagePredictor(confidence_threshold=threshold, geo_config=geo_config)
    benchmark = DetectionBenchmark(
        predictor=predictor,
        output_dir=output_dir,
        confidence_threshold=threshold,
    )

    prompt_types = prompts.split(",")
    info(f"Running detection benchmark with prompts: [bold]{prompt_types}[/bold]")
    run = benchmark.run(ds, prompt_types=prompt_types, max_samples=max_samples, verbose=True)

    success(f"Completed [bold]{len(run)}[/bold] benchmark results")

    scores = benchmark.calculate_scores(run)

    console.print("\n[bold]Detection Metrics:[/bold]")
    console.print(f"  mAP@50:  [cyan]{scores['mAP50']:.3f}[/cyan]")
    console.print(f"  mAP@75:  [cyan]{scores['mAP75']:.3f}[/cyan]")
    console.print(f"  Mean IoU: [green]{scores['mean_iou_avg']:.3f}[/green]")
    console.print(f"  Precision: [green]{scores['precision_avg']:.3f}[/green]")
    console.print(f"  Recall: [green]{scores['recall_avg']:.3f}[/green]")
    console.print(f"  Samples: {scores['total_samples']}, GT objects: {scores['n_gt_total']}")


@app.command()
def single(
    reference: Annotated[Path, typer.Option("--reference", "-r", help="Path to reference image")],
    target: Annotated[Optional[Path], typer.Option("--target", "-t", help="Path to target image (for transfer)")] = None,
    text: Annotated[Optional[str], typer.Option("--text", help="Text prompt content")] = None,
    point: Annotated[Optional[str], typer.Option("--point", help="Point coordinates as 'x,y' (can use multiple)")] = None,
    box: Annotated[Optional[str], typer.Option("--box", help="Box as 'x,y,w,h'")] = None,
    mask: Annotated[Optional[Path], typer.Option("--mask", help="Path to mask image")] = None,
    threshold: Annotated[float, typer.Option("--threshold", help="Confidence threshold")] = 0.5,
    output: Annotated[Optional[Path], typer.Option("--output", "-o", help="Output image path")] = None,
    # Geometry encoder flags
    points_direct: Annotated[bool, typer.Option("--points-direct/--no-points-direct", help="Points direct projection")] = True,
    points_pool: Annotated[bool, typer.Option("--points-pool/--no-points-pool", help="Points pooling")] = True,
    points_pos: Annotated[bool, typer.Option("--points-pos/--no-points-pos", help="Points position encoding")] = True,
    boxes_direct: Annotated[bool, typer.Option("--boxes-direct/--no-boxes-direct", help="Boxes direct projection")] = True,
    boxes_pool: Annotated[bool, typer.Option("--boxes-pool/--no-boxes-pool", help="Boxes pooling")] = True,
    boxes_pos: Annotated[bool, typer.Option("--boxes-pos/--no-boxes-pos", help="Boxes position encoding")] = True,
    masks_direct: Annotated[bool, typer.Option("--masks-direct/--no-masks-direct", help="Masks direct projection")] = True,
    masks_pool: Annotated[bool, typer.Option("--masks-pool/--no-masks-pool", help="Masks pooling")] = True,
    masks_pos: Annotated[bool, typer.Option("--masks-pos/--no-masks-pos", help="Masks position encoding")] = True,
) -> None:
    """Run on single image(s) for quick testing.

    Prompt type is inferred from provided arguments. Multiple prompts can be combined.

    Examples:
        cross-bench single -r ref.jpg --mask mask.png
        cross-bench single -r ref.jpg --point 100,200
        cross-bench single -r ref.jpg --box 50,50,100,100
        cross-bench single -r ref.jpg --text "a cat"
        cross-bench single -r ref.jpg --mask mask.png --point 100,200  # combined
    """
    from PIL import Image
    import numpy as np
    import matplotlib.pyplot as plt

    # Collect all prompts as PromptMap: dict[PromptType, Prompt]
    prompts: PromptMap = {}

    if text:
        prompts[PromptType.TEXT] = Prompt.from_text(text)

    if point:
        x, y = map(float, point.split(","))
        prompts[PromptType.POINT] = Prompt.from_point(x, y)

    if box:
        x, y, w, h = map(float, box.split(","))
        prompts[PromptType.BOX] = Prompt.from_box(x, y, w, h)

    if mask:
        mask_img = Image.open(mask).convert("L")
        mask_arr = (np.array(mask_img) > 128).astype(np.float32)
        prompts[PromptType.MASK] = Prompt.from_mask(mask_arr)

    if prompts:
        display_names = [format_prompt_display(pt, p) for pt, p in prompts.items()]
        console.print(f"[bold]Prompts:[/bold] {', '.join(display_names)}")
    else:
        console.print("[bold]Prompts:[/bold] [dim](empty)[/dim]")

    # Load images
    info(f"Loading reference: [bold]{reference}[/bold]")
    ref_img = Image.open(reference).convert("RGB")

    tgt_img = None
    if target:
        info(f"Loading target: [bold]{target}[/bold]")
        tgt_img = Image.open(target).convert("RGB")

    # Build geometry encoder config
    geo_config = {
        'points_direct_project': points_direct,
        'points_pool': points_pool,
        'points_pos_enc': points_pos,
        'boxes_direct_project': boxes_direct,
        'boxes_pool': boxes_pool,
        'boxes_pos_enc': boxes_pos,
        'masks_direct_project': masks_direct,
        'masks_pool': masks_pool,
        'masks_pos_enc': masks_pos,
    }
    
    info("Initializing predictor...")
    predictor = CrossImagePredictor(confidence_threshold=threshold, geo_config=geo_config)

    if len(prompts) > 1:
        info(f"Combining {len(prompts)} prompts together")

    if tgt_img:
        # Run transfer
        info("Running cross-image transfer...")
        ref_result, tgt_result = predictor.segment_and_transfer(
            ref_img, tgt_img, prompts, threshold=threshold
        )

        console.print()
        console.print(Panel(
            f"[cyan]Reference:[/cyan] [bold]{ref_result.num_detections}[/bold] detections\n"
            f"[green]Target:[/green] [bold]{tgt_result.num_detections}[/bold] detections",
            title="[bold]Results[/bold]",
            border_style="blue"
        ))

        if output:
            from cross_bench.visualization import plot_segmentation

            # Calculate aspect ratios (height/width) for both images
            ref_aspect = ref_img.height / ref_img.width
            tgt_aspect = tgt_img.height / tgt_img.width
            # Use the taller aspect ratio so both fit in same-height boxes
            max_aspect = max(ref_aspect, tgt_aspect)

            # Create figure with 2 columns
            fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(16, 8))

            # Set same box aspect for both image axes (ensures equal width)
            ax_left.set_box_aspect(max_aspect)
            ax_right.set_box_aspect(max_aspect)

            # Plot images
            plot_segmentation(ref_img, ref_result, prompts=prompts, ax=ax_left,
                            title="Reference")
            plot_segmentation(tgt_img, tgt_result, prompts={}, ax=ax_right,
                            title="Target", show_prompt=False)
            plt.savefig(output, dpi=150, bbox_inches="tight")
            success(f"Saved to [bold]{output}[/bold]")
        else:
            plt.show()
    else:
        # Run segmentation only
        info("Running segmentation...")
        result = predictor.segment(ref_img, prompts, threshold=threshold)

        console.print()
        console.print(Panel(
            f"[bold]{result.num_detections}[/bold] detections",
            title="[bold]Results[/bold]",
            border_style="blue"
        ))

        if output:
            from cross_bench.visualization import plot_segmentation
            fig, ax = plt.subplots(1, 1, figsize=(10, 10))
            plot_segmentation(ref_img, result, prompts=prompts, ax=ax)
            plt.savefig(output, dpi=150, bbox_inches="tight")
            success(f"Saved to [bold]{output}[/bold]")
        else:
            plt.show()


def main() -> None:
    """Entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()
