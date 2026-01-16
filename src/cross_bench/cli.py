"""Command-line interface for cross-bench using Typer."""

from pathlib import Path
from typing import Annotated, Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from cross_bench.datasets import CrossImageDataset
from cross_bench.predictor import CrossImagePredictor, PromptType, Prompt, PromptMap, format_prompt_display
from cross_bench.benchmarks import SegmentationBenchmark, ConceptTransferBenchmark
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

    # Create predictor and benchmark
    predictor = CrossImagePredictor(confidence_threshold=threshold)
    benchmark = SegmentationBenchmark(
        predictor=predictor,
        output_dir=output_dir,
        confidence_threshold=threshold,
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

    # Create predictor and benchmark
    predictor = CrossImagePredictor(confidence_threshold=threshold)
    benchmark = ConceptTransferBenchmark(
        predictor=predictor,
        output_dir=output_dir,
        confidence_threshold=threshold,
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

    # Print summary
    console.print("\n[bold]Summary:[/bold]")
    for ptype in prompt_types:
        results = run.filter_by_prompt_type(ptype)
        if results:
            total_ref = sum(r.metadata.get("reference_detections", 0) for r in results)
            total_tgt = sum(r.metadata.get("target_detections", 0) for r in results)
            console.print(f"  [magenta]{ptype}[/magenta]: {len(results)} samples, "
                         f"[cyan]{total_ref}[/cyan] ref detections, "
                         f"[green]{total_tgt}[/green] target detections")

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

        success(f"Saved visualizations to [bold]{vis_dir}[/bold]")


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

    info("Initializing predictor...")
    predictor = CrossImagePredictor(confidence_threshold=threshold)

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
