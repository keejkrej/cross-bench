"""Plotting utilities for visualizing segmentation and transfer results."""

from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.figure import Figure
from matplotlib.axes import Axes

from cross_bench.predictor import SegmentationResult, Prompt, PromptType, PromptMap
from cross_bench.datasets import DatasetSample
from cross_bench.benchmarks.base import BenchmarkResult


# Color palette for masks (colorblind-friendly)
MASK_COLORS = [
    (0.12, 0.47, 0.71, 0.5),  # blue
    (1.00, 0.50, 0.05, 0.5),  # orange
    (0.17, 0.63, 0.17, 0.5),  # green
    (0.84, 0.15, 0.16, 0.5),  # red
    (0.58, 0.40, 0.74, 0.5),  # purple
    (0.55, 0.34, 0.29, 0.5),  # brown
    (0.89, 0.47, 0.76, 0.5),  # pink
    (0.50, 0.50, 0.50, 0.5),  # gray
    (0.74, 0.74, 0.13, 0.5),  # olive
    (0.09, 0.75, 0.81, 0.5),  # cyan
]


def _overlay_mask(
    ax: Axes,
    mask: np.ndarray,
    color: tuple[float, ...] = (0.12, 0.47, 0.71, 0.5),
    contour_color: str = "white",
    contour_width: float = 2,
    contour_linestyle: str = "-",
    contour_only: bool = False,
) -> None:
    """Overlay a mask on an axes with color fill and contour.

    Args:
        ax: Matplotlib axes
        mask: Binary mask array (H, W)
        color: RGBA color for the fill
        contour_color: Color for the contour line
        contour_width: Width of the contour line
        contour_linestyle: Line style for contour ("-" solid, "--" dashed)
        contour_only: If True, only draw contour without fill
    """
    # Squeeze batch dimension if present (handle both 2D and 3D masks)
    if mask.ndim == 3:
        mask = mask.squeeze(0)
    
    # Create colored overlay (skip if contour_only)
    if not contour_only:
        h, w = mask.shape
        overlay = np.zeros((h, w, 4))
        overlay[mask > 0.5] = color
        ax.imshow(overlay)

    # Add contour
    ax.contour(
        mask,
        levels=[0.5],
        colors=[contour_color],
        linewidths=[contour_width],
        linestyles=[contour_linestyle],
    )


def _draw_bbox(
    ax: Axes,
    box: tuple[float, float, float, float],
    color: str = "yellow",
    linewidth: float = 2,
    linestyle: str = "--",
    label: str | None = None,
) -> None:
    """Draw a bounding box on axes.

    Args:
        ax: Matplotlib axes
        box: Box in XYXY format (x1, y1, x2, y2)
        color: Box color
        linewidth: Line width
        linestyle: Line style
        label: Optional label to display
    """
    x1, y1, x2, y2 = box
    rect = patches.Rectangle(
        (x1, y1),
        x2 - x1,
        y2 - y1,
        linewidth=linewidth,
        linestyle=linestyle,
        edgecolor=color,
        facecolor="none",
    )
    ax.add_patch(rect)

    if label:
        ax.text(
            x1, y1 - 5,
            label,
            color=color,
            fontsize=10,
            fontweight="bold",
            va="bottom",
        )


def _draw_point(
    ax: Axes,
    point: tuple[float, float],
    color: str = "yellow",
    marker: str = "o",
    size: int = 100,
    label: str | None = None,
) -> None:
    """Draw a point marker on axes.

    Args:
        ax: Matplotlib axes
        point: (x, y) coordinates
        color: Point color
        marker: Marker style
        size: Marker size
        label: Optional label
    """
    ax.scatter(
        [point[0]], [point[1]],
        c=color,
        marker=marker,
        s=size,
        edgecolors="black",
        linewidths=2,
        zorder=10,
    )

    if label:
        ax.text(
            point[0] + 10, point[1],
            label,
            color=color,
            fontsize=10,
            fontweight="bold",
            va="center",
        )


# Distinct colors for each prompt type visualization
PROMPT_COLORS = {
    PromptType.TEXT: "#3498db",    # blue
    PromptType.POINT: "#2ecc71",   # green
    PromptType.BOX: "#f1c40f",     # yellow
    PromptType.MASK: "#e74c3c",    # red
}


def _hex_to_rgba(hex_color: str, alpha: float = 0.3) -> tuple[float, float, float, float]:
    """Convert hex color to RGBA tuple."""
    hex_color = hex_color.lstrip("#")
    r, g, b = tuple(int(hex_color[i:i+2], 16) / 255.0 for i in (0, 2, 4))
    return (r, g, b, alpha)


def _draw_prompt(
    ax: Axes,
    prompt: Prompt,
    color: str | None = None,
) -> None:
    """Draw a prompt visualization on axes.

    Args:
        ax: Matplotlib axes
        prompt: Prompt to visualize
        color: Color override (uses type-specific color if None)
    """
    # Use type-specific color if not overridden
    color = color or PROMPT_COLORS.get(prompt.prompt_type, "yellow")

    if prompt.prompt_type == PromptType.POINT:
        _draw_point(ax, prompt.value, color=color)

    elif prompt.prompt_type == PromptType.BOX:
        x, y, w, h = prompt.value
        box = (x, y, x + w, y + h)
        _draw_bbox(ax, box, color=color)

    elif prompt.prompt_type == PromptType.MASK:
        _overlay_mask(
            ax,
            prompt.value,
            contour_color=color,
            contour_linestyle="--",  # dashed for prompt masks
            contour_only=True,  # only draw contour, not fill
        )

    elif prompt.prompt_type == PromptType.TEXT:
        # Text prompts are shown in the legend only, not drawn on the image
        pass


def _draw_prompt_legend(
    ax: Axes,
    prompts: PromptMap,
    position: str = "upper left",
) -> None:
    """Draw a legend showing active prompts with visual indicators.

    Args:
        ax: Matplotlib axes
        prompts: PromptMap of active prompts
        position: Legend position ("upper left", "upper right", etc.)
    """
    if not prompts:
        return

    # Build legend entries: (label, color)
    legend_entries = []
    for prompt_type, prompt in prompts.items():
        color = PROMPT_COLORS.get(prompt_type, "yellow")

        if prompt_type == PromptType.TEXT:
            legend_entries.append((f"text: {prompt.value}", color))
        elif prompt_type == PromptType.POINT:
            legend_entries.append(("point: ●", color))
        elif prompt_type == PromptType.BOX:
            legend_entries.append(("box: ╍╍╍", color))
        elif prompt_type == PromptType.MASK:
            legend_entries.append(("mask: ╍╍╍", color))

    if not legend_entries:
        return

    # Position mapping
    positions = {
        "upper left": (0.02, 0.98, "top", "left"),
        "upper right": (0.98, 0.98, "top", "right"),
        "lower left": (0.02, 0.02, "bottom", "left"),
        "lower right": (0.98, 0.02, "bottom", "right"),
    }
    x, y, va, ha = positions.get(position, positions["upper left"])

    # Draw each entry vertically stacked
    y_offset = 0
    for label, color in legend_entries:
        ax.text(
            x, y - y_offset,
            label,
            transform=ax.transAxes,
            color=color,
            fontsize=9,
            fontweight="bold",
            va=va,
            ha=ha,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.7),
        )
        y_offset += 0.045


def plot_segmentation(
    image: Image.Image,
    result: SegmentationResult,
    prompts: PromptMap,
    ax: Axes | None = None,
    title: str | None = None,
    show_scores: bool = True,
    show_boxes: bool = True,
    show_prompt: bool = True,
) -> Axes:
    """Plot segmentation results on an image.

    Args:
        image: PIL Image to display
        result: SegmentationResult with masks and scores
        prompts: PromptMap (dict[PromptType, Prompt]) to visualize
        ax: Matplotlib axes (created if None)
        title: Optional title for the plot
        show_scores: Whether to show confidence scores
        show_boxes: Whether to show bounding boxes
        show_prompt: Whether to show the prompts

    Returns:
        Matplotlib axes with the plot
    """
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(10, 10))

    # Display image
    ax.imshow(image)

    # Draw masks
    for i, (mask, score) in enumerate(zip(result.masks, result.scores)):
        color = MASK_COLORS[i % len(MASK_COLORS)]
        _overlay_mask(ax, mask, color=color)

        if show_scores and result.boxes:
            box = result.boxes[i]
            ax.text(
                box[0], box[1] - 5,
                f"{score:.2f}",
                color="white",
                fontsize=9,
                fontweight="bold",
                bbox=dict(boxstyle="round", facecolor=color[:3], alpha=0.8),
            )

    # Draw bounding boxes
    if show_boxes:
        for i, box in enumerate(result.boxes):
            color_rgb = MASK_COLORS[i % len(MASK_COLORS)][:3]
            _draw_bbox(ax, box, color=color_rgb, linewidth=1.5, linestyle="-")

    # Draw prompts and legend
    if show_prompt and prompts:
        for p in prompts.values():
            _draw_prompt(ax, p)
        _draw_prompt_legend(ax, prompts)

    # Set title
    if title:
        ax.set_title(title, fontsize=12, fontweight="bold")

    ax.axis("off")
    return ax


def plot_transfer_comparison(
    sample: DatasetSample,
    ref_result: SegmentationResult,
    tgt_result: SegmentationResult,
    prompts: PromptMap,
    figsize: tuple[int, int] = (16, 8),
    title: str | None = None,
    ref_iou: Optional[float] = None,
    tgt_iou: Optional[float] = None,
) -> Figure:
    """Plot reference and target segmentation side by side.

    Args:
        sample: Dataset sample with images
        ref_result: Segmentation result for reference image
        tgt_result: Segmentation result for target image
        prompts: PromptMap (dict[PromptType, Prompt]) used for segmentation
        figsize: Figure size
        title: Optional overall title
        ref_iou: IoU score for reference image
        tgt_iou: IoU score for target image

    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Get actual image names from file paths (with extension)
    ref_name = sample.reference_image_path.name
    tgt_name = sample.target_image_path.name
    
    # Reference image
    ref_title = f"Reference: {ref_name}"
    if ref_iou is not None:
        ref_title += f" (IoU: {ref_iou:.3f})"
    
    plot_segmentation(
        sample.reference_image,
        ref_result,
        prompts=prompts,
        ax=axes[0],
        title=ref_title,
        show_prompt=True,
    )

    # Target image
    tgt_title = f"Target: {tgt_name}"
    if tgt_iou is not None:
        tgt_title += f" (IoU: {tgt_iou:.3f})"
    
    plot_segmentation(
        sample.target_image,
        tgt_result,
        prompts={},  # Don't show prompts on target
        ax=axes[1],
        title=tgt_title,
        show_prompt=False,
    )
    
    # Draw target ground truth mask contour if available
    if sample.target_mask is not None:
        _overlay_mask(
            axes[1],
            sample.target_mask,
            contour_color="lime",
            contour_linestyle="--",
            contour_only=True,
        )

    if title:
        fig.suptitle(title, fontsize=14, fontweight="bold")

    plt.tight_layout()
    return fig


def plot_benchmark_grid(
    sample: DatasetSample,
    results: dict[str, BenchmarkResult],
    figsize: tuple[int, int] = (20, 10),
    title: str | None = None,
) -> Figure:
    """Plot benchmark results for multiple prompt types in a grid.

    Args:
        sample: Dataset sample with images
        results: Dictionary mapping prompt_type to BenchmarkResult
        figsize: Figure size
        title: Optional overall title

    Returns:
        Matplotlib figure
    """
    prompt_types = list(results.keys())
    n_prompts = len(prompt_types)

    # 2 rows: reference and target; n_prompts columns
    fig, axes = plt.subplots(2, n_prompts, figsize=figsize)

    if n_prompts == 1:
        axes = axes.reshape(2, 1)

    for col, ptype in enumerate(prompt_types):
        result = results[ptype]

        ref_result = result.results.get("reference")
        tgt_result = result.results.get("target")

        # Reference row
        if ref_result:
            # Calculate IoU
            ref_iou = result.calculate_iou("reference")
            ref_title = f"{ptype.upper()} - Reference ({ref_result.num_detections})"
            if ref_iou is not None:
                ref_title += f" [IoU: {ref_iou:.3f}]"
            
            plot_segmentation(
                sample.reference_image,
                ref_result,
                prompts=ref_result.prompts,
                ax=axes[0, col],
                title=ref_title,
            )
        else:
            axes[0, col].text(0.5, 0.5, "No result", ha="center", va="center")
            axes[0, col].axis("off")

        # Target row
        if tgt_result:
            # Calculate IoU
            tgt_iou = result.calculate_iou("target")
            tgt_title = f"{ptype.upper()} - Target ({tgt_result.num_detections})"
            if tgt_iou is not None:
                tgt_title += f" [IoU: {tgt_iou:.3f}]"
            
            plot_segmentation(
                sample.target_image,
                tgt_result,
                ax=axes[1, col],
                title=tgt_title,
                show_prompt=False,
            )
        else:
            axes[1, col].text(0.5, 0.5, "No result", ha="center", va="center")
            axes[1, col].axis("off")

    if title:
        fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)

    plt.tight_layout()
    return fig


def create_benchmark_figure(
    sample: DatasetSample,
    benchmark_result: BenchmarkResult,
    show_ground_truth: bool = True,
    figsize: tuple[int, int] = (18, 6),
) -> Figure:
    """Create a comprehensive figure for a single benchmark result.

    Shows: Ground truth | Reference result | Target result (if available)

    Args:
        sample: Dataset sample
        benchmark_result: Benchmark result to visualize
        show_ground_truth: Whether to show ground truth mask
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    has_target = "target" in benchmark_result.results

    n_cols = 3 if has_target else 2
    if show_ground_truth:
        n_cols += 1

    fig, axes = plt.subplots(1, n_cols, figsize=figsize)
    col = 0

    # Ground truth
    if show_ground_truth:
        axes[col].imshow(sample.reference_image)
        _overlay_mask(
            axes[col],
            sample.reference_mask,
            color=(0.0, 1.0, 0.0, 0.4),  # green
            contour_color="lime",
        )
        axes[col].set_title("Ground Truth Mask", fontsize=11, fontweight="bold")
        axes[col].axis("off")
        col += 1

    # Reference result
    ref_result = benchmark_result.results.get("reference")
    if ref_result:
        plot_segmentation(
            sample.reference_image,
            ref_result,
            prompts=ref_result.prompts,
            ax=axes[col],
            title=f"Reference ({benchmark_result.prompt_type})",
        )
    col += 1

    # Target result
    if has_target:
        tgt_result = benchmark_result.results.get("target")
        if tgt_result:
            plot_segmentation(
                sample.target_image,
                tgt_result,
                ax=axes[col],
                title=f"Target Transfer ({tgt_result.num_detections} found)",
                show_prompt=False,
            )

    # Overall title
    title = f"Sample: {sample.sample_id} | Prompt: {benchmark_result.prompt_type}"
    if sample.category:
        title += f" | Category: {sample.category}"
    fig.suptitle(title, fontsize=12, fontweight="bold", y=1.02)

    plt.tight_layout()
    return fig


def save_figure(
    fig: Figure,
    output_path: Path | str,
    dpi: int = 150,
    close: bool = True,
) -> None:
    """Save a figure to file.

    Args:
        fig: Matplotlib figure
        output_path: Path to save the figure
        dpi: Resolution
        close: Whether to close the figure after saving
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    if close:
        plt.close(fig)
