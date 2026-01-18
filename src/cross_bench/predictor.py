"""Cross-image predictor wrapper for SAM3.

This module provides a clean interface to SAM3's cross-image segmentation
capabilities, wrapping the model and processor from the SAM3 fork.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Any

import numpy as np
import torch
from PIL import Image


class PromptType(Enum):
    """Types of prompts supported for segmentation."""
    TEXT = "text"
    POINT = "point"
    BOX = "box"
    MASK = "mask"

    def display_color(self) -> str:
        """Return rich color for this prompt type."""
        colors = {
            PromptType.TEXT: "blue",
            PromptType.POINT: "green",
            PromptType.BOX: "yellow",
            PromptType.MASK: "magenta",
        }
        return colors.get(self, "white")


# Type alias for prompt collections
PromptMap = dict[PromptType, "Prompt"]


def format_prompt_display(prompt_type: PromptType, prompt: "Prompt") -> str:
    """Generate a rich-formatted display string for a prompt.

    Args:
        prompt_type: The type of the prompt
        prompt: The prompt object

    Returns:
        Rich-formatted string like "[blue]text[/blue]('cat')"
    """
    color = prompt_type.display_color()
    name = prompt_type.value

    if prompt_type == PromptType.TEXT:
        return f"[{color}]{name}[/{color}]('{prompt.value}')"
    elif prompt_type == PromptType.POINT:
        x, y = prompt.value
        return f"[{color}]{name}[/{color}]({x},{y})"
    elif prompt_type == PromptType.BOX:
        x, y, w, h = prompt.value
        return f"[{color}]{name}[/{color}]({x},{y},{w},{h})"
    elif prompt_type == PromptType.MASK:
        return f"[{color}]{name}[/{color}]"
    else:
        return f"[{color}]{name}[/{color}]"


@dataclass
class Prompt:
    """A segmentation prompt.

    Attributes:
        prompt_type: Type of the prompt
        value: The prompt value (text string, point coords, box coords, or mask)
        label: Whether this is a positive (True) or negative (False) prompt
    """
    prompt_type: PromptType
    value: Any
    label: bool = True

    @classmethod
    def from_text(cls, text: str) -> "Prompt":
        """Create a text prompt."""
        return cls(PromptType.TEXT, text, True)

    @classmethod
    def from_point(cls, x: float, y: float, label: bool = True) -> "Prompt":
        """Create a point prompt.

        Args:
            x: X coordinate of the point
            y: Y coordinate of the point
            label: True for foreground, False for background
        """
        return cls(PromptType.POINT, (x, y), label)

    @classmethod
    def from_box(cls, x: float, y: float, w: float, h: float) -> "Prompt":
        """Create a box prompt in XYWH format.

        Args:
            x: X coordinate of top-left corner
            y: Y coordinate of top-left corner
            w: Width of the box
            h: Height of the box
        """
        return cls(PromptType.BOX, (x, y, w, h), True)

    @classmethod
    def from_mask(cls, mask: np.ndarray) -> "Prompt":
        """Create a mask prompt.

        Args:
            mask: Binary mask array (H, W) with 1 for object, 0 for background
        """
        return cls(PromptType.MASK, mask, True)


@dataclass
class SegmentationResult:
    """Result of a segmentation operation.

    Attributes:
        masks: List of predicted binary masks
        scores: Confidence scores for each mask
        boxes: Bounding boxes for each mask (XYXY format)
        prompts: The prompts used for segmentation
        image_size: Original image size (width, height)
    """
    masks: list[np.ndarray]
    scores: list[float]
    boxes: list[tuple[float, float, float, float]]
    prompts: PromptMap = field(default_factory=dict)
    image_size: tuple[int, int] = (0, 0)

    @property
    def num_detections(self) -> int:
        """Number of detected objects."""
        return len(self.masks)

    def filter_by_score(self, threshold: float) -> "SegmentationResult":
        """Filter results by confidence score.

        Args:
            threshold: Minimum score threshold

        Returns:
            New SegmentationResult with filtered detections
        """
        filtered_masks = []
        filtered_scores = []
        filtered_boxes = []

        for mask, score, box in zip(self.masks, self.scores, self.boxes):
            if score >= threshold:
                filtered_masks.append(mask)
                filtered_scores.append(score)
                filtered_boxes.append(box)

        return SegmentationResult(
            masks=filtered_masks,
            scores=filtered_scores,
            boxes=filtered_boxes,
            prompts=self.prompts,
            image_size=self.image_size,
        )


@dataclass
class ConceptEmbedding:
    """Extracted concept embedding from a prompted segmentation.

    This captures the "concept" of an object from a reference image,
    which can be transferred to find similar objects in other images.

    Attributes:
        prompt_embedding: The prompt embedding tensor
        prompt_mask: The prompt mask tensor
        source_prompts: The original prompts used
        source_image_size: Size of the source image
    """
    prompt_embedding: torch.Tensor
    prompt_mask: torch.Tensor
    source_prompts: PromptMap = field(default_factory=dict)
    source_image_size: tuple[int, int] = (0, 0)


class CrossImagePredictor:
    """Wrapper for SAM3 cross-image segmentation.

    This class provides a clean interface for:
    1. Segmenting objects in an image using various prompt types
    2. Extracting concept embeddings from prompted segmentations
    3. Transferring concepts to find similar objects in other images

    Usage:
        from cross_bench.predictor import CrossImagePredictor, PromptType, Prompt

        predictor = CrossImagePredictor()

        # Segment with prompts (PromptMap: dict[PromptType, Prompt])
        prompts = {
            PromptType.MASK: Prompt.from_mask(mask_array),
            PromptType.POINT: Prompt.from_point(100, 200),
        }
        result = predictor.segment(reference_image, prompts)

        # Extract concept and transfer to target
        concept = predictor.extract_concept(reference_image, prompts)
        transfer_result = predictor.transfer_concept(target_image, concept)
    """

    def __init__(
        self,
        model: Optional[Any] = None,
        device: Optional[str] = None,
        confidence_threshold: float = 0.5,
        geo_config: Optional[dict] = None,
    ):
        """Initialize the predictor.

        Args:
            model: Pre-loaded SAM3 model (if None, will be loaded on first use)
            device: Device to use ('cuda' or 'cpu', auto-detected if None)
            confidence_threshold: Default confidence threshold for detections
            geo_config: Geometry encoder config dict with keys like 'points_direct_project',
                       'points_pool', 'points_pos_enc', 'boxes_*', 'masks_*'
        """
        self._model = model
        self._processor = None
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.confidence_threshold = confidence_threshold
        self._geo_config = geo_config
        self._initialized = False

    def _ensure_initialized(self) -> None:
        """Ensure model and processor are loaded."""
        if self._initialized:
            return

        # Import SAM3 components
        try:
            from sam3 import build_sam3_image_model
            from sam3.model.sam3_image_processor import Sam3Processor
            from importlib.resources import files

            bpe_path = str(files("sam3").joinpath("assets/bpe_simple_vocab_16e6.txt.gz"))

            if self._model is None:
                self._model = build_sam3_image_model(bpe_path=bpe_path, geo_config=self._geo_config)

            self._processor = Sam3Processor(
                self._model,
                confidence_threshold=self.confidence_threshold,
            )
            self._initialized = True

        except ImportError as e:
            raise ImportError(
                "SAM3 is not installed. Please install from the fork:\n"
                "pip install git+https://github.com/keejkrej/sam3.git@feat/cross-image"
            ) from e

    def _prompt_to_payload(self, prompt: Prompt) -> dict:
        """Convert a Prompt to SAM3 processor payload format."""
        if prompt.prompt_type == PromptType.TEXT:
            return {"type": "text", "text": prompt.value}
        elif prompt.prompt_type == PromptType.POINT:
            return {"type": "point", "point": prompt.value, "label": prompt.label}
        elif prompt.prompt_type == PromptType.BOX:
            return {"type": "box", "box": prompt.value, "label": prompt.label}
        elif prompt.prompt_type == PromptType.MASK:
            mask_tensor = torch.from_numpy(prompt.value.astype(np.float32))
            return {"type": "mask", "mask": mask_tensor, "label": prompt.label}
        else:
            raise ValueError(f"Unknown prompt type: {prompt.prompt_type}")

    def _add_prompts_to_state(self, prompts: list[Prompt], state: dict) -> dict:
        """Add multiple prompts to the processor state.

        This method handles combining multiple prompts by adding them
        sequentially to the geometric prompt before encoding.
        """
        # Separate text prompts from geometric prompts
        text_prompts = [p for p in prompts if p.prompt_type == PromptType.TEXT]
        geo_prompts = [p for p in prompts if p.prompt_type != PromptType.TEXT]

        # Handle text prompt (only one supported)
        if text_prompts:
            if len(text_prompts) > 1:
                print(f"Warning: Multiple text prompts provided, using first one")
            text_prompt = text_prompts[0]
            text_outputs = self._processor.model.backbone.forward_text(
                [text_prompt.value], device=self._device
            )
            state["backbone_out"].update(text_outputs)
        else:
            # Set up dummy text for visual-only mode
            if "language_features" not in state["backbone_out"]:
                dummy_text_outputs = self._processor.model.backbone.forward_text(
                    ["visual"], device=self._device
                )
                state["backbone_out"].update(dummy_text_outputs)

        # Initialize geometric prompt
        state["geometric_prompt"] = self._processor.model._get_dummy_prompt()

        img_w = state["original_width"]
        img_h = state["original_height"]

        # Add each geometric prompt
        for prompt in geo_prompts:
            if prompt.prompt_type == PromptType.POINT:
                normalized_point = [prompt.value[0] / img_w, prompt.value[1] / img_h]
                points = torch.tensor(
                    normalized_point, device=self._device, dtype=torch.float32
                ).view(1, 1, 2)
                labels = torch.tensor(
                    [prompt.label], device=self._device, dtype=torch.bool
                ).view(1, 1)
                state["geometric_prompt"].append_points(points, labels)

            elif prompt.prompt_type == PromptType.BOX:
                from sam3.utils import box_ops
                box_tensor = torch.tensor(
                    prompt.value, device=self._device, dtype=torch.float32
                ).view(1, 4)
                box_cxcywh = box_ops.box_xywh_to_cxcywh(box_tensor)
                normalized_box = box_cxcywh / torch.tensor(
                    [img_w, img_h, img_w, img_h], device=self._device, dtype=torch.float32
                )
                boxes = normalized_box.view(1, 1, 4)
                labels = torch.tensor(
                    [prompt.label], device=self._device, dtype=torch.bool
                ).view(1, 1)
                state["geometric_prompt"].append_boxes(boxes, labels)

            elif prompt.prompt_type == PromptType.MASK:
                mask = prompt.value
                if not isinstance(mask, torch.Tensor):
                    mask = torch.from_numpy(mask)

                # Resize mask to match original image dimensions if needed
                mask_h, mask_w = mask.shape[-2:]
                if mask_h != img_h or mask_w != img_w:
                    mask_pil = Image.fromarray((mask.numpy() * 255).astype(np.uint8))
                    mask_pil = mask_pil.resize((img_w, img_h), Image.Resampling.NEAREST)
                    mask = torch.from_numpy(np.array(mask_pil) / 255.0).float()

                # Resize to processor resolution
                mask_pil = Image.fromarray((mask.numpy() * 255).astype(np.uint8))
                mask_resized = mask_pil.resize(
                    (self._processor.resolution, self._processor.resolution),
                    Image.Resampling.NEAREST
                )
                mask_tensor = torch.from_numpy(np.array(mask_resized) / 255.0).float()

                masks = mask_tensor.to(
                    device=self._device, dtype=torch.float32
                ).view(1, 1, 1, *mask_tensor.shape[-2:])
                labels = torch.tensor(
                    [prompt.label], device=self._device, dtype=torch.long
                ).view(1, 1)
                state["geometric_prompt"].append_masks(masks, labels)

        # Encode all prompts together
        with torch.profiler.record_function("SAM3Image._encode_prompt"):
            encoded = self._processor.model._encode_prompt(
                backbone_out=state["backbone_out"],
                find_input=self._processor.find_stage,
                geometric_prompt=state["geometric_prompt"],
            )

        # Store encoded results
        for key, value in encoded.items():
            state[key] = value

        # Clean up
        if "geometric_prompt" in state:
            del state["geometric_prompt"]

        return state

    def _parse_result(
        self,
        state: dict,
        image_size: tuple[int, int],
        prompts: PromptMap,
    ) -> SegmentationResult:
        """Parse SAM3 processor state into SegmentationResult."""
        masks = state.get("masks", [])
        scores = state.get("scores", [])
        boxes = state.get("boxes", [])

        # Convert tensors to numpy/lists
        if isinstance(masks, torch.Tensor):
            masks = [m.cpu().numpy() for m in masks]
        if isinstance(scores, torch.Tensor):
            scores = scores.cpu().tolist()
        if isinstance(boxes, torch.Tensor):
            boxes = [tuple(b.cpu().tolist()) for b in boxes]

        # Ensure lists
        if not isinstance(masks, list):
            masks = [masks] if masks is not None else []
        if not isinstance(scores, list):
            scores = [scores] if scores is not None else []
        if not isinstance(boxes, list):
            boxes = [boxes] if boxes is not None else []

        return SegmentationResult(
            masks=masks,
            scores=scores,
            boxes=boxes,
            prompts=prompts,
            image_size=image_size,
        )

    def segment(
        self,
        image: Image.Image,
        prompts: PromptMap,
        threshold: Optional[float] = None,
    ) -> SegmentationResult:
        """Segment objects in an image using prompts.

        Args:
            image: PIL Image to segment
            prompts: PromptMap (dict[PromptType, Prompt])
            threshold: Confidence threshold (uses default if None)

        Returns:
            SegmentationResult with detected masks and scores
        """
        self._ensure_initialized()

        threshold = threshold or self.confidence_threshold
        image_size = image.size  # (width, height)

        with torch.autocast(self._device, dtype=torch.bfloat16):
            # Set up image
            state = self._processor.set_image(image)

            # Add all prompts (convert to list for internal processing)
            state = self._add_prompts_to_state(list(prompts.values()), state)

            # Run inference
            state = self._processor._forward_grounding(state)

        return self._parse_result(state, image_size, prompts=prompts)

    def extract_concept(
        self,
        image: Image.Image,
        prompts: PromptMap,
    ) -> ConceptEmbedding:
        """Extract a concept embedding from a prompted image.

        This captures the "concept" of the prompted object, which can
        then be transferred to find similar objects in other images.

        Args:
            image: PIL Image containing the reference object
            prompts: PromptMap (dict[PromptType, Prompt])

        Returns:
            ConceptEmbedding that can be transferred to other images
        """
        self._ensure_initialized()

        image_size = image.size

        with torch.autocast(self._device, dtype=torch.bfloat16):
            # Set up image and add prompts
            state = self._processor.set_image(image)
            state = self._add_prompts_to_state(list(prompts.values()), state)

            # Extract the prompt embeddings (the "concept")
            prompt_embedding = state["prompt"].clone()
            prompt_mask = state["prompt_mask"].clone()

        return ConceptEmbedding(
            prompt_embedding=prompt_embedding,
            prompt_mask=prompt_mask,
            source_prompts=prompts,
            source_image_size=image_size,
        )

    def transfer_concept(
        self,
        image: Image.Image,
        concept: ConceptEmbedding,
        threshold: Optional[float] = None,
    ) -> SegmentationResult:
        """Transfer a concept to find similar objects in a new image.

        Args:
            image: PIL Image to search for similar objects
            concept: ConceptEmbedding extracted from a reference image
            threshold: Confidence threshold (uses default if None)

        Returns:
            SegmentationResult with detected similar objects
        """
        self._ensure_initialized()

        threshold = threshold or self.confidence_threshold
        image_size = image.size

        with torch.autocast(self._device, dtype=torch.bfloat16):
            # Set up new image
            state = self._processor.set_image(image)

            # Inject the concept embeddings
            state["prompt"] = concept.prompt_embedding
            state["prompt_mask"] = concept.prompt_mask

            # Run inference with transferred concept
            state = self._processor._forward_grounding(state)

        result = self._parse_result(state, image_size, prompts=concept.source_prompts)
        return result.filter_by_score(threshold)

    def segment_and_transfer(
        self,
        reference_image: Image.Image,
        target_image: Image.Image,
        prompts: PromptMap,
        threshold: Optional[float] = None,
    ) -> tuple[SegmentationResult, SegmentationResult]:
        """Segment in reference image and transfer to target.

        Convenience method that combines segment, extract_concept, and transfer_concept.

        Args:
            reference_image: Image containing the reference object
            target_image: Image to find similar objects in
            prompts: PromptMap (dict[PromptType, Prompt])
            threshold: Confidence threshold

        Returns:
            Tuple of (reference_result, target_result)
        """
        self._ensure_initialized()

        threshold = threshold or self.confidence_threshold

        with torch.autocast(self._device, dtype=torch.bfloat16):
            # Process reference image
            ref_state = self._processor.set_image(reference_image)
            ref_state = self._add_prompts_to_state(list(prompts.values()), ref_state)

            # Run inference on reference
            ref_state_result = self._processor._forward_grounding(ref_state.copy())
            ref_result = self._parse_result(ref_state_result, reference_image.size, prompts=prompts)

            # Extract concept
            prompt_embedding = ref_state["prompt"].clone()
            prompt_mask = ref_state["prompt_mask"].clone()

            # Process target image with transferred concept
            tgt_state = self._processor.set_image(target_image)
            tgt_state["prompt"] = prompt_embedding
            tgt_state["prompt_mask"] = prompt_mask

            # Run inference on target
            tgt_state = self._processor._forward_grounding(tgt_state)
            tgt_result = self._parse_result(tgt_state, target_image.size, prompts=prompts)

        return ref_result, tgt_result.filter_by_score(threshold)

    def set_confidence_threshold(self, threshold: float) -> None:
        """Update the default confidence threshold."""
        self.confidence_threshold = threshold
        if self._processor is not None:
            self._processor.set_confidence_threshold(threshold)
