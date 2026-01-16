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
        prompt: The prompt used for segmentation
        image_size: Original image size (width, height)
    """
    masks: list[np.ndarray]
    scores: list[float]
    boxes: list[tuple[float, float, float, float]]
    prompt: Optional[Prompt] = None
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
            prompt=self.prompt,
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
        source_prompt: The original prompt used
        source_image_size: Size of the source image
    """
    prompt_embedding: torch.Tensor
    prompt_mask: torch.Tensor
    source_prompt: Optional[Prompt] = None
    source_image_size: tuple[int, int] = (0, 0)


class CrossImagePredictor:
    """Wrapper for SAM3 cross-image segmentation.

    This class provides a clean interface for:
    1. Segmenting objects in an image using various prompt types
    2. Extracting concept embeddings from prompted segmentations
    3. Transferring concepts to find similar objects in other images

    Usage:
        predictor = CrossImagePredictor()

        # Segment in reference image
        result = predictor.segment(reference_image, prompt)

        # Extract concept and transfer to target
        concept = predictor.extract_concept(reference_image, prompt)
        transfer_result = predictor.transfer_concept(target_image, concept)
    """

    def __init__(
        self,
        model: Optional[Any] = None,
        device: Optional[str] = None,
        confidence_threshold: float = 0.5,
    ):
        """Initialize the predictor.

        Args:
            model: Pre-loaded SAM3 model (if None, will be loaded on first use)
            device: Device to use ('cuda' or 'cpu', auto-detected if None)
            confidence_threshold: Default confidence threshold for detections
        """
        self._model = model
        self._processor = None
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.confidence_threshold = confidence_threshold
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
                self._model = build_sam3_image_model(bpe_path=bpe_path)

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

    def _parse_result(
        self,
        state: dict,
        image_size: tuple[int, int],
        prompt: Optional[Prompt] = None,
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
            prompt=prompt,
            image_size=image_size,
        )

    def segment(
        self,
        image: Image.Image,
        prompt: Prompt,
        threshold: Optional[float] = None,
    ) -> SegmentationResult:
        """Segment objects in an image using a prompt.

        Args:
            image: PIL Image to segment
            prompt: Segmentation prompt (text, point, box, or mask)
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

            # Add prompt
            payload = self._prompt_to_payload(prompt)
            state = self._processor.add_prompt(payload, state)

            # Run inference
            state = self._processor._forward_grounding(state)

        return self._parse_result(state, image_size, prompt)

    def extract_concept(
        self,
        image: Image.Image,
        prompt: Prompt,
    ) -> ConceptEmbedding:
        """Extract a concept embedding from a prompted image.

        This captures the "concept" of the prompted object, which can
        then be transferred to find similar objects in other images.

        Args:
            image: PIL Image containing the reference object
            prompt: Prompt identifying the object

        Returns:
            ConceptEmbedding that can be transferred to other images
        """
        self._ensure_initialized()

        image_size = image.size

        with torch.autocast(self._device, dtype=torch.bfloat16):
            # Set up image and add prompt
            state = self._processor.set_image(image)
            payload = self._prompt_to_payload(prompt)
            state = self._processor.add_prompt(payload, state)

            # Extract the prompt embeddings (the "concept")
            prompt_embedding = state["prompt"].clone()
            prompt_mask = state["prompt_mask"].clone()

        return ConceptEmbedding(
            prompt_embedding=prompt_embedding,
            prompt_mask=prompt_mask,
            source_prompt=prompt,
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

        result = self._parse_result(state, image_size, concept.source_prompt)
        return result.filter_by_score(threshold)

    def segment_and_transfer(
        self,
        reference_image: Image.Image,
        target_image: Image.Image,
        prompt: Prompt,
        threshold: Optional[float] = None,
    ) -> tuple[SegmentationResult, SegmentationResult]:
        """Segment in reference image and transfer to target.

        Convenience method that combines segment, extract_concept, and transfer_concept.

        Args:
            reference_image: Image containing the reference object
            target_image: Image to find similar objects in
            prompt: Prompt identifying the reference object
            threshold: Confidence threshold

        Returns:
            Tuple of (reference_result, target_result)
        """
        self._ensure_initialized()

        threshold = threshold or self.confidence_threshold

        with torch.autocast(self._device, dtype=torch.bfloat16):
            # Process reference image
            ref_state = self._processor.set_image(reference_image)
            payload = self._prompt_to_payload(prompt)
            ref_state = self._processor.add_prompt(payload, ref_state)

            # Run inference on reference
            ref_state_result = self._processor._forward_grounding(ref_state.copy())
            ref_result = self._parse_result(
                ref_state_result, reference_image.size, prompt
            )

            # Extract concept
            prompt_embedding = ref_state["prompt"].clone()
            prompt_mask = ref_state["prompt_mask"].clone()

            # Process target image with transferred concept
            # Need a fresh processor to avoid state pollution
            tgt_state = self._processor.set_image(target_image)
            tgt_state["prompt"] = prompt_embedding
            tgt_state["prompt_mask"] = prompt_mask

            # Run inference on target
            tgt_state = self._processor._forward_grounding(tgt_state)
            tgt_result = self._parse_result(tgt_state, target_image.size, prompt)

        return ref_result, tgt_result.filter_by_score(threshold)

    def set_confidence_threshold(self, threshold: float) -> None:
        """Update the default confidence threshold."""
        self.confidence_threshold = threshold
        if self._processor is not None:
            self._processor.set_confidence_threshold(threshold)
