"""Base dataset classes for cross-image segmentation benchmarking.

A dataset consists of triplets:
- Reference image: The source image containing the object of interest
- Reference mask: Binary mask indicating the object in the reference image
- Target image: Image containing similar objects to segment/detect
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Iterator
import json

import numpy as np
from PIL import Image


@dataclass
class DatasetSample:
    """A single sample in the cross-image benchmark dataset.

    Attributes:
        sample_id: Unique identifier for this sample
        reference_image_path: Path to the reference image
        reference_mask_path: Path to the reference mask (binary, white=object)
        target_image_path: Path to the target image
        category: Optional category/class name for the object
        metadata: Optional additional metadata
    """
    sample_id: str
    reference_image_path: Path
    reference_mask_path: Path
    target_image_path: Path
    category: Optional[str] = None
    metadata: dict = field(default_factory=dict)

    # Cached loaded data
    _reference_image: Optional[Image.Image] = field(default=None, repr=False)
    _reference_mask: Optional[np.ndarray] = field(default=None, repr=False)
    _target_image: Optional[Image.Image] = field(default=None, repr=False)

    @property
    def reference_image(self) -> Image.Image:
        """Load and return the reference image."""
        if self._reference_image is None:
            self._reference_image = Image.open(self.reference_image_path).convert("RGB")
        return self._reference_image

    @property
    def reference_mask(self) -> np.ndarray:
        """Load and return the reference mask as binary numpy array."""
        if self._reference_mask is None:
            mask_img = Image.open(self.reference_mask_path).convert("L")
            mask_np = np.array(mask_img)
            self._reference_mask = (mask_np > 128).astype(np.float32)
        return self._reference_mask

    @property
    def target_image(self) -> Image.Image:
        """Load and return the target image."""
        if self._target_image is None:
            self._target_image = Image.open(self.target_image_path).convert("RGB")
        return self._target_image

    def clear_cache(self) -> None:
        """Clear cached image data to free memory."""
        self._reference_image = None
        self._reference_mask = None
        self._target_image = None

    def get_mask_bbox(self) -> tuple[int, int, int, int]:
        """Get bounding box of the mask in XYWH format.

        Returns:
            Tuple of (x, y, width, height) for the mask bounding box.
        """
        mask = self.reference_mask
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        y_indices = np.where(rows)[0]
        x_indices = np.where(cols)[0]

        if len(y_indices) == 0 or len(x_indices) == 0:
            return (0, 0, 0, 0)

        y_min, y_max = y_indices[0], y_indices[-1]
        x_min, x_max = x_indices[0], x_indices[-1]

        return (int(x_min), int(y_min), int(x_max - x_min + 1), int(y_max - y_min + 1))

    def get_mask_centroid(self) -> tuple[float, float]:
        """Get centroid point of the mask.

        Returns:
            Tuple of (x, y) coordinates for the mask centroid.
        """
        mask = self.reference_mask
        y_indices, x_indices = np.where(mask > 0)

        if len(y_indices) == 0:
            h, w = mask.shape
            return (w / 2, h / 2)

        return (float(np.mean(x_indices)), float(np.mean(y_indices)))


class CrossImageDataset:
    """Dataset for cross-image segmentation benchmarking.

    Supports loading datasets from:
    - A directory with a manifest.json file
    - A directory with standard structure (reference/, masks/, target/)
    - Programmatic construction

    Manifest JSON format:
    {
        "name": "dataset_name",
        "samples": [
            {
                "id": "sample_001",
                "reference": "reference/img001.jpg",
                "mask": "masks/img001.png",
                "target": "target/img001.jpg",
                "category": "dog"  // optional
            },
            ...
        ]
    }
    """

    def __init__(
        self,
        name: str = "unnamed",
        samples: Optional[list[DatasetSample]] = None,
    ):
        """Initialize the dataset.

        Args:
            name: Name of the dataset
            samples: Optional list of DatasetSample objects
        """
        self.name = name
        self._samples: list[DatasetSample] = samples or []

    def __len__(self) -> int:
        return len(self._samples)

    def __iter__(self) -> Iterator[DatasetSample]:
        return iter(self._samples)

    def __getitem__(self, idx: int) -> DatasetSample:
        return self._samples[idx]

    def add_sample(self, sample: DatasetSample) -> None:
        """Add a sample to the dataset."""
        self._samples.append(sample)

    @classmethod
    def from_manifest(cls, manifest_path: Path | str) -> "CrossImageDataset":
        """Load dataset from a manifest JSON file.

        Args:
            manifest_path: Path to the manifest.json file

        Returns:
            CrossImageDataset instance
        """
        manifest_path = Path(manifest_path)
        base_dir = manifest_path.parent

        with open(manifest_path) as f:
            manifest = json.load(f)

        dataset = cls(name=manifest.get("name", manifest_path.stem))

        for sample_data in manifest.get("samples", []):
            sample = DatasetSample(
                sample_id=sample_data["id"],
                reference_image_path=base_dir / sample_data["reference"],
                reference_mask_path=base_dir / sample_data["mask"],
                target_image_path=base_dir / sample_data["target"],
                category=sample_data.get("category"),
                metadata=sample_data.get("metadata", {}),
            )
            dataset.add_sample(sample)

        return dataset

    @classmethod
    def from_directory(
        cls,
        directory: Path | str,
        images_dir: str = "images",
        mask_dir: str = "masks",
        image_extensions: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".webp"),
        sample_ratio: float = 0.1,
    ) -> "CrossImageDataset":
        """Load dataset from a directory with images and masks.

        Expected structure:
        directory/
            images/
                img001.jpg
                img002.jpg
            masks/
                img001.png
                img002.png

        Files are matched by stem (filename without extension).
        For cross-image transfer, each image is used as reference with its mask,
        and other images in the dataset serve as targets.

        Args:
            directory: Root directory of the dataset
            images_dir: Subdirectory name for images
            mask_dir: Subdirectory name for masks
            image_extensions: Tuple of valid image extensions
            sample_ratio: Fraction of pairs to sample (0.1 = 10%)

        Returns:
            CrossImageDataset instance
        """
        import random
        
        directory = Path(directory)

        img_dir = directory / images_dir
        msk_dir = directory / mask_dir

        # Build lookup dictionaries by stem
        def get_files_by_stem(dir_path: Path) -> dict[str, Path]:
            files = {}
            if dir_path.exists():
                for f in dir_path.iterdir():
                    if f.suffix.lower() in image_extensions:
                        files[f.stem] = f
            return files

        img_files = get_files_by_stem(img_dir)
        msk_files = get_files_by_stem(msk_dir)

        # Find common stems (images with masks)
        common_stems = sorted(set(img_files.keys()) & set(msk_files.keys()))

        # Generate all possible pairs
        all_pairs = [
            (ref_stem, tgt_stem)
            for i, ref_stem in enumerate(common_stems)
            for j, tgt_stem in enumerate(common_stems)
            if i != j
        ]
        
        # Sample pairs
        n_samples = max(1, int(len(all_pairs) * sample_ratio))
        sampled_pairs = random.sample(all_pairs, min(n_samples, len(all_pairs)))

        dataset = cls(name=directory.name)

        for ref_stem, tgt_stem in sampled_pairs:
            sample = DatasetSample(
                sample_id=f"{ref_stem}_to_{tgt_stem}",
                reference_image_path=img_files[ref_stem],
                reference_mask_path=msk_files[ref_stem],
                target_image_path=img_files[tgt_stem],
            )
            dataset.add_sample(sample)

        return dataset

    def filter_by_category(self, category: str) -> "CrossImageDataset":
        """Create a filtered dataset containing only samples of a specific category.

        Args:
            category: Category to filter by

        Returns:
            New CrossImageDataset with filtered samples
        """
        filtered_samples = [s for s in self._samples if s.category == category]
        return CrossImageDataset(name=f"{self.name}_{category}", samples=filtered_samples)

    @property
    def categories(self) -> list[str]:
        """Get list of unique categories in the dataset."""
        cats = set(s.category for s in self._samples if s.category is not None)
        return sorted(cats)

    def summary(self) -> str:
        """Get a summary string of the dataset."""
        lines = [
            f"Dataset: {self.name}",
            f"  Samples: {len(self)}",
        ]
        if self.categories:
            lines.append(f"  Categories: {', '.join(self.categories)}")
        return "\n".join(lines)
