"""COCO format dataset loader for object detection benchmarking.

Loads classic object detection datasets from:
- Local COCO structure (images/ + annotations/)
- Hugging Face Hub (datasets library, e.g. detection-datasets/coco)

Uses HF cache when available (~/.cache/huggingface/hub/datasets--detection-datasets--coco).

COCO annotation format:
- bbox: [x, y, width, height] in pixels
- category_id: integer class id

Local structure:
    coco_root/
    ├── images/
    │   └── val2017/
    └── annotations/
        └── instances_val2017.json
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional
import json
import random

import numpy as np
from PIL import Image

from cross_bench.datasets.base import DatasetSample


@dataclass
class COCODetectionSample(DatasetSample):
    """Dataset sample for detection benchmark with explicit box annotations.

    Extends DatasetSample with:
    - reference_bbox: Ground truth box [x, y, w, h] for the reference object
    - target_bboxes: List of GT boxes in target image (same category)
    - category_id: COCO category id
    """

    reference_bbox: tuple[float, float, float, float] = (0, 0, 0, 0)
    target_bboxes: list[tuple[float, float, float, float]] = field(default_factory=list)
    category_id: int = 0

    @property
    def reference_mask(self) -> np.ndarray:
        """Return bbox-derived mask (avoids loading non-existent mask file)."""
        if self._reference_mask is not None:
            return self._reference_mask
        img_info = (self.reference_image.width, self.reference_image.height)
        return _bbox_xywh_to_mask(
            self.reference_bbox[0], self.reference_bbox[1],
            self.reference_bbox[2], self.reference_bbox[3],
            img_info[0], img_info[1],
        )

    def get_reference_box_prompt(self) -> tuple[float, float, float, float]:
        """Return XYWH format for box prompt."""
        return self.reference_bbox

    def get_reference_centroid(self) -> tuple[float, float]:
        """Return centroid of reference bbox for point prompt."""
        x, y, w, h = self.reference_bbox
        return (x + w / 2, y + h / 2)


# COCO 80 class name -> id (official instance category ids)
COCO_CATEGORY_NAMES: dict[str, int] = {
    "person": 1, "bicycle": 2, "car": 3, "motorcycle": 4, "airplane": 5,
    "bus": 6, "train": 7, "truck": 8, "boat": 9, "traffic light": 10,
    "fire hydrant": 11, "stop sign": 13, "parking meter": 14, "bench": 15,
    "bird": 16, "cat": 17, "dog": 18, "horse": 19, "sheep": 20, "cow": 21,
    "elephant": 22, "bear": 23, "zebra": 24, "giraffe": 25, "backpack": 27,
    "umbrella": 28, "handbag": 31, "tie": 32, "suitcase": 33, "frisbee": 34,
    "skis": 35, "snowboard": 36, "sports ball": 37, "kite": 38, "baseball bat": 39,
    "baseball glove": 40, "skateboard": 41, "surfboard": 42, "tennis racket": 43,
    "bottle": 44, "wine glass": 46, "cup": 47, "fork": 48, "knife": 49,
    "spoon": 50, "bowl": 51, "banana": 52, "apple": 53, "sandwich": 54,
    "orange": 55, "broccoli": 56, "carrot": 57, "hot dog": 58, "pizza": 59,
    "donut": 60, "cake": 61, "chair": 62, "couch": 63, "potted plant": 64,
    "bed": 65, "dining table": 67, "toilet": 70, "tv": 72, "laptop": 73,
    "mouse": 74, "remote": 75, "keyboard": 76, "cell phone": 77, "microwave": 78,
    "oven": 79, "toaster": 80, "sink": 81, "refrigerator": 82, "book": 84,
    "clock": 85, "vase": 86, "scissors": 87, "teddy bear": 88, "hair drier": 89,
    "toothbrush": 90,
}


def _resolve_categories(categories: Optional[list[str | int]]) -> Optional[set[int]]:
    """Resolve category names to COCO ids."""
    if not categories:
        return None
    out = set()
    for c in categories:
        if isinstance(c, int):
            out.add(c)
        elif isinstance(c, str) and c.strip().isdigit():
            out.add(int(c.strip()))
        else:
            out.add(COCO_CATEGORY_NAMES.get(c.lower().strip(), -1))
    out.discard(-1)
    return out if out else None


def _bbox_xywh_to_mask(x: float, y: float, w: float, h: float, img_w: int, img_h: int) -> np.ndarray:
    """Create binary mask from bbox for compatibility with mask-based prompts."""
    mask = np.zeros((img_h, img_w), dtype=np.float32)
    x1 = max(0, int(x))
    y1 = max(0, int(y))
    x2 = min(img_w, int(x + w))
    y2 = min(img_h, int(y + h))
    mask[y1:y2, x1:x2] = 1.0
    return mask


def _load_coco_annotations(annot_path: Path) -> tuple[dict, dict, dict]:
    """Load COCO JSON and return images, annotations, categories."""
    with open(annot_path) as f:
        data = json.load(f)

    images = {img["id"]: img for img in data["images"]}
    categories = {cat["id"]: cat for cat in data.get("categories", [])}
    annos = data.get("annotations", [])

    return images, annos, categories


def _build_imgid_to_annos(annos: list, use_segmentation: bool = False) -> dict[int, list]:
    """Group annotations by image_id."""
    by_img = {}
    for ann in annos:
        img_id = ann["image_id"]
        if use_segmentation and ann.get("segmentation"):
            pass  # Keep full anno
        by_img.setdefault(img_id, []).append(ann)
    return by_img


class COCODetectionDataset:
    """Dataset for detection benchmarking on COCO-format data.

    Creates cross-image samples: for each (ref_image, ref_annotation), finds
    target images with the same category and builds transfer pairs.

    Usage:
        dataset = COCODetectionDataset.from_coco(
            root=Path("/path/to/coco"),
            split="val",
            max_samples=500,
            categories=["person", "dog"],
        )
    """

    def __init__(
        self,
        name: str = "coco_detection",
        samples: Optional[list[COCODetectionSample]] = None,
    ):
        self.name = name
        self._samples: list[COCODetectionSample] = samples or []

    def __len__(self) -> int:
        return len(self._samples)

    def __iter__(self) -> Iterator[COCODetectionSample]:
        return iter(self._samples)

    def __getitem__(self, idx: int) -> COCODetectionSample:
        return self._samples[idx]

    @classmethod
    def from_coco(
        cls,
        root: Path | str,
        split: str = "val",
        annotation_file: Optional[str] = None,
        images_dir: Optional[str] = None,
        max_samples: Optional[int] = 500,
        categories: Optional[list[str | int]] = None,
        min_targets_per_ref: int = 1,
        seed: int = 42,
    ) -> "COCODetectionDataset":
        """Load COCO-format dataset for detection benchmark.

        Args:
            root: Root directory (contains images/ and annotations/)
            split: 'train', 'val', or 'train2017'/'val2017' etc.
            annotation_file: Override annotation path (e.g. instances_val2017.json)
            images_dir: Override images subdir (e.g. val2017)
            max_samples: Max number of (ref, target) pairs to sample
            categories: Filter by category names or ids (e.g. ["person", "dog"])
            min_targets_per_ref: Require at least N target images per ref
            seed: Random seed for sampling

        Returns:
            COCODetectionDataset
        """
        root = Path(root)
        rng = random.Random(seed)

        # Resolve paths
        if annotation_file is None:
            if split in ("train", "val"):
                annotation_file = f"annotations/instances_{split}2017.json"
            else:
                annotation_file = f"annotations/instances_{split}.json"
        annot_path = root / annotation_file
        if not annot_path.exists():
            annot_path = root / "annotations" / Path(annotation_file).name

        if not annot_path.exists():
            raise FileNotFoundError(
                f"COCO annotations not found: {annot_path}\n"
                "Download from https://cocodataset.org and place in root/annotations/"
            )

        images_map, annos, categories_map = _load_coco_annotations(annot_path)
        cat_name_to_id = {c["name"]: c["id"] for c in categories_map.values()}

        # Resolve category filter
        cat_ids = None
        if categories is not None:
            cat_ids = set()
            for c in categories:
                if isinstance(c, str):
                    cat_ids.add(cat_name_to_id.get(c, -1))
                else:
                    cat_ids.add(c)
            cat_ids.discard(-1)

        # Build image_id -> list of annos (by category)
        img_cat_annos: dict[int, dict[int, list]] = {}
        for ann in annos:
            if ann.get("iscrowd"):
                continue
            img_id = ann["image_id"]
            cat_id = ann["category_id"]
            if cat_ids and cat_id not in cat_ids:
                continue
            img_cat_annos.setdefault(img_id, {}).setdefault(cat_id, []).append(ann)

        # Resolve image paths
        if images_dir is None:
            if split == "train":
                images_dir = "train2017"
            elif split == "val":
                images_dir = "val2017"
            else:
                images_dir = split
        img_base = root / "images" / images_dir

        # Build samples: (ref_img_id, ref_anno) -> target_img_ids with same cat
        samples: list[COCODetectionSample] = []
        ref_candidates: list[tuple[int, dict]] = []

        for img_id, cat_annos in img_cat_annos.items():
            img_info = images_map.get(img_id)
            if not img_info:
                continue
            img_path = img_base / img_info["file_name"]
            if not img_path.exists():
                img_path = root / img_info["file_name"]
            if not img_path.exists():
                continue

            for cat_id, ann_list in cat_annos.items():
                for ann in ann_list:
                    bbox = ann["bbox"]
                    if len(bbox) != 4 or bbox[2] <= 0 or bbox[3] <= 0:
                        continue
                    ref_candidates.append((img_id, ann, cat_id, img_path, ann_list))

        rng.shuffle(ref_candidates)

        # For each ref, find targets with same category
        img_ids_by_cat: dict[int, set[int]] = {}
        for img_id, cat_annos in img_cat_annos.items():
            for cat_id in cat_annos:
                img_ids_by_cat.setdefault(cat_id, set()).add(img_id)

        for img_id, ref_ann, cat_id, ref_img_path, ref_ann_list in ref_candidates:
            target_img_ids = img_ids_by_cat.get(cat_id, set()) - {img_id}
            if len(target_img_ids) < min_targets_per_ref:
                continue

            target_img_id = rng.choice(list(target_img_ids))
            tgt_img_info = images_map.get(target_img_id)
            if not tgt_img_info:
                continue
            tgt_img_path = img_base / tgt_img_info["file_name"]
            if not tgt_img_path.exists():
                tgt_img_path = root / tgt_img_info["file_name"]
            if not tgt_img_path.exists():
                continue

            tgt_annos = img_cat_annos.get(target_img_id, {}).get(cat_id, [])
            tgt_bboxes = [tuple(a["bbox"]) for a in tgt_annos if len(a.get("bbox", [])) == 4]

            cat_name = categories_map.get(cat_id, {}).get("name", "unknown")

            ref_bbox = tuple(ref_ann["bbox"])
            ref_w = images_map[img_id].get("width", 640)
            ref_h = images_map[img_id].get("height", 480)

            # Mask path: use a synthetic path for compatibility (we use bbox primarily)
            masks_dir = root / "masks"
            masks_dir.mkdir(parents=True, exist_ok=True)
            mask_stem = f"{img_id}_{ref_ann['id']}_to_{target_img_id}"
            mask_path = masks_dir / f"{mask_stem}.npy"

            sample = COCODetectionSample(
                sample_id=mask_stem,
                reference_image_path=ref_img_path,
                reference_mask_path=mask_path,
                target_image_path=tgt_img_path,
                target_mask_path=None,
                category=cat_name,
                reference_bbox=ref_bbox,
                target_bboxes=tgt_bboxes,
                category_id=cat_id,
                metadata={
                    "ref_image_id": img_id,
                    "ref_ann_id": ref_ann["id"],
                    "target_image_id": target_img_id,
                    "target_ann_ids": [a["id"] for a in tgt_annos],
                },
            )
            # Override _reference_mask with bbox-derived mask
            sample._reference_mask = _bbox_xywh_to_mask(
                ref_bbox[0], ref_bbox[1], ref_bbox[2], ref_bbox[3], ref_w, ref_h
            )
            samples.append(sample)

            if max_samples and len(samples) >= max_samples:
                break

        return cls(name=f"coco_{split}", samples=samples)

    @classmethod
    def from_huggingface(
        cls,
        name: str = "detection-datasets/coco",
        split: str = "val",
        max_samples: Optional[int] = 500,
        categories: Optional[list[str | int]] = None,
        min_targets_per_ref: int = 1,
        seed: int = 42,
    ) -> "COCODetectionDataset":
        """Load COCO from Hugging Face Hub via datasets library.

        Uses cached data when available (~/.cache/huggingface/hub/).
        Uses HF cache when available (~/.cache/huggingface/hub/datasets--detection-datasets--coco).

        Args:
            name: HF dataset name (default: detection-datasets/coco)
            split: 'train' or 'val'
            max_samples: Max (ref, target) pairs to sample
            categories: Filter by category ids (HF uses int ids)
            min_targets_per_ref: Min target images with same category per ref
            seed: Random seed

        Returns:
            COCODetectionDataset
        """
        from datasets import load_dataset

        rng = random.Random(seed)
        cat_ids = _resolve_categories(categories)
        hf_ds = load_dataset(name, split=split)

        # Build idx -> row data; idx_by_cat: category -> set of row indices
        rows: dict[int, dict] = {}
        idx_by_cat: dict[int, set[int]] = {}

        for idx in range(len(hf_ds)):
            row = hf_ds[idx]
            objs = row.get("objects") or {}
            bboxes = objs.get("bbox") or []
            cats = objs.get("category") or []

            valid_objs = []
            for b, c in zip(bboxes, cats):
                b = tuple(float(x) for x in b)
                if len(b) != 4 or b[2] <= 0 or b[3] <= 0:
                    continue
                if cat_ids is not None and c not in cat_ids:
                    continue
                valid_objs.append((b, int(c)))
            if not valid_objs:
                continue

            img = row["image"]
            rows[idx] = {
                "image": img,
                "image_id": row.get("image_id", idx),
                "objects": valid_objs,
                "width": row.get("width", img.width if img else 640),
                "height": row.get("height", img.height if img else 480),
            }
            for _, cat_id in valid_objs:
                idx_by_cat.setdefault(cat_id, set()).add(idx)

        # Build samples: for each (idx, obj_idx) with bbox, pick target idx with same cat
        samples: list[COCODetectionSample] = []
        ref_candidates: list[tuple[int, int, tuple, int]] = []

        for idx, r in rows.items():
            for obj_idx, (bbox, cat_id) in enumerate(r["objects"]):
                target_idxs = idx_by_cat.get(cat_id, set()) - {idx}
                if len(target_idxs) < min_targets_per_ref:
                    continue
                ref_candidates.append((idx, obj_idx, bbox, cat_id))

        rng.shuffle(ref_candidates)

        for idx, obj_idx, ref_bbox, cat_id in ref_candidates:
            ref_row = rows[idx]
            target_idxs = idx_by_cat.get(cat_id, set()) - {idx}
            target_idx = rng.choice(list(target_idxs))
            tgt_row = rows[target_idx]
            tgt_bboxes = [tuple(float(x) for x in b) for b, c in tgt_row["objects"] if c == cat_id]

            sample_id = f"hf_{idx}_{obj_idx}_to_{target_idx}"
            # Use placeholder paths - images provided in-memory
            dummy = Path("/tmp/cross_bench_hf_placeholder")
            sample = COCODetectionSample(
                sample_id=sample_id,
                reference_image_path=dummy / "ref.jpg",
                reference_mask_path=dummy / "mask.npy",
                target_image_path=dummy / "tgt.jpg",
                target_mask_path=None,
                category=str(cat_id),
                reference_bbox=ref_bbox,
                target_bboxes=tgt_bboxes,
                category_id=cat_id,
                metadata={"source": "huggingface", "ref_idx": idx, "target_idx": target_idx},
            )
            sample._reference_image = ref_row["image"].convert("RGB")
            sample._target_image = tgt_row["image"].convert("RGB")
            sample._reference_mask = _bbox_xywh_to_mask(
                ref_bbox[0], ref_bbox[1], ref_bbox[2], ref_bbox[3],
                ref_row["width"], ref_row["height"],
            )
            samples.append(sample)

            if max_samples and len(samples) >= max_samples:
                break

        return cls(name=f"hf_coco_{split}", samples=samples)
