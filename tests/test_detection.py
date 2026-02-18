"""Tests for detection benchmark and COCO dataset."""

import json
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from cross_bench.datasets.coco import (
    COCODetectionDataset,
    COCODetectionSample,
    _bbox_xywh_to_mask,
)
from cross_bench.benchmarks.detection import (
    _box_iou,
    _match_detections_to_gt,
    compute_ap,
)


def _create_mini_coco(tmp_path: Path) -> Path:
    """Create minimal COCO-style dataset for testing."""
    img_dir = tmp_path / "images" / "val2017"
    annot_dir = tmp_path / "annotations"
    img_dir.mkdir(parents=True)
    annot_dir.mkdir(parents=True)

    # Create 3 small images
    images = []
    for i in range(1, 4):
        w, h = 100, 80
        img = Image.fromarray(np.random.randint(0, 255, (h, w, 3), dtype=np.uint8))
        fname = f"{i:012d}.jpg"
        img.save(img_dir / fname)
        images.append({"id": i, "file_name": fname, "width": w, "height": h})

    # Annotations: image 1 has cat, image 2 has cat, image 3 has dog
    categories = [{"id": 1, "name": "cat"}, {"id": 2, "name": "dog"}]
    annotations = [
        {"id": 1, "image_id": 1, "category_id": 1, "bbox": [10, 10, 30, 40], "iscrowd": 0},
        {"id": 2, "image_id": 2, "category_id": 1, "bbox": [20, 15, 25, 35], "iscrowd": 0},
        {"id": 3, "image_id": 3, "category_id": 2, "bbox": [15, 20, 35, 30], "iscrowd": 0},
    ]

    ann_file = annot_dir / "instances_val2017.json"
    with open(ann_file, "w") as f:
        json.dump(
            {"images": images, "annotations": annotations, "categories": categories},
            f,
        )
    return tmp_path


def test_bbox_xywh_to_mask():
    """Test bbox to mask conversion."""
    mask = _bbox_xywh_to_mask(10, 10, 20, 30, 100, 100)
    assert mask.shape == (100, 100)
    assert mask.sum() == 20 * 30
    assert mask[10, 10] == 1.0
    assert mask[39, 29] == 1.0
    assert mask[40, 30] == 0.0


def test_box_iou():
    """Test box IoU computation."""
    # Same box
    box = (10, 10, 20, 20)  # xywh
    assert abs(_box_iou(box, box, "xywh", "xywh") - 1.0) < 1e-6

    # No overlap
    a = (0, 0, 10, 10)
    b = (20, 20, 10, 10)
    assert _box_iou(a, b, "xywh", "xywh") == 0.0

    # Half overlap
    a = (0, 0, 10, 10)
    b = (5, 0, 10, 10)
    iou = _box_iou(a, b, "xywh", "xywh")
    # overlap 5x10=50, union 100+100-50=150
    assert abs(iou - 50 / 150) < 1e-6


def test_match_detections():
    """Test detection matching."""
    gt = [(0, 0, 10, 10)]  # xywh
    pred_xyxy = [(1, 1, 9, 9)]  # good overlap
    scores = [0.9]
    tp, _, ious = _match_detections_to_gt(pred_xyxy, scores, gt, iou_threshold=0.5)
    assert tp[0] is True
    assert ious[0] > 0.5


def test_compute_ap():
    """Test AP computation."""
    tp = [True, False, True]
    scores = [0.9, 0.8, 0.7]
    n_gt = 2
    ap = compute_ap(tp, scores, n_gt)
    assert 0 <= ap <= 1


def test_coco_dataset_load():
    """Test COCO dataset loading."""
    with tempfile.TemporaryDirectory() as tmp:
        root = _create_mini_coco(Path(tmp))
        ds = COCODetectionDataset.from_coco(
            root=root,
            split="val",
            max_samples=5,
            categories=None,
            seed=42,
        )
        assert len(ds) > 0
        sample = ds[0]
        assert isinstance(sample, COCODetectionSample)
        assert hasattr(sample, "reference_bbox")
        assert hasattr(sample, "target_bboxes")
        assert sample.reference_image is not None
        assert sample.target_image is not None
        mask = sample.reference_mask
        assert mask is not None
        assert mask.ndim == 2
        assert mask.sum() > 0
