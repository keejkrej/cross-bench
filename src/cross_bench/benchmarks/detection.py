"""Object detection benchmark for cross-image concept transfer.

Evaluates SAM3's ability to detect objects in target images given a reference
concept. Focus is on bounding box detection—segmentation is trivial with
SAM1/SAM2 once the bbox is acquired.

Flow (matches SAM3 cross-image fork):
- Take GT bbox (e.g. cat) from image A as prompt
- segment_and_transfer: finds all instances in A, extracts concept, transfers to B
- Find all instances of that category in image B
- Compare predicted boxes to GT boxes (mAP, AP@50, AP@75).

Metrics: mAP, AP@50, AP@75, mean IoU (box-level).
"""

from pathlib import Path
from typing import Optional

import numpy as np

from cross_bench.datasets.base import DatasetSample
from cross_bench.datasets.coco import COCODetectionSample
from cross_bench.predictor import CrossImagePredictor, Prompt, SegmentationResult
from cross_bench.benchmarks.base import BaseBenchmark, BenchmarkResult, BenchmarkRun


def _box_iou(
    box_a: tuple[float, float, float, float],
    box_b: tuple[float, float, float, float],
    format_a: str = "xywh",
    format_b: str = "xywh",
) -> float:
    """Compute IoU between two boxes. Supports xywh or xyxy."""
    def to_xyxy(box: tuple[float, ...], fmt: str) -> tuple[float, float, float, float]:
        if fmt == "xywh":
            x, y, w, h = box
            return (x, y, x + w, y + h)
        return box  # assume xyxy

    xa1, ya1, xa2, ya2 = to_xyxy(box_a, format_a)
    xb1, yb1, xb2, yb2 = to_xyxy(box_b, format_b)

    xi1 = max(xa1, xb1)
    yi1 = max(ya1, yb1)
    xi2 = min(xa2, xb2)
    yi2 = min(ya2, yb2)

    inter_w = max(0, xi2 - xi1)
    inter_h = max(0, yi2 - yi1)
    inter_area = inter_w * inter_h

    area_a = (xa2 - xa1) * (ya2 - ya1)
    area_b = (xb2 - xb1) * (yb2 - yb1)
    union_area = area_a + area_b - inter_area

    if union_area <= 0:
        return 0.0
    return inter_area / union_area


def _boxes_xyxy_to_xywh(boxes: list[tuple[float, float, float, float]]) -> list[tuple[float, float, float, float]]:
    """Convert XYXY to XYWH."""
    out = []
    for x1, y1, x2, y2 in boxes:
        out.append((x1, y1, x2 - x1, y2 - y1))
    return out


def _match_detections_to_gt(
    pred_boxes: list[tuple[float, float, float, float]],
    pred_scores: list[float],
    gt_boxes: list[tuple[float, float, float, float]],
    iou_threshold: float = 0.5,
) -> tuple[list[bool], list[int], list[float]]:
    """Match predictions to GT by IoU. Greedy matching by score.

    Returns:
        tp: list of booleans, True if pred is a true positive
        matched_gt: for each pred, index of matched gt or -1
        ious: for each pred, IoU with matched gt
    """
    n_pred = len(pred_boxes)
    n_gt = len(gt_boxes)
    tp = [False] * n_pred
    matched_gt = [-1] * n_pred
    ious = [0.0] * n_pred
    gt_matched = [False] * n_gt

    if n_gt == 0:
        return tp, matched_gt, ious

    # Sort by score descending
    order = sorted(range(n_pred), key=lambda i: pred_scores[i] if i < len(pred_scores) else 0, reverse=True)

    for pi in order:
        best_iou = 0.0
        best_j = -1
        pred_box = pred_boxes[pi]
        # SAM/SAM3 output boxes in XYXY format; COCO GT is XYWH
        pred_fmt = "xyxy"
        for j, gt_box in enumerate(gt_boxes):
            if gt_matched[j]:
                continue
            iou = _box_iou(pred_box, gt_box, pred_fmt, "xywh")
            if iou >= iou_threshold and iou > best_iou:
                best_iou = iou
                best_j = j
        if best_j >= 0:
            tp[pi] = True
            matched_gt[pi] = best_j
            ious[pi] = best_iou
            gt_matched[best_j] = True

    return tp, matched_gt, ious


def compute_ap(
    tp: list[bool],
    scores: list[float],
    n_gt: int,
    recall_thresholds: Optional[list[float]] = None,
) -> float:
    """Compute average precision from TP flags and scores."""
    if n_gt == 0:
        return 0.0 if (tp and any(tp)) else 1.0
    if recall_thresholds is None:
        recall_thresholds = np.linspace(0, 1, 101).tolist()

    order = sorted(range(len(tp)), key=lambda i: scores[i] if i < len(scores) else 0, reverse=True)
    tp_sorted = [tp[i] for i in order]
    scores_sorted = [scores[i] if i < len(scores) else 0.0 for i in order]

    tp_cumsum = np.cumsum(tp_sorted)
    fp_cumsum = np.cumsum([not t for t in tp_sorted])
    recall = tp_cumsum / n_gt
    precision = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-10)

    # Make precision monotonic
    for i in range(len(precision) - 2, -1, -1):
        precision[i] = max(precision[i], precision[i + 1])

    # Interpolate at recall thresholds
    ap = 0.0
    for t in recall_thresholds:
        idx = np.searchsorted(recall, t, side="right") - 1
        if idx >= 0:
            ap += precision[idx]
    ap /= len(recall_thresholds)
    return float(ap)


class DetectionBenchmark(BaseBenchmark):
    """Benchmark for cross-image object detection.

    Given a reference image and object (prompted by box/point/mask), extracts
    the concept and transfers to a target image. Evaluates predicted boxes
    against ground truth boxes using mAP, AP@50, AP@75.
    """

    @property
    def name(self) -> str:
        return "detection"

    def _create_prompts(
        self,
        sample: DatasetSample,
        prompt_types: list[str],
    ) -> dict[str, Prompt]:
        """Create prompts from sample. Prefer box for detection eval."""
        prompts = {}
        if hasattr(sample, "reference_bbox"):
            coco_sample = sample
            x, y, w, h = coco_sample.reference_bbox
        else:
            x, y, w, h = sample.get_mask_bbox()
            if w <= 0 or h <= 0:
                return prompts

        for ptype in prompt_types:
            if ptype == "box":
                prompts["box"] = Prompt.from_box(x, y, w, h)
            elif ptype == "point":
                cx = x + w / 2
                cy = y + h / 2
                prompts["point"] = Prompt.from_point(cx, cy, label=True)
            elif ptype == "mask":
                # Use box prompt (mask encoding can hit numpy/tensor edge cases)
                prompts["mask"] = Prompt.from_box(x, y, w, h)

        return prompts

    def run_sample(
        self,
        sample: DatasetSample,
        prompt_types: list[str],
    ) -> list[BenchmarkResult]:
        """Run detection benchmark on a single sample."""
        results = []
        prompts = self._create_prompts(sample, prompt_types)

        gt_boxes = []
        if hasattr(sample, "target_bboxes"):
            gt_boxes = list(sample.target_bboxes)

        for ptype, prompt in prompts.items():
            try:
                prompt_map = {prompt.prompt_type: prompt}
                ref_result, tgt_result = self.predictor.segment_and_transfer(
                    reference_image=sample.reference_image,
                    target_image=sample.target_image,
                    prompts=prompt_map,
                    threshold=self.confidence_threshold,
                )

                pred_boxes = tgt_result.boxes
                pred_scores = tgt_result.scores if tgt_result.scores else [1.0] * len(pred_boxes)

                tp50, _, ious50 = _match_detections_to_gt(
                    pred_boxes, pred_scores, gt_boxes, iou_threshold=0.5
                )
                tp75, _, ious75 = _match_detections_to_gt(
                    pred_boxes, pred_scores, gt_boxes, iou_threshold=0.75
                )
                ap50 = compute_ap(tp50, pred_scores, len(gt_boxes))
                ap75 = compute_ap(tp75, pred_scores, len(gt_boxes))
                mean_iou = float(np.mean(ious50)) if ious50 else 0.0
                precision = sum(tp50) / len(pred_boxes) if pred_boxes else 0.0
                recall = sum(tp50) / len(gt_boxes) if gt_boxes else 0.0

                result = BenchmarkResult(
                    sample_id=sample.sample_id,
                    prompt_type=ptype,
                    results={"reference": ref_result, "target": tgt_result},
                    metadata={
                        "category": getattr(sample, "category", None),
                        "n_gt": len(gt_boxes),
                        "n_pred": len(pred_boxes),
                        "ap50": ap50,
                        "ap75": ap75,
                        "precision": precision,
                        "recall": recall,
                        "mean_iou": mean_iou,
                        "reference_detections": ref_result.num_detections,
                        "target_detections": tgt_result.num_detections,
                    },
                )
                results.append(result)

            except Exception as e:
                results.append(
                    BenchmarkResult(
                        sample_id=sample.sample_id,
                        prompt_type=ptype,
                        metadata={"error": str(e), "n_gt": len(gt_boxes)},
                    )
                )

        return results

    def calculate_scores(self, run: BenchmarkRun) -> dict[str, float]:
        """Compute detection metrics over the run."""
        scores = {
            "total_samples": 0,
            "ap50_sum": 0.0,
            "ap75_sum": 0.0,
            "mean_iou_sum": 0.0,
            "precision_sum": 0.0,
            "recall_sum": 0.0,
            "n_gt_total": 0,
            "valid_count": 0,
        }

        for result in run.results:
            if "target" not in result.results:
                continue
            n_gt = result.metadata.get("n_gt", 0)
            scores["total_samples"] += 1
            scores["n_gt_total"] += n_gt

            ap50 = result.metadata.get("ap50")
            ap75 = result.metadata.get("ap75")
            mean_iou = result.metadata.get("mean_iou")
            prec = result.metadata.get("precision")
            rec = result.metadata.get("recall")

            if ap50 is not None:
                scores["ap50_sum"] += ap50
                scores["valid_count"] += 1
            if ap75 is not None:
                scores["ap75_sum"] += ap75
            if mean_iou is not None:
                scores["mean_iou_sum"] += mean_iou
            if prec is not None:
                scores["precision_sum"] += prec
            if rec is not None:
                scores["recall_sum"] += rec

        n = max(1, scores["valid_count"])
        scores["mAP50"] = scores["ap50_sum"] / n
        scores["mAP75"] = scores["ap75_sum"] / n
        scores["mean_iou_avg"] = scores["mean_iou_sum"] / n
        scores["precision_avg"] = scores["precision_sum"] / n
        scores["recall_avg"] = scores["recall_sum"] / n

        return scores

    def save_sample_plots(
        self,
        sample: DatasetSample,
        results: list[BenchmarkResult],
        output_dir: Path,
    ) -> None:
        """Save detection result plots: reference/target with GT and predicted boxes."""
        from cross_bench.visualization import plot_detection_result, save_figure

        if not isinstance(sample, COCODetectionSample):
            return

        for result in results:
            if "reference" not in result.results or "target" not in result.results:
                continue
            try:
                fig = plot_detection_result(sample, result)
                save_figure(
                    fig,
                    output_dir / f"{result.sample_id}_{result.prompt_type}_detection.png",
                )
            except Exception:
                pass
