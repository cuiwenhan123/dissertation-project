from __future__ import annotations

import contextlib
import io
from typing import Any

import numpy as np

from .domain import Box


def iou(a: Box, b: Box) -> float:
    x1, y1 = max(a.x, b.x), max(a.y, b.y)
    x2, y2 = min(a.x + a.w, b.x + b.w), min(a.y + a.h, b.y + b.h)
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = a.w * a.h + b.w * b.h - inter
    return inter / union if union else 0.0


def empty_class_metric() -> dict[str, Any]:
    return {
        "gt": 0,
        "matched": 0,
        "missed": 0,
        "falsePositive": 0,
        "localisation": 0,
        "classification": 0,
        "iouTotal": 0.0,
        "iouCount": 0,
        "meanIou": None,
        "ap50": None,
        "map": None,
    }


def _failure_analysis(gt: list[Box], preds: list[Box]) -> tuple[dict[str, int], dict[str, dict[str, Any]]]:
    failures = {
        "missed": 0,
        "falsePositive": 0,
        "localisation": 0,
        "classification": 0,
    }
    class_metrics: dict[str, dict[str, Any]] = {}
    for truth in gt:
        class_metrics.setdefault(truth.label, empty_class_metric())["gt"] += 1

    unmatched = set(range(len(preds)))
    truths = sorted(gt, key=lambda box: box.w * box.h, reverse=True)
    for truth in truths:
        metric = class_metrics.setdefault(truth.label, empty_class_metric())
        same_class = [(idx, iou(truth, preds[idx])) for idx in unmatched if preds[idx].label == truth.label]
        correct = max(same_class, key=lambda item: item[1], default=(None, 0.0))
        if correct[0] is not None and correct[1] >= 0.5:
            unmatched.remove(correct[0])
            metric["matched"] += 1
            metric["iouTotal"] += correct[1]
            metric["iouCount"] += 1
            continue

        any_class = [(idx, iou(truth, preds[idx])) for idx in unmatched]
        classification = max(any_class, key=lambda item: item[1], default=(None, 0.0))
        if classification[0] is not None and classification[1] >= 0.5:
            unmatched.remove(classification[0])
            failures["classification"] += 1
            metric["classification"] += 1
            continue

        if correct[0] is not None and correct[1] >= 0.1:
            unmatched.remove(correct[0])
            failures["localisation"] += 1
            metric["localisation"] += 1
            continue

        failures["missed"] += 1
        metric["missed"] += 1

    failures["falsePositive"] = len(unmatched)
    for idx in unmatched:
        class_metrics.setdefault(preds[idx].label, empty_class_metric())["falsePositive"] += 1
    return failures, class_metrics


def _mean_valid(values: np.ndarray) -> float | None:
    valid = values[values > -1]
    return float(np.mean(valid)) if valid.size else None


def _safe_stat(value: float) -> float:
    return max(0.0, float(value))


def evaluate_detection_dataset(samples: list[dict[str, Any]]) -> dict[str, Any]:
    """Evaluate detections with the official COCO AP/AR protocol."""
    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("pycocotools is required for standard COCO evaluation") from exc

    labels = sorted({
        box.label
        for sample in samples
        for box in [*sample.get("groundTruth", []), *sample.get("predictions", [])]
    })
    category_ids = {label: index + 1 for index, label in enumerate(labels)}
    images: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    detections: list[dict[str, Any]] = []
    all_failures = {"missed": 0, "falsePositive": 0, "localisation": 0, "classification": 0}
    all_class_metrics: dict[str, dict[str, Any]] = {}
    annotation_id = 1

    for index, sample in enumerate(samples):
        image_id = index + 1
        width = int(sample.get("width") or 1)
        height = int(sample.get("height") or 1)
        gt: list[Box] = sample.get("groundTruth", [])
        preds: list[Box] = sample.get("predictions", [])
        images.append({"id": image_id, "width": width, "height": height, "file_name": sample.get("name", str(image_id))})
        for truth in gt:
            area = max(0.0, truth.w) * max(0.0, truth.h)
            annotations.append({
                "id": annotation_id,
                "image_id": image_id,
                "category_id": category_ids[truth.label],
                "bbox": [truth.x, truth.y, truth.w, truth.h],
                "area": area,
                "iscrowd": 0,
            })
            annotation_id += 1
        for pred in preds:
            detections.append({
                "image_id": image_id,
                "category_id": category_ids[pred.label],
                "bbox": [pred.x, pred.y, pred.w, pred.h],
                "score": float(pred.score if pred.score is not None else 1.0),
            })
        failures, class_metrics = _failure_analysis(gt, preds)
        for key, value in failures.items():
            all_failures[key] += value
        for label, values in class_metrics.items():
            target = all_class_metrics.setdefault(label, empty_class_metric())
            for key in ("gt", "matched", "missed", "falsePositive", "localisation", "classification", "iouCount"):
                target[key] += int(values.get(key) or 0)
            target["iouTotal"] += float(values.get("iouTotal") or 0.0)

    gt_count = len(annotations)
    if not samples or not labels or gt_count == 0:
        return {
            "map": None,
            "ap50": None,
            "ap75": None,
            "sizeAP": {"small": None, "medium": None, "large": None},
            "ar100": None,
            "sizeAR": {"small": None, "medium": None, "large": None},
            "failures": all_failures,
            "classMetrics": _finalise_class_metrics(all_class_metrics),
            "gtCount": gt_count,
        }

    with contextlib.redirect_stdout(io.StringIO()):
        coco_gt = COCO()
        coco_gt.dataset = {
            "info": {"description": "Detection Robustness Workbench evaluation"},
            "licenses": [],
            "images": images,
            "annotations": annotations,
            "categories": [{"id": category_id, "name": label} for label, category_id in category_ids.items()],
        }
        coco_gt.createIndex()
        if detections:
            coco_dt = coco_gt.loadRes(detections)
        else:
            coco_dt = COCO()
            coco_dt.dataset = {
                "info": coco_gt.dataset["info"],
                "licenses": [],
                "images": images,
                "annotations": [],
                "categories": coco_gt.dataset["categories"],
            }
            coco_dt.createIndex()
        evaluator = COCOeval(coco_gt, coco_dt, "bbox")
        evaluator.params.imgIds = [image["id"] for image in images]
        evaluator.params.catIds = list(category_ids.values())
        evaluator.params.maxDets = [1, 10, 100]
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()

    precision = evaluator.eval.get("precision")
    if precision is not None:
        for category_index, label in enumerate(labels):
            metric = all_class_metrics.setdefault(label, empty_class_metric())
            metric["map"] = _mean_valid(precision[:, :, category_index, 0, -1])
            metric["ap50"] = _mean_valid(precision[0, :, category_index, 0, -1])

    return {
        "map": _safe_stat(evaluator.stats[0]),
        "ap50": _safe_stat(evaluator.stats[1]),
        "ap75": _safe_stat(evaluator.stats[2]),
        "sizeAP": {
            "small": _safe_stat(evaluator.stats[3]),
            "medium": _safe_stat(evaluator.stats[4]),
            "large": _safe_stat(evaluator.stats[5]),
        },
        "ar100": _safe_stat(evaluator.stats[8]),
        "sizeAR": {
            "small": _safe_stat(evaluator.stats[9]),
            "medium": _safe_stat(evaluator.stats[10]),
            "large": _safe_stat(evaluator.stats[11]),
        },
        "failures": all_failures,
        "classMetrics": _finalise_class_metrics(all_class_metrics),
        "gtCount": gt_count,
    }


def _finalise_class_metrics(metrics: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    finalised: dict[str, dict[str, Any]] = {}
    for label, values in sorted(metrics.items()):
        item = dict(values)
        iou_count = int(item.get("iouCount") or 0)
        item["meanIou"] = float(item.get("iouTotal") or 0.0) / iou_count if iou_count else None
        item.pop("iouTotal", None)
        item.pop("iouCount", None)
        finalised[label] = item
    return finalised


def evaluate(gt: list[Box], preds: list[Box]) -> dict[str, Any]:
    width = int(max([box.x + box.w for box in [*gt, *preds]] or [1]))
    height = int(max([box.y + box.h for box in [*gt, *preds]] or [1]))
    return evaluate_detection_dataset([{
        "name": "single-image",
        "width": width,
        "height": height,
        "groundTruth": gt,
        "predictions": preds,
    }])


def evaluate_uploaded_ground_truth(gt: list[Box], preds: list[Box]) -> dict[str, Any]:
    if not gt:
        return {
            "labelAvailable": False,
            "gtCount": 0,
            "matched": 0,
            "missed": 0,
            "falsePositive": len(preds),
            "map": None,
            "ap50": None,
            "ap75": None,
            "meanIou": None,
            "classMetrics": {},
        }
    result = evaluate(gt, preds)
    class_metrics = result["classMetrics"]
    matched = sum(int(item.get("matched") or 0) for item in class_metrics.values())
    iou_values = [
        (float(item["meanIou"]), int(item.get("matched") or 0))
        for item in class_metrics.values()
        if item.get("meanIou") is not None
    ]
    iou_weight = sum(weight for _, weight in iou_values)
    mean_iou = sum(value * weight for value, weight in iou_values) / iou_weight if iou_weight else 0.0
    return {
        "labelAvailable": True,
        "gtCount": len(gt),
        "matched": matched,
        "missed": result["failures"]["missed"],
        "falsePositive": result["failures"]["falsePositive"],
        "map": result["map"],
        "ap50": result["ap50"],
        "ap75": result["ap75"],
        "meanIou": mean_iou,
        "classMetrics": class_metrics,
    }
