from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path
from typing import Any

from PIL import Image

from .annotations import (
    coco_label_from_yolo_id,
    parse_coco_annotation_data,
    parse_coco_boxes,
    parse_yolo_dataset_yaml,
    parse_yolo_label_text,
)
from .domain import Box
from .images import bytes_from_data_url, degradation_parameters, degrade, draw_boxes, image_data_url, stable_seed
from .metrics import evaluate_detection_dataset, evaluate_uploaded_ground_truth
from .models import run_uploaded_detector
from .runtime import runtime_status
from .storage import dataset_summary_from_rows, persist_run

SUPPORTED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def upload_summary(preds: list[Box], degradation: str, severity: int, model: str) -> dict[str, Any]:
    scores = [box.score for box in preds if box.score is not None]
    size_counts = {"small": 0, "medium": 0, "large": 0}
    for box in preds:
        size_counts[box.size] += 1
    return {
        "predictionCount": len(preds),
        "meanConfidence": sum(scores) / len(scores) if scores else 0,
        "sizeCounts": size_counts,
        "note": f"{model} produced {len(preds)} predictions after {degradation} severity {severity}.",
    }


def evaluate_uploaded_image(
    name: str,
    clean: Image.Image,
    model: str,
    degradation: str,
    severity: int,
    ground_truth: list[Box] | None = None,
) -> dict[str, Any]:
    parameters = degradation_parameters(degradation, severity)
    degraded = degrade(clean, degradation, severity, seed=stable_seed(name))
    preds, backend = run_uploaded_detector(model, degraded)
    boxed = draw_boxes(degraded, ground_truth or [], "#176b63") if ground_truth else degraded
    result = draw_boxes(boxed, preds, "#b84d36")
    summary = upload_summary(preds, degradation, severity, model)
    gt_metrics = evaluate_uploaded_ground_truth(ground_truth or [], preds)
    return {
        "imageName": name,
        "model": model,
        "degradation": degradation,
        "severity": severity,
        "degradationParameters": parameters,
        "backend": backend,
        "cleanImage": image_data_url(draw_boxes(clean, ground_truth or [], "#176b63") if ground_truth else clean),
        "resultImage": image_data_url(result),
        "groundTruth": [box.to_dict() for box in ground_truth or []],
        "predictions": [box.to_dict() for box in preds],
        "summary": {**summary, "groundTruthMetrics": gt_metrics},
        "row": {
            "imageName": name,
            "model": model,
            "degradation": degradation,
            "severity": severity,
            "degradationParameters": parameters,
            "backend": backend,
            "predictionCount": summary["predictionCount"],
            "meanConfidence": summary["meanConfidence"],
            "small": summary["sizeCounts"]["small"],
            "medium": summary["sizeCounts"]["medium"],
            "large": summary["sizeCounts"]["large"],
            "labelAvailable": gt_metrics["labelAvailable"],
            "gtCount": gt_metrics["gtCount"],
            "matched": gt_metrics["matched"],
            "missed": gt_metrics["missed"],
            "falsePositive": gt_metrics["falsePositive"],
            "map": gt_metrics.get("map"),
            "ap50": gt_metrics["ap50"],
            "ap75": gt_metrics.get("ap75"),
            "meanIou": gt_metrics["meanIou"],
            "classMetrics": gt_metrics.get("classMetrics", {}),
            "cleanImage": image_data_url(draw_boxes(clean, ground_truth or [], "#176b63") if ground_truth else clean),
            "resultImage": image_data_url(result),
            "error": "",
        },
    }


def inspect_zip_upload(zip_data_url: str) -> dict[str, Any]:
    raw = bytes_from_data_url(zip_data_url)
    images: list[str] = []
    labels: dict[str, str] = {}
    coco_by_stem: dict[str, dict[str, Any]] = {}
    yolo_class_counts: dict[str, int] = {}
    coco_class_counts: dict[str, int] = {}
    yolo_class_names: list[str] | None = None
    unreadable_images = 0
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        for item in archive.infolist():
            if item.is_dir() or item.filename.startswith("__MACOSX/"):
                continue
            suffix = Path(item.filename).suffix.lower()
            if suffix in SUPPORTED_IMAGE_SUFFIXES:
                images.append(item.filename)
                try:
                    with archive.open(item.filename) as handle:
                        Image.open(handle).verify()
                except Exception:
                    unreadable_images += 1
            elif suffix == ".txt":
                try:
                    labels[Path(item.filename).stem] = archive.read(item).decode("utf-8", errors="ignore")
                except Exception:
                    continue
            elif suffix == ".json":
                try:
                    data = json.loads(archive.read(item).decode("utf-8", errors="ignore"))
                    if isinstance(data, dict) and "annotations" in data and "images" in data:
                        parsed_coco, parsed_counts = parse_coco_annotation_data(data)
                        coco_by_stem.update(parsed_coco)
                        for label, count in parsed_counts.items():
                            coco_class_counts[label] = coco_class_counts.get(label, 0) + count
                except Exception:
                    continue
            elif suffix in {".yaml", ".yml"}:
                try:
                    parsed_names = parse_yolo_dataset_yaml(archive.read(item).decode("utf-8", errors="ignore"))
                    if parsed_names:
                        yolo_class_names = parsed_names
                except Exception:
                    continue

    image_stems = {Path(name).stem for name in images}
    yolo_labelled_stems = {stem for stem in image_stems if stem in labels}
    coco_labelled_stems = {stem for stem in image_stems if stem in coco_by_stem and coco_by_stem[stem].get("annotations")}
    for stem in yolo_labelled_stems:
        for line in labels.get(stem, "").splitlines():
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            try:
                class_id = int(float(parts[0]))
            except ValueError:
                continue
            label = coco_label_from_yolo_id(class_id, yolo_class_names)
            yolo_class_counts[label] = yolo_class_counts.get(label, 0) + 1

    annotation_stems = coco_labelled_stems or yolo_labelled_stems
    class_counts = coco_class_counts if coco_labelled_stems else yolo_class_counts
    annotation_format = "COCO JSON" if coco_labelled_stems else "YOLO TXT" if yolo_labelled_stems else "none"
    image_count = len(images)
    return {
        "imageCount": image_count,
        "annotationFormat": annotation_format,
        "classMappingSource": "COCO categories" if coco_labelled_stems else "dataset YAML" if yolo_class_names else "standard COCO80",
        "labelFileCount": len(labels),
        "cocoImageCount": len(coco_by_stem),
        "matchedLabelCount": len(annotation_stems),
        "labelCoverage": len(annotation_stems) / image_count if image_count else 0,
        "unreadableImageCount": unreadable_images,
        "truncatedOnEvaluation": image_count > 100,
        "classCounts": dict(sorted(class_counts.items(), key=lambda item: (-item[1], item[0]))),
        "sampleImages": images[:8],
        "sampleLabels": sorted(annotation_stems)[:8],
        "runtime": runtime_status(),
    }


def evaluate_zip_upload(zip_data_url: str, model: str, degradation: str, severity: int) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    last_result: dict[str, Any] | None = None
    raw = bytes_from_data_url(zip_data_url)
    annotation_format = "none"
    yolo_class_names: list[str] | None = None
    evaluation_samples: list[dict[str, Any]] = []
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        label_text_by_stem: dict[str, str] = {}
        coco_by_stem: dict[str, dict[str, Any]] = {}
        for item in archive.infolist():
            if item.is_dir() or item.filename.startswith("__MACOSX/"):
                continue
            suffix = Path(item.filename).suffix.lower()
            if suffix == ".txt":
                try:
                    label_text_by_stem[Path(item.filename).stem] = archive.read(item).decode("utf-8", errors="ignore")
                except Exception:
                    continue
            elif suffix == ".json":
                try:
                    data = json.loads(archive.read(item).decode("utf-8", errors="ignore"))
                    if isinstance(data, dict) and "annotations" in data and "images" in data:
                        parsed_coco, _counts = parse_coco_annotation_data(data)
                        coco_by_stem.update(parsed_coco)
                except Exception:
                    continue
            elif suffix in {".yaml", ".yml"}:
                try:
                    parsed_names = parse_yolo_dataset_yaml(archive.read(item).decode("utf-8", errors="ignore"))
                    if parsed_names:
                        yolo_class_names = parsed_names
                except Exception:
                    continue
        names = [
            item.filename
            for item in archive.infolist()
            if not item.is_dir()
            and not item.filename.startswith("__MACOSX/")
            and Path(item.filename).suffix.lower() in SUPPORTED_IMAGE_SUFFIXES
        ]
        if not names:
            raise ValueError("Zip file does not contain supported image files.")
        if coco_by_stem:
            annotation_format = "COCO JSON"
        elif label_text_by_stem:
            annotation_format = "YOLO TXT"
        for name in names[:100]:
            try:
                with archive.open(name) as handle:
                    clean = Image.open(handle).convert("RGB")
                    clean.thumbnail((960, 720))
                    stem = Path(name).stem
                    if stem in coco_by_stem and coco_by_stem[stem].get("annotations"):
                        ground_truth = parse_coco_boxes(coco_by_stem[stem], clean.width, clean.height)
                    else:
                        label_text = label_text_by_stem.get(stem)
                        ground_truth = parse_yolo_label_text(
                            label_text,
                            clean.width,
                            clean.height,
                            yolo_class_names,
                        ) if label_text else None
                    last_result = evaluate_uploaded_image(name, clean, model, degradation, severity, ground_truth)
                    last_result["row"]["annotationFormat"] = annotation_format if ground_truth else "none"
                    rows.append(last_result["row"])
                    if ground_truth:
                        predictions = [
                            Box(
                                float(item["x"]), float(item["y"]), float(item["w"]), float(item["h"]),
                                str(item["label"]), str(item["size"]), float(item.get("score", 1.0)),
                            )
                            for item in last_result["predictions"]
                        ]
                        evaluation_samples.append({
                            "name": name,
                            "width": clean.width,
                            "height": clean.height,
                            "groundTruth": ground_truth,
                            "predictions": predictions,
                        })
            except Exception as exc:
                rows.append({
                    "imageName": name,
                    "model": model,
                    "degradation": degradation,
                    "severity": severity,
                    "backend": "failed",
                    "predictionCount": 0,
                    "meanConfidence": 0,
                    "small": 0,
                    "medium": 0,
                    "large": 0,
                    "labelAvailable": False,
                    "gtCount": 0,
                    "matched": 0,
                    "missed": 0,
                    "falsePositive": 0,
                    "ap50": None,
                    "meanIou": None,
                    "classMetrics": {},
                    "annotationFormat": "none",
                    "error": str(exc),
                })
    aggregate_metrics = evaluate_detection_dataset(evaluation_samples) if evaluation_samples else None
    summary = dataset_summary_from_rows(rows)
    if aggregate_metrics:
        summary.update({
            "map": aggregate_metrics["map"],
            "ap50": aggregate_metrics["ap50"],
            "ap75": aggregate_metrics["ap75"],
            "sizeAP": aggregate_metrics["sizeAP"],
            "ar100": aggregate_metrics["ar100"],
            "sizeAR": aggregate_metrics["sizeAR"],
            "failures": aggregate_metrics["failures"],
            "classMetrics": aggregate_metrics["classMetrics"],
        })
    payload = {
        "model": model,
        "degradation": degradation,
        "severity": severity,
        "annotationFormat": annotation_format,
        "classMappingSource": "COCO categories" if coco_by_stem else "dataset YAML" if yolo_class_names else "standard COCO80",
        "rows": rows,
        "summary": summary,
    }
    saved_run = persist_run("dataset-batch", payload)
    return {
        "model": model,
        "degradation": degradation,
        "severity": severity,
        "annotationFormat": annotation_format,
        "classMappingSource": payload["classMappingSource"],
        "rows": rows,
        "summary": summary,
        "lastResult": last_result,
        "truncated": len(names) > 100,
        "savedRun": saved_run,
        "runtime": runtime_status(),
    }
