from __future__ import annotations

import hashlib
import importlib.metadata
import io
import json
import platform
import random
import threading
import time
import uuid
import zipfile
from collections import Counter
from dataclasses import dataclass
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
from .config import (
    BUILTIN_DATASET_PATH,
    COCO_VAL2017_ANNOTATIONS,
    COCO_VAL2017_IMAGES,
    DETR_MODEL_PATH,
    EVALUATION_MAX_DETECTIONS,
    EVALUATION_SCORE_THRESHOLD,
    FASTER_RCNN_MODEL_PATH,
    MAX_STUDY_IMAGES,
)
from .images import bytes_from_data_url, degradation_parameters, degrade, stable_seed
from .metrics import evaluate_detection_dataset
from .models import run_uploaded_detector
from .storage import now_iso
from .study_storage import latest_completed_study, list_studies, save_study
from .uploads import SUPPORTED_IMAGE_SUFFIXES

DEGRADATIONS = ("blur", "lowlight", "jpeg")
MODELS = ("transformer", "cnn")
JOBS: dict[str, dict[str, Any]] = {}
JOB_CANCEL: dict[str, threading.Event] = {}
JOBS_LOCK = threading.Lock()
HASH_CACHE: dict[str, str] = {}


@dataclass
class DatasetSource:
    kind: str
    name: str
    raw: bytes | None = None
    image_root: Path | None = None
    annotation_path: Path | None = None


def _file_hash(path: Path) -> str:
    cache_key = str(path)
    if cache_key in HASH_CACHE:
        return HASH_CACHE[cache_key]
    if not path.exists():
        raise FileNotFoundError(path)
    digest = hashlib.sha256()
    paths = sorted(item for item in path.rglob("*") if item.is_file()) if path.is_dir() else [path]
    for item in paths:
        digest.update(str(item.relative_to(path) if path.is_dir() else item.name).encode("utf-8"))
        with item.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    result = digest.hexdigest()
    HASH_CACHE[cache_key] = result
    return result


def _source_hash(source: DatasetSource) -> str:
    if source.kind == "zip":
        return hashlib.sha256(source.raw or b"").hexdigest()
    dataset_root = (source.image_root or Path()).parent.parent
    manifest_path = dataset_root / "dataset_manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        checksums = manifest.get("sha256", {})
        if checksums:
            return hashlib.sha256(json.dumps(checksums, sort_keys=True).encode("utf-8")).hexdigest()
    digest = hashlib.sha256()
    digest.update(_file_hash(source.annotation_path or Path()).encode("ascii"))
    digest.update(_file_hash(source.image_root or Path()).encode("ascii"))
    return digest.hexdigest()


def _versions() -> dict[str, str]:
    versions = {"python": platform.python_version()}
    for package in ("torch", "torchvision", "transformers", "numpy", "opencv-python", "pycocotools", "Pillow"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    return versions


def _resolve_dataset(payload: dict[str, Any]) -> DatasetSource:
    archive = payload.get("archive")
    if archive:
        return DatasetSource(
            kind="zip",
            name=str(payload.get("datasetName") or "uploaded-dataset.zip"),
            raw=bytes_from_data_url(str(archive)),
        )
    if payload.get("datasetSource") == "coco-val2017":
        if not COCO_VAL2017_IMAGES.is_dir() or not COCO_VAL2017_ANNOTATIONS.is_file():
            raise FileNotFoundError(
                f"COCO val2017 is incomplete. Expected {COCO_VAL2017_IMAGES} and {COCO_VAL2017_ANNOTATIONS}"
            )
        return DatasetSource(
            kind="directory",
            name="COCO val2017",
            image_root=COCO_VAL2017_IMAGES,
            annotation_path=COCO_VAL2017_ANNOTATIONS,
        )
    if not BUILTIN_DATASET_PATH.exists():
        raise FileNotFoundError(f"Built-in dataset not found: {BUILTIN_DATASET_PATH}")
    return DatasetSource(kind="zip", name=BUILTIN_DATASET_PATH.name, raw=BUILTIN_DATASET_PATH.read_bytes())


def _yolo_labels(text: str, class_names: list[str] | None) -> set[str]:
    labels: set[str] = set()
    for line in text.splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        try:
            class_id = int(float(parts[0]))
        except ValueError:
            continue
        labels.add(coco_label_from_yolo_id(class_id, class_names))
    return labels


def _balanced_select(candidates: list[dict[str, Any]], limit: int, seed: int) -> list[dict[str, Any]]:
    """Create a deterministic class-aware subset without assuming one label per image."""
    ordered = sorted(candidates, key=lambda item: item["name"])
    if limit >= len(ordered):
        return ordered
    rng = random.Random(seed)
    rng.shuffle(ordered)
    selected: list[dict[str, Any]] = []
    selected_names: set[str] = set()
    class_counts: Counter[str] = Counter()

    all_labels = sorted({label for item in ordered for label in item["labels"]})
    for label in all_labels:
        choices = [item for item in ordered if item["name"] not in selected_names and label in item["labels"]]
        if not choices or len(selected) >= limit:
            continue
        choice = min(
            choices,
            key=lambda item: (sum(class_counts[value] for value in item["labels"]), item["name"]),
        )
        selected.append(choice)
        selected_names.add(choice["name"])
        class_counts.update(choice["labels"])

    remaining = [item for item in ordered if item["name"] not in selected_names]
    while remaining and len(selected) < limit:
        probe = rng.sample(remaining, min(256, len(remaining)))
        choice = min(
            probe,
            key=lambda item: (
                sum(class_counts[label] for label in item["labels"]) / max(1, len(item["labels"])),
                item["name"],
            ),
        )
        selected.append(choice)
        class_counts.update(choice["labels"])
        remaining.remove(choice)
    return sorted(selected, key=lambda item: item["name"])


def _load_zip_samples(source: DatasetSource, limit: int, seed: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    labels: dict[str, str] = {}
    coco_by_stem: dict[str, dict[str, Any]] = {}
    yolo_names: list[str] | None = None
    raw = source.raw or b""
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        for item in archive.infolist():
            if item.is_dir() or item.filename.startswith("__MACOSX/"):
                continue
            suffix = Path(item.filename).suffix.lower()
            try:
                if suffix == ".txt":
                    labels[Path(item.filename).stem] = archive.read(item).decode("utf-8", errors="ignore")
                elif suffix in {".yaml", ".yml"}:
                    yolo_names = parse_yolo_dataset_yaml(archive.read(item).decode("utf-8", errors="ignore")) or yolo_names
                elif suffix == ".json":
                    data = json.loads(archive.read(item).decode("utf-8", errors="ignore"))
                    if isinstance(data, dict) and "images" in data and "annotations" in data:
                        parsed, _ = parse_coco_annotation_data(data)
                        coco_by_stem.update(parsed)
            except Exception:
                continue

        names = sorted(
            item.filename
            for item in archive.infolist()
            if not item.is_dir()
            and not item.filename.startswith("__MACOSX/")
            and Path(item.filename).suffix.lower() in SUPPORTED_IMAGE_SUFFIXES
        )
        candidates: list[dict[str, Any]] = []
        for name in names:
            stem = Path(name).stem
            if stem in coco_by_stem and coco_by_stem[stem].get("annotations"):
                item_labels = {str(item["label"]) for item in coco_by_stem[stem]["annotations"]}
                candidates.append({"name": name, "labels": item_labels, "annotation": coco_by_stem[stem]})
            elif stem in labels:
                item_labels = _yolo_labels(labels[stem], yolo_names)
                if item_labels:
                    candidates.append({"name": name, "labels": item_labels, "labelText": labels[stem]})

        selected = _balanced_select(candidates, min(limit, len(candidates)), seed)
        samples: list[dict[str, Any]] = []
        for item in selected:
            try:
                with archive.open(item["name"]) as handle:
                    image = Image.open(handle).convert("RGB")
                    image.thumbnail((960, 720))
                ground_truth = (
                    parse_coco_boxes(item["annotation"], image.width, image.height)
                    if item.get("annotation")
                    else parse_yolo_label_text(item["labelText"], image.width, image.height, yolo_names)
                )
                if ground_truth:
                    samples.append({
                        "name": item["name"],
                        "archiveName": item["name"],
                        "width": image.width,
                        "height": image.height,
                        "groundTruth": ground_truth,
                    })
            except Exception:
                continue
    annotation_format = "COCO JSON" if coco_by_stem else "YOLO TXT"
    mapping = "COCO categories" if coco_by_stem else "dataset YAML" if yolo_names else "standard COCO80"
    return samples, _dataset_info(
        source,
        samples,
        available_images=len(names),
        eligible_images=len(candidates),
        annotation_format=annotation_format,
        mapping=mapping,
        requested=limit,
        seed=seed,
    )


def _load_coco_directory_samples(
    source: DatasetSource,
    limit: int,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    data = json.loads((source.annotation_path or Path()).read_text(encoding="utf-8"))
    coco_by_stem, _ = parse_coco_annotation_data(data)
    image_root = source.image_root or Path()
    candidates: list[dict[str, Any]] = []
    for stem, entry in coco_by_stem.items():
        if not entry.get("annotations"):
            continue
        file_name = Path(str(entry.get("image", {}).get("fileName") or f"{stem}.jpg")).name
        image_path = image_root / file_name
        if not image_path.is_file():
            continue
        candidates.append({
            "name": file_name,
            "imagePath": image_path,
            "labels": {str(item["label"]) for item in entry["annotations"]},
            "annotation": entry,
        })

    selected = _balanced_select(candidates, min(limit, len(candidates)), seed)
    samples: list[dict[str, Any]] = []
    for item in selected:
        image_info = item["annotation"].get("image", {})
        width = int(float(image_info.get("width") or 0))
        height = int(float(image_info.get("height") or 0))
        if width <= 0 or height <= 0:
            try:
                with Image.open(item["imagePath"]) as image:
                    width, height = image.size
            except Exception:
                continue
        ground_truth = parse_coco_boxes(item["annotation"], width, height)
        if ground_truth:
            samples.append({
                "name": item["name"],
                "imagePath": str(item["imagePath"]),
                "width": width,
                "height": height,
                "groundTruth": ground_truth,
            })
    available_images = len(list(image_root.glob("*.jpg")))
    return samples, _dataset_info(
        source,
        samples,
        available_images=available_images,
        eligible_images=len(candidates),
        annotation_format="COCO JSON",
        mapping="COCO categories",
        requested=limit,
        seed=seed,
    )


def _dataset_info(
    source: DatasetSource,
    samples: list[dict[str, Any]],
    *,
    available_images: int,
    eligible_images: int,
    annotation_format: str,
    mapping: str,
    requested: int,
    seed: int,
) -> dict[str, Any]:
    class_counts: Counter[str] = Counter(
        box.label for sample in samples for box in sample["groundTruth"]
    )
    return {
        "sourceType": source.kind,
        "sourcePath": str(source.image_root.parent.parent) if source.image_root else "uploaded archive",
        "availableImages": available_images,
        "eligibleImages": eligible_images,
        "requestedImages": requested,
        "evaluatedImages": len(samples),
        "annotationFormat": annotation_format,
        "classMappingSource": mapping,
        "samplingMethod": "all eligible images" if requested >= eligible_images else "seeded class-aware balanced subset",
        "samplingSeed": seed,
        "fingerprintMethod": "verified import manifest" if source.kind == "directory" and ((source.image_root or Path()).parent.parent / "dataset_manifest.json").is_file() else "direct content hash",
        "selectedClassCounts": dict(sorted(class_counts.items())),
    }


def load_dataset_samples(
    source: DatasetSource | bytes,
    limit: int,
    seed: int = 2026,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    source = source if isinstance(source, DatasetSource) else DatasetSource("zip", "in-memory.zip", raw=source)
    samples, info = (
        _load_coco_directory_samples(source, limit, seed)
        if source.kind == "directory"
        else _load_zip_samples(source, limit, seed)
    )
    if not samples:
        raise ValueError("The dataset contains no readable images with matching COCO or YOLO annotations.")
    return samples, info


def _load_sample_image(source: DatasetSource, sample: dict[str, Any], archive: zipfile.ZipFile | None) -> Image.Image:
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            if source.kind == "directory":
                raw = Path(sample["imagePath"]).read_bytes()
            else:
                if archive is None:
                    raise RuntimeError("ZIP dataset reader is not open")
                raw = archive.read(sample["archiveName"])
            with Image.open(io.BytesIO(raw)) as handle:
                handle.load()
                image = handle.convert("RGB")
            if source.kind == "zip":
                image.thumbnail((960, 720))
            return image
        except (OSError, ValueError) as exc:
            last_error = exc
            time.sleep(0.05 * (attempt + 1))
    raise RuntimeError(f"Unable to decode {sample['name']} after 3 attempts: {last_error}")


def _job_update(study_id: str, **values: Any) -> None:
    with JOBS_LOCK:
        if study_id in JOBS:
            JOBS[study_id].update(values)


def _metric_row(
    model: str,
    degradation: str,
    severity: int,
    metrics: dict[str, Any],
    image_count: int,
    elapsed: float,
    baseline_map: float,
    dataset_name: str,
) -> dict[str, Any]:
    current_map = float(metrics.get("map") or 0.0)
    failures = metrics["failures"]
    return {
        "scene": "dataset",
        "sceneName": f"{dataset_name} evaluation subset",
        "model": model,
        "degradation": degradation,
        "severity": severity,
        "degradationParameters": degradation_parameters(degradation, severity),
        "imageCount": image_count,
        "inferenceSeconds": elapsed,
        "map": current_map,
        "ap50": float(metrics.get("ap50") or 0.0),
        "ap75": float(metrics.get("ap75") or 0.0),
        "ar100": float(metrics.get("ar100") or 0.0),
        "smallAP": float(metrics["sizeAP"].get("small") or 0.0),
        "mediumAP": float(metrics["sizeAP"].get("medium") or 0.0),
        "largeAP": float(metrics["sizeAP"].get("large") or 0.0),
        "smallAR": float(metrics["sizeAR"].get("small") or 0.0),
        "mediumAR": float(metrics["sizeAR"].get("medium") or 0.0),
        "largeAR": float(metrics["sizeAR"].get("large") or 0.0),
        "missed": int(failures["missed"]),
        "falsePositive": int(failures["falsePositive"]),
        "classification": int(failures["classification"]),
        "localisation": int(failures["localisation"]),
        "cleanMap": baseline_map,
        "absoluteDrop": baseline_map - current_map,
        "retention": current_map / baseline_map if baseline_map else 0.0,
        "classMetrics": metrics["classMetrics"],
    }


def _study_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    average_map: dict[str, float] = {}
    robustness_auc: dict[str, dict[str, float]] = {}
    degradation_impact: dict[str, float] = {}
    for model in MODELS:
        subset = [row for row in rows if row["model"] == model]
        average_map[model] = sum(row["map"] for row in subset) / len(subset) if subset else 0.0
        robustness_auc[model] = {}
        for degradation in DEGRADATIONS:
            curve = sorted(
                [row for row in subset if row["degradation"] == degradation],
                key=lambda row: row["severity"],
            )
            area = sum((curve[index]["map"] + curve[index + 1]["map"]) / 2 for index in range(len(curve) - 1))
            robustness_auc[model][degradation] = area / 5 if len(curve) == 6 else 0.0
    for degradation in DEGRADATIONS:
        clean = [row["map"] for row in rows if row["degradation"] == degradation and row["severity"] == 0]
        severe = [row["map"] for row in rows if row["degradation"] == degradation and row["severity"] == 5]
        degradation_impact[degradation] = (sum(clean) / len(clean) - sum(severe) / len(severe)) if clean and severe else 0.0
    worst = min(rows, key=lambda row: row["map"])
    return {
        "runCount": len(rows),
        "averageMap": average_map,
        "modelGap": average_map.get("transformer", 0.0) - average_map.get("cnn", 0.0),
        "bestModel": max(average_map, key=average_map.get),
        "worstDegradation": max(degradation_impact, key=degradation_impact.get),
        "worstCase": worst,
        "degradationImpact": degradation_impact,
        "robustnessAuc": robustness_auc,
    }


def _evaluate_condition(
    study_id: str,
    cancel_event: threading.Event,
    source: DatasetSource,
    samples: list[dict[str, Any]],
    model: str,
    degradation: str,
    severity: int,
    progress: list[int],
    total: int,
    random_seed: int,
) -> tuple[dict[str, Any], float]:
    evaluation_samples: list[dict[str, Any]] = []
    started = time.perf_counter()
    archive = zipfile.ZipFile(io.BytesIO(source.raw or b"")) if source.kind == "zip" else None
    try:
        for sample in samples:
            if cancel_event.is_set():
                raise InterruptedError("Study cancelled by user")
            clean_image = _load_sample_image(source, sample, archive)
            image = degrade(
                clean_image,
                degradation,
                severity,
                seed=stable_seed(f"{random_seed}:{sample['name']}"),
            )
            predictions, backend = run_uploaded_detector(model, image)
            evaluation_samples.append({
                "name": sample["name"],
                "width": sample["width"],
                "height": sample["height"],
                "groundTruth": sample["groundTruth"],
                "predictions": predictions,
            })
            progress[0] += 1
            _job_update(
                study_id,
                completedTasks=progress[0],
                progress=progress[0] / total,
                current=f"{model} / {degradation} / severity {severity} / {sample['name']}",
                backend=backend,
            )
    finally:
        if archive:
            archive.close()
    return evaluate_detection_dataset(evaluation_samples), time.perf_counter() - started


def _run_study(study_id: str, source: DatasetSource, max_images: int, seed: int) -> None:
    cancel_event = JOB_CANCEL[study_id]
    try:
        _job_update(study_id, status="loading-dataset", current="Reading annotations and selecting a balanced subset")
        samples, dataset_info = load_dataset_samples(source, max_images, seed)
        total = len(samples) * len(MODELS) * (1 + len(DEGRADATIONS) * 5)
        config = {
            "dataset": {"name": source.name, "sha256": _source_hash(source), **dataset_info},
            "models": list(MODELS),
            "degradations": list(DEGRADATIONS),
            "severities": list(range(6)),
            "seed": seed,
            "scoreThreshold": EVALUATION_SCORE_THRESHOLD,
            "maxDetections": EVALUATION_MAX_DETECTIONS,
            "evaluator": "pycocotools COCOeval bbox",
            "versions": _versions(),
            "modelHashes": {
                "transformer": _file_hash(DETR_MODEL_PATH),
                "cnn": _file_hash(FASTER_RCNN_MODEL_PATH),
            },
        }
        _job_update(
            study_id,
            status="running",
            config=config,
            totalTasks=total,
            completedTasks=0,
            progress=0.0,
            current="Loading model weights",
        )
        save_study(study_id, "running", config)
        rows: list[dict[str, Any]] = []
        progress = [0]
        for model in MODELS:
            clean_metrics, clean_elapsed = _evaluate_condition(
                study_id, cancel_event, source, samples, model, "blur", 0, progress, total, seed
            )
            baseline_map = float(clean_metrics.get("map") or 0.0)
            for degradation in DEGRADATIONS:
                rows.append(_metric_row(
                    model, degradation, 0, clean_metrics, len(samples), clean_elapsed, baseline_map, source.name
                ))
            for degradation in DEGRADATIONS:
                for severity in range(1, 6):
                    metrics, elapsed = _evaluate_condition(
                        study_id, cancel_event, source, samples, model, degradation, severity, progress, total, seed
                    )
                    rows.append(_metric_row(
                        model, degradation, severity, metrics, len(samples), elapsed, baseline_map, source.name
                    ))
        result = {
            "id": study_id,
            "completedAt": now_iso(),
            "rows": rows,
            "summary": _study_summary(rows),
            "config": config,
        }
        _job_update(
            study_id,
            status="completed",
            completedAt=result["completedAt"],
            completedTasks=total,
            progress=1.0,
            current="Study completed",
            result=result,
        )
        save_study(study_id, "completed", config, result=result)
    except InterruptedError as exc:
        config = get_study_status(study_id).get("config", {})
        _job_update(study_id, status="cancelled", current=str(exc), error=str(exc))
        save_study(study_id, "cancelled", config, error=str(exc))
    except Exception as exc:
        config = get_study_status(study_id).get("config", {})
        _job_update(study_id, status="failed", current="Study failed", error=str(exc))
        save_study(study_id, "failed", config, error=str(exc))


def start_study(payload: dict[str, Any]) -> dict[str, Any]:
    with JOBS_LOCK:
        active = next((job for job in JOBS.values() if job["status"] in {"queued", "loading-dataset", "running"}), None)
        if active:
            return dict(active)
    max_images = max(1, min(MAX_STUDY_IMAGES, int(payload.get("maxImages", 4))))
    seed = int(payload.get("seed", 2026))
    source = _resolve_dataset(payload)
    study_id = f"study-{uuid.uuid4().hex[:12]}"
    job = {
        "id": study_id,
        "status": "queued",
        "createdAt": now_iso(),
        "datasetName": source.name,
        "maxImages": max_images,
        "completedTasks": 0,
        "totalTasks": None,
        "progress": 0.0,
        "current": "Queued",
        "error": None,
    }
    with JOBS_LOCK:
        JOBS[study_id] = job
        JOB_CANCEL[study_id] = threading.Event()
    thread = threading.Thread(
        target=_run_study,
        args=(study_id, source, max_images, seed),
        name=f"robustness-{study_id}",
        daemon=True,
    )
    thread.start()
    return dict(job)


def get_study_status(study_id: str) -> dict[str, Any]:
    with JOBS_LOCK:
        job = JOBS.get(study_id)
        return dict(job) if job else {"id": study_id, "status": "not-found", "error": "Unknown study ID"}


def cancel_study(study_id: str) -> dict[str, Any]:
    with JOBS_LOCK:
        event = JOB_CANCEL.get(study_id)
    if not event:
        return {"id": study_id, "status": "not-found", "error": "Unknown study ID"}
    event.set()
    _job_update(study_id, current="Cancellation requested")
    return get_study_status(study_id)


def latest_study_result() -> dict[str, Any] | None:
    stored = latest_completed_study()
    return stored["result"] if stored else None


def study_history() -> list[dict[str, Any]]:
    return list_studies()


def sweep_from_latest(degradation: str) -> dict[str, Any] | None:
    result = latest_study_result()
    if not result:
        return None
    rows = []
    for severity in range(6):
        transformer = next((row for row in result["rows"] if row["model"] == "transformer" and row["degradation"] == degradation and row["severity"] == severity), None)
        cnn = next((row for row in result["rows"] if row["model"] == "cnn" and row["degradation"] == degradation and row["severity"] == severity), None)
        if transformer and cnn:
            rows.append({
                "severity": severity,
                "transformer": {
                    "map": transformer["map"], "ap50": transformer["ap50"], "ap75": transformer["ap75"],
                    "sizeAP": {"small": transformer["smallAP"], "medium": transformer["mediumAP"], "large": transformer["largeAP"]},
                    "failures": {"missed": transformer["missed"], "falsePositive": transformer["falsePositive"], "classification": transformer["classification"], "localisation": transformer["localisation"]},
                },
                "cnn": {
                    "map": cnn["map"], "ap50": cnn["ap50"], "ap75": cnn["ap75"],
                    "sizeAP": {"small": cnn["smallAP"], "medium": cnn["mediumAP"], "large": cnn["largeAP"]},
                    "failures": {"missed": cnn["missed"], "falsePositive": cnn["falsePositive"], "classification": cnn["classification"], "localisation": cnn["localisation"]},
                },
            })
    return {"degradation": degradation, "rows": rows, "studyId": result["id"], "config": result["config"]}
