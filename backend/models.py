from __future__ import annotations

from .config import ALLOW_MODEL_DOWNLOAD, DETR_MODEL_PATH, FASTER_RCNN_MODEL_PATH, USE_REAL_MODELS
from .domain import COCO_LABELS, SCENES, Box
from .images import stable_random
from .runtime import MODEL_CACHE, MODEL_ERRORS, MODEL_LOCKS

try:
    from PIL import Image
except Exception as exc:  # pragma: no cover
    raise SystemExit("Pillow is required. Install the project dependencies in the selected environment.") from exc


def fallback_detector(scene_id: str, model: str, degradation: str, severity: int) -> list[Box]:
    profile = {
        "transformer": {"base": 0.97, "sens": {"blur": 0.072, "lowlight": 0.06, "jpeg": 0.046}},
        "cnn": {"base": 0.94, "sens": {"blur": 0.054, "lowlight": 0.078, "jpeg": 0.06}},
    }[model]
    penalties = {"small": 0.062, "medium": 0.024, "large": 0.01}
    preds = []
    for i, box in enumerate(SCENES[scene_id]["boxes"]):
        keep = profile["base"] - profile["sens"][degradation] * severity - penalties[box.size] * severity
        if stable_random(f"{scene_id}-{model}-{degradation}-{severity}-{i}") < keep:
            jitter = 2 + severity * (2.4 if box.size == "small" else 1.7)
            dx = (stable_random(f"dx-{scene_id}-{model}-{severity}-{i}") - 0.5) * jitter * 2
            dy = (stable_random(f"dy-{scene_id}-{model}-{severity}-{i}") - 0.5) * jitter * 2
            label = "wrong class" if stable_random(f"cls-{scene_id}-{model}-{severity}-{i}") < 0.032 * severity else box.label
            preds.append(Box(box.x + dx, box.y + dy, box.w, box.h, label, box.size, max(0.1, min(0.99, keep))))
    return preds


def detect_with_torchvision(image: Image.Image) -> tuple[list[Box], str]:
    try:
        import torch
        import torchvision
        from torchvision.transforms import functional as F
    except Exception as exc:
        MODEL_ERRORS["torchvision"] = f"import failed: {exc}"
        return [], "torchvision-unavailable"

    try:
        with MODEL_LOCKS["torchvision"]:
            if "torchvision" not in MODEL_CACHE:
                if not USE_REAL_MODELS:
                    MODEL_ERRORS["torchvision"] = "real model mode disabled; using demonstration fallback where available"
                    return [], "torchvision-disabled"
                if FASTER_RCNN_MODEL_PATH.exists():
                    detector = torchvision.models.detection.fasterrcnn_resnet50_fpn(
                        weights=None,
                        weights_backbone=None,
                    )
                    state_dict = torch.load(
                        FASTER_RCNN_MODEL_PATH,
                        map_location="cpu",
                        weights_only=True,
                    )
                    detector.load_state_dict(state_dict)
                elif ALLOW_MODEL_DOWNLOAD:
                    weights = torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT
                    detector = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=weights)
                else:
                    raise FileNotFoundError(f"Faster R-CNN weights not found: {FASTER_RCNN_MODEL_PATH}")
                detector.eval()
                MODEL_CACHE["torchvision"] = detector
            detector = MODEL_CACHE["torchvision"]
            tensor = F.to_tensor(image)
            with torch.no_grad():
                output = detector([tensor])[0]
        boxes = []
        for raw_box, raw_label, raw_score in zip(output["boxes"], output["labels"], output["scores"]):
            score = float(raw_score)
            if score < 0.05:
                continue
            x1, y1, x2, y2 = [float(v) for v in raw_box.tolist()]
            label_id = int(raw_label)
            label = COCO_LABELS[label_id] if label_id < len(COCO_LABELS) else str(label_id)
            area = (x2 - x1) * (y2 - y1)
            size = "small" if area < 32 * 32 else "medium" if area < 96 * 96 else "large"
            boxes.append(Box(x1, y1, x2 - x1, y2 - y1, label, size, score))
            if len(boxes) >= 100:
                break
        MODEL_ERRORS.pop("torchvision", None)
        return boxes, "torchvision-fasterrcnn"
    except Exception as exc:
        MODEL_ERRORS["torchvision"] = str(exc)
        return [], "torchvision-load-failed"


def detect_with_detr(image: Image.Image) -> tuple[list[Box], str]:
    if not USE_REAL_MODELS and "detr" not in MODEL_CACHE:
        MODEL_ERRORS["detr"] = "real model mode disabled; using demonstration fallback where available"
        return [], "detr-disabled"
    try:
        import torch
        from transformers import DetrForObjectDetection, DetrImageProcessor
    except Exception as exc:
        MODEL_ERRORS["detr"] = f"import failed: {exc}"
        return [], "detr-unavailable"

    try:
        with MODEL_LOCKS["detr"]:
            if "detr" not in MODEL_CACHE:
                model_source = str(DETR_MODEL_PATH) if DETR_MODEL_PATH.exists() else "facebook/detr-resnet-50"
                use_local_only = DETR_MODEL_PATH.exists() or not ALLOW_MODEL_DOWNLOAD
                processor = DetrImageProcessor.from_pretrained(model_source, local_files_only=use_local_only)
                detector = DetrForObjectDetection.from_pretrained(
                    model_source,
                    local_files_only=use_local_only,
                    low_cpu_mem_usage=False,
                )
                detector.eval()
                MODEL_CACHE["detr"] = (processor, detector)
            processor, detector = MODEL_CACHE["detr"]
            inputs = processor(images=image, return_tensors="pt")
            with torch.no_grad():
                outputs = detector(**inputs)
            target_sizes = torch.tensor([[image.height, image.width]])
            results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.05)[0]
        boxes = []
        for raw_box, raw_label, raw_score in zip(results["boxes"], results["labels"], results["scores"]):
            score = float(raw_score)
            x1, y1, x2, y2 = [float(v) for v in raw_box.tolist()]
            label = detector.config.id2label[int(raw_label)]
            area = (x2 - x1) * (y2 - y1)
            size = "small" if area < 32 * 32 else "medium" if area < 96 * 96 else "large"
            boxes.append(Box(x1, y1, x2 - x1, y2 - y1, label, size, score))
            if len(boxes) >= 100:
                break
        MODEL_ERRORS.pop("detr", None)
        return boxes, "transformers-detr-resnet-50"
    except Exception as exc:
        MODEL_ERRORS["detr"] = str(exc)
        return [], "detr-load-failed"


def run_detector(scene_id: str, model: str, degradation: str, severity: int, image: Image.Image) -> tuple[list[Box], str]:
    if model == "cnn":
        preds, backend = detect_with_torchvision(image)
        if preds or backend == "torchvision-fasterrcnn":
            return preds, backend
    if model == "transformer":
        preds, backend = detect_with_detr(image)
        if preds or backend == "transformers-detr-resnet-50":
            return preds, backend
    if USE_REAL_MODELS:
        error_key = "torchvision" if model == "cnn" else "detr"
        detail = MODEL_ERRORS.get(error_key, "unknown model error")
        raise RuntimeError(f"{model} real-model inference failed: {detail}")
    return fallback_detector(scene_id, model, degradation, severity), "fallback-detector"


def run_uploaded_detector(model: str, image: Image.Image) -> tuple[list[Box], str]:
    if model == "cnn":
        preds, backend = detect_with_torchvision(image)
    elif model == "transformer":
        preds, backend = detect_with_detr(image)
    else:
        raise ValueError(f"Unsupported detector: {model}")
    if USE_REAL_MODELS and backend not in {"torchvision-fasterrcnn", "transformers-detr-resnet-50"}:
        error_key = "torchvision" if model == "cnn" else "detr"
        detail = MODEL_ERRORS.get(error_key, "unknown model error")
        raise RuntimeError(f"{model} real-model inference failed: {detail}")
    return preds, backend
