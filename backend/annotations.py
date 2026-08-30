from __future__ import annotations

from pathlib import Path
from typing import Any

from .domain import COCO80_LABELS, Box
from .images import box_size_from_area


def coco_label_from_yolo_id(class_id: int, class_names: list[str] | None = None) -> str:
    names = class_names or COCO80_LABELS
    if 0 <= class_id < len(names):
        return str(names[class_id])
    return f"class_{class_id}"


def parse_yolo_dataset_yaml(text: str) -> list[str] | None:
    """Read the optional `names` mapping from a YOLO dataset YAML file."""
    try:
        import yaml

        data = yaml.safe_load(text)
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    names = data.get("names")
    if isinstance(names, list):
        return [str(name) for name in names]
    if isinstance(names, dict):
        try:
            return [str(names[key]) for key in sorted(names, key=lambda value: int(value))]
        except (TypeError, ValueError, KeyError):
            return None
    return None


def coco_category_name(category_id: int, categories: dict[int, str]) -> str:
    return categories.get(category_id, f"category_{category_id}")


def parse_coco_annotation_data(data: dict[str, Any]) -> tuple[dict[str, dict[str, Any]], dict[str, int]]:
    categories = {
        int(item.get("id")): str(item.get("name", f"category_{item.get('id')}"))
        for item in data.get("categories", [])
        if item.get("id") is not None
    }
    images_by_id: dict[int, dict[str, Any]] = {}
    annotations_by_stem: dict[str, dict[str, Any]] = {}
    class_counts: dict[str, int] = {}
    for image in data.get("images", []):
        image_id = image.get("id")
        file_name = image.get("file_name")
        if image_id is None or not file_name:
            continue
        try:
            image_id_int = int(image_id)
        except (TypeError, ValueError):
            continue
        stem = Path(str(file_name)).stem
        images_by_id[image_id_int] = {
            "stem": stem,
            "fileName": str(file_name),
            "width": float(image.get("width") or 0),
            "height": float(image.get("height") or 0),
        }
        annotations_by_stem.setdefault(stem, {"image": images_by_id[image_id_int], "annotations": []})

    for ann in data.get("annotations", []):
        image_id = ann.get("image_id")
        try:
            image_info = images_by_id[int(image_id)]
        except (TypeError, ValueError, KeyError):
            continue
        bbox = ann.get("bbox") or []
        if len(bbox) < 4:
            continue
        try:
            x, y, w, h = [float(value) for value in bbox[:4]]
            category_id = int(ann.get("category_id"))
        except (TypeError, ValueError):
            continue
        if w <= 0 or h <= 0:
            continue
        label = coco_category_name(category_id, categories)
        annotations_by_stem.setdefault(image_info["stem"], {"image": image_info, "annotations": []})["annotations"].append({
            "x": x,
            "y": y,
            "w": w,
            "h": h,
            "label": label,
        })
        class_counts[label] = class_counts.get(label, 0) + 1
    return annotations_by_stem, class_counts


def parse_coco_boxes(entry: dict[str, Any], width: int, height: int) -> list[Box]:
    image_info = entry.get("image", {})
    original_width = float(image_info.get("width") or width or 1)
    original_height = float(image_info.get("height") or height or 1)
    scale_x = width / original_width if original_width else 1
    scale_y = height / original_height if original_height else 1
    boxes: list[Box] = []
    for ann in entry.get("annotations", []):
        x = float(ann["x"]) * scale_x
        y = float(ann["y"]) * scale_y
        w = float(ann["w"]) * scale_x
        h = float(ann["h"]) * scale_y
        boxes.append(Box(x, y, w, h, str(ann["label"]), box_size_from_area(w * h)))
    return boxes


def parse_yolo_label_text(
    text: str,
    width: int,
    height: int,
    class_names: list[str] | None = None,
) -> list[Box]:
    boxes: list[Box] = []
    for line in text.splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        try:
            class_id = int(float(parts[0]))
            cx, cy, bw, bh = [float(value) for value in parts[1:5]]
        except ValueError:
            continue
        w = bw * width
        h = bh * height
        x = cx * width - w / 2
        y = cy * height - h / 2
        label = coco_label_from_yolo_id(class_id, class_names)
        boxes.append(Box(x, y, w, h, label, box_size_from_area(w * h)))
    return boxes
