from __future__ import annotations

import importlib.metadata
import sys
import threading
from typing import Any

from .config import (
    BUILTIN_DATASET_PATH,
    COCO_VAL2017_ANNOTATIONS,
    COCO_VAL2017_IMAGES,
    DETR_MODEL_PATH,
    FASTER_RCNN_MODEL_PATH,
)

MODEL_CACHE: dict[str, Any] = {}
MODEL_ERRORS: dict[str, str] = {}
MODEL_LOCKS = {
    "torchvision": threading.Lock(),
    "detr": threading.Lock(),
}


def package_status() -> dict[str, bool]:
    packages = {}
    distribution_names = {
        "torch": "torch",
        "torchvision": "torchvision",
        "transformers": "transformers",
        "numpy": "numpy",
        "cv2": "opencv-python",
        "streamlit": "streamlit",
    }
    for import_name, distribution_name in distribution_names.items():
        try:
            importlib.metadata.version(distribution_name)
            packages[import_name] = True
        except importlib.metadata.PackageNotFoundError:
            packages[import_name] = False
    return packages


def local_model_status() -> dict[str, Any]:
    detr_required = ["config.json", "preprocessor_config.json", "model.safetensors"]
    detr_files = {name: (DETR_MODEL_PATH / name).exists() for name in detr_required}
    return {
        "detrResnet50": {
            "path": str(DETR_MODEL_PATH),
            "available": DETR_MODEL_PATH.exists() and all(detr_files.values()),
            "files": detr_files,
        },
        "fasterRcnnResnet50Fpn": {
            "path": str(FASTER_RCNN_MODEL_PATH),
            "available": FASTER_RCNN_MODEL_PATH.exists(),
            "bytes": FASTER_RCNN_MODEL_PATH.stat().st_size if FASTER_RCNN_MODEL_PATH.exists() else 0,
        },
    }


def runtime_status() -> dict[str, Any]:
    return {
        "python": sys.executable,
        "packages": package_status(),
        "localModels": local_model_status(),
        "localDatasets": {
            "coco128": {
                "path": str(BUILTIN_DATASET_PATH),
                "available": BUILTIN_DATASET_PATH.exists(),
            },
            "cocoVal2017": {
                "imagePath": str(COCO_VAL2017_IMAGES),
                "annotationPath": str(COCO_VAL2017_ANNOTATIONS),
                "available": COCO_VAL2017_IMAGES.is_dir() and COCO_VAL2017_ANNOTATIONS.is_file(),
            },
        },
        "modelsLoaded": sorted(MODEL_CACHE.keys()),
        "modelErrors": dict(MODEL_ERRORS),
    }
