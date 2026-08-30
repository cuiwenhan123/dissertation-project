from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _boolean(name: str, default: bool, *, legacy: str | None = None) -> bool:
    raw = os.environ.get(name)
    if raw is None and legacy:
        raw = os.environ.get(legacy)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True, slots=True)
class Settings:
    root: Path
    host: str
    port: int
    log_level: str
    max_request_bytes: int
    use_real_models: bool
    allow_model_download: bool
    coco_val2017_root: Path
    detr_model_path: Path
    faster_rcnn_model_path: Path
    builtin_dataset_path: Path

    @classmethod
    def from_environment(cls, root: Path) -> "Settings":
        return cls(
            root=root,
            host=os.environ.get("ROBUSTNESS_HOST", "127.0.0.1"),
            port=int(os.environ.get("ROBUSTNESS_PORT", "8877")),
            log_level=os.environ.get("ROBUSTNESS_LOG_LEVEL", "INFO").upper(),
            max_request_bytes=int(os.environ.get("ROBUSTNESS_MAX_REQUEST_BYTES", str(128 * 1024 * 1024))),
            use_real_models=_boolean(
                "ROBUSTNESS_USE_REAL_MODELS",
                True,
                legacy="PROTOTYPE_USE_REAL_MODELS",
            ),
            allow_model_download=_boolean(
                "ROBUSTNESS_ALLOW_MODEL_DOWNLOAD",
                False,
                legacy="PROTOTYPE_ALLOW_MODEL_DOWNLOAD",
            ),
            coco_val2017_root=Path(
                os.environ.get(
                    "ROBUSTNESS_COCO_VAL2017_ROOT",
                    str(Path.home() / "Desktop" / "DetectionRobustnessDatasets" / "coco_val2017"),
                )
            ).expanduser(),
            detr_model_path=Path(
                os.environ.get("ROBUSTNESS_DETR_MODEL_PATH", str(root / "models" / "detr-resnet-50"))
            ).expanduser(),
            faster_rcnn_model_path=Path(
                os.environ.get(
                    "ROBUSTNESS_FASTER_RCNN_MODEL_PATH",
                    str(root / "models" / "fasterrcnn-resnet50-fpn-coco.pth"),
                )
            ).expanduser(),
            builtin_dataset_path=Path(
                os.environ.get("ROBUSTNESS_BUILTIN_DATASET_PATH", str(root / "datasets" / "coco128.zip"))
            ).expanduser(),
        )
