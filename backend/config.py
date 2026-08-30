from __future__ import annotations

from pathlib import Path

from .settings import Settings

ROOT = Path(__file__).resolve().parent.parent
SETTINGS = Settings.from_environment(ROOT)
STATIC_DIR = Path(__file__).resolve().parent / "static"
RUNS_DIR = ROOT / "runs"
RUNS_FILE = RUNS_DIR / "experiments.json"
STUDY_DB = RUNS_DIR / "experiments.sqlite3"
RESEARCH_DATA_DIR = Path(__file__).resolve().parent / "research_data"
TRANSITION_ANALYSIS_PATH = RESEARCH_DATA_DIR / "object_failure_transitions.json"
COMPLETED_STUDY_ARCHIVE = RESEARCH_DATA_DIR / "chapter4_completed_study.json"
BUILTIN_DATASET_PATH = SETTINGS.builtin_dataset_path
COCO_VAL2017_ROOT = SETTINGS.coco_val2017_root
COCO_VAL2017_IMAGES = COCO_VAL2017_ROOT / "images" / "val2017"
COCO_VAL2017_ANNOTATIONS = COCO_VAL2017_ROOT / "annotations" / "instances_val2017.json"
DETR_MODEL_PATH = SETTINGS.detr_model_path
FASTER_RCNN_MODEL_PATH = SETTINGS.faster_rcnn_model_path
WIDTH, HEIGHT = 640, 420
EVALUATION_SCORE_THRESHOLD = 0.05
EVALUATION_MAX_DETECTIONS = 100
MAX_STUDY_IMAGES = 5000

ALLOW_MODEL_DOWNLOAD = SETTINGS.allow_model_download
USE_REAL_MODELS = SETTINGS.use_real_models
