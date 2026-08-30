from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .study_storage import save_study


REQUIRED_ROW_FIELDS = {
    "model",
    "degradation",
    "severity",
    "map",
    "ap50",
    "smallAP",
    "mediumAP",
    "largeAP",
    "retention",
    "classMetrics",
}


def load_completed_archive(path: Path) -> dict[str, Any]:
    archive = json.loads(path.read_text(encoding="utf-8"))
    validate_completed_archive(archive)
    return archive


def validate_completed_archive(archive: dict[str, Any]) -> None:
    if archive.get("schemaVersion") != 1:
        raise ValueError("unsupported study archive schema")
    study = archive.get("study")
    if not isinstance(study, dict) or study.get("status") != "completed":
        raise ValueError("the archive does not contain a completed study")
    result = study.get("result")
    config = study.get("config")
    if not isinstance(result, dict) or not isinstance(config, dict):
        raise ValueError("study configuration or result is missing")
    if study.get("id") != result.get("id"):
        raise ValueError("study identifiers do not match")
    rows = result.get("rows")
    if not isinstance(rows, list) or len(rows) != 36:
        raise ValueError("the completed study must contain 36 reported condition rows")
    if any(not REQUIRED_ROW_FIELDS.issubset(row) for row in rows):
        raise ValueError("one or more condition rows are incomplete")
    dataset = config.get("dataset", {})
    if dataset.get("evaluatedImages") != 500:
        raise ValueError("the archive is not the expected 500-image study")
    if result.get("summary", {}).get("runCount") != len(rows):
        raise ValueError("summary run count does not match the condition rows")


def import_completed_archive(path: Path) -> dict[str, Any]:
    archive = load_completed_archive(path)
    study = archive["study"]
    save_study(
        study_id=study["id"],
        status="completed",
        config=study["config"],
        result=study["result"],
    )
    return {
        "id": study["id"],
        "status": "completed",
        "imageCount": study["config"]["dataset"]["evaluatedImages"],
        "rowCount": len(study["result"]["rows"]),
    }
