from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .config import TRANSITION_ANALYSIS_PATH


MODELS = {"transformer", "cnn"}
DEGRADATIONS = {"blur", "lowlight", "jpeg"}
STATUSES = {"correct", "localisation", "classification", "missed"}


def _validate(payload: dict[str, Any]) -> None:
    if payload.get("schemaVersion") != 1:
        raise ValueError("unsupported transition-analysis schema")
    if int(payload.get("objectCount") or 0) <= 0:
        raise ValueError("transition analysis has no tracked objects")
    combinations = payload.get("combinations")
    if not isinstance(combinations, list) or len(combinations) != 6:
        raise ValueError("transition analysis must contain six model-degradation combinations")
    identities = {(item.get("model"), item.get("degradation")) for item in combinations}
    if identities != {(model, degradation) for model in MODELS for degradation in DEGRADATIONS}:
        raise ValueError("transition-analysis combinations are incomplete")
    for item in combinations:
        if len(item.get("severityCounts") or []) != 6 or len(item.get("steps") or []) != 5:
            raise ValueError("transition analysis has an incomplete severity trajectory")
        for step in item["steps"]:
            transitions = step.get("transitions") or []
            pairs = {(value.get("from"), value.get("to")) for value in transitions}
            if pairs != {(source, target) for source in STATUSES for target in STATUSES}:
                raise ValueError("transition matrix is incomplete")


def load_transition_analysis(path: Path = TRANSITION_ANALYSIS_PATH) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(
            "Object-level transition analysis is unavailable. Run experiments/analyze_object_transitions.py first."
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    _validate(payload)
    return payload


def transition_analysis(model: str, degradation: str) -> dict[str, Any]:
    if model not in MODELS:
        raise ValueError(f"unsupported transition model: {model}")
    if degradation not in DEGRADATIONS:
        raise ValueError(f"unsupported transition degradation: {degradation}")
    payload = load_transition_analysis()
    combination = next(
        item
        for item in payload["combinations"]
        if item["model"] == model and item["degradation"] == degradation
    )
    return {
        "schemaVersion": payload["schemaVersion"],
        "studyId": payload["studyId"],
        "generatedAt": payload["generatedAt"],
        "method": payload["method"],
        "objectCount": payload["objectCount"],
        "selection": {"model": model, "degradation": degradation},
        "analysis": combination,
    }
