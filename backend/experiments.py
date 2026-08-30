from __future__ import annotations

from typing import Any

from .domain import SCENES
from .images import degrade, draw_boxes, image_data_url, make_scene
from .metrics import evaluate
from .models import fallback_detector, run_detector
from .runtime import runtime_status


def build_evaluation(scene_id: str, model: str, degradation: str, severity: int, *, actual: bool = True) -> dict[str, Any]:
    clean = make_scene(scene_id)
    degraded = degrade(clean, degradation, severity)
    gt = SCENES[scene_id]["boxes"]
    if actual:
        preds, backend = run_detector(scene_id, model, degradation, severity, degraded)
    else:
        preds, backend = fallback_detector(scene_id, model, degradation, severity), "demonstration-sweep-model"
    result = draw_boxes(draw_boxes(degraded, gt, "#176b63"), preds, "#b84d36")
    return {
        "scene": scene_id,
        "sceneName": SCENES[scene_id]["name"],
        "model": model,
        "degradation": degradation,
        "severity": severity,
        "backend": backend,
        "cleanImage": image_data_url(draw_boxes(clean, gt, "#176b63")),
        "resultImage": image_data_url(result),
        "groundTruth": [box.to_dict() for box in gt],
        "predictions": [box.to_dict() for box in preds],
        "metrics": evaluate(gt, preds),
        "runtime": runtime_status(),
    }


def benchmark_row(scene_id: str, model: str, degradation: str, severity: int) -> dict[str, Any]:
    result = build_evaluation(scene_id, model, degradation, severity, actual=False)
    metrics = result["metrics"]
    failures = metrics["failures"]
    return {
        "scene": scene_id,
        "sceneName": result["sceneName"],
        "model": model,
        "degradation": degradation,
        "severity": severity,
        "map": metrics["map"],
        "ap50": metrics["ap50"],
        "smallAP": metrics["sizeAP"]["small"],
        "mediumAP": metrics["sizeAP"]["medium"],
        "largeAP": metrics["sizeAP"]["large"],
        "missed": failures["missed"],
        "falsePositive": failures["falsePositive"],
        "classification": failures["classification"],
        "localisation": failures["localisation"],
    }


def summarise_benchmark(rows: list[dict[str, Any]]) -> dict[str, Any]:
    averages = {}
    for model in ["transformer", "cnn"]:
        subset = [row for row in rows if row["model"] == model]
        averages[model] = sum(row["map"] for row in subset) / len(subset)
    degradation_impact = {}
    for degradation in ["blur", "lowlight", "jpeg"]:
        base = [row["map"] for row in rows if row["degradation"] == degradation and row["severity"] == 0]
        severe = [row["map"] for row in rows if row["degradation"] == degradation and row["severity"] == 5]
        degradation_impact[degradation] = ((sum(base) / len(base)) - (sum(severe) / len(severe))) if base and severe else 0
    worst = min(rows, key=lambda row: row["map"])
    best_model = "transformer" if averages["transformer"] >= averages["cnn"] else "cnn"
    worst_degradation = max(degradation_impact, key=degradation_impact.get)
    return {
        "runCount": len(rows),
        "averageMap": averages,
        "bestModel": best_model,
        "worstDegradation": worst_degradation,
        "worstCase": worst,
        "degradationImpact": degradation_impact,
    }
