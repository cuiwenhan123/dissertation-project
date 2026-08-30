from __future__ import annotations

import json
import time
from typing import Any

from .config import RUNS_DIR, RUNS_FILE


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def strip_heavy_fields(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: strip_heavy_fields(item)
            for key, item in value.items()
            if key not in {"cleanImage", "resultImage", "image", "archive"}
        }
    if isinstance(value, list):
        return [strip_heavy_fields(item) for item in value]
    return value


def load_runs() -> list[dict[str, Any]]:
    if not RUNS_FILE.exists():
        return []
    try:
        data = json.loads(RUNS_FILE.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except Exception:
        return []


def save_runs(runs: list[dict[str, Any]]) -> None:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    RUNS_FILE.write_text(json.dumps(runs[-200:], indent=2), encoding="utf-8")


def persist_run(kind: str, payload: dict[str, Any]) -> dict[str, Any]:
    entry = {
        "id": f"{int(time.time() * 1000)}-{kind}",
        "createdAt": now_iso(),
        "kind": kind,
        "payload": strip_heavy_fields(payload),
    }
    runs = load_runs()
    runs.append(entry)
    save_runs(runs)
    return entry


def dataset_summary_from_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    labelled = [row for row in rows if row.get("labelAvailable")]
    completed = [row for row in rows if not row.get("error")]
    total_predictions = sum(int(row.get("predictionCount") or 0) for row in rows)
    mean_confidence = sum(float(row.get("meanConfidence") or 0) for row in rows) / len(rows) if rows else 0
    return {
        "imageCount": len(rows),
        "completedCount": len(completed),
        "failedCount": len(rows) - len(completed),
        "labelledCount": len(labelled),
        "totalGt": sum(int(row.get("gtCount") or 0) for row in labelled),
        "totalMissed": sum(int(row.get("missed") or 0) for row in labelled),
        "totalFalsePositive": sum(int(row.get("falsePositive") or 0) for row in labelled),
        "meanAp50": sum(float(row.get("ap50") or 0) for row in labelled) / len(labelled) if labelled else None,
        "totalPredictions": total_predictions,
        "meanConfidence": mean_confidence,
    }
