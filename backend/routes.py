from __future__ import annotations

import json
import logging
import mimetypes
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from .config import RUNS_FILE, SETTINGS, STATIC_DIR
from .domain import SCENES
from .experiments import build_evaluation
from .images import image_from_data_url
from .runtime import runtime_status
from .storage import load_runs, persist_run
from .studies import (
    cancel_study,
    get_study_status,
    latest_study_result,
    start_study,
    study_history,
    sweep_from_latest,
)
from .transitions import transition_analysis
from .uploads import evaluate_uploaded_image, evaluate_zip_upload, inspect_zip_upload


LOGGER = logging.getLogger(__name__)
STATIC_SUFFIXES = frozenset({".html", ".js", ".css"})
POST_ROUTES = frozenset({
    "/api/upload-evaluate",
    "/api/upload-zip-evaluate",
    "/api/inspect-zip",
    "/api/save-run",
    "/api/study/start",
    "/api/study/cancel",
})
GET_API_ROUTES = frozenset({
    "/api/runs",
    "/api/study/status",
    "/api/study/latest",
    "/api/study/history",
    "/api/transitions",
    "/api/status",
    "/api/evaluate",
    "/api/compare",
    "/api/sweep",
    "/api/benchmark",
})


def resolve_static_asset(path: str) -> Path | None:
    name = "index.html" if path in {"", "/"} else path.lstrip("/")
    relative = Path(name)
    if relative.is_absolute() or ".." in relative.parts or relative.suffix not in STATIC_SUFFIXES:
        return None
    static_root = STATIC_DIR.resolve()
    target = (static_root / relative).resolve()
    try:
        target.relative_to(static_root)
    except ValueError:
        return None
    return target if target.is_file() else None


def parse_evaluation_query(query: str) -> tuple[str, str, str, int]:
    qs = parse_qs(query)
    scene_id = qs.get("scene", ["street"])[0]
    model = qs.get("model", ["transformer"])[0]
    degradation = qs.get("degradation", ["blur"])[0]
    if scene_id not in SCENES:
        scene_id = "street"
    if model not in ["transformer", "cnn"]:
        model = "transformer"
    if degradation not in ["blur", "lowlight", "jpeg"]:
        degradation = "blur"
    try:
        severity = int(qs.get("severity", ["3"])[0])
    except ValueError:
        severity = 3
    return scene_id, model, degradation, max(0, min(5, severity))


def normalise_run_options(payload: dict[str, Any]) -> tuple[str, str, int]:
    model = payload.get("model", "transformer")
    degradation = payload.get("degradation", "blur")
    severity = int(payload.get("severity", 3))
    if model not in ["transformer", "cnn"]:
        model = "transformer"
    if degradation not in ["blur", "lowlight", "jpeg"]:
        degradation = "blur"
    return model, degradation, max(0, min(5, severity))


class Handler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path not in POST_ROUTES:
            self.send_error(404)
            return
        try:
            payload = self.read_json_body()
            if parsed.path == "/api/save-run":
                return self.json({"savedRun": persist_run(str(payload.get("kind", "frontend-run")), payload.get("payload", payload))})
            if parsed.path == "/api/study/start":
                return self.json(start_study(payload), status=202)
            if parsed.path == "/api/study/cancel":
                return self.json(cancel_study(str(payload.get("id", ""))))
            model, degradation, severity = normalise_run_options(payload)
            if parsed.path == "/api/inspect-zip":
                return self.json(inspect_zip_upload(payload["archive"]))
            if parsed.path == "/api/upload-zip-evaluate":
                return self.json(evaluate_zip_upload(payload["archive"], model, degradation, severity))
            clean = image_from_data_url(payload["image"])
            result = evaluate_uploaded_image(payload.get("imageName", "uploaded-image"), clean, model, degradation, severity)
            saved_run = persist_run("uploaded-image", {"model": model, "degradation": degradation, "severity": severity, "row": result["row"]})
            return self.json({
                "model": model,
                "degradation": degradation,
                "severity": severity,
                "backend": result["backend"],
                "cleanImage": result["cleanImage"],
                "resultImage": result["resultImage"],
                "predictions": result["predictions"],
                "summary": result["summary"],
                "row": result["row"],
                "savedRun": saved_run,
                "runtime": runtime_status(),
            })
        except (KeyError, TypeError, ValueError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            return self.json({
                "error": str(exc),
                "runtime": runtime_status(),
            }, status=400)
        except Exception as exc:
            LOGGER.exception("POST %s failed", parsed.path)
            return self.json({
                "error": "The request could not be completed.",
                "detail": str(exc),
                "runtime": runtime_status(),
            }, status=500)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path.startswith("/api/") and parsed.path not in GET_API_ROUTES:
            return self.json({"error": "Unknown API endpoint"}, status=404)
        if parsed.path == "/api/runs":
            return self.json({"runs": load_runs(), "path": str(RUNS_FILE)})
        if parsed.path == "/api/study/status":
            study_id = parse_qs(parsed.query).get("id", [""])[0]
            result = get_study_status(study_id)
            return self.json(result, status=404 if result.get("status") == "not-found" else 200)
        if parsed.path == "/api/study/latest":
            result = latest_study_result()
            return self.json(result or {"error": "No completed real study is available yet."}, status=200 if result else 404)
        if parsed.path == "/api/study/history":
            return self.json({"studies": study_history()})
        if parsed.path == "/api/transitions":
            try:
                query = parse_qs(parsed.query)
                model = query.get("model", ["transformer"])[0]
                degradation = query.get("degradation", ["blur"])[0]
                return self.json(transition_analysis(model, degradation))
            except Exception as exc:
                return self.json({"error": str(exc)}, status=400)
        if parsed.path == "/api/status":
            status = runtime_status()
            status["scenes"] = [{"id": k, "name": v["name"]} for k, v in SCENES.items()]
            return self.json(status)
        if parsed.path == "/api/evaluate":
            try:
                scene_id, model, degradation, severity = parse_evaluation_query(parsed.query)
                result = build_evaluation(scene_id, model, degradation, severity)
                result["savedRun"] = persist_run("single-run", result)
                return self.json(result)
            except Exception as exc:
                return self.json({"error": str(exc), "runtime": runtime_status()}, status=500)
        if parsed.path == "/api/compare":
            try:
                scene_id, _model, degradation, severity = parse_evaluation_query(parsed.query)
                return self.json({
                    "scene": scene_id,
                    "sceneName": SCENES[scene_id]["name"],
                    "degradation": degradation,
                    "severity": severity,
                    "transformer": build_evaluation(scene_id, "transformer", degradation, severity),
                    "cnn": build_evaluation(scene_id, "cnn", degradation, severity),
                    "runtime": runtime_status(),
                })
            except Exception as exc:
                return self.json({"error": str(exc), "runtime": runtime_status()}, status=500)
        if parsed.path == "/api/sweep":
            _scene_id, _model, degradation, _severity = parse_evaluation_query(parsed.query)
            result = sweep_from_latest(degradation)
            if not result:
                return self.json({"error": "Run a real dataset study before opening robustness curves."}, status=409)
            result["runtime"] = runtime_status()
            return self.json(result)
        if parsed.path == "/api/benchmark":
            result = latest_study_result()
            if not result:
                return self.json({"error": "No completed real study is available. Start one on the Benchmark page."}, status=409)
            result["runtime"] = runtime_status()
            return self.json(result)
        return self.send_static(parsed.path)

    def send_static(self, path: str) -> None:
        target = resolve_static_asset(path)
        if target is None:
            self.send_error(404)
            return
        content_type = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
        data = target.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", f"{content_type}; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(data)

    def read_json_body(self) -> dict[str, Any]:
        raw_length = self.headers.get("Content-Length")
        if raw_length is None:
            raise ValueError("Content-Length is required")
        length = int(raw_length)
        if length <= 0:
            raise ValueError("Request body is empty")
        if length > SETTINGS.max_request_bytes:
            raise ValueError(
                f"Request body exceeds the {SETTINGS.max_request_bytes}-byte limit"
            )
        payload = json.loads(self.rfile.read(length).decode("utf-8"))
        if not isinstance(payload, dict):
            raise TypeError("JSON request body must be an object")
        return payload

    def json(self, payload: Any, status: int = 200) -> None:
        data = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, format: str, *args: Any) -> None:
        LOGGER.info("%s - %s", self.client_address[0], format % args)
