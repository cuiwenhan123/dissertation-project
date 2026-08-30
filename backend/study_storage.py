from __future__ import annotations

import json
import sqlite3
from typing import Any

from .config import RUNS_DIR, STUDY_DB
from .storage import now_iso


def _connect() -> sqlite3.Connection:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(STUDY_DB, timeout=30)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS studies (
            id TEXT PRIMARY KEY,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            status TEXT NOT NULL,
            config_json TEXT NOT NULL,
            result_json TEXT,
            error TEXT
        )
        """
    )
    return connection


def save_study(
    study_id: str,
    status: str,
    config: dict[str, Any],
    result: dict[str, Any] | None = None,
    error: str | None = None,
) -> None:
    timestamp = now_iso()
    with _connect() as connection:
        connection.execute(
            """
            INSERT INTO studies (id, created_at, updated_at, status, config_json, result_json, error)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                updated_at = excluded.updated_at,
                status = excluded.status,
                config_json = excluded.config_json,
                result_json = excluded.result_json,
                error = excluded.error
            """,
            (
                study_id,
                timestamp,
                timestamp,
                status,
                json.dumps(config),
                json.dumps(result) if result is not None else None,
                error,
            ),
        )


def _decode(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "id": row["id"],
        "createdAt": row["created_at"],
        "updatedAt": row["updated_at"],
        "status": row["status"],
        "config": json.loads(row["config_json"]),
        "result": json.loads(row["result_json"]) if row["result_json"] else None,
        "error": row["error"],
    }


def latest_completed_study() -> dict[str, Any] | None:
    with _connect() as connection:
        row = connection.execute(
            "SELECT * FROM studies WHERE status = 'completed' ORDER BY updated_at DESC LIMIT 1"
        ).fetchone()
    return _decode(row) if row else None


def list_studies(limit: int = 20) -> list[dict[str, Any]]:
    with _connect() as connection:
        rows = connection.execute(
            "SELECT * FROM studies ORDER BY updated_at DESC LIMIT ?",
            (max(1, min(100, limit)),),
        ).fetchall()
    return [_decode(row) for row in rows]
