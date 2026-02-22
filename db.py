"""SQLite persistence for brand generation runs."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

DB_PATH = Path(__file__).parent / "runs.db"


def _conn() -> sqlite3.Connection:
    con = sqlite3.connect(str(DB_PATH))
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA journal_mode=WAL")
    return con


def init_db() -> None:
    with _conn() as con:
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS runs (
                id          TEXT PRIMARY KEY,
                created_at  DATETIME DEFAULT (datetime('now')),
                concept     TEXT NOT NULL,
                tone        TEXT NOT NULL,
                seed        INTEGER,
                status      TEXT DEFAULT 'pending',
                error_msg   TEXT,
                brand_data  TEXT,   -- JSON
                images      TEXT,   -- JSON  {stage: local_path}
                prompts     TEXT,   -- JSON  {stage: prompt}
                api_calls   TEXT,   -- JSON  list
                settings    TEXT,   -- JSON
                cost_data   TEXT,   -- JSON  {openai_cost, anthropic_cost, replicate_cost, total, items[]}
                duration    REAL
            )
            """
        )
        # Add cost_data column to existing DBs (safe no-op if already present)
        try:
            con.execute("ALTER TABLE runs ADD COLUMN cost_data TEXT")
        except Exception:
            pass


# ---------------------------------------------------------------------------
# CRUD helpers
# ---------------------------------------------------------------------------

def create_run(
    run_id: str,
    concept: str,
    tone: str,
    seed: Optional[int],
    settings: Dict,
) -> None:
    with _conn() as con:
        con.execute(
            "INSERT INTO runs (id, concept, tone, seed, settings, status) "
            "VALUES (?, ?, ?, ?, ?, 'pending')",
            (run_id, concept, tone, seed, json.dumps(settings)),
        )


def set_run_running(run_id: str) -> None:
    with _conn() as con:
        con.execute(
            "UPDATE runs SET status='running' WHERE id=?", (run_id,)
        )


def complete_run(
    run_id: str,
    brand_data: Dict,
    images: Dict,
    prompts: Dict,
    api_calls: List,
    duration: float,
    cost_data: Optional[Dict] = None,
) -> None:
    with _conn() as con:
        con.execute(
            """
            UPDATE runs SET
                status     = 'complete',
                brand_data = ?,
                images     = ?,
                prompts    = ?,
                api_calls  = ?,
                cost_data  = ?,
                duration   = ?
            WHERE id = ?
            """,
            (
                json.dumps(brand_data),
                json.dumps(images),
                json.dumps(prompts),
                json.dumps(api_calls),
                json.dumps(cost_data) if cost_data else None,
                duration,
                run_id,
            ),
        )


def fail_run(run_id: str, error_msg: str) -> None:
    with _conn() as con:
        con.execute(
            "UPDATE runs SET status='failed', error_msg=? WHERE id=?",
            (error_msg, run_id),
        )


def update_run_images(run_id: str, images: Dict) -> None:
    with _conn() as con:
        con.execute(
            "UPDATE runs SET images=? WHERE id=?",
            (json.dumps(images), run_id),
        )


def get_run(run_id: str) -> Optional[Dict]:
    with _conn() as con:
        row = con.execute("SELECT * FROM runs WHERE id=?", (run_id,)).fetchone()
    if not row:
        return None
    return _deserialise(dict(row))


def list_runs(limit: int = 50) -> List[Dict]:
    with _conn() as con:
        rows = con.execute(
            "SELECT * FROM runs ORDER BY created_at DESC LIMIT ?", (limit,)
        ).fetchall()
    return [_deserialise(dict(r)) for r in rows]


def _deserialise(row: Dict) -> Dict:
    for key in ("brand_data", "images", "prompts", "api_calls", "settings", "cost_data"):
        val = row.get(key)
        if val:
            try:
                row[key] = json.loads(val)
            except (json.JSONDecodeError, TypeError):
                row[key] = {}
    return row
