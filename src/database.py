"""
database.py — Persist results to SQLite + JSON.
All writes are atomic; reads return pandas DataFrames.
"""

import json
import sqlite3
import os
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import pandas as pd

from src.config import RESULTS_DB, RESULTS_JSON
from src.conditions import TaskResult


# ─────────────────────────────────────────────────────────────────────────────
# Schema
# ─────────────────────────────────────────────────────────────────────────────

CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS results (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id         INTEGER NOT NULL,
    category        TEXT NOT NULL,
    difficulty      TEXT NOT NULL,
    condition       TEXT NOT NULL,
    final_answer    TEXT,
    score           REAL NOT NULL,
    grader_method   TEXT,
    grader_detail   TEXT,
    failure_mode    TEXT DEFAULT 'None',
    tokens_total    INTEGER,
    latency_ms      REAL,
    model_responses TEXT,     -- JSON string
    model_scores    TEXT,     -- JSON string
    model_tokens    TEXT,     -- JSON string
    actual_models   TEXT,     -- JSON string
    fallbacks_used  TEXT,     -- JSON string
    round1_responses TEXT,    -- JSON string
    round2_responses TEXT,    -- JSON string
    error           TEXT,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(task_id, condition)
);
"""

CREATE_INDEX = """
CREATE INDEX IF NOT EXISTS idx_cat_cond ON results(category, condition);
"""


# ─────────────────────────────────────────────────────────────────────────────
# Init
# ─────────────────────────────────────────────────────────────────────────────

def init_db() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(RESULTS_DB), exist_ok=True)
    conn = sqlite3.connect(RESULTS_DB, timeout=30.0)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute(CREATE_TABLE)
    conn.execute(CREATE_INDEX)
    # Migration: check if failure_mode / actual_models / fallbacks_used columns exist
    cursor = conn.execute("PRAGMA table_info(results);")
    cols = [r[1] for r in cursor.fetchall()]
    if "failure_mode" not in cols:
        conn.execute("ALTER TABLE results ADD COLUMN failure_mode TEXT DEFAULT 'None';")
    if "actual_models" not in cols:
        conn.execute("ALTER TABLE results ADD COLUMN actual_models TEXT;")
    if "fallbacks_used" not in cols:
        conn.execute("ALTER TABLE results ADD COLUMN fallbacks_used TEXT;")
    conn.commit()
    return conn


def clear_db() -> None:
    """Clear all records from results table."""
    if Path(RESULTS_DB).exists():
        conn = sqlite3.connect(RESULTS_DB, timeout=30.0)
        conn.execute("DELETE FROM results;")
        conn.commit()
        conn.close()


# ─────────────────────────────────────────────────────────────────────────────
# Write
# ─────────────────────────────────────────────────────────────────────────────

def save_result(result: TaskResult, conn: Optional[sqlite3.Connection] = None) -> None:
    """Insert or replace a single TaskResult into the DB atomically."""
    close = False
    if conn is None:
        conn = sqlite3.connect(RESULTS_DB, timeout=60.0)
        conn.execute("PRAGMA journal_mode=WAL;")
        close = True

    def j(v) -> str:
        return json.dumps(v) if v else "{}"

    failure_mode = getattr(result, "failure_mode", "None") or "None"
    actual_models = getattr(result, "actual_models", {}) or {}
    fallbacks_used = getattr(result, "fallbacks_used", {}) or {}

    conn.execute("""
        INSERT OR REPLACE INTO results
        (task_id, category, difficulty, condition, final_answer, score,
         grader_method, grader_detail, failure_mode, tokens_total, latency_ms,
         model_responses, model_scores, model_tokens, actual_models, fallbacks_used,
         round1_responses, round2_responses, error)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, (
        result.task_id, result.category, result.difficulty,
        result.condition, result.final_answer, result.score,
        result.grader_method, result.grader_detail, failure_mode,
        result.tokens_total, result.latency_ms,
        j(result.model_responses), j(result.model_scores), j(result.model_tokens),
        j(actual_models), j(fallbacks_used),
        j(result.round1_responses), j(result.round2_responses),
        result.error,
    ))
    conn.commit()
    if close:
        conn.close()


def save_results_batch(results: list[TaskResult]) -> None:
    conn = init_db()
    for r in results:
        save_result(r, conn)
    conn.close()


# ─────────────────────────────────────────────────────────────────────────────
# Read
# ─────────────────────────────────────────────────────────────────────────────

def load_results() -> pd.DataFrame:
    """Load all results into a DataFrame."""
    if not Path(RESULTS_DB).exists():
        return pd.DataFrame()
    conn = sqlite3.connect(RESULTS_DB)
    df = pd.read_sql_query("SELECT * FROM results ORDER BY task_id, condition", conn)
    conn.close()
    # Parse JSON columns back to dicts
    for col in ["model_responses", "model_scores", "model_tokens", "actual_models", "fallbacks_used",
                "round1_responses", "round2_responses"]:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: json.loads(x) if x else {})
    return df


def load_results_for(category: Optional[str] = None,
                     condition: Optional[str] = None) -> pd.DataFrame:
    df = load_results()
    if df.empty:
        return df
    if category:
        df = df[df["category"] == category]
    if condition:
        df = df[df["condition"] == condition]
    return df


def get_completed_combos() -> set[tuple]:
    """Return set of (task_id, condition) already completed — for resuming."""
    if not Path(RESULTS_DB).exists():
        return set()
    conn = sqlite3.connect(RESULTS_DB)
    rows = conn.execute("""
        SELECT task_id, condition FROM results
        WHERE tokens_total > 0
          AND (error IS NULL OR error = '')
    """).fetchall()
    conn.close()
    return set(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Export
# ─────────────────────────────────────────────────────────────────────────────

def export_to_json() -> None:
    """Export full results table to JSON for external analysis."""
    df = load_results()
    if df.empty:
        print("No results to export.")
        return
    os.makedirs(os.path.dirname(RESULTS_JSON), exist_ok=True)
    records = df.to_dict(orient="records")
    with open(RESULTS_JSON, "w") as f:
        json.dump(records, f, indent=2, default=str)
    print(f"Exported {len(records)} records -> {RESULTS_JSON}")


def summary_stats() -> dict:
    """Quick in-DB aggregation for status checks."""
    if not Path(RESULTS_DB).exists():
        return {}
    conn = sqlite3.connect(RESULTS_DB)
    rows = conn.execute("""
        SELECT condition, COUNT(*) as n, AVG(score) as avg_score,
               SUM(tokens_total) as total_tokens
        FROM results GROUP BY condition
    """).fetchall()
    conn.close()
    return {r[0]: {"n": r[1], "avg_score": round(r[2], 4), "total_tokens": r[3]}
            for r in rows}
