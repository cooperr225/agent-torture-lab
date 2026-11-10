"""Logging every agent run to SQLite.

Two tables:
- runs: one row per run (task, final answer, status, tokens, cost...)
- tool_calls: one row per tool call (step, tool, arguments, result)
"""

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at TEXT NOT NULL,
    model TEXT NOT NULL,
    task TEXT NOT NULL,
    status TEXT NOT NULL,          -- running, completed, step_limit, refused, truncated, error
    final_answer TEXT,
    steps INTEGER NOT NULL DEFAULT 0,
    input_tokens INTEGER NOT NULL DEFAULT 0,
    output_tokens INTEGER NOT NULL DEFAULT 0,
    cost_usd REAL                  -- NULL if the model has no price listed
);
CREATE TABLE IF NOT EXISTS tool_calls (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL REFERENCES runs(id),
    step INTEGER NOT NULL,
    tool_name TEXT NOT NULL,
    arguments TEXT NOT NULL,       -- JSON
    result TEXT NOT NULL,
    is_error INTEGER NOT NULL
);
"""


class RunLog:
    def __init__(self, db_path: Path):
        db_path = Path(db_path)
        if str(db_path) != ":memory:":
            db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)

    def start_run(self, model: str, task: str) -> int:
        cur = self.conn.execute(
            "INSERT INTO runs (started_at, model, task, status) VALUES (?, ?, ?, 'running')",
            (datetime.now(timezone.utc).isoformat(timespec="seconds"), model, task),
        )
        self.conn.commit()
        return cur.lastrowid

    def log_tool_call(self, run_id: int, step: int, name: str, arguments, result: str, is_error: bool):
        self.conn.execute(
            "INSERT INTO tool_calls (run_id, step, tool_name, arguments, result, is_error)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            (run_id, step, name, json.dumps(arguments, default=str), result, int(is_error)),
        )
        self.conn.commit()

    def finish_run(self, run_id: int, status: str, final_answer: str | None, steps: int,
                   input_tokens: int, output_tokens: int, cost_usd: float | None):
        self.conn.execute(
            "UPDATE runs SET status=?, final_answer=?, steps=?, input_tokens=?,"
            " output_tokens=?, cost_usd=? WHERE id=?",
            (status, final_answer, steps, input_tokens, output_tokens, cost_usd, run_id),
        )
        self.conn.commit()

    def get_run(self, run_id: int) -> sqlite3.Row:
        return self.conn.execute("SELECT * FROM runs WHERE id=?", (run_id,)).fetchone()

    def get_tool_calls(self, run_id: int) -> list[sqlite3.Row]:
        return self.conn.execute(
            "SELECT * FROM tool_calls WHERE run_id=? ORDER BY id", (run_id,)
        ).fetchall()

    def close(self):
        self.conn.close()
