from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Union


class QALogger:
    """Log user query and final answer to a SQLite table for easy retrieval."""

    _CREATE_TABLE_SQL = (
        "CREATE TABLE IF NOT EXISTS qa_log ("
        "id INTEGER PRIMARY KEY AUTOINCREMENT, "
        "thread_id TEXT NOT NULL, "
        "query TEXT NOT NULL, "
        "final_answer TEXT NOT NULL, "
        "created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP"
        ")"
    )

    def __init__(self, db_path: Union[str, Path]) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, *, thread_id: str, query: str, final_answer: str) -> None:
        conn = sqlite3.connect(str(self.db_path))
        try:
            conn.execute(self._CREATE_TABLE_SQL)
            conn.execute(
                "INSERT INTO qa_log (thread_id, query, final_answer) VALUES (?, ?, ?)",
                (thread_id, query, final_answer),
            )
            conn.commit()
        finally:
            conn.close()
