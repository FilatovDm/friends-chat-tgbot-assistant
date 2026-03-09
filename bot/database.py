from __future__ import annotations

import sqlite3
from pathlib import Path
from threading import Lock


class ThreadDatabase:
    """Stores one OpenAI thread_id per Telegram chat_id."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _initialize(self) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS chat_threads (
                    chat_id INTEGER PRIMARY KEY,
                    thread_id TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.commit()

    def get_thread_id(self, chat_id: int) -> str | None:
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT thread_id FROM chat_threads WHERE chat_id = ?",
                (chat_id,),
            ).fetchone()
            return row["thread_id"] if row else None

    def set_thread_id(self, chat_id: int, thread_id: str) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO chat_threads (chat_id, thread_id)
                VALUES (?, ?)
                ON CONFLICT(chat_id) DO UPDATE SET
                    thread_id = excluded.thread_id,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (chat_id, thread_id),
            )
            conn.commit()
