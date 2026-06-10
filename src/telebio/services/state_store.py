"""SQLite-backed persistence for bot runtime state and bio history.

Two tables:
* ``settings`` — key/value: mode, prompt_name, paused, last_bio, last_update.
* ``bio_history`` — append-only log of bio updates (id, ts, bio, mode).
"""

from __future__ import annotations

import logging
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


_SCHEMA = """
CREATE TABLE IF NOT EXISTS settings (
    key   TEXT PRIMARY KEY,
    value TEXT
);
CREATE TABLE IF NOT EXISTS bio_history (
    id   INTEGER PRIMARY KEY AUTOINCREMENT,
    ts   TEXT NOT NULL,
    bio  TEXT NOT NULL,
    mode TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_bio_history_ts ON bio_history(ts);
"""


class StateStore:
    """Thin wrapper over a single SQLite connection."""

    def __init__(self, path: Path | str) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(
            str(self._path),
            check_same_thread=False,
            isolation_level=None,
        )
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.executescript(_SCHEMA)

    # ------------------------------------------------------------------
    # Settings (key/value)
    # ------------------------------------------------------------------

    def load_settings(self) -> dict[str, str]:
        cur = self._conn.execute("SELECT key, value FROM settings")
        return {key: value for key, value in cur.fetchall() if value is not None}

    def save_setting(self, key: str, value: str | None) -> None:
        self._conn.execute(
            "INSERT INTO settings(key, value) VALUES(?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, value),
        )

    # ------------------------------------------------------------------
    # Bio history (append-only)
    # ------------------------------------------------------------------

    def append_bio(self, *, bio: str, mode: str, ts: datetime) -> None:
        self._conn.execute(
            "INSERT INTO bio_history(ts, bio, mode) VALUES(?, ?, ?)",
            (ts.strftime("%Y-%m-%d %H:%M:%S"), bio, mode),
        )

    def load_history(self) -> list[dict[str, Any]]:
        cur = self._conn.execute(
            "SELECT ts, bio, mode FROM bio_history ORDER BY id ASC"
        )
        return [
            {"timestamp": ts, "bio": bio, "mode": mode}
            for ts, bio, mode in cur.fetchall()
        ]

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        self._conn.close()
