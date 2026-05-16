"""Utilities for exporting Telegram context messages to JSON fixtures."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from telebio.services.telegram import ContextMessage


def messages_to_fixture_rows(messages: list[ContextMessage]) -> list[dict[str, str]]:
    """Convert collected context messages to the fixture/report JSON shape."""
    return [
        {
            "date": message.date.isoformat(timespec="seconds"),
            "dialog": message.dialog,
            "text": message.text,
        }
        for message in messages
    ]


def write_context_fixture(path: Path, messages: list[ContextMessage]) -> None:
    """Write collected context messages as a UTF-8 JSON array."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = messages_to_fixture_rows(messages)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(rows, fh, ensure_ascii=False, indent=2)
        fh.write("\n")


def fixture_rows_to_messages(rows: list[dict[str, Any]]) -> list[ContextMessage]:
    """Parse fixture/report JSON rows into ContextMessage objects."""
    from datetime import datetime

    messages: list[ContextMessage] = []
    for row in rows:
        messages.append(
            ContextMessage(
                date=datetime.fromisoformat(row["date"]),
                dialog=row["dialog"],
                text=row["text"],
            )
        )
    return messages
