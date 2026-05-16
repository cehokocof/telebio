"""Tests for persisted runtime state."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from telebio.state import RuntimeState, validate_context_settings


def test_load_missing_file_uses_defaults(tmp_path: Path) -> None:
    state = RuntimeState.load(
        tmp_path / "state.json",
        default_mode="context",
        default_context_days=7,
        default_context_limit=300,
    )

    assert state.mode == "context"
    assert state.context_days == 7
    assert state.context_limit == 300


def test_save_and_reload(tmp_path: Path) -> None:
    path = tmp_path / "state.json"
    state = RuntimeState.load(
        path,
        default_mode="list",
        default_context_days=14,
        default_context_limit=500,
    )

    state.set_mode("context")
    state.set_context_settings(3, 42)
    state.set_last_context_fingerprint("abc")

    reloaded = RuntimeState.load(
        path,
        default_mode="list",
        default_context_days=14,
        default_context_limit=500,
    )
    assert reloaded.mode == "context"
    assert reloaded.context_days == 3
    assert reloaded.context_limit == 42
    assert reloaded.last_context_fingerprint == "abc"


def test_invalid_json_falls_back_to_defaults(tmp_path: Path) -> None:
    path = tmp_path / "state.json"
    path.write_text("{bad json", encoding="utf-8")

    state = RuntimeState.load(
        path,
        default_mode="llm",
        default_context_days=14,
        default_context_limit=500,
    )

    assert state.mode == "llm"
    assert state.context_days == 14
    assert state.context_limit == 500


def test_invalid_values_are_ignored_on_load(tmp_path: Path) -> None:
    path = tmp_path / "state.json"
    path.write_text(
        json.dumps({"mode": "magic", "context_days": -1, "context_limit": "many"}),
        encoding="utf-8",
    )

    state = RuntimeState.load(
        path,
        default_mode="list",
        default_context_days=14,
        default_context_limit=500,
    )

    assert state.mode == "list"
    assert state.context_days == 14
    assert state.context_limit == 500


@pytest.mark.parametrize(
    ("days", "limit"),
    [(0, 10), (1, 0), (91, 10), (1, 2001)],
)
def test_context_settings_validation(days: int, limit: int) -> None:
    with pytest.raises(ValueError):
        validate_context_settings(days, limit)
