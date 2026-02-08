"""Shared fixtures for the telebio test suite."""

from __future__ import annotations

import json
from pathlib import Path
from typing import AsyncIterator
from unittest.mock import AsyncMock, MagicMock

import pytest


# ------------------------------------------------------------------
# Temporary phrase / example files
# ------------------------------------------------------------------

@pytest.fixture()
def phrases_file(tmp_path: Path) -> Path:
    """Create a temporary phrases.json with a few entries."""
    phrases = ["Фраза раз", "Фраза два", "Фраза три"]
    p = tmp_path / "phrases.json"
    p.write_text(json.dumps(phrases, ensure_ascii=False), encoding="utf-8")
    return p


@pytest.fixture()
def single_phrase_file(tmp_path: Path) -> Path:
    """phrases.json with exactly one entry."""
    p = tmp_path / "phrases.json"
    p.write_text(json.dumps(["Одна единственная фраза"]), encoding="utf-8")
    return p


@pytest.fixture()
def empty_list_file(tmp_path: Path) -> Path:
    """phrases.json with an empty array."""
    p = tmp_path / "phrases.json"
    p.write_text("[]", encoding="utf-8")
    return p


@pytest.fixture()
def long_phrases_file(tmp_path: Path) -> Path:
    """phrases.json where one phrase exceeds 70 chars."""
    phrases = ["Короткая", "А" * 100]
    p = tmp_path / "phrases.json"
    p.write_text(json.dumps(phrases, ensure_ascii=False), encoding="utf-8")
    return p


@pytest.fixture()
def invalid_json_file(tmp_path: Path) -> Path:
    """A file with invalid JSON content (not an array of strings)."""
    p = tmp_path / "phrases.json"
    p.write_text('{"key": "value"}', encoding="utf-8")
    return p


@pytest.fixture()
def examples_file(tmp_path: Path) -> Path:
    """Create a temporary examples.json for LLM few-shot."""
    examples = ["Борщ — это UI-фреймворк", "Кот одобрил мой коммит"]
    p = tmp_path / "examples.json"
    p.write_text(json.dumps(examples, ensure_ascii=False), encoding="utf-8")
    return p


@pytest.fixture()
def missing_examples_path(tmp_path: Path) -> Path:
    """Path to a non-existent examples file."""
    return tmp_path / "no_such_file.json"


# ------------------------------------------------------------------
# Mock Telegram service
# ------------------------------------------------------------------

@pytest.fixture()
def mock_telegram() -> AsyncMock:
    """A mock TelegramService with an async update_bio method."""
    tg = AsyncMock()
    tg.update_bio = AsyncMock()
    return tg
