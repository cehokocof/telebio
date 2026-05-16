"""Tests for ContextBioProvider."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock

import httpx
import pytest
import respx

from telebio.context_relevance import RelevanceOptions
from telebio.providers.context_provider import (
    ContextBioProvider,
    ContextUnchanged,
    _YANDEX_API_URL,
)
from telebio.services.telegram import ContextMessage


def _make_yandex_response(text: str) -> dict[str, Any]:
    return {
        "result": {
            "alternatives": [
                {"message": {"role": "assistant", "text": text}}
            ],
        }
    }


def _make_provider(telegram: AsyncMock) -> ContextBioProvider:
    return ContextBioProvider(
        telegram=telegram,
        api_key="test-key",
        folder_id="test-folder",
        days=7,
        limit=300,
        relevance_options=RelevanceOptions(enable_nli=False),
        model="yandexgpt-lite/latest",
        temperature=0.5,
    )


def _context_message(text: str, dialog: str = "Dima") -> ContextMessage:
    return ContextMessage(
        date=datetime(2026, 5, 15, 12, 0, tzinfo=timezone.utc),
        dialog=dialog,
        text=text,
    )


def test_build_request_body_contains_messages() -> None:
    telegram = AsyncMock()
    provider = _make_provider(telegram)

    body = provider._build_request_body([
        _context_message("пишу тесты", "Chat A"),
        _context_message("доделываю био", "Chat B"),
    ])

    assert body["modelUri"] == "gpt://test-folder/yandexgpt-lite/latest"
    assert body["completionOptions"]["temperature"] == 0.5
    prompt = body["messages"][1]["text"]
    assert "Chat A" in prompt
    assert "Chat B" in prompt
    assert "пишу тесты" in prompt
    assert "доделываю био" in prompt


def test_extract_text_truncates_long_result() -> None:
    result = ContextBioProvider._extract_text(_make_yandex_response("А" * 100))
    assert len(result) == 70


def test_extract_text_raises_on_bad_response() -> None:
    with pytest.raises(RuntimeError, match="Unexpected YandexGPT response"):
        ContextBioProvider._extract_text({"bad": "data"})


@respx.mock
async def test_get_bio_collects_messages_and_calls_yandex() -> None:
    telegram = AsyncMock()
    telegram.collect_recent_outgoing_texts.return_value = [
        _context_message("учу матан"),
        _context_message("пишу код"),
    ]
    provider = _make_provider(telegram)
    route = respx.post(_YANDEX_API_URL).mock(
        return_value=httpx.Response(200, json=_make_yandex_response("живу между матаном и кодом"))
    )

    result = await provider.get_bio()

    assert result == "живу между матаном и кодом"
    telegram.collect_recent_outgoing_texts.assert_awaited_once_with(
        days=7,
        limit=300,
        dialog_scan_limit=None,
        per_dialog_limit=None,
    )
    assert route.called


async def test_get_bio_raises_without_messages() -> None:
    telegram = AsyncMock()
    telegram.collect_recent_outgoing_texts.return_value = []
    provider = _make_provider(telegram)

    with pytest.raises(RuntimeError, match="No outgoing text messages"):
        await provider.get_bio()


@respx.mock
async def test_get_bio_raises_when_context_unchanged(tmp_path) -> None:
    from telebio.state import RuntimeState
    from telebio.context_relevance import fingerprint_context

    telegram = AsyncMock()
    messages = [_context_message("пишу код")]
    telegram.collect_recent_outgoing_texts.return_value = messages
    state = RuntimeState.load(
        tmp_path / "state.json",
        default_mode="context",
        default_context_days=7,
        default_context_limit=300,
    )
    state.set_last_context_fingerprint(fingerprint_context(messages))
    provider = ContextBioProvider(
        telegram=telegram,
        api_key="test-key",
        folder_id="test-folder",
        days=7,
        limit=300,
        relevance_options=RelevanceOptions(enable_nli=False),
        runtime_state=state,
    )

    with pytest.raises(ContextUnchanged):
        await provider.get_bio()


@respx.mock
async def test_commit_successful_update_persists_fingerprint(tmp_path) -> None:
    from telebio.state import RuntimeState

    telegram = AsyncMock()
    telegram.collect_recent_outgoing_texts.return_value = [_context_message("пишу код")]
    state = RuntimeState.load(
        tmp_path / "state.json",
        default_mode="context",
        default_context_days=7,
        default_context_limit=300,
    )
    provider = ContextBioProvider(
        telegram=telegram,
        api_key="test-key",
        folder_id="test-folder",
        days=7,
        limit=300,
        relevance_options=RelevanceOptions(enable_nli=False),
        runtime_state=state,
    )
    respx.post(_YANDEX_API_URL).mock(
        return_value=httpx.Response(200, json=_make_yandex_response("пишу код"))
    )

    await provider.get_bio()
    assert state.last_context_fingerprint == ""

    provider.commit_successful_update()
    assert state.last_context_fingerprint
