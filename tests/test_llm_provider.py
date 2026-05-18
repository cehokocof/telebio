"""Tests for LLMBioProvider (YandexGPT)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch

import httpx
import pytest
import respx

from telebio.providers.llm_provider import (
    LLMBioProvider,
    _SYSTEM_PROMPT,
    _TELEGRAM_BIO_MAX_LENGTH,
    _YANDEX_API_URL,
)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _make_yandex_response(text: str) -> dict[str, Any]:
    """Build a minimal YandexGPT-shaped response payload."""
    return {
        "result": {
            "alternatives": [
                {"message": {"role": "assistant", "text": text}}
            ],
            "usage": {"inputTextTokens": "10", "completionTokens": "5", "totalTokens": "15"},
            "modelVersion": "latest",
        }
    }


def _make_provider(
    examples_path: Path,
    api_key: str = "test-key",
    folder_id: str = "test-folder",
) -> LLMBioProvider:
    return LLMBioProvider(
        api_key=api_key,
        folder_id=folder_id,
        examples_path=examples_path,
        model="yandexgpt-lite/latest",
        temperature=0.8,
    )


# ------------------------------------------------------------------
# Construction
# ------------------------------------------------------------------

class TestLLMBioProviderInit:

    def test_loads_examples(self, examples_file: Path) -> None:
        provider = _make_provider(examples_file)
        assert len(provider._examples) == 2

    def test_missing_examples_file_yields_empty(self, missing_examples_path: Path) -> None:
        provider = _make_provider(missing_examples_path)
        assert provider._examples == []

    def test_invalid_examples_raises(self, tmp_path: Path) -> None:
        p = tmp_path / "bad.json"
        p.write_text('{"not": "array"}', encoding="utf-8")
        with pytest.raises(ValueError, match="JSON array of strings"):
            _make_provider(p)

    def test_model_uri_constructed_correctly(self, examples_file: Path) -> None:
        provider = _make_provider(examples_file, folder_id="abc123")
        assert provider._model_uri == "gpt://abc123/yandexgpt-lite/latest"


# ------------------------------------------------------------------
# _build_request_body
# ------------------------------------------------------------------

class TestBuildRequestBody:

    def test_contains_system_prompt(self, examples_file: Path) -> None:
        provider = _make_provider(examples_file)
        body = provider._build_request_body()
        assert body["messages"][0] == {"role": "system", "text": _SYSTEM_PROMPT}

    def test_few_shot_pairs(self, examples_file: Path) -> None:
        provider = _make_provider(examples_file)
        body = provider._build_request_body()
        msgs = body["messages"]
        # system + 2*(user+assistant) + final user = 1 + 4 + 1 = 6
        assert len(msgs) == 6
        assert msgs[1]["role"] == "user"
        assert msgs[2]["role"] == "assistant"
        assert msgs[-1]["role"] == "user"

    def test_no_examples_still_has_system_and_user(self, missing_examples_path: Path) -> None:
        provider = _make_provider(missing_examples_path)
        body = provider._build_request_body()
        msgs = body["messages"]
        assert len(msgs) == 2  # system + user
        assert msgs[0]["role"] == "system"
        assert msgs[1]["role"] == "user"

    def test_completion_options(self, examples_file: Path) -> None:
        provider = _make_provider(examples_file)
        body = provider._build_request_body()
        opts = body["completionOptions"]
        assert opts["stream"] is False
        assert opts["temperature"] == 0.8
        assert opts["maxTokens"] == 100


# ------------------------------------------------------------------
# _extract_text
# ------------------------------------------------------------------

class TestExtractText:

    def test_extracts_text(self) -> None:
        data = _make_yandex_response("  Кот на Луне  ")
        assert LLMBioProvider._extract_text(data) == "Кот на Луне"

    def test_truncates_long_text(self) -> None:
        long = "Б" * 100
        data = _make_yandex_response(long)
        result = LLMBioProvider._extract_text(data)
        assert len(result) == _TELEGRAM_BIO_MAX_LENGTH

    def test_raises_on_bad_structure_missing_result(self) -> None:
        with pytest.raises(RuntimeError, match="Unexpected YandexGPT response"):
            LLMBioProvider._extract_text({"bad": "data"})

    def test_raises_on_empty_alternatives(self) -> None:
        with pytest.raises(RuntimeError, match="Unexpected YandexGPT response"):
            LLMBioProvider._extract_text({"result": {"alternatives": []}})


# ------------------------------------------------------------------
# get_bio (integration with mocked HTTP)
# ------------------------------------------------------------------

class TestGetBio:

    @respx.mock
    async def test_successful_generation(self, examples_file: Path) -> None:
        provider = _make_provider(examples_file)
        respx.post(_YANDEX_API_URL).mock(
            return_value=httpx.Response(200, json=_make_yandex_response("Сгенерированное био"))
        )

        result = await provider.get_bio()
        assert result == "Сгенерированное био"

    @respx.mock
    async def test_sends_correct_headers(self, examples_file: Path) -> None:
        provider = _make_provider(examples_file, api_key="my-key", folder_id="my-folder")
        route = respx.post(_YANDEX_API_URL).mock(
            return_value=httpx.Response(200, json=_make_yandex_response("ok"))
        )

        await provider.get_bio()

        request = route.calls.last.request
        assert request.headers["Authorization"] == "Api-Key my-key"
        assert request.headers["x-folder-id"] == "my-folder"

    @respx.mock
    async def test_http_error_propagates(self, examples_file: Path) -> None:
        provider = _make_provider(examples_file)
        respx.post(_YANDEX_API_URL).mock(
            return_value=httpx.Response(500, text="Internal Server Error")
        )

        with pytest.raises(httpx.HTTPStatusError):
            await provider.get_bio()
