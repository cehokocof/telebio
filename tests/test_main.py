"""Tests for the provider factory in main.py."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from main import _build_provider
from telebio.config import Settings
from telebio.providers.context_provider import ContextBioProvider
from telebio.providers.list_provider import ListBioProvider
from telebio.providers.llm_provider import LLMBioProvider
from telebio.state import RuntimeState


@pytest.fixture()
def base_settings(tmp_path: Path) -> Settings:
    """Settings pointing at real temp files for the list provider."""
    phrases = tmp_path / "phrases.json"
    phrases.write_text('["Test phrase"]', encoding="utf-8")
    examples = tmp_path / "examples.json"
    examples.write_text('["Example"]', encoding="utf-8")
    return Settings(
        api_id=1,
        api_hash="h",
        project_root=tmp_path,
        phrases_file="phrases.json",
        examples_file="examples.json",
        yandex_api_key="key",
        yandex_folder_id="folder",
    )


class TestBuildProvider:

    def test_list_provider(self, base_settings: Settings) -> None:
        provider = _build_provider(base_settings)
        assert isinstance(provider, ListBioProvider)

    def test_llm_provider(self, base_settings: Settings) -> None:
        from dataclasses import replace
        s = replace(base_settings, bio_provider="llm")
        provider = _build_provider(s)
        assert isinstance(provider, LLMBioProvider)

    def test_llm_without_api_key_raises(self, base_settings: Settings) -> None:
        from dataclasses import replace
        s = replace(base_settings, bio_provider="llm", yandex_api_key="")
        with pytest.raises(EnvironmentError, match="YANDEX_API_KEY"):
            _build_provider(s)

    def test_llm_without_folder_id_raises(self, base_settings: Settings) -> None:
        from dataclasses import replace
        s = replace(base_settings, bio_provider="llm", yandex_folder_id="")
        with pytest.raises(EnvironmentError, match="YANDEX_FOLDER_ID"):
            _build_provider(s)

    def test_unknown_provider_raises(self, base_settings: Settings) -> None:
        from dataclasses import replace
        s = replace(base_settings, bio_provider="magic")
        with pytest.raises(ValueError, match="Unknown BIO_PROVIDER"):
            _build_provider(s)

    def test_context_provider(self, base_settings: Settings, tmp_path: Path) -> None:
        from dataclasses import replace
        from main import _build_provider_by_mode

        s = replace(base_settings, bio_provider="context")
        state = RuntimeState.load(
            tmp_path / "state.json",
            default_mode="context",
            default_context_days=3,
            default_context_limit=42,
        )
        provider = _build_provider_by_mode("context", s, telegram=object(), runtime_state=state)
        assert isinstance(provider, ContextBioProvider)

    def test_context_without_telegram_raises(self, base_settings: Settings) -> None:
        from dataclasses import replace
        from main import _build_provider_by_mode

        s = replace(base_settings, bio_provider="context")
        with pytest.raises(ValueError, match="TelegramService"):
            _build_provider_by_mode("context", s)
