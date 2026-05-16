"""Tests for configuration loading."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from telebio.config import Settings, _get_env, load_settings


# ------------------------------------------------------------------
# _get_env
# ------------------------------------------------------------------

class TestGetEnv:

    def test_returns_env_value(self) -> None:
        with patch.dict(os.environ, {"MY_VAR": "hello"}):
            assert _get_env("MY_VAR") == "hello"

    def test_returns_default_when_missing(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            assert _get_env("MISSING_VAR", default="fallback") == "fallback"

    def test_required_raises_when_missing(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(EnvironmentError, match="IMPORTANT_KEY"):
                _get_env("IMPORTANT_KEY", required=True)

    def test_required_does_not_raise_when_present(self) -> None:
        with patch.dict(os.environ, {"IMPORTANT_KEY": "val"}):
            assert _get_env("IMPORTANT_KEY", required=True) == "val"


# ------------------------------------------------------------------
# Settings dataclass
# ------------------------------------------------------------------

class TestSettings:

    def test_phrases_path(self, tmp_path: Path) -> None:
        s = Settings(api_id=1, api_hash="h", project_root=tmp_path, phrases_file="data/p.json")
        assert s.phrases_path == tmp_path / "data/p.json"

    def test_examples_path(self, tmp_path: Path) -> None:
        s = Settings(api_id=1, api_hash="h", project_root=tmp_path, examples_file="data/e.json")
        assert s.examples_path == tmp_path / "data/e.json"

    def test_session_path(self, tmp_path: Path) -> None:
        s = Settings(api_id=1, api_hash="h", project_root=tmp_path, session_name="mysession")
        assert s.session_path == str(tmp_path / "mysession")

    def test_state_path(self, tmp_path: Path) -> None:
        s = Settings(api_id=1, api_hash="h", project_root=tmp_path, state_file="state.json")
        assert s.state_path == tmp_path / "state.json"

    def test_frozen(self) -> None:
        s = Settings(api_id=1, api_hash="h")
        with pytest.raises(AttributeError):
            s.api_id = 2  # type: ignore[misc]

    def test_default_values(self) -> None:
        s = Settings(api_id=1, api_hash="h")
        assert s.update_interval_minutes == 60
        assert s.bio_provider == "list"
        assert s.log_level == "INFO"
        assert s.yandex_temperature == 0.9
        assert s.context_days == 14
        assert s.context_limit == 500
        assert s.context_dialog_scan_limit == 10
        assert s.context_per_dialog_limit == 50
        assert s.context_top_k == 15
        assert s.context_min_score == 0.55
        assert s.context_excluded_dialogs == "telebio"
        assert s.context_enable_nli is True
        assert s.context_semantic_scorer == "nli"


# ------------------------------------------------------------------
# load_settings
# ------------------------------------------------------------------

class TestLoadSettings:

    def test_loads_from_env(self) -> None:
        env = {
            "TELEGRAM_API_ID": "999",
            "TELEGRAM_API_HASH": "abc123",
            "BIO_PROVIDER": "llm",
            "UPDATE_INTERVAL_MINUTES": "30",
            "YANDEX_API_KEY": "key",
            "YANDEX_FOLDER_ID": "folder",
            "YANDEX_TEMPERATURE": "0.5",
            "CONTEXT_DAYS": "7",
            "CONTEXT_LIMIT": "300",
            "CONTEXT_DIALOG_SCAN_LIMIT": "20",
            "CONTEXT_PER_DIALOG_LIMIT": "80",
            "CONTEXT_TOP_K": "9",
            "CONTEXT_MIN_SCORE": "0.7",
            "CONTEXT_EXCLUDED_DIALOGS": "telebio,Saved Messages",
            "CONTEXT_ENABLE_NLI": "false",
            "CONTEXT_SEMANTIC_SCORER": "embedding",
            "CONTEXT_NLI_MODEL": "test-nli",
            "CONTEXT_EMBEDDING_MODEL": "test-embedding",
            "STATE_FILE": "state.json",
        }
        with patch.dict(os.environ, env, clear=False):
            s = load_settings()

        assert s.api_id == 999
        assert s.api_hash == "abc123"
        assert s.bio_provider == "llm"
        assert s.update_interval_minutes == 30
        assert s.yandex_api_key == "key"
        assert s.yandex_folder_id == "folder"
        assert s.yandex_temperature == 0.5
        assert s.context_days == 7
        assert s.context_limit == 300
        assert s.context_dialog_scan_limit == 20
        assert s.context_per_dialog_limit == 80
        assert s.context_top_k == 9
        assert s.context_min_score == 0.7
        assert s.context_excluded_dialogs == "telebio,Saved Messages"
        assert s.context_enable_nli is False
        assert s.context_semantic_scorer == "embedding"
        assert s.context_nli_model == "test-nli"
        assert s.context_embedding_model == "test-embedding"
        assert s.state_file == "state.json"

    def test_raises_without_api_id(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(EnvironmentError, match="TELEGRAM_API_ID"):
                load_settings()
