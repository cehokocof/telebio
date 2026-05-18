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

    def test_raises_without_api_id(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(EnvironmentError, match="TELEGRAM_API_ID"):
                load_settings()
