"""Application configuration loaded from environment variables."""

from __future__ import annotations

import os
import logging
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Load .env from project root (two levels up from this file)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_ENV_PATH = _PROJECT_ROOT / ".env"

load_dotenv(_ENV_PATH)


def _get_env(key: str, *, default: str | None = None, required: bool = False) -> str:
    """Retrieve an environment variable with validation."""
    value = os.getenv(key, default)
    if required and not value:
        raise EnvironmentError(
            f"Required environment variable '{key}' is not set. "
            f"Check your .env file at {_ENV_PATH}"
        )
    return value  # type: ignore[return-value]


@dataclass(frozen=True, slots=True)
class Settings:
    """Immutable application settings."""

    # Telegram credentials (required)
    api_id: int
    api_hash: str

    # Bot token for management bot (optional)
    bot_token: str = ""

    # Session name for Telethon (stored in project root)
    session_name: str = "telebio"

    # Interval between bio changes in minutes
    update_interval_minutes: int = 60

    # Provider type: "list" | "llm"
    bio_provider: str = "list"

    # Path to the phrases data file (relative to project root)
    phrases_file: str = "data/phrases.json"

    # Path to few-shot examples for LLM provider (relative to project root)
    examples_file: str = "data/examples.json"

    # ── YandexGPT settings (required only when bio_provider="llm") ──
    yandex_api_key: str = ""
    yandex_folder_id: str = ""
    yandex_model: str = "yandexgpt-lite/latest"
    yandex_temperature: float = 0.9

    # Logging level
    log_level: str = "INFO"

    # Resolved paths (computed after init)
    project_root: Path = field(default=_PROJECT_ROOT)

    @property
    def phrases_path(self) -> Path:
        return self.project_root / self.phrases_file

    @property
    def examples_path(self) -> Path:
        return self.project_root / self.examples_file

    @property
    def session_path(self) -> str:
        """Full path to the Telethon .session file (without extension)."""
        return str(self.project_root / self.session_name)


def load_settings() -> Settings:
    """Build Settings from environment variables."""
    return Settings(
        api_id=int(_get_env("TELEGRAM_API_ID", required=True)),
        api_hash=_get_env("TELEGRAM_API_HASH", required=True),
        bot_token=_get_env("BOT_TOKEN", default=""),
        session_name=_get_env("SESSION_NAME", default="telebio"),
        update_interval_minutes=int(
            _get_env("UPDATE_INTERVAL_MINUTES", default="60")
        ),
        bio_provider=_get_env("BIO_PROVIDER", default="list"),
        phrases_file=_get_env("PHRASES_FILE", default="data/phrases.json"),
        examples_file=_get_env("EXAMPLES_FILE", default="data/examples.json"),
        yandex_api_key=_get_env("YANDEX_API_KEY", default=""),
        yandex_folder_id=_get_env("YANDEX_FOLDER_ID", default=""),
        yandex_model=_get_env("YANDEX_MODEL", default="yandexgpt-lite/latest"),
        yandex_temperature=float(
            _get_env("YANDEX_TEMPERATURE", default="0.9")
        ),
        log_level=_get_env("LOG_LEVEL", default="INFO"),
    )
