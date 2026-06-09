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

    # Provider type: "list" | "llm_prompt_generation" | "telegram_context"
    bio_provider: str = "list"

    # Path to the phrases data file (relative to project root)
    phrases_file: str = "data/phrases.json"

    # Path to few-shot examples for the llm_prompt_generation provider
    examples_file: str = "data/examples.json"

    # Path to named system prompts for llm_prompt_generation
    prompts_file: str = "data/prompts.json"

    # ── YandexGPT settings (required for llm_prompt_generation/telegram_context) ──
    yandex_api_key: str = ""
    yandex_folder_id: str = ""
    yandex_model: str = "yandexgpt-lite/latest"
    yandex_temperature: float = 0.9

    # Logging level
    log_level: str = "INFO"

    # Production context collection/classification settings
    telegram_context_poll_minutes: int = 60
    telegram_context_fetch_days: int = 7
    telegram_context_min_batch: int = 20
    telegram_context_fallback_min_batch: int = 10
    telegram_context_fallback_max_age_days: int = 7
    telegram_context_max_prompt_messages: int = 20
    telegram_context_max_maybe_prompt_messages: int = 5
    # Hard cap on the assembled YandexGPT prompt size (chars). Messages are
    # dropped to fit so the request never exceeds the model's context window.
    telegram_context_max_prompt_chars: int = 6000
    telegram_context_dataset: str = "data/telegram_context.parquet"
    telegram_context_report_dir: str = "logs/context_api_reports"
    telegram_context_model_dir: str = "data/prod_models/mix0035"
    telegram_context_stage1_model: str = "cointegrated/rubert-tiny2"
    telegram_context_stage2_model: str = (
        "sentence-transformers/distiluse-base-multilingual-cased-v2"
    )
    telegram_context_feature_embedding_model: str = (
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    telegram_context_enable_nli_score: bool = False
    telegram_context_nli_model: str = "cointegrated/rubert-base-cased-nli-threeway"
    telegram_context_dialog_scan_limit: int = 500
    telegram_context_per_dialog_limit: int = 100
    telegram_context_merge_gap_seconds: int = 1800
    telegram_context_max_message_length: int = 1000

    # Resolved paths (computed after init)
    project_root: Path = field(default=_PROJECT_ROOT)

    @property
    def phrases_path(self) -> Path:
        return self.project_root / self.phrases_file

    @property
    def examples_path(self) -> Path:
        return self.project_root / self.examples_file

    @property
    def prompts_path(self) -> Path:
        return self.project_root / self.prompts_file

    @property
    def session_path(self) -> str:
        """Full path to the Telethon .session file (without extension)."""
        return str(self.project_root / self.session_name)

    @property
    def telegram_context_dataset_path(self) -> Path:
        return self.project_root / self.telegram_context_dataset

    @property
    def telegram_context_report_path(self) -> Path:
        return self.project_root / self.telegram_context_report_dir

    @property
    def telegram_context_model_path(self) -> Path:
        return self.project_root / self.telegram_context_model_dir


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
        prompts_file=_get_env("PROMPTS_FILE", default="data/prompts.json"),
        yandex_api_key=_get_env("YANDEX_API_KEY", default=""),
        yandex_folder_id=_get_env("YANDEX_FOLDER_ID", default=""),
        yandex_model=_get_env("YANDEX_MODEL", default="yandexgpt-lite/latest"),
        yandex_temperature=float(
            _get_env("YANDEX_TEMPERATURE", default="0.9")
        ),
        log_level=_get_env("LOG_LEVEL", default="INFO"),
        telegram_context_poll_minutes=int(
            _get_env("TELEGRAM_CONTEXT_POLL_MINUTES", default="60")
        ),
        telegram_context_fetch_days=int(_get_env("TELEGRAM_CONTEXT_FETCH_DAYS", default="7")),
        telegram_context_min_batch=int(_get_env("TELEGRAM_CONTEXT_MIN_BATCH", default="20")),
        telegram_context_fallback_min_batch=int(
            _get_env("TELEGRAM_CONTEXT_FALLBACK_MIN_BATCH", default="10")
        ),
        telegram_context_fallback_max_age_days=int(
            _get_env("TELEGRAM_CONTEXT_FALLBACK_MAX_AGE_DAYS", default="7")
        ),
        telegram_context_max_prompt_messages=int(
            _get_env("TELEGRAM_CONTEXT_MAX_PROMPT_MESSAGES", default="20")
        ),
        telegram_context_max_maybe_prompt_messages=int(
            _get_env("TELEGRAM_CONTEXT_MAX_MAYBE_PROMPT_MESSAGES", default="5")
        ),
        telegram_context_max_prompt_chars=int(
            _get_env("TELEGRAM_CONTEXT_MAX_PROMPT_CHARS", default="6000")
        ),
        telegram_context_dataset=_get_env(
            "TELEGRAM_CONTEXT_DATASET", default="data/telegram_context.parquet"
        ),
        telegram_context_report_dir=_get_env(
            "TELEGRAM_CONTEXT_REPORT_DIR", default="logs/context_api_reports"
        ),
        telegram_context_model_dir=_get_env(
            "TELEGRAM_CONTEXT_MODEL_DIR", default="data/prod_models/mix0035"
        ),
        telegram_context_stage1_model=_get_env(
            "TELEGRAM_CONTEXT_STAGE1_MODEL", default="cointegrated/rubert-tiny2"
        ),
        telegram_context_stage2_model=_get_env(
            "TELEGRAM_CONTEXT_STAGE2_MODEL",
            default="sentence-transformers/distiluse-base-multilingual-cased-v2",
        ),
        telegram_context_feature_embedding_model=_get_env(
            "TELEGRAM_CONTEXT_FEATURE_EMBEDDING_MODEL",
            default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        ),
        telegram_context_enable_nli_score=_get_env(
            "TELEGRAM_CONTEXT_ENABLE_NLI_SCORE", default="false"
        ).lower()
        in {"1", "true", "yes", "on"},
        telegram_context_nli_model=_get_env(
            "TELEGRAM_CONTEXT_NLI_MODEL",
            default="cointegrated/rubert-base-cased-nli-threeway",
        ),
        telegram_context_dialog_scan_limit=int(
            _get_env("TELEGRAM_CONTEXT_DIALOG_SCAN_LIMIT", default="500")
        ),
        telegram_context_per_dialog_limit=int(
            _get_env("TELEGRAM_CONTEXT_PER_DIALOG_LIMIT", default="100")
        ),
        telegram_context_merge_gap_seconds=int(
            _get_env("TELEGRAM_CONTEXT_MERGE_GAP_SECONDS", default="1800")
        ),
        telegram_context_max_message_length=int(
            _get_env("TELEGRAM_CONTEXT_MAX_MESSAGE_LENGTH", default="1000")
        ),
    )
