"""TeleBio — automatic Telegram bio changer.

Entry point: can be run directly (`python main.py`)
or via the package script (`telebio`).
"""

from __future__ import annotations

import asyncio
import logging
import signal
import sys
from pathlib import Path

_SRC_PATH = str(Path(__file__).resolve().parent / "src")
if _SRC_PATH not in sys.path:
    sys.path.insert(0, _SRC_PATH)

from telebio.config import load_settings, Settings
from telebio.prompts import Prompt, get_prompt, load_prompts
from telebio.providers.base import BioProvider
from telebio.providers.list_provider import ListBioProvider
from telebio.providers.llm_provider import LLMBioProvider
from telebio.services.state_store import StateStore
from telebio.services.telegram import TelegramService
from telebio.services.bot import BotService
from telebio.scheduler import run_scheduler

logger = logging.getLogger("telebio")


# ------------------------------------------------------------------
# Factory
# ------------------------------------------------------------------

def _build_provider(
    settings: Settings,
    telegram: TelegramService | None = None,
    prompts: list[Prompt] | None = None,
    prompt_name: str | None = None,
) -> BioProvider:
    """Instantiate the correct bio provider based on config."""
    return _build_provider_by_mode(
        settings.bio_provider, settings, telegram, prompts, prompt_name
    )


def _build_provider_by_mode(
    mode: str,
    settings: Settings,
    telegram: TelegramService | None = None,
    prompts: list[Prompt] | None = None,
    prompt_name: str | None = None,
) -> BioProvider:
    """Build a provider for a specific mode."""
    match mode:
        case "list":
            return ListBioProvider(settings.phrases_path)
        case "llm_prompt_generation":
            if not settings.yandex_api_key or not settings.yandex_folder_id:
                raise EnvironmentError(
                    "BIO_PROVIDER=llm_prompt_generation requires YANDEX_API_KEY "
                    "and YANDEX_FOLDER_ID to be set in .env"
                )
            system_prompt = (
                get_prompt(prompts, prompt_name).system if prompts else None
            )
            return LLMBioProvider(
                api_key=settings.yandex_api_key,
                folder_id=settings.yandex_folder_id,
                examples_path=settings.examples_path,
                model=settings.yandex_model,
                temperature=settings.yandex_temperature,
                system_prompt=system_prompt,
            )
        case "telegram_context":
            from telebio.providers.telegram_context_provider import (
                TelegramContextBioProvider,
                TelegramContextProviderConfig,
            )

            if telegram is None:
                raise EnvironmentError(
                    "BIO_PROVIDER=telegram_context requires an active TelegramService."
                )
            if not settings.yandex_api_key or not settings.yandex_folder_id:
                raise EnvironmentError(
                    "BIO_PROVIDER=telegram_context requires YANDEX_API_KEY and "
                    "YANDEX_FOLDER_ID to be set in .env"
                )
            return TelegramContextBioProvider(
                telegram=telegram,
                config=TelegramContextProviderConfig(
                    dataset_path=settings.telegram_context_dataset_path,
                    report_dir=settings.telegram_context_report_path,
                    model_dir=settings.telegram_context_model_path,
                    stage1_model=settings.telegram_context_stage1_model,
                    stage2_model=settings.telegram_context_stage2_model,
                    feature_embedding_model=settings.telegram_context_feature_embedding_model,
                    enable_nli_score=settings.telegram_context_enable_nli_score,
                    nli_model=settings.telegram_context_nli_model,
                    yandex_api_key=settings.yandex_api_key,
                    yandex_folder_id=settings.yandex_folder_id,
                    yandex_model=settings.yandex_model,
                    yandex_temperature=settings.yandex_temperature,
                    fetch_days=settings.telegram_context_fetch_days,
                    min_batch=settings.telegram_context_min_batch,
                    fallback_min_batch=settings.telegram_context_fallback_min_batch,
                    fallback_max_age_days=settings.telegram_context_fallback_max_age_days,
                    max_prompt_messages=settings.telegram_context_max_prompt_messages,
                    max_maybe_prompt_messages=(
                        settings.telegram_context_max_maybe_prompt_messages
                    ),
                    max_prompt_chars=settings.telegram_context_max_prompt_chars,
                    dialog_scan_limit=settings.telegram_context_dialog_scan_limit,
                    per_dialog_limit=settings.telegram_context_per_dialog_limit,
                    merge_gap_seconds=settings.telegram_context_merge_gap_seconds,
                    max_message_length=settings.telegram_context_max_message_length,
                ),
            )
        case other:
            raise ValueError(
                f"Unknown BIO_PROVIDER: '{other}'. Use 'list', "
                "'llm_prompt_generation', or 'telegram_context'."
            )


# ------------------------------------------------------------------
# Logging
# ------------------------------------------------------------------

def _configure_logging(level: str) -> None:
    logging.basicConfig(
        level=level.upper(),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
    )


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

async def _async_main() -> None:
    settings = load_settings()
    _configure_logging(settings.log_level)

    prompts = load_prompts(settings.prompts_path)

    # Restore persisted mode/prompt before constructing providers so the
    # scheduler starts in the user's last-known configuration.
    store = StateStore(settings.state_db_path)
    persisted = store.load_settings()
    initial_mode = persisted.get("mode", settings.bio_provider)
    initial_prompt = persisted.get("prompt_name", prompts[0].name)

    scheduler_interval = (
        settings.telegram_context_poll_minutes
        if initial_mode == "telegram_context"
        else settings.update_interval_minutes
    )

    logger.info("Starting TeleBio (interval=%d min, provider=%s)",
                scheduler_interval, initial_mode)

    async with TelegramService(
        api_id=settings.api_id,
        api_hash=settings.api_hash,
        session_path=settings.session_path,
    ) as tg:
        # Shared mutable runtime state (mode + active prompt) — read by both the
        # scheduler and the management bot.
        state = {"mode": initial_mode, "prompt_name": initial_prompt}

        # Provider factory for (re)building on mode / prompt change
        def provider_factory(mode: str) -> BioProvider:
            return _build_provider_by_mode(
                mode, settings, tg, prompts, state.get("prompt_name")
            )

        provider = provider_factory(initial_mode)

        # Start management bot if token is provided
        bot = None
        if settings.bot_token:
            me = await tg._client.get_me()
            bot = BotService(
                bot_token=settings.bot_token,
                api_id=settings.api_id,
                api_hash=settings.api_hash,
                current_mode=state,
                telegram=tg,
                provider_factory=provider_factory,
                prompts=prompts,
                store=store,
            )
            await bot.start(owner_id=me.id)
            logger.info("Management bot enabled")

        # Graceful shutdown on SIGINT / SIGTERM
        loop = asyncio.get_running_loop()
        stop_event = asyncio.Event()

        def _shutdown(sig: signal.Signals) -> None:
            logger.info("Received %s — shutting down…", sig.name)
            stop_event.set()

        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, _shutdown, sig)

        scheduler_task = asyncio.create_task(
            run_scheduler(
                tg,
                provider,
                scheduler_interval,
                provider_factory=provider_factory,
                current_mode=state,
                bot=bot,
            )
        )

        # Wait until a stop signal arrives, then cancel the scheduler
        await stop_event.wait()
        scheduler_task.cancel()

        try:
            await scheduler_task
        except asyncio.CancelledError:
            pass
        
        # Stop bot if running
        if bot:
            await bot.stop()

    store.close()
    logger.info("TeleBio stopped.")


def main() -> None:
    """Synchronous wrapper for the async entry point."""
    asyncio.run(_async_main())


if __name__ == "__main__":
    main()
