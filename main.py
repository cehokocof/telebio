"""TeleBio — automatic Telegram bio changer.

Entry point: can be run directly (`python main.py`)
or via the package script (`telebio`).
"""

from __future__ import annotations

import asyncio
import logging
import signal
import sys

from telebio.config import load_settings, Settings
from telebio.providers.base import BioProvider
from telebio.providers.list_provider import ListBioProvider
from telebio.providers.llm_provider import LLMBioProvider
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
) -> BioProvider:
    """Instantiate the correct bio provider based on config."""
    return _build_provider_by_mode(settings.bio_provider, settings, telegram)


def _build_provider_by_mode(
    mode: str,
    settings: Settings,
    telegram: TelegramService | None = None,
) -> BioProvider:
    """Build a provider for a specific mode."""
    match mode:
        case "list":
            return ListBioProvider(settings.phrases_path)
        case "llm":
            if not settings.yandex_api_key or not settings.yandex_folder_id:
                raise EnvironmentError(
                    "BIO_PROVIDER=llm requires YANDEX_API_KEY and "
                    "YANDEX_FOLDER_ID to be set in .env"
                )
            return LLMBioProvider(
                api_key=settings.yandex_api_key,
                folder_id=settings.yandex_folder_id,
                examples_path=settings.examples_path,
                model=settings.yandex_model,
                temperature=settings.yandex_temperature,
            )
        case "context_prod":
            from telebio.providers.context_prod_provider import (
                ContextProdBioProvider,
                ContextProdProviderConfig,
            )

            if telegram is None:
                raise EnvironmentError(
                    "BIO_PROVIDER=context_prod requires an active TelegramService."
                )
            if not settings.yandex_api_key or not settings.yandex_folder_id:
                raise EnvironmentError(
                    "BIO_PROVIDER=context_prod requires YANDEX_API_KEY and "
                    "YANDEX_FOLDER_ID to be set in .env"
                )
            return ContextProdBioProvider(
                telegram=telegram,
                config=ContextProdProviderConfig(
                    db_path=settings.context_prod_db_path,
                    model_dir=settings.context_prod_model_path,
                    stage1_model=settings.context_prod_stage1_model,
                    stage2_model=settings.context_prod_stage2_model,
                    feature_embedding_model=settings.context_prod_feature_embedding_model,
                    enable_nli_score=settings.context_prod_enable_nli_score,
                    nli_model=settings.context_prod_nli_model,
                    yandex_api_key=settings.yandex_api_key,
                    yandex_folder_id=settings.yandex_folder_id,
                    yandex_model=settings.yandex_model,
                    yandex_temperature=settings.yandex_temperature,
                    fetch_days=settings.context_prod_fetch_days,
                    min_batch=settings.context_prod_min_batch,
                    fallback_min_batch=settings.context_prod_fallback_min_batch,
                    fallback_max_age_days=settings.context_prod_fallback_max_age_days,
                    max_prompt_messages=settings.context_prod_max_prompt_messages,
                    dialog_scan_limit=settings.context_prod_dialog_scan_limit,
                    per_dialog_limit=settings.context_prod_per_dialog_limit,
                ),
            )
        case other:
            raise ValueError(
                f"Unknown BIO_PROVIDER: '{other}'. Use 'list', 'llm', or 'context_prod'."
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
    scheduler_interval = (
        settings.context_prod_poll_minutes
        if settings.bio_provider == "context_prod"
        else settings.update_interval_minutes
    )

    logger.info("Starting TeleBio (interval=%d min, provider=%s)",
                scheduler_interval, settings.bio_provider)

    async with TelegramService(
        api_id=settings.api_id,
        api_hash=settings.api_hash,
        session_path=settings.session_path,
    ) as tg:
        provider = _build_provider(settings, tg)

        # Track current mode for dynamic switching
        current_mode = {"mode": settings.bio_provider}

        # Provider factory for rebuilding on mode change
        def provider_factory(mode: str) -> BioProvider:
            return _build_provider_by_mode(mode, settings, tg)

        # Start management bot if token is provided
        bot = None
        if settings.bot_token:
            me = await tg._client.get_me()
            bot = BotService(
                bot_token=settings.bot_token,
                api_id=settings.api_id,
                api_hash=settings.api_hash,
                current_mode=current_mode,
                telegram=tg,
                provider_factory=provider_factory,
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
                current_mode=current_mode,
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

    logger.info("TeleBio stopped.")


def main() -> None:
    """Synchronous wrapper for the async entry point."""
    asyncio.run(_async_main())


if __name__ == "__main__":
    main()
