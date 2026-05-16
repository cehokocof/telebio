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
from telebio.context_relevance import RelevanceOptions
from telebio.providers.base import BioProvider
from telebio.providers.context_provider import ContextBioProvider
from telebio.providers.list_provider import ListBioProvider
from telebio.providers.llm_provider import LLMBioProvider
from telebio.services.telegram import TelegramService
from telebio.services.bot import BotService
from telebio.scheduler import run_scheduler
from telebio.state import RuntimeState

logger = logging.getLogger("telebio")


# ------------------------------------------------------------------
# Factory
# ------------------------------------------------------------------

def _build_provider(settings: Settings) -> BioProvider:
    """Instantiate the correct bio provider based on config."""
    return _build_provider_by_mode(settings.bio_provider, settings)


def _build_provider_by_mode(
    mode: str,
    settings: Settings,
    telegram: TelegramService | None = None,
    runtime_state: RuntimeState | None = None,
) -> BioProvider:
    """Build a provider for a specific mode."""
    match mode:
        case "list":
            return ListBioProvider(settings.phrases_path)
        case "llm":
            _require_yandex_settings(mode, settings)
            return LLMBioProvider(
                api_key=settings.yandex_api_key,
                folder_id=settings.yandex_folder_id,
                examples_path=settings.examples_path,
                model=settings.yandex_model,
                temperature=settings.yandex_temperature,
            )
        case "context":
            _require_yandex_settings(mode, settings)
            if telegram is None:
                raise ValueError("BIO_PROVIDER=context requires TelegramService")
            days = runtime_state.context_days if runtime_state else settings.context_days
            limit = runtime_state.context_limit if runtime_state else settings.context_limit
            return ContextBioProvider(
                telegram=telegram,
                api_key=settings.yandex_api_key,
                folder_id=settings.yandex_folder_id,
                days=days,
                limit=limit,
                dialog_scan_limit=settings.context_dialog_scan_limit,
                per_dialog_limit=settings.context_per_dialog_limit,
                relevance_options=_build_relevance_options(settings),
                runtime_state=runtime_state,
                model=settings.yandex_model,
                temperature=settings.yandex_temperature,
            )
        case other:
            raise ValueError(
                f"Unknown BIO_PROVIDER: '{other}'. Use 'list', 'llm' or 'context'."
            )


def _require_yandex_settings(mode: str, settings: Settings) -> None:
    if not settings.yandex_api_key or not settings.yandex_folder_id:
        raise EnvironmentError(
            f"BIO_PROVIDER={mode} requires YANDEX_API_KEY and "
            "YANDEX_FOLDER_ID to be set in .env"
        )


def _build_relevance_options(settings: Settings) -> RelevanceOptions:
    excluded_dialogs = tuple(
        dialog.strip()
        for dialog in settings.context_excluded_dialogs.split(",")
        if dialog.strip()
    )
    return RelevanceOptions(
        top_k=settings.context_top_k,
        min_score=settings.context_min_score,
        excluded_dialogs=excluded_dialogs,
        enable_nli=settings.context_enable_nli,
        semantic_scorer=settings.context_semantic_scorer,
        nli_model=settings.context_nli_model,
        embedding_model=settings.context_embedding_model,
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

    logger.info("Starting TeleBio (interval=%d min, provider=%s)",
                settings.update_interval_minutes, settings.bio_provider)

    runtime_state = RuntimeState.load(
        settings.state_path,
        default_mode=settings.bio_provider,
        default_context_days=settings.context_days,
        default_context_limit=settings.context_limit,
    )

    # Track current mode for dynamic switching
    current_mode = {"mode": runtime_state.mode}

    async with TelegramService(
        api_id=settings.api_id,
        api_hash=settings.api_hash,
        session_path=settings.session_path,
    ) as tg:
        # Provider factory for rebuilding on mode change
        def provider_factory(mode: str) -> BioProvider:
            return _build_provider_by_mode(mode, settings, tg, runtime_state)

        provider = provider_factory(current_mode["mode"])

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
                runtime_state=runtime_state,
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
                settings.update_interval_minutes,
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
