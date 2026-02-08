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
from telebio.scheduler import run_scheduler

logger = logging.getLogger("telebio")


# ------------------------------------------------------------------
# Factory
# ------------------------------------------------------------------

def _build_provider(settings: Settings) -> BioProvider:
    """Instantiate the correct bio provider based on config."""
    match settings.bio_provider:
        case "list":
            return ListBioProvider(settings.phrases_path)
        case "llm":
            return LLMBioProvider()
        case other:
            raise ValueError(f"Unknown BIO_PROVIDER: '{other}'. Use 'list' or 'llm'.")


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

    provider = _build_provider(settings)

    async with TelegramService(
        api_id=settings.api_id,
        api_hash=settings.api_hash,
        session_path=settings.session_path,
    ) as tg:
        # Graceful shutdown on SIGINT / SIGTERM
        loop = asyncio.get_running_loop()
        stop_event = asyncio.Event()

        def _shutdown(sig: signal.Signals) -> None:
            logger.info("Received %s — shutting down…", sig.name)
            stop_event.set()

        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, _shutdown, sig)

        scheduler_task = asyncio.create_task(
            run_scheduler(tg, provider, settings.update_interval_minutes)
        )

        # Wait until a stop signal arrives, then cancel the scheduler
        await stop_event.wait()
        scheduler_task.cancel()

        try:
            await scheduler_task
        except asyncio.CancelledError:
            pass

    logger.info("TeleBio stopped.")


def main() -> None:
    """Synchronous wrapper for the async entry point."""
    asyncio.run(_async_main())


if __name__ == "__main__":
    main()
