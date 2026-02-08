"""Periodic scheduler that drives bio updates."""

from __future__ import annotations

import asyncio
import logging

from telebio.providers.base import BioProvider
from telebio.services.telegram import TelegramService

logger = logging.getLogger(__name__)


async def run_scheduler(
    telegram: TelegramService,
    provider: BioProvider,
    interval_minutes: int,
) -> None:
    """Infinite loop: get a new bio → push it to Telegram → sleep.

    The very first update happens immediately on start.
    """
    interval_seconds = interval_minutes * 60
    logger.info("Scheduler started — interval every %d min", interval_minutes)

    while True:
        try:
            new_bio = await provider.get_bio()
            await telegram.update_bio(new_bio)
        except Exception:
            logger.exception("Unhandled error during bio update — will retry next cycle")

        await asyncio.sleep(interval_seconds)
