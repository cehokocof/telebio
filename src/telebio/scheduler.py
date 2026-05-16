"""Periodic scheduler that drives bio updates."""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from telebio.providers.base import BioProvider
from telebio.providers.context_provider import ContextUnchanged
from telebio.services.telegram import TelegramService

if TYPE_CHECKING:
    from telebio.services.bot import BotService

logger = logging.getLogger(__name__)


async def run_scheduler(
    telegram: TelegramService,
    provider: BioProvider,
    interval_minutes: int,
    provider_factory: callable | None = None,
    current_mode: dict[str, str] | None = None,
    bot: BotService | None = None,
) -> None:
    """Infinite loop: get a new bio → push it to Telegram → sleep.

    The very first update happens immediately on start.
    
    Args:
        telegram: Telegram service for updating bio
        provider: Initial bio provider
        interval_minutes: Update interval in minutes
        provider_factory: Optional factory to rebuild provider when mode changes
        current_mode: Optional dict to track and detect mode changes
        bot: Optional bot service to record updates
    """
    interval_seconds = interval_minutes * 60
    logger.info("Scheduler started — interval every %d min", interval_minutes)

    active_provider = provider
    last_mode = current_mode.get("mode") if current_mode else None
    last_context_settings = (
        (bot.context_days, bot.context_limit)
        if bot and last_mode == "context"
        else None
    )

    while True:
        try:
            # Skip update if paused
            if bot and bot.paused:
                await asyncio.sleep(interval_seconds)
                continue

            # Check if mode changed and rebuild provider if needed
            if current_mode and provider_factory:
                new_mode = current_mode.get("mode")
                context_settings = (
                    (bot.context_days, bot.context_limit)
                    if bot and new_mode == "context"
                    else None
                )
                if (
                    new_mode
                    and (
                        new_mode != last_mode
                        or context_settings != last_context_settings
                    )
                ):
                    logger.info("Mode changed from '%s' to '%s', rebuilding provider", last_mode, new_mode)
                    active_provider = provider_factory(new_mode)
                    last_mode = new_mode
                    last_context_settings = context_settings
            
            new_bio = await active_provider.get_bio()
            await telegram.update_bio(new_bio)
            _commit_successful_update(active_provider)
            
            # Record in bot history if bot is available
            if bot and current_mode:
                bot.record_bio_update(new_bio, current_mode.get("mode", "unknown"))
                
        except ContextUnchanged as exc:
            logger.info("Context unchanged — skipping bio update: %s", exc)
        except Exception:
            logger.exception("Unhandled error during bio update — will retry next cycle")

        await asyncio.sleep(interval_seconds)


def _commit_successful_update(provider: BioProvider) -> None:
    commit = getattr(type(provider), "commit_successful_update", None)
    if commit:
        commit(provider)
