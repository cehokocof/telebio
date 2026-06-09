"""Reusable bot actions shared by typed commands and inline-button callbacks.

Each function returns the HTML text to show the user; the calling handler is
responsible for delivering it (``event.respond`` / ``event.edit``).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from telebio.context_exceptions import ContextBatchNotReady
from telebio.modes import (
    MODE_LIST,
    MODE_TELEGRAM_CONTEXT,
    is_valid,
)
from telebio.services import texts

if TYPE_CHECKING:
    from telebio.services.bot import BotService

logger = logging.getLogger(__name__)


def menu_text(bot: BotService) -> str:
    """Header text for the main menu."""
    return texts.menu_text(
        mode=bot.current_mode.get("mode", "unknown"),
        bio=bot.last_bio,
        prompt_name=bot.prompt_name,
    )


def status_text(bot: BotService) -> str:
    last_update = (
        bot.last_update.strftime("%Y-%m-%d %H:%M:%S") if bot.last_update else None
    )
    return texts.status_text(
        mode=bot.current_mode.get("mode", "unknown"),
        paused=bot.paused,
        bio=bot.last_bio,
        last_update=last_update,
    )


def history_text(bot: BotService) -> str:
    return texts.history_text(bot.history)


def toggle_pause_text(bot: BotService) -> str:
    bot.toggle_pause()
    if bot.paused:
        logger.info("Auto-update paused")
        return texts.PAUSE_PAUSED
    logger.info("Auto-update resumed")
    return texts.PAUSE_RESUMED


def apply_mode(bot: BotService, mode: str) -> str:
    """Switch the active bio-provider mode."""
    mode = mode.strip().lower()
    if not is_valid(mode):
        return texts.MODE_UNKNOWN
    if mode == bot.current_mode.get("mode", ""):
        return texts.mode_already(mode)
    bot.current_mode["mode"] = mode
    logger.info("Mode switched to '%s'", mode)
    return texts.mode_switched(mode)


def apply_prompt(bot: BotService, name: str) -> str:
    """Select the active named prompt for llm_prompt_generation."""
    bot.set_prompt(name)
    logger.info("Active prompt set to '%s'", name)
    return texts.prompt_applied(name)


async def run_new(bot: BotService) -> str:
    """Generate and apply a fresh bio using the current mode."""
    if not bot.telegram or not bot.provider_factory:
        return texts.NEW_NOT_CONFIGURED

    mode = bot.current_mode.get("mode", MODE_LIST)
    try:
        provider = bot.provider_factory(mode)
        if mode == MODE_TELEGRAM_CONTEXT:
            new_bio = await provider.get_bio(force=True)
        else:
            new_bio = await provider.get_bio()
        await bot.telegram.update_bio(new_bio)
        commit = getattr(provider, "commit_successful_update", None)
        if commit:
            await commit(new_bio)
        bot.record_bio_update(new_bio, mode)
        logger.info("Bio updated via /new: %s", new_bio)
        return texts.new_success(new_bio)
    except ContextBatchNotReady as exc:
        logger.info("Bio update skipped: %s", exc)
        return texts.new_batch_not_ready(str(exc))
    except Exception:
        logger.exception("Error during /new")
        return texts.NEW_ERROR


async def run_collect(bot: BotService) -> str:
    """Collect and classify context rows into the parquet dataset."""
    if not bot.provider_factory:
        return texts.COLLECT_NOT_CONFIGURED
    try:
        provider = bot.provider_factory(MODE_TELEGRAM_CONTEXT)
        collect = getattr(provider, "collect_context", None)
        if collect is None:
            return texts.COLLECT_NOT_SUPPORTED
        stats = await collect()
        logger.info("Context collected: %s", stats)
        return texts.collect_success(stats)
    except Exception:
        logger.exception("Error during /collect")
        return texts.COLLECT_ERROR
