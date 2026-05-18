"""Handler for the /pause command."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from telethon import events

if TYPE_CHECKING:
    from telebio.services.bot import BotService

logger = logging.getLogger(__name__)


async def handle_pause(event: events.NewMessage.Event, bot: BotService) -> None:
    """Toggle auto-update on / off."""
    bot.toggle_pause()
    if bot.paused:
        await event.respond(
            "⏸ Автообновление приостановлено.\n"
            "Текущее био сохранено. Отправьте /pause снова, чтобы возобновить.",
        )
        logger.info("Auto-update paused via /pause command")
    else:
        await event.respond(
            "▶️ Автообновление возобновлено.",
        )
        logger.info("Auto-update resumed via /pause command")
