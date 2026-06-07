"""Handler for the /pause command."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from telethon import events

from telebio.services.actions import toggle_pause_text

if TYPE_CHECKING:
    from telebio.services.bot import BotService

logger = logging.getLogger(__name__)


async def handle_pause(event: events.NewMessage.Event, bot: BotService) -> None:
    """Toggle auto-update on / off."""
    await event.respond(toggle_pause_text(bot))
