"""Handler for the /status command."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from telethon import events

from telebio.services.actions import status_text

if TYPE_CHECKING:
    from telebio.services.bot import BotService

logger = logging.getLogger(__name__)


async def handle_status(event: events.NewMessage.Event, bot: BotService) -> None:
    """Reply with current TeleBio status."""
    await event.respond(status_text(bot), parse_mode="html")
