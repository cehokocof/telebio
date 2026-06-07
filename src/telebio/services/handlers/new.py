"""Handler for the /new command."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from telethon import events

from telebio.services.actions import run_new

if TYPE_CHECKING:
    from telebio.services.bot import BotService

logger = logging.getLogger(__name__)


async def handle_new(event: events.NewMessage.Event, bot: BotService) -> None:
    """Immediately generate and apply a new bio."""
    await event.respond(await run_new(bot), parse_mode="html")
