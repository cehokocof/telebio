"""Handler for the /history command."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from telethon import events

from telebio.services.actions import history_text

if TYPE_CHECKING:
    from telebio.services.bot import BotService

logger = logging.getLogger(__name__)


async def handle_history(event: events.NewMessage.Event, bot: BotService) -> None:
    """Reply with recent bio update history."""
    await event.respond(history_text(bot), parse_mode="html")
