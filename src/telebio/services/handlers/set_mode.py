"""Handler for the /set_mode command."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from telethon import events

from telebio.services.actions import apply_mode

if TYPE_CHECKING:
    from telebio.services.bot import BotService

logger = logging.getLogger(__name__)


async def handle_set_mode(event: events.NewMessage.Event, bot: BotService) -> None:
    """Switch bio provider mode (list / llm_prompt_generation / telegram_context)."""
    mode = event.pattern_match.group(1)
    await event.respond(apply_mode(bot, mode), parse_mode="html")
