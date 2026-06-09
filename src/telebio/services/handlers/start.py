"""Handler for the /start command — onboarding and mode selection."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from telethon import events

from telebio.services.keyboards import mode_menu
from telebio.services.texts import onboarding_text

if TYPE_CHECKING:
    from telebio.services.bot import BotService

logger = logging.getLogger(__name__)

__all__ = ["handle_start", "onboarding_text"]


async def handle_start(event: events.NewMessage.Event, bot: BotService) -> None:
    """Greet the owner and offer the mode picker."""
    current = bot.current_mode.get("mode", "")
    await event.respond(
        onboarding_text(), parse_mode="html", buttons=mode_menu(current)
    )
