"""Handler for the /menu command — shows the main inline menu."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from telethon import events

from telebio.services.actions import menu_text
from telebio.services.keyboards import main_menu

if TYPE_CHECKING:
    from telebio.services.bot import BotService

logger = logging.getLogger(__name__)


async def handle_menu(event: events.NewMessage.Event, bot: BotService) -> None:
    """Show the main menu."""
    await event.respond(menu_text(bot), parse_mode="html", buttons=main_menu(bot))
