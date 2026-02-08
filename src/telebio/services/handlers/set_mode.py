"""Handler for the /set_mode command."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from telethon import events

if TYPE_CHECKING:
    from telebio.services.bot import BotService

logger = logging.getLogger(__name__)


async def handle_set_mode(event: events.NewMessage.Event, bot: BotService) -> None:
    """Switch bio provider mode (list / llm)."""
    mode = event.pattern_match.group(1).lower()

    if mode not in ("list", "llm"):
        await event.respond(
            "❌ Invalid mode. Use <code>list</code> or <code>llm</code>.",
            parse_mode="html",
        )
        return

    current = bot.current_mode.get("mode", "")
    if mode == current:
        await event.respond(f"ℹ️ Already in <code>{mode}</code> mode.", parse_mode="html")
        return

    bot.current_mode["mode"] = mode

    await event.respond(
        f"✅ Mode switched to <code>{mode}</code>\n"
        f"Next bio update will use the new provider.",
        parse_mode="html",
    )
    logger.info("Mode switched to '%s' via bot command", mode)
