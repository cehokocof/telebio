"""Handler for the /history command."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from telethon import events

if TYPE_CHECKING:
    from telebio.services.bot import BotService

logger = logging.getLogger(__name__)


async def handle_history(event: events.NewMessage.Event, bot: BotService) -> None:
    """Reply with recent bio update history."""
    if not bot.history:
        await event.respond("📜 No history available yet.")
        return

    history_lines = ["📜 <b>Recent Bio Updates:</b>", ""]
    for i, entry in enumerate(reversed(bot.history), 1):
        history_lines.append(
            f"{i}. [{entry['timestamp']}] <code>{entry['mode']}</code>\n"
            f"   {entry['bio']}"
        )

    await event.respond("\n\n".join(history_lines), parse_mode="html")
