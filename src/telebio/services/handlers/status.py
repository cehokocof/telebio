"""Handler for the /status command."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from telethon import events

if TYPE_CHECKING:
    from telebio.services.bot import BotService

logger = logging.getLogger(__name__)


async def handle_status(event: events.NewMessage.Event, bot: BotService) -> None:
    """Reply with current TeleBio status."""
    mode = bot.current_mode.get("mode", "unknown")
    paused_label = "⏸ приостановлено" if bot.paused else "▶️ активно"
    status_lines = [
        "🤖 <b>TeleBio Status</b>",
        "",
        f"📊 <b>Mode:</b> <code>{mode}</code>",
        f"⏯ <b>State:</b> {paused_label}",
        f"📝 <b>Current bio:</b> {bot.last_bio or '(none)'}",
    ]

    if bot.last_update:
        status_lines.append(
            f"🕐 <b>Last update:</b> {bot.last_update.strftime('%Y-%m-%d %H:%M:%S')}"
        )

    await event.respond("\n".join(status_lines), parse_mode="html")
