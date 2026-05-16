"""Handler for the /context command."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from telethon import events

if TYPE_CHECKING:
    from telebio.services.bot import BotService

logger = logging.getLogger(__name__)


async def handle_context(event: events.NewMessage.Event, bot: BotService) -> None:
    """Show or update context mode settings."""
    days_raw = event.pattern_match.group(1)
    limit_raw = event.pattern_match.group(2)

    if days_raw is None and limit_raw is None:
        await event.respond(
            "🧠 <b>Context settings</b>\n"
            f"Days: <code>{bot.context_days}</code>\n"
            f"Messages: <code>{bot.context_limit}</code>\n\n"
            "Use <code>/context 14 500</code> to update.",
            parse_mode="html",
        )
        return

    try:
        days = int(days_raw)
        limit = int(limit_raw)
        bot.set_context_settings(days, limit)
    except ValueError as exc:
        await event.respond(
            f"❌ Invalid context settings: {exc}",
            parse_mode="html",
        )
        return

    await event.respond(
        "✅ Context settings updated:\n"
        f"Days: <code>{days}</code>\n"
        f"Messages: <code>{limit}</code>",
        parse_mode="html",
    )
    logger.info("Context settings updated via bot: days=%d limit=%d", days, limit)
