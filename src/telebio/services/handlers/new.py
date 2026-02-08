"""Handler for the /new command."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from telethon import events

if TYPE_CHECKING:
    from telebio.services.bot import BotService

logger = logging.getLogger(__name__)


async def handle_new(event: events.NewMessage.Event, bot: BotService) -> None:
    """Immediately generate and apply a new bio."""
    if not bot.telegram or not bot.provider_factory:
        await event.respond("❌ Bot не настроен для обновления био.")
        return

    mode = bot.current_mode.get("mode", "list")
    try:
        provider = bot.provider_factory(mode)
        new_bio = await provider.get_bio()
        await bot.telegram.update_bio(new_bio)
        bot.record_bio_update(new_bio, mode)
        await event.respond(
            f"✅ Био обновлено:\n<code>{new_bio}</code>",
            parse_mode="html",
        )
        logger.info("Bio updated via /new command: %s", new_bio)
    except Exception:
        logger.exception("Error during /new command")
        await event.respond("❌ Ошибка при обновлении био. Проверьте логи.")
