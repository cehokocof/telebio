"""Handler for the /new command."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from telethon import events

from telebio.providers.context_provider import ContextUnchanged

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
        _commit_successful_update(provider)
        bot.record_bio_update(new_bio, mode)
        await event.respond(
            f"✅ Био обновлено:\n<code>{new_bio}</code>",
            parse_mode="html",
        )
        logger.info("Bio updated via /new command: %s", new_bio)
    except ContextUnchanged:
        await event.respond("ℹ️ Контекст не изменился, новое био не генерирую.")
        logger.info("Context unchanged during /new command")
    except Exception:
        logger.exception("Error during /new command")
        await event.respond("❌ Ошибка при обновлении био. Проверьте логи.")


def _commit_successful_update(provider: object) -> None:
    commit = getattr(type(provider), "commit_successful_update", None)
    if commit:
        commit(provider)
