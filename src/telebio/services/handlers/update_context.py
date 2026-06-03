"""Handler for updating bio from already collected context."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from telethon import events

from telebio.context_exceptions import ContextBatchNotReady

if TYPE_CHECKING:
    from telebio.services.bot import BotService

logger = logging.getLogger(__name__)


async def handle_update_context(event: events.NewMessage.Event, bot: BotService) -> None:
    """Generate and apply a bio from the existing parquet context dataset."""
    if not bot.telegram or not bot.provider_factory:
        await event.respond("❌ Bot не настроен для обновления био.")
        return

    try:
        provider = bot.provider_factory("context_prod")
        await event.respond("⏳ Обновляю bio из уже собранного context dataset…")
        new_bio = await provider.get_bio(force=True)
        await bot.telegram.update_bio(new_bio)
        commit = getattr(provider, "commit_successful_update", None)
        if commit:
            await commit(new_bio)
        bot.record_bio_update(new_bio, "context_prod")
        await event.respond(
            f"✅ Био обновлено из context dataset:\n<code>{new_bio}</code>",
            parse_mode="html",
        )
        logger.info("Context bio updated via /update_context command: %s", new_bio)
    except ContextBatchNotReady as exc:
        logger.info("Context update skipped: %s", exc)
        await event.respond(
            "⏳ Context batch ещё не готов.\n"
            f"<code>{exc}</code>\n\n"
            "Сначала собери больше новых сообщений через /collect_context.",
            parse_mode="html",
        )
    except Exception:
        logger.exception("Error during /update_context command")
        await event.respond("❌ Ошибка при обновлении bio из контекста. Проверьте логи.")
