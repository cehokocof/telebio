"""Handler for collecting context messages without updating bio."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from telethon import events

if TYPE_CHECKING:
    from telebio.services.bot import BotService

logger = logging.getLogger(__name__)


async def handle_collect_context(event: events.NewMessage.Event, bot: BotService) -> None:
    """Collect and classify context rows into the parquet dataset."""
    if not bot.provider_factory:
        await event.respond("❌ Bot не настроен для сбора контекста.")
        return

    try:
        provider = bot.provider_factory("context_prod")
        collect = getattr(provider, "collect_context", None)
        if collect is None:
            await event.respond("❌ Текущий provider не поддерживает сбор контекста.")
            return

        await event.respond("⏳ Собираю сообщения и обновляю parquet dataset…")
        stats = await collect()
        await event.respond(
            "✅ Context collected\n"
            f"collected: <code>{stats['collected']}</code>\n"
            f"changed_rows: <code>{stats['changed_rows']}</code>\n"
            f"classified: <code>{stats['classified']}</code>\n"
            f"pending_keep: <code>{stats['pending_keep']}</code>\n"
            f"pending_maybe: <code>{stats['pending_maybe']}</code>",
            parse_mode="html",
        )
        logger.info("Context collected via bot command: %s", stats)
    except Exception:
        logger.exception("Error during /collect_context command")
        await event.respond("❌ Ошибка при сборе контекста. Проверьте логи.")
