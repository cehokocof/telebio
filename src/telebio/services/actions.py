"""Reusable bot actions shared by typed commands and inline-button callbacks.

Each function returns the HTML text to show the user; the calling handler is
responsible for delivering it (``event.respond`` / ``event.edit``).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from telebio.context_exceptions import ContextBatchNotReady
from telebio.modes import (
    MODE_LIST,
    MODE_TELEGRAM_CONTEXT,
    display,
    is_valid,
)

if TYPE_CHECKING:
    from telebio.services.bot import BotService

logger = logging.getLogger(__name__)


def menu_text(bot: BotService) -> str:
    """Header text for the main menu."""
    mode = bot.current_mode.get("mode", "unknown")
    state = "⏸ приостановлено" if bot.paused else "▶️ активно"
    lines = [
        "🤖 <b>TeleBio</b>",
        f"📊 <b>Режим:</b> {display(mode)}",
        f"⏯ <b>Состояние:</b> {state}",
        f"📝 <b>Био:</b> {bot.last_bio or '(none)'}",
    ]
    if bot.prompt_name:
        lines.insert(2, f"🧩 <b>Промпт:</b> <code>{bot.prompt_name}</code>")
    return "\n".join(lines)


def status_text(bot: BotService) -> str:
    mode = bot.current_mode.get("mode", "unknown")
    paused_label = "⏸ приостановлено" if bot.paused else "▶️ активно"
    lines = [
        "🤖 <b>TeleBio Status</b>",
        "",
        f"📊 <b>Mode:</b> {display(mode)}",
        f"⏯ <b>State:</b> {paused_label}",
        f"📝 <b>Current bio:</b> {bot.last_bio or '(none)'}",
    ]
    if bot.last_update:
        lines.append(
            f"🕐 <b>Last update:</b> {bot.last_update.strftime('%Y-%m-%d %H:%M:%S')}"
        )
    return "\n".join(lines)


def history_text(bot: BotService) -> str:
    if not bot.history:
        return "📜 No history available yet."
    lines = ["📜 <b>Recent Bio Updates:</b>", ""]
    for i, entry in enumerate(reversed(bot.history), 1):
        lines.append(
            f"{i}. [{entry['timestamp']}] <code>{entry['mode']}</code>\n"
            f"   {entry['bio']}"
        )
    return "\n\n".join(lines)


def toggle_pause_text(bot: BotService) -> str:
    bot.toggle_pause()
    if bot.paused:
        logger.info("Auto-update paused")
        return (
            "⏸ Автообновление приостановлено.\n"
            "Текущее био сохранено. Нажми ещё раз, чтобы возобновить."
        )
    logger.info("Auto-update resumed")
    return "▶️ Автообновление возобновлено."


def apply_mode(bot: BotService, mode: str) -> str:
    """Switch the active bio-provider mode."""
    mode = mode.strip().lower()
    if not is_valid(mode):
        return (
            "❌ Неизвестный режим. Доступны: <code>list</code>, "
            "<code>llm_prompt_generation</code>, <code>telegram_context</code>."
        )
    if mode == bot.current_mode.get("mode", ""):
        return f"ℹ️ Уже выбран режим {display(mode)}."
    bot.current_mode["mode"] = mode
    logger.info("Mode switched to '%s'", mode)
    return (
        f"✅ Режим переключён на {display(mode)}\n"
        "Следующее обновление использует новый провайдер."
    )


def apply_prompt(bot: BotService, name: str) -> str:
    """Select the active named prompt for llm_prompt_generation."""
    bot.set_prompt(name)
    logger.info("Active prompt set to '%s'", name)
    return f"✅ Активный промпт: <code>{name}</code>"


async def run_new(bot: BotService) -> str:
    """Generate and apply a fresh bio using the current mode."""
    if not bot.telegram or not bot.provider_factory:
        return "❌ Bot не настроен для обновления био."

    mode = bot.current_mode.get("mode", MODE_LIST)
    try:
        provider = bot.provider_factory(mode)
        if mode == MODE_TELEGRAM_CONTEXT:
            new_bio = await provider.get_bio(force=True)
        else:
            new_bio = await provider.get_bio()
        await bot.telegram.update_bio(new_bio)
        commit = getattr(provider, "commit_successful_update", None)
        if commit:
            await commit(new_bio)
        bot.record_bio_update(new_bio, mode)
        logger.info("Bio updated via /new: %s", new_bio)
        return f"✅ Био обновлено:\n<code>{new_bio}</code>"
    except ContextBatchNotReady as exc:
        logger.info("Bio update skipped: %s", exc)
        return (
            "⏳ Context batch ещё не готов.\n"
            f"<code>{exc}</code>\n\n"
            "Сначала собери больше новых сообщений через /collect."
        )
    except Exception:
        logger.exception("Error during /new")
        return "❌ Ошибка при обновлении био. Проверьте логи."


async def run_collect(bot: BotService) -> str:
    """Collect and classify context rows into the parquet dataset."""
    if not bot.provider_factory:
        return "❌ Bot не настроен для сбора контекста."
    try:
        provider = bot.provider_factory(MODE_TELEGRAM_CONTEXT)
        collect = getattr(provider, "collect_context", None)
        if collect is None:
            return "❌ Текущий provider не поддерживает сбор контекста."
        stats = await collect()
        logger.info("Context collected: %s", stats)
        return (
            "✅ Context collected\n"
            f"collected: <code>{stats['collected']}</code>\n"
            f"changed_rows: <code>{stats['changed_rows']}</code>\n"
            f"classified: <code>{stats['classified']}</code>\n"
            f"pending_keep: <code>{stats['pending_keep']}</code>\n"
            f"pending_maybe: <code>{stats['pending_maybe']}</code>"
        )
    except Exception:
        logger.exception("Error during /collect")
        return "❌ Ошибка при сборе контекста. Проверьте логи."
