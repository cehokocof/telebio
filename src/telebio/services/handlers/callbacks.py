"""Inline-button callback router for the management bot."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from telethon import events
from telethon.errors import MessageNotModifiedError

from telebio.modes import MODE_LLM
from telebio.services import actions
from telebio.services.keyboards import main_menu, mode_menu, prompts_menu

if TYPE_CHECKING:
    from telebio.services.bot import BotService

logger = logging.getLogger(__name__)

_MODE_MENU_TEXT = "🔄 Выбери режим:"


def _prompts_header(bot: BotService) -> str:
    return f"🧩 Промпты · активный: <code>{bot.prompt_name or '—'}</code>"


async def _safe_edit(event: events.CallbackQuery.Event, text: str, buttons) -> None:
    try:
        await event.edit(text, parse_mode="html", buttons=buttons)
    except MessageNotModifiedError:
        pass


async def handle_callback(event: events.CallbackQuery.Event, bot: BotService) -> None:
    """Route an inline-button press to the matching action."""
    data = (event.data or b"").decode("utf-8", "ignore")

    if data == "menu:main":
        await _safe_edit(event, actions.menu_text(bot), main_menu(bot))
    elif data == "menu:modes":
        await _safe_edit(event, _MODE_MENU_TEXT, mode_menu(bot.current_mode.get("mode", "")))
    elif data == "menu:prompts":
        await _safe_edit(event, _prompts_header(bot), prompts_menu(bot.prompts, bot.prompt_name))
    elif data.startswith("mode:"):
        result = actions.apply_mode(bot, data.split(":", 1)[1])
        await event.answer(_toast(result))
        if bot.current_mode.get("mode") == MODE_LLM:
            await _safe_edit(event, _prompts_header(bot), prompts_menu(bot.prompts, bot.prompt_name))
        else:
            await _safe_edit(event, actions.menu_text(bot), main_menu(bot))
    elif data.startswith("pset:"):
        prompt = _prompt_at(bot, data)
        if prompt is not None:
            await event.answer(_toast(actions.apply_prompt(bot, prompt.name)))
            await _safe_edit(event, _prompts_header(bot), prompts_menu(bot.prompts, bot.prompt_name))
    elif data.startswith("pview:"):
        prompt = _prompt_at(bot, data)
        if prompt is not None:
            await event.answer()
            await event.respond(
                f"🧩 <b>{prompt.name}</b>\n\n{prompt.system}", parse_mode="html"
            )
    elif data == "act:new":
        await event.answer("Генерирую…")
        await event.respond(await actions.run_new(bot), parse_mode="html")
    elif data == "act:collect":
        await event.answer("Собираю…")
        await event.respond("⏳ Собираю сообщения и обновляю parquet dataset…")
        await event.respond(await actions.run_collect(bot), parse_mode="html")
    elif data == "act:status":
        await event.answer()
        await event.respond(actions.status_text(bot), parse_mode="html")
    elif data == "act:history":
        await event.answer()
        await event.respond(actions.history_text(bot), parse_mode="html")
    elif data == "act:pause":
        await event.answer(_toast(actions.toggle_pause_text(bot)))
        await _safe_edit(event, actions.menu_text(bot), main_menu(bot))
    else:
        await event.answer()


def _prompt_at(bot: BotService, data: str):
    try:
        index = int(data.split(":", 1)[1])
    except (ValueError, IndexError):
        return None
    if 0 <= index < len(bot.prompts):
        return bot.prompts[index]
    return None


def _toast(text: str) -> str:
    """Strip simple HTML tags for the short callback toast (max ~200 chars)."""
    plain = text.replace("<code>", "").replace("</code>", "")
    plain = plain.replace("<b>", "").replace("</b>", "")
    return plain[:195]
