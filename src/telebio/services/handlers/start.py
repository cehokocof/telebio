"""Handler for the /start command — onboarding and mode selection."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from telethon import events

from telebio.modes import MODE_DESCRIPTIONS, MODE_LABELS, MODES
from telebio.services.keyboards import mode_menu

if TYPE_CHECKING:
    from telebio.services.bot import BotService

logger = logging.getLogger(__name__)


def onboarding_text() -> str:
    lines = [
        "👋 Привет! Это <b>TeleBio</b> — он сам меняет твоё Telegram bio.",
        "",
        "Выбери режим работы:",
        "",
    ]
    for key in MODES:
        lines.append(f"• <b>{MODE_LABELS[key]}</b> (<code>{key}</code>)")
        lines.append(f"  {MODE_DESCRIPTIONS[key]}")
    return "\n".join(lines)


async def handle_start(event: events.NewMessage.Event, bot: BotService) -> None:
    """Greet the owner and offer the mode picker."""
    current = bot.current_mode.get("mode", "")
    await event.respond(
        onboarding_text(), parse_mode="html", buttons=mode_menu(current)
    )
