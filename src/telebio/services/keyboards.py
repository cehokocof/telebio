"""Inline-keyboard builders for the management bot.

Callback-data format (kept well under Telegram's 64-byte limit):
    menu:main | menu:modes | menu:prompts
    mode:<key>
    pset:<index> | pview:<index>
    act:new | act:pause | act:status | act:history | act:collect
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from telethon import Button

from telebio.modes import MODE_LABELS, MODE_LLM, MODE_TELEGRAM_CONTEXT, MODES

if TYPE_CHECKING:
    from telebio.prompts import Prompt
    from telebio.services.bot import BotService


def main_menu(bot: BotService) -> list[list[Button]]:
    mode = bot.current_mode.get("mode", "")
    rows: list[list[Button]] = [
        [Button.inline("✨ Новое био", b"act:new")],
        [
            Button.inline("🔄 Режим", b"menu:modes"),
            Button.inline("📜 История", b"act:history"),
            Button.inline("📊 Статус", b"act:status"),
        ],
    ]
    if mode == MODE_LLM:
        rows.append([Button.inline("🧩 Промпты", b"menu:prompts")])
    if mode == MODE_TELEGRAM_CONTEXT:
        rows.append([Button.inline("📥 Собрать контекст", b"act:collect")])
    state_label = (
        "⏸ На паузе — нажми, чтобы продолжить"
        if bot.paused
        else "▶️ Активно — нажми, чтобы пауза"
    )
    rows.append([Button.inline(state_label, b"act:pause")])
    return rows


def mode_menu(current: str) -> list[list[Button]]:
    rows: list[list[Button]] = []
    for key in MODES:
        mark = "✅ " if key == current else ""
        rows.append([Button.inline(f"{mark}{MODE_LABELS[key]} ({key})", f"mode:{key}")])
    rows.append([Button.inline("⬅️ Назад", b"menu:main")])
    return rows


def prompts_menu(prompts: list[Prompt], active: str | None) -> list[list[Button]]:
    rows: list[list[Button]] = []
    for i, prompt in enumerate(prompts):
        mark = "✅ " if prompt.name == active else ""
        rows.append(
            [
                Button.inline(f"{mark}{prompt.name}", f"pset:{i}"),
                Button.inline("👁", f"pview:{i}"),
            ]
        )
    rows.append([Button.inline("⬅️ Назад", b"menu:main")])
    return rows
