"""Central registry of user-facing strings for the management bot.

All bot message text lives here so wording stays consistent and is edited in
one place. Model/LLM prompts belong with their providers, not in this module.
Functions take plain values (not ``BotService``) to stay import-light.
"""

from __future__ import annotations

from telebio.modes import MODE_DESCRIPTIONS, MODE_LABELS, MODES, display

# ── Shared fragments ──────────────────────────────────────────────────


def pause_state(paused: bool) -> str:
    return "⏸ приостановлено" if paused else "▶️ активно"


# ── Main menu ─────────────────────────────────────────────────────────


def menu_text(*, mode: str, bio: str, prompt_name: str | None) -> str:
    lines = [
        "🤖 <b>TeleBio</b>",
        f"📊 <b>Режим:</b> {display(mode)}",
        f"📝 <b>Био:</b> {bio or '(none)'}",
    ]
    if prompt_name:
        lines.insert(2, f"🧩 <b>Промпт:</b> <code>{prompt_name}</code>")
    return "\n".join(lines)


# ── Status ────────────────────────────────────────────────────────────


def status_text(*, mode: str, paused: bool, bio: str, last_update: str | None) -> str:
    lines = [
        "🤖 <b>TeleBio Status</b>",
        "",
        f"📊 <b>Mode:</b> {display(mode)}",
        f"⏯ <b>State:</b> {pause_state(paused)}",
        f"📝 <b>Current bio:</b> {bio or '(none)'}",
    ]
    if last_update:
        lines.append(f"🕐 <b>Last update:</b> {last_update}")
    return "\n".join(lines)


# ── History ───────────────────────────────────────────────────────────

HISTORY_EMPTY = "📜 No history available yet."


def history_text(history: list[dict]) -> str:
    if not history:
        return HISTORY_EMPTY
    lines = ["📜 <b>Recent Bio Updates:</b>", ""]
    for i, entry in enumerate(reversed(history), 1):
        lines.append(
            f"{i}. [{entry['timestamp']}] <code>{entry['mode']}</code>\n"
            f"   {entry['bio']}"
        )
    return "\n\n".join(lines)


# ── Pause toggle ──────────────────────────────────────────────────────

PAUSE_PAUSED = (
    "⏸ Автообновление приостановлено.\n"
    "Текущее био сохранено. Нажми ещё раз, чтобы возобновить."
)
PAUSE_RESUMED = "▶️ Автообновление возобновлено."


# ── Mode switching ────────────────────────────────────────────────────

MODE_UNKNOWN = (
    "❌ Неизвестный режим. Доступны: <code>list</code>, "
    "<code>llm_prompt_generation</code>, <code>telegram_context</code>."
)


def mode_already(mode: str) -> str:
    return f"ℹ️ Уже выбран режим {display(mode)}."


def mode_switched(mode: str) -> str:
    return (
        f"✅ Режим переключён на {display(mode)}\n"
        "Следующее обновление использует новый провайдер."
    )


# ── Prompts ───────────────────────────────────────────────────────────

MODE_MENU = "🔄 Выбери режим:"


def prompt_applied(name: str) -> str:
    return f"✅ Активный промпт: <code>{name}</code>"


def prompts_header(active: str | None) -> str:
    return f"🧩 Промпты · активный: <code>{active or '—'}</code>"


def prompt_view(name: str, system: str) -> str:
    return f"🧩 <b>{name}</b>\n\n{system}"


# ── /new ──────────────────────────────────────────────────────────────

NEW_NOT_CONFIGURED = "❌ Bot не настроен для обновления био."
NEW_ERROR = "❌ Ошибка при обновлении био. Проверьте логи."


def new_success(bio: str) -> str:
    return f"✅ Био обновлено:\n<code>{bio}</code>"


def new_batch_not_ready(detail: str) -> str:
    return (
        "⏳ Context batch ещё не готов.\n"
        f"<code>{detail}</code>\n\n"
        "Сначала собери больше новых сообщений через /collect."
    )


# ── /collect ──────────────────────────────────────────────────────────

COLLECT_PROGRESS = "⏳ Собираю сообщения и обновляю parquet dataset…"
COLLECT_NOT_CONFIGURED = "❌ Bot не настроен для сбора контекста."
COLLECT_NOT_SUPPORTED = "❌ Текущий provider не поддерживает сбор контекста."
COLLECT_ERROR = "❌ Ошибка при сборе контекста. Проверьте логи."


def collect_success(stats: dict) -> str:
    return (
        "✅ Context collected\n"
        f"collected: <code>{stats['collected']}</code>\n"
        f"changed_rows: <code>{stats['changed_rows']}</code>\n"
        f"classified: <code>{stats['classified']}</code>\n"
        f"pending_keep: <code>{stats['pending_keep']}</code>\n"
        f"pending_maybe: <code>{stats['pending_maybe']}</code>"
    )


# ── Callback toasts ───────────────────────────────────────────────────

TOAST_GENERATING = "Генерирую…"
TOAST_COLLECTING = "Собираю…"


# ── Onboarding (/start) ───────────────────────────────────────────────


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
