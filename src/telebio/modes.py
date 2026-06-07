"""Bio-provider mode identifiers and their human-friendly labels."""

from __future__ import annotations

MODE_LIST = "list"
MODE_LLM = "llm_prompt_generation"
MODE_TELEGRAM_CONTEXT = "telegram_context"

# Stable order used for menus and onboarding.
MODES: tuple[str, ...] = (MODE_LIST, MODE_LLM, MODE_TELEGRAM_CONTEXT)

# Friendly Russian labels shown on buttons (technical key kept alongside).
MODE_LABELS: dict[str, str] = {
    MODE_LIST: "Список фраз",
    MODE_LLM: "Генерация по промпту",
    MODE_TELEGRAM_CONTEXT: "Контекст из переписки",
}

# Short descriptions used in the /start onboarding.
MODE_DESCRIPTIONS: dict[str, str] = {
    MODE_LIST: "крутит готовые фразы из списка",
    MODE_LLM: "генерит bio нейросетью по выбранному промпту",
    MODE_TELEGRAM_CONTEXT: "пишет bio по твоим реальным последним сообщениям",
}


def is_valid(mode: str) -> bool:
    return mode in MODE_LABELS


def display(mode: str) -> str:
    """Return ``"RU-подпись (key)"`` for a known mode, or the raw key."""
    label = MODE_LABELS.get(mode)
    return f"{label} ({mode})" if label else mode
