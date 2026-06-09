"""Tests for the inline-button UI: keyboards, callbacks, prompts, onboarding."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

from telebio.prompts import Prompt, get_prompt, load_prompts
from telebio.services import keyboards
from telebio.services.bot import BotService
from telebio.services.handlers.callbacks import handle_callback
from telebio.services.handlers.start import onboarding_text


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


_PROMPTS = [Prompt(name="Линал", system="sys linal"), Prompt(name="Абсурд", system="sys absurd")]


def _make_bot(mode: str = "list", **overrides) -> BotService:
    kwargs = dict(
        bot_token="tok",
        api_id=1,
        api_hash="hash",
        current_mode={"mode": mode, "prompt_name": "Линал"},
        prompts=_PROMPTS,
    )
    kwargs.update(overrides)
    bot = BotService(**kwargs)
    bot._bot = MagicMock()
    return bot


def _cb_event(data: str) -> AsyncMock:
    event = AsyncMock()
    event.data = data.encode("utf-8")
    event.sender_id = 1
    event.answer = AsyncMock()
    event.edit = AsyncMock()
    event.respond = AsyncMock()
    return event


def _texts(rows) -> list[str]:
    return [btn.text for row in rows for btn in row]


def _datas(rows) -> list[str]:
    return [btn.data.decode() for row in rows for btn in row]


# ------------------------------------------------------------------
# prompts
# ------------------------------------------------------------------


class TestPrompts:

    def test_load_prompts_from_file(self, tmp_path) -> None:
        path = tmp_path / "p.json"
        path.write_text(
            json.dumps([{"name": "A", "system": "sa"}, {"name": "B", "system": "sb"}]),
            encoding="utf-8",
        )
        prompts = load_prompts(path)
        assert [p.name for p in prompts] == ["A", "B"]

    def test_load_prompts_missing_file_fallback(self, tmp_path) -> None:
        prompts = load_prompts(tmp_path / "nope.json")
        assert len(prompts) == 1
        assert prompts[0].name == "default"

    def test_load_prompts_empty_fallback(self, tmp_path) -> None:
        path = tmp_path / "p.json"
        path.write_text("[]", encoding="utf-8")
        prompts = load_prompts(path)
        assert prompts[0].name == "default"

    def test_get_prompt_by_name(self) -> None:
        assert get_prompt(_PROMPTS, "Абсурд").system == "sys absurd"

    def test_get_prompt_unknown_falls_back_to_first(self) -> None:
        assert get_prompt(_PROMPTS, "missing").name == "Линал"


# ------------------------------------------------------------------
# keyboards
# ------------------------------------------------------------------


class TestKeyboards:

    def test_main_menu_list_mode_has_no_extra_buttons(self) -> None:
        rows = keyboards.main_menu(_make_bot("list"))
        datas = _datas(rows)
        assert "act:new" in datas
        assert "menu:modes" in datas
        assert "act:collect" not in datas
        assert "menu:prompts" not in datas

    def test_main_menu_telegram_context_has_collect(self) -> None:
        rows = keyboards.main_menu(_make_bot("telegram_context"))
        assert "act:collect" in _datas(rows)

    def test_main_menu_llm_has_prompts(self) -> None:
        rows = keyboards.main_menu(_make_bot("llm_prompt_generation"))
        assert "menu:prompts" in _datas(rows)

    def test_main_menu_state_button_reflects_state(self) -> None:
        active = _make_bot("list")
        assert any("Активно" in t for t in _texts(keyboards.main_menu(active)))
        paused = _make_bot("list")
        paused.toggle_pause()
        assert any("На паузе" in t for t in _texts(keyboards.main_menu(paused)))

    def test_main_menu_state_button_is_last_row(self) -> None:
        rows = keyboards.main_menu(_make_bot("telegram_context"))
        assert [b.data.decode() for b in rows[-1]] == ["act:pause"]

    def test_mode_menu_marks_current(self) -> None:
        rows = keyboards.mode_menu("telegram_context")
        datas = _datas(rows)
        assert datas == ["mode:list", "mode:llm_prompt_generation", "mode:telegram_context", "menu:main"]
        assert any("✅" in t and "telegram_context" in t for t in _texts(rows))

    def test_prompts_menu_indices(self) -> None:
        rows = keyboards.prompts_menu(_PROMPTS, "Линал")
        datas = _datas(rows)
        assert "pset:0" in datas and "pview:0" in datas
        assert "pset:1" in datas and "pview:1" in datas
        assert "menu:main" in datas


# ------------------------------------------------------------------
# callbacks
# ------------------------------------------------------------------


class TestCallbacks:

    async def test_mode_switch_to_telegram_context_shows_main_menu(self) -> None:
        bot = _make_bot("list")
        event = _cb_event("mode:telegram_context")

        await handle_callback(event, bot)

        assert bot.current_mode["mode"] == "telegram_context"
        event.edit.assert_awaited()
        assert "act:collect" in _datas(event.edit.call_args.kwargs["buttons"])

    async def test_mode_switch_to_llm_opens_prompts(self) -> None:
        bot = _make_bot("list")
        event = _cb_event("mode:llm_prompt_generation")

        await handle_callback(event, bot)

        assert bot.current_mode["mode"] == "llm_prompt_generation"
        assert "pset:0" in _datas(event.edit.call_args.kwargs["buttons"])

    async def test_pset_selects_prompt(self) -> None:
        bot = _make_bot("llm_prompt_generation")
        event = _cb_event("pset:1")

        await handle_callback(event, bot)

        assert bot.prompt_name == "Абсурд"

    async def test_pview_shows_prompt_text(self) -> None:
        bot = _make_bot("llm_prompt_generation")
        event = _cb_event("pview:1")

        await handle_callback(event, bot)

        event.respond.assert_awaited_once()
        assert "sys absurd" in event.respond.call_args[0][0]

    async def test_act_pause_toggles_and_refreshes(self) -> None:
        bot = _make_bot("list")
        event = _cb_event("act:pause")

        await handle_callback(event, bot)

        assert bot.paused
        event.edit.assert_awaited()

    async def test_menu_modes_navigation(self) -> None:
        bot = _make_bot("list")
        event = _cb_event("menu:modes")

        await handle_callback(event, bot)

        assert "mode:list" in _datas(event.edit.call_args.kwargs["buttons"])


# ------------------------------------------------------------------
# onboarding
# ------------------------------------------------------------------


class TestOnboarding:

    def test_onboarding_lists_all_modes(self) -> None:
        text = onboarding_text()
        assert "list" in text
        assert "llm_prompt_generation" in text
        assert "telegram_context" in text


# ------------------------------------------------------------------
# BotService prompt state
# ------------------------------------------------------------------


class TestBotPromptState:

    def test_set_prompt_updates_state(self) -> None:
        bot = _make_bot("llm_prompt_generation")
        bot.set_prompt("Абсурд")
        assert bot.prompt_name == "Абсурд"

    def test_prompts_property(self) -> None:
        assert [p.name for p in _make_bot().prompts] == ["Линал", "Абсурд"]
