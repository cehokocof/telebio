"""Tests for bot command handlers."""

from __future__ import annotations

import asyncio
from collections import deque
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from telebio.services.bot import BotService
from telebio.services.handlers.status import handle_status
from telebio.services.handlers.history import handle_history
from telebio.services.handlers.set_mode import handle_set_mode
from telebio.services.handlers.new import handle_new
from telebio.services.handlers.pause import handle_pause


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _make_bot(**overrides) -> BotService:
    """Build a BotService with sensible defaults (no real Telegram client)."""
    kwargs = dict(
        bot_token="tok",
        api_id=1,
        api_hash="hash",
        current_mode={"mode": "list"},
    )
    kwargs.update(overrides)
    bot = BotService(**kwargs)
    # Replace the real TelegramClient so nothing touches the network
    bot._bot = MagicMock()
    return bot


def _make_event(pattern_match_group: str | None = None) -> AsyncMock:
    """Create a mock Telethon NewMessage event."""
    event = AsyncMock()
    event.respond = AsyncMock()
    if pattern_match_group is not None:
        match = MagicMock()
        match.group.return_value = pattern_match_group
        event.pattern_match = match
    return event


# ------------------------------------------------------------------
# /status
# ------------------------------------------------------------------


class TestHandleStatus:

    async def test_status_shows_mode_and_state(self) -> None:
        bot = _make_bot(current_mode={"mode": "llm"})
        event = _make_event()

        await handle_status(event, bot)

        event.respond.assert_awaited_once()
        text = event.respond.call_args[0][0]
        assert "llm" in text
        assert "активно" in text

    async def test_status_shows_paused(self) -> None:
        bot = _make_bot()
        bot.toggle_pause()
        event = _make_event()

        await handle_status(event, bot)

        text = event.respond.call_args[0][0]
        assert "приостановлено" in text

    async def test_status_shows_last_bio(self) -> None:
        bot = _make_bot()
        bot.record_bio_update("hello world", "list")
        event = _make_event()

        await handle_status(event, bot)

        text = event.respond.call_args[0][0]
        assert "hello world" in text

    async def test_status_shows_last_update_timestamp(self) -> None:
        bot = _make_bot()
        bot.record_bio_update("bio", "list")
        event = _make_event()

        await handle_status(event, bot)

        text = event.respond.call_args[0][0]
        assert "Last update" in text


# ------------------------------------------------------------------
# /history
# ------------------------------------------------------------------


class TestHandleHistory:

    async def test_history_empty(self) -> None:
        bot = _make_bot()
        event = _make_event()

        await handle_history(event, bot)

        text = event.respond.call_args[0][0]
        assert "No history" in text

    async def test_history_with_entries(self) -> None:
        bot = _make_bot()
        bot.record_bio_update("first", "list")
        bot.record_bio_update("second", "llm")
        event = _make_event()

        await handle_history(event, bot)

        text = event.respond.call_args[0][0]
        assert "first" in text
        assert "second" in text
        assert "list" in text
        assert "llm" in text


# ------------------------------------------------------------------
# /set_mode
# ------------------------------------------------------------------


class TestHandleSetMode:

    async def test_set_mode_switches(self) -> None:
        bot = _make_bot(current_mode={"mode": "list"})
        event = _make_event(pattern_match_group="llm")

        await handle_set_mode(event, bot)

        assert bot.current_mode["mode"] == "llm"
        text = event.respond.call_args[0][0]
        assert "llm" in text

    async def test_set_mode_same_mode(self) -> None:
        bot = _make_bot(current_mode={"mode": "list"})
        event = _make_event(pattern_match_group="list")

        await handle_set_mode(event, bot)

        text = event.respond.call_args[0][0]
        assert "Already" in text

    async def test_set_mode_invalid(self) -> None:
        bot = _make_bot()
        event = _make_event(pattern_match_group="unknown")

        await handle_set_mode(event, bot)

        text = event.respond.call_args[0][0]
        assert "Invalid" in text
        assert bot.current_mode["mode"] == "list"

    async def test_set_mode_case_insensitive(self) -> None:
        bot = _make_bot(current_mode={"mode": "list"})
        event = _make_event(pattern_match_group="LLM")

        await handle_set_mode(event, bot)

        assert bot.current_mode["mode"] == "llm"


# ------------------------------------------------------------------
# /new
# ------------------------------------------------------------------


class TestHandleNew:

    async def test_new_generates_and_applies_bio(self) -> None:
        mock_tg = AsyncMock()
        mock_provider = AsyncMock()
        mock_provider.get_bio.return_value = "свежее био"

        bot = _make_bot(
            telegram=mock_tg,
            provider_factory=lambda _mode: mock_provider,
        )
        event = _make_event()

        await handle_new(event, bot)

        mock_provider.get_bio.assert_awaited_once()
        mock_tg.update_bio.assert_awaited_once_with("свежее био")
        text = event.respond.call_args[0][0]
        assert "свежее био" in text

    async def test_new_records_history(self) -> None:
        mock_tg = AsyncMock()
        mock_provider = AsyncMock()
        mock_provider.get_bio.return_value = "new bio"

        bot = _make_bot(
            telegram=mock_tg,
            provider_factory=lambda _mode: mock_provider,
        )
        event = _make_event()

        await handle_new(event, bot)

        assert bot.last_bio == "new bio"
        assert len(bot.history) == 1

    async def test_new_without_telegram(self) -> None:
        bot = _make_bot()  # no telegram, no provider_factory
        event = _make_event()

        await handle_new(event, bot)

        text = event.respond.call_args[0][0]
        assert "не настроен" in text

    async def test_new_handles_provider_error(self) -> None:
        mock_tg = AsyncMock()

        def bad_factory(_mode):
            p = AsyncMock()
            p.get_bio.side_effect = RuntimeError("boom")
            return p

        bot = _make_bot(telegram=mock_tg, provider_factory=bad_factory)
        event = _make_event()

        await handle_new(event, bot)

        text = event.respond.call_args[0][0]
        assert "Ошибка" in text

    async def test_new_uses_current_mode(self) -> None:
        mock_tg = AsyncMock()
        captured_modes: list[str] = []

        def factory(mode: str):
            captured_modes.append(mode)
            p = AsyncMock()
            p.get_bio.return_value = "bio"
            return p

        bot = _make_bot(
            current_mode={"mode": "llm"},
            telegram=mock_tg,
            provider_factory=factory,
        )
        event = _make_event()

        await handle_new(event, bot)

        assert captured_modes == ["llm"]


# ------------------------------------------------------------------
# /pause
# ------------------------------------------------------------------


class TestHandlePause:

    async def test_pause_toggles_on(self) -> None:
        bot = _make_bot()
        assert not bot.paused
        event = _make_event()

        await handle_pause(event, bot)

        assert bot.paused
        text = event.respond.call_args[0][0]
        assert "приостановлено" in text

    async def test_pause_toggles_off(self) -> None:
        bot = _make_bot()
        bot.toggle_pause()  # pause
        event = _make_event()

        await handle_pause(event, bot)

        assert not bot.paused
        text = event.respond.call_args[0][0]
        assert "возобновлено" in text

    async def test_pause_double_toggle(self) -> None:
        bot = _make_bot()
        event = _make_event()

        await handle_pause(event, bot)
        assert bot.paused

        await handle_pause(event, bot)
        assert not bot.paused


# ------------------------------------------------------------------
# BotService unit tests
# ------------------------------------------------------------------


class TestBotServiceUnit:

    def test_record_bio_update(self) -> None:
        bot = _make_bot()
        bot.record_bio_update("test bio", "list")

        assert bot.last_bio == "test bio"
        assert bot.last_update is not None
        assert len(bot.history) == 1
        assert bot.history[0]["bio"] == "test bio"
        assert bot.history[0]["mode"] == "list"

    def test_history_max_len(self) -> None:
        bot = _make_bot()
        for i in range(15):
            bot.record_bio_update(f"bio {i}", "list")

        assert len(bot.history) == 10

    def test_toggle_pause(self) -> None:
        bot = _make_bot()
        assert not bot.paused
        bot.toggle_pause()
        assert bot.paused
        bot.toggle_pause()
        assert not bot.paused

    def test_properties_expose_init_values(self) -> None:
        mock_tg = AsyncMock()
        factory = lambda m: AsyncMock()
        bot = _make_bot(
            current_mode={"mode": "llm"},
            telegram=mock_tg,
            provider_factory=factory,
        )
        assert bot.current_mode == {"mode": "llm"}
        assert bot.telegram is mock_tg
        assert bot.provider_factory is factory
        assert bot.last_bio == ""
        assert bot.last_update is None
        assert len(bot.history) == 0


# ------------------------------------------------------------------
# Scheduler respects pause
# ------------------------------------------------------------------


class TestSchedulerPause:

    async def test_scheduler_skips_when_paused(self, mock_telegram: AsyncMock) -> None:
        """When bot.paused is True the scheduler should NOT call the provider."""
        from telebio.scheduler import run_scheduler

        provider = AsyncMock()
        provider.get_bio.return_value = "bio"
        bot = _make_bot()
        bot.toggle_pause()

        task = asyncio.create_task(
            run_scheduler(
                mock_telegram,
                provider,
                interval_minutes=0,
                bot=bot,
            )
        )

        await asyncio.sleep(0.1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        provider.get_bio.assert_not_awaited()
        mock_telegram.update_bio.assert_not_awaited()

    async def test_scheduler_resumes_after_unpause(self, mock_telegram: AsyncMock) -> None:
        from telebio.scheduler import run_scheduler

        provider = AsyncMock()
        provider.get_bio.return_value = "bio"
        bot = _make_bot()
        bot.toggle_pause()

        task = asyncio.create_task(
            run_scheduler(
                mock_telegram,
                provider,
                interval_minutes=0,
                bot=bot,
            )
        )

        # Let the scheduler notice it's paused
        await asyncio.sleep(0.05)
        assert provider.get_bio.await_count == 0

        # Unpause and let it run
        bot.toggle_pause()
        await asyncio.sleep(0.1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        assert provider.get_bio.await_count >= 1
        mock_telegram.update_bio.assert_awaited()
