"""Tests for TelegramService."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from telebio.services.telegram import TelegramService


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

@pytest.fixture()
def mock_client() -> AsyncMock:
    """A mocked TelegramClient."""
    client = AsyncMock()
    me = MagicMock()
    me.first_name = "TestUser"
    me.id = 12345
    client.get_me.return_value = me
    return client


@pytest.fixture()
def service(mock_client: AsyncMock) -> TelegramService:
    """TelegramService with the internal client mocked."""
    svc = TelegramService(api_id=1, api_hash="hash", session_path="/tmp/test")
    svc._client = mock_client
    return svc


# ------------------------------------------------------------------
# Lifecycle
# ------------------------------------------------------------------

class TestLifecycle:

    async def test_start_calls_client_start(self, service: TelegramService, mock_client: AsyncMock) -> None:
        await service.start()
        mock_client.start.assert_awaited_once()
        mock_client.get_me.assert_awaited_once()

    async def test_stop_disconnects(self, service: TelegramService, mock_client: AsyncMock) -> None:
        await service.stop()
        mock_client.disconnect.assert_awaited_once()

    async def test_context_manager(self, service: TelegramService, mock_client: AsyncMock) -> None:
        async with service:
            pass
        mock_client.start.assert_awaited_once()
        mock_client.disconnect.assert_awaited_once()


# ------------------------------------------------------------------
# update_bio
# ------------------------------------------------------------------

class TestUpdateBio:

    async def test_successful_update(self, service: TelegramService, mock_client: AsyncMock) -> None:
        await service.update_bio("New bio text")
        mock_client.assert_awaited_once()

    async def test_flood_wait_retry(self, service: TelegramService, mock_client: AsyncMock) -> None:
        """FloodWaitError should cause a sleep then retry."""
        from telethon.errors import FloodWaitError

        # First call raises FloodWaitError, second call succeeds
        exc = FloodWaitError(request=MagicMock(), capture=0)
        exc.seconds = 0  # Don't actually sleep in tests
        mock_client.side_effect = [exc, None]

        with patch("telebio.services.telegram.asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            await service.update_bio("retry bio")
            mock_sleep.assert_awaited_once_with(0)

        assert mock_client.await_count == 2

    async def test_rpc_error_reraises(self, service: TelegramService, mock_client: AsyncMock) -> None:
        from telethon.errors import RPCError

        mock_client.side_effect = RPCError(request=MagicMock(), message="TEST_ERROR")

        with pytest.raises(RPCError):
            await service.update_bio("will fail")


# ------------------------------------------------------------------
# collect_recent_outgoing_texts
# ------------------------------------------------------------------


class _AsyncMessages:
    def __init__(self, messages: list[MagicMock]) -> None:
        self._messages = messages

    def __aiter__(self):
        self._iter = iter(self._messages)
        return self

    async def __anext__(self):
        try:
            return next(self._iter)
        except StopIteration as exc:
            raise StopAsyncIteration from exc


def _message(
    text: str | None,
    date: datetime | None = None,
    *,
    out: bool = True,
) -> MagicMock:
    msg = MagicMock()
    msg.raw_text = text
    msg.message = text
    msg.date = date or datetime.now(timezone.utc)
    msg.out = out
    return msg


def _dialog(entity: str, title: str | None = None) -> MagicMock:
    dialog = MagicMock()
    dialog.entity = entity
    dialog.title = title
    return dialog


class TestCollectRecentOutgoingTexts:

    async def test_collects_texts_in_chronological_order(
        self,
        service: TelegramService,
        mock_client: AsyncMock,
    ) -> None:
        now = datetime.now(timezone.utc)
        mock_client.get_dialogs.return_value = [_dialog("chat", "Chat Title")]
        mock_client.iter_messages = MagicMock(return_value=_AsyncMessages([
            _message("new", now),
            _message("incoming", out=False),
            _message(None),
            _message(""),
            _message("old", now - timedelta(minutes=1)),
        ]))

        result = await service.collect_recent_outgoing_texts(days=7, limit=10)

        assert [message.text for message in result] == ["old", "new"]
        assert [message.dialog for message in result] == ["Chat Title", "Chat Title"]
        mock_client.get_dialogs.assert_awaited_once_with(limit=10)
        mock_client.iter_messages.assert_called_once_with(
            "chat",
            limit=10,
            wait_time=1,
        )

    async def test_stops_at_cutoff(
        self,
        service: TelegramService,
        mock_client: AsyncMock,
    ) -> None:
        now = datetime.now(timezone.utc)
        mock_client.get_dialogs.return_value = [_dialog("chat")]
        mock_client.iter_messages = MagicMock(return_value=_AsyncMessages([
            _message("fresh", now),
            _message("too old", now - timedelta(days=10)),
            _message("never reached", now - timedelta(days=11)),
        ]))

        result = await service.collect_recent_outgoing_texts(days=7, limit=10)

        assert [message.text for message in result] == ["fresh"]

    async def test_applies_global_limit_across_dialogs(
        self,
        service: TelegramService,
        mock_client: AsyncMock,
    ) -> None:
        now = datetime.now(timezone.utc)
        mock_client.get_dialogs.return_value = [_dialog("a"), _dialog("b")]
        mock_client.iter_messages = MagicMock(side_effect=[
            _AsyncMessages([
                _message("a-new", now),
                _message("a-old", now - timedelta(minutes=3)),
            ]),
            _AsyncMessages([
                _message("b-new", now - timedelta(minutes=1)),
                _message("b-old", now - timedelta(minutes=2)),
            ]),
        ])

        result = await service.collect_recent_outgoing_texts(days=7, limit=2)

        assert [message.text for message in result] == ["b-new", "a-new"]
        mock_client.get_dialogs.assert_awaited_once_with(limit=2)
        assert mock_client.iter_messages.call_count == 2

    async def test_caps_scan_size_for_large_limit(
        self,
        service: TelegramService,
        mock_client: AsyncMock,
    ) -> None:
        mock_client.get_dialogs.return_value = [_dialog("chat")]
        mock_client.iter_messages = MagicMock(return_value=_AsyncMessages([]))

        await service.collect_recent_outgoing_texts(days=7, limit=500)

        mock_client.get_dialogs.assert_awaited_once_with(limit=10)
        mock_client.iter_messages.assert_called_once_with(
            "chat",
            limit=50,
            wait_time=1,
        )

    async def test_accepts_custom_scan_limits(
        self,
        service: TelegramService,
        mock_client: AsyncMock,
    ) -> None:
        mock_client.get_dialogs.return_value = [_dialog("chat")]
        mock_client.iter_messages = MagicMock(return_value=_AsyncMessages([]))

        await service.collect_recent_outgoing_texts(
            days=7,
            limit=500,
            dialog_scan_limit=30,
            per_dialog_limit=100,
        )

        mock_client.get_dialogs.assert_awaited_once_with(limit=30)
        mock_client.iter_messages.assert_called_once_with(
            "chat",
            limit=100,
            wait_time=1,
        )

    async def test_rejects_invalid_window(self, service: TelegramService) -> None:
        with pytest.raises(ValueError):
            await service.collect_recent_outgoing_texts(days=0, limit=10)

    async def test_logs_selected_messages(
        self,
        service: TelegramService,
        mock_client: AsyncMock,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        now = datetime.now(timezone.utc)
        mock_client.get_dialogs.return_value = [_dialog("chat", "Dima")]
        mock_client.iter_messages = MagicMock(return_value=_AsyncMessages([
            _message("обсуждаю линал", now),
        ]))

        with caplog.at_level("INFO", logger="telebio.services.telegram"):
            await service.collect_recent_outgoing_texts(days=7, limit=10)

        assert "Dima" in caplog.text
        assert "обсуждаю линал" in caplog.text
