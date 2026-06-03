"""Tests for TelegramService."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from telethon.tl.types import Message

from telebio.services.telegram import TelegramService


class _AsyncIter:
    def __init__(self, items):
        self._items = list(items)

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self._items:
            raise StopAsyncIteration
        return self._items.pop(0)


def _make_message(
    *,
    msg_id: int,
    text: str,
    when: datetime,
    peer_user_id: int,
    out: bool = True,
) -> MagicMock:
    msg = MagicMock(spec=Message)
    msg.id = msg_id
    msg.out = out
    msg.date = when
    msg.raw_text = text
    msg.message = text
    peer = MagicMock(spec=["user_id"])
    peer.user_id = peer_user_id
    msg.peer_id = peer
    return msg


def _make_dialog(*, peer_id: int, title: str) -> MagicMock:
    entity = MagicMock()
    entity.id = peer_id
    entity.title = title
    dialog = MagicMock()
    dialog.entity = entity
    dialog.name = title
    return dialog


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


def _bind_iter(mock_client: AsyncMock, dialogs_with_messages: list[tuple[MagicMock, list]]) -> None:
    dialogs = [dialog for dialog, _ in dialogs_with_messages]
    by_entity_id = {id(dialog.entity): msgs for dialog, msgs in dialogs_with_messages}

    mock_client.iter_dialogs = MagicMock(return_value=_AsyncIter(dialogs))

    def iter_messages_factory(entity, **_kwargs):
        return _AsyncIter(list(by_entity_id.get(id(entity), [])))

    mock_client.iter_messages = MagicMock(side_effect=iter_messages_factory)


class TestCollectRecentOutgoing:

    async def test_merges_burst_within_gap(
        self, service: TelegramService, mock_client: AsyncMock
    ) -> None:
        peer_id = 100
        base = datetime.now(UTC) - timedelta(hours=1)
        # iter_messages yields newest-first
        dialog_messages = [
            _make_message(msg_id=3, text="осталось 2 главы",
                          when=base + timedelta(seconds=60), peer_user_id=peer_id),
            _make_message(msg_id=2, text="пилю диплом",
                          when=base + timedelta(seconds=30), peer_user_id=peer_id),
            _make_message(msg_id=1, text="я сейчас",
                          when=base, peer_user_id=peer_id),
        ]
        _bind_iter(mock_client, [(_make_dialog(peer_id=peer_id, title="Alice"), dialog_messages)])

        collected = await service.collect_recent_outgoing_texts(
            days=7, limit=100, dialog_scan_limit=10,
            per_dialog_limit=10, merge_gap_seconds=300,
        )

        assert len(collected) == 1
        msg = collected[0]
        assert msg.message_key == f"{peer_id}:1"
        assert msg.message_id == 3
        assert msg.date == base + timedelta(seconds=60)
        assert msg.text == "я сейчас\nпилю диплом\nосталось 2 главы"

    async def test_does_not_merge_when_gap_exceeded(
        self, service: TelegramService, mock_client: AsyncMock
    ) -> None:
        peer_id = 200
        base = datetime.now(UTC) - timedelta(hours=2)
        dialog_messages = [
            _make_message(msg_id=2, text="через час продолжаю",
                          when=base + timedelta(seconds=600), peer_user_id=peer_id),
            _make_message(msg_id=1, text="первое сообщение",
                          when=base, peer_user_id=peer_id),
        ]
        _bind_iter(mock_client, [(_make_dialog(peer_id=peer_id, title="Bob"), dialog_messages)])

        collected = await service.collect_recent_outgoing_texts(
            days=7, limit=100, dialog_scan_limit=10,
            per_dialog_limit=10, merge_gap_seconds=300,
        )

        assert len(collected) == 2
        assert {msg.text for msg in collected} == {"первое сообщение", "через час продолжаю"}

    async def test_disabled_when_gap_is_zero(
        self, service: TelegramService, mock_client: AsyncMock
    ) -> None:
        peer_id = 300
        base = datetime.now(UTC) - timedelta(hours=1)
        dialog_messages = [
            _make_message(msg_id=2, text="вторая мысль",
                          when=base + timedelta(seconds=10), peer_user_id=peer_id),
            _make_message(msg_id=1, text="первая мысль",
                          when=base, peer_user_id=peer_id),
        ]
        _bind_iter(mock_client, [(_make_dialog(peer_id=peer_id, title="Carol"), dialog_messages)])

        collected = await service.collect_recent_outgoing_texts(
            days=7, limit=100, dialog_scan_limit=10,
            per_dialog_limit=10, merge_gap_seconds=0,
        )

        assert len(collected) == 2

    async def test_different_dialogs_are_not_merged(
        self, service: TelegramService, mock_client: AsyncMock
    ) -> None:
        base = datetime.now(UTC) - timedelta(hours=1)
        alice = [_make_message(msg_id=1, text="привет от alice",
                               when=base, peer_user_id=10)]
        bob = [_make_message(msg_id=2, text="привет от bob",
                             when=base + timedelta(seconds=5), peer_user_id=20)]
        _bind_iter(mock_client, [
            (_make_dialog(peer_id=10, title="Alice"), alice),
            (_make_dialog(peer_id=20, title="Bob"), bob),
        ])

        collected = await service.collect_recent_outgoing_texts(
            days=7, limit=100, dialog_scan_limit=10,
            per_dialog_limit=10, merge_gap_seconds=300,
        )

        assert len(collected) == 2
        assert {msg.dialog_title for msg in collected} == {"Alice", "Bob"}

    async def test_messages_older_than_cutoff_are_dropped(
        self, service: TelegramService, mock_client: AsyncMock
    ) -> None:
        peer_id = 400
        recent = datetime.now(UTC) - timedelta(hours=1)
        old = datetime.now(UTC) - timedelta(days=30)
        dialog_messages = [
            _make_message(msg_id=2, text="свежее", when=recent, peer_user_id=peer_id),
            _make_message(msg_id=1, text="древнее", when=old, peer_user_id=peer_id),
        ]
        _bind_iter(mock_client, [(_make_dialog(peer_id=peer_id, title="Dave"), dialog_messages)])

        collected = await service.collect_recent_outgoing_texts(
            days=7, limit=100, dialog_scan_limit=10,
            per_dialog_limit=10, merge_gap_seconds=300,
        )

        assert len(collected) == 1
        assert collected[0].text == "свежее"

    async def test_incoming_reply_breaks_group(
        self, service: TelegramService, mock_client: AsyncMock
    ) -> None:
        """Even within gap, an incoming message between own breaks the group."""
        peer_id = 500
        base = datetime.now(UTC) - timedelta(hours=1)
        # iter_messages newest-first
        dialog_messages = [
            _make_message(msg_id=4, text="и продолжение",
                          when=base + timedelta(seconds=90), peer_user_id=peer_id),
            _make_message(msg_id=3, text="ага",
                          when=base + timedelta(seconds=60),
                          peer_user_id=peer_id, out=False),
            _make_message(msg_id=2, text="вторая мысль",
                          when=base + timedelta(seconds=30), peer_user_id=peer_id),
            _make_message(msg_id=1, text="первая мысль",
                          when=base, peer_user_id=peer_id),
        ]
        _bind_iter(mock_client, [(_make_dialog(peer_id=peer_id, title="Eve"), dialog_messages)])

        collected = await service.collect_recent_outgoing_texts(
            days=7, limit=100, dialog_scan_limit=10,
            per_dialog_limit=10, merge_gap_seconds=1800,
        )

        # Без incoming все 3 own сообщения склеились бы в одно;
        # incoming msg_id=3 разделяет группу на две.
        assert len(collected) == 2
        texts = sorted([msg.text for msg in collected])
        assert texts == ["и продолжение", "первая мысль\nвторая мысль"]

    async def test_multiple_groups_separated_by_incoming(
        self, service: TelegramService, mock_client: AsyncMock
    ) -> None:
        peer_id = 600
        base = datetime.now(UTC) - timedelta(hours=1)
        dialog_messages = [
            _make_message(msg_id=6, text="ну ок",
                          when=base + timedelta(seconds=200), peer_user_id=peer_id),
            _make_message(msg_id=5, text="понятно",
                          when=base + timedelta(seconds=190), peer_user_id=peer_id),
            _make_message(msg_id=4, text="ответ",
                          when=base + timedelta(seconds=150),
                          peer_user_id=peer_id, out=False),
            _make_message(msg_id=3, text="и второе",
                          when=base + timedelta(seconds=60), peer_user_id=peer_id),
            _make_message(msg_id=2, text="первое",
                          when=base + timedelta(seconds=30), peer_user_id=peer_id),
            _make_message(msg_id=1, text="привет",
                          when=base, peer_user_id=peer_id),
        ]
        _bind_iter(mock_client, [(_make_dialog(peer_id=peer_id, title="Fox"), dialog_messages)])

        collected = await service.collect_recent_outgoing_texts(
            days=7, limit=100, dialog_scan_limit=10,
            per_dialog_limit=10, merge_gap_seconds=1800,
        )

        assert len(collected) == 2
        texts = [msg.text for msg in collected]
        assert texts == ["привет\nпервое\nи второе", "понятно\nну ок"]

    async def test_leading_incoming_does_not_affect_group(
        self, service: TelegramService, mock_client: AsyncMock
    ) -> None:
        peer_id = 700
        base = datetime.now(UTC) - timedelta(hours=1)
        dialog_messages = [
            _make_message(msg_id=3, text="и второе",
                          when=base + timedelta(seconds=60), peer_user_id=peer_id),
            _make_message(msg_id=2, text="первое",
                          when=base + timedelta(seconds=30), peer_user_id=peer_id),
            _make_message(msg_id=1, text="старая реплика собеседника",
                          when=base, peer_user_id=peer_id, out=False),
        ]
        _bind_iter(mock_client, [(_make_dialog(peer_id=peer_id, title="Gail"), dialog_messages)])

        collected = await service.collect_recent_outgoing_texts(
            days=7, limit=100, dialog_scan_limit=10,
            per_dialog_limit=10, merge_gap_seconds=1800,
        )

        assert len(collected) == 1
        assert collected[0].text == "первое\nи второе"

    async def test_incoming_text_not_stored(
        self, service: TelegramService, mock_client: AsyncMock
    ) -> None:
        """Incoming messages must not leak into the collected output."""
        peer_id = 800
        base = datetime.now(UTC) - timedelta(hours=1)
        dialog_messages = [
            _make_message(msg_id=2, text="ответ собеседника, не должен попасть",
                          when=base + timedelta(seconds=30),
                          peer_user_id=peer_id, out=False),
            _make_message(msg_id=1, text="моё сообщение",
                          when=base, peer_user_id=peer_id),
        ]
        _bind_iter(mock_client, [(_make_dialog(peer_id=peer_id, title="Hugo"), dialog_messages)])

        collected = await service.collect_recent_outgoing_texts(
            days=7, limit=100, dialog_scan_limit=10,
            per_dialog_limit=10, merge_gap_seconds=1800,
        )

        assert len(collected) == 1
        assert collected[0].text == "моё сообщение"
        for msg in collected:
            assert "собеседника" not in msg.text
