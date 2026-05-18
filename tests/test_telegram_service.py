"""Tests for TelegramService."""

from __future__ import annotations

import asyncio
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
