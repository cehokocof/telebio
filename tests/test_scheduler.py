"""Tests for the scheduler loop."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import pytest

from telebio.scheduler import run_scheduler


class TestRunScheduler:

    async def test_calls_provider_and_telegram(self, mock_telegram: AsyncMock) -> None:
        """Scheduler should get bio from provider and push it to Telegram."""
        provider = AsyncMock()
        provider.get_bio.return_value = "test bio"

        task = asyncio.create_task(run_scheduler(mock_telegram, provider, interval_minutes=60))

        # Give the loop one iteration to run
        await asyncio.sleep(0.05)
        task.cancel()

        with pytest.raises(asyncio.CancelledError):
            await task

        provider.get_bio.assert_awaited_once()
        mock_telegram.update_bio.assert_awaited_once_with("test bio")

    async def test_continues_on_provider_error(self, mock_telegram: AsyncMock) -> None:
        """If the provider raises, the scheduler should NOT crash."""
        provider = AsyncMock()
        provider.get_bio.side_effect = [RuntimeError("boom"), "ok bio"]

        task = asyncio.create_task(run_scheduler(mock_telegram, provider, interval_minutes=0))

        # Let two iterations run (both happen almost instantly because interval=0)
        await asyncio.sleep(0.1)
        task.cancel()

        with pytest.raises(asyncio.CancelledError):
            await task

        assert provider.get_bio.await_count >= 2
        # The second call should have succeeded
        mock_telegram.update_bio.assert_awaited_with("ok bio")

    async def test_continues_on_telegram_error(self, mock_telegram: AsyncMock) -> None:
        """If Telegram service raises, the scheduler should NOT crash."""
        provider = AsyncMock()
        provider.get_bio.return_value = "bio"
        mock_telegram.update_bio.side_effect = [Exception("network"), None]

        task = asyncio.create_task(run_scheduler(mock_telegram, provider, interval_minutes=0))

        await asyncio.sleep(0.1)
        task.cancel()

        with pytest.raises(asyncio.CancelledError):
            await task

        assert provider.get_bio.await_count >= 2
