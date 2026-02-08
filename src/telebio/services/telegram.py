"""Telegram service — thin wrapper around Telethon for profile operations."""

from __future__ import annotations

import asyncio
import logging

from telethon import TelegramClient, errors, functions

logger = logging.getLogger(__name__)


class TelegramService:
    """Manages a Telethon user-bot session and exposes high-level helpers."""

    def __init__(self, api_id: int, api_hash: str, session_path: str) -> None:
        self._client = TelegramClient(session_path, api_id, api_hash)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Start the client and ensure the user is authorised."""
        logger.info("Connecting to Telegram…")
        await self._client.start()
        me = await self._client.get_me()
        logger.info("Signed in as %s (id=%s)", me.first_name, me.id)

    async def stop(self) -> None:
        """Gracefully disconnect."""
        logger.info("Disconnecting from Telegram…")
        await self._client.disconnect()

    # ------------------------------------------------------------------
    # Bio operations
    # ------------------------------------------------------------------

    async def update_bio(self, text: str) -> None:
        """Set the user's "About" (bio) field.

        Handles:
        - FloodWaitError  → waits the required time then retries once.
        - RPCError        → logs and re-raises.
        """
        try:
            await self._client(
                functions.account.UpdateProfileRequest(about=text)
            )
            logger.info("Bio updated → '%s'", text)

        except errors.FloodWaitError as exc:
            wait = exc.seconds
            logger.warning(
                "FloodWaitError: Telegram asks to wait %d s. Sleeping…", wait
            )
            await asyncio.sleep(wait)
            # One retry after the cooldown
            await self._client(
                functions.account.UpdateProfileRequest(about=text)
            )
            logger.info("Bio updated after flood-wait → '%s'", text)

        except errors.RPCError as exc:
            logger.error("Telegram RPC error while updating bio: %s", exc)
            raise

    # ------------------------------------------------------------------
    # Context manager support
    # ------------------------------------------------------------------

    async def __aenter__(self) -> TelegramService:
        await self.start()
        return self

    async def __aexit__(self, *_exc) -> None:
        await self.stop()
