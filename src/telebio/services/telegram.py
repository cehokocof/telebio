"""Telegram service — thin wrapper around Telethon for profile operations."""

from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime, timedelta

from telethon import TelegramClient, errors, functions
from telethon.tl.types import Message

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
    # Context collection
    # ------------------------------------------------------------------

    async def collect_recent_outgoing_texts(
        self,
        *,
        days: int,
        limit: int,
        dialog_scan_limit: int,
        per_dialog_limit: int,
    ) -> list["ContextMessage"]:
        """Collect recent outgoing text messages across dialogs."""
        from telebio.context_prod import ContextMessage

        cutoff = datetime.now(UTC) - timedelta(days=days)
        collected: list[ContextMessage] = []
        inspected = 0
        dialogs_seen = 0

        logger.info(
            "Collecting context messages (days=%d, limit=%d, dialogs<=%d, per_dialog<=%d)",
            days,
            limit,
            dialog_scan_limit,
            per_dialog_limit,
        )

        async for dialog in self._client.iter_dialogs(limit=dialog_scan_limit):
            dialogs_seen += 1
            entity = getattr(dialog, "entity", None)
            if entity is None:
                continue

            peer_id = getattr(entity, "id", None)
            title = getattr(dialog, "name", None) or getattr(entity, "title", None)
            if not title:
                title = getattr(entity, "username", None) or str(peer_id or "unknown")

            try:
                async for message in self._client.iter_messages(
                    entity,
                    limit=per_dialog_limit,
                    from_user="me",
                ):
                    inspected += 1
                    if len(collected) >= limit:
                        break
                    if not isinstance(message, Message) or not message.out:
                        continue
                    if message.date is None:
                        continue
                    message_date = message.date
                    if message_date.tzinfo is None:
                        message_date = message_date.replace(tzinfo=UTC)
                    message_date = message_date.astimezone(UTC)
                    if message_date < cutoff:
                        break

                    text = message.raw_text or message.message or ""
                    text = text.strip()
                    if not text:
                        continue

                    msg_peer_id = _message_peer_id(message) or peer_id
                    msg_key = f"{msg_peer_id or 'unknown'}:{message.id}"
                    collected.append(
                        ContextMessage(
                            message_key=msg_key,
                            message_id=message.id,
                            peer_id=msg_peer_id,
                            dialog_title=str(title),
                            date=message_date,
                            text=text,
                        )
                    )
            except errors.FloodWaitError:
                raise
            except Exception as exc:
                logger.warning("Skipping dialog %s while collecting context: %s", title, exc)

            if len(collected) >= limit:
                break

        collected.sort(key=lambda item: (item.date, item.message_key))
        logger.info(
            "Collected %d outgoing messages for context (days=%d, dialogs=%d, inspected=%d)",
            len(collected),
            days,
            dialogs_seen,
            inspected,
        )
        return collected

    # ------------------------------------------------------------------
    # Context manager support
    # ------------------------------------------------------------------

    async def __aenter__(self) -> TelegramService:
        await self.start()
        return self

    async def __aexit__(self, *_exc) -> None:
        await self.stop()


def _message_peer_id(message: Message) -> int | None:
    peer = getattr(message, "peer_id", None)
    if peer is None:
        return None
    for attr in ("channel_id", "chat_id", "user_id"):
        value = getattr(peer, attr, None)
        if value is not None:
            return int(value)
    return None
