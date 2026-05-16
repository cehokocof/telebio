"""Telegram service — thin wrapper around Telethon for profile operations."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from telethon import TelegramClient, errors, functions

logger = logging.getLogger(__name__)

_CONTEXT_DIALOG_SCAN_LIMIT = 10
_CONTEXT_PER_DIALOG_SCAN_LIMIT = 50


@dataclass(frozen=True, slots=True)
class ContextMessage:
    """Outgoing Telegram message selected for context generation."""

    date: datetime
    dialog: str
    text: str


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

    async def collect_recent_outgoing_texts(
        self,
        *,
        days: int,
        limit: int,
        dialog_scan_limit: int | None = None,
        per_dialog_limit: int | None = None,
    ) -> list[ContextMessage]:
        """Collect recent outgoing text messages for context generation.

        Returns messages in chronological order (oldest to newest), capped by
        ``limit`` and restricted to the last ``days`` days.
        """
        if days <= 0:
            raise ValueError("days must be positive")
        if limit <= 0:
            raise ValueError("limit must be positive")

        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        collected: list[ContextMessage] = []
        dialog_limit = min(dialog_scan_limit or _CONTEXT_DIALOG_SCAN_LIMIT, max(1, limit))
        message_limit = min(per_dialog_limit or _CONTEXT_PER_DIALOG_SCAN_LIMIT, limit)

        logger.info(
            "Collecting context messages (days=%d, limit=%d, dialogs<=%d, per_dialog<=%d)",
            days,
            limit,
            dialog_limit,
            message_limit,
        )

        dialogs = await self._client.get_dialogs(limit=dialog_limit)
        logger.debug("Loaded %d dialogs for context scan", len(dialogs))

        scanned_dialogs = 0
        inspected_messages = 0
        for dialog in dialogs:
            entity = getattr(dialog, "entity", None)
            if entity is None:
                logger.debug("Skipping dialog without entity: %r", dialog)
                continue

            scanned_dialogs += 1
            dialog_title = _get_dialog_title(dialog, entity)
            dialog_collected = 0
            dialog_inspected = 0
            stopped_by_cutoff = False

            async for message in self._client.iter_messages(
                entity,
                limit=message_limit,
                wait_time=1,
            ):
                inspected_messages += 1
                dialog_inspected += 1

                if not getattr(message, "out", False):
                    continue

                message_date = message.date
                if message_date is None:
                    continue
                if message_date.tzinfo is None:
                    message_date = message_date.replace(tzinfo=timezone.utc)
                if message_date < cutoff:
                    stopped_by_cutoff = True
                    break

                text = getattr(message, "raw_text", None)
                if text is None:
                    text = getattr(message, "message", None)
                if not isinstance(text, str):
                    continue
                text = text.strip()
                if not text:
                    continue
                collected.append(
                    ContextMessage(
                        date=message_date,
                        dialog=dialog_title,
                        text=text,
                    )
                )
                dialog_collected += 1

            logger.debug(
                "Context scan dialog %d/%d title=%r inspected=%d collected=%d cutoff=%s",
                scanned_dialogs,
                len(dialogs),
                dialog_title,
                dialog_inspected,
                dialog_collected,
                stopped_by_cutoff,
            )

        collected.sort(key=lambda message: message.date, reverse=True)
        messages = sorted(collected[:limit], key=lambda message: message.date)
        logger.info(
            "Collected %d outgoing messages for context (days=%d, limit=%d, dialogs=%d, inspected=%d)",
            len(messages),
            days,
            limit,
            scanned_dialogs,
            inspected_messages,
        )
        for index, message in enumerate(messages, 1):
            logger.info(
                "Context message %03d | %s | %s | %s",
                index,
                message.date.isoformat(timespec="seconds"),
                message.dialog,
                message.text,
            )
        return messages

    # ------------------------------------------------------------------
    # Context manager support
    # ------------------------------------------------------------------

    async def __aenter__(self) -> TelegramService:
        await self.start()
        return self

    async def __aexit__(self, *_exc) -> None:
        await self.stop()


def _get_dialog_title(dialog: object, entity: object) -> str:
    """Build a readable dialog label for context debug logs."""
    title = getattr(dialog, "title", None) or getattr(entity, "title", None)
    if title:
        return str(title)

    first_name = getattr(entity, "first_name", None)
    last_name = getattr(entity, "last_name", None)
    full_name = " ".join(part for part in (first_name, last_name) if part)
    if full_name:
        return full_name

    username = getattr(entity, "username", None)
    if username:
        return f"@{username}"

    entity_id = getattr(entity, "id", None)
    return f"id={entity_id}" if entity_id else "unknown"
