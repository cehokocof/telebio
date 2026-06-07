"""Telegram service — thin wrapper around Telethon for profile operations."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

from telethon import TelegramClient, errors, functions
from telethon.tl.types import Message

if TYPE_CHECKING:
    from telebio.telegram_context import ContextMessage


@dataclass(frozen=True, slots=True)
class _TimelineEvent:
    """One message in a dialog timeline. ``message=None`` → incoming barrier."""

    date: datetime
    message: "ContextMessage | None"

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
        merge_gap_seconds: int = 0,
    ) -> list["ContextMessage"]:
        """Collect recent outgoing text messages across dialogs.

        If ``merge_gap_seconds`` > 0, consecutive own messages in a dialog are
        glued into one ContextMessage when:
            * the gap between neighbours is ≤ merge_gap_seconds, AND
            * no incoming message arrived between them.
        Incoming messages are fetched alongside (no ``from_user`` filter) but
        their text is discarded — only their timestamps act as a barrier.
        """
        from telebio.telegram_context import ContextMessage

        cutoff = datetime.now(UTC) - timedelta(days=days)
        collected: list[ContextMessage] = []
        inspected = 0
        dialogs_seen = 0
        own_seen = 0
        incoming_seen = 0
        merged_away = 0

        logger.info(
            "Collecting context messages (days=%d, limit=%d, dialogs<=%d, "
            "per_dialog<=%d, merge_gap=%ds)",
            days,
            limit,
            dialog_scan_limit,
            per_dialog_limit,
            merge_gap_seconds,
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

            events: list[_TimelineEvent] = []
            dialog_own = 0
            dialog_incoming = 0
            try:
                async for message in self._client.iter_messages(
                    entity,
                    limit=per_dialog_limit,
                ):
                    inspected += 1
                    if not isinstance(message, Message):
                        continue
                    if message.date is None:
                        continue
                    message_date = message.date
                    if message_date.tzinfo is None:
                        message_date = message_date.replace(tzinfo=UTC)
                    message_date = message_date.astimezone(UTC)
                    if message_date < cutoff:
                        break

                    if not message.out:
                        # Incoming — keep as barrier only, text не нужен.
                        events.append(_TimelineEvent(date=message_date, message=None))
                        dialog_incoming += 1
                        continue

                    text = message.raw_text or message.message or ""
                    text = text.strip()
                    if not text:
                        continue

                    msg_peer_id = _message_peer_id(message) or peer_id
                    msg_key = f"{msg_peer_id or 'unknown'}:{message.id}"
                    events.append(
                        _TimelineEvent(
                            date=message_date,
                            message=ContextMessage(
                                message_key=msg_key,
                                message_id=message.id,
                                peer_id=msg_peer_id,
                                dialog_title=str(title),
                                date=message_date,
                                text=text,
                            ),
                        )
                    )
                    dialog_own += 1
            except errors.FloodWaitError:
                raise
            except Exception as exc:
                logger.warning("Skipping dialog %s while collecting context: %s", title, exc)

            own_seen += dialog_own
            incoming_seen += dialog_incoming

            grouped = _group_with_barriers(events, gap_seconds=merge_gap_seconds)
            if merge_gap_seconds > 0:
                merged_away += max(0, dialog_own - len(grouped))

            for message in grouped:
                if len(collected) >= limit:
                    break
                collected.append(message)

            if len(collected) >= limit:
                break

        collected.sort(key=lambda item: (item.date, item.message_key))
        logger.info(
            "Collected %d outgoing context items "
            "(days=%d, dialogs=%d, inspected=%d, own_seen=%d, incoming_seen=%d, "
            "merged_away=%d)",
            len(collected),
            days,
            dialogs_seen,
            inspected,
            own_seen,
            incoming_seen,
            merged_away,
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


def _group_with_barriers(
    events: list[_TimelineEvent],
    *,
    gap_seconds: int,
) -> list["ContextMessage"]:
    """Group own messages from a dialog timeline, breaking on incoming events.

    Two own messages glue iff:
      * no incoming event lies between them, AND
      * either gap_seconds <= 0 is disabled (still глюёт неограниченно), or
        Δt(neighbour) ≤ gap_seconds.
    """
    from telebio.telegram_context import ContextMessage

    if not events:
        return []

    if gap_seconds <= 0:
        # Склейка отключена — возвращаем все own-сообщения как есть.
        return [event.message for event in events if event.message is not None]

    ordered = sorted(events, key=lambda event: event.date)

    groups: list[list["ContextMessage"]] = []
    barrier_pending = True  # before any own message we have no group
    for event in ordered:
        if event.message is None:
            # incoming barrier — закрывает текущую группу
            barrier_pending = True
            continue

        message = event.message
        if barrier_pending or not groups:
            groups.append([message])
            barrier_pending = False
            continue

        previous = groups[-1][-1]
        delta = (message.date - previous.date).total_seconds()
        if gap_seconds > 0 and delta > gap_seconds:
            groups.append([message])
        else:
            groups[-1].append(message)

    merged: list["ContextMessage"] = []
    for group in groups:
        if len(group) == 1:
            merged.append(group[0])
            continue
        first = group[0]
        last = group[-1]
        merged.append(
            ContextMessage(
                message_key=first.message_key,
                message_id=last.message_id,
                peer_id=first.peer_id,
                dialog_title=first.dialog_title,
                date=last.date,
                text="\n".join(part.text for part in group),
            )
        )
    return merged
