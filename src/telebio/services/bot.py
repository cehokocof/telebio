"""Telegram bot service for managing telebio application."""

from __future__ import annotations

import asyncio
import logging
from collections import deque
from datetime import datetime
from typing import TYPE_CHECKING

from telethon import TelegramClient, events

if TYPE_CHECKING:
    from telebio.providers.base import BioProvider

logger = logging.getLogger(__name__)


class BotService:
    """Manages a Telegram bot for controlling the telebio application."""

    def __init__(
        self,
        bot_token: str,
        api_id: int,
        api_hash: str,
        current_mode: dict[str, str],
    ) -> None:
        """Initialize bot service.
        
        Args:
            bot_token: Telegram bot token from @BotFather
            api_id: Telegram API ID
            api_hash: Telegram API hash
            current_mode: Dict reference to track current bio provider mode
        """
        self._bot = TelegramClient("bot_session", api_id, api_hash)
        self._token = bot_token
        self._current_mode = current_mode
        self._history: deque[dict] = deque(maxlen=10)
        self._last_bio: str = ""
        self._last_update: datetime | None = None
        self._owner_id: int | None = None

    async def start(self, owner_id: int) -> None:
        """Start the bot and register handlers."""
        self._owner_id = owner_id
        logger.info("Starting management bot...")
        await self._bot.start(bot_token=self._token)
        
        # Register command handlers
        self._bot.add_event_handler(
            self._handle_status,
            events.NewMessage(pattern="/status", from_users=owner_id)
        )
        self._bot.add_event_handler(
            self._handle_history,
            events.NewMessage(pattern="/history", from_users=owner_id)
        )
        self._bot.add_event_handler(
            self._handle_set_mode,
            events.NewMessage(pattern=r"/set_mode (\w+)", from_users=owner_id)
        )
        
        me = await self._bot.get_me()
        logger.info("Management bot started as @%s", me.username)

    async def stop(self) -> None:
        """Stop the bot."""
        logger.info("Stopping management bot...")
        await self._bot.disconnect()

    def record_bio_update(self, bio: str, mode: str) -> None:
        """Record a bio update for history tracking."""
        self._last_bio = bio
        self._last_update = datetime.now()
        self._history.append({
            "bio": bio,
            "mode": mode,
            "timestamp": self._last_update.strftime("%Y-%m-%d %H:%M:%S"),
        })
        logger.debug("Recorded bio update: %s", bio)

    async def _handle_status(self, event: events.NewMessage.Event) -> None:
        """Handle /status command."""
        mode = self._current_mode.get("mode", "unknown")
        status_lines = [
            "🤖 <b>TeleBio Status</b>",
            "",
            f"📊 <b>Mode:</b> <code>{mode}</code>",
            f"📝 <b>Current bio:</b> {self._last_bio or '(none)'}",
        ]
        
        if self._last_update:
            status_lines.append(
                f"🕐 <b>Last update:</b> {self._last_update.strftime('%Y-%m-%d %H:%M:%S')}"
            )
        
        await event.respond("\n".join(status_lines), parse_mode="html")

    async def _handle_history(self, event: events.NewMessage.Event) -> None:
        """Handle /history command."""
        if not self._history:
            await event.respond("📜 No history available yet.")
            return
        
        history_lines = ["📜 <b>Recent Bio Updates:</b>", ""]
        for i, entry in enumerate(reversed(self._history), 1):
            history_lines.append(
                f"{i}. [{entry['timestamp']}] <code>{entry['mode']}</code>\n"
                f"   {entry['bio']}"
            )
        
        await event.respond("\n\n".join(history_lines), parse_mode="html")

    async def _handle_set_mode(self, event: events.NewMessage.Event) -> None:
        """Handle /set_mode command."""
        mode = event.pattern_match.group(1).lower()
        
        if mode not in ("list", "llm"):
            await event.respond(
                "❌ Invalid mode. Use <code>list</code> or <code>llm</code>.",
                parse_mode="html"
            )
            return
        
        current = self._current_mode.get("mode", "")
        if mode == current:
            await event.respond(f"ℹ️ Already in <code>{mode}</code> mode.", parse_mode="html")
            return
        
        # Update mode reference
        self._current_mode["mode"] = mode
        
        await event.respond(
            f"✅ Mode switched to <code>{mode}</code>\n"
            f"Next bio update will use the new provider.",
            parse_mode="html"
        )
        logger.info("Mode switched to '%s' via bot command", mode)

    async def __aenter__(self) -> BotService:
        return self

    async def __aexit__(self, *_exc) -> None:
        await self.stop()
