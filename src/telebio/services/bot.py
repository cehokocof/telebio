"""Telegram bot service for managing telebio application."""

from __future__ import annotations

import logging
from collections import deque
from collections.abc import Callable
from datetime import datetime
from typing import TYPE_CHECKING

from telethon import TelegramClient

from telebio.services.handlers import register_all

if TYPE_CHECKING:
    from telebio.providers.base import BioProvider
    from telebio.services.telegram import TelegramService

logger = logging.getLogger(__name__)


class BotService:
    """Manages a Telegram bot for controlling the telebio application."""

    def __init__(
        self,
        bot_token: str,
        api_id: int,
        api_hash: str,
        current_mode: dict[str, str],
        telegram: TelegramService | None = None,
        provider_factory: Callable[[str], BioProvider] | None = None,
    ) -> None:
        """Initialize bot service.

        Args:
            bot_token: Telegram bot token from @BotFather
            api_id: Telegram API ID
            api_hash: Telegram API hash
            current_mode: Dict reference to track current bio provider mode
            telegram: Telegram service for updating bio
            provider_factory: Factory to build a provider by mode name
        """
        self._bot = TelegramClient("bot_session", api_id, api_hash)
        self._token = bot_token
        self._current_mode = current_mode
        self._telegram = telegram
        self._provider_factory = provider_factory
        self._history: deque[dict] = deque(maxlen=10)
        self._last_bio: str = ""
        self._last_update: datetime | None = None
        self._owner_id: int | None = None
        self._paused: bool = False

    # ------------------------------------------------------------------
    # Public read-only properties used by handlers
    # ------------------------------------------------------------------

    @property
    def current_mode(self) -> dict[str, str]:
        return self._current_mode

    @property
    def telegram(self) -> TelegramService | None:
        return self._telegram

    @property
    def provider_factory(self) -> Callable[[str], BioProvider] | None:
        return self._provider_factory

    @property
    def last_bio(self) -> str:
        return self._last_bio

    @property
    def last_update(self) -> datetime | None:
        return self._last_update

    @property
    def history(self) -> deque[dict]:
        return self._history

    @property
    def paused(self) -> bool:
        """Whether automatic bio updates are paused."""
        return self._paused

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self, owner_id: int) -> None:
        """Start the bot and register handlers."""
        self._owner_id = owner_id
        logger.info("Starting management bot...")
        await self._bot.start(bot_token=self._token)

        register_all(self._bot, self, owner_id)

        me = await self._bot.get_me()
        logger.info("Management bot started as @%s", me.username)

    async def stop(self) -> None:
        """Stop the bot."""
        logger.info("Stopping management bot...")
        await self._bot.disconnect()

    # ------------------------------------------------------------------
    # Mutations called by handlers
    # ------------------------------------------------------------------

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

    def toggle_pause(self) -> None:
        """Flip the paused flag."""
        self._paused = not self._paused

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    async def __aenter__(self) -> BotService:
        return self

    async def __aexit__(self, *_exc) -> None:
        await self.stop()
