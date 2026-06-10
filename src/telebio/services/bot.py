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
    from telebio.prompts import Prompt
    from telebio.providers.base import BioProvider
    from telebio.services.state_store import StateStore
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
        prompts: list[Prompt] | None = None,
        store: StateStore | None = None,
    ) -> None:
        """Initialize bot service.

        Args:
            bot_token: Telegram bot token from @BotFather
            api_id: Telegram API ID
            api_hash: Telegram API hash
            current_mode: Dict reference to track current mode and active prompt
            telegram: Telegram service for updating bio
            provider_factory: Factory to build a provider by mode name
            prompts: Named prompts available for llm_prompt_generation
            store: Optional SQLite-backed persistence for state and history
        """
        self._bot = TelegramClient("bot_session", api_id, api_hash)
        self._token = bot_token
        self._current_mode = current_mode
        self._telegram = telegram
        self._provider_factory = provider_factory
        self._prompts: list[Prompt] = prompts or []
        self._history: deque[dict] = deque()
        self._last_bio: str = ""
        self._last_update: datetime | None = None
        self._owner_id: int | None = None
        self._paused: bool = False
        self._store = store
        if store is not None:
            self._restore_from_store(store)

    def _restore_from_store(self, store: StateStore) -> None:
        settings = store.load_settings()
        if "mode" in settings:
            self._current_mode["mode"] = settings["mode"]
        if "prompt_name" in settings:
            self._current_mode["prompt_name"] = settings["prompt_name"]
        self._paused = settings.get("paused", "0") == "1"
        self._last_bio = settings.get("last_bio", "")
        last_update_str = settings.get("last_update")
        if last_update_str:
            try:
                self._last_update = datetime.strptime(
                    last_update_str, "%Y-%m-%d %H:%M:%S"
                )
            except ValueError:
                logger.warning("Bad last_update in store: %r", last_update_str)
        self._history.extend(store.load_history())

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
    def prompts(self) -> list[Prompt]:
        return self._prompts

    @property
    def prompt_name(self) -> str | None:
        return self._current_mode.get("prompt_name")

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
        if self._store is not None:
            self._store.append_bio(bio=bio, mode=mode, ts=self._last_update)
            self._store.save_setting("last_bio", bio)
            self._store.save_setting(
                "last_update", self._last_update.strftime("%Y-%m-%d %H:%M:%S")
            )
        logger.debug("Recorded bio update: %s", bio)

    def toggle_pause(self) -> None:
        """Flip the paused flag."""
        self._paused = not self._paused
        if self._store is not None:
            self._store.save_setting("paused", "1" if self._paused else "0")

    def set_prompt(self, name: str) -> None:
        """Set the active named prompt for llm_prompt_generation."""
        self._current_mode["prompt_name"] = name
        if self._store is not None:
            self._store.save_setting("prompt_name", name)

    def set_mode(self, mode: str) -> None:
        """Switch the active bio-provider mode and persist it."""
        self._current_mode["mode"] = mode
        if self._store is not None:
            self._store.save_setting("mode", mode)

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    async def __aenter__(self) -> BotService:
        return self

    async def __aexit__(self, *_exc) -> None:
        await self.stop()
