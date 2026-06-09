"""Handler for the /collect command (collect context without updating bio)."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from telethon import events

from telebio.services.actions import run_collect
from telebio.services.texts import COLLECT_PROGRESS

if TYPE_CHECKING:
    from telebio.services.bot import BotService

logger = logging.getLogger(__name__)


async def handle_collect(event: events.NewMessage.Event, bot: BotService) -> None:
    """Collect and classify context rows into the parquet dataset."""
    await event.respond(COLLECT_PROGRESS)
    await event.respond(await run_collect(bot), parse_mode="html")
