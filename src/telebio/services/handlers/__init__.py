"""Bot command handlers package."""

from __future__ import annotations

from typing import TYPE_CHECKING

from telethon import events

from telebio.services.handlers.status import handle_status
from telebio.services.handlers.history import handle_history
from telebio.services.handlers.set_mode import handle_set_mode
from telebio.services.handlers.new import handle_new
from telebio.services.handlers.pause import handle_pause

if TYPE_CHECKING:
    from telethon import TelegramClient
    from telebio.services.bot import BotService


def register_all(client: TelegramClient, bot: BotService, owner_id: int) -> None:
    """Register every command handler on *client*."""
    client.add_event_handler(
        lambda e: handle_status(e, bot),
        events.NewMessage(pattern="/status", from_users=owner_id),
    )
    client.add_event_handler(
        lambda e: handle_history(e, bot),
        events.NewMessage(pattern="/history", from_users=owner_id),
    )
    client.add_event_handler(
        lambda e: handle_set_mode(e, bot),
        events.NewMessage(pattern=r"/set_mode (\w+)", from_users=owner_id),
    )
    client.add_event_handler(
        lambda e: handle_new(e, bot),
        events.NewMessage(pattern="/new", from_users=owner_id),
    )
    client.add_event_handler(
        lambda e: handle_pause(e, bot),
        events.NewMessage(pattern="/pause", from_users=owner_id),
    )
