"""Bot command handlers package."""

from __future__ import annotations

from typing import TYPE_CHECKING

from telethon import events

from telebio.services.handlers.callbacks import handle_callback
from telebio.services.handlers.collect import handle_collect
from telebio.services.handlers.history import handle_history
from telebio.services.handlers.menu import handle_menu
from telebio.services.handlers.new import handle_new
from telebio.services.handlers.pause import handle_pause
from telebio.services.handlers.set_mode import handle_set_mode
from telebio.services.handlers.start import handle_start
from telebio.services.handlers.status import handle_status

if TYPE_CHECKING:
    from telethon import TelegramClient
    from telebio.services.bot import BotService


def register_all(client: TelegramClient, bot: BotService, owner_id: int) -> None:
    """Register every command and callback handler on *client*."""
    client.add_event_handler(
        lambda e: handle_start(e, bot),
        events.NewMessage(pattern="/start", from_users=owner_id),
    )
    client.add_event_handler(
        lambda e: handle_menu(e, bot),
        events.NewMessage(pattern="/menu", from_users=owner_id),
    )
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
        lambda e: handle_collect(e, bot),
        events.NewMessage(pattern="/collect", from_users=owner_id),
    )
    client.add_event_handler(
        lambda e: handle_pause(e, bot),
        events.NewMessage(pattern="/pause", from_users=owner_id),
    )
    client.add_event_handler(
        lambda e: handle_callback(e, bot),
        events.CallbackQuery(func=lambda e: e.sender_id == owner_id),
    )
