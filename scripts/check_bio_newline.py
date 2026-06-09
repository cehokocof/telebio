"""Manual check: does Telegram preserve newlines in the account About/bio field?

Sets a temporary two-line bio via ``UpdateProfileRequest``, reads it back via
``GetFullUserRequest``, reports whether the ``\\n`` survived, then restores the
previous bio.

Run from the project root with the existing Telethon session:

    PYTHONPATH=src uv run python scripts/check_bio_newline.py
"""

from __future__ import annotations

import asyncio

from telethon import TelegramClient, functions

from telebio.config import load_settings

TEST_BIO = "- строка один\n- строка два"


async def main() -> None:
    settings = load_settings()
    client = TelegramClient(settings.session_path, settings.api_id, settings.api_hash)
    await client.start()
    try:
        full = await client(functions.users.GetFullUserRequest("me"))
        previous = full.full_user.about or ""
        print(f"Текущее bio: {previous!r}\n")

        await client(functions.account.UpdateProfileRequest(about=TEST_BIO))
        full_after = await client(functions.users.GetFullUserRequest("me"))
        readback = full_after.full_user.about or ""

        print(f"Записали:           {TEST_BIO!r}")
        print(f"Прочитали обратно:  {readback!r}\n")

        if "\n" in readback:
            print("РЕЗУЛЬТАТ: перенос строки СОХРАНЁН — можно делать 2-строчное bio.")
        else:
            print(
                "РЕЗУЛЬТАТ: перенос строки ВЫРЕЗАН Telegram — "
                "нужен видимый разделитель в одну строку."
            )

        await client(functions.account.UpdateProfileRequest(about=previous))
        print(f"\nВосстановили прежнее bio: {previous!r}")
    finally:
        await client.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
