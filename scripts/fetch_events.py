"""Fetch raw timeline events from Telegram once, save to parquet for re-use.

Connects to Telegram via the existing session, walks dialogs the same way
``/collect_context`` does, and writes every relevant message (both own and
incoming) into a local parquet cache. Use ``scripts/live_dry_run.py`` to
analyse this cache with different gap / filter parameters without touching
Telegram again.

Output columns:
    peer_id, dialog_title, message_id, date (ISO), is_own (bool), text,
    is_forward (bool — message.fwd_from is not None),
    is_via_bot (bool — message.via_bot_id is not None)

Note: incoming messages KEEP their text in this cache (it's a local debug
file). In production, incoming text is dropped at collection time.

Usage:
  uv run python scripts/fetch_events.py [--days N] [--dialog-scan-limit N] \\
      [--per-dialog-limit N] [--output PATH]
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from telethon import TelegramClient  # noqa: E402
from telethon.tl.types import Message  # noqa: E402

from telebio.config import load_settings  # noqa: E402
from telebio.services.telegram import _message_peer_id  # noqa: E402


async def _fetch(args, settings) -> list[dict]:
    cutoff = datetime.now(UTC) - timedelta(days=args.days)
    rows: list[dict] = []
    dialogs_seen = 0

    client = TelegramClient(settings.session_path, settings.api_id, settings.api_hash)
    await client.start()
    try:
        print(f"Walking up to {args.dialog_scan_limit} dialogs, "
              f"≤{args.per_dialog_limit} messages each, days={args.days}…")
        async for dialog in client.iter_dialogs(limit=args.dialog_scan_limit):
            dialogs_seen += 1
            entity = getattr(dialog, "entity", None)
            if entity is None:
                continue
            peer_id = getattr(entity, "id", None)
            title = (
                getattr(dialog, "name", None)
                or getattr(entity, "title", None)
                or getattr(entity, "username", None)
                or str(peer_id or "unknown")
            )

            dialog_rows: list[dict] = []
            try:
                async for message in client.iter_messages(entity, limit=args.per_dialog_limit):
                    if not isinstance(message, Message):
                        continue
                    if message.date is None:
                        continue
                    msg_date = message.date
                    if msg_date.tzinfo is None:
                        msg_date = msg_date.replace(tzinfo=UTC)
                    msg_date = msg_date.astimezone(UTC)
                    if msg_date < cutoff:
                        break

                    is_own = bool(message.out)
                    text = (message.raw_text or message.message or "").strip()
                    if is_own and not text:
                        # skip empty own messages (media-only)
                        continue
                    msg_peer_id = _message_peer_id(message) or peer_id
                    dialog_rows.append({
                        "peer_id": int(msg_peer_id) if msg_peer_id is not None else None,
                        "dialog_title": str(title),
                        "message_id": int(message.id),
                        "date": msg_date.isoformat(),
                        "is_own": is_own,
                        "text": text,
                        "is_forward": getattr(message, "fwd_from", None) is not None,
                        "is_via_bot": getattr(message, "via_bot_id", None) is not None,
                    })
            except Exception as exc:
                print(f"  ! skip {title!r}: {exc}")
                continue

            if dialog_rows:
                own = sum(1 for r in dialog_rows if r["is_own"])
                inc = sum(1 for r in dialog_rows if not r["is_own"])
                if own > 0:
                    print(f"  · {title!r}: own={own}, incoming={inc}")
            rows.extend(dialog_rows)
    finally:
        await client.disconnect()

    print(f"\nDialogs seen: {dialogs_seen}")
    print(f"Total rows  : {len(rows)}")
    return rows


async def main() -> None:
    settings = load_settings()
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=settings.context_prod_fetch_days)
    parser.add_argument(
        "--dialog-scan-limit", type=int,
        default=settings.context_prod_dialog_scan_limit,
    )
    parser.add_argument(
        "--per-dialog-limit", type=int,
        default=settings.context_prod_per_dialog_limit,
    )
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "data" / "dry_run_events.parquet",
    )
    args = parser.parse_args()

    rows = await _fetch(args, settings)
    if not rows:
        print("Nothing to write.")
        return

    frame = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(args.output, index=False)

    own_mask = frame["is_own"]
    print(
        "\nOwn breakdown: "
        f"organic={int((own_mask & ~frame['is_forward'] & ~frame['is_via_bot']).sum())}, "
        f"forwarded={int((own_mask & frame['is_forward']).sum())}, "
        f"via_bot={int((own_mask & frame['is_via_bot']).sum())}"
    )
    print(f"Saved → {args.output} ({args.output.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    asyncio.run(main())
