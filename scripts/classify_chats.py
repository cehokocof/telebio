"""One-shot: classify every chat seen in existing parquets via Telegram API.

For each unique ``peer_id`` found in ``data/*.parquet`` files, resolve the
entity through your Telethon session and record:

* ``peer_type``: ``user`` | ``group`` | ``channel``
* ``is_bot``: bool (only meaningful for ``user``)
* ``category``: ``personal`` | ``bot`` | ``group`` | ``channel``

Result is saved to ``data/chat_categories.json``. Idempotent — peers already
in the file are skipped, so re-running just fills the gap for newly fetched
data.

Category rules:
    User(is_bot=False)      -> personal
    User(is_bot=True)       -> bot
    Chat                    -> group        (legacy basic groups)
    Channel(megagroup=True) -> group        (supergroups behave like groups)
    Channel(broadcast=True) -> channel
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd
from telethon import TelegramClient
from telethon.tl.types import Channel, Chat, User

from telebio.config import load_settings

OUTPUT_PATH = ROOT / "data" / "chat_categories.json"
PARQUETS = [
    ROOT / "data" / "context_prod.parquet",
    ROOT / "data" / "dry_run_events.parquet",
    ROOT / "data" / "context_labeling_train.parquet",
    ROOT / "data" / "context_labeling_val_raw.parquet",
]


def _collect_peers() -> dict[int, str]:
    """Return ``{peer_id: dialog_title}`` from every available parquet."""
    peers: dict[int, str] = {}
    for path in PARQUETS:
        if not path.exists():
            continue
        frame = pd.read_parquet(path)
        if "peer_id" not in frame.columns:
            continue
        title_col = "dialog_title" if "dialog_title" in frame.columns else "dialog"
        if title_col not in frame.columns:
            continue
        for _, row in frame[["peer_id", title_col]].drop_duplicates().iterrows():
            peer = row["peer_id"]
            if pd.isna(peer):
                continue
            peer_id = int(peer)
            title = str(row[title_col])
            peers.setdefault(peer_id, title)
    return peers


def _categorize(entity) -> tuple[str, bool, str]:
    if isinstance(entity, User):
        is_bot = bool(getattr(entity, "bot", False))
        return ("user", is_bot, "bot" if is_bot else "personal")
    if isinstance(entity, Chat):
        return ("group", False, "group")
    if isinstance(entity, Channel):
        if getattr(entity, "megagroup", False):
            return ("group", False, "group")
        return ("channel", False, "channel")
    return ("unknown", False, "unknown")


async def _resolve_async(peers: dict[int, str], existing: dict[str, dict]) -> dict[str, dict]:
    settings = load_settings()
    client = TelegramClient(settings.session_path, settings.api_id, settings.api_hash)
    await client.start()
    try:
        # Iter dialogs once — populates entity cache so later get_entity calls
        # don't trigger heavy MTProto lookups.
        print("Warming entity cache via iter_dialogs…")
        seen_via_dialogs: dict[int, object] = {}
        async for dialog in client.iter_dialogs(limit=500):
            entity = getattr(dialog, "entity", None)
            if entity is not None and getattr(entity, "id", None) is not None:
                seen_via_dialogs[int(entity.id)] = entity

        unresolved: list[int] = []
        new_entries = 0
        for peer_id, title in peers.items():
            if str(peer_id) in existing:
                continue
            entity = seen_via_dialogs.get(peer_id)
            if entity is None:
                try:
                    entity = await client.get_entity(peer_id)
                except Exception as exc:
                    unresolved.append(peer_id)
                    print(f"  ! cannot resolve {peer_id} ({title!r}): {exc}")
                    continue
            peer_type, is_bot, category = _categorize(entity)
            existing[str(peer_id)] = {
                "title": title,
                "peer_type": peer_type,
                "is_bot": is_bot,
                "category": category,
            }
            new_entries += 1
            print(f"  + {peer_id:>14}  [{category:<8}] {title}")

        if unresolved:
            print(f"\nUnresolved peers (will be saved as 'unknown'): {len(unresolved)}")
            for peer_id in unresolved:
                existing[str(peer_id)] = {
                    "title": peers[peer_id],
                    "peer_type": "unknown",
                    "is_bot": False,
                    "category": "unknown",
                }
        print(f"\nNew entries this run: {new_entries}")
        return existing
    finally:
        await client.disconnect()


async def _amain() -> None:
    peers = _collect_peers()
    print(f"Collected {len(peers)} unique peer_id(s) from parquet files")

    existing: dict[str, dict] = {}
    if OUTPUT_PATH.exists():
        existing = json.loads(OUTPUT_PATH.read_text(encoding="utf-8"))
        print(f"Loaded existing categories: {len(existing)} peers")

    result = await _resolve_async(peers, existing)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    by_cat: dict[str, int] = {}
    for info in result.values():
        by_cat[info["category"]] = by_cat.get(info["category"], 0) + 1
    print()
    print(f"Saved → {OUTPUT_PATH}")
    print(f"Total peers: {len(result)}")
    for cat, count in sorted(by_cat.items()):
        print(f"  {cat:<10} {count}")


def main() -> None:
    asyncio.run(_amain())


if __name__ == "__main__":
    main()
