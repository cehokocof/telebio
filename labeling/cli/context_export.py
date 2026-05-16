"""CLI for exporting real Telegram context history to a JSON dataset."""

from __future__ import annotations

import argparse
import asyncio
import logging
from pathlib import Path

from telebio.config import load_settings
from labeling.core.context_dataset import write_context_fixture
from telebio.services.telegram import TelegramService

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export recent outgoing Telegram messages to a context JSON fixture."
    )
    parser.add_argument(
        "--output",
        default="tests/fixtures/context_messages_live.json",
        help="Output JSON path.",
    )
    parser.add_argument("--days", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--dialogs",
        type=int,
        default=30,
        help="How many recent dialogs to scan.",
    )
    parser.add_argument(
        "--per-dialog",
        type=int,
        default=100,
        help="How many recent messages to inspect per dialog.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level for the export run.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )
    asyncio.run(_async_main(args))


async def _async_main(args: argparse.Namespace) -> None:
    settings = load_settings()
    days = args.days if args.days is not None else settings.context_days
    limit = args.limit if args.limit is not None else settings.context_limit

    async with TelegramService(
        api_id=settings.api_id,
        api_hash=settings.api_hash,
        session_path=settings.session_path,
    ) as telegram:
        messages = await telegram.collect_recent_outgoing_texts(
            days=days,
            limit=limit,
            dialog_scan_limit=args.dialogs,
            per_dialog_limit=args.per_dialog,
        )

    output = Path(args.output)
    write_context_fixture(output, messages)
    logger.info(
        "Exported %d context messages to %s (days=%d, limit=%d, dialogs=%d, per_dialog=%d)",
        len(messages),
        output,
        days,
        limit,
        args.dialogs,
        args.per_dialog,
    )


if __name__ == "__main__":
    main()
