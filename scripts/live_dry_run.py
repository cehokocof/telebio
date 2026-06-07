"""Analyse cached Telegram events under different gap settings.

Reads events previously dumped by ``scripts/fetch_events.py`` and runs the
new collection logic locally — no Telegram round trips. Lets you iterate on
``--gap`` and other knobs cheaply.

Shows:
  * per-fetch totals (own / incoming)
  * how many ``ContextMessage`` you get under TWO strategies:
      - barrier-on (new): incoming reply closes a group
      - barrier-off (old): only gap_seconds matters
  * groups where the new barrier saved you from a bad merge (--show-saved)
  * largest groups so you can eyeball quality

Usage:
  uv run python scripts/fetch_events.py          # one-time slow part
  uv run python scripts/live_dry_run.py [--gap N] [--samples N] \\
      [--dialog SUBSTRING] [--show-saved]
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from telebio.config import load_settings  # noqa: E402
from telebio.telegram_context import ContextMessage, _from_iso  # noqa: E402
from telebio.services.telegram import (  # noqa: E402
    _TimelineEvent,
    _group_with_barriers,
)


def _load_events(path: Path, dialog_filter: str) -> dict[tuple, list[_TimelineEvent]]:
    if not path.exists():
        sys.exit(
            f"Cache not found: {path}\n"
            "Run `uv run python scripts/fetch_events.py` first."
        )
    frame = pd.read_parquet(path)
    if dialog_filter:
        needle = dialog_filter.lower()
        frame = frame[frame["dialog_title"].str.lower().str.contains(needle, na=False)]
    by_dialog: dict[tuple, list[_TimelineEvent]] = defaultdict(list)
    for _, row in frame.iterrows():
        peer_id = None if pd.isna(row["peer_id"]) else int(row["peer_id"])
        title = str(row["dialog_title"])
        key = (peer_id, title)
        date = _from_iso(str(row["date"]))
        if bool(row["is_own"]):
            msg_peer = peer_id
            event = _TimelineEvent(
                date=date,
                message=ContextMessage(
                    message_key=f"{msg_peer or 'unknown'}:{int(row['message_id'])}",
                    message_id=int(row["message_id"]),
                    peer_id=msg_peer,
                    dialog_title=title,
                    date=date,
                    text=str(row["text"]),
                ),
            )
        else:
            event = _TimelineEvent(date=date, message=None)
        by_dialog[key].append(event)
    # ensure chronological order per dialog
    for key in by_dialog:
        by_dialog[key].sort(key=lambda e: e.date)
    return by_dialog


def _group_without_barriers(
    events: list[_TimelineEvent], *, gap_seconds: int
) -> list[ContextMessage]:
    own_only = [e for e in events if e.message is not None]
    return _group_with_barriers(own_only, gap_seconds=gap_seconds)


def _build_groups(messages: list[ContextMessage]) -> list[list[ContextMessage]]:
    groups: list[list[ContextMessage]] = []
    for merged in messages:
        if "\n" in merged.text:
            parts = merged.text.split("\n")
            groups.append([_synthetic(merged, text=part) for part in parts])
        else:
            groups.append([merged])
    return groups


def _synthetic(template: ContextMessage, *, text: str) -> ContextMessage:
    return ContextMessage(
        message_key=template.message_key,
        message_id=template.message_id,
        peer_id=template.peer_id,
        dialog_title=template.dialog_title,
        date=template.date,
        text=text,
    )


def _format_group(group: list[ContextMessage], indent: str = "    ") -> str:
    lines = []
    for part in group:
        preview = part.text.replace("\n", " ↵ ")
        if len(preview) > 120:
            preview = preview[:117] + "…"
        lines.append(f"{indent}[{part.date:%Y-%m-%d %H:%M:%S}] {preview}")
    return "\n".join(lines)


def main() -> None:
    settings = load_settings()
    parser = argparse.ArgumentParser()
    parser.add_argument("--gap", type=int, default=settings.telegram_context_merge_gap_seconds)
    parser.add_argument(
        "--cache", type=Path,
        default=ROOT / "data" / "dry_run_events.parquet",
        help="parquet cache produced by fetch_events.py",
    )
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--min-size", type=int, default=2)
    parser.add_argument("--dialog", type=str, default="")
    parser.add_argument(
        "--show-saved", action="store_true",
        help="show concrete groups where the barrier prevented a bad merge",
    )
    parser.add_argument(
        "--show-incoming", action="store_true",
        help="when listing rescued groups, also print the incoming replies between them",
    )
    args = parser.parse_args()

    by_dialog = _load_events(args.cache, args.dialog)
    if not by_dialog:
        print(f"No events match filters (cache={args.cache}, dialog={args.dialog!r}).")
        return

    print(f"Cache  : {args.cache}")
    print(f"Params : gap={args.gap}s")
    if args.dialog:
        print(f"Filter : dialog~={args.dialog!r}")
    print()

    total_own = sum(
        sum(1 for e in events if e.message is not None) for events in by_dialog.values()
    )
    total_incoming = sum(
        sum(1 for e in events if e.message is None) for events in by_dialog.values()
    )
    print(f"Dialogs touched : {len(by_dialog)}")
    print(f"Own messages    : {total_own}")
    print(f"Incoming (barrier source) : {total_incoming}")
    print()

    all_with_barrier: list[ContextMessage] = []
    all_without_barrier: list[ContextMessage] = []
    saved_examples: list[tuple[str, int, int, list[list[ContextMessage]], list[_TimelineEvent]]] = []

    for (_, title), events in by_dialog.items():
        with_barrier = _group_with_barriers(events, gap_seconds=args.gap)
        without_barrier = _group_without_barriers(events, gap_seconds=args.gap)
        all_with_barrier.extend(with_barrier)
        all_without_barrier.extend(without_barrier)

        if len(with_barrier) > len(without_barrier):
            saved_examples.append((
                title, len(without_barrier), len(with_barrier),
                _build_groups(with_barrier), events,
            ))

    print(f"Merged ContextMessages with barrier (NEW): {len(all_with_barrier)}")
    print(f"Merged ContextMessages no barrier  (OLD): {len(all_without_barrier)}")
    if total_own:
        print(
            f"Compression vs raw own : "
            f"{len(all_with_barrier) / total_own * 100:.1f}% (NEW), "
            f"{len(all_without_barrier) / total_own * 100:.1f}% (OLD)"
        )
    extra = len(all_with_barrier) - len(all_without_barrier)
    print(
        f"Groups that barrier rescued from merging: {len(saved_examples)} "
        f"dialogs (extra +{extra} entries)"
    )
    print()

    groups_new = _build_groups(all_with_barrier)
    sizes = [len(g) for g in groups_new]
    buckets = {"1": 0, "2": 0, "3-4": 0, "5-9": 0, "10-19": 0, "20+": 0}
    for s in sizes:
        if s == 1:
            buckets["1"] += 1
        elif s == 2:
            buckets["2"] += 1
        elif s <= 4:
            buckets["3-4"] += 1
        elif s <= 9:
            buckets["5-9"] += 1
        elif s <= 19:
            buckets["10-19"] += 1
        else:
            buckets["20+"] += 1
    print("Group size distribution (NEW): " + "  ".join(f"{k}:{v}" for k, v in buckets.items()))
    print()

    if args.show_saved and saved_examples:
        print(f"=== groups rescued by barrier (up to {args.samples}) ===")
        for i, (title, before, after, groups_after, events) in enumerate(
            saved_examples[: args.samples], 1
        ):
            print(f"\n[{i}] {title!r}: {before} → {after} group(s) after barrier")
            if args.show_incoming:
                _print_timeline_with_incoming(groups_after, events)
            else:
                for j, group_after in enumerate(groups_after, 1):
                    print(f"  group {j} (size={len(group_after)}):")
                    print(_format_group(group_after, indent="    "))

    largest = sorted(
        ((idx, group) for idx, group in enumerate(groups_new) if len(group) >= args.min_size),
        key=lambda pair: len(pair[1]),
        reverse=True,
    )
    print()
    print(f"=== largest groups (NEW, ≥{args.min_size}, up to {args.samples}) ===")
    for i, (_, group) in enumerate(largest[: args.samples], 1):
        title = group[0].dialog_title
        span = (group[-1].date - group[0].date).total_seconds()
        print(f"\n[{i}] {title!r}  size={len(group)}  span={span:.0f}s")
        print(_format_group(group))


def _print_timeline_with_incoming(
    groups_after: list[list[ContextMessage]],
    events: list[_TimelineEvent],
) -> None:
    """Print rescued groups interleaved with the incoming replies that split them."""
    # Walk events chronologically; render own messages by group, incoming as separator.
    group_idx = 0
    pos_in_group = 0
    pending_label = True
    last_incoming: datetime | None = None
    for event in events:
        if event.message is None:
            last_incoming = event.date
            continue
        if pending_label:
            print(f"  group {group_idx + 1} (size={len(groups_after[group_idx])}):")
            if last_incoming is not None and group_idx > 0:
                print(f"    ↩ incoming reply at "
                      f"{last_incoming:%Y-%m-%d %H:%M:%S} closed previous group")
            pending_label = False
            last_incoming = None
        msg = event.message
        preview = msg.text.replace("\n", " ↵ ")
        if len(preview) > 120:
            preview = preview[:117] + "…"
        print(f"    [{msg.date:%Y-%m-%d %H:%M:%S}] {preview}")
        pos_in_group += 1
        if pos_in_group >= len(groups_after[group_idx]):
            group_idx += 1
            pos_in_group = 0
            pending_label = True
            if group_idx >= len(groups_after):
                break


if __name__ == "__main__":
    main()
