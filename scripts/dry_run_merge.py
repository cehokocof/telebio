"""Dry-run: apply _merge_consecutive_outgoing to existing parquet rows.

Reads data/context_prod.parquet, reconstructs ContextMessage-like items per
(peer_id) bucket, runs the merge function with a configurable gap, then
reports:
  - row counts before / after merge
  - per-label breakdown (drop / maybe / keep / unlabeled)
  - sample groups with size >= 2 so you can eyeball the joined text
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from telebio.context_prod import ContextMessage, _from_iso  # noqa: E402
from telebio.services.telegram import _merge_consecutive_outgoing  # noqa: E402


def _row_to_message(row: pd.Series) -> ContextMessage:
    return ContextMessage(
        message_key=str(row["message_key"]),
        message_id=int(row["message_id"]),
        peer_id=None if pd.isna(row["peer_id"]) else int(row["peer_id"]),
        dialog_title=str(row["dialog_title"]),
        date=_from_iso(str(row["message_date"])),
        text=str(row["text"]),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gap", type=int, default=300, help="merge gap in seconds")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=ROOT / "data" / "context_prod.parquet",
    )
    parser.add_argument("--samples", type=int, default=10, help="how many merged groups to show (0 = all)")
    parser.add_argument(
        "--only-pending",
        action="store_true",
        help="restrict to pending rows: label ∈ {maybe, keep} and used_at is NaN",
    )
    parser.add_argument(
        "--min-size",
        type=int,
        default=2,
        help="show only groups of at least this size (default 2 = non-trivial)",
    )
    parser.add_argument(
        "--max-size",
        type=int,
        default=0,
        help="show only groups up to this size (0 = no cap)",
    )
    parser.add_argument(
        "--dialog",
        type=str,
        default="",
        help="filter samples to dialog titles containing this substring (case-insensitive)",
    )
    parser.add_argument(
        "--sort",
        choices=("date", "size", "span"),
        default="date",
        help="sort sample groups by date / size / span (default date)",
    )
    parser.add_argument(
        "--with-labels",
        action="store_true",
        help="annotate each line in the sample with its stored label",
    )
    args = parser.parse_args()

    frame = pd.read_parquet(args.dataset)
    print(f"Dataset: {args.dataset}")
    print(f"Total rows: {len(frame)}")

    if args.only_pending:
        mask = frame["label"].isin(["maybe", "keep"]) & frame["used_at"].isna()
        frame = frame[mask].copy()
        print(f"Filtered to pending rows: {len(frame)}")

    if frame.empty:
        print("Empty frame, nothing to do.")
        return

    label_counts = frame["label"].value_counts(dropna=False).to_dict()
    print(f"Labels: {label_counts}")

    by_peer: dict[object, list[ContextMessage]] = {}
    labels_by_key: dict[str, str] = {}
    for _, row in frame.iterrows():
        peer = None if pd.isna(row["peer_id"]) else int(row["peer_id"])
        by_peer.setdefault(peer, []).append(_row_to_message(row))
        labels_by_key[str(row["message_key"])] = (
            "—" if pd.isna(row["label"]) else str(row["label"])
        )

    before = sum(len(items) for items in by_peer.values())
    merged_all: list[tuple[list[ContextMessage], ContextMessage]] = []
    for items in by_peer.values():
        ordered = sorted(items, key=lambda m: (m.date, m.message_id))
        # та же группировка что в _merge_consecutive_outgoing — но мы храним сами группы
        groups: list[list[ContextMessage]] = []
        for message in ordered:
            if groups:
                previous = groups[-1][-1]
                delta = (message.date - previous.date).total_seconds()
                if 0 <= delta <= args.gap and args.gap > 0:
                    groups[-1].append(message)
                    continue
            groups.append([message])
        merged = _merge_consecutive_outgoing(ordered, gap_seconds=args.gap)
        assert len(groups) == len(merged), "group reconstruction mismatch"
        merged_all.extend(zip(groups, merged))

    after = len(merged_all)
    multi = [(g, m) for g, m in merged_all if len(g) >= args.min_size]
    if args.max_size > 0:
        multi = [(g, m) for g, m in multi if len(g) <= args.max_size]
    if args.dialog:
        needle = args.dialog.lower()
        multi = [(g, m) for g, m in multi if needle in m.dialog_title.lower()]

    if args.sort == "size":
        multi.sort(key=lambda pair: len(pair[0]), reverse=True)
    elif args.sort == "span":
        multi.sort(
            key=lambda pair: (pair[0][-1].date - pair[0][0].date).total_seconds(),
            reverse=True,
        )

    # сводка по всем нетривиальным группам (>=2) до фильтров — для honest stats
    all_multi = [(g, m) for g, m in merged_all if len(g) > 1]

    print()
    print(f"gap_seconds   : {args.gap}")
    print(f"rows before   : {before}")
    print(f"rows after    : {after}")
    print(f"compression   : {after / before * 100:.1f}%  ({before - after} rows merged away)")
    print(f"non-trivial groups (size>=2): {len(all_multi)}")
    if all_multi:
        sizes = [len(g) for g, _ in all_multi]
        buckets = {"2": 0, "3-4": 0, "5-9": 0, "10-19": 0, "20+": 0}
        for s in sizes:
            if s == 2:
                buckets["2"] += 1
            elif s <= 4:
                buckets["3-4"] += 1
            elif s <= 9:
                buckets["5-9"] += 1
            elif s <= 19:
                buckets["10-19"] += 1
            else:
                buckets["20+"] += 1
        print(f"group sizes  : min={min(sizes)}, max={max(sizes)}, mean={sum(sizes)/len(sizes):.1f}")
        print(f"distribution : " + "  ".join(f"{k}:{v}" for k, v in buckets.items()))

    print()
    filter_desc = []
    if args.min_size != 2:
        filter_desc.append(f"min_size>={args.min_size}")
    if args.max_size > 0:
        filter_desc.append(f"max_size<={args.max_size}")
    if args.dialog:
        filter_desc.append(f"dialog~={args.dialog!r}")
    if args.sort != "date":
        filter_desc.append(f"sort={args.sort}")
    suffix = f"  [{', '.join(filter_desc)}]" if filter_desc else ""
    cap = args.samples if args.samples > 0 else len(multi)
    print(f"=== sample merged groups: {len(multi)} match, showing up to {cap}{suffix} ===")

    for i, (group, merged_msg) in enumerate(multi[:cap], 1):
        span = (group[-1].date - group[0].date).total_seconds()
        print(
            f"\n[{i}] peer={merged_msg.peer_id}  dialog={merged_msg.dialog_title!r}  "
            f"size={len(group)}  span={span:.0f}s"
        )
        for j, part in enumerate(group, 1):
            preview = part.text.replace("\n", " ↵ ")
            if len(preview) > 120:
                preview = preview[:117] + "…"
            label_tag = ""
            if args.with_labels:
                label_tag = f" [{labels_by_key.get(part.message_key, '—'):>5}]"
            print(f"    {j}.{label_tag} [{part.date:%Y-%m-%d %H:%M:%S}] {preview}")
        merged_preview = merged_msg.text.replace("\n", " ↵ ")
        if len(merged_preview) > 200:
            merged_preview = merged_preview[:197] + "…"
        print(f"    → merged: {merged_preview}")


if __name__ == "__main__":
    main()
