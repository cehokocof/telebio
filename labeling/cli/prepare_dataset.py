"""Unified CLI: prepare a labeling-ready parquet from any source schema.

Replaces the previous trio of build scripts (``labeling/cli/build_dataset.py``,
``scripts/build_labeling_dataset.py``, ``scripts/build_val_dataset.py``).
Pick a source parquet, the rest is auto-detected.

Examples
--------
Live production dump → validation set (drop train-overlap, filter by date)::

    uv run python -m labeling.cli.prepare_dataset \\
        --source data/context_prod.parquet \\
        --output data/context_labeling_val_raw.parquet \\
        --filter-after 2026-05-17T00:00:00+00:00 \\
        --dedup-against data/context_labeling_train.parquet

Raw Telegram fetch → grouped labeling candidate::

    uv run python -m labeling.cli.prepare_dataset \\
        --source data/dry_run_events.parquet \\
        --output data/context_labeling_dryrun.parquet

Smoke test without heavy scoring::

    uv run python -m labeling.cli.prepare_dataset \\
        --source data/context_prod.parquet \\
        --output /tmp/smoke.parquet \\
        --no-scores --limit 50
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path

_SRC = Path(__file__).resolve().parents[2] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import pandas as pd

from labeling.core.scoring import compute_embedding_and_nli, compute_heuristic
from labeling.core.source_adapters import adapt, detect_source_type

_DEFAULT_CATEGORIES_PATH = Path(__file__).resolve().parents[2] / "data" / "chat_categories.json"
_DEFAULT_EXCLUDE = "bot,channel"


def _backup_if_has_labels(output: Path) -> None:
    """Refuse to silently clobber a parquet that already has manual labels.

    If the output file exists and has any non-null ``label``/``true_state``,
    copy it to ``<stem>.bak.<utc>.parquet`` before the run overwrites it.
    """
    if not output.exists():
        return
    try:
        existing = pd.read_parquet(output)
    except Exception:
        return
    has_labels = (
        ("label" in existing.columns and existing["label"].notna().any())
        or ("true_state" in existing.columns and existing["true_state"].notna().any())
    )
    if not has_labels:
        return
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    backup = output.with_name(f"{output.stem}.bak.{stamp}{output.suffix}")
    shutil.copy2(output, backup)
    n = int(existing.get("label", pd.Series(dtype=object)).notna().sum())
    print(f"⚠️  Found {n} existing manual labels in {output.name}; backed up to {backup.name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", type=Path, required=True, help="Input parquet.")
    parser.add_argument("--output", type=Path, required=True, help="Output parquet.")
    parser.add_argument(
        "--source-type",
        choices=("auto", "prod", "raw_events", "labeling"),
        default="auto",
        help="Force a source schema. Default auto-detects from columns.",
    )
    parser.add_argument(
        "--filter-after",
        help="ISO datetime; rows with the source date column < this are dropped.",
    )
    parser.add_argument(
        "--dedup-against",
        type=Path,
        help="Labeling parquet (usually the train set) whose texts are removed.",
    )
    parser.add_argument(
        "--group-gap-seconds",
        type=int,
        default=1800,
        help="Neighbor-merge sanity cap for raw_events source (seconds).",
    )
    parser.add_argument(
        "--categories",
        type=Path,
        default=_DEFAULT_CATEGORIES_PATH,
        help=(
            "JSON map {peer_id: {category, …}} produced by classify_chats.py. "
            "Missing file → all rows tagged 'unknown' and no category filter."
        ),
    )
    parser.add_argument(
        "--exclude-categories",
        default=_DEFAULT_EXCLUDE,
        help=(
            "Comma-separated chat_category values to drop. Default: 'bot,channel'. "
            "Pass empty string to disable category filtering."
        ),
    )
    parser.add_argument("--limit", type=int, help="Cap row count after dedup.")
    parser.add_argument(
        "--no-scores",
        action="store_true",
        help="Skip embedding_score / nli_score (heuristic only, fast).",
    )
    parser.add_argument(
        "--nli",
        action="store_true",
        help="Also compute nli_score (heavy; off by default).",
    )
    args = parser.parse_args()

    if not args.source.exists():
        sys.exit(f"Source not found: {args.source}")

    raw = pd.read_parquet(args.source)
    print(f"Source: {args.source}  rows={len(raw)}")

    source_type = (
        args.source_type if args.source_type != "auto" else detect_source_type(raw)
    )
    print(f"Source type: {source_type}")

    if args.filter_after:
        date_col = "message_date" if source_type == "prod" else "date"
        before = len(raw)
        raw = raw[raw[date_col].astype(str) >= args.filter_after]
        print(
            f"After --filter-after {args.filter_after}: {len(raw)} "
            f"(dropped {before - len(raw)})"
        )

    categories: dict[int, str] = {}
    if args.categories and args.categories.exists():
        raw_map = json.loads(args.categories.read_text(encoding="utf-8"))
        categories = {int(k): v["category"] for k, v in raw_map.items()}
        print(f"Loaded categories: {len(categories)} peers from {args.categories.name}")
    else:
        print(f"No categories file at {args.categories} — all rows will be 'unknown'.")

    frame = adapt(
        raw,
        source_type=source_type,
        group_gap_seconds=args.group_gap_seconds,
        categories=categories,
    )
    frame = frame.drop_duplicates("message_id", keep="last").reset_index(drop=True)
    print(f"After adapter + dedup by message_id: {len(frame)}")

    excluded = {c.strip() for c in args.exclude_categories.split(",") if c.strip()}
    if excluded:
        before = len(frame)
        by_cat = frame["chat_category"].value_counts(dropna=False).to_dict()
        frame = frame[~frame["chat_category"].isin(excluded)].reset_index(drop=True)
        dropped = before - len(frame)
        print(
            f"After --exclude-categories {sorted(excluded)}: {len(frame)} "
            f"(dropped {dropped})"
        )
        print(f"  pre-filter distribution: {by_cat}")
        print(f"  post-filter distribution: "
              f"{frame['chat_category'].value_counts(dropna=False).to_dict()}")

    if args.dedup_against:
        if not args.dedup_against.exists():
            sys.exit(f"--dedup-against parquet not found: {args.dedup_against}")
        train = pd.read_parquet(args.dedup_against)
        train_texts = set(train["text"].astype(str).tolist())
        before = len(frame)
        frame = frame[~frame["text"].astype(str).isin(train_texts)].reset_index(drop=True)
        print(
            f"After --dedup-against {args.dedup_against.name}: {len(frame)} "
            f"(dropped {before - len(frame)})"
        )

    if args.limit and args.limit < len(frame):
        frame = frame.head(args.limit).reset_index(drop=True)
        print(f"Capped to --limit {args.limit}")

    if frame.empty:
        print("Nothing to write.")
        return

    print("Computing heuristic_score…")
    compute_heuristic(frame)
    if not args.no_scores:
        suffix = " + nli_score" if args.nli else ""
        print(f"Computing embedding_score{suffix}…")
        compute_embedding_and_nli(frame, enable_nli=args.nli)

    frame["label"] = frame["label"].astype("Int64")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    _backup_if_has_labels(args.output)
    frame.to_parquet(args.output, index=False)

    size_kb = args.output.stat().st_size / 1024
    print()
    print(f"Saved → {args.output} ({size_kb:.1f} KB, {len(frame)} rows)")
    print()
    print("Next step: label it via the Streamlit UI:")
    print("  uv run streamlit run labeling/ui/app.py")
    print(f"  (in sidebar, set Dataset path to: {args.output})")


if __name__ == "__main__":
    main()
