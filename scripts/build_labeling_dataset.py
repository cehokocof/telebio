"""Build a labeling-ready parquet from cached Telegram events.

Pipeline:
  1. Read raw events produced by ``scripts/fetch_events.py``.
  2. Drop forwarded / via-bot messages (configurable).
  3. Group consecutive own messages with the same barrier rule as production
     (``_group_with_barriers``): incoming reply closes a group, ``--gap``
     acts as a sanity cap.
  4. (Optional) Classify each merged ContextMessage with the production
     ``Mix0035Classifier`` (CatBoost stage1 + nearest-centroid stage2).
  5. Write a parquet whose schema is compatible with the labeling UI's
     ``data/context_labeling.parquet`` so an existing labeling tool can
     pick it up directly.

Output parquet columns (labeling-UI compatible):
    message_id (sha-20 hash of date+dialog+text),
    date (ISO), dialog, text, text_hash, text_len, word_count,
    has_link, is_command,
    heuristic_score, nli_score, embedding_score,
    label, label_name, label_source, labeled_at,            (empty by default)
    catboost_label, catboost_label_name, catboost_confidence,
    catboost_proba_drop, catboost_proba_maybe, catboost_proba_keep,
    catboost_model_version, catboost_predicted_at,
Plus per-group extras (for review): peer_id, first_message_id, last_message_id,
group_size, span_seconds, has_forward_in_group (always false after default
filter), has_via_bot_in_group.

Usage:
  uv run python scripts/build_labeling_dataset.py
  uv run python scripts/build_labeling_dataset.py --gap 600 --no-classify
  uv run python scripts/build_labeling_dataset.py --include-forwards --dialog "ярик"
"""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from telebio.config import load_settings  # noqa: E402
from telebio.telegram_context import (  # noqa: E402
    ContextMessage,
    Mix0035Classifier,
    QueuedContextMessage,
    _from_iso,
    _heuristic_score,
    numeric_features,
)
from telebio.services.telegram import (  # noqa: E402
    _TimelineEvent,
    _group_with_barriers,
)

_LABEL_TO_INT = {"drop": 1, "maybe": 2, "keep": 3}
_URL_PATTERN = re.compile(r"https?://", re.IGNORECASE)


def _stable_message_id(*, date: str, dialog: str, text: str) -> str:
    payload = f"{date}\0{dialog}\0{text}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:20]


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:20]


def _load_events(
    cache_path: Path,
    *,
    include_forwards: bool,
    include_via_bot: bool,
    dialog_filter: str,
) -> tuple[dict[tuple, list[_TimelineEvent]], dict[str, int]]:
    """Return events grouped by (peer_id, dialog_title) and a stats dict."""
    if not cache_path.exists():
        sys.exit(
            f"Cache not found: {cache_path}\n"
            "Run `uv run python scripts/fetch_events.py` first."
        )

    frame = pd.read_parquet(cache_path)
    stats = {
        "raw_rows": len(frame),
        "raw_own": int(frame["is_own"].sum()),
        "raw_incoming": int((~frame["is_own"]).sum()),
        "dropped_forwards_own": 0,
        "dropped_via_bot_own": 0,
    }

    if dialog_filter:
        needle = dialog_filter.lower()
        frame = frame[frame["dialog_title"].str.lower().str.contains(needle, na=False)]

    own_mask = frame["is_own"]
    if not include_forwards and "is_forward" in frame.columns:
        drop_mask = own_mask & frame["is_forward"].fillna(False)
        stats["dropped_forwards_own"] = int(drop_mask.sum())
        frame = frame[~drop_mask]
    if not include_via_bot and "is_via_bot" in frame.columns:
        drop_mask = frame["is_own"] & frame["is_via_bot"].fillna(False)
        stats["dropped_via_bot_own"] = int(drop_mask.sum())
        frame = frame[~drop_mask]

    by_dialog: dict[tuple, list[_TimelineEvent]] = defaultdict(list)
    for _, row in frame.iterrows():
        peer_id = None if pd.isna(row["peer_id"]) else int(row["peer_id"])
        title = str(row["dialog_title"])
        date = _from_iso(str(row["date"]))
        if bool(row["is_own"]):
            event = _TimelineEvent(
                date=date,
                message=ContextMessage(
                    message_key=f"{peer_id or 'unknown'}:{int(row['message_id'])}",
                    message_id=int(row["message_id"]),
                    peer_id=peer_id,
                    dialog_title=title,
                    date=date,
                    text=str(row["text"]),
                ),
            )
        else:
            event = _TimelineEvent(date=date, message=None)
        by_dialog[(peer_id, title)].append(event)

    for key in by_dialog:
        by_dialog[key].sort(key=lambda e: e.date)

    return by_dialog, stats


def _build_rows(
    by_dialog: dict[tuple, list[_TimelineEvent]],
    *,
    gap_seconds: int,
) -> tuple[list[dict], list[QueuedContextMessage]]:
    """Run grouping and produce labeling-parquet rows + classifier inputs."""
    rows: list[dict] = []
    classifier_inputs: list[QueuedContextMessage] = []
    synthetic_id = 0

    for _, events in by_dialog.items():
        merged = _group_with_barriers(events, gap_seconds=gap_seconds)
        # reconstruct group sizes: scan events chronologically and split where
        # _group_with_barriers would have split. We can derive group_size from
        # the merged text since lines are joined with '\n', but that breaks for
        # multi-line originals. So we recount the same way: walk events.
        own_events_per_group = _recount_group_sizes(events, gap_seconds=gap_seconds)
        assert len(own_events_per_group) == len(merged)

        for merged_msg, members in zip(merged, own_events_per_group):
            first_msg = members[0].message
            last_msg = members[-1].message
            assert first_msg is not None and last_msg is not None

            text = merged_msg.text
            date_iso = merged_msg.date.isoformat()
            dialog = merged_msg.dialog_title
            message_id = _stable_message_id(date=date_iso, dialog=dialog, text=text)
            stripped = text.strip()
            words = re.findall(r"\w+", text, flags=re.UNICODE)

            row = {
                "message_id": message_id,
                "date": date_iso,
                "dialog": dialog,
                "text": text,
                "text_hash": _hash_text(text),
                "text_len": len(stripped),
                "word_count": len(words),
                "has_link": bool(_URL_PATTERN.search(text)),
                "is_command": stripped.startswith("/"),
                "heuristic_score": pd.NA,
                "nli_score": pd.NA,
                "embedding_score": pd.NA,
                "label": pd.NA,
                "label_name": pd.NA,
                "label_source": pd.NA,
                "labeled_at": pd.NA,
                "catboost_label": pd.NA,
                "catboost_label_name": pd.NA,
                "catboost_confidence": pd.NA,
                "catboost_proba_drop": pd.NA,
                "catboost_proba_maybe": pd.NA,
                "catboost_proba_keep": pd.NA,
                "catboost_model_version": pd.NA,
                "catboost_predicted_at": pd.NA,
                "peer_id": merged_msg.peer_id,
                "first_message_id": first_msg.message_id,
                "last_message_id": last_msg.message_id,
                "group_size": len(members),
                "span_seconds": int(
                    (last_msg.date - first_msg.date).total_seconds()
                ),
            }
            rows.append(row)
            classifier_inputs.append(
                QueuedContextMessage(
                    id=synthetic_id,
                    message_key=merged_msg.message_key,
                    date=merged_msg.date,
                    dialog_title=dialog,
                    text=text,
                    label=None,
                )
            )
            synthetic_id += 1

    return rows, classifier_inputs


def _recount_group_sizes(
    events: list[_TimelineEvent], *, gap_seconds: int
) -> list[list[_TimelineEvent]]:
    """Repeat the grouping algorithm but keep raw events per group."""
    ordered = sorted(events, key=lambda e: e.date)
    groups: list[list[_TimelineEvent]] = []
    barrier_pending = True
    for event in ordered:
        if event.message is None:
            barrier_pending = True
            continue
        if barrier_pending or not groups:
            groups.append([event])
            barrier_pending = False
            continue
        previous_event = groups[-1][-1]
        delta = (event.date - previous_event.date).total_seconds()
        if gap_seconds > 0 and delta > gap_seconds:
            groups.append([event])
        else:
            groups[-1].append(event)
    return groups


def _classify(
    rows: list[dict],
    inputs: list[QueuedContextMessage],
    *,
    enable_nli: bool,
) -> None:
    """Run Mix0035Classifier and per-row scorers in place; mutates ``rows``."""
    if not inputs:
        return
    settings = load_settings()
    classifier = Mix0035Classifier(
        settings.telegram_context_model_path,
        stage1_model_name=settings.telegram_context_stage1_model,
        stage2_model_name=settings.telegram_context_stage2_model,
        feature_embedding_model_name=settings.telegram_context_feature_embedding_model,
        enable_nli_score=enable_nli,
        nli_model_name=settings.telegram_context_nli_model,
    )
    print(
        "Loading Mix0035 model artifacts from "
        f"{settings.telegram_context_model_path}… this is the slow part."
    )
    # Force load + clamp max_seq_length on all embedders.
    # rubert-tiny2 advertises max_position_embeddings=2048, which makes
    # SentenceTransformer pad to that length and blow up attention memory
    # to 12+ GiB on a batch of 64. Real merged messages are short, so 256
    # tokens is more than enough.
    classifier._load()
    for embedder_attr in ("_stage1_embedder", "_stage2_embedder", "_feature_embedder"):
        embedder = getattr(classifier, embedder_attr, None)
        if embedder is not None and hasattr(embedder, "max_seq_length"):
            current = embedder.max_seq_length
            embedder.max_seq_length = min(current or 256, 256)
            print(f"  {embedder_attr}.max_seq_length: {current} → {embedder.max_seq_length}")
    labels = classifier.classify(inputs)
    print(f"Classified {len(labels)} merged items.")

    now_iso = datetime.now(UTC).isoformat(timespec="seconds")
    for queued in inputs:
        label = labels.get(queued.id)
        if label is None:
            continue
        row = rows[queued.id]
        row["catboost_label"] = _LABEL_TO_INT[label]
        row["catboost_label_name"] = label
        row["catboost_model_version"] = "mix0035"
        row["catboost_predicted_at"] = now_iso

    counts = {label: sum(1 for v in labels.values() if v == label)
              for label in ("drop", "maybe", "keep")}
    print(f"Label distribution: {counts}")

    # --- Per-row scorers exposed for the labeling UI -------------------------
    texts = [queued.text for queued in inputs]

    # embedding_score — always available, classifier has loaded the model
    print("Computing per-row embedding_score…")
    embedding_scores = classifier._embedding_scores(texts)
    for queued, score in zip(inputs, embedding_scores, strict=True):
        rows[queued.id]["embedding_score"] = float(score)

    # nli_score — only if NLI was enabled (heavy)
    if enable_nli:
        print(f"Computing per-row nli_score for {len(texts)} items… "
              "(NLI is the slow part)")
        nli_scores = classifier._nli_scores(texts)
        for queued, score in zip(inputs, nli_scores, strict=True):
            rows[queued.id]["nli_score"] = float(score)

    # heuristic_score — cheap, compute even without classifier
    print("Computing per-row heuristic_score…")
    for queued in inputs:
        stripped = queued.text.strip()
        words = re.findall(r"\w+", queued.text, flags=re.UNICODE)
        rows[queued.id]["heuristic_score"] = float(_heuristic_score(stripped, words))


def _fill_heuristic_only(rows: list[dict], inputs: list[QueuedContextMessage]) -> None:
    """Fast path: heuristic_score only (no classifier load)."""
    for queued in inputs:
        stripped = queued.text.strip()
        words = re.findall(r"\w+", queued.text, flags=re.UNICODE)
        rows[queued.id]["heuristic_score"] = float(_heuristic_score(stripped, words))


def main() -> None:
    settings = load_settings()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cache", type=Path,
        default=ROOT / "data" / "dry_run_events.parquet",
    )
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "data" / "context_labeling_dryrun.parquet",
        help=(
            "where to write the labeling-ready parquet. Defaults to a "
            "*_dryrun file so it does not clobber data/context_labeling.parquet "
            "(which holds your manual labels). Pass the canonical path "
            "explicitly when you are ready to overwrite."
        ),
    )
    parser.add_argument("--gap", type=int, default=settings.telegram_context_merge_gap_seconds)
    parser.add_argument("--dialog", type=str, default="", help="substring filter (case-insensitive)")
    parser.add_argument(
        "--include-forwards",
        action="store_true",
        help="keep own forwarded messages (default: drop them)",
    )
    parser.add_argument(
        "--include-via-bot",
        action="store_true",
        help="keep own inline-bot messages (default: drop them)",
    )
    parser.add_argument(
        "--no-classify",
        action="store_true",
        help="skip CatBoost (faster; just produce merged rows without labels)",
    )
    parser.add_argument(
        "--no-nli",
        action="store_true",
        help="skip NLI per-row scoring (saves several minutes)",
    )
    args = parser.parse_args()

    by_dialog, stats = _load_events(
        args.cache,
        include_forwards=args.include_forwards,
        include_via_bot=args.include_via_bot,
        dialog_filter=args.dialog,
    )
    print(f"Cache: {args.cache}")
    print(
        f"Raw events: {stats['raw_rows']} "
        f"(own={stats['raw_own']}, incoming={stats['raw_incoming']})"
    )
    if stats["dropped_forwards_own"]:
        print(f"Dropped own forwards: {stats['dropped_forwards_own']}")
    if stats["dropped_via_bot_own"]:
        print(f"Dropped own via-bot:  {stats['dropped_via_bot_own']}")
    print(f"Dialogs touched: {len(by_dialog)}")

    rows, classifier_inputs = _build_rows(by_dialog, gap_seconds=args.gap)
    print(f"Merged rows (after barrier groupping, gap={args.gap}s): {len(rows)}")
    if not rows:
        print("Nothing to write.")
        return

    sizes = [r["group_size"] for r in rows]
    print(
        "Group size distribution: "
        f"size=1:{sum(1 for s in sizes if s == 1)}  "
        f"size=2:{sum(1 for s in sizes if s == 2)}  "
        f"3-4:{sum(1 for s in sizes if 3 <= s <= 4)}  "
        f"5-9:{sum(1 for s in sizes if 5 <= s <= 9)}  "
        f"10+:{sum(1 for s in sizes if s >= 10)}"
    )

    if not args.no_classify:
        _classify(rows, classifier_inputs, enable_nli=not args.no_nli)
    else:
        print("Skipping CatBoost (--no-classify).")
        _fill_heuristic_only(rows, classifier_inputs)

    frame = pd.DataFrame(rows)
    frame = frame.sort_values(["date", "message_id"]).reset_index(drop=True)
    if "label" in frame:
        frame["label"] = frame["label"].astype("Int64")
    if "catboost_label" in frame:
        frame["catboost_label"] = frame["catboost_label"].astype("Int64")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        print(f"Note: {args.output} exists; overwriting (no merge with existing labels here).")
    frame.to_parquet(args.output, index=False)
    print(f"\nSaved → {args.output} ({args.output.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
