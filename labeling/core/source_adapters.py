"""Adapt various raw parquet schemas to the labeling schema.

Three input shapes are auto-detected by their column set:

* ``prod`` — ``data/context_prod.parquet``: per-message rows from the live
  production stream (``row_id``, ``message_key``, ``message_date``, …).
* ``raw_events`` — ``data/dry_run_events.parquet``: raw Telegram fetch with
  outgoing + incoming events (``peer_id``, ``is_own``, ``is_forward``, …).
  Consecutive own messages are grouped with the same barrier rule as
  production (``_group_with_barriers``) before being written out.
* ``labeling`` — anything already in the labeling schema (``message_id``,
  ``date``, ``dialog``, ``text``); features are recomputed from text so a
  stale parquet can be refreshed.

Every output row carries ``peer_id`` and ``chat_category`` so downstream
filters (e.g. ``--exclude-categories bot,channel``) work without re-fetching
Telegram metadata. ``chat_category`` falls back to ``"unknown"`` when the
peer is not in the provided category map.
"""

from __future__ import annotations

import hashlib
import re
from collections import defaultdict

import pandas as pd

from labeling.core.dataset import stable_message_id

_URL_PATTERN = re.compile(r"https?://", re.IGNORECASE)
_UNKNOWN = "unknown"


def detect_source_type(frame: pd.DataFrame) -> str:
    cols = set(frame.columns)
    if {"row_id", "message_key", "message_date"} <= cols:
        return "prod"
    if {"peer_id", "is_own", "date"} <= cols:
        return "raw_events"
    if {"message_id", "date", "dialog", "text"} <= cols:
        return "labeling"
    raise ValueError(
        "Cannot auto-detect source schema. Columns present: " f"{sorted(cols)}"
    )


def adapt(
    frame: pd.DataFrame,
    *,
    source_type: str,
    group_gap_seconds: int = 1800,
    categories: dict[int, str] | None = None,
) -> pd.DataFrame:
    """Dispatch to the right adapter based on ``source_type``."""
    cats = categories or {}
    match source_type:
        case "prod":
            return _adapt_prod(frame, cats)
        case "raw_events":
            return _adapt_raw_events(frame, gap_seconds=group_gap_seconds, cats=cats)
        case "labeling":
            return _adapt_labeling(frame, cats)
        case other:
            raise ValueError(f"Unsupported source_type: {other!r}")


# ---------------------------------------------------------------------------
# Adapters
# ---------------------------------------------------------------------------


def _adapt_prod(frame: pd.DataFrame, cats: dict[int, str]) -> pd.DataFrame:
    records = [
        _labeling_record(
            date=str(row["message_date"]),
            dialog=str(row["dialog_title"]),
            text=str(row["text"]),
            peer_id=_safe_peer_id(row.get("peer_id")),
            cats=cats,
        )
        for _, row in frame.iterrows()
    ]
    return pd.DataFrame(records)


def _adapt_raw_events(
    frame: pd.DataFrame, *, gap_seconds: int, cats: dict[int, str]
) -> pd.DataFrame:
    from telebio.services.telegram import _TimelineEvent, _group_with_barriers, ContextMessage
    from telebio.telegram_context import _from_iso

    by_dialog: dict[tuple, list] = defaultdict(list)
    for _, row in frame.iterrows():
        peer_id = _safe_peer_id(row["peer_id"])
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

    records: list[dict] = []
    for (peer_id, _), events in by_dialog.items():
        for merged in _group_with_barriers(events, gap_seconds=gap_seconds):
            records.append(
                _labeling_record(
                    date=merged.date.isoformat(),
                    dialog=merged.dialog_title,
                    text=merged.text,
                    peer_id=peer_id,
                    cats=cats,
                )
            )
    return pd.DataFrame(records)


def _adapt_labeling(frame: pd.DataFrame, cats: dict[int, str]) -> pd.DataFrame:
    has_peer = "peer_id" in frame.columns
    records = [
        _labeling_record(
            date=str(row["date"]),
            dialog=str(row["dialog"]),
            text=str(row["text"]),
            peer_id=_safe_peer_id(row["peer_id"]) if has_peer else None,
            cats=cats,
        )
        for _, row in frame.iterrows()
    ]
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _labeling_record(
    *,
    date: str,
    dialog: str,
    text: str,
    peer_id: int | None,
    cats: dict[int, str],
) -> dict:
    stripped = text.strip()
    words = re.findall(r"\w+", text, flags=re.UNICODE)
    category = cats.get(peer_id, _UNKNOWN) if peer_id is not None else _UNKNOWN
    return {
        "message_id": stable_message_id(date=date, dialog=dialog, text=text),
        "date": date,
        "dialog": dialog,
        "peer_id": peer_id,
        "chat_category": category,
        "text": text,
        "text_hash": _hash_text(text),
        "text_len": len(stripped),
        "word_count": len(words),
        "has_link": bool(_URL_PATTERN.search(text)),
        "is_command": stripped.startswith("/"),
        "heuristic_score": pd.NA,
        "nli_score": pd.NA,
        "embedding_score": pd.NA,
        "true_state": pd.NA,
        "label": pd.NA,
        "label_name": pd.NA,
        "label_source": pd.NA,
        "labeled_at": pd.NA,
    }


def _safe_peer_id(value) -> int | None:
    if value is None or (hasattr(pd, "isna") and pd.isna(value)):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:20]
