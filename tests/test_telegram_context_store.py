from __future__ import annotations

from datetime import UTC, datetime, timedelta

from telebio.telegram_context import ContextMessage, TelegramContextStore


def _message(index: int, *, days_ago: int = 0, text: str | None = None) -> ContextMessage:
    return ContextMessage(
        message_key=f"peer:{index}",
        message_id=index,
        peer_id=1,
        dialog_title="dialog",
        date=datetime.now(UTC) - timedelta(days=days_ago, minutes=index),
        text=text or f"meaningful context message {index}",
    )


def test_store_deduplicates_messages(tmp_path) -> None:
    store = TelegramContextStore(tmp_path / "context.sqlite3")
    messages = [_message(1), _message(1)]

    assert store.upsert_messages(messages) == 1
    assert len(store.unclassified_messages()) == 1


def test_ready_batch_uses_latest_prompt_messages(tmp_path) -> None:
    store = TelegramContextStore(tmp_path / "context.sqlite3")
    store.upsert_messages([_message(index) for index in range(25)])
    store.save_labels({message.id: "keep" for message in store.unclassified_messages()})

    batch = store.ready_batch(
        min_batch=20,
        fallback_min_batch=10,
        fallback_max_age_days=7,
        max_prompt_messages=20,
        max_maybe_messages=5,
    )

    assert batch is not None
    assert batch.reason.startswith("min_batch:")
    assert len(batch.messages) == 15
    assert [message.message_key for message in batch.messages] == [
        f"peer:{index}" for index in range(14, -1, -1)
    ]
    assert batch.pending_keep_count == 25
    assert batch.pending_maybe_count == 0
    assert batch.included_keep_count == 15
    assert batch.included_maybe_count == 0


def test_store_updates_unprocessed_edited_message_and_resets_label(tmp_path) -> None:
    store = TelegramContextStore(tmp_path / "context.sqlite3")
    store.upsert_messages([_message(1, text="old text")])
    first = store.unclassified_messages()[0]
    store.save_labels({first.id: "drop"})

    store.upsert_messages([_message(1, text="new text with more meaning")])

    unclassified = store.unclassified_messages()
    assert len(unclassified) == 1
    assert unclassified[0].text == "new text with more meaning"


def test_ready_batch_fallback_after_age_threshold(tmp_path) -> None:
    store = TelegramContextStore(tmp_path / "context.sqlite3")
    store.upsert_messages([_message(index, days_ago=8) for index in range(10)])
    store.save_labels({message.id: "maybe" for message in store.unclassified_messages()})

    batch = store.ready_batch(
        min_batch=20,
        fallback_min_batch=10,
        fallback_max_age_days=7,
        max_prompt_messages=20,
        max_maybe_messages=5,
    )

    assert batch is not None
    assert batch.reason.startswith("fallback:")
    assert len(batch.messages) == 5
    assert batch.included_keep_count == 0
    assert batch.included_maybe_count == 5


def test_ready_batch_limits_maybe_but_keeps_strong_context(tmp_path) -> None:
    store = TelegramContextStore(tmp_path / "context.sqlite3")
    store.upsert_messages([_message(index) for index in range(30)])
    rows = store.unclassified_messages()
    store.save_labels(
        {
            message.id: "maybe" if index < 12 else "keep"
            for index, message in enumerate(rows)
        }
    )

    batch = store.ready_batch(
        min_batch=20,
        fallback_min_batch=10,
        fallback_max_age_days=7,
        max_prompt_messages=20,
        max_maybe_messages=5,
    )

    assert batch is not None
    assert batch.pending_keep_count == 18
    assert batch.pending_maybe_count == 12
    assert batch.included_keep_count == 15
    assert batch.included_maybe_count == 5
    assert all(message.label in {"keep", "maybe"} for message in batch.messages)


def test_ready_batch_does_not_use_previous_history_before_min_batch(tmp_path) -> None:
    store = TelegramContextStore(tmp_path / "context.parquet")
    store.upsert_messages([_message(index) for index in range(18)])
    store.save_labels({message.id: "keep" for message in store.unclassified_messages()})

    batch = store.ready_batch(
        min_batch=20,
        fallback_min_batch=10,
        fallback_max_age_days=7,
        max_prompt_messages=20,
        max_maybe_messages=5,
    )

    assert batch is None


def test_ready_batch_force_uses_recent_keep_maybe_before_min_batch(tmp_path) -> None:
    store = TelegramContextStore(tmp_path / "context.parquet")
    store.upsert_messages([_message(index, days_ago=1) for index in range(9)])
    store.save_labels({message.id: "maybe" for message in store.unclassified_messages()})

    batch = store.ready_batch(
        min_batch=20,
        fallback_min_batch=10,
        fallback_max_age_days=7,
        max_prompt_messages=20,
        max_maybe_messages=5,
        force=True,
        force_max_age_days=7,
    )

    assert batch is not None
    assert batch.reason == "manual_force_recent:9<= 7d"
    assert batch.pending_keep_count == 0
    assert batch.pending_maybe_count == 9
    assert batch.included_maybe_count == 5


def test_ready_batch_force_ignores_old_pending_messages(tmp_path) -> None:
    store = TelegramContextStore(tmp_path / "context.parquet")
    store.upsert_messages([_message(index, days_ago=8) for index in range(9)])
    store.save_labels({message.id: "maybe" for message in store.unclassified_messages()})

    batch = store.ready_batch(
        min_batch=20,
        fallback_min_batch=10,
        fallback_max_age_days=7,
        max_prompt_messages=20,
        max_maybe_messages=5,
        force=True,
        force_max_age_days=7,
    )

    assert batch is None


def test_mark_used_removes_messages_from_pending_queue(tmp_path) -> None:
    store = TelegramContextStore(tmp_path / "context.sqlite3")
    store.upsert_messages([_message(index) for index in range(3)])
    store.save_labels({message.id: "keep" for message in store.unclassified_messages()})
    pending = store.pending_selected_messages()

    assert store.mark_used([message.id for message in pending], bio="test bio") == 3
    assert store.pending_selected_messages() == []
