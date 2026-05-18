from __future__ import annotations

from datetime import UTC, datetime, timedelta

from telebio.context_prod import ContextMessage, ContextProdStore


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
    store = ContextProdStore(tmp_path / "context.sqlite3")
    messages = [_message(1), _message(1)]

    assert store.upsert_messages(messages) == 1
    assert len(store.unclassified_messages()) == 1


def test_ready_batch_uses_latest_prompt_messages(tmp_path) -> None:
    store = ContextProdStore(tmp_path / "context.sqlite3")
    store.upsert_messages([_message(index) for index in range(25)])
    store.save_labels({message.id: "keep" for message in store.unclassified_messages()})

    batch = store.ready_batch(
        min_batch=20,
        fallback_min_batch=10,
        fallback_max_age_days=7,
        max_prompt_messages=20,
    )

    assert batch is not None
    assert batch.reason.startswith("min_batch:")
    assert len(batch.messages) == 20
    assert [message.message_key for message in batch.messages] == [
        f"peer:{index}" for index in range(19, -1, -1)
    ]


def test_store_updates_unprocessed_edited_message_and_resets_label(tmp_path) -> None:
    store = ContextProdStore(tmp_path / "context.sqlite3")
    store.upsert_messages([_message(1, text="old text")])
    first = store.unclassified_messages()[0]
    store.save_labels({first.id: "drop"})

    store.upsert_messages([_message(1, text="new text with more meaning")])

    unclassified = store.unclassified_messages()
    assert len(unclassified) == 1
    assert unclassified[0].text == "new text with more meaning"


def test_ready_batch_fallback_after_age_threshold(tmp_path) -> None:
    store = ContextProdStore(tmp_path / "context.sqlite3")
    store.upsert_messages([_message(index, days_ago=8) for index in range(10)])
    store.save_labels({message.id: "maybe" for message in store.unclassified_messages()})

    batch = store.ready_batch(
        min_batch=20,
        fallback_min_batch=10,
        fallback_max_age_days=7,
        max_prompt_messages=20,
    )

    assert batch is not None
    assert batch.reason.startswith("fallback:")


def test_mark_used_removes_messages_from_pending_queue(tmp_path) -> None:
    store = ContextProdStore(tmp_path / "context.sqlite3")
    store.upsert_messages([_message(index) for index in range(3)])
    store.save_labels({message.id: "keep" for message in store.unclassified_messages()})
    pending = store.pending_selected_messages()

    assert store.mark_used([message.id for message in pending], bio="test bio") == 3
    assert store.pending_selected_messages() == []
