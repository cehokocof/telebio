from __future__ import annotations

from datetime import UTC, datetime

from telebio.context_prod import ContextBatch, QueuedContextMessage
from telebio.providers.context_prod_provider import _build_prompt, _normalise_generated_bio


def _queued(index: int, label: str, text: str) -> QueuedContextMessage:
    return QueuedContextMessage(
        id=index,
        message_key=f"peer:{index}",
        date=datetime(2026, 5, 19, 12, index, tzinfo=UTC),
        dialog_title="dialog",
        text=text,
        label=label,
    )


def test_build_prompt_splits_keep_and_maybe_sections() -> None:
    batch = ContextBatch(
        messages=[
            _queued(1, "keep", "делаю prod контекст для bio"),
            _queued(2, "maybe", "немного сомневаюсь в сигнале"),
        ],
        reason="test",
        pending_keep_count=1,
        pending_maybe_count=1,
        included_keep_count=1,
        included_maybe_count=1,
    )

    prompt = _build_prompt(batch)

    assert "- Сильный контекст keep:" in prompt
    assert "делаю prod контекст для bio" in prompt
    assert "- Слабые дополнительные сигналы maybe:" in prompt
    assert "немного сомневаюсь в сигнале" in prompt
    assert "drop-сообщения уже отфильтрованы" in prompt
    assert "- Стиль: смешно, саркастично" in prompt
    assert "- Финальный ответ: только bio, ровно 2 строки." in prompt
    assert "- Каждая строка итогового bio должна начинаться с - и пробела." in prompt
    assert "ЗАКОНЧЕННАЯ мысль" in prompt
    assert "одна строка" not in prompt


def test_build_prompt_does_not_include_drop_messages() -> None:
    batch = ContextBatch(
        messages=[
            _queued(1, "keep", "важное сообщение"),
            _queued(2, "drop", "/new"),
        ],
        reason="test",
    )

    prompt = _build_prompt(batch)

    assert "важное сообщение" in prompt
    assert "/new" not in prompt


def test_normalise_generated_bio_keeps_complete_short_lines() -> None:
    text = (
        "- ML творит чудеса, не только в бизнесе, но и в науке.\n"
        "- Иногда цена ошибки выше дедлайна"
    )

    bio = _normalise_generated_bio(text)

    assert len(bio) <= 70
    assert bio.splitlines() == [
        "- ML творит чудеса, не только в",
        "- Иногда цена ошибки выше",
    ]


def test_normalise_generated_bio_drops_line_that_does_not_fit() -> None:
    text = (
        "- Порой один курс — и всё\n"
        "- Иногда ML важнее сна.\n"
        "- Стараюсь не решать задачи во сне"
    )

    bio = _normalise_generated_bio(text)

    assert len(bio) <= 70
    assert bio.splitlines() == [
        "- Порой один курс — и всё",
        "- Иногда ML важнее сна.",
    ]
