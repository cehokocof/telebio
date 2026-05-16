"""Tests for context relevance filtering and scoring."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from telebio.context_relevance import (
    LocalNliScorer,
    RelevanceOptions,
    compare_semantic_scorers,
    fingerprint_context,
    select_context_messages,
)
from telebio.services.telegram import ContextMessage


class FixedScorer:
    def __init__(self, scores: dict[str, float]) -> None:
        self._scores = scores

    def score(self, messages: list[ContextMessage]) -> dict[ContextMessage, float]:
        return {message: self._scores.get(message.text, 0.0) for message in messages}


def _load_fixture() -> list[tuple[ContextMessage, str]]:
    path = Path(__file__).parent / "fixtures" / "context_messages_sample.json"
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
    else:
        data = [
            {
                "date": "2026-05-09T22:25:18+00:00",
                "dialog": "cehokoc(Arseniy)",
                "text": "https://x.com/atenov_d/status/2052766098756395025?s=52",
                "expected": "drop",
            },
            {
                "date": "2026-05-10T03:57:42+00:00",
                "dialog": "cehokoc(Arseniy)",
                "text": "я пишу сообщение в тех поддержку платформы, чтобы попасть на стажировку в тбанк.",
                "expected": "keep",
            },
            {
                "date": "2026-05-10T03:57:49+00:00",
                "dialog": "cehokoc(Arseniy)",
                "text": "Привет! Отличный подход. Пожалуйста, ответь на следующие вопросы.",
                "expected": "drop",
            },
            {
                "date": "2026-05-15T12:38:49+00:00",
                "dialog": "telebio",
                "text": "/status",
                "expected": "drop",
            },
            {
                "date": "2026-05-15T12:40:23+00:00",
                "dialog": "Ярик",
                "text": "cos sim",
                "expected": "drop",
            },
            {
                "date": "2026-05-15T12:46:08+00:00",
                "dialog": "Ярик",
                "text": "да я тут приколы с telebio делаю",
                "expected": "keep",
            },
        ]
    return [
        (
            ContextMessage(
                date=datetime.fromisoformat(row["date"]),
                dialog=row["dialog"],
                text=row["text"],
            ),
            row["expected"],
        )
        for row in data
    ]


def test_heuristic_filters_prepared_messages() -> None:
    fixture = _load_fixture()
    messages = [message for message, _ in fixture]
    expected = {message.text: label for message, label in fixture}
    selection = select_context_messages(
        messages,
        options=RelevanceOptions(enable_nli=False, min_score=0.55),
        scorer=None,
    )
    decisions = {row.message.text: row.decision for row in selection.scored}

    for text, label in expected.items():
        assert decisions[text] == label


def test_nli_can_rescue_semantic_message_after_heuristic() -> None:
    message = ContextMessage(
        date=datetime.fromisoformat("2026-05-15T12:40:23+00:00"),
        dialog="Ярик",
        text="cos sim",
    )
    selection = select_context_messages(
        [message],
        options=RelevanceOptions(enable_nli=True, min_score=0.55),
        scorer=FixedScorer({"cos sim": 1.0}),
    )

    assert [message.text for message in selection.selected] == ["cos sim"]
    assert selection.scored[0].nli_score == 1.0


def test_compare_nli_and_embedding_scorers_on_prepared_fixture() -> None:
    fixture = _load_fixture()
    messages = [message for message, _ in fixture]
    nli_scores = {
        "да я тут приколы с telebio делаю": 0.9,
        "теперь он смотрит на все мои сообщения и делает промпт по ним": 0.9,
        "и это идет в био": 0.8,
    }
    embedding_scores = {
        "cos sim": 0.95,
        "да я тут приколы с telebio делаю": 0.65,
        "теперь он смотрит на все мои сообщения и делает промпт по ним": 0.7,
    }

    report = compare_semantic_scorers(
        messages,
        options=RelevanceOptions(min_score=0.55),
        nli_scorer=FixedScorer(nli_scores),
        embedding_scorer=FixedScorer(embedding_scores),
    )

    assert "nli" in report
    assert "embedding" in report
    assert report["nli_selected"] != report["embedding_selected"]


def test_fingerprint_is_stable_for_same_context() -> None:
    message = ContextMessage(
        date=datetime.fromisoformat("2026-05-15T12:46:08+00:00"),
        dialog="Ярик",
        text="да я тут приколы с telebio делаю",
    )

    assert fingerprint_context([message]) == fingerprint_context([message])


def test_nli_pipeline_uses_truncation_for_long_messages() -> None:
    calls: list[dict[str, object]] = []

    def fake_pipeline(text: str, **kwargs):
        calls.append(kwargs)
        return [[{"label": "entailment", "score": 0.5}]]

    LocalNliScorer._entailment_score(fake_pipeline, "очень длинный текст " * 200, "гипотеза")

    assert calls == [{"truncation": True, "max_length": 512}]
