"""Filtering and relevance scoring for context-mode Telegram messages."""

from __future__ import annotations

import hashlib
import json
import logging
import math
import re
from dataclasses import dataclass, field
from typing import Protocol

from telebio.services.telegram import ContextMessage

logger = logging.getLogger(__name__)

_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_COMMAND_RE = re.compile(r"^/\w+(?:\s|$)")
_WORD_RE = re.compile(r"[A-Za-zА-Яа-яЁё0-9]+")
_LAUGHTER_RE = re.compile(r"^(?:а?х[авх]+|х[аx]+|lol|lmao|kek)+$", re.IGNORECASE)
_PROFANITY_RE = re.compile(
    r"\b(?:бля|блять|пиздец|пизд|хуй|хуя|аху|еба|ёба|ебать)\w*\b",
    re.IGNORECASE,
)

POSITIVE_HINTS = {
    "делаю",
    "пишу",
    "работаю",
    "учусь",
    "стажиров",
    "тест",
    "ml",
    "код",
    "codex",
    "telebio",
    "проект",
    "задач",
    "собесед",
    "команда",
    "позици",
    "био",
    "контекст",
    "промпт",
}


@dataclass(frozen=True, slots=True)
class RelevanceOptions:
    """Tunable context relevance settings."""

    top_k: int = 15
    min_score: float = 0.55
    excluded_dialogs: tuple[str, ...] = ("telebio",)
    enable_nli: bool = True
    semantic_scorer: str = "nli"
    nli_model: str = "cointegrated/rubert-base-cased-nli-threeway"
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


@dataclass(frozen=True, slots=True)
class ScoredContextMessage:
    """Context message with scoring metadata for logging and ranking."""

    message: ContextMessage
    heuristic_score: float
    nli_score: float | None
    final_score: float
    decision: str
    reasons: tuple[str, ...] = field(default_factory=tuple)

    def to_json_dict(self) -> dict[str, object]:
        return {
            "date": self.message.date.isoformat(timespec="seconds"),
            "dialog": self.message.dialog,
            "text": self.message.text,
            "heuristic_score": round(self.heuristic_score, 3),
            "nli_score": None if self.nli_score is None else round(self.nli_score, 3),
            "final_score": round(self.final_score, 3),
            "decision": self.decision,
            "reasons": list(self.reasons),
        }


@dataclass(frozen=True, slots=True)
class ContextSelection:
    """Selected context messages and diagnostics."""

    selected: list[ContextMessage]
    scored: list[ScoredContextMessage]
    fingerprint: str


class RelevanceScorer(Protocol):
    """Semantic relevance scorer interface."""

    def score(self, messages: list[ContextMessage]) -> dict[ContextMessage, float]:
        """Return relevance scores from 0.0 to 1.0."""
        ...


class LocalNliScorer:
    """Lazy local NLI scorer based on Hugging Face transformers."""

    _positive_hypothesis = (
        "Это сообщение описывает текущее состояние, занятие, проект, проблему "
        "или интерес пользователя."
    )
    _negative_hypothesis = (
        "Это сообщение является шумом, командой, реакцией или не несёт смысла "
        "для Telegram bio."
    )

    def __init__(self, model: str) -> None:
        self._model = model
        self._pipeline = None
        self._unavailable = False

    def score(self, messages: list[ContextMessage]) -> dict[ContextMessage, float]:
        if not messages:
            return {}

        pipeline = self._load_pipeline()
        if pipeline is None:
            return {message: 0.0 for message in messages}

        scores: dict[ContextMessage, float] = {}
        for message in messages:
            positive = self._entailment_score(pipeline, message.text, self._positive_hypothesis)
            negative = self._entailment_score(pipeline, message.text, self._negative_hypothesis)
            scores[message] = max(0.0, min(1.0, positive * (1.0 - negative)))
        return scores

    def _load_pipeline(self):
        if self._pipeline is not None:
            return self._pipeline
        if self._unavailable:
            return None

        try:
            from transformers import pipeline
        except ImportError:
            logger.warning(
                "Local NLI scorer unavailable: install transformers/torch to enable it"
            )
            self._unavailable = True
            return None

        self._pipeline = pipeline(
            "text-classification",
            model=self._model,
            tokenizer=self._model,
            return_all_scores=True,
        )
        return self._pipeline

    @staticmethod
    def _entailment_score(pipeline, premise: str, hypothesis: str) -> float:
        result = pipeline(
            f"{premise} [SEP] {hypothesis}",
            truncation=True,
            max_length=512,
        )
        rows = result[0] if result and isinstance(result[0], list) else result
        best = 0.0
        for row in rows:
            label = str(row.get("label", "")).lower()
            if "entail" in label or "entailment" in label:
                best = max(best, float(row.get("score", 0.0)))
        return best


class LocalEmbeddingScorer:
    """Lazy local embedding scorer based on transformer mean pooling."""

    _anchors = (
        "текущее состояние пользователя",
        "чем пользователь сейчас занимается",
        "проект работа учеба проблема интерес пользователя",
        "сообщение полезно для актуального Telegram bio",
    )
    _negative_anchors = (
        "команда боту",
        "короткая реакция без смысла",
        "ссылка без текста",
        "случайный шум для bio",
    )

    def __init__(self, model: str) -> None:
        self._model = model
        self._tokenizer = None
        self._model_obj = None
        self._anchor_vectors: list[list[float]] | None = None
        self._negative_anchor_vectors: list[list[float]] | None = None
        self._unavailable = False

    def score(self, messages: list[ContextMessage]) -> dict[ContextMessage, float]:
        if not messages:
            return {}

        if not self._load_model():
            return {message: 0.0 for message in messages}

        if self._anchor_vectors is None:
            self._anchor_vectors = [self._embed(anchor) for anchor in self._anchors]
            self._negative_anchor_vectors = [
                self._embed(anchor) for anchor in self._negative_anchors
            ]

        scores: dict[ContextMessage, float] = {}
        for message in messages:
            vector = self._embed(message.text)
            positive = max(_cosine_similarity(vector, anchor) for anchor in self._anchor_vectors)
            negative = max(
                _cosine_similarity(vector, anchor)
                for anchor in self._negative_anchor_vectors or []
            )
            scores[message] = max(0.0, min(1.0, (positive - negative + 1.0) / 2.0))
        return scores

    def _load_model(self) -> bool:
        if self._model_obj is not None and self._tokenizer is not None:
            return True
        if self._unavailable:
            return False

        try:
            from transformers import AutoModel, AutoTokenizer
        except ImportError:
            logger.warning(
                "Local embedding scorer unavailable: install transformers/torch to enable it"
            )
            self._unavailable = True
            return False

        self._tokenizer = AutoTokenizer.from_pretrained(self._model)
        self._model_obj = AutoModel.from_pretrained(self._model)
        self._model_obj.eval()
        return True

    def _embed(self, text: str) -> list[float]:
        import torch

        assert self._tokenizer is not None
        assert self._model_obj is not None

        encoded = self._tokenizer(
            text,
            truncation=True,
            max_length=256,
            return_tensors="pt",
        )
        with torch.no_grad():
            output = self._model_obj(**encoded)
        token_embeddings = output.last_hidden_state
        mask = encoded["attention_mask"].unsqueeze(-1).expand(token_embeddings.size()).float()
        summed = (token_embeddings * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        vector = (summed / counts).squeeze(0)
        return [float(value) for value in vector]


def select_context_messages(
    messages: list[ContextMessage],
    *,
    options: RelevanceOptions,
    scorer: RelevanceScorer | None,
) -> ContextSelection:
    """Filter, score, rank, and log context messages."""
    heuristic_rows = [_score_heuristic(message, options) for message in messages]
    nli_input = [message for message, score, _ in heuristic_rows if score > 0.0]
    nli_scores = scorer.score(nli_input) if options.enable_nli and scorer else {}

    scored: list[ScoredContextMessage] = []
    for message, heuristic_score, reasons in heuristic_rows:
        nli_score = nli_scores.get(message)
        final_score = (
            heuristic_score
            if nli_score is None
            else heuristic_score * 0.45 + nli_score * 0.55
        )
        decision = "keep" if final_score >= options.min_score else "drop"
        if decision == "drop" and "below_min_score" not in reasons:
            reasons = (*reasons, "below_min_score")
        scored.append(
            ScoredContextMessage(
                message=message,
                heuristic_score=heuristic_score,
                nli_score=nli_score,
                final_score=final_score,
                decision=decision,
                reasons=reasons,
            )
        )

    ranked = sorted(
        (row for row in scored if row.decision == "keep"),
        key=lambda row: (row.final_score, row.message.date),
        reverse=True,
    )
    selected = sorted(
        [row.message for row in ranked[: options.top_k]],
        key=lambda message: message.date,
    )
    fingerprint = fingerprint_context(selected)

    logger.info(
        "Context relevance JSON: %s",
        json.dumps(
            {
                "raw_candidates": [message_to_json(message) for message in messages],
                "scored_candidates": [row.to_json_dict() for row in scored],
                "selected_context": [message_to_json(message) for message in selected],
                "fingerprint": fingerprint,
            },
            ensure_ascii=False,
        ),
    )
    return ContextSelection(selected=selected, scored=scored, fingerprint=fingerprint)


def compare_semantic_scorers(
    messages: list[ContextMessage],
    *,
    options: RelevanceOptions,
    nli_scorer: RelevanceScorer | None,
    embedding_scorer: RelevanceScorer | None,
) -> dict[str, object]:
    """Score the same prepared messages with NLI and embeddings for logs/tests."""
    nli_options = RelevanceOptions(
        top_k=options.top_k,
        min_score=options.min_score,
        excluded_dialogs=options.excluded_dialogs,
        enable_nli=True,
        semantic_scorer="nli",
        nli_model=options.nli_model,
        embedding_model=options.embedding_model,
    )
    embedding_options = RelevanceOptions(
        top_k=options.top_k,
        min_score=options.min_score,
        excluded_dialogs=options.excluded_dialogs,
        enable_nli=True,
        semantic_scorer="embedding",
        nli_model=options.nli_model,
        embedding_model=options.embedding_model,
    )
    nli_selection = select_context_messages(
        messages,
        options=nli_options,
        scorer=nli_scorer,
    )
    embedding_selection = select_context_messages(
        messages,
        options=embedding_options,
        scorer=embedding_scorer,
    )
    report = {
        "nli": [row.to_json_dict() for row in nli_selection.scored],
        "embedding": [row.to_json_dict() for row in embedding_selection.scored],
        "nli_selected": [message_to_json(message) for message in nli_selection.selected],
        "embedding_selected": [
            message_to_json(message) for message in embedding_selection.selected
        ],
    }
    logger.info("Context scorer comparison JSON: %s", json.dumps(report, ensure_ascii=False))
    return report


def fingerprint_context(messages: list[ContextMessage]) -> str:
    payload = [
        {
            "date": message.date.isoformat(timespec="seconds"),
            "dialog": message.dialog,
            "text": message.text,
        }
        for message in messages
    ]
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def message_to_json(message: ContextMessage) -> dict[str, str]:
    return {
        "date": message.date.isoformat(timespec="seconds"),
        "dialog": message.dialog,
        "text": message.text,
    }


def _score_heuristic(
    message: ContextMessage,
    options: RelevanceOptions,
) -> tuple[ContextMessage, float, tuple[str, ...]]:
    text = " ".join(message.text.strip().split())
    lower_text = text.lower()
    lower_dialog = message.dialog.lower()
    reasons: list[str] = []

    if any(lower_dialog == excluded.lower() for excluded in options.excluded_dialogs):
        return message, 0.0, ("excluded_dialog",)
    if _COMMAND_RE.match(text):
        return message, 0.0, ("bot_command",)
    if _is_link_only(text):
        return message, 0.0, ("link_only",)
    if _looks_like_assistant_paste(lower_text):
        return message, 0.0, ("assistant_like_paste",)

    words = _WORD_RE.findall(text)
    word_count = len(words)
    if word_count <= 1:
        return message, 0.0, ("too_short",)
    if _is_vague_fragment(lower_text, word_count):
        return message, 0.15, ("vague_fragment",)
    if word_count <= 3 and not _has_positive_hint(lower_text):
        return message, 0.15, ("short_low_information",)
    if _LAUGHTER_RE.match(lower_text):
        return message, 0.0, ("laughter_only",)
    if _is_numeric_only(text):
        return message, 0.0, ("numeric_only",)
    if _PROFANITY_RE.search(lower_text) and word_count < 7:
        return message, 0.25, ("short_profanity_rant",)

    score = 0.35
    if word_count >= 5:
        score += 0.2
        reasons.append("enough_words")
    if _has_positive_hint(lower_text):
        score += 0.25
        reasons.append("positive_hint")
    if "?" in text and word_count >= 5:
        score += 0.05
        reasons.append("substantive_question")
    if _PROFANITY_RE.search(lower_text):
        score -= 0.15
        reasons.append("profanity_penalty")
    if len(text) > 900:
        score -= 0.35
        reasons.append("very_long_penalty")
    elif len(text) > 450:
        score -= 0.2
        reasons.append("long_penalty")

    score = max(0.0, min(1.0, score))
    if not reasons:
        reasons.append("generic_text")
    return message, score, tuple(reasons)


def _has_positive_hint(lower_text: str) -> bool:
    return any(hint in lower_text for hint in POSITIVE_HINTS)


def _is_link_only(text: str) -> bool:
    without_links = _URL_RE.sub("", text).strip()
    return bool(_URL_RE.search(text)) and not without_links


def _is_numeric_only(text: str) -> bool:
    stripped = re.sub(r"[\s.,:;!?()\[\]{}+\-*/=_]", "", text)
    return bool(stripped) and stripped.isdigit()


def _looks_like_assistant_paste(lower_text: str) -> bool:
    markers = (
        "привет! отличный подход",
        "пожалуйста, ответь",
        "жду твоих ответов",
        "как только напишешь",
    )
    return any(marker in lower_text for marker in markers)


def _is_vague_fragment(lower_text: str, word_count: int) -> bool:
    if word_count <= 4 and "позици" in lower_text and lower_text.startswith("и на"):
        return True
    if word_count <= 5 and "скок" in lower_text:
        return True
    return False


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot / (left_norm * right_norm)
