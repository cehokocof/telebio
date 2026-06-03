"""Production context queue and mix0035 classifier runtime."""

from __future__ import annotations

import hashlib
import json
import logging
import math
import pickle
import re
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Iterable, Literal

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sentence_transformers import SentenceTransformer

from telebio.context_exceptions import ContextBatchNotReady

logger = logging.getLogger(__name__)

ContextLabel = Literal["drop", "maybe", "keep"]

NUMERIC_COLUMNS = (
    "text_len",
    "word_count",
    "has_link",
    "is_command",
    "heuristic_score",
    "nli_score",
    "embedding_score",
)

_URL_RE = re.compile(r"https?://|t\.me/|www\.", re.IGNORECASE)
_WORD_RE = re.compile(r"\w+", re.UNICODE)
_LOW_VALUE_RE = re.compile(
    r"^(?:да|нет|ок|окей|ага|угу|спс|хз|лол|ах+|ахв+|авх+|бля|супер|норм|"
    r"видел\??|почему|прикол|чего|че|чё|как|понял|ясно|0(?:\.\d+)?)$",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class ContextMessage:
    """Outgoing Telegram text message collected for context processing."""

    message_key: str
    message_id: int
    peer_id: int | None
    dialog_title: str
    date: datetime
    text: str

    @property
    def text_hash(self) -> str:
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class QueuedContextMessage:
    id: int
    message_key: str
    date: datetime
    dialog_title: str
    text: str
    label: ContextLabel | None


@dataclass(frozen=True, slots=True)
class ContextBatch:
    messages: list[QueuedContextMessage]
    reason: str
    pending_keep_count: int = 0
    pending_maybe_count: int = 0
    included_keep_count: int = 0
    included_maybe_count: int = 0


class ContextProdStore:
    """Parquet-backed context dataset used by runtime and future training."""

    _columns = (
        "row_id",
        "message_key",
        "message_id",
        "peer_id",
        "dialog_title",
        "message_date",
        "text",
        "text_hash",
        "label",
        "classified_at",
        "used_at",
        "batch_key",
        "created_at",
    )

    def __init__(self, dataset_path: Path) -> None:
        self._dataset_path = dataset_path
        self._dataset_path.parent.mkdir(parents=True, exist_ok=True)

    def upsert_messages(self, messages: Iterable[ContextMessage]) -> int:
        rows = list(messages)
        if not rows:
            return 0

        frame = self._load_frame()
        by_key = {str(row.message_key): index for index, row in frame.iterrows()}
        changed = 0
        next_row_id = _next_row_id(frame)
        now = _now_iso()

        for msg in rows:
            row = {
                "row_id": next_row_id,
                "message_key": msg.message_key,
                "message_id": msg.message_id,
                "peer_id": msg.peer_id,
                "dialog_title": msg.dialog_title,
                "message_date": _to_iso(msg.date),
                "text": msg.text,
                "text_hash": msg.text_hash,
                "label": pd.NA,
                "classified_at": pd.NA,
                "used_at": pd.NA,
                "batch_key": pd.NA,
                "created_at": now,
            }
            existing_index = by_key.get(msg.message_key)
            if existing_index is None:
                frame.loc[len(frame)] = row
                by_key[msg.message_key] = len(frame) - 1
                next_row_id += 1
                changed += 1
                continue

            existing = frame.loc[existing_index]
            if _is_missing(existing.get("used_at")) and existing.get("text_hash") != msg.text_hash:
                for key in ("dialog_title", "message_date", "text", "text_hash"):
                    frame.at[existing_index, key] = row[key]
                frame.at[existing_index, "label"] = pd.NA
                frame.at[existing_index, "classified_at"] = pd.NA
                changed += 1

        if changed:
            self._save_frame(frame)
        return changed

    def unclassified_messages(self, *, limit: int = 1000) -> list[QueuedContextMessage]:
        frame = self._load_frame()
        if frame.empty:
            return []
        subset = frame[frame["label"].isna()].sort_values(["message_date", "row_id"])
        return [_queued_from_row(row) for _, row in subset.head(limit).iterrows()]

    def save_labels(self, labels: dict[int, ContextLabel]) -> None:
        if not labels:
            return

        now = _now_iso()
        frame = self._load_frame()
        for row_id, label in labels.items():
            mask = frame["row_id"] == row_id
            frame.loc[mask, "label"] = label
            frame.loc[mask, "classified_at"] = now
        self._save_frame(frame)

    def ready_batch(
        self,
        *,
        min_batch: int,
        fallback_min_batch: int,
        fallback_max_age_days: int,
        max_prompt_messages: int,
        max_maybe_messages: int,
        force: bool = False,
        force_max_age_days: int | None = None,
    ) -> ContextBatch | None:
        selected = self.pending_selected_messages()
        if force and force_max_age_days is not None:
            cutoff = datetime.now(UTC) - timedelta(days=force_max_age_days)
            selected = [message for message in selected if message.date >= cutoff]
        prompt_messages = _select_prompt_messages(
            selected,
            max_prompt_messages=max_prompt_messages,
            max_maybe_messages=max_maybe_messages,
        )
        pending_keep_count = _label_count(selected, "keep")
        pending_maybe_count = _label_count(selected, "maybe")
        included_keep_count = _label_count(prompt_messages, "keep")
        included_maybe_count = _label_count(prompt_messages, "maybe")

        if len(selected) >= min_batch:
            return ContextBatch(
                messages=prompt_messages,
                reason=f"min_batch:{len(selected)}>={min_batch}",
                pending_keep_count=pending_keep_count,
                pending_maybe_count=pending_maybe_count,
                included_keep_count=included_keep_count,
                included_maybe_count=included_maybe_count,
            )

        if force and selected:
            return ContextBatch(
                messages=prompt_messages,
                reason=(
                    f"manual_force_recent:{len(selected)}"
                    + (
                        f"<= {force_max_age_days}d"
                        if force_max_age_days is not None
                        else ""
                    )
                ),
                pending_keep_count=pending_keep_count,
                pending_maybe_count=pending_maybe_count,
                included_keep_count=included_keep_count,
                included_maybe_count=included_maybe_count,
            )

        if len(selected) >= fallback_min_batch:
            oldest = selected[0].date
            if datetime.now(UTC) - oldest >= timedelta(days=fallback_max_age_days):
                return ContextBatch(
                    messages=prompt_messages,
                    reason=(
                        f"fallback:{len(selected)}>={fallback_min_batch},"
                        f"oldest>={fallback_max_age_days}d"
                    ),
                    pending_keep_count=pending_keep_count,
                    pending_maybe_count=pending_maybe_count,
                    included_keep_count=included_keep_count,
                    included_maybe_count=included_maybe_count,
                )
        return None

    def pending_selected_messages(self) -> list[QueuedContextMessage]:
        frame = self._load_frame()
        if frame.empty:
            return []
        mask = frame["label"].isin(["maybe", "keep"]) & frame["used_at"].isna()
        subset = frame[mask].sort_values(["message_date", "row_id"])
        return [_queued_from_row(row) for _, row in subset.iterrows()]

    def mark_used(self, message_ids: Iterable[int], *, bio: str) -> int:
        ids = list(message_ids)
        if not ids:
            return 0

        batch_key = hashlib.sha256(
            f"{_now_iso()}:{','.join(map(str, ids))}:{bio}".encode("utf-8")
        ).hexdigest()[:16]
        now = _now_iso()
        frame = self._load_frame()
        mask = frame["row_id"].isin(ids)
        changed = int(mask.sum())
        frame.loc[mask, "used_at"] = now
        frame.loc[mask, "batch_key"] = batch_key
        self._save_frame(frame)
        return changed

    def _load_frame(self) -> pd.DataFrame:
        if not self._dataset_path.exists():
            return pd.DataFrame(columns=self._columns)
        frame = pd.read_parquet(self._dataset_path)
        for column in self._columns:
            if column not in frame.columns:
                frame[column] = pd.NA
        return frame[list(self._columns)]

    def _save_frame(self, frame: pd.DataFrame) -> None:
        frame = frame.copy()
        frame = frame.sort_values(["message_date", "row_id"], kind="stable")
        frame.to_parquet(self._dataset_path, index=False)


class Mix0035Classifier:
    """Inference-only runtime for the selected two-stage experiment pipeline."""

    def __init__(
        self,
        model_dir: Path,
        *,
        stage1_model_name: str,
        stage2_model_name: str,
        feature_embedding_model_name: str,
        enable_nli_score: bool,
        nli_model_name: str,
        max_message_length: int = 0,
    ) -> None:
        self._model_dir = model_dir
        self._stage1_model_name = stage1_model_name
        self._stage2_model_name = stage2_model_name
        self._feature_embedding_model_name = feature_embedding_model_name
        self._enable_nli_score = enable_nli_score
        self._nli_model_name = nli_model_name
        self._max_message_length = max_message_length
        self._stage1: CatBoostClassifier | None = None
        self._stage2 = None
        self._stage1_numeric_scaler = None
        self._stage1_embedder: SentenceTransformer | None = None
        self._stage2_embedder: SentenceTransformer | None = None
        self._feature_embedder: SentenceTransformer | None = None
        self._feature_anchor_embeddings = None
        self._feature_negative_anchor_embeddings = None
        self._nli_pipeline = None

    def classify(self, messages: list[QueuedContextMessage]) -> dict[int, ContextLabel]:
        if not messages:
            return {}

        result: dict[int, ContextLabel] = {}
        eligible: list[QueuedContextMessage] = []
        if self._max_message_length > 0:
            for message in messages:
                if len(message.text) > self._max_message_length:
                    result[message.id] = "drop"
                else:
                    eligible.append(message)
            if len(eligible) < len(messages):
                logger.info(
                    "Pre-dropped %d/%d messages exceeding max_message_length=%d",
                    len(messages) - len(eligible),
                    len(messages),
                    self._max_message_length,
                )
            if not eligible:
                return result
        else:
            eligible = messages

        self._load()

        messages = eligible
        texts = [message.text for message in messages]
        stage1_embeddings = self._stage1_embedder.encode(
            texts, batch_size=64, show_progress_bar=False, normalize_embeddings=False
        )
        embedding_scores = self._embedding_scores(texts)
        nli_scores = self._nli_scores(texts) if self._enable_nli_score else [-1.0] * len(texts)
        numeric = np.array(
            [
                numeric_features(
                    text,
                    nli_score=nli_score,
                    embedding_score=embedding_score,
                )
                for text, nli_score, embedding_score in zip(
                    texts, nli_scores, embedding_scores, strict=True
                )
            ],
            dtype=np.float32,
        )
        if self._stage1_numeric_scaler is not None:
            numeric = self._stage1_numeric_scaler.transform(numeric)
        stage1_features = np.hstack([stage1_embeddings, numeric])
        stage1_pred = self._stage1.predict(stage1_features)
        stage1_is_not_drop = [
            _normalize_binary(pred) == 1 for pred in np.asarray(stage1_pred).ravel()
        ]

        stage2_indices = [
            index for index, is_not_drop in enumerate(stage1_is_not_drop) if is_not_drop
        ]
        for index, is_not_drop in enumerate(stage1_is_not_drop):
            if not is_not_drop:
                result[messages[index].id] = "drop"

        if stage2_indices:
            stage2_texts = [texts[index] for index in stage2_indices]
            stage2_embeddings = self._stage2_embedder.encode(
                stage2_texts,
                batch_size=64,
                show_progress_bar=False,
                normalize_embeddings=False,
            )
            stage2_pred = self._stage2.predict(stage2_embeddings)
            for source_index, pred in zip(stage2_indices, np.asarray(stage2_pred).ravel()):
                result[messages[source_index].id] = (
                    "keep" if _normalize_binary(pred) == 1 else "maybe"
                )

        counts = {label: list(result.values()).count(label) for label in ("drop", "maybe", "keep")}
        logger.info("Classified %d context messages: %s", len(messages), counts)
        return result

    def _load(self) -> None:
        if self._stage1 is not None:
            return

        stage1_path = self._model_dir / "stage1_catboost.cbm"
        stage2_path = self._model_dir / "stage2_nearest_centroid.pkl"
        scaler_path = self._model_dir / "stage1_numeric_scaler.pkl"
        missing = [str(path) for path in (stage1_path, stage2_path) if not path.exists()]
        if missing:
            raise FileNotFoundError(
                "context_prod model artifacts are missing: "
                + ", ".join(missing)
                + ". Expected mix0035 artifacts in CONTEXT_PROD_MODEL_DIR."
            )

        logger.info("Loading context_prod model artifacts from %s", self._model_dir)
        stage1 = CatBoostClassifier()
        stage1.load_model(stage1_path)
        with stage2_path.open("rb") as fh:
            stage2 = pickle.load(fh)
        if scaler_path.exists():
            with scaler_path.open("rb") as fh:
                self._stage1_numeric_scaler = pickle.load(fh)
            logger.info("Loaded optional stage1 numeric scaler from %s", scaler_path)
        else:
            logger.info("No stage1 numeric scaler found; using raw numeric features")

        self._stage1 = stage1
        self._stage2 = stage2
        self._stage1_embedder = SentenceTransformer(self._stage1_model_name)
        self._stage2_embedder = SentenceTransformer(self._stage2_model_name)
        self._feature_embedder = SentenceTransformer(self._feature_embedding_model_name)
        for embedder in (self._stage1_embedder, self._stage2_embedder, self._feature_embedder):
            current = getattr(embedder, "max_seq_length", None) or 512
            embedder.max_seq_length = min(current, 256)
        logger.info(
            "Feature scores: embedding_model=%s, nli_enabled=%s",
            self._feature_embedding_model_name,
            self._enable_nli_score,
        )

    def _embedding_scores(self, texts: list[str]) -> list[float]:
        anchors = (
            "текущее состояние пользователя",
            "чем пользователь сейчас занимается",
            "проект работа учеба проблема интерес пользователя",
            "сообщение полезно для актуального Telegram bio",
        )
        negative_anchors = (
            "команда боту",
            "короткая реакция без смысла",
            "ссылка без текста",
            "случайный шум для bio",
        )
        if self._feature_anchor_embeddings is None:
            self._feature_anchor_embeddings = self._feature_embedder.encode(
                list(anchors),
                show_progress_bar=False,
                normalize_embeddings=True,
            )
            self._feature_negative_anchor_embeddings = self._feature_embedder.encode(
                list(negative_anchors),
                show_progress_bar=False,
                normalize_embeddings=True,
            )
        vectors = self._feature_embedder.encode(
            texts,
            batch_size=64,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        scores: list[float] = []
        for vector in vectors:
            positive = float(np.max(np.dot(self._feature_anchor_embeddings, vector)))
            negative = float(np.max(np.dot(self._feature_negative_anchor_embeddings, vector)))
            scores.append(max(0.0, min(1.0, (positive - negative + 1.0) / 2.0)))
        return scores

    def _nli_scores(self, texts: list[str]) -> list[float]:
        pipeline = self._load_nli_pipeline()
        positive_hypothesis = (
            "Это сообщение описывает текущее состояние, занятие, проект, проблему "
            "или интерес пользователя."
        )
        negative_hypothesis = (
            "Это сообщение является шумом, командой, реакцией или не несёт смысла "
            "для Telegram bio."
        )
        scores: list[float] = []
        for text in texts:
            positive = _entailment_score(pipeline, text, positive_hypothesis)
            negative = _entailment_score(pipeline, text, negative_hypothesis)
            scores.append(max(0.0, min(1.0, positive * (1.0 - negative))))
        return scores

    def _load_nli_pipeline(self):
        if self._nli_pipeline is None:
            from transformers import pipeline

            self._nli_pipeline = pipeline(
                "text-classification",
                model=self._nli_model_name,
                tokenizer=self._nli_model_name,
                return_all_scores=True,
            )
        return self._nli_pipeline


def numeric_features(
    text: str,
    *,
    nli_score: float = -1.0,
    embedding_score: float = -1.0,
) -> list[float]:
    words = _WORD_RE.findall(text)
    stripped = text.strip()
    heuristic_score = _heuristic_score(stripped, words)
    return [
        float(len(stripped)),
        float(len(words)),
        1.0 if _URL_RE.search(stripped) else 0.0,
        1.0 if stripped.startswith("/") else 0.0,
        heuristic_score,
        float(nli_score),
        float(embedding_score),
    ]


def _heuristic_score(text: str, words: list[str]) -> float:
    if not text:
        return 0.0
    if text.startswith("/"):
        return 0.0
    if _URL_RE.search(text) and len(words) <= 3:
        return 0.05
    if _LOW_VALUE_RE.match(text):
        return 0.05
    score = 0.2
    if len(words) >= 6:
        score += 0.35
    if len(words) >= 14:
        score += 0.2
    if any(char in text for char in "?!"):
        score += 0.05
    if any(marker in text.lower() for marker in ("думаю", "делаю", "нужно", "хочу", "сейчас")):
        score += 0.15
    return min(score, 1.0)


def _select_prompt_messages(
    messages: list[QueuedContextMessage],
    *,
    max_prompt_messages: int,
    max_maybe_messages: int,
) -> list[QueuedContextMessage]:
    maybe_limit = max(0, min(max_maybe_messages, max_prompt_messages))
    keep_limit = max(0, max_prompt_messages - maybe_limit)

    keep = [message for message in messages if message.label == "keep"]
    maybe = [message for message in messages if message.label == "maybe"]

    selected = keep[-keep_limit:] if keep_limit else []
    if maybe_limit:
        selected = [*selected, *maybe[-maybe_limit:]]

    return sorted(selected, key=lambda message: (message.date, message.id))


def _label_count(messages: list[QueuedContextMessage], label: ContextLabel) -> int:
    return sum(1 for message in messages if message.label == label)


def _queued_from_row(row) -> QueuedContextMessage:
    return QueuedContextMessage(
        id=int(row["row_id"]),
        message_key=str(row["message_key"]),
        date=_from_iso(str(row["message_date"])),
        dialog_title=str(row["dialog_title"]),
        text=str(row["text"]),
        label=None if _is_missing(row["label"]) else row["label"],
    )


def _next_row_id(frame: pd.DataFrame) -> int:
    if frame.empty or frame["row_id"].isna().all():
        return 1
    return int(frame["row_id"].max()) + 1


def _is_missing(value: object) -> bool:
    return value is None or pd.isna(value)


def _normalize_binary(value: object) -> int:
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    label = str(value).strip().lower()
    if label in {"0", "0.0", "drop", "maybe", "false"}:
        return 0
    if label in {"1", "1.0", "not_drop", "keep", "true"}:
        return 1
    raise ValueError(f"Unknown binary label from model: {value!r}")


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
    if math.isnan(best):
        return 0.0
    return best


def _to_iso(value: datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.astimezone(UTC).isoformat()


def _from_iso(value: str) -> datetime:
    return datetime.fromisoformat(value).astimezone(UTC)


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()
