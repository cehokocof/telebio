"""Production context-based bio provider."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import httpx

from telebio.context_prod import (
    ContextBatch,
    ContextBatchNotReady,
    ContextProdStore,
    Mix0035Classifier,
)
from telebio.services.telegram import TelegramService

logger = logging.getLogger(__name__)

_TELEGRAM_BIO_MAX_LENGTH = 70
_YANDEX_COMPLETION_URL = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"


@dataclass(frozen=True, slots=True)
class ContextProdProviderConfig:
    db_path: Path
    model_dir: Path
    stage1_model: str
    stage2_model: str
    feature_embedding_model: str
    enable_nli_score: bool
    nli_model: str
    yandex_api_key: str
    yandex_folder_id: str
    yandex_model: str
    yandex_temperature: float
    fetch_days: int
    min_batch: int
    fallback_min_batch: int
    fallback_max_age_days: int
    max_prompt_messages: int
    dialog_scan_limit: int
    per_dialog_limit: int


class ContextProdBioProvider:
    """Collects recent outgoing messages, classifies them, then generates bio."""

    def __init__(
        self,
        *,
        telegram: TelegramService,
        config: ContextProdProviderConfig,
    ) -> None:
        self._telegram = telegram
        self._config = config
        self._store = ContextProdStore(config.db_path)
        self._classifier = Mix0035Classifier(
            config.model_dir,
            stage1_model_name=config.stage1_model,
            stage2_model_name=config.stage2_model,
            feature_embedding_model_name=config.feature_embedding_model,
            enable_nli_score=config.enable_nli_score,
            nli_model_name=config.nli_model,
        )
        self._pending_batch: ContextBatch | None = None
        self._pending_bio: str | None = None
        logger.info(
            "ContextProdBioProvider initialised (db=%s, model_dir=%s, fetch_days=%d, "
            "min_batch=%d, fallback=%d/%dd)",
            config.db_path,
            config.model_dir,
            config.fetch_days,
            config.min_batch,
            config.fallback_min_batch,
            config.fallback_max_age_days,
        )

    async def get_bio(self) -> str:
        """Refresh context queue and return a new generated bio when ready."""
        self._pending_batch = None
        self._pending_bio = None

        collected = await self._telegram.collect_recent_outgoing_texts(
            days=self._config.fetch_days,
            limit=self._config.dialog_scan_limit * self._config.per_dialog_limit,
            dialog_scan_limit=self._config.dialog_scan_limit,
            per_dialog_limit=self._config.per_dialog_limit,
        )
        inserted = self._store.upsert_messages(collected)
        logger.info("Context queue refreshed: collected=%d inserted=%d", len(collected), inserted)

        unclassified = self._store.unclassified_messages(limit=2000)
        if unclassified:
            labels = self._classifier.classify(unclassified)
            self._store.save_labels(labels)
        else:
            logger.info("No new context messages require classification")

        batch = self._store.ready_batch(
            min_batch=self._config.min_batch,
            fallback_min_batch=self._config.fallback_min_batch,
            fallback_max_age_days=self._config.fallback_max_age_days,
            max_prompt_messages=self._config.max_prompt_messages,
        )
        pending_count = len(self._store.pending_selected_messages())
        if batch is None:
            raise ContextBatchNotReady(
                "Context batch is not ready: "
                f"pending maybe/keep={pending_count}, "
                f"need {self._config.min_batch} or fallback "
                f"{self._config.fallback_min_batch} after "
                f"{self._config.fallback_max_age_days}d"
            )

        logger.info(
            "Generating context bio from %d messages (reason=%s, pending=%d)",
            len(batch.messages),
            batch.reason,
            pending_count,
        )
        for index, message in enumerate(batch.messages, start=1):
            logger.info(
                "Context batch message %03d | %s | %s | %s | %s",
                index,
                message.date.isoformat(),
                message.label,
                message.dialog_title,
                _one_line(message.text),
            )

        bio = await self._generate_bio(batch)
        self._pending_batch = batch
        self._pending_bio = bio
        return bio

    async def commit_successful_update(self, bio: str) -> None:
        """Mark messages used only after Telegram accepted the generated bio."""
        if self._pending_batch is None:
            return
        if self._pending_bio != bio:
            logger.warning("Generated bio changed before commit; context batch left pending")
            return

        marked = self._store.mark_used([message.id for message in self._pending_batch.messages], bio=bio)
        logger.info("Marked %d context messages as used after bio update", marked)
        self._pending_batch = None
        self._pending_bio = None

    async def _generate_bio(self, batch: ContextBatch) -> str:
        prompt = _build_prompt(batch)
        headers = {
            "Authorization": f"Api-Key {self._config.yandex_api_key}",
            "x-folder-id": self._config.yandex_folder_id,
        }
        payload = {
            "modelUri": _model_uri(
                self._config.yandex_folder_id,
                self._config.yandex_model,
            ),
            "completionOptions": {
                "stream": False,
                "temperature": self._config.yandex_temperature,
                "maxTokens": 80,
            },
            "messages": [
                {
                    "role": "system",
                    "text": (
                        "Ты пишешь короткое Telegram bio на русском. "
                        "До 70 символов. Без кавычек, без эмодзи, без пояснений. "
                        "Сохраняй живой стиль автора, но не копируй личные данные."
                    ),
                },
                {"role": "user", "text": prompt},
            ],
        }
        async with httpx.AsyncClient(timeout=60) as client:
            response = await client.post(_YANDEX_COMPLETION_URL, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()

        text = (
            data["result"]["alternatives"][0]["message"]["text"]
            .strip()
            .strip('"')
            .strip("'")
        )
        if len(text) > _TELEGRAM_BIO_MAX_LENGTH:
            text = text[:_TELEGRAM_BIO_MAX_LENGTH].rstrip()
        if not text:
            raise RuntimeError("YandexGPT returned an empty bio")
        return text


def _build_prompt(batch: ContextBatch) -> str:
    lines = [
        "Сделай новое bio по моим последним содержательным исходящим сообщениям.",
        "Цель: отразить, что у меня сейчас происходит, без пересказа и без имен собеседников.",
        "",
        "Сообщения:",
    ]
    for message in batch.messages:
        lines.append(
            f"- [{message.date:%Y-%m-%d %H:%M}, {message.label}, {message.dialog_title}] "
            f"{_one_line(message.text, limit=500)}"
        )
    return "\n".join(lines)


def _model_uri(folder_id: str, model: str) -> str:
    if model.startswith("gpt://"):
        return model
    return f"gpt://{folder_id}/{model}"


def _one_line(text: str, *, limit: int = 240) -> str:
    value = " ".join(text.split())
    if len(value) <= limit:
        return value
    return value[: limit - 1].rstrip() + "…"
