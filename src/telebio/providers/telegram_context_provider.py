"""Production context-based bio provider."""

from __future__ import annotations

import logging
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import httpx

from telebio.telegram_context import (
    ContextBatch,
    ContextBatchNotReady,
    TelegramContextStore,
    Mix0035Classifier,
)
from telebio.services.telegram import TelegramService

if TYPE_CHECKING:
    from telebio.telegram_context import QueuedContextMessage

logger = logging.getLogger(__name__)

_TELEGRAM_BIO_MAX_LENGTH = 70
_YANDEX_COMPLETION_URL = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"


@dataclass(frozen=True, slots=True)
class TelegramContextProviderConfig:
    dataset_path: Path
    report_dir: Path
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
    max_maybe_prompt_messages: int
    dialog_scan_limit: int
    per_dialog_limit: int
    merge_gap_seconds: int = 0
    max_message_length: int = 0
    max_prompt_chars: int = 0


class TelegramContextBioProvider:
    """Collects recent outgoing messages, classifies them, then generates bio."""

    def __init__(
        self,
        *,
        telegram: TelegramService,
        config: TelegramContextProviderConfig,
    ) -> None:
        self._telegram = telegram
        self._config = config
        self._store = TelegramContextStore(config.dataset_path)
        self._classifier = Mix0035Classifier(
            config.model_dir,
            stage1_model_name=config.stage1_model,
            stage2_model_name=config.stage2_model,
            feature_embedding_model_name=config.feature_embedding_model,
            enable_nli_score=config.enable_nli_score,
            nli_model_name=config.nli_model,
            max_message_length=config.max_message_length,
        )
        self._pending_batch: ContextBatch | None = None
        self._pending_bio: str | None = None
        logger.info(
            "TelegramContextBioProvider initialised (dataset=%s, model_dir=%s, fetch_days=%d, "
            "min_batch=%d, fallback=%d/%dd)",
            config.dataset_path,
            config.model_dir,
            config.fetch_days,
            config.min_batch,
            config.fallback_min_batch,
            config.fallback_max_age_days,
        )

    async def get_bio(self, *, force: bool = False) -> str:
        """Return a generated bio from the already collected parquet dataset."""
        self._pending_batch = None
        self._pending_bio = None
        batch = self._ready_batch_or_raise(force=force)

        logger.info(
            "Generating context bio from %d pre-collected messages (reason=%s)",
            len(batch.messages),
            batch.reason,
        )
        _log_prompt_context(batch)

        bio = await self._generate_bio(batch)
        self._pending_batch = batch
        self._pending_bio = bio
        return bio

    async def collect_context(self) -> dict[str, int]:
        """Collect recent outgoing messages and classify only new parquet rows."""
        collected = await self._telegram.collect_recent_outgoing_texts(
            days=self._config.fetch_days,
            limit=self._config.dialog_scan_limit * self._config.per_dialog_limit,
            dialog_scan_limit=self._config.dialog_scan_limit,
            per_dialog_limit=self._config.per_dialog_limit,
            merge_gap_seconds=self._config.merge_gap_seconds,
        )
        changed = self._store.upsert_messages(collected)
        unclassified = self._store.unclassified_messages(limit=2000)
        labels = self._classifier.classify(unclassified) if unclassified else {}
        self._store.save_labels(labels)
        pending = self._store.pending_selected_messages()
        counts = {
            "collected": len(collected),
            "changed_rows": changed,
            "classified": len(labels),
            "pending_keep": sum(1 for message in pending if message.label == "keep"),
            "pending_maybe": sum(1 for message in pending if message.label == "maybe"),
        }
        logger.info("Context collect completed: %s", counts)
        return counts

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
        prompt = _build_prompt(batch, max_chars=self._config.max_prompt_chars)
        report_path = _write_api_report(
            self._config.report_dir,
            batch=batch,
            prompt=prompt,
            model_uri=_model_uri(self._config.yandex_folder_id, self._config.yandex_model),
            generated_bio=None,
        )
        logger.info("Context API request report saved: %s", report_path)
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
                        "Ты пишешь короткое Telegram bio на русском: смешное, "
                        "саркастичное, живое, с легкой самоиронией. Формат: "
                        "ровно 2 строки, каждая начинается с дефиса и пробела. "
                        "Каждая строка ≤ 32 символа и обязательно ЗАКОНЧЕННАЯ "
                        "мысль — нельзя обрывать на предлоге, союзе, частице "
                        "или посреди слова. Если не помещается — переформулируй "
                        "короче, не обрезай. Без кавычек, без эмодзи, без "
                        "пояснений. Не раскрывай личные данные и имена собеседников."
                    ),
                },
                {"role": "user", "text": prompt},
            ],
        }
        async with httpx.AsyncClient(timeout=60) as client:
            response = await client.post(_YANDEX_COMPLETION_URL, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()

        raw_text = (
            data["result"]["alternatives"][0]["message"]["text"]
            .strip()
            .strip('"')
            .strip("'")
        )
        text = _normalise_generated_bio(raw_text)
        if not text:
            raise RuntimeError("YandexGPT returned an empty bio")
        _write_api_report(
            self._config.report_dir,
            batch=batch,
            prompt=prompt,
            model_uri=_model_uri(self._config.yandex_folder_id, self._config.yandex_model),
            generated_bio=text,
            report_path=report_path,
        )
        return text

    def _ready_batch_or_raise(self, *, force: bool = False) -> ContextBatch:
        batch = self._store.ready_batch(
            min_batch=self._config.min_batch,
            fallback_min_batch=self._config.fallback_min_batch,
            fallback_max_age_days=self._config.fallback_max_age_days,
            max_prompt_messages=self._config.max_prompt_messages,
            max_maybe_messages=self._config.max_maybe_prompt_messages,
            force=force,
            force_max_age_days=self._config.fetch_days,
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
        return batch


def _build_prompt(batch: ContextBatch, *, max_chars: int = 0) -> str:
    """Assemble the YandexGPT prompt, dropping messages to fit ``max_chars``.

    ``max_chars <= 0`` disables the budget. When the rendered prompt is over
    budget, the oldest ``maybe`` messages are dropped first, then the oldest
    ``keep`` messages, until it fits (instructions are always kept).
    """
    keep_messages = [message for message in batch.messages if message.label == "keep"]
    maybe_messages = [message for message in batch.messages if message.label == "maybe"]

    prompt = _render_prompt(keep_messages, maybe_messages)
    if max_chars <= 0 or len(prompt) <= max_chars:
        return prompt

    dropped_keep = dropped_maybe = 0
    while len(prompt) > max_chars and (maybe_messages or keep_messages):
        if maybe_messages:
            maybe_messages.pop(0)
            dropped_maybe += 1
        else:
            keep_messages.pop(0)
            dropped_keep += 1
        prompt = _render_prompt(keep_messages, maybe_messages)

    logger.warning(
        "Prompt exceeded %d chars; dropped %d keep + %d maybe to fit (final=%d)",
        max_chars,
        dropped_keep,
        dropped_maybe,
        len(prompt),
    )
    return prompt


def _render_prompt(
    keep_messages: list["QueuedContextMessage"],
    maybe_messages: list["QueuedContextMessage"],
) -> str:
    lines = [
        "- Задача: придумай новое Telegram bio по моим последним сообщениям.",
        "- Стиль: смешно, саркастично, немного дерзко, но без кринжового стендапа.",
        "- Не делай сухое summary. Лучше возьми 1-2 конкретные детали из сообщений "
        "и преврати их в короткую фразу.",
        "- Можно прямо использовать удачные куски формулировок из сообщений, но без кавычек.",
        "- Не пиши имена собеседников, даты, команды, ссылки и приватные детали.",
        "- Сильный контекст важнее слабого.",
        "- Слабые сигналы используй только если они добавляют смешную или точную деталь.",
        "- Не делай bio только по одному слабому сообщению.",
        "- drop-сообщения уже отфильтрованы и не переданы в этот запрос.",
        "- Финальный ответ: только bio, ровно 2 строки.",
        "- Каждая строка итогового bio должна начинаться с - и пробела.",
        "- Каждая строка ≤ 32 символа И обязательно ЗАКОНЧЕННАЯ мысль.",
        "- Нельзя обрывать строку на предлоге (в, на, за, с, к, у, для, "
        "из, по, при, о, об), союзе (и, а, но, или, что), частице (не, "
        "же, ли, бы, уже) или посреди слова.",
        "- Если мысль не помещается в 32 символа — переформулируй короче, "
        "не обрезай на полуслове.",
        "- Всё bio суммарно ≤ 70 символов (это лимит Telegram).",
        "",
        "- Сильный контекст keep:",
    ]
    if keep_messages:
        for message in keep_messages:
            lines.append(_prompt_message_line(message))
    else:
        lines.append("- нет")

    lines.extend(["", "- Слабые дополнительные сигналы maybe:"])
    if maybe_messages:
        for message in maybe_messages:
            lines.append(_prompt_message_line(message))
    else:
        lines.append("- нет")

    return "\n".join(lines)


def _prompt_message_line(message: "QueuedContextMessage") -> str:
    return (
        f"- [{message.date:%Y-%m-%d %H:%M}, {message.dialog_title}] "
        f"{_one_line(message.text, limit=500)}"
    )


def _log_prompt_context(batch: ContextBatch) -> None:
    logger.info(
        "Context prompt split | pending_keep=%d pending_maybe=%d "
        "included_keep=%d included_maybe=%d excluded_keep_by_limit=%d "
        "excluded_maybe_by_limit=%d drop_sent=0",
        batch.pending_keep_count,
        batch.pending_maybe_count,
        batch.included_keep_count,
        batch.included_maybe_count,
        max(0, batch.pending_keep_count - batch.included_keep_count),
        max(0, batch.pending_maybe_count - batch.included_maybe_count),
    )

    keep_index = 0
    maybe_index = 0
    for message in batch.messages:
        if message.label == "keep":
            keep_index += 1
            logger.info(
                "Context prompt KEEP %03d | %s | %s | %s",
                keep_index,
                message.date.isoformat(),
                message.dialog_title,
                _one_line(message.text),
            )
        elif message.label == "maybe":
            maybe_index += 1
            logger.info(
                "Context prompt MAYBE %03d | %s | %s | %s",
                maybe_index,
                message.date.isoformat(),
                message.dialog_title,
                _one_line(message.text),
            )


def _write_api_report(
    report_dir: Path,
    *,
    batch: ContextBatch,
    prompt: str,
    model_uri: str,
    generated_bio: str | None,
    report_path: Path | None = None,
) -> Path:
    report_dir.mkdir(parents=True, exist_ok=True)
    if report_path is None:
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        report_path = report_dir / f"context_api_request_{stamp}.json"
    payload = {
        "created_at": datetime.now(UTC).isoformat(),
        "model_uri": model_uri,
        "reason": batch.reason,
        "counts": {
            "pending_keep": batch.pending_keep_count,
            "pending_maybe": batch.pending_maybe_count,
            "included_keep": batch.included_keep_count,
            "included_maybe": batch.included_maybe_count,
            "drop_sent": 0,
        },
        "messages": [
            {
                "row_id": message.id,
                "message_key": message.message_key,
                "date": message.date.isoformat(),
                "dialog": message.dialog_title,
                "label": message.label,
                "text": message.text,
            }
            for message in batch.messages
        ],
        "prompt": prompt,
        "generated_bio": generated_bio,
    }
    report_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return report_path


def _model_uri(folder_id: str, model: str) -> str:
    if model.startswith("gpt://"):
        return model
    return f"gpt://{folder_id}/{model}"


def _one_line(text: str, *, limit: int = 240) -> str:
    value = " ".join(text.split())
    if len(value) <= limit:
        return value
    return value[: limit - 1].rstrip() + "…"


def _normalise_generated_bio(text: str) -> str:
    lines = []
    for raw_line in text.splitlines():
        line = raw_line.strip().strip('"').strip("'")
        if not line:
            continue
        if line.startswith("-"):
            line = line[1:].strip()
        line = " ".join(line.split())
        if not line:
            continue
        lines.append(f"- {_truncate_line(line, limit=30)}")

    if not lines and text.strip():
        lines = [f"- {_truncate_line(text.strip(), limit=30)}"]

    selected: list[str] = []
    for line in lines[:3]:
        candidate = "\n".join([*selected, line])
        if len(candidate) > _TELEGRAM_BIO_MAX_LENGTH:
            break
        selected.append(line)

    result = "\n".join(selected).strip()
    if len(result) <= _TELEGRAM_BIO_MAX_LENGTH:
        return result
    return _truncate_line(result, limit=_TELEGRAM_BIO_MAX_LENGTH)


def _truncate_line(text: str, *, limit: int) -> str:
    if len(text) <= limit:
        return text
    words = text.split()
    result = ""
    for word in words:
        candidate = f"{result} {word}".strip()
        if len(candidate) > limit:
            break
        result = candidate
    if result:
        return result.rstrip(" ,.;:-")
    return text[:limit].rstrip(" ,.;:-")
