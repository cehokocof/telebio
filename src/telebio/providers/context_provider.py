"""Bio provider that generates a current-state bio from sent messages."""

from __future__ import annotations

import logging
from typing import Any

import httpx

from telebio.context_relevance import (
    LocalNliScorer,
    LocalEmbeddingScorer,
    RelevanceOptions,
    select_context_messages,
)
from telebio.services.telegram import ContextMessage, TelegramService
from telebio.state import RuntimeState

logger = logging.getLogger(__name__)

_TELEGRAM_BIO_MAX_LENGTH = 70
_YANDEX_API_URL = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"
_DEFAULT_TIMEOUT = 30
_MAX_CONTEXT_CHARS = 12_000

_SYSTEM_PROMPT = (
    "Ты генерируешь короткое bio для Telegram по недавним исходящим сообщениям "
    "пользователя. Задача bio - отразить, что у пользователя происходит сейчас: "
    "темы, настроение, занятия, фокус внимания. Не цитируй сообщения дословно, "
    "а сожми общий контекст в одну живую фразу. Правила: длина до 70 символов, "
    "только текст bio, без кавычек, пояснений, списков и эмодзи."
)


class ContextUnchanged(RuntimeError):
    """Raised when selected context is unchanged since the last successful update."""


class ContextBioProvider:
    """Generates bio text via YandexGPT using recent outgoing messages."""

    def __init__(
        self,
        *,
        telegram: TelegramService,
        api_key: str,
        folder_id: str,
        days: int,
        limit: int,
        dialog_scan_limit: int | None = None,
        per_dialog_limit: int | None = None,
        relevance_options: RelevanceOptions | None = None,
        runtime_state: RuntimeState | None = None,
        model: str = "yandexgpt-lite/latest",
        temperature: float = 0.7,
    ) -> None:
        self._telegram = telegram
        self._api_key = api_key
        self._folder_id = folder_id
        self._model_uri = f"gpt://{folder_id}/{model}"
        self._temperature = temperature
        self._days = days
        self._limit = limit
        self._dialog_scan_limit = dialog_scan_limit
        self._per_dialog_limit = per_dialog_limit
        self._relevance_options = relevance_options or RelevanceOptions()
        self._runtime_state = runtime_state
        self._semantic_scorer = self._build_semantic_scorer(self._relevance_options)
        self._pending_context_fingerprint: str | None = None

        logger.info(
            "ContextBioProvider initialised (model=%s, days=%d, limit=%d, top_k=%d, min_score=%.2f, nli=%s)",
            self._model_uri,
            days,
            limit,
            self._relevance_options.top_k,
            self._relevance_options.min_score,
            self._relevance_options.semantic_scorer,
        )

    async def get_bio(self) -> str:
        """Collect recent messages, call YandexGPT, and return a bio."""
        messages = await self._telegram.collect_recent_outgoing_texts(
            days=self._days,
            limit=self._limit,
            dialog_scan_limit=self._dialog_scan_limit,
            per_dialog_limit=self._per_dialog_limit,
        )
        if not messages:
            raise RuntimeError(
                "No outgoing text messages found for context bio generation."
            )

        selection = select_context_messages(
            messages,
            options=self._relevance_options,
            scorer=self._semantic_scorer,
        )
        if not selection.selected:
            raise RuntimeError("No relevant context messages selected for bio generation.")

        if (
            self._runtime_state
            and self._runtime_state.last_context_fingerprint == selection.fingerprint
        ):
            raise ContextUnchanged("Selected context has not changed since last update.")

        self._pending_context_fingerprint = selection.fingerprint

        body = self._build_request_body(selection.selected)
        async with httpx.AsyncClient(timeout=_DEFAULT_TIMEOUT) as client:
            response = await client.post(
                _YANDEX_API_URL,
                json=body,
                headers={
                    "Authorization": f"Api-Key {self._api_key}",
                    "x-folder-id": self._folder_id,
                },
            )
            response.raise_for_status()
            data = response.json()

        text = self._extract_text(data)
        logger.info("YandexGPT generated context bio: '%s'", text)
        return text

    def commit_successful_update(self) -> None:
        """Persist selected-context fingerprint after Telegram update succeeds."""
        if self._runtime_state and self._pending_context_fingerprint:
            self._runtime_state.set_last_context_fingerprint(
                self._pending_context_fingerprint
            )
            self._pending_context_fingerprint = None

    @staticmethod
    def _build_semantic_scorer(options: RelevanceOptions):
        if not options.enable_nli:
            return None
        match options.semantic_scorer:
            case "nli":
                return LocalNliScorer(options.nli_model)
            case "embedding":
                return LocalEmbeddingScorer(options.embedding_model)
            case other:
                raise ValueError(f"Unknown semantic scorer: {other}")

    def _build_request_body(self, messages: list[ContextMessage]) -> dict[str, Any]:
        """Construct a YandexGPT payload with message history as context."""
        context = _format_messages(messages)
        return {
            "modelUri": self._model_uri,
            "completionOptions": {
                "stream": False,
                "temperature": self._temperature,
                "maxTokens": 100,
            },
            "messages": [
                {"role": "system", "text": _SYSTEM_PROMPT},
                {
                    "role": "user",
                    "text": (
                        "Вот мои недавние исходящие сообщения, от старых к новым. "
                        "Сгенерируй одно актуальное bio:\n\n"
                        f"{context}"
                    ),
                },
            ],
        }

    @staticmethod
    def _extract_text(data: dict[str, Any]) -> str:
        """Pull the generated text out of the YandexGPT response."""
        try:
            alternatives = data["result"]["alternatives"]
            text = alternatives[0]["message"]["text"].strip()
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(
                f"Unexpected YandexGPT response structure: {data}"
            ) from exc

        if len(text) > _TELEGRAM_BIO_MAX_LENGTH:
            logger.warning(
                "Generated context bio too long (%d chars), truncating: '%s...'",
                len(text),
                text[:30],
            )
            text = text[:_TELEGRAM_BIO_MAX_LENGTH]

        return text


def _format_messages(messages: list[ContextMessage]) -> str:
    """Format messages for a prompt while keeping payload size bounded."""
    lines: list[str] = []
    total = 0
    for message in reversed(messages):
        timestamp = message.date.isoformat(timespec="minutes")
        line = f"- [{timestamp}] {message.dialog}: {message.text}"
        total += len(line) + 1
        if total > _MAX_CONTEXT_CHARS:
            break
        lines.append(line)

    lines.reverse()
    return "\n".join(lines)
