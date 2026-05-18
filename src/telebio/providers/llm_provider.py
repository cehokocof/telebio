"""Bio provider that generates absurd/surreal phrases via YandexGPT API.

Uses the Foundation Models Text Generation REST API (synchronous):
POST https://llm.api.cloud.yandex.net/foundationModels/v1/completion

Examples for few-shot prompting are loaded from a separate JSON file.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import httpx

logger = logging.getLogger(__name__)

_TELEGRAM_BIO_MAX_LENGTH = 70

_SYSTEM_PROMPT = (
    "Ты — циничный, дерзкий и гениальный студент-математик, который перечитал лекции Димы Трушина по Линейной Алгебре (пилотный поток 2025-2026). Ты обожаешь линал, но ненавидишь тупость и громоздкие вычисления. Твой стиль — это смесь стендапа и разговора в курилке мехмата."\
    "Твоя задача: генерировать короткие, хлесткие и очень смешные шутки на тему линейной алгебры, основываясь на стиле и терминах из конспекта Трушина."\
    "Правила тона:"\
        "Используй нецензурную лексику (мат), но уместно: для выражения экзистенциальной боли от вычислений, восторга от красоты теоремы или презрения к вырожденным матрицам. Мат должен быть «в тему», как эмоциональный усилитель."\
        "Ссылайся на локальные мемы из лекций (список ниже)."\
        "Будь дерзким. Ты смотришь на гуманитариев как на людей, не знающих, что такое тензорное произведение."\

    "Обязательно используй этот контекст из лекций:"\
    "— «Метод пристального взгляда» (используй это как универсальное решение всех проблем, когда лень доказывать)."\
    "— «Большие дяди и тети» (те, кто используют тензоры и знают, что такое SVD)."\
    "— «Жульничество» (когда определение выглядит подозрительно просто)."\
    "— Алгоритм Гаусса — это боль, страдания и кубическая сложность."\
    "— Двойственное пространство — это когда ты пытаешься понять душу, а получаешь линейный функционал."\
    "— Поля (особенно когда 1+1=0) — это место, где здравый смысл идет нах*й."\
    "Constraints:\n"
    "1. Длина: до 60 символов.\n"
    "2. Тон: хаотичный, непредсказуемый, абсурдный.\n"
    "3. Сочетай несочетаемое (еду и технологии, животных и политику, космос и быт).\n"
    "4. Выводи ТОЛЬКО текст."
)

_YANDEX_API_URL = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"
_DEFAULT_TIMEOUT = 30  # seconds


class LLMBioProvider:
    """Generates bio text via YandexGPT Foundation Models API."""

    def __init__(
        self,
        *,
        api_key: str,
        folder_id: str,
        examples_path: Path,
        model: str = "yandexgpt-lite/latest",
        temperature: float = 0.9,
    ) -> None:
        self._api_key = api_key
        self._folder_id = folder_id
        self._model_uri = f"gpt://{folder_id}/{model}"
        self._temperature = temperature
        self._examples = self._load_examples(examples_path)

        logger.info(
            "LLMBioProvider initialised (model=%s, examples=%d)",
            self._model_uri,
            len(self._examples),
        )

    # ------------------------------------------------------------------
    # Public API (matches BioProvider protocol)
    # ------------------------------------------------------------------

    async def get_bio(self) -> str:
        """Call YandexGPT and return a generated bio string."""
        body = self._build_request_body()

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
        logger.info("YandexGPT generated bio: '%s'", text)
        return text

    # ------------------------------------------------------------------
    # Request building
    # ------------------------------------------------------------------

    def _build_request_body(self) -> dict[str, Any]:
        """Construct the JSON payload with system prompt + few-shot examples."""
        messages: list[dict[str, str]] = [
            {"role": "system", "text": _SYSTEM_PROMPT},
        ]

        # Few-shot: each example is presented as a user request → assistant response pair
        for example in self._examples:
            messages.append({"role": "user", "text": "Придумай фразу для био."})
            messages.append({"role": "assistant", "text": example})

        # Final user turn that triggers generation
        messages.append({"role": "user", "text": "Придумай фразу для био."})

        return {
            "modelUri": self._model_uri,
            "completionOptions": {
                "stream": False,
                "temperature": self._temperature,
                "maxTokens": 100,
            },
            "messages": messages,
        }

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_text(data: dict[str, Any]) -> str:
        """Pull the generated text out of the API response.

        Response shape:
        {
          "result": {
            "alternatives": [
              { "message": { "role": "assistant", "text": "..." } }
            ],
            ...
          }
        }
        """
        try:
            alternatives = data["result"]["alternatives"]
            text = alternatives[0]["message"]["text"].strip()
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(
                f"Unexpected YandexGPT response structure: {data}"
            ) from exc

        # Enforce Telegram bio length limit
        if len(text) > _TELEGRAM_BIO_MAX_LENGTH:
            logger.warning(
                "Generated bio too long (%d chars), truncating: '%s…'",
                len(text),
                text[:30],
            )
            text = text[:_TELEGRAM_BIO_MAX_LENGTH]

        return text

    # ------------------------------------------------------------------
    # Examples loading
    # ------------------------------------------------------------------

    @staticmethod
    def _load_examples(path: Path) -> list[str]:
        """Load few-shot examples from a JSON file (array of strings)."""
        if not path.exists():
            logger.warning("Examples file not found: %s — proceeding without examples", path)
            return []

        with path.open(encoding="utf-8") as fh:
            data = json.load(fh)

        if not isinstance(data, list) or not all(isinstance(s, str) for s in data):
            raise ValueError(f"Expected a JSON array of strings in {path}")

        logger.info("Loaded %d few-shot examples from %s", len(data), path)
        return data
