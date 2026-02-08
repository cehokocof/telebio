"""Placeholder provider for future LLM-based bio generation.

Replace the body of `get_bio` with an actual API call
(e.g. OpenAI ChatCompletion) when you're ready.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_TELEGRAM_BIO_MAX_LENGTH = 70


class LLMBioProvider:
    """Generates bio text via an LLM API (stub)."""

    def __init__(self) -> None:
        # TODO: accept api_key, model, prompt template, etc.
        logger.info("LLMBioProvider initialised (stub)")

    async def get_bio(self) -> str:
        """Call an LLM and return a bio string.

        Currently raises NotImplementedError so the app fails fast
        if someone selects this provider before it's implemented.
        """
        # TODO: implement real LLM call, e.g.:
        #
        # import openai
        # response = await openai.ChatCompletion.acreate(
        #     model="gpt-4o-mini",
        #     messages=[{"role": "user", "content": "..."}],
        # )
        # text = response.choices[0].message.content.strip()
        # return text[:_TELEGRAM_BIO_MAX_LENGTH]

        raise NotImplementedError(
            "LLMBioProvider is not yet implemented. "
            "Set BIO_PROVIDER=list or implement this method."
        )
