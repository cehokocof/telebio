"""Bio provider that picks phrases from a local JSON file.

The file must contain a JSON array of strings:
    ["phrase one", "phrase two", ...]

Selection strategy: sequential with wrap-around.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_TELEGRAM_BIO_MAX_LENGTH = 70


class ListBioProvider:
    """Reads phrases from a JSON file and yields them sequentially."""

    def __init__(self, phrases_path: Path) -> None:
        self._phrases = self._load(phrases_path)
        self._index = 0
        logger.info("Loaded %d phrases from %s", len(self._phrases), phrases_path)

    # ------------------------------------------------------------------
    # Public API (matches BioProvider protocol)
    # ------------------------------------------------------------------

    async def get_bio(self) -> str:
        """Return the next phrase, cycling through the list."""
        if not self._phrases:
            raise RuntimeError("Phrase list is empty — nothing to set as bio.")

        phrase = self._phrases[self._index]
        self._index = (self._index + 1) % len(self._phrases)
        return phrase

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _load(path: Path) -> list[str]:
        if not path.exists():
            raise FileNotFoundError(f"Phrases file not found: {path}")

        with path.open(encoding="utf-8") as fh:
            data = json.load(fh)

        if not isinstance(data, list) or not all(isinstance(s, str) for s in data):
            raise ValueError(f"Expected a JSON array of strings in {path}")

        valid: list[str] = []
        for phrase in data:
            if len(phrase) > _TELEGRAM_BIO_MAX_LENGTH:
                logger.warning(
                    "Phrase truncated to %d chars: '%s…'",
                    _TELEGRAM_BIO_MAX_LENGTH,
                    phrase[:30],
                )
                phrase = phrase[:_TELEGRAM_BIO_MAX_LENGTH]
            valid.append(phrase)

        if not valid:
            raise ValueError("Phrases file is empty.")

        return valid
