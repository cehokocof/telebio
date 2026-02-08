"""Abstract base for bio-text providers.

Any new provider (LLM, database, API, etc.) must implement `BioProvider`.
This keeps the rest of the application decoupled from the data source.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class BioProvider(Protocol):
    """Contract that every bio provider must satisfy."""

    async def get_bio(self) -> str:
        """Return the next bio string (max 70 chars for Telegram)."""
        ...
