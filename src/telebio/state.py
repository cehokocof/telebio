"""Small persisted runtime state for bot-controlled settings."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

VALID_MODES = {"list", "llm", "context"}
MAX_CONTEXT_DAYS = 90
MAX_CONTEXT_LIMIT = 2_000


@dataclass(slots=True)
class RuntimeState:
    """Runtime settings that can be changed through the management bot."""

    path: Path
    mode: str
    context_days: int
    context_limit: int
    last_context_fingerprint: str = ""

    @classmethod
    def load(
        cls,
        path: Path,
        *,
        default_mode: str,
        default_context_days: int,
        default_context_limit: int,
    ) -> RuntimeState:
        """Load state from disk, falling back to supplied defaults."""
        state = cls(
            path=path,
            mode=_valid_mode(default_mode, "list"),
            context_days=_valid_int(default_context_days, 14, maximum=MAX_CONTEXT_DAYS),
            context_limit=_valid_int(
                default_context_limit,
                500,
                maximum=MAX_CONTEXT_LIMIT,
            ),
            last_context_fingerprint="",
        )

        if not path.exists():
            return state

        try:
            with path.open(encoding="utf-8") as fh:
                data = json.load(fh)
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Could not load runtime state from %s: %s", path, exc)
            return state

        if not isinstance(data, dict):
            logger.warning("Runtime state must be a JSON object: %s", path)
            return state

        state.mode = _valid_mode(data.get("mode"), state.mode)
        state.context_days = _valid_int(
            data.get("context_days"),
            state.context_days,
            maximum=MAX_CONTEXT_DAYS,
        )
        state.context_limit = _valid_int(
            data.get("context_limit"),
            state.context_limit,
            maximum=MAX_CONTEXT_LIMIT,
        )
        state.last_context_fingerprint = (
            data["last_context_fingerprint"]
            if isinstance(data.get("last_context_fingerprint"), str)
            else ""
        )
        return state

    def save(self) -> None:
        """Persist state to disk."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "mode": self.mode,
            "context_days": self.context_days,
            "context_limit": self.context_limit,
            "last_context_fingerprint": self.last_context_fingerprint,
        }
        with self.path.open("w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False, indent=2)
            fh.write("\n")

    def set_mode(self, mode: str) -> None:
        """Persist a provider mode selected through the bot."""
        if mode not in VALID_MODES:
            raise ValueError(f"Invalid mode: {mode}")
        self.mode = mode
        self.save()

    def set_context_settings(self, days: int, limit: int) -> None:
        """Persist context collection settings."""
        validate_context_settings(days, limit)
        self.context_days = days
        self.context_limit = limit
        self.save()

    def set_last_context_fingerprint(self, fingerprint: str) -> None:
        """Persist fingerprint of the last context that updated Telegram bio."""
        self.last_context_fingerprint = fingerprint
        self.save()


def validate_context_settings(days: int, limit: int) -> None:
    """Validate the user-controlled context collection window."""
    if days <= 0:
        raise ValueError("days must be positive")
    if limit <= 0:
        raise ValueError("limit must be positive")
    if days > MAX_CONTEXT_DAYS:
        raise ValueError(f"days must be <= {MAX_CONTEXT_DAYS}")
    if limit > MAX_CONTEXT_LIMIT:
        raise ValueError(f"limit must be <= {MAX_CONTEXT_LIMIT}")


def _valid_mode(value: Any, default: str) -> str:
    return value if isinstance(value, str) and value in VALID_MODES else default


def _valid_int(value: Any, default: int, *, maximum: int) -> int:
    return value if isinstance(value, int) and 0 < value <= maximum else default
