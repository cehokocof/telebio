"""Named system prompts for the llm_prompt_generation provider.

The prompts file is a JSON array of objects: ``[{"name": ..., "system": ...}]``.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class Prompt:
    name: str
    system: str


_FALLBACK = Prompt(
    name="default",
    system=(
        "Ты придумываешь короткое смешное Telegram bio на русском. "
        "Только текст, без кавычек и пояснений, до 70 символов."
    ),
)


def load_prompts(path: Path) -> list[Prompt]:
    """Load named prompts; fall back to a single built-in prompt if absent/invalid."""
    if not path.exists():
        logger.warning("Prompts file not found: %s — using built-in default", path)
        return [_FALLBACK]

    with path.open(encoding="utf-8") as fh:
        data = json.load(fh)

    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON array of prompts in {path}")

    prompts: list[Prompt] = []
    for entry in data:
        name = str(entry["name"]).strip()
        system = str(entry["system"]).strip()
        if name and system:
            prompts.append(Prompt(name=name, system=system))

    if not prompts:
        logger.warning("Prompts file %s is empty — using built-in default", path)
        return [_FALLBACK]

    logger.info("Loaded %d named prompts from %s", len(prompts), path)
    return prompts


def get_prompt(prompts: list[Prompt], name: str | None) -> Prompt:
    """Return the prompt with *name*, or the first one as a fallback."""
    if name:
        for prompt in prompts:
            if prompt.name == name:
                return prompt
    return prompts[0]
