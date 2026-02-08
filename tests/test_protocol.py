"""Tests for BioProvider protocol conformance."""

from __future__ import annotations

from pathlib import Path

from telebio.providers.base import BioProvider
from telebio.providers.list_provider import ListBioProvider
from telebio.providers.llm_provider import LLMBioProvider


class TestProtocolConformance:
    """Both providers must satisfy the BioProvider protocol."""

    def test_list_provider_is_bio_provider(self, phrases_file: Path) -> None:
        provider = ListBioProvider(phrases_file)
        assert isinstance(provider, BioProvider)

    def test_llm_provider_is_bio_provider(self, examples_file: Path) -> None:
        provider = LLMBioProvider(
            api_key="k",
            folder_id="f",
            examples_path=examples_file,
        )
        assert isinstance(provider, BioProvider)
