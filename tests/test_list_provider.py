"""Tests for ListBioProvider."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from telebio.providers.list_provider import ListBioProvider


# ------------------------------------------------------------------
# Construction / loading
# ------------------------------------------------------------------

class TestListBioProviderInit:
    """Tests for __init__ and _load logic."""

    def test_loads_phrases_from_valid_file(self, phrases_file: Path) -> None:
        provider = ListBioProvider(phrases_file)
        assert provider._phrases == ["Фраза раз", "Фраза два", "Фраза три"]

    def test_raises_on_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="not found"):
            ListBioProvider(tmp_path / "nope.json")

    def test_raises_on_invalid_json_structure(self, invalid_json_file: Path) -> None:
        with pytest.raises(ValueError, match="JSON array of strings"):
            ListBioProvider(invalid_json_file)

    def test_raises_on_empty_list(self, empty_list_file: Path) -> None:
        with pytest.raises(ValueError, match="empty"):
            ListBioProvider(empty_list_file)

    def test_truncates_long_phrases(self, long_phrases_file: Path) -> None:
        provider = ListBioProvider(long_phrases_file)
        assert provider._phrases[0] == "Короткая"
        assert len(provider._phrases[1]) == 70

    def test_raises_on_mixed_types(self, tmp_path: Path) -> None:
        p = tmp_path / "bad.json"
        p.write_text('[1, "ok"]', encoding="utf-8")
        with pytest.raises(ValueError, match="JSON array of strings"):
            ListBioProvider(p)


# ------------------------------------------------------------------
# get_bio — sequential cycling
# ------------------------------------------------------------------

class TestListBioProviderGetBio:
    """Tests for the get_bio method (sequential with wrap-around)."""

    @pytest.fixture()
    def provider(self, phrases_file: Path) -> ListBioProvider:
        return ListBioProvider(phrases_file)

    async def test_returns_first_phrase(self, provider: ListBioProvider) -> None:
        assert await provider.get_bio() == "Фраза раз"

    async def test_sequential_order(self, provider: ListBioProvider) -> None:
        results = [await provider.get_bio() for _ in range(3)]
        assert results == ["Фраза раз", "Фраза два", "Фраза три"]

    async def test_wraps_around(self, provider: ListBioProvider) -> None:
        for _ in range(3):
            await provider.get_bio()
        # Should wrap back to first
        assert await provider.get_bio() == "Фраза раз"

    async def test_single_phrase_always_returns_same(
        self, single_phrase_file: Path
    ) -> None:
        provider = ListBioProvider(single_phrase_file)
        for _ in range(5):
            assert await provider.get_bio() == "Одна единственная фраза"
