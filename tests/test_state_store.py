"""Tests for the SQLite StateStore."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from telebio.services.bot import BotService
from telebio.services.state_store import StateStore


def _store(tmp_path: Path) -> StateStore:
    return StateStore(tmp_path / "state.sqlite3")


class TestSettingsKV:

    def test_load_empty(self, tmp_path: Path) -> None:
        assert _store(tmp_path).load_settings() == {}

    def test_save_and_load(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        store.save_setting("mode", "telegram_context")
        store.save_setting("paused", "1")
        assert store.load_settings() == {
            "mode": "telegram_context",
            "paused": "1",
        }

    def test_upsert_overwrites(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        store.save_setting("mode", "list")
        store.save_setting("mode", "llm_prompt_generation")
        assert store.load_settings()["mode"] == "llm_prompt_generation"

    def test_persists_across_connections(self, tmp_path: Path) -> None:
        path = tmp_path / "state.sqlite3"
        s1 = StateStore(path)
        s1.save_setting("prompt_name", "Линал")
        s1.close()

        s2 = StateStore(path)
        assert s2.load_settings()["prompt_name"] == "Линал"


class TestBioHistory:

    def test_empty(self, tmp_path: Path) -> None:
        assert _store(tmp_path).load_history() == []

    def test_append_and_load_in_order(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        ts1 = datetime(2026, 1, 1, 12, 0, 0)
        ts2 = datetime(2026, 1, 1, 12, 5, 0)
        store.append_bio(bio="первое", mode="list", ts=ts1)
        store.append_bio(bio="второе", mode="telegram_context", ts=ts2)

        rows = store.load_history()
        assert [r["bio"] for r in rows] == ["первое", "второе"]
        assert [r["mode"] for r in rows] == ["list", "telegram_context"]
        assert rows[0]["timestamp"] == "2026-01-01 12:00:00"

    def test_history_is_unbounded(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        ts = datetime(2026, 1, 1, 0, 0, 0)
        for i in range(50):
            store.append_bio(bio=f"bio {i}", mode="list", ts=ts)
        assert len(store.load_history()) == 50


class TestBotServiceRestore:

    def _bot(self, store: StateStore, mode: str = "list") -> BotService:
        bot = BotService(
            bot_token="tok",
            api_id=1,
            api_hash="hash",
            current_mode={"mode": mode},
            store=store,
        )
        return bot

    def test_fresh_store_keeps_defaults(self, tmp_path: Path) -> None:
        bot = self._bot(_store(tmp_path), mode="list")
        assert bot.current_mode["mode"] == "list"
        assert bot.paused is False
        assert bot.last_bio == ""
        assert list(bot.history) == []

    def test_persists_mode_change(self, tmp_path: Path) -> None:
        path = tmp_path / "state.sqlite3"
        bot1 = self._bot(StateStore(path))
        bot1.set_mode("telegram_context")
        bot1.set_prompt("Абсурд")
        bot1.toggle_pause()

        bot2 = self._bot(StateStore(path), mode="list")
        assert bot2.current_mode["mode"] == "telegram_context"
        assert bot2.current_mode["prompt_name"] == "Абсурд"
        assert bot2.paused is True

    def test_persists_bio_history(self, tmp_path: Path) -> None:
        path = tmp_path / "state.sqlite3"
        bot1 = self._bot(StateStore(path))
        bot1.record_bio_update("bio A", "list")
        bot1.record_bio_update("bio B", "telegram_context")

        bot2 = self._bot(StateStore(path))
        bios = [row["bio"] for row in bot2.history]
        assert bios == ["bio A", "bio B"]
        assert bot2.last_bio == "bio B"
        assert bot2.last_update is not None
