from __future__ import annotations

import json

import pandas as pd

from labeling.core.dataset import build_labeling_dataset, save_dataset, set_label, stable_message_id
from labeling.ui.catboost_review import _source_for_label


def test_build_labeling_dataset_merges_scores(tmp_path):
    messages = [
        {"date": "2026-05-15T12:00:00+00:00", "dialog": "A", "text": "long useful text"},
        {"date": "2026-05-15T12:01:00+00:00", "dialog": "B", "text": "ok"},
    ]
    messages_path = tmp_path / "messages.json"
    messages_path.write_text(json.dumps(messages), encoding="utf-8")

    heuristic_path = tmp_path / "heuristic.json"
    heuristic_path.write_text(
        json.dumps([{**messages[0], "heuristic_score": 0.7}]),
        encoding="utf-8",
    )
    nli_path = tmp_path / "nli.json"
    nli_path.write_text(
        json.dumps([{**messages[0], "nli_score": 0.8}]),
        encoding="utf-8",
    )
    embedding_path = tmp_path / "embedding.json"
    embedding_path.write_text(
        json.dumps([{**messages[0], "nli_score": 0.9}]),
        encoding="utf-8",
    )

    dataset = build_labeling_dataset(
        messages_path=messages_path,
        heuristic_report_path=heuristic_path,
        nli_report_path=nli_path,
        embedding_report_path=embedding_path,
    )

    first = dataset[dataset["dialog"] == "A"].iloc[0]
    second = dataset[dataset["dialog"] == "B"].iloc[0]

    assert first["heuristic_score"] == 0.7
    assert first["nli_score"] == 0.8
    assert first["embedding_score"] == 0.9
    assert pd.isna(second["heuristic_score"])
    assert pd.isna(second["label"])


def test_build_labeling_dataset_preserves_existing_labels(tmp_path):
    message = {"date": "2026-05-15T12:00:00+00:00", "dialog": "A", "text": "long useful text"}
    messages_path = tmp_path / "messages.json"
    messages_path.write_text(json.dumps([message]), encoding="utf-8")
    output_path = tmp_path / "dataset.parquet"

    message_id = stable_message_id(
        date=message["date"],
        dialog=message["dialog"],
        text=message["text"],
    )
    existing = build_labeling_dataset(messages_path=messages_path)
    existing = set_label(existing, message_id, 3)
    save_dataset(existing, output_path)

    rebuilt = build_labeling_dataset(
        messages_path=messages_path,
        existing_dataset_path=output_path,
    )

    assert rebuilt.iloc[0]["label"] == 3
    assert rebuilt.iloc[0]["label_name"] == "keep"


def test_set_label_supports_custom_source(tmp_path):
    message = {"date": "2026-05-15T12:00:00+00:00", "dialog": "A", "text": "long useful text"}
    messages_path = tmp_path / "messages.json"
    messages_path.write_text(json.dumps([message]), encoding="utf-8")

    message_id = stable_message_id(
        date=message["date"],
        dialog=message["dialog"],
        text=message["text"],
    )
    dataset = build_labeling_dataset(messages_path=messages_path)

    updated = set_label(dataset, message_id, 2, source="catboost_accept")

    assert updated.iloc[0]["label"] == 2
    assert updated.iloc[0]["label_name"] == "maybe"
    assert updated.iloc[0]["label_source"] == "catboost_accept"


def test_catboost_review_treats_matching_manual_label_as_accept():
    assert _source_for_label(1, 1) == "catboost_accept"
    assert _source_for_label(2, 1) == "manual_review"
