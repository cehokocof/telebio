from __future__ import annotations

import pandas as pd
import pytest

from labeling.core.catboost_experiments import (
    FEATURE_SETS,
    build_gold_dataset,
    classification_metrics,
    create_stratified_splits,
    _assert_no_leakage,
)


def _dataset() -> pd.DataFrame:
    rows = []
    for label, source, count in [
        (1, "manual", 10),
        (2, "manual_review", 6),
        (3, "catboost_accept", 6),
    ]:
        for index in range(count):
            rows.append(
                {
                    "message_id": f"{label}-{index}",
                    "date": f"2026-05-15T12:{index:02d}:00+00:00",
                    "dialog": "A",
                    "text": f"text {label} {index}",
                    "text_len": 10,
                    "word_count": 3,
                    "has_link": False,
                    "is_command": False,
                    "heuristic_score": 0.5,
                    "nli_score": 0.4,
                    "embedding_score": 0.6,
                    "label": label,
                    "label_name": {1: "drop", 2: "maybe", 3: "keep"}[label],
                    "label_source": source,
                    "catboost_label": label,
                    "catboost_confidence": 0.9,
                }
            )
    rows.append(
        {
            "message_id": "unlabeled",
            "date": "2026-05-15T13:00:00+00:00",
            "dialog": "B",
            "text": "unlabeled text",
            "label": pd.NA,
            "label_source": pd.NA,
        }
    )
    return pd.DataFrame(rows)


def test_build_gold_dataset_uses_verified_sources_and_removes_leakage():
    gold = build_gold_dataset(_dataset())

    assert len(gold) == 22
    assert set(gold["label_source"]) == {"manual", "manual_review", "catboost_accept"}
    assert "catboost_label" not in gold
    assert "catboost_confidence" not in gold
    assert gold["label"].dtype == "int64"


def test_build_gold_dataset_prefers_true_state_as_target():
    dataset = _dataset()
    dataset["true_state"] = dataset["label_name"]
    dataset.loc[dataset["message_id"] == "unlabeled", "true_state"] = "keep"
    dataset.loc[dataset["message_id"] == "unlabeled", "label_source"] = pd.NA

    gold = build_gold_dataset(dataset)

    unlabeled = gold[gold["message_id"] == "unlabeled"].iloc[0]
    assert len(gold) == 23
    assert unlabeled["label"] == 3


def test_create_stratified_splits_is_stable_and_keeps_classes():
    gold = build_gold_dataset(_dataset())

    first = create_stratified_splits(gold, seed=123)
    second = create_stratified_splits(gold, seed=123)

    assert first[["message_id", "split"]].equals(second[["message_id", "split"]])
    assert set(first["split"]) == {"train", "valid", "test"}
    for label in [1, 2, 3]:
        assert set(first[first["label"] == label]["split"]) == {"train", "valid", "test"}


def test_feature_sets_do_not_include_catboost_prediction_columns():
    for columns in FEATURE_SETS.values():
        _assert_no_leakage(columns)

    with pytest.raises(ValueError):
        _assert_no_leakage(["text", "catboost_confidence"])


def test_classification_metrics_reports_macro_and_per_class_scores():
    metrics = classification_metrics([1, 1, 2, 3], [1, 2, 2, 1])

    assert metrics["accuracy"] == 0.5
    assert set(metrics["per_class"]) == {"drop", "maybe", "keep"}
    assert metrics["per_class"]["drop"]["support"] == 2
    assert metrics["confusion_matrix"]["drop"]["maybe"] == 1
