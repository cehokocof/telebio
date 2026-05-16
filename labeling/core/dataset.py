"""Dataset helpers for manual context relevance labeling."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

SCORE_COLUMNS = ["heuristic_score", "nli_score", "embedding_score"]
LABELS = {
    1: "drop",
    2: "maybe",
    3: "keep",
}
VISIBLE_COLUMNS = ["date", "dialog", "text", *SCORE_COLUMNS, "label"]


def build_labeling_dataset(
    *,
    messages_path: Path,
    heuristic_report_path: Path | None = None,
    nli_report_path: Path | None = None,
    embedding_report_path: Path | None = None,
    existing_dataset_path: Path | None = None,
) -> pd.DataFrame:
    """Build a parquet-friendly dataframe from exported messages and score reports."""
    dataset = _messages_dataframe(messages_path)
    dataset = _merge_score_report(
        dataset,
        heuristic_report_path,
        source_column="heuristic_score",
        target_column="heuristic_score",
    )
    dataset = _merge_score_report(
        dataset,
        nli_report_path,
        source_column="nli_score",
        target_column="nli_score",
    )
    dataset = _merge_score_report(
        dataset,
        embedding_report_path,
        source_column="nli_score",
        target_column="embedding_score",
    )

    for column in SCORE_COLUMNS:
        if column not in dataset:
            dataset[column] = pd.NA

    dataset["label"] = pd.Series([pd.NA] * len(dataset), dtype="Int64")
    dataset["label_name"] = pd.NA
    dataset["label_source"] = pd.NA
    dataset["labeled_at"] = pd.NA

    if existing_dataset_path and existing_dataset_path.exists():
        dataset = preserve_existing_labels(dataset, load_dataset(existing_dataset_path))

    return dataset.sort_values(["date", "message_id"]).reset_index(drop=True)


def load_dataset(path: Path) -> pd.DataFrame:
    """Load a labeling dataset from parquet."""
    dataset = pd.read_parquet(path)
    if "label" in dataset:
        dataset["label"] = dataset["label"].astype("Int64")
    return dataset


def save_dataset(dataset: pd.DataFrame, path: Path) -> None:
    """Persist the labeling dataset to parquet."""
    path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_parquet(path, index=False)


def set_label(
    dataset: pd.DataFrame,
    message_id: str,
    label: int,
    *,
    source: str = "manual",
) -> pd.DataFrame:
    """Return a copy of the dataset with one manual label updated."""
    if label not in LABELS:
        raise ValueError(f"Unsupported label: {label}")

    updated = dataset.copy()
    mask = updated["message_id"] == message_id
    if not mask.any():
        raise KeyError(f"Unknown message_id: {message_id}")

    updated.loc[mask, "label"] = label
    updated.loc[mask, "label_name"] = LABELS[label]
    updated.loc[mask, "label_source"] = source
    updated.loc[mask, "labeled_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    updated["label"] = updated["label"].astype("Int64")
    return updated


def preserve_existing_labels(dataset: pd.DataFrame, existing: pd.DataFrame) -> pd.DataFrame:
    """Carry manual labels and model predictions over when rebuilding scores."""
    preserved_columns = [
        "message_id",
        "label",
        "label_name",
        "label_source",
        "labeled_at",
        "catboost_label",
        "catboost_label_name",
        "catboost_confidence",
        "catboost_proba_drop",
        "catboost_proba_maybe",
        "catboost_proba_keep",
        "catboost_model_version",
        "catboost_predicted_at",
    ]
    existing_labels = existing[
        [column for column in preserved_columns if column in existing]
    ].copy()
    if existing_labels.empty or "message_id" not in existing_labels:
        return dataset

    value_columns = [column for column in existing_labels.columns if column != "message_id"]
    existing_labels = existing_labels.dropna(subset=value_columns, how="all")
    existing_labels = existing_labels.drop_duplicates("message_id", keep="last")
    if existing_labels.empty:
        return dataset

    merged = dataset.drop(columns=[c for c in value_columns if c in dataset]).merge(
        existing_labels,
        on="message_id",
        how="left",
    )
    if "label" not in merged:
        merged["label"] = pd.NA
    merged["label"] = merged["label"].astype("Int64")
    for column in ["label_name", "label_source", "labeled_at"]:
        if column not in merged:
            merged[column] = pd.NA
    if "catboost_label" in merged:
        merged["catboost_label"] = merged["catboost_label"].astype("Int64")
    return merged


def _messages_dataframe(path: Path) -> pd.DataFrame:
    rows = _read_json_rows(path)
    records: list[dict[str, Any]] = []
    for row in rows:
        date = str(row.get("date") or "")
        dialog = str(row.get("dialog") or "")
        text = str(row.get("text") or "")
        normalized_text = text.strip()
        records.append(
            {
                "message_id": stable_message_id(date=date, dialog=dialog, text=text),
                "date": date,
                "dialog": dialog,
                "text": text,
                "text_hash": _hash_text(text),
                "text_len": len(text),
                "word_count": len(normalized_text.split()) if normalized_text else 0,
                "has_link": "http://" in text or "https://" in text,
                "is_command": normalized_text.startswith("/"),
            }
        )

    if not records:
        return pd.DataFrame(columns=["message_id", "date", "dialog", "text"])
    return pd.DataFrame(records).drop_duplicates("message_id", keep="last")


def _merge_score_report(
    dataset: pd.DataFrame,
    path: Path | None,
    *,
    source_column: str,
    target_column: str,
) -> pd.DataFrame:
    if not path or not path.exists():
        dataset[target_column] = pd.NA
        return dataset

    rows = _read_json_rows(path)
    records: list[dict[str, Any]] = []
    for row in rows:
        if source_column not in row:
            continue
        date = str(row.get("date") or "")
        dialog = str(row.get("dialog") or "")
        text = str(row.get("text") or "")
        records.append(
            {
                "message_id": stable_message_id(date=date, dialog=dialog, text=text),
                target_column: row.get(source_column),
            }
        )

    if not records:
        dataset[target_column] = pd.NA
        return dataset

    scores = pd.DataFrame(records).drop_duplicates("message_id", keep="last")
    return dataset.drop(columns=[target_column], errors="ignore").merge(
        scores,
        on="message_id",
        how="left",
    )


def _read_json_rows(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Expected JSON array in {path}")
    return data


def stable_message_id(*, date: str, dialog: str, text: str) -> str:
    payload = f"{date}\0{dialog}\0{text}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:20]


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:20]
