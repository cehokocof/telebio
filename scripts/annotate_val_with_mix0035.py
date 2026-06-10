"""One-shot: run Mix0035 over data/context_labeling_val_raw.parquet.

Fills the ``catboost_label`` / ``catboost_label_name`` / ``catboost_model_version``
/ ``catboost_predicted_at`` columns in-place so the labeling UI's "CatBoost
review" page can be used to accept/reject the predictions in batch.

Run:
  uv run python scripts/annotate_val_with_mix0035.py
"""

from __future__ import annotations

import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

from telebio.config import load_settings
from telebio.telegram_context import Mix0035Classifier, QueuedContextMessage, _from_iso

VAL_PATH = ROOT / "data" / "context_labeling_val_raw.parquet"
_LABEL_TO_INT = {"drop": 1, "maybe": 2, "keep": 3}


def main() -> None:
    if not VAL_PATH.exists():
        sys.exit(f"Missing: {VAL_PATH}")

    frame = pd.read_parquet(VAL_PATH)
    print(f"Loaded {len(frame)} rows from {VAL_PATH.name}")

    queued = [
        QueuedContextMessage(
            id=row_index,
            message_key=str(row["message_id"]),
            date=_from_iso(str(row["date"])),
            dialog_title=str(row["dialog"]),
            text=str(row["text"]),
            label=None,
        )
        for row_index, (_, row) in enumerate(frame.iterrows())
    ]

    settings = load_settings()
    classifier = Mix0035Classifier(
        settings.telegram_context_model_path,
        stage1_model_name=settings.telegram_context_stage1_model,
        stage2_model_name=settings.telegram_context_stage2_model,
        feature_embedding_model_name=settings.telegram_context_feature_embedding_model,
        enable_nli_score=settings.telegram_context_enable_nli_score,
        nli_model_name=settings.telegram_context_nli_model,
    )
    print(f"Loading Mix0035 from {settings.telegram_context_model_path}…")
    classifier._load()
    for attr in ("_stage1_embedder", "_stage2_embedder", "_feature_embedder"):
        emb = getattr(classifier, attr, None)
        if emb is not None and hasattr(emb, "max_seq_length"):
            emb.max_seq_length = min(emb.max_seq_length or 256, 256)

    print(f"Classifying {len(queued)} messages…")
    predictions = classifier.classify(queued)
    print(f"Got {len(predictions)} predictions")

    now_iso = datetime.now(UTC).isoformat(timespec="seconds")
    for column in (
        "catboost_label",
        "catboost_label_name",
        "catboost_confidence",
        "catboost_proba_drop",
        "catboost_proba_maybe",
        "catboost_proba_keep",
        "catboost_model_version",
        "catboost_predicted_at",
    ):
        if column not in frame:
            frame[column] = pd.NA

    for queued_msg in queued:
        label = predictions.get(queued_msg.id)
        if label is None:
            continue
        index = frame.index[queued_msg.id]
        frame.loc[index, "catboost_label"] = _LABEL_TO_INT[label]
        frame.loc[index, "catboost_label_name"] = label
        frame.loc[index, "catboost_model_version"] = "mix0035"
        frame.loc[index, "catboost_predicted_at"] = now_iso

    frame["catboost_label"] = frame["catboost_label"].astype("Int64")
    frame.to_parquet(VAL_PATH, index=False)

    counts = frame["catboost_label_name"].value_counts(dropna=False).to_dict()
    total = sum(v for k, v in counts.items() if pd.notna(k))
    print()
    print(f"Saved → {VAL_PATH}")
    print(f"Predictions: {counts}")
    if total:
        print("Distribution (% of predicted):")
        for label_name in ("drop", "maybe", "keep"):
            n = counts.get(label_name, 0)
            print(f"  {label_name:<6} {n:>4} ({n / total * 100:5.1f}%)")


if __name__ == "__main__":
    main()
