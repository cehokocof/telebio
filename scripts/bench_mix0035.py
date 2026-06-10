"""Benchmark mix0035 predictions against the gold-labeled validation set.

Reads ``data/context_labeling_val_gold.parquet`` (must have both ``label``
and ``catboost_label`` populated) and writes per-class
precision/recall/F1, macro/weighted F1, accuracy, and a confusion matrix
to ``data/prod_models/mix0035/val_metrics.json``.

Run:
  uv run python scripts/bench_mix0035.py
"""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import pandas as pd

from labeling.core.catboost_experiments import classification_metrics
from labeling.core.dataset import LABELS

VAL_PATH = ROOT / "data" / "context_labeling_val_gold.parquet"
OUT_PATH = ROOT / "data" / "prod_models" / "mix0035" / "val_metrics.json"


def _print_table(metrics: dict) -> None:
    print()
    print(f"{'class':<8} {'precision':>10} {'recall':>10} {'f1':>10} {'support':>9}")
    print("-" * 50)
    for name, info in metrics["per_class"].items():
        print(
            f"{name:<8} {info['precision']:>10.3f} {info['recall']:>10.3f} "
            f"{info['f1']:>10.3f} {info['support']:>9}"
        )
    print("-" * 50)
    print(f"{'accuracy':<8} {metrics['accuracy']:>10.3f}")
    print(f"{'macro_f1':<8} {metrics['macro_f1']:>10.3f}")
    print(f"{'weighted':<8} {metrics['weighted_f1']:>10.3f}")
    print()
    print("Confusion matrix (rows = gold, cols = mix0035):")
    matrix = pd.DataFrame(metrics["confusion_matrix"])
    matrix = matrix.reindex(index=["drop", "maybe", "keep"], columns=["drop", "maybe", "keep"], fill_value=0)
    print(matrix.to_string())


def main() -> None:
    if not VAL_PATH.exists():
        sys.exit(f"Missing: {VAL_PATH}")

    df = pd.read_parquet(VAL_PATH)
    if df["label"].isna().any() or df["catboost_label"].isna().any():
        sys.exit(
            f"Found NaN in label or catboost_label. "
            f"label NaN={int(df['label'].isna().sum())}, "
            f"catboost NaN={int(df['catboost_label'].isna().sum())}"
        )

    truth = df["label"].astype(int).tolist()
    predicted = df["catboost_label"].astype(int).tolist()
    overall = classification_metrics(truth, predicted)

    by_category: dict[str, dict] = {}
    for cat, sub in df.groupby("chat_category"):
        sub_truth = sub["label"].astype(int).tolist()
        sub_pred = sub["catboost_label"].astype(int).tolist()
        by_category[str(cat)] = classification_metrics(sub_truth, sub_pred)

    label_distribution = {
        "gold": {LABELS[k]: int(v) for k, v in df["label"].value_counts().sort_index().to_dict().items()},
        "predicted": {
            LABELS[k]: int(v)
            for k, v in df["catboost_label"].value_counts().sort_index().to_dict().items()
        },
    }

    payload = {
        "model_version": "mix0035",
        "evaluated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "dataset": str(VAL_PATH.relative_to(ROOT)),
        "n_total": int(len(df)),
        "n_personal": int((df["chat_category"] == "personal").sum()),
        "n_group": int((df["chat_category"] == "group").sum()),
        "label_distribution": label_distribution,
        "overall": overall,
        "by_category": by_category,
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"=== mix0035 benchmark on {VAL_PATH.name} (n={len(df)}) ===")
    print(f"Categories: personal={payload['n_personal']}, group={payload['n_group']}")
    _print_table(overall)

    for cat, metrics in sorted(by_category.items()):
        print()
        print(f"=== by chat_category = {cat!r} (n={sum(v['support'] for v in metrics['per_class'].values())}) ===")
        _print_table(metrics)

    print()
    print(f"Saved → {OUT_PATH}")


if __name__ == "__main__":
    main()
