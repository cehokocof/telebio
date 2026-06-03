"""Fill missing NLI/embedding scores in an existing parquet labeling dataset."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd

from telebio.context_relevance import LocalEmbeddingScorer, LocalNliScorer, RelevanceOptions
from telebio.services.telegram import ContextMessage


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute missing semantic scores directly for all parquet rows."
    )
    parser.add_argument(
        "--dataset",
        default="data/context_labeling.parquet",
        help="Parquet dataset to update in place.",
    )
    parser.add_argument(
        "--nli-model",
        default=RelevanceOptions.nli_model,
    )
    parser.add_argument(
        "--embedding-model",
        default=RelevanceOptions.embedding_model,
    )
    args = parser.parse_args()

    path = Path(args.dataset)
    dataset = pd.read_parquet(path)

    nli_count = _fill_missing_nli(dataset, model=args.nli_model)
    embedding_count = _fill_missing_embedding(dataset, model=args.embedding_model)
    dataset.to_parquet(path, index=False)

    print(f"Updated {path}")
    print(f"Filled nli_score: {nli_count}")
    print(f"Filled embedding_score: {embedding_count}")
    for column in ["heuristic_score", "nli_score", "embedding_score"]:
        print(f"{column}: {int(dataset[column].notna().sum())}/{len(dataset)}")


def _fill_missing_nli(dataset: pd.DataFrame, *, model: str) -> int:
    missing = dataset[dataset["nli_score"].isna()].copy()
    if missing.empty:
        return 0

    scorer = LocalNliScorer(model)
    messages = _rows_to_messages(missing)
    scores = scorer.score(messages)
    for index, message in zip(missing.index, messages, strict=True):
        dataset.loc[index, "nli_score"] = scores.get(message)
    return len(missing)


def _fill_missing_embedding(dataset: pd.DataFrame, *, model: str) -> int:
    missing = dataset[dataset["embedding_score"].isna()].copy()
    if missing.empty:
        return 0

    scorer = LocalEmbeddingScorer(model)
    messages = _rows_to_messages(missing)
    scores = scorer.score(messages)
    for index, message in zip(missing.index, messages, strict=True):
        dataset.loc[index, "embedding_score"] = scores.get(message)
    return len(missing)


def _rows_to_messages(rows: pd.DataFrame) -> list[ContextMessage]:
    return [
        ContextMessage(
            date=datetime.fromisoformat(str(row.date)),
            dialog=str(row.dialog),
            text=str(row.text),
        )
        for row in rows.itertuples(index=False)
    ]


if __name__ == "__main__":
    main()
