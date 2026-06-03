"""Benchmark context scoring runtime on a 100-message sample."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd

from labeling.core.context_dataset import fixture_rows_to_messages
from telebio.context_relevance import (
    LocalEmbeddingScorer,
    LocalNliScorer,
    RelevanceOptions,
    select_context_messages,
)
from telebio.services.telegram import ContextMessage


@dataclass(frozen=True)
class BenchmarkRow:
    name: str
    messages: int
    seconds: float
    messages_per_second: float


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure scoring runtime on a fixed-size message sample."
    )
    parser.add_argument(
        "--dataset",
        default="data/context_labeling.parquet",
        help="Parquet dataset path. Used when it exists.",
    )
    parser.add_argument(
        "--fixture",
        default="tests/fixtures/context_messages_live_large.json",
        help="JSON fixture fallback with date/dialog/text rows.",
    )
    parser.add_argument("--sample-size", type=int, default=100)
    parser.add_argument(
        "--sample",
        choices=("head", "tail", "random"),
        default="random",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--json-output",
        default="logs/labeling_benchmark_100.json",
        help="Where to save benchmark JSON. Pass empty string to skip.",
    )
    args = parser.parse_args()

    messages = _load_messages(
        dataset_path=Path(args.dataset),
        fixture_path=Path(args.fixture),
        sample_size=args.sample_size,
        sample=args.sample,
        seed=args.seed,
    )
    options = RelevanceOptions(top_k=min(40, len(messages)))

    results = [
        _measure("heuristic", messages, options=options, scorer=None),
        _measure("nli", messages, options=options, scorer=LocalNliScorer(options.nli_model)),
        _measure(
            "embedding",
            messages,
            options=options,
            scorer=LocalEmbeddingScorer(options.embedding_model),
        ),
    ]

    _print_report(results)
    if args.json_output:
        output = Path(args.json_output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(
                {
                    "created_at": datetime.now().isoformat(timespec="seconds"),
                    "sample_size": len(messages),
                    "results": [asdict(row) for row in results],
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"Wrote {output}")


def _measure(
    name: str,
    messages: list[ContextMessage],
    *,
    options: RelevanceOptions,
    scorer,
) -> BenchmarkRow:
    started = time.perf_counter()
    select_context_messages(
        messages,
        options=RelevanceOptions(
            top_k=options.top_k,
            min_score=options.min_score,
            excluded_dialogs=options.excluded_dialogs,
            enable_nli=scorer is not None,
            semantic_scorer=name,
            nli_model=options.nli_model,
            embedding_model=options.embedding_model,
        ),
        scorer=scorer,
    )
    seconds = time.perf_counter() - started
    messages_per_second = len(messages) / seconds if seconds > 0 else 0.0
    return BenchmarkRow(
        name=name,
        messages=len(messages),
        seconds=round(seconds, 4),
        messages_per_second=round(messages_per_second, 2),
    )


def _load_messages(
    *,
    dataset_path: Path,
    fixture_path: Path,
    sample_size: int,
    sample: str,
    seed: int,
) -> list[ContextMessage]:
    if dataset_path.exists():
        dataset = pd.read_parquet(dataset_path)
        rows = dataset[["date", "dialog", "text"]].drop_duplicates().reset_index(drop=True)
        rows = _sample_rows(rows, sample_size=sample_size, sample=sample, seed=seed)
        return [
            ContextMessage(
                date=datetime.fromisoformat(str(row.date)),
                dialog=str(row.dialog),
                text=str(row.text),
            )
            for row in rows.itertuples(index=False)
        ]

    rows = json.loads(fixture_path.read_text(encoding="utf-8"))
    messages = fixture_rows_to_messages(rows)
    frame = pd.DataFrame(
        [
            {
                "date": message.date.isoformat(timespec="seconds"),
                "dialog": message.dialog,
                "text": message.text,
            }
            for message in messages
        ]
    )
    sampled = _sample_rows(frame, sample_size=sample_size, sample=sample, seed=seed)
    return [
        ContextMessage(
            date=datetime.fromisoformat(str(row.date)),
            dialog=str(row.dialog),
            text=str(row.text),
        )
        for row in sampled.itertuples(index=False)
    ]


def _sample_rows(
    rows: pd.DataFrame,
    *,
    sample_size: int,
    sample: str,
    seed: int,
) -> pd.DataFrame:
    if len(rows) <= sample_size:
        return rows
    match sample:
        case "head":
            return rows.head(sample_size)
        case "tail":
            return rows.tail(sample_size)
        case "random":
            return rows.sample(n=sample_size, random_state=seed).sort_values("date")
        case other:
            raise ValueError(f"Unknown sample strategy: {other}")


def _print_report(results: list[BenchmarkRow]) -> None:
    print(f"{'stage':<12} {'messages':>8} {'seconds':>10} {'msg/s':>10}")
    print("-" * 44)
    for row in results:
        print(
            f"{row.name:<12} {row.messages:>8} "
            f"{row.seconds:>10.4f} {row.messages_per_second:>10.2f}"
        )


if __name__ == "__main__":
    main()
