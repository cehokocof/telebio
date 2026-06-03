"""CLI for building the manual context labeling dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

from labeling.core.dataset import SCORE_COLUMNS, build_labeling_dataset, save_dataset


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a parquet dataset for manual context relevance labeling."
    )
    parser.add_argument(
        "--messages",
        default="tests/fixtures/context_messages_live_large.json",
        help="JSON export with date/dialog/text rows.",
    )
    parser.add_argument(
        "--heuristic-report",
        default="logs/context_heuristic.json",
        help="JSON report produced with --scorer heuristic.",
    )
    parser.add_argument(
        "--nli-report",
        default="logs/context_nli.json",
        help="JSON report produced with --scorer nli.",
    )
    parser.add_argument(
        "--embedding-report",
        default="logs/context_embedding.json",
        help="JSON report produced with --scorer embedding.",
    )
    parser.add_argument(
        "--output",
        default="data/context_labeling.parquet",
        help="Output parquet path. Existing manual labels are preserved.",
    )
    args = parser.parse_args()

    output_path = Path(args.output)
    dataset = build_labeling_dataset(
        messages_path=Path(args.messages),
        heuristic_report_path=Path(args.heuristic_report),
        nli_report_path=Path(args.nli_report),
        embedding_report_path=Path(args.embedding_report),
        existing_dataset_path=output_path,
    )
    save_dataset(dataset, output_path)

    scored = {
        column: int(dataset[column].notna().sum())
        for column in SCORE_COLUMNS
        if column in dataset
    }
    labeled = int(dataset["label"].notna().sum()) if "label" in dataset else 0
    print(f"Wrote {len(dataset)} rows to {output_path}")
    print(f"Preserved labels: {labeled}")
    print("Scores: " + ", ".join(f"{name}={count}" for name, count in scored.items()))


if __name__ == "__main__":
    main()
