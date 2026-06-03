"""CLI report for context relevance experiments on prepared JSON messages."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from telebio.context_relevance import (
    LocalEmbeddingScorer,
    LocalNliScorer,
    RelevanceOptions,
    select_context_messages,
)
from labeling.core.context_dataset import fixture_rows_to_messages
from telebio.services.telegram import ContextMessage


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect which prepared messages pass context relevance scoring."
    )
    parser.add_argument(
        "fixture",
        nargs="?",
        default="tests/fixtures/context_messages_live.json",
        help="Path to JSON fixture with date/dialog/text rows.",
    )
    parser.add_argument(
        "--scorer",
        choices=("heuristic", "nli", "embedding"),
        default="nli",
        help="Semantic scorer to use after heuristic filtering.",
    )
    parser.add_argument("--min-score", type=float, default=0.55)
    parser.add_argument("--top-k", type=int, default=15)
    parser.add_argument(
        "--exclude-dialog",
        action="append",
        default=["telebio"],
        help="Dialog name to exclude. Can be repeated.",
    )
    parser.add_argument(
        "--nli-model",
        default="cointegrated/rubert-base-cased-nli-threeway",
    )
    parser.add_argument(
        "--embedding-model",
        default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print full scored JSON instead of a compact table.",
    )
    parser.add_argument(
        "--log-output",
        help="Write the compact table report to this file instead of stdout.",
    )
    parser.add_argument(
        "--json-output",
        help="Write the full scored JSON report to this file.",
    )
    args = parser.parse_args()

    messages = _load_messages(Path(args.fixture))
    options = RelevanceOptions(
        top_k=args.top_k,
        min_score=args.min_score,
        excluded_dialogs=tuple(args.exclude_dialog),
        enable_nli=args.scorer != "heuristic",
        semantic_scorer=args.scorer,
        nli_model=args.nli_model,
        embedding_model=args.embedding_model,
    )
    scorer = _build_scorer(args.scorer, options)
    selection = select_context_messages(messages, options=options, scorer=scorer)

    json_report = json.dumps(
        [row.to_json_dict() for row in selection.scored],
        ensure_ascii=False,
        indent=2,
    )
    table_report = _build_table_report(
        selection,
        scorer_name=args.scorer,
        min_score=args.min_score,
        top_k=args.top_k,
    )

    if args.json_output:
        _write_text(Path(args.json_output), json_report)
    if args.log_output:
        _write_text(Path(args.log_output), table_report)

    if args.json:
        print(json_report)
        return

    if not args.log_output:
        print(table_report)
    else:
        print(f"Wrote report to {args.log_output}")
        if args.json_output:
            print(f"Wrote JSON report to {args.json_output}")


def _build_table_report(selection, *, scorer_name: str, min_score: float, top_k: int) -> str:
    lines = [
        f"scorer={scorer_name} min_score={min_score} top_k={top_k}",
        f"selected={len(selection.selected)} fingerprint={selection.fingerprint}",
        "",
        f"{'#':>2} {'decision':<4} {'heur':>5} {'sem':>5} {'final':>5} "
        f"{'dialog':<18} {'reasons':<34} text",
        "-" * 140,
    ]
    for index, row in enumerate(selection.scored, 1):
        semantic = "-" if row.nli_score is None else f"{row.nli_score:.2f}"
        text = row.message.text.replace("\n", " ")
        if len(text) > 90:
            text = f"{text[:87]}..."
        lines.append(
            f"{index:>2} {row.decision:<4} "
            f"{row.heuristic_score:>5.2f} {semantic:>5} {row.final_score:>5.2f} "
            f"{row.message.dialog[:18]:<18} "
            f"{','.join(row.reasons)[:34]:<34} {text}"
        )
    return "\n".join(lines)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"{text}\n", encoding="utf-8")


def _build_scorer(scorer_name: str, options: RelevanceOptions):
    match scorer_name:
        case "heuristic":
            return None
        case "nli":
            return LocalNliScorer(options.nli_model)
        case "embedding":
            return LocalEmbeddingScorer(options.embedding_model)
        case other:
            raise ValueError(f"Unknown scorer: {other}")


def _load_messages(path: Path) -> list[ContextMessage]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return fixture_rows_to_messages(data)


if __name__ == "__main__":
    main()
