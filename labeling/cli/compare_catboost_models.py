"""CLI for comparing CatBoost feature sets and hyperparameters."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from labeling.core.catboost_experiments import (
    DEFAULT_PARAM_GRID,
    FEATURE_SETS,
    QUICK_PARAM_GRID,
    build_gold_dataset,
    compare_models,
    create_stratified_splits,
    save_experiment_outputs,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare CatBoost models on verified labeling data."
    )
    parser.add_argument(
        "--dataset",
        default="data/context_labeling.parquet",
        help="Labeling parquet with verified labels.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/catboost_experiments",
        help="Directory for gold dataset, splits, results, and models.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use a small grid for fast smoke comparisons.",
    )
    parser.add_argument(
        "--max-experiments",
        type=int,
        default=None,
        help="Optional cap across feature sets and params.",
    )
    parser.add_argument(
        "--feature-set",
        action="append",
        choices=tuple(FEATURE_SETS),
        help="Feature set to run. Can be repeated. Defaults to all.",
    )
    parser.add_argument(
        "--task-type",
        choices=("CPU", "GPU"),
        default="CPU",
        help="CatBoost task_type. Use GPU on CUDA servers.",
    )
    parser.add_argument(
        "--devices",
        default=None,
        help="CatBoost GPU devices, for example '0'. Used with --task-type GPU.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    dataset = pd.read_parquet(args.dataset)
    gold = build_gold_dataset(dataset)
    split_dataset = create_stratified_splits(gold, seed=args.seed)
    feature_sets = (
        {name: FEATURE_SETS[name] for name in args.feature_set}
        if args.feature_set
        else FEATURE_SETS
    )
    results = compare_models(
        split_dataset,
        output_dir=output_dir,
        param_grid=QUICK_PARAM_GRID if args.quick else DEFAULT_PARAM_GRID,
        feature_sets=feature_sets,
        seed=args.seed,
        max_experiments=args.max_experiments,
        task_type=args.task_type,
        devices=args.devices,
        progress=True,
    )
    save_experiment_outputs(
        output_dir=output_dir,
        gold=gold,
        split_dataset=split_dataset,
        results=results,
    )

    best = max(results, key=lambda row: row.valid_macro_f1) if results else None
    print(f"Gold rows: {len(gold)}")
    print(f"Experiments: {len(results)}")
    print(f"Wrote: {output_dir}")
    if best:
        print(
            "Best valid macro F1: "
            f"{best.valid_macro_f1:.3f} "
            f"({best.experiment_id}, {best.feature_set}, test={best.test_macro_f1:.3f})"
        )


if __name__ == "__main__":
    main()
