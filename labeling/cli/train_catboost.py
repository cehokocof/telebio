"""Train CatBoost on manual labels and add predictions to the parquet dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

from labeling.core.catboost_pipeline import train_and_predict
from labeling.core.dataset import load_dataset, save_dataset


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train CatBoost on manual labels and predict unlabeled messages."
    )
    parser.add_argument(
        "--dataset",
        default="data/context_labeling.parquet",
        help="Input/output parquet dataset path.",
    )
    parser.add_argument(
        "--model-output",
        default="data/context_catboost_model.cbm",
        help="Where to save the CatBoost model.",
    )
    parser.add_argument("--iterations", type=int, default=350)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    dataset = load_dataset(dataset_path)
    updated, result = train_and_predict(
        dataset,
        model_path=Path(args.model_output),
        iterations=args.iterations,
        random_seed=args.seed,
    )
    save_dataset(updated, dataset_path)

    accuracy = "-" if result.validation_accuracy is None else f"{result.validation_accuracy:.3f}"
    print(f"Trained on {result.labeled_rows} manual labels")
    print(f"Predicted {result.predicted_rows} unlabeled rows")
    print(f"Validation accuracy: {accuracy}")
    print(f"Model version: {result.model_version}")
    print(f"Saved model: {result.model_path}")
    print(f"Updated dataset: {dataset_path}")


if __name__ == "__main__":
    main()
