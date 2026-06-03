"""CatBoost experiment helpers for comparing feature sets and hyperparameters."""

from __future__ import annotations

import itertools
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from labeling.core.catboost_pipeline import (
    CAT_FEATURES,
    FEATURE_COLUMNS,
    NUMERIC_FEATURES,
    TEXT_FEATURES,
    _prepare_dataset,
)
from labeling.core.dataset import LABELS, SCORE_COLUMNS

VERIFIED_LABEL_SOURCES = ("manual", "manual_review", "catboost_accept")
TARGET_COLUMN = "label"
TRUE_STATE_COLUMN = "true_state"
TRUE_STATE_TO_LABEL = {name: label for label, name in LABELS.items()}
SPLIT_COLUMN = "split"
LEAKAGE_COLUMNS = {
    "catboost_label",
    "catboost_label_name",
    "catboost_confidence",
    "catboost_proba_drop",
    "catboost_proba_maybe",
    "catboost_proba_keep",
    "catboost_model_version",
    "catboost_predicted_at",
}
FEATURE_SETS: dict[str, list[str]] = {
    "text_only": ["text"],
    "scores_only": ["text_len", "word_count", "has_link", "is_command", *SCORE_COLUMNS],
    "text_scores": ["text", "text_len", "word_count", "has_link", "is_command", *SCORE_COLUMNS],
    "all": FEATURE_COLUMNS,
}
DEFAULT_PARAM_GRID = {
    "iterations": [300, 600, 1000],
    "depth": [4, 6, 8],
    "learning_rate": [0.03, 0.05, 0.1],
    "l2_leaf_reg": [3, 10],
    "auto_class_weights": ["Balanced", None],
}
QUICK_PARAM_GRID = {
    "iterations": [300],
    "depth": [4, 6],
    "learning_rate": [0.05],
    "l2_leaf_reg": [3],
    "auto_class_weights": ["Balanced"],
}


@dataclass(frozen=True)
class ExperimentResult:
    experiment_id: str
    feature_set: str
    params: dict[str, Any]
    train_size: int
    valid_size: int
    test_size: int
    valid_macro_f1: float
    valid_accuracy: float
    valid_weighted_f1: float
    test_macro_f1: float
    test_accuracy: float
    test_weighted_f1: float
    manual_review_macro_f1: float | None
    model_path: str
    created_at: str
    per_class_metrics: dict[str, Any]
    confusion_matrix: dict[str, Any]


def build_gold_dataset(dataset: pd.DataFrame) -> pd.DataFrame:
    """Return verified labels with leakage columns removed."""
    if TARGET_COLUMN not in dataset and TRUE_STATE_COLUMN not in dataset:
        raise ValueError("Dataset must contain a label column")

    gold = dataset.copy()
    if TRUE_STATE_COLUMN in gold:
        gold = gold[gold[TRUE_STATE_COLUMN].notna()].copy()
        gold[TARGET_COLUMN] = gold[TRUE_STATE_COLUMN].map(TRUE_STATE_TO_LABEL)
        if gold[TARGET_COLUMN].isna().any():
            unknown = sorted(gold.loc[gold[TARGET_COLUMN].isna(), TRUE_STATE_COLUMN].unique())
            raise ValueError(f"Unknown true_state values: {unknown}")
    else:
        gold = gold[gold[TARGET_COLUMN].notna()].copy()

    if TRUE_STATE_COLUMN not in gold and "label_source" in gold:
        gold = gold[gold["label_source"].isin(VERIFIED_LABEL_SOURCES)]
    if gold.empty:
        raise ValueError("No verified labels found")

    gold = gold.drop(columns=[column for column in LEAKAGE_COLUMNS if column in gold])
    gold[TARGET_COLUMN] = gold[TARGET_COLUMN].astype(int)
    return gold.reset_index(drop=True)


def create_stratified_splits(
    gold: pd.DataFrame,
    *,
    seed: int = 42,
    train_ratio: float = 0.70,
    valid_ratio: float = 0.15,
) -> pd.DataFrame:
    """Create stable train/valid/test splits stratified by label."""
    if TARGET_COLUMN not in gold:
        raise ValueError("Gold dataset must contain a label column")

    split_parts: list[pd.DataFrame] = []
    for _, group in gold.groupby(TARGET_COLUMN, dropna=False):
        shuffled = group.sample(frac=1.0, random_state=seed).copy()
        total = len(shuffled)
        train_size = int(round(total * train_ratio))
        valid_size = int(round(total * valid_ratio))
        if total >= 3:
            train_size = min(max(train_size, 1), total - 2)
            valid_size = min(max(valid_size, 1), total - train_size - 1)
        else:
            train_size = max(total - 1, 1)
            valid_size = 0

        shuffled[SPLIT_COLUMN] = "test"
        shuffled.iloc[:train_size, shuffled.columns.get_loc(SPLIT_COLUMN)] = "train"
        if valid_size:
            shuffled.iloc[
                train_size : train_size + valid_size,
                shuffled.columns.get_loc(SPLIT_COLUMN),
            ] = "valid"
        split_parts.append(shuffled)

    return (
        pd.concat(split_parts)
        .sample(frac=1.0, random_state=seed)
        .reset_index(drop=True)
    )


def compare_models(
    split_dataset: pd.DataFrame,
    *,
    output_dir: Path,
    param_grid: dict[str, list[Any]] | None = None,
    feature_sets: dict[str, list[str]] | None = None,
    seed: int = 42,
    max_experiments: int | None = None,
    task_type: str = "CPU",
    devices: str | None = None,
    progress: bool = False,
) -> list[ExperimentResult]:
    """Train and evaluate CatBoost models across feature sets and hyperparameters."""
    from catboost import CatBoostClassifier

    output_dir.mkdir(parents=True, exist_ok=True)
    models_dir = output_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    grid = param_grid or DEFAULT_PARAM_GRID
    selected_feature_sets = feature_sets or FEATURE_SETS
    combinations = list(_param_combinations(grid))
    results: list[ExperimentResult] = []
    experiment_number = 0

    for feature_set_name, columns in selected_feature_sets.items():
        for params in combinations:
            experiment_number += 1
            if max_experiments is not None and experiment_number > max_experiments:
                return results

            experiment_id = f"{feature_set_name}-{experiment_number:04d}"
            if progress:
                print(
                    f"[{experiment_number}] training {experiment_id} "
                    f"feature_set={feature_set_name} params={params}",
                    flush=True,
                )
            prepared = _prepare_dataset(split_dataset)
            train = prepared[prepared[SPLIT_COLUMN] == "train"]
            valid = prepared[prepared[SPLIT_COLUMN] == "valid"]
            test = prepared[prepared[SPLIT_COLUMN] == "test"]

            model = CatBoostClassifier(
                loss_function="MultiClass",
                random_seed=seed,
                verbose=False,
                allow_writing_files=False,
                task_type=task_type,
                **({"devices": devices} if devices else {}),
                **params,
            )
            model.fit(
                _pool_for_columns(train, columns, labels=train[TARGET_COLUMN].astype(int)),
                eval_set=_pool_for_columns(valid, columns, labels=valid[TARGET_COLUMN].astype(int))
                if not valid.empty
                else None,
                use_best_model=not valid.empty,
            )

            model_path = models_dir / f"{experiment_id}.cbm"
            model.save_model(str(model_path))

            valid_metrics = evaluate_model(model, valid, columns)
            test_metrics = evaluate_model(model, test, columns)
            manual_review_metrics = _evaluate_subset(
                model,
                test,
                columns,
                label_source="manual_review",
            )
            results.append(
                ExperimentResult(
                    experiment_id=experiment_id,
                    feature_set=feature_set_name,
                    params=params,
                    train_size=len(train),
                    valid_size=len(valid),
                    test_size=len(test),
                    valid_macro_f1=valid_metrics["macro_f1"],
                    valid_accuracy=valid_metrics["accuracy"],
                    valid_weighted_f1=valid_metrics["weighted_f1"],
                    test_macro_f1=test_metrics["macro_f1"],
                    test_accuracy=test_metrics["accuracy"],
                    test_weighted_f1=test_metrics["weighted_f1"],
                    manual_review_macro_f1=None
                    if manual_review_metrics is None
                    else manual_review_metrics["macro_f1"],
                    model_path=str(model_path),
                    created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    per_class_metrics=test_metrics["per_class"],
                    confusion_matrix=test_metrics["confusion_matrix"],
                )
            )
    return results


def evaluate_model(model, dataset: pd.DataFrame, columns: list[str]) -> dict[str, Any]:
    """Evaluate a CatBoost model on a labeled split."""
    if dataset.empty:
        return _empty_metrics()

    truth = dataset[TARGET_COLUMN].astype(int).tolist()
    predicted = [int(value) for value in model.predict(_pool_for_columns(dataset, columns)).reshape(-1)]
    return classification_metrics(truth, predicted)


def classification_metrics(truth: list[int], predicted: list[int]) -> dict[str, Any]:
    """Compute accuracy, macro/weighted F1, per-class metrics, and confusion matrix."""
    if not truth:
        return _empty_metrics()

    labels = sorted(set(LABELS) | set(truth) | set(predicted))
    per_class: dict[str, dict[str, float | int | str]] = {}
    total_correct = sum(1 for actual, guess in zip(truth, predicted, strict=True) if actual == guess)
    weighted_f1_sum = 0.0
    macro_f1_values: list[float] = []
    confusion = {
        LABELS[label]: {LABELS[inner]: 0 for inner in labels}
        for label in labels
    }

    for actual, guess in zip(truth, predicted, strict=True):
        confusion[LABELS[actual]][LABELS[guess]] += 1

    for label in labels:
        tp = sum(
            1
            for actual, guess in zip(truth, predicted, strict=True)
            if actual == label and guess == label
        )
        fp = sum(
            1
            for actual, guess in zip(truth, predicted, strict=True)
            if actual != label and guess == label
        )
        fn = sum(
            1
            for actual, guess in zip(truth, predicted, strict=True)
            if actual == label and guess != label
        )
        support = sum(1 for actual in truth if actual == label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        macro_f1_values.append(f1)
        weighted_f1_sum += f1 * support
        per_class[LABELS[label]] = {
            "label": label,
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "f1": round(f1, 6),
            "support": support,
        }

    return {
        "accuracy": round(total_correct / len(truth), 6),
        "macro_f1": round(sum(macro_f1_values) / len(macro_f1_values), 6),
        "weighted_f1": round(weighted_f1_sum / len(truth), 6),
        "per_class": per_class,
        "confusion_matrix": confusion,
    }


def save_experiment_outputs(
    *,
    output_dir: Path,
    gold: pd.DataFrame,
    split_dataset: pd.DataFrame,
    results: list[ExperimentResult],
) -> None:
    """Persist gold dataset, splits, and experiment metrics."""
    output_dir.mkdir(parents=True, exist_ok=True)
    gold.to_parquet(output_dir / "gold_dataset.parquet", index=False)
    split_dataset.to_parquet(output_dir / "splits.parquet", index=False)
    rows = []
    for result in results:
        row = result.__dict__.copy()
        row["params"] = json.dumps(row["params"], ensure_ascii=False, sort_keys=True)
        row["per_class_metrics"] = json.dumps(row["per_class_metrics"], ensure_ascii=False)
        row["confusion_matrix"] = json.dumps(row["confusion_matrix"], ensure_ascii=False)
        rows.append(row)
    pd.DataFrame(rows).to_csv(output_dir / "results.csv", index=False)


def _pool_for_columns(dataset: pd.DataFrame, columns: list[str], *, labels: pd.Series | None = None):
    from catboost import Pool

    _assert_no_leakage(columns)
    text_features = [column for column in TEXT_FEATURES if column in columns]
    cat_features = [column for column in CAT_FEATURES if column in columns]
    return Pool(
        dataset[columns],
        label=labels,
        text_features=text_features,
        cat_features=cat_features,
    )


def _assert_no_leakage(columns: list[str]) -> None:
    leaked = sorted(set(columns) & LEAKAGE_COLUMNS)
    if leaked:
        raise ValueError(f"CatBoost experiment features contain leakage columns: {leaked}")


def _evaluate_subset(
    model,
    dataset: pd.DataFrame,
    columns: list[str],
    *,
    label_source: str,
) -> dict[str, Any] | None:
    if "label_source" not in dataset:
        return None
    subset = dataset[dataset["label_source"] == label_source]
    if subset.empty:
        return None
    return evaluate_model(model, subset, columns)


def _param_combinations(grid: dict[str, list[Any]]):
    keys = list(grid)
    for values in itertools.product(*(grid[key] for key in keys)):
        yield dict(zip(keys, values, strict=True))


def _empty_metrics() -> dict[str, Any]:
    return {
        "accuracy": 0.0,
        "macro_f1": 0.0,
        "weighted_f1": 0.0,
        "per_class": {},
        "confusion_matrix": {},
    }
