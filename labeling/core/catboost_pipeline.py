"""CatBoost training and prediction helpers for the labeling dataset."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from labeling.core.dataset import LABELS, SCORE_COLUMNS

TRUE_STATE_TO_LABEL = {name: label for label, name in LABELS.items()}
TEXT_FEATURES = ["text"]
CAT_FEATURES = ["dialog"]
NUMERIC_FEATURES = [
    "text_len",
    "word_count",
    "has_link",
    "is_command",
    *SCORE_COLUMNS,
]
FEATURE_COLUMNS = [*TEXT_FEATURES, *CAT_FEATURES, *NUMERIC_FEATURES]
CATBOOST_COLUMNS = [
    "catboost_label",
    "catboost_label_name",
    "catboost_confidence",
    "catboost_proba_drop",
    "catboost_proba_maybe",
    "catboost_proba_keep",
    "catboost_model_version",
    "catboost_predicted_at",
]


@dataclass(frozen=True)
class TrainResult:
    labeled_rows: int
    predicted_rows: int
    model_path: Path
    model_version: str
    validation_accuracy: float | None


def train_and_predict(
    dataset: pd.DataFrame,
    *,
    model_path: Path,
    iterations: int = 350,
    random_seed: int = 42,
) -> tuple[pd.DataFrame, TrainResult]:
    """Train CatBoost on manual labels and write predictions for unlabeled rows."""
    from catboost import CatBoostClassifier, Pool

    prepared = _prepare_dataset(dataset)
    if "true_state" in prepared:
        prepared["label"] = prepared["true_state"].map(TRUE_STATE_TO_LABEL)
    train_df = prepared[prepared["label"].notna()].copy()
    if train_df.empty:
        raise ValueError("No manual labels found in dataset")
    if train_df["label"].nunique() < 2:
        raise ValueError("At least two label classes are required to train CatBoost")

    fit_df, eval_df = _split_train_eval(train_df, random_seed=random_seed)
    model = CatBoostClassifier(
        loss_function="MultiClass",
        iterations=iterations,
        depth=6,
        learning_rate=0.05,
        auto_class_weights="Balanced",
        random_seed=random_seed,
        verbose=False,
        allow_writing_files=False,
    )
    model.fit(
        _pool(fit_df, labels=fit_df["label"].astype(int)),
        eval_set=_pool(eval_df, labels=eval_df["label"].astype(int)) if not eval_df.empty else None,
        use_best_model=not eval_df.empty,
    )

    validation_accuracy = _validation_accuracy(model, eval_df)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(model_path))

    model_version = _model_version()
    predicted = _write_predictions(
        dataset.copy(),
        prepared,
        model=model,
        model_version=model_version,
    )
    result = TrainResult(
        labeled_rows=len(train_df),
        predicted_rows=int(predicted["catboost_label"].notna().sum()),
        model_path=model_path,
        model_version=model_version,
        validation_accuracy=validation_accuracy,
    )
    return predicted, result


def _prepare_dataset(dataset: pd.DataFrame) -> pd.DataFrame:
    prepared = dataset.copy()
    for column in ["text", "dialog"]:
        prepared[column] = prepared[column].fillna("").astype(str)
    for column in NUMERIC_FEATURES:
        if column not in prepared:
            prepared[column] = 0
    prepared["has_link"] = prepared["has_link"].fillna(False).astype(int)
    prepared["is_command"] = prepared["is_command"].fillna(False).astype(int)
    for column in ["text_len", "word_count", *SCORE_COLUMNS]:
        prepared[column] = pd.to_numeric(prepared[column], errors="coerce").fillna(-1.0)
    return prepared


def _pool(dataset: pd.DataFrame, *, labels: pd.Series | None = None):
    from catboost import Pool

    return Pool(
        dataset[FEATURE_COLUMNS],
        label=labels,
        text_features=TEXT_FEATURES,
        cat_features=CAT_FEATURES,
    )


def _split_train_eval(dataset: pd.DataFrame, *, random_seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    if len(dataset) < 30 or dataset["label"].value_counts().min() < 2:
        return dataset, dataset.iloc[0:0].copy()

    eval_parts = []
    fit_parts = []
    for _, group in dataset.groupby("label", dropna=False):
        shuffled = group.sample(frac=1.0, random_state=random_seed)
        eval_size = max(1, int(round(len(shuffled) * 0.2)))
        eval_parts.append(shuffled.iloc[:eval_size])
        fit_parts.append(shuffled.iloc[eval_size:])
    fit_df = pd.concat(fit_parts).sample(frac=1.0, random_state=random_seed)
    eval_df = pd.concat(eval_parts).sample(frac=1.0, random_state=random_seed)
    return fit_df, eval_df


def _validation_accuracy(model, eval_df: pd.DataFrame) -> float | None:
    if eval_df.empty:
        return None
    predictions = model.predict(_pool(eval_df)).reshape(-1)
    truth = eval_df["label"].astype(int).to_numpy()
    return float((predictions.astype(int) == truth).mean())


def _write_predictions(
    dataset: pd.DataFrame,
    prepared: pd.DataFrame,
    *,
    model,
    model_version: str,
) -> pd.DataFrame:
    for column in CATBOOST_COLUMNS:
        if column not in dataset:
            dataset[column] = pd.NA

    target = prepared[prepared["label"].isna()].copy()
    if target.empty:
        return dataset

    classes = [int(value) for value in model.classes_]
    probabilities = model.predict_proba(_pool(target))
    labels = model.predict(_pool(target)).reshape(-1).astype(int)
    predicted_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

    for row_index, label, proba in zip(target.index, labels, probabilities, strict=True):
        probability_by_class = {
            class_label: float(class_probability)
            for class_label, class_probability in zip(classes, proba, strict=True)
        }
        dataset.loc[row_index, "catboost_label"] = int(label)
        dataset.loc[row_index, "catboost_label_name"] = LABELS[int(label)]
        dataset.loc[row_index, "catboost_confidence"] = probability_by_class[int(label)]
        dataset.loc[row_index, "catboost_proba_drop"] = probability_by_class.get(1, 0.0)
        dataset.loc[row_index, "catboost_proba_maybe"] = probability_by_class.get(2, 0.0)
        dataset.loc[row_index, "catboost_proba_keep"] = probability_by_class.get(3, 0.0)
        dataset.loc[row_index, "catboost_model_version"] = model_version
        dataset.loc[row_index, "catboost_predicted_at"] = predicted_at

    dataset["catboost_label"] = dataset["catboost_label"].astype("Int64")
    return dataset


def _model_version() -> str:
    return datetime.now(timezone.utc).strftime("catboost-%Y%m%dT%H%M%SZ")
