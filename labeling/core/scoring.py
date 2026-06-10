"""Score computation for labeling parquets: heuristic / embedding / NLI."""

from __future__ import annotations

import re

import pandas as pd


def compute_heuristic(frame: pd.DataFrame) -> None:
    """Fill ``heuristic_score`` in-place. Cheap; no model load."""
    from telebio.telegram_context import _heuristic_score

    for index, row in frame.iterrows():
        text = str(row["text"])
        stripped = text.strip()
        words = re.findall(r"\w+", text, flags=re.UNICODE)
        frame.loc[index, "heuristic_score"] = float(_heuristic_score(stripped, words))


def compute_embedding_and_nli(frame: pd.DataFrame, *, enable_nli: bool) -> None:
    """Load Mix0035 once, fill ``embedding_score`` (+``nli_score`` if asked).

    Slow on first run: triggers HuggingFace model downloads. ``embedding_score``
    is always populated; ``nli_score`` only when ``enable_nli=True``.
    """
    from telebio.config import load_settings
    from telebio.telegram_context import Mix0035Classifier

    if frame.empty:
        return

    settings = load_settings()
    classifier = Mix0035Classifier(
        settings.telegram_context_model_path,
        stage1_model_name=settings.telegram_context_stage1_model,
        stage2_model_name=settings.telegram_context_stage2_model,
        feature_embedding_model_name=settings.telegram_context_feature_embedding_model,
        enable_nli_score=enable_nli,
        nli_model_name=settings.telegram_context_nli_model,
    )
    classifier._load()
    for attr in ("_stage1_embedder", "_stage2_embedder", "_feature_embedder"):
        embedder = getattr(classifier, attr, None)
        if embedder is not None and hasattr(embedder, "max_seq_length"):
            embedder.max_seq_length = min(embedder.max_seq_length or 256, 256)

    texts = frame["text"].astype(str).tolist()
    embedding_scores = classifier._embedding_scores(texts)
    for index, score in zip(frame.index, embedding_scores, strict=True):
        frame.loc[index, "embedding_score"] = float(score)

    if enable_nli:
        nli_scores = classifier._nli_scores(texts)
        for index, score in zip(frame.index, nli_scores, strict=True):
            frame.loc[index, "nli_score"] = float(score)
