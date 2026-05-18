"""Streamlit page for reviewing CatBoost predictions."""

from __future__ import annotations

import html
import sys
from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from labeling.core.dataset import LABELS, SCORE_COLUMNS, load_dataset, save_dataset, set_label

DEFAULT_DATASET_PATH = "data/context_labeling.parquet"


def render(*, configure_page: bool = True) -> None:
    if configure_page:
        st.set_page_config(page_title="CatBoost review", layout="wide")
    _inject_styles()
    _inject_hotkeys()

    st.title("CatBoost review")
    dataset_path = Path(
        st.sidebar.text_input("Dataset path", value=DEFAULT_DATASET_PATH).strip()
        or DEFAULT_DATASET_PATH
    )
    if not dataset_path.exists():
        st.info("Dataset not found.")
        return

    dataset = _load_cached_dataset(str(dataset_path))
    dataset = _ensure_dtypes(dataset)
    if "catboost_label" not in dataset:
        st.info(
            "No CatBoost predictions found. Run: "
            "`PYTHONPATH=src:. uv run python -m labeling.train_catboost`"
        )
        return

    confidence_min = st.sidebar.slider("Min confidence", 0.0, 1.0, 0.0, 0.05)
    prediction_filter = st.sidebar.radio(
        "CatBoost class",
        options=["all", "drop", "maybe", "keep"],
        horizontal=False,
    )
    only_unlabeled = st.sidebar.checkbox("Only unlabeled", value=True)
    st.session_state.catboost_review_only_unlabeled = only_unlabeled

    filtered = _filter_dataset(
        dataset,
        confidence_min=confidence_min,
        prediction_filter=prediction_filter,
        only_unlabeled=only_unlabeled,
    )
    filtered = filtered.sort_values(
        ["catboost_confidence", "date"],
        ascending=[False, False],
    ).reset_index(drop=True)

    _render_progress(dataset)
    _render_review_summary(dataset)
    if filtered.empty:
        st.success("No rows for current filters.")
        return

    if "catboost_review_offset" not in st.session_state:
        st.session_state.catboost_review_offset = 0
    st.session_state.catboost_review_offset = min(
        st.session_state.catboost_review_offset,
        len(filtered) - 1,
    )

    row = filtered.iloc[st.session_state.catboost_review_offset]
    _render_navigation(len(filtered))
    _render_message(row)

    _render_label_buttons(row, dataset_path=dataset_path)


@st.cache_data(show_spinner=False)
def _load_cached_dataset(path: str) -> pd.DataFrame:
    return load_dataset(Path(path))


def _ensure_dtypes(dataset: pd.DataFrame) -> pd.DataFrame:
    dataset = dataset.copy()
    for column in ["label", "catboost_label"]:
        if column in dataset:
            dataset[column] = dataset[column].astype("Int64")
    return dataset


def _filter_dataset(
    dataset: pd.DataFrame,
    *,
    confidence_min: float,
    prediction_filter: str,
    only_unlabeled: bool,
) -> pd.DataFrame:
    filtered = dataset[dataset["catboost_label"].notna()].copy()
    filtered = filtered[filtered["catboost_confidence"].fillna(0.0) >= confidence_min]
    if only_unlabeled:
        filtered = filtered[filtered["label"].isna()]
    if prediction_filter != "all":
        label = next(value for value, name in LABELS.items() if name == prediction_filter)
        filtered = filtered[filtered["catboost_label"] == label]
    return filtered


def _render_progress(dataset: pd.DataFrame) -> None:
    labeled = int(dataset["label"].notna().sum())
    predicted = int(dataset.get("catboost_label", pd.Series(dtype="Int64")).notna().sum())
    pending_review = int(
        dataset["label"].isna().sum()
        if "catboost_label" not in dataset
        else ((dataset["label"].isna()) & (dataset["catboost_label"].notna())).sum()
    )
    cols = st.columns(4)
    cols[0].metric("manual labels", labeled)
    cols[1].metric("catboost predictions", predicted)
    cols[2].metric("pending review", pending_review)
    cols[3].metric("accepted", int((dataset["label_source"] == "catboost_accept").sum()))


def _render_review_summary(dataset: pd.DataFrame) -> None:
    reviewed = dataset[dataset["label_source"].isin(["catboost_accept", "manual_review"])].copy()
    if reviewed.empty:
        st.caption("No CatBoost review decisions yet.")
        return

    accepted = int((reviewed["label_source"] == "catboost_accept").sum())
    overridden = int((reviewed["label_source"] == "manual_review").sum())
    agreement_rate = accepted / len(reviewed) if len(reviewed) else 0.0
    cols = st.columns(3)
    cols[0].metric("reviewed", len(reviewed))
    cols[1].metric("catboost ok", accepted)
    cols[2].metric("manual overrides", overridden, f"{agreement_rate:.1%} ok")

    with st.expander("CatBoost review breakdown", expanded=False):
        display = reviewed.copy()
        display["catboost"] = display["catboost_label"].map(_format_label)
        display["final"] = display["label"].map(_format_label)
        matrix = pd.crosstab(
            display["catboost"].fillna("missing"),
            display["final"].fillna("missing"),
            margins=True,
        )
        st.dataframe(matrix, use_container_width=True)

        overrides = display[display["label_source"] == "manual_review"]
        if not overrides.empty:
            st.caption("Manual overrides by direction")
            direction = pd.crosstab(
                overrides["catboost"].fillna("missing"),
                overrides["final"].fillna("missing"),
                margins=True,
            )
            st.dataframe(direction, use_container_width=True)


def _render_navigation(total_filtered: int) -> None:
    prev_col, position_col, next_col = st.columns([1, 2, 1])
    prev_col.button(
        "Previous",
        key="catboost_review_previous",
        use_container_width=True,
        on_click=_move_offset,
        args=(-1, total_filtered),
    )
    position_col.caption(f"Row {st.session_state.catboost_review_offset + 1} / {total_filtered}")
    next_col.button(
        "Skip",
        key="catboost_review_skip",
        use_container_width=True,
        on_click=_move_offset,
        args=(1, total_filtered),
    )


def _render_message(row: pd.Series) -> None:
    st.caption(f"{row['date']} | {row['dialog']}")
    escaped_text = html.escape(str(row["text"]))
    st.markdown(
        f"<div class='message-box'>{escaped_text}</div>",
        unsafe_allow_html=True,
    )

    label = int(row["catboost_label"])
    st.markdown(
        f"<div class='catboost-label catboost-label-{label}'>"
        f"CatBoost: {label} {LABELS[label]} "
        f"<span>{_format_score(row.get('catboost_confidence'))}</span>"
        "</div>",
        unsafe_allow_html=True,
    )

    proba_cols = st.columns(3)
    proba_cols[0].metric("drop", _format_score(row.get("catboost_proba_drop")))
    proba_cols[1].metric("maybe", _format_score(row.get("catboost_proba_maybe")))
    proba_cols[2].metric("keep", _format_score(row.get("catboost_proba_keep")))

    score_cols = st.columns(3)
    for column, container in zip(SCORE_COLUMNS, score_cols, strict=True):
        container.metric(column, _format_score(row.get(column)))

    current_label = row.get("label")
    if pd.notna(current_label):
        current = int(current_label)
        st.markdown(
            f"<div class='current-label current-label-{current}'>current label: "
            f"{current} {LABELS[current]}</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            "<div class='current-label current-label-empty'>current label: empty</div>",
            unsafe_allow_html=True,
        )


def _render_label_buttons(row: pd.Series, *, dataset_path: Path) -> None:
    predicted = int(row["catboost_label"])
    message_id = str(row["message_id"])
    st.caption("Hotkeys: 1 = drop, 2 = maybe, 3 = keep, 4 = accept CatBoost, 5 = skip")
    cols = st.columns(5)
    cols[0].button(
        "1 Drop",
        key=f"catboost_review_drop_{message_id}",
        use_container_width=True,
        on_click=_apply_label,
        args=(str(dataset_path), message_id, 1, _source_for_label(1, predicted)),
    )
    cols[1].button(
        "2 Maybe",
        key=f"catboost_review_maybe_{message_id}",
        use_container_width=True,
        on_click=_apply_label,
        args=(str(dataset_path), message_id, 2, _source_for_label(2, predicted)),
    )
    cols[2].button(
        "3 Keep",
        key=f"catboost_review_keep_{message_id}",
        use_container_width=True,
        on_click=_apply_label,
        args=(str(dataset_path), message_id, 3, _source_for_label(3, predicted)),
    )
    cols[3].button(
        "4 CatBoost OK",
        key=f"catboost_review_accept_{message_id}",
        use_container_width=True,
        on_click=_apply_label,
        args=(str(dataset_path), message_id, predicted, "catboost_accept"),
    )
    cols[4].button(
        "5 Skip",
        key=f"catboost_review_skip_action_{message_id}",
        use_container_width=True,
        on_click=_move_offset,
        args=(1, 10**9),
    )


def _source_for_label(label: int, predicted: int) -> str:
    return "catboost_accept" if label == predicted else "manual_review"


def _move_offset(delta: int, total_filtered: int) -> None:
    current = int(st.session_state.get("catboost_review_offset", 0))
    st.session_state.catboost_review_offset = min(
        max(current + delta, 0),
        max(total_filtered - 1, 0),
    )


def _apply_label(dataset_path: str, message_id: str, label: int, source: str) -> None:
    path = Path(dataset_path)
    dataset = load_dataset(path)
    updated = set_label(dataset, message_id, label, source=source)
    save_dataset(updated, path)
    _load_cached_dataset.clear()
    current = int(st.session_state.get("catboost_review_offset", 0))
    if not bool(st.session_state.get("catboost_review_only_unlabeled", True)):
        st.session_state.catboost_review_offset = current + 1
    else:
        st.session_state.catboost_review_offset = current


def _format_score(value: object) -> str:
    if value is None or pd.isna(value):
        return "-"
    return f"{float(value):.3f}"


def _format_label(value: object) -> str:
    if value is None or pd.isna(value):
        return "missing"
    label = int(value)
    return f"{label} {LABELS[label]}"


def _inject_hotkeys() -> None:
    components.html(
        """
        <script>
        const bindHotkeys = () => {
          const doc = window.parent.document;
          if (doc.__telebioCatboostHotkeysBoundV3) return;
          doc.__telebioCatboostHotkeysBoundV3 = true;
          doc.addEventListener("keydown", (event) => {
            if (event.metaKey || event.ctrlKey || event.altKey) return;
            const active = doc.activeElement;
            const tag = active && active.tagName ? active.tagName.toLowerCase() : "";
            const editableField = (tag === "input" || tag === "textarea") && !active.disabled && !active.readOnly;
            if (editableField || active?.isContentEditable) return;
            const labels = {
              "1": "1 Drop",
              "2": "2 Maybe",
              "3": "3 Keep",
              "4": "4 CatBoost OK",
              "5": "5 Skip"
            };
            const label = labels[event.key];
            if (!label) return;
            const buttons = Array.from(doc.querySelectorAll("button"));
            const button = buttons.find((candidate) => candidate.innerText.trim().includes(label));
            if (!button) return;
            event.preventDefault();
            button.click();
          });
        };
        bindHotkeys();
        </script>
        """,
        height=0,
    )


def _inject_styles() -> None:
    st.markdown(
        """
        <style>
        .message-box {
            min-height: 480px;
            white-space: pre-wrap;
            overflow-wrap: anywhere;
            padding: 0.9rem 1rem;
            border: 1px solid rgba(49, 51, 63, 0.2);
            border-radius: 6px;
            background: rgba(250, 250, 250, 0.04);
            font-size: 24px;
            line-height: 1.65;
        }
        .catboost-label {
            margin: 1rem 0;
            font-size: 34px;
            font-weight: 800;
            line-height: 1.25;
        }
        .catboost-label span {
            font-size: 24px;
            font-weight: 700;
            margin-left: 0.75rem;
        }
        .catboost-label-1, .current-label-1 {
            color: #dc2626;
        }
        .catboost-label-2, .current-label-2 {
            color: #ca8a04;
        }
        .catboost-label-3, .current-label-3 {
            color: #16a34a;
        }
        .current-label {
            margin: 0.75rem 0 0.25rem;
            font-size: 24px;
            font-weight: 700;
            line-height: 1.35;
        }
        .current-label-empty {
            color: #6b7280;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    render()
