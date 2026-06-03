"""Streamlit UI for fast manual context relevance labeling."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from labeling.core.dataset import LABELS, SCORE_COLUMNS, load_dataset, save_dataset, set_label
from labeling.ui.catboost_review import render as render_catboost_review
from labeling.ui.merged_groups import render as render_merged_groups

DEFAULT_DATASET_PATH = "data/context_labeling_merged.parquet"
DEFAULT_EVENTS_PATH = "data/dry_run_events.parquet"


def main() -> None:
    st.set_page_config(page_title="Context labeling", layout="wide")
    page = st.sidebar.radio(
        "Page",
        options=["Manual labeling", "CatBoost review", "Merged groups"],
        horizontal=False,
    )
    if page == "CatBoost review":
        _render_catboost_review()
        return
    if page == "Merged groups":
        render_merged_groups(configure_page=False)
        return

    _inject_styles()
    _inject_hotkeys()

    st.title("Context labeling")
    dataset_path = Path(
        st.sidebar.text_input("Dataset path", value=DEFAULT_DATASET_PATH).strip()
        or DEFAULT_DATASET_PATH
    )

    if not dataset_path.exists():
        st.info(
            "Dataset not found. Build it with: "
            "`uv run python -m labeling.build_dataset --output data/context_labeling.parquet`"
        )
        return

    dataset = _load_cached_dataset(str(dataset_path))
    dataset = _ensure_label_dtype(dataset)

    label_filter = st.sidebar.radio(
        "Rows",
        options=["unlabeled", "all", "drop", "maybe", "keep"],
        horizontal=False,
    )
    dialog_filter = st.sidebar.selectbox(
        "Dialog",
        options=["all", *sorted(dataset["dialog"].dropna().unique().tolist())],
    )
    sort_order = st.sidebar.radio("Sort", options=["newest", "oldest"], horizontal=True)

    filtered = _filter_dataset(dataset, label_filter=label_filter, dialog_filter=dialog_filter)
    filtered = filtered.sort_values("date", ascending=sort_order == "oldest").reset_index(drop=True)
    if filtered.empty:
        _render_progress(dataset)
        st.success("No rows for current filters.")
        return

    if "row_offset" not in st.session_state:
        st.session_state.row_offset = 0
    st.session_state.row_offset = min(st.session_state.row_offset, len(filtered) - 1)

    row = filtered.iloc[st.session_state.row_offset]
    _render_progress(dataset)
    _render_navigation(len(filtered))
    _render_message(row)

    label = _render_label_buttons()
    if label:
        updated = set_label(dataset, str(row["message_id"]), label)
        save_dataset(updated, dataset_path)
        _load_cached_dataset.clear()
        st.session_state.row_offset = min(st.session_state.row_offset, max(len(filtered) - 2, 0))
        st.rerun()


def _render_catboost_review() -> None:
    render_catboost_review(configure_page=False)


@st.cache_data(show_spinner=False)
def _load_cached_dataset(path: str) -> pd.DataFrame:
    return load_dataset(Path(path))


def _ensure_label_dtype(dataset: pd.DataFrame) -> pd.DataFrame:
    dataset = dataset.copy()
    if "label" in dataset:
        dataset["label"] = dataset["label"].astype("Int64")
    return dataset


def _filter_dataset(
    dataset: pd.DataFrame,
    *,
    label_filter: str,
    dialog_filter: str,
) -> pd.DataFrame:
    filtered = dataset
    if label_filter == "unlabeled":
        filtered = filtered[filtered["label"].isna()]
    elif label_filter != "all":
        label = next(value for value, name in LABELS.items() if name == label_filter)
        filtered = filtered[filtered["label"] == label]

    if dialog_filter != "all":
        filtered = filtered[filtered["dialog"] == dialog_filter]
    return filtered


def _render_progress(dataset: pd.DataFrame) -> None:
    labeled = int(dataset["label"].notna().sum())
    total = len(dataset)
    keep = int((dataset["label"] == 3).sum())
    maybe = int((dataset["label"] == 2).sum())
    drop = int((dataset["label"] == 1).sum())

    cols = st.columns(5)
    cols[0].metric("total", total)
    cols[1].metric("labeled", labeled)
    cols[2].metric("keep", keep)
    cols[3].metric("maybe", maybe)
    cols[4].metric("drop", drop)


def _render_navigation(total_filtered: int) -> None:
    prev_col, position_col, next_col = st.columns([1, 2, 1])
    if prev_col.button("Previous", key="manual_previous", use_container_width=True):
        st.session_state.row_offset = max(st.session_state.row_offset - 1, 0)
        st.rerun()
    position_col.caption(f"Row {st.session_state.row_offset + 1} / {total_filtered}")
    if next_col.button("Skip", key="manual_skip", use_container_width=True):
        st.session_state.row_offset = min(st.session_state.row_offset + 1, total_filtered - 1)
        st.rerun()


def _render_message(row: pd.Series) -> None:
    st.caption(f"{row['date']} | {row['dialog']}")
    st.text_area(
        "Message",
        value=str(row["text"]),
        height=420,
        disabled=True,
        key=f"manual_message_{row['message_id']}",
    )

    score_cols = st.columns(3)
    for column, container in zip(SCORE_COLUMNS, score_cols, strict=True):
        container.metric(column, _format_score(row.get(column)))

    current_label = row.get("label")
    if pd.notna(current_label):
        label_value = int(current_label)
        st.markdown(
            f"<div class='current-label current-label-{label_value}'>current label: "
            f"{label_value} {LABELS[label_value]}</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            "<div class='current-label current-label-empty'>current label: empty</div>",
            unsafe_allow_html=True,
        )


def _render_label_buttons() -> int | None:
    st.caption("Hotkeys: 1 = drop, 2 = maybe, 3 = keep")
    cols = st.columns(3)
    if cols[0].button("1 Drop", key="manual_drop", use_container_width=True):
        return 1
    if cols[1].button("2 Maybe", key="manual_maybe", use_container_width=True):
        return 2
    if cols[2].button("3 Keep", key="manual_keep", use_container_width=True):
        return 3
    return None


def _format_score(value: object) -> str:
    if value is None or pd.isna(value):
        return "-"
    return f"{float(value):.3f}"


def _inject_hotkeys() -> None:
    components.html(
        """
        <script>
        const bindHotkeys = () => {
          const doc = window.parent.document;
          if (doc.__telebioLabelHotkeysBoundV2) return;
          doc.__telebioLabelHotkeysBoundV2 = true;
          doc.addEventListener("keydown", (event) => {
            if (event.metaKey || event.ctrlKey || event.altKey) return;
            const active = doc.activeElement;
            const tag = active && active.tagName ? active.tagName.toLowerCase() : "";
            const editableField = (tag === "input" || tag === "textarea") && !active.disabled && !active.readOnly;
            if (editableField || active?.isContentEditable) return;
            const labels = {"1": "1 Drop", "2": "2 Maybe", "3": "3 Keep"};
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
        textarea[aria-label="Message"] {
            font-size: 20px !important;
            line-height: 1.55 !important;
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
        .current-label-1 {
            color: #dc2626;
        }
        .current-label-2 {
            color: #ca8a04;
        }
        .current-label-3 {
            color: #16a34a;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
