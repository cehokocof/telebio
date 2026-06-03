"""Streamlit page for inspecting merged context groups."""

from __future__ import annotations

import html
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from labeling.core.dataset import load_dataset

DEFAULT_DATASET_PATH = "data/context_labeling_merged.parquet"
DEFAULT_EVENTS_PATH = "data/dry_run_events.parquet"


def render(*, configure_page: bool = True) -> None:
    if configure_page:
        st.set_page_config(page_title="Merged groups", layout="wide")
    _inject_styles()

    st.title("Merged groups")
    st.caption(
        "Группы из подряд идущих собственных сообщений, склеенные в один контекстный "
        "элемент. Дай путь к events parquet чтобы видеть точные исходные сообщения."
    )

    dataset_path = Path(
        st.sidebar.text_input("Dataset path", value=DEFAULT_DATASET_PATH).strip()
        or DEFAULT_DATASET_PATH
    )
    events_path_raw = st.sidebar.text_input(
        "Events parquet (optional)", value=DEFAULT_EVENTS_PATH
    ).strip()
    events_path = Path(events_path_raw) if events_path_raw else None

    if not dataset_path.exists():
        st.info(f"Dataset not found: {dataset_path}")
        return

    dataset = _load_cached_dataset(str(dataset_path))
    if "group_size" not in dataset:
        st.error(
            "В этом parquet нет колонки `group_size`. "
            "Это значит датасет не был построен через scripts/build_labeling_dataset.py "
            "с включённой склейкой сообщений."
        )
        return

    events = None
    if events_path and events_path.exists():
        events = _load_cached_events(str(events_path))
    elif events_path:
        st.sidebar.warning(f"Events parquet не найден: {events_path}")

    merged = dataset[dataset["group_size"].fillna(0).astype(int) > 1].copy()
    if merged.empty:
        st.success("Склеек нет — все группы размера 1.")
        return

    dialogs = ["all", *sorted(merged["dialog"].dropna().unique().tolist())]
    dialog_filter = st.sidebar.selectbox("Dialog", options=dialogs)

    max_size = int(merged["group_size"].max())
    min_size = st.sidebar.slider("Min group size", min_value=2, max_value=max_size, value=2)

    sort_by = st.sidebar.radio(
        "Sort by",
        options=["group_size desc", "span_seconds desc", "date desc", "date asc"],
        horizontal=False,
    )

    filtered = merged
    if dialog_filter != "all":
        filtered = filtered[filtered["dialog"] == dialog_filter]
    filtered = filtered[filtered["group_size"] >= min_size]

    if sort_by == "group_size desc":
        filtered = filtered.sort_values(["group_size", "span_seconds"], ascending=[False, False])
    elif sort_by == "span_seconds desc":
        filtered = filtered.sort_values("span_seconds", ascending=False)
    elif sort_by == "date desc":
        filtered = filtered.sort_values("date", ascending=False)
    else:
        filtered = filtered.sort_values("date", ascending=True)
    filtered = filtered.reset_index(drop=True)

    cols = st.columns(4)
    cols[0].metric("groups (filtered)", len(filtered))
    cols[1].metric("total groups", len(merged))
    cols[2].metric("avg size", f"{filtered['group_size'].mean():.2f}" if len(filtered) else "-")
    cols[3].metric(
        "avg span",
        f"{filtered['span_seconds'].mean():.0f}s" if len(filtered) else "-",
    )

    if events is None:
        st.caption(
            "_Events parquet не подключён — оригинальные сообщения восстановлены "
            "только разбиением склееного текста по `\\n`. Это неточно, если в исходном "
            "сообщении были переносы строк._"
        )

    for idx, row in filtered.iterrows():
        _render_group(row, events, idx)


def _render_group(row: pd.Series, events: pd.DataFrame | None, idx: int) -> None:
    title = (
        f"[{int(row['group_size'])}] {row.get('dialog', '?')} · "
        f"{str(row.get('date', ''))[:19]} · "
        f"span={int(row.get('span_seconds') or 0)}s"
    )
    with st.expander(title, expanded=idx < 5):
        info_cols = st.columns(4)
        info_cols[0].caption(f"peer_id: `{row.get('peer_id')}`")
        info_cols[1].caption(f"first_msg_id: `{row.get('first_message_id')}`")
        info_cols[2].caption(f"last_msg_id: `{row.get('last_message_id')}`")
        info_cols[3].caption(f"label: `{_format_label(row.get('label'))}`")

        originals = _lookup_originals(row, events)
        if originals is not None and not originals.empty:
            st.markdown(f"**{len(originals)} исходных сообщений** (из events parquet):")
            for _, orig in originals.iterrows():
                date_str = str(orig.get("date", ""))[:19]
                text = html.escape(str(orig.get("text", "")))
                msg_id = int(orig.get("message_id", 0))
                st.markdown(
                    f"<div class='merged-orig'>"
                    f"<span class='merged-orig-meta'>{date_str} · id={msg_id}</span>"
                    f"<div class='merged-orig-text'>{text}</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
        else:
            parts = str(row.get("text") or "").split("\n")
            st.markdown(f"**{len(parts)} строк** (разбиение по `\\n` склеенного текста):")
            for part in parts:
                st.markdown(
                    f"<div class='merged-orig'>"
                    f"<div class='merged-orig-text'>{html.escape(part)}</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

        with st.expander("Склееный текст (сырой)", expanded=False):
            st.text(row.get("text") or "")


def _lookup_originals(row: pd.Series, events: pd.DataFrame | None) -> pd.DataFrame | None:
    if events is None:
        return None
    peer = row.get("peer_id")
    first = row.get("first_message_id")
    last = row.get("last_message_id")
    if pd.isna(peer) or pd.isna(first) or pd.isna(last):
        return None
    mask = (
        (events["peer_id"] == int(peer))
        & (events["message_id"].astype("Int64") >= int(first))
        & (events["message_id"].astype("Int64") <= int(last))
        & events["is_own"].fillna(False)
    )
    return events.loc[mask].sort_values("message_id")


def _format_label(value: object) -> str:
    if value is None or pd.isna(value):
        return "—"
    return str(int(value))


@st.cache_data(show_spinner=False)
def _load_cached_dataset(path: str) -> pd.DataFrame:
    return load_dataset(Path(path))


@st.cache_data(show_spinner=False)
def _load_cached_events(path: str) -> pd.DataFrame:
    return pd.read_parquet(path)


def _inject_styles() -> None:
    st.markdown(
        """
        <style>
        .merged-orig {
            padding: 0.6rem 0.8rem;
            margin: 0.4rem 0;
            border-left: 3px solid #2563eb;
            background: rgba(37, 99, 235, 0.06);
            border-radius: 4px;
        }
        .merged-orig-meta {
            font-size: 12px;
            color: #6b7280;
            font-family: monospace;
        }
        .merged-orig-text {
            margin-top: 0.3rem;
            white-space: pre-wrap;
            font-size: 15px;
            line-height: 1.45;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    render()
