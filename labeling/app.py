"""Compatibility wrapper for `streamlit run labeling/app.py`."""

from __future__ import annotations

from labeling.ui.app import main


if __name__ == "__main__":
    main()
