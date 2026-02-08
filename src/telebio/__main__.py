"""Allow running as `python -m telebio`."""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is on sys.path so `main.py` imports work regardless
# of how the package is invoked.
_root = str(Path(__file__).resolve().parent.parent.parent)
if _root not in sys.path:
    sys.path.insert(0, _root)

from main import main  # noqa: E402

main()
