"""Shared lightweight exceptions for context-based providers."""

from __future__ import annotations


class ContextBatchNotReady(RuntimeError):
    """Raised when context was refreshed but not enough useful messages exist."""
