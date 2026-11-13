"""Output formatting helpers for PC PPO evaluation."""

from __future__ import annotations

from typing import Any


def _format_summary_value(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, str):
        return value
    return format(value, ".6g")
