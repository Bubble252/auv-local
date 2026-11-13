"""Row-wise state cloning and merging for batched episode resets."""

from __future__ import annotations

from dataclasses import fields, is_dataclass, replace
from typing import Any

import torch
from torch import Tensor

_MISSING_ROW_STATE = object()


def _clone_row_state(value: Any, count: int) -> Any:
    """Clone row-aligned tensors from a nested dataclass/dict state tree."""

    if isinstance(value, Tensor):
        if value.ndim >= 1 and value.shape[0] == count:
            return value.clone()
        return _MISSING_ROW_STATE
    if is_dataclass(value) and not isinstance(value, type):
        cloned: dict[str, Any] = {}
        for field in fields(value):
            state = _clone_row_state(getattr(value, field.name), count)
            if state is not _MISSING_ROW_STATE:
                cloned[field.name] = state
        if cloned:
            return ("dataclass", type(value), cloned)
        return _MISSING_ROW_STATE
    if isinstance(value, dict):
        cloned = {}
        for key, item in value.items():
            state = _clone_row_state(item, count)
            if state is not _MISSING_ROW_STATE:
                cloned[key] = state
        return cloned if cloned else _MISSING_ROW_STATE
    return _MISSING_ROW_STATE


def _merge_row_state(old: Any, new: Any, active_mask: Tensor) -> Any:
    """Keep ``old`` rows where ``active_mask`` is false and ``new`` elsewhere."""

    if isinstance(old, Tensor):
        if not isinstance(new, Tensor) or old.shape != new.shape:
            return new
        view = (active_mask.shape[0],) + (1,) * (old.ndim - 1)
        return torch.where(active_mask.view(view), new, old)
    if isinstance(old, tuple) and len(old) == 3 and old[0] == "dataclass":
        _, value_type, old_fields = old
        if not isinstance(new, value_type):
            return new
        merged = {
            name: _merge_row_state(value, getattr(new, name), active_mask)
            for name, value in old_fields.items()
        }
        return replace(new, **merged)
    if isinstance(old, dict):
        if not isinstance(new, dict):
            return new
        merged_dict = dict(new)
        for name, value in old.items():
            merged_dict[name] = _merge_row_state(value, new.get(name), active_mask)
        return merged_dict
    return new


__all__ = ["_clone_row_state", "_merge_row_state"]
