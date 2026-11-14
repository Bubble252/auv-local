"""Apply evaluation-only history and path ablations to flattened PC observations.

Non-responsibility: does not train policies or modify checkpoint contents.
"""

from __future__ import annotations

import torch
from torch import Tensor

from ..pc_path import PC_PATH_BLOCK_FEATURE_INDICES
from ..pc_policy import PCPolicyLayout

PC_HISTORY_ABLATION_MODES = (
    "none",
    "current-only",
    "current-zero",
    "history-zero",
    "history-reverse",
)
PC_PATH_ABLATION_MODES = (
    "none",
    "path-zero",
    "path-geom-zero",
    "path-context-zero",
)


def ablate_pc_history(
    observation: Tensor,
    layout: PCPolicyLayout,
    mode: str,
) -> Tensor:
    """Return a modified copy of a flattened PC observation.

    The interventions are applied after normalization, matching the tensor
    consumed by the policy. They never mutate the original observation.

    ``current-only`` repeats the current token in both history windows and
    therefore removes all older samples. ``current-zero`` keeps older history
    but removes both the top-level current state and the newest history row.
    ``history-zero`` removes the history windows while keeping the current
    state. ``history-reverse`` reverses each window to test temporal order.
    """

    if mode not in PC_HISTORY_ABLATION_MODES:
        raise ValueError(
            f"unknown PC history ablation mode {mode!r}; "
            f"expected one of {PC_HISTORY_ABLATION_MODES}"
        )
    if mode == "none":
        return observation.clone()

    state, path, command, short, long = layout.split_with_path(observation)
    state = state.clone()
    short = short.clone()
    long = long.clone()

    if mode == "current-only":
        current_side = long[:, -1, layout.state_dim :]
        current_token = torch.cat([state, current_side], dim=1)
        short = current_token[:, None, :].expand_as(short).clone()
        long = current_token[:, None, :].expand_as(long).clone()
    elif mode == "current-zero":
        state.zero_()
        short[:, -1, :].zero_()
        long[:, -1, :].zero_()
    elif mode == "history-zero":
        short.zero_()
        long.zero_()
    elif mode == "history-reverse":
        short = short.flip(1)
        long = long.flip(1)

    return layout.flatten(
        state=state,
        path=path,
        command=command,
        short=short,
        long=long,
    )


def ablate_pc_path(
    observation: Tensor,
    layout: PCPolicyLayout,
    mode: str,
) -> Tensor:
    """Return a path-ablated copy of a flattened PC observation.

    The intervention is applied before the actor encoder, exactly where the
    environment's active path-block mask is applied during training.  History
    tokens never contain path columns, so this function changes only the
    current path block and leaves the DADE input untouched.
    """

    if mode not in PC_PATH_ABLATION_MODES:
        raise ValueError(
            f"unknown PC path ablation mode {mode!r}; expected one of {PC_PATH_ABLATION_MODES}"
        )
    if mode == "none":
        return observation.clone()
    if layout.path_dim == 0:
        raise ValueError("path ablation requires a policy layout with path_dim > 0")

    state, path, current, command, short, long = layout.split_with_optional_current(observation)
    assert path is not None
    path = path.clone()
    if mode == "path-zero":
        path.zero_()
    elif mode == "path-geom-zero":
        path[:, PC_PATH_BLOCK_FEATURE_INDICES["path_geom"]] = 0.0
    elif mode == "path-context-zero":
        path[:, PC_PATH_BLOCK_FEATURE_INDICES["path_context"]] = 0.0

    return layout.flatten(
        state=state,
        path=path,
        current=current,
        command=command,
        short=short,
        long=long,
    )


__all__ = [
    "PC_HISTORY_ABLATION_MODES",
    "PC_PATH_ABLATION_MODES",
    "ablate_pc_history",
    "ablate_pc_path",
]
