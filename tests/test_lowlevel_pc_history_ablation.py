from __future__ import annotations

import pytest
import torch

from auv_lowlevel.batch.pc_history_ablation import (
    PC_HISTORY_ABLATION_MODES,
    ablate_pc_history,
)
from auv_lowlevel.batch.pc_policy import PCPolicyLayout


def _layout() -> PCPolicyLayout:
    return PCPolicyLayout(state_dim=3, short_window=2, long_window=4)


def _observation(layout: PCPolicyLayout) -> torch.Tensor:
    batch = 2
    state = torch.arange(batch * 3, dtype=torch.float32).reshape(batch, 3)
    command = 10.0 + torch.arange(batch * 3, dtype=torch.float32).reshape(batch, 3)
    short = 20.0 + torch.arange(batch * 2 * layout.token_dim, dtype=torch.float32).reshape(
        batch, 2, layout.token_dim
    )
    long = 40.0 + torch.arange(batch * 4 * layout.token_dim, dtype=torch.float32).reshape(
        batch, 4, layout.token_dim
    )
    return layout.flatten(state=state, command=command, short=short, long=long)


def test_current_only_repeats_newest_token_and_preserves_current_state() -> None:
    layout = _layout()
    observation = _observation(layout)
    original = observation.clone()
    ablated = ablate_pc_history(observation, layout, "current-only")
    state, command, short, long = layout.split(ablated)
    base_state, base_command, _, base_long = layout.split(observation)

    expected_token = torch.cat([base_state, base_long[:, -1, layout.state_dim :]], dim=1)
    assert torch.equal(state, base_state)
    assert torch.equal(command, base_command)
    assert torch.equal(short, expected_token[:, None, :].expand_as(short))
    assert torch.equal(long, expected_token[:, None, :].expand_as(long))
    assert torch.equal(observation, original)


def test_current_zero_removes_only_current_state_and_newest_history_row() -> None:
    layout = _layout()
    observation = _observation(layout)
    _, _, base_short, base_long = layout.split(observation)
    ablated = ablate_pc_history(observation, layout, "current-zero")
    state, _, short, long = layout.split(ablated)

    assert torch.count_nonzero(state) == 0
    assert torch.count_nonzero(short[:, -1, :]) == 0
    assert torch.count_nonzero(long[:, -1, :]) == 0
    assert torch.equal(short[:, :-1, :], base_short[:, :-1, :])
    assert torch.equal(long[:, :-1, :], base_long[:, :-1, :])


def test_history_zero_removes_windows_but_keeps_current_state() -> None:
    layout = _layout()
    observation = _observation(layout)
    base_state, base_command, _, _ = layout.split(observation)
    state, command, short, long = layout.split(
        ablate_pc_history(observation, layout, "history-zero")
    )

    assert torch.equal(state, base_state)
    assert torch.equal(command, base_command)
    assert torch.count_nonzero(short) == 0
    assert torch.count_nonzero(long) == 0


def test_history_reverse_flips_both_windows() -> None:
    layout = _layout()
    observation = _observation(layout)
    _, _, base_short, base_long = layout.split(observation)
    state, command, short, long = layout.split(
        ablate_pc_history(observation, layout, "history-reverse")
    )
    base_state, base_command, _, _ = layout.split(observation)

    assert torch.equal(state, base_state)
    assert torch.equal(command, base_command)
    assert torch.equal(short, base_short.flip(1))
    assert torch.equal(long, base_long.flip(1))


def test_none_is_a_copy_and_unknown_mode_is_rejected() -> None:
    layout = _layout()
    observation = _observation(layout)
    ablated = ablate_pc_history(observation, layout, "none")
    assert torch.equal(ablated, observation)
    assert ablated.data_ptr() != observation.data_ptr()
    assert PC_HISTORY_ABLATION_MODES == (
        "none",
        "current-only",
        "current-zero",
        "history-zero",
        "history-reverse",
    )
    with pytest.raises(ValueError, match="unknown PC history ablation mode"):
        ablate_pc_history(observation, layout, "invalid")
