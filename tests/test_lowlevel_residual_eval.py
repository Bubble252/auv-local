from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from auv_lowlevel.batch.checkpoint import build_checkpoint_infos

SCRIPT = (
    Path(__file__).resolve().parents[1] / "scripts" / "evaluate_odarrl_control_residual.py"
)
SPEC = importlib.util.spec_from_file_location("evaluate_odarrl_control_residual", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _checkpoint(*, dim: int = 26, schema: str = "v2") -> dict:
    return {
        "model_state_dict": {},
        "infos": build_checkpoint_infos(
            observation_schema=schema,
            observation_dim=dim,
            vector_env_backend="batch_torch",
            physics_dtype="float64",
            training_config_sha256="a" * 64,
        ),
    }


def test_26d_checkpoint_requires_matching_metadata() -> None:
    metadata = MODULE._validate_policy_checkpoint(
        _checkpoint(),
        observation_dim=26,
        actor_warmstart=False,
    )
    assert metadata is not None
    assert metadata["observation_schema"] == "v2"

    with pytest.raises(ValueError, match="observation schema mismatch"):
        MODULE._validate_policy_checkpoint(
            _checkpoint(dim=27, schema="v1"),
            observation_dim=26,
            actor_warmstart=False,
        )


def test_26d_checkpoint_rejects_missing_metadata() -> None:
    with pytest.raises(ValueError, match="no 'auv_batch_checkpoint' metadata"):
        MODULE._validate_policy_checkpoint(
            {"model_state_dict": {}, "infos": {}},
            observation_dim=26,
            actor_warmstart=False,
        )


def test_legacy_27d_checkpoint_can_omit_metadata() -> None:
    assert (
        MODULE._validate_policy_checkpoint(
            {"model_state_dict": {}, "infos": {}},
            observation_dim=27,
            actor_warmstart=False,
        )
        is None
    )


def test_actor_warmstart_checks_observation_dimension() -> None:
    checkpoint = {
        "schema_version": "auv_odarrl_actor_warmstart.v1",
        "observation_dim": 26,
        "observation_schema": "v2",
        "actor_state_dict": {},
    }
    metadata = MODULE._validate_policy_checkpoint(
        checkpoint,
        observation_dim=26,
        actor_warmstart=True,
    )
    assert metadata is not None
    assert metadata["kind"] == "actor_warmstart"

    checkpoint["observation_dim"] = 27
    with pytest.raises(ValueError, match="dimension mismatch"):
        MODULE._validate_policy_checkpoint(
            checkpoint,
            observation_dim=26,
            actor_warmstart=True,
        )

    checkpoint["observation_dim"] = 26
    checkpoint["observation_schema"] = "v1"
    with pytest.raises(ValueError, match="schema mismatch"):
        MODULE._validate_policy_checkpoint(
            checkpoint,
            observation_dim=26,
            actor_warmstart=True,
        )


def test_residual_gains_parse_and_validate() -> None:
    assert MODULE._parse_residual_gains("1,0.5,0") == (1.0, 0.5, 0.0)

    with pytest.raises(ValueError, match="exactly three"):
        MODULE._parse_residual_gains("1,0")
    with pytest.raises(ValueError, match="numeric"):
        MODULE._parse_residual_gains("1,nan-ish,0")
    with pytest.raises(ValueError, match="finite and non-negative"):
        MODULE._parse_residual_gains("1,-0.1,0")
    with pytest.raises(ValueError, match="finite and non-negative"):
        MODULE._parse_residual_gains("1,inf,0")


def test_history_layout_metadata_is_validated() -> None:
    layout = MODULE._history_layout_from_metadata(
        {
            "observation_dim": 26,
            "history_short_window": 10,
            "history_long_window": 50,
            "policy_input_dim": 26 * 61,
        },
        observation_dim=26,
    )
    assert layout is not None
    assert layout.policy_input_dim == 26 * 61
    assert layout.short_stride == 1
    assert layout.long_stride == 1

    strided = MODULE._history_layout_from_metadata(
        {
            "observation_dim": 26,
            "history_short_window": 10,
            "history_long_window": 50,
            "history_short_stride": 5,
            "history_long_stride": 5,
            "policy_input_dim": 26 * 61,
        },
        observation_dim=26,
    )
    assert strided is not None
    assert strided.short_span_steps == 50
    assert strided.long_span_steps == 250

    assert MODULE._history_layout_from_metadata(None, observation_dim=26) is None
    with pytest.raises(ValueError, match="provided together"):
        MODULE._history_layout_from_metadata(
            {"history_short_window": 10}, observation_dim=26
        )
    with pytest.raises(ValueError, match="inconsistent"):
        MODULE._history_layout_from_metadata(
            {
                "history_short_window": 10,
                "history_long_window": 50,
                "policy_input_dim": 26,
            },
            observation_dim=26,
        )


def test_history_ablation_replaces_only_the_requested_horizon() -> None:
    current = np.arange(4, dtype=np.float32)
    short = np.tile(current, (3, 1)) + 10.0
    long = np.tile(current, (5, 1)) + 20.0

    actual_short, actual_long = MODULE._ablate_history_windows(
        current,
        short,
        long,
        mode="none",
    )
    np.testing.assert_array_equal(actual_short, short)
    np.testing.assert_array_equal(actual_long, long)

    current_short, current_long = MODULE._ablate_history_windows(
        current,
        short,
        long,
        mode="current-only",
    )
    assert np.all(current_short == current)
    assert np.all(current_long == current)

    short_only_short, short_only_long = MODULE._ablate_history_windows(
        current,
        short,
        long,
        mode="short-only",
    )
    np.testing.assert_array_equal(short_only_short, short)
    assert np.all(short_only_long == current)

    long_only_short, long_only_long = MODULE._ablate_history_windows(
        current,
        short,
        long,
        mode="long-only",
    )
    assert np.all(long_only_short == current)
    np.testing.assert_array_equal(long_only_long, long)

    zero_short, zero_long = MODULE._ablate_history_windows(
        current,
        short,
        long,
        mode="zero",
    )
    assert np.all(zero_short == 0.0)
    assert np.all(zero_long == 0.0)


def test_history_ablation_rejects_invalid_modes_and_shapes() -> None:
    with pytest.raises(ValueError, match="unsupported"):
        MODULE._ablate_history_windows(
            np.zeros(4),
            np.zeros((3, 4)),
            np.zeros((5, 4)),
            mode="shuffle",
        )


def test_current_ablation_zeros_only_the_current_frame() -> None:
    current = np.arange(1, 5, dtype=np.float32)
    short = np.tile(current, (3, 1)) + 10.0
    long = np.tile(current, (5, 1)) + 20.0

    unchanged = MODULE._ablate_current_frame(
        current, short, long, mode="none"
    )
    for actual, expected in zip(unchanged, (current, short, long), strict=True):
        np.testing.assert_array_equal(actual, expected)

    masked_current, masked_short, masked_long = MODULE._ablate_current_frame(
        current, short, long, mode="zero"
    )
    np.testing.assert_array_equal(masked_current, np.zeros_like(current))
    np.testing.assert_array_equal(masked_short[:-1], short[:-1])
    np.testing.assert_array_equal(masked_long[:-1], long[:-1])
    np.testing.assert_array_equal(masked_short[-1], np.zeros_like(current))
    np.testing.assert_array_equal(masked_long[-1], np.zeros_like(current))

    reset_current, reset_short, reset_long = MODULE._ablate_current_frame(
        current, short, long, mode="zero", reset=True
    )
    np.testing.assert_array_equal(reset_current, np.zeros_like(current))
    np.testing.assert_array_equal(reset_short, np.zeros_like(short))
    np.testing.assert_array_equal(reset_long, np.zeros_like(long))
    with pytest.raises(ValueError, match="unsupported"):
        MODULE._ablate_current_frame(current, short, long, mode="current-only")


def test_current_ablation_keeps_live_history_and_only_changes_policy_input() -> None:
    policy = object.__new__(MODULE._DeterministicActorPolicy)
    policy.history_layout = MODULE._history_layout_from_metadata(
        {
            "observation_dim": 2,
            "history_short_window": 2,
            "history_long_window": 3,
            "policy_input_dim": 12,
        },
        observation_dim=2,
    )
    policy.history_ablation = "none"
    policy.current_ablation = "zero"
    policy._history_short = None
    policy._history_long = None
    policy._fresh_history = True

    augmented = policy._augment_observation(np.array([3.0, 4.0], dtype=np.float32))
    assert augmented.shape == (12,)
    np.testing.assert_array_equal(augmented[:2], np.zeros(2))
    np.testing.assert_array_equal(augmented[2:], np.zeros(10, dtype=np.float32))
    np.testing.assert_array_equal(policy._history_short[-1], [3.0, 4.0])
    np.testing.assert_array_equal(policy._history_long[-1], [3.0, 4.0])

    augmented = policy._augment_observation(np.array([5.0, 6.0], dtype=np.float32))
    np.testing.assert_array_equal(augmented[:2], np.zeros(2))
    np.testing.assert_array_equal(augmented[2:4], [3.0, 4.0])
    np.testing.assert_array_equal(augmented[4:6], np.zeros(2))
    with pytest.raises(ValueError, match="one-dimensional"):
        MODULE._ablate_history_windows(
            np.zeros((1, 4)),
            np.zeros((3, 4)),
            np.zeros((5, 4)),
            mode="none",
        )
    with pytest.raises(ValueError, match="width"):
        MODULE._ablate_history_windows(
            np.zeros(4),
            np.zeros((3, 5)),
            np.zeros((5, 4)),
            mode="none",
        )


class _ConstantActor:
    def act_inference(self, observations: torch.Tensor) -> torch.Tensor:
        return torch.ones((observations.shape[0], 3), dtype=observations.dtype)


def _terminal_policy(*, radius: float | None):
    policy = object.__new__(MODULE._DeterministicActorPolicy)
    policy.model = _ConstantActor()
    policy.device = torch.device("cpu")
    policy.action_scale = 1.0
    policy.action_gains = np.ones(3, dtype=np.float64)
    policy.history_layout = None
    policy.terminal_residual_governor = (
        None
        if radius is None
        else MODULE.TerminalResidualGovernor(
            zero_radius_m=1.25,
            full_scale_radius_m=radius,
        )
    )
    policy._environment = None
    policy._augment_observation = lambda observation: np.asarray(observation, dtype=np.float32)
    return policy


def _bind_goal_distance(policy, distance_m: float) -> None:
    policy.bind_environment(
        SimpleNamespace(
            last_context=SimpleNamespace(
                state=SimpleNamespace(position_ned_m=np.array([distance_m, 0.0, 0.0]))
            ),
            path=SimpleNamespace(goal_ned_m=np.zeros(3, dtype=np.float64)),
        )
    )


def test_terminal_residual_governor_requires_environment_binding() -> None:
    policy = _terminal_policy(radius=4.0)
    with pytest.raises(RuntimeError, match="bind_environment"):
        policy(np.zeros(26, dtype=np.float32), 0)


def test_terminal_residual_governor_scales_actor_residual_by_goal_distance() -> None:
    policy = _terminal_policy(radius=4.0)
    observation = np.zeros(26, dtype=np.float32)

    _bind_goal_distance(policy, 4.0)
    np.testing.assert_allclose(policy(observation, 0), np.ones(3))

    _bind_goal_distance(policy, 2.625)
    np.testing.assert_allclose(policy(observation, 0), np.full(3, 0.5))

    _bind_goal_distance(policy, 1.25)
    np.testing.assert_allclose(policy(observation, 0), np.zeros(3))


def test_terminal_residual_governor_supports_early_handoff() -> None:
    policy = _terminal_policy(radius=4.0)
    policy.terminal_residual_governor = MODULE.TerminalResidualGovernor(
        zero_radius_m=2.0,
        full_scale_radius_m=4.0,
    )
    observation = np.zeros(26, dtype=np.float32)

    _bind_goal_distance(policy, 4.0)
    np.testing.assert_allclose(policy(observation, 0), np.ones(3))
    _bind_goal_distance(policy, 2.0)
    np.testing.assert_allclose(policy(observation, 0), np.zeros(3))
    _bind_goal_distance(policy, 1.25)
    np.testing.assert_allclose(policy(observation, 0), np.zeros(3))


def test_disabled_terminal_residual_governor_preserves_action() -> None:
    policy = _terminal_policy(radius=None)
    _bind_goal_distance(policy, 1.25)
    np.testing.assert_allclose(policy(np.zeros(26, dtype=np.float32), 0), np.ones(3))
