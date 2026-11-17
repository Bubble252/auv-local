from __future__ import annotations

import hashlib
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from scripts.evaluate_pc_ppo import _parse_args

from auv_lowlevel.batch.pc_path import PC_PATH_DIM_ENDPOINT, PC_PATH_DIM_SAFE, PC_PATH_DIM_V1
from auv_lowlevel.batch.pc_policy import pc_schema_hash
from auv_lowlevel.config import load_config
from auv_lowlevel.cli.evaluate_pc_ppo.policy import (
    _InferenceEmpiricalNormalization,
    _load_observation_normalizer,
    _ObservationNormalizedPolicy,
)
from auv_lowlevel.cli.evaluate_pc_ppo.validation import (
    _apply_eval_start_contract,
    _final_path_blocks,
    _validate_eval_start_contract,
)

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "evaluate_pc_ppo.py"
HELP_SHA256 = "895873e097531e9bbef5fdc847b9887dae63a3be82f21d8c5b1309201afd9b87"


def test_help_text_is_byte_stable() -> None:
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join(
        part for part in (str(ROOT / "src"), environment.get("PYTHONPATH", "")) if part
    )
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"],
        cwd=ROOT,
        env=environment,
        check=False,
        capture_output=True,
    )

    assert result.returncode == 0
    assert result.stderr == b""
    assert hashlib.sha256(result.stdout).hexdigest() == HELP_SHA256


def test_default_namespace_is_stable() -> None:
    args = _parse_args(
        ["--training-config", "train.json", "--checkpoint", "ck.pt", "--output", "out"]
    )

    assert vars(args) == {
        "action_mode": "policy",
        "actuator_residual_channel_gain_far_distance_m": None,
        "actuator_residual_channel_gain_near_distance_m": None,
        "actuator_residual_channel_gains": None,
        "actuator_residual_near_channel_gains": None,
        "actuator_residual_scale_far": None,
        "actuator_residual_scale_near": None,
        "actuator_residual_scale_override": None,
        "actuator_residual_scale_switch_distance_m": None,
        "allow_actor_observation_override": False,
        "allow_alos_current_mode_override": False,
        "allow_diversity_environment_override": False,
        "allow_guidance_rate_limit_override": False,
        "allow_non_nominal_start": False,
        "allow_sensor_override": False,
        "allow_stage_override": False,
        "allow_start_progress_training_override": False,
        "checkpoint": Path("ck.pt"),
        "controller_config": None,
        "curriculum_current_speed_range_mps": None,
        "curriculum_shear_scale": None,
        "curriculum_stage": None,
        "device": "cpu",
        "disable_curriculum_direct_command": False,
        "dtype": "float64",
        "environment_config": None,
        "guidance_config": None,
        "history_ablation": "none",
        "injected_residual_profile": ("zero", None),
        "output": Path("out"),
        "path_ablation": "none",
        "ppo_residual_attenuation_enter_m": 3.0,
        "ppo_residual_attenuation_full_m": 1.25,
        "seed_count": 64,
        "seed_start": 0,
        "success_frame": "both",
        "terminal_gate_enter_m": 20.0,
        "terminal_gate_full_m": 5.0,
        "trace_estimator_state": False,
        "trace_output": None,
        "trace_reward_breakdown": False,
        "trace_sensor_ages": False,
        "training_config": Path("train.json"),
    }

    override = _parse_args(
        [
            "--training-config",
            "train.json",
            "--checkpoint",
            "ck.pt",
            "--output",
            "out",
            "--allow-non-nominal-start",
        ]
    )
    assert override.allow_non_nominal_start is True


def test_formal_eval_requires_nominal_start() -> None:
    config = load_config(
        ROOT / "configs" / "lowlevel" / "goal182" / "env_path_fixed_speed_c3.json"
    )

    with pytest.raises(ValueError, match="INVALID_EVAL_START_CONTRACT"):
        _validate_eval_start_contract(config)

    forced = _apply_eval_start_contract(config)
    assert forced.guidance.nominal_initial_state is True
    _validate_eval_start_contract(forced)

    randomized = _apply_eval_start_contract(config, require_nominal_start=False)
    assert randomized.guidance.nominal_initial_state is False
    _validate_eval_start_contract(randomized, require_nominal_start=False)


def test_canonical_pc_path_dimensions_are_accepted_by_evaluator() -> None:
    from auv_lowlevel.cli.evaluate_pc_ppo.runner import _validate_control_path_dim

    for path_dim in (PC_PATH_DIM_SAFE, PC_PATH_DIM_V1, PC_PATH_DIM_ENDPOINT):
        _validate_control_path_dim("path_fixed_speed", path_dim)

    try:
        _validate_control_path_dim("path_fixed_speed", 7)
    except ValueError as exc:
        assert "pc.path_dim in" in str(exc)
    else:
        raise AssertionError("unsupported path_dim must be rejected")


def test_five_dim_path_uses_safe_path_blocks() -> None:
    training = SimpleNamespace(pc=SimpleNamespace(path_dim=PC_PATH_DIM_SAFE, path_curriculum=[]))

    assert _final_path_blocks(training) == ("path_geom", "path_lookahead")


def test_pc_schema_hash_includes_privileged_critic_path() -> None:
    common = {
        "path_dim": PC_PATH_DIM_SAFE,
        "path_blocks": ("path_geom", "path_lookahead"),
        "dade_history_mode": "encoded",
        "actor_command_dim": 0,
        "critic_command_dim": 1,
        "history_path_dim": PC_PATH_DIM_SAFE,
    }

    assert pc_schema_hash(**common, critic_path_dim=30) != pc_schema_hash(
        **common, critic_path_dim=0
    )


def _normalizer_state(dim: int, *, mean: float, std: float) -> dict[str, torch.Tensor]:
    return {
        "_mean": torch.full((1, dim), mean),
        "_var": torch.full((1, dim), std * std),
        "_std": torch.full((1, dim), std),
        "count": torch.tensor(123, dtype=torch.long),
    }


def test_evaluator_applies_checkpoint_empirical_normalizer() -> None:
    checkpoint = {
        "obs_norm_state_dict": _normalizer_state(3, mean=1.0, std=2.0),
        "privileged_obs_norm_state_dict": _normalizer_state(4, mean=0.5, std=3.0),
    }
    training = SimpleNamespace(runner={"empirical_normalization": True})

    normalizer = _load_observation_normalizer(
        checkpoint,
        training,
        num_actor_obs=3,
        num_critic_obs=4,
        device=torch.device("cpu"),
    )
    assert isinstance(normalizer, _InferenceEmpiricalNormalization)
    observations = torch.tensor([[3.0, 5.0, 7.0]])
    expected = (observations - 1.0) / (2.0 + 1e-2)
    assert torch.allclose(normalizer(observations), expected)


def test_normalized_policy_wraps_all_actor_entry_points() -> None:
    class DummyPolicy:
        actor_layout = object()

        def act_inference(self, observations: torch.Tensor) -> torch.Tensor:
            return observations + 1.0

        def base_action(self, observations: torch.Tensor) -> torch.Tensor:
            return observations + 2.0

        def residual_action(self, observations: torch.Tensor) -> torch.Tensor:
            return observations + 3.0

    normalizer = _InferenceEmpiricalNormalization(2)
    normalizer._mean.fill_(1.0)
    normalizer._std.fill_(2.0)
    policy = _ObservationNormalizedPolicy(DummyPolicy(), normalizer)
    observations = torch.tensor([[3.0, 5.0]])
    normalized = (observations - 1.0) / (2.0 + 1e-2)

    assert policy.actor_layout is DummyPolicy.actor_layout
    assert torch.allclose(policy.act_inference(observations), normalized + 1.0)
    assert torch.allclose(policy.base_action(observations), normalized + 2.0)
    assert torch.allclose(policy.residual_action(observations), normalized + 3.0)


def test_evaluator_rejects_enabled_normalization_without_checkpoint_state() -> None:
    training = SimpleNamespace(runner={"empirical_normalization": True})
    with pytest.raises(ValueError, match="obs_norm_state_dict"):
        _load_observation_normalizer(
            {},
            training,
            num_actor_obs=3,
            num_critic_obs=4,
            device=torch.device("cpu"),
        )
