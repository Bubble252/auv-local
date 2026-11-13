from __future__ import annotations

import hashlib
import os
import subprocess
import sys
from pathlib import Path

from scripts.evaluate_pc_ppo import _parse_args

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "evaluate_pc_ppo.py"
HELP_SHA256 = "944559f9b0e88678d3d0560e17cd3462ab97692a2c0bff355eeaba2ab2e11dd4"


def test_help_text_is_byte_stable() -> None:
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(ROOT / "src")
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
        "allow_diversity_environment_override": False,
        "allow_guidance_rate_limit_override": False,
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
