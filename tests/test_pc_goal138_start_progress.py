from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from auv_lowlevel.batch.pc_env import BatchedPCLowLevelEnv
from auv_lowlevel.config import load_config


ROOT = Path(__file__).resolve().parents[1]
BASE_ENV = ROOT / "configs/lowlevel/goal134/pc_goal134_nominal_strict_20261111.json"


def _config_with_start_progress(path: Path) -> Path:
    data = json.loads(BASE_ENV.read_text(encoding="utf-8"))
    data["reference"]["start_progress_m"] = 20.0
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def test_start_progress_parses_and_defaults_to_zero() -> None:
    base = load_config(BASE_ENV)
    assert base.reference.start_progress_m == 0.0

    data = json.loads(BASE_ENV.read_text(encoding="utf-8"))
    data["reference"]["start_progress_m"] = 20.0
    path = BASE_ENV.parent / "goal138_test_start_progress.json"
    try:
        path.write_text(json.dumps(data), encoding="utf-8")
        config = load_config(path)
        assert config.reference.start_progress_m == pytest.approx(20.0)
    finally:
        path.unlink(missing_ok=True)


def test_start_progress_must_be_less_than_length(tmp_path: Path) -> None:
    data = json.loads(BASE_ENV.read_text(encoding="utf-8"))
    data["reference"]["start_progress_m"] = float(data["reference"]["length_m"])
    path = tmp_path / "bad_start_progress.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="start_progress_m must be less"):
        load_config(path)


def test_default_start_progress_preserves_path_start(tmp_path: Path) -> None:
    config_path = _config_with_start_progress(tmp_path / "start20.json")
    config = load_config(config_path)
    no_jitter = replace(
        config,
        reference=replace(
            config.reference,
            start_jitter_xy_m=0.0,
            start_jitter_z_m=0.0,
            start_jitter_yaw_rad=0.0,
        ),
    )
    env = BatchedPCLowLevelEnv(
        no_jitter, num_envs=2, device="cpu", dtype=torch.float64, path_dim=12
    )
    env.reset([20, 21])
    projection = env.path.project(env.eta[:, :3])
    assert torch.allclose(
        projection.progress_m,
        torch.full((2,), 20.0, dtype=torch.float64),
        atol=1e-6,
        rtol=0.0,
    )
    assert torch.all(
        (env._projection_hint - projection.segment_index).abs() <= 2
    )


def test_zero_start_progress_uses_historical_start(tmp_path: Path) -> None:
    config = load_config(BASE_ENV)
    no_jitter = replace(
        config,
        reference=replace(
            config.reference,
            start_jitter_xy_m=0.0,
            start_jitter_z_m=0.0,
            start_jitter_yaw_rad=0.0,
        ),
    )
    env = BatchedPCLowLevelEnv(
        no_jitter, num_envs=2, device="cpu", dtype=torch.float64, path_dim=12
    )
    env.reset([20, 21])
    assert torch.allclose(
        env.eta[:, :3],
        env.path.start_ned_m_for(2),
        atol=1e-6,
        rtol=0.0,
    )


def test_ppo_evaluator_allows_only_zero_eval_start_progress(tmp_path: Path) -> None:
    from scripts.evaluate_pc_ppo import _validate_environment_override

    start20 = load_config(_config_with_start_progress(tmp_path / "start20.json"))
    eval_zero = load_config(BASE_ENV)
    _validate_environment_override(
        start20,
        eval_zero,
        allow_start_progress_training_override=True,
    )
    with pytest.raises(ValueError, match="start_progress_m"):
        _validate_environment_override(
            start20,
            eval_zero,
            allow_start_progress_training_override=False,
        )
    eval_nonzero = replace(
        eval_zero,
        reference=replace(eval_zero.reference, start_progress_m=5.0),
    )
    with pytest.raises(ValueError, match="start_progress_m=0"):
        _validate_environment_override(
            start20,
            eval_nonzero,
            allow_start_progress_training_override=True,
        )
