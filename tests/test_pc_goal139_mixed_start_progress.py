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


def _write_mixed_config(path: Path, *, progress_m: float, full_fraction: float) -> Path:
    data = json.loads(BASE_ENV.read_text(encoding="utf-8"))
    data["reference"]["start_progress_m"] = progress_m
    data["reference"]["start_progress_full_fraction"] = full_fraction
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def test_mixed_start_requires_progress_and_valid_fraction(tmp_path: Path) -> None:
    def _bad(progress_m: float, fraction: float) -> Path:
        data = json.loads(BASE_ENV.read_text(encoding="utf-8"))
        data["reference"]["start_progress_m"] = progress_m
        data["reference"]["start_progress_full_fraction"] = fraction
        path = tmp_path / f"bad_{progress_m}_{fraction}.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        return path

    with pytest.raises(ValueError, match=r"must lie in \[0, 1\]"):
        load_config(_bad(20.0, 1.1))
    with pytest.raises(ValueError, match="requires reference.start_progress_m > 0"):
        load_config(_bad(0.0, 0.5))


def test_mixed_start_is_deterministic_and_roughly_balanced(tmp_path: Path) -> None:
    config = load_config(
        _write_mixed_config(
            tmp_path / "m50.json",
            progress_m=20.0,
            full_fraction=0.5,
        )
    )
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
        no_jitter, num_envs=256, device="cpu", dtype=torch.float64, path_dim=12
    )
    seeds = list(range(1000, 1256))
    env.reset(seeds)
    first_progress = env.episode_start_progress_m.clone()
    full_mask = env._full_path_mask_for(env.episode_seed, env.num_envs)
    mask_fraction = float(full_mask.to(torch.float64).mean().item())
    assert 0.40 <= mask_fraction <= 0.60
    assert torch.allclose(
        first_progress[full_mask],
        torch.zeros_like(first_progress[full_mask]),
        atol=0.0,
        rtol=0.0,
    )
    assert torch.allclose(
        first_progress[~full_mask],
        torch.full_like(first_progress[~full_mask], 20.0),
        atol=1e-6,
        rtol=0.0,
    )
    env.reset(seeds)
    assert torch.allclose(first_progress, env.episode_start_progress_m)
    assert torch.equal(full_mask, env._full_path_mask_for(env.episode_seed, env.num_envs))


def test_mixed_start_positions_match_selected_horizon(tmp_path: Path) -> None:
    config = load_config(
        _write_mixed_config(
            tmp_path / "m75.json",
            progress_m=20.0,
            full_fraction=0.25,
        )
    )
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
        no_jitter, num_envs=256, device="cpu", dtype=torch.float64, path_dim=12
    )
    env.reset(list(range(2000, 2256)))
    full_mask = env.episode_start_progress_m <= 0.0
    assert bool(full_mask.any()) and bool((~full_mask).any())
    assert torch.allclose(
        env.eta[full_mask, :3],
        env.path.start_ned_m_for(env.num_envs)[full_mask],
        atol=1e-6,
        rtol=0.0,
    )
    expected = env.path.point_at_s(env.episode_start_progress_m)
    assert torch.allclose(env.eta[:, :3], expected, atol=1e-6, rtol=0.0)
    near_projection = env.path.project(env.eta[:, :3])
    assert torch.allclose(
        near_projection.progress_m[~full_mask],
        torch.full_like(near_projection.progress_m[~full_mask], 20.0),
        atol=1e-5,
        rtol=0.0,
    )


def test_ppo_evaluator_forces_mixed_training_fraction_to_zero(tmp_path: Path) -> None:
    from scripts.evaluate_pc_ppo import _validate_environment_override

    mixed = load_config(
        _write_mixed_config(
            tmp_path / "mixed.json",
            progress_m=20.0,
            full_fraction=0.5,
        )
    )
    eval_zero = load_config(BASE_ENV)
    _validate_environment_override(
        mixed,
        eval_zero,
        allow_start_progress_training_override=True,
    )
    eval_mixed = replace(
        eval_zero,
        reference=replace(
            eval_zero.reference,
            start_progress_m=20.0,
            start_progress_full_fraction=0.5,
        ),
    )
    with pytest.raises(ValueError, match="start_progress_full_fraction=0"):
        _validate_environment_override(
            mixed,
            eval_mixed,
            allow_start_progress_training_override=True,
        )
