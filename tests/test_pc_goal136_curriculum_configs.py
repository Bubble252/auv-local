from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from auv_lowlevel.batch.pc_env import BatchedPCLowLevelEnv
from auv_lowlevel.config import load_config
from auv_lowlevel.curriculum import CurriculumStageSampler, curriculum_stage_mix_target
from auv_lowlevel.ppo_config import load_training_config
from scripts.build_goal136_curriculum_configs import (
    PILOT_TRANSITIONS_SAC,
    STAGES,
    STAGE_DEPTH,
    STAGE_Z_MAX,
    build_environment_config,
    build_ppo_pilot,
    build_sac_pilot,
)


@pytest.mark.parametrize("stage", STAGES)
def test_curriculum_environment_configs_load(tmp_path: Path, stage: str) -> None:
    path = tmp_path / f"{stage}.json"
    path.write_text(json.dumps(build_environment_config(stage)), encoding="utf-8")
    config = load_config(path)
    assert config.curriculum.enabled is True
    assert config.curriculum.initial_stage == stage
    for key, weight in config.curriculum.stage_mix:
        assert weight == pytest.approx(dict(curriculum_stage_mix_target(stage, preview_enabled=True))[key])
    assert config.reward.hold_bonus is (stage in {"C0", "C1"})
    assert config.plant.bounds_ned_m[2].tolist() == [0.0, STAGE_Z_MAX[stage]]
    assert config.termination.min_depth_m == 0.0
    assert config.termination.max_depth_m == STAGE_Z_MAX[stage]
    depth_low, depth_high = STAGE_DEPTH[stage]
    assert config.reference.depth_start_m == pytest.approx(0.5 * (depth_low + depth_high))


def test_curriculum_deep_stages_extend_only_sim_bounds() -> None:
    assert STAGE_Z_MAX["C0"] == 200.0
    assert STAGE_Z_MAX["C3"] == 200.0
    assert STAGE_Z_MAX["C4"] > STAGE_DEPTH["C4"][1]
    assert STAGE_Z_MAX["C5"] > STAGE_DEPTH["C5"][1]
    assert STAGE_Z_MAX["C6"] > STAGE_DEPTH["C6"][1]


def test_curriculum_pilot_configs_share_budget_and_protocol(tmp_path: Path) -> None:
    env_path = tmp_path / "C0.json"
    env_path.write_text(json.dumps(build_environment_config("C0")), encoding="utf-8")
    for arm in ("C-full", "C-hard", "C-no-current"):
        ppo_path = tmp_path / f"{arm}_ppo.json"
        ppo_path.write_text(json.dumps(build_ppo_pilot(arm, env_path)), encoding="utf-8")
        ppo = load_training_config(ppo_path)
        assert ppo.iterations == 40
        assert ppo.num_envs == 1000
        assert ppo.pc is not None
        assert ppo.pc.actor_current_encoder_enabled is True
        assert ppo.environment_config.endswith("C0.json") if arm != "C-hard" else True

        payload = build_sac_pilot(arm, env_path)
        assert payload["training"]["total_env_transitions"] == PILOT_TRANSITIONS_SAC
        assert payload["training"]["num_envs"] == 256
        assert payload["training"]["authority_regularization"]["enabled"] is False
        assert payload["observation"]["current_dim"] == 3
        assert payload["policy"]["current_encoder_alpha"] == (
            0.0 if arm in {"C-full", "C-no-current"} else 1.0
        )
        assert payload["evaluation"]["seed_start"] == 20
        assert payload["evaluation"]["seed_count"] == 40
        assert payload["experiment"]["heldout_seed_start"] == 100
        assert payload["experiment"]["heldout_closed"] is True


def test_mixed_curriculum_direct_commands_are_per_row(tmp_path: Path) -> None:
    path = tmp_path / "C0.json"
    path.write_text(json.dumps(build_environment_config("C0")), encoding="utf-8")
    config = load_config(path)
    sampler = CurriculumStageSampler(
        stage_mix=(("C0", 1.0), ("C1", 1.0), ("C2", 1.0))
    )
    seeds_by_stage: dict[str, int] = {}
    for seed in range(512):
        spec = sampler.sample(seed)
        seeds_by_stage.setdefault(spec.stage, seed)
        if set(seeds_by_stage) == {"C0", "C1", "C2"}:
            break
    assert set(seeds_by_stage) == {"C0", "C1", "C2"}
    seeds = [seeds_by_stage[stage] for stage in ("C0", "C0", "C1", "C1", "C2", "C2")]
    env = BatchedPCLowLevelEnv(
        config,
        num_envs=len(seeds),
        device="cpu",
        dtype=torch.float64,
        path_dim=12,
        curriculum_sampler=sampler,
    )
    observation = env.reset(seeds)
    specs = sampler.sample_batch(seeds)
    expected = torch.tensor([spec.stage in {"C0", "C1"} for spec in specs])
    assert env._curriculum_direct_mask is not None
    assert torch.equal(env._curriculum_direct_mask.cpu(), expected)
    for row, direct in enumerate(expected.tolist()):
        if direct:
            assert torch.allclose(
                observation.command.heading_rad[row],
                env._direct_command.heading_rad[row],
            )
            assert torch.allclose(
                observation.command.speed_mps[row],
                env._direct_command.speed_mps[row],
            )

    disabled = BatchedPCLowLevelEnv(
        config,
        num_envs=len(seeds),
        device="cpu",
        dtype=torch.float64,
        path_dim=12,
        curriculum_sampler=sampler,
        curriculum_direct_commands_enabled=False,
    )
    disabled.reset(seeds)
    assert disabled._curriculum_direct_mask is not None
    assert not bool(disabled._curriculum_direct_mask.any())


def test_non_curriculum_c1_direct_scaffold_can_be_disabled(tmp_path: Path) -> None:
    path = tmp_path / "C1.json"
    path.write_text(json.dumps(build_environment_config("C1")), encoding="utf-8")
    config = load_config(path)
    enabled = BatchedPCLowLevelEnv(
        config,
        num_envs=1,
        device="cpu",
        dtype=torch.float64,
        path_dim=12,
    )
    assert enabled.direct_commands is True
    disabled = BatchedPCLowLevelEnv(
        config,
        num_envs=1,
        device="cpu",
        dtype=torch.float64,
        path_dim=12,
        curriculum_direct_commands_enabled=False,
    )
    assert disabled.direct_commands is False
