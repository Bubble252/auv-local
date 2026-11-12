from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch

from auv_lowlevel.base_controller import sha256_file
from auv_lowlevel.bc import ACTION_SCALES, BCActor
from auv_lowlevel.config import load_config
from auv_lowlevel.vector_env import PvsVecEnv

CONFIG_PATH = (
    Path(__file__).resolve().parents[1]
    / "configs"
    / "lowlevel"
    / "odarrl_control_i_residual.json"
)


def _write_checkpoint(path: Path) -> None:
    model = BCActor(27, (8,))
    torch.save(
        {
            "schema_version": "auv_odarrl_bc.v1",
            "observation_dim": 27,
            "hidden_sizes": [8],
            "state_dict": model.state_dict(),
            "observation_mean": torch.zeros(27),
            "observation_std": torch.ones(27),
            "action_scales": torch.from_numpy(ACTION_SCALES),
        },
        path,
    )


def test_pvs_vec_env_contract_autoreset_and_worker_recovery(tmp_path: Path) -> None:
    checkpoint = tmp_path / "pi0.pt"
    _write_checkpoint(checkpoint)
    config = load_config(CONFIG_PATH)
    env = PvsVecEnv(
        config,
        CONFIG_PATH,
        num_envs=2,
        base_mode="frozen_bc",
        checkpoint_path=checkpoint,
        checkpoint_sha256=sha256_file(checkpoint),
        max_episode_steps=2,
        reset_timeout_s=60.0,
        step_timeout_s=20.0,
        max_restarts=1,
    )
    try:
        observations, extras = env.get_observations()
        assert observations.shape == (2, 27)
        assert observations.dtype == torch.float32
        assert extras == {"observations": {}}
        assert env.max_episode_length == 2

        first = env.step(torch.zeros((2, 3), dtype=torch.float32))
        assert first[0].shape == (2, 27)
        assert first[1].shape == (2,)
        assert first[3]["observations"] == {}
        assert not bool(first[2].any())

        second = env.step(torch.zeros((2, 3), dtype=torch.float32))
        assert bool(second[2].all())
        assert bool(second[3]["time_outs"].all())
        assert "/episode/success_rate" in second[3]["log"]
        assert torch.equal(env.episode_length_buf, torch.zeros(2, dtype=torch.long))
        assert np.all(np.isfinite(second[0].numpy()))

        pid = env.worker_pids[0]
        assert pid is not None
        os.kill(pid, 9)
        env._slots[0].process.join(timeout=5.0)  # type: ignore[attr-defined]
        recovered = env.step(torch.zeros((2, 3), dtype=torch.float32))
        assert bool(recovered[2][0])
        assert env.worker_restart_counts[0] == 1
        assert np.all(np.isfinite(recovered[0].numpy()))
    finally:
        env.close()
