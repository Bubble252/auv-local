"""IsaacLab DirectRLEnv wrapper for the batched P1 planar task."""

from __future__ import annotations

# pyright: reportMissingImports=false
from collections.abc import Sequence
from typing import Any

import gymnasium as gym
import torch
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

from auv_isaaclab.config import P1AvoidanceEnvCfg
from auv_isaaclab.tasks.p1_avoidance.batch import BatchStep, P1BatchCore


@configclass
class P1DirectRLEnvCfg(DirectRLEnvCfg):
    """IsaacLab-side configuration derived from the JSON P1 contract."""

    decimation = 1
    episode_length_s = 60.0
    action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,))
    observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(272,))
    state_space = 0
    sim: SimulationCfg = SimulationCfg(dt=0.02, render_interval=1)
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=1,
        env_spacing=1.0,
        replicate_physics=True,
        clone_in_fabric=True,
    )
    wait_for_textures = False
    rerender_on_reset = False
    ui_window_class_type = None


def build_p1_env_cfg(
    experiment_cfg: P1AvoidanceEnvCfg,
    *,
    device: str = "cuda:0",
) -> P1DirectRLEnvCfg:
    """Build an IsaacLab config without changing the frozen JSON semantics."""

    if not isinstance(experiment_cfg, P1AvoidanceEnvCfg):
        raise TypeError("experiment_cfg must be a P1AvoidanceEnvCfg")
    if abs(experiment_cfg.env.dt_s - experiment_cfg.env.integration_step_s) > 1e-12:
        raise ValueError("P1 DirectRLEnv requires dt_s == integration_step_s")

    cfg = P1DirectRLEnvCfg()
    cfg.seed = experiment_cfg.experiment.seed
    cfg.decimation = 1
    cfg.episode_length_s = (
        experiment_cfg.env.max_episode_steps * experiment_cfg.env.integration_step_s
    )
    cfg.observation_space = gym.spaces.Box(
        low=-1.0,
        high=1.0,
        shape=(experiment_cfg.policy_input_dim,),
    )
    cfg.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,))
    cfg.state_space = 0
    cfg.sim = SimulationCfg(
        dt=experiment_cfg.env.integration_step_s,
        render_interval=1,
        device=device,
    )
    cfg.scene = InteractiveSceneCfg(
        num_envs=experiment_cfg.env.num_envs,
        env_spacing=1.0,
        replicate_physics=True,
        clone_in_fabric=True,
    )
    cfg.wait_for_textures = False
    cfg.rerender_on_reset = False
    cfg.ui_window_class_type = None
    return cfg


class P1IsaacLabEnv(DirectRLEnv):
    """Expose the verified torch batch core through IsaacLab's direct workflow."""

    cfg: P1DirectRLEnvCfg

    def __init__(
        self,
        cfg: P1DirectRLEnvCfg,
        experiment_cfg: P1AvoidanceEnvCfg,
        render_mode: str | None = None,
        **kwargs: Any,
    ) -> None:
        self.experiment_cfg = experiment_cfg
        self._pending_actions: torch.Tensor | None = None
        self._last_step: BatchStep | None = None
        super().__init__(cfg, render_mode, **kwargs)
        self.core = P1BatchCore(
            experiment_cfg,
            device=self.device,
            dtype=torch.float32,
            seed=experiment_cfg.experiment.seed,
        )

    def _setup_scene(self) -> None:
        pass

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        if actions.shape != (self.num_envs, 2):
            raise ValueError(f"actions must have shape ({self.num_envs}, 2)")
        scale = torch.tensor(
            [
                self.experiment_cfg.env.action_max_accel_mps2,
                self.experiment_cfg.env.action_max_yaw_rate_rps,
            ],
            device=self.device,
            dtype=torch.float32,
        )
        self._pending_actions = torch.clamp(actions, -1.0, 1.0) * scale

    def _apply_action(self) -> None:
        if self._pending_actions is None:
            raise RuntimeError("_pre_physics_step must be called before _apply_action")
        self._last_step = self.core.step(self._pending_actions)

    def _get_observations(self) -> dict[str, torch.Tensor]:
        return {"policy": self.core.observation}

    def _get_rewards(self) -> torch.Tensor:
        if self._last_step is None:
            raise RuntimeError("_apply_action must be called before _get_rewards")
        return self._last_step.ppo_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self._last_step is None:
            raise RuntimeError("_apply_action must be called before _get_dones")
        return self._last_step.terminated, self._last_step.truncated

    def _reset_idx(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        elif torch.is_tensor(env_ids):
            ids = env_ids.to(device=self.device, dtype=torch.long)
        else:
            ids = torch.tensor(list(env_ids), device=self.device, dtype=torch.long)
        self.core.reset(ids, resample=True)
        self._last_step = None
        super()._reset_idx(ids)


__all__ = ["P1DirectRLEnvCfg", "P1IsaacLabEnv", "build_p1_env_cfg"]
