from __future__ import annotations

import copy
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch  # type: ignore[import-not-found, unused-ignore]
from torch import nn  # type: ignore[import-not-found, unused-ignore]

from .networks import Critic, NoisyLinear, PolicyActor
from .replay import PrioritizedSequenceReplayBuffer, ReplayBatch


@dataclass(frozen=True)
class DDPGConfig:
    observation_dim: int
    action_low: tuple[float, ...] = (-0.5, -math.radians(15.0))
    action_high: tuple[float, ...] = (0.5, math.radians(15.0))
    hidden_sizes: tuple[int, ...] = (160, 154)
    gru_hidden_size: int = 64
    history_length: int = 10
    gamma: float = 0.98
    tau: float = 0.005
    actor_lr: float = 2e-4
    critic_lr: float = 2e-4
    batch_size: int = 64
    buffer_size: int = 100000
    learning_starts: int = 1000
    use_noisy: bool = True
    use_gru: bool = True
    activation: str = "relu"
    actor_activation: str | None = None
    critic_activation: str | None = None
    layer_norm: bool = False
    actor_output_noise: bool = True
    actor_output_sigma_init: float = 0.5
    actor_hidden_sigma_init: float = 0.5
    actor_noise_sigma_trainable: bool = True
    actor_output_weight_init_bound: float | None = None
    actor_output_scale: float = 1.0
    actor_output_squash: str = "tanh"
    policy_delay: int = 1
    actor_warmup_updates: int = 0
    twin_critics: bool = False
    target_policy_noise_sigma: float = 0.0
    target_policy_noise_clip: float = 0.5
    saturation_penalty: float = 0.0
    saturation_threshold_fraction: float = 0.9
    privileged_dim: int = 0
    bc_auxiliary_weight: float = 0.0
    bc_auxiliary_decay_updates: int = 0
    use_pper: bool = True
    per_alpha: float = 0.6
    per_beta_start: float = 0.4
    per_beta_steps: int = 100000
    per_epsilon: float = 1e-6
    grad_clip_norm: float = 1.0
    cuda_graph_update: bool = False
    device: str = "cpu"

    def __post_init__(self) -> None:
        if self.observation_dim <= 0:
            raise ValueError("observation_dim must be positive")
        if len(self.action_low) != len(self.action_high) or not self.action_low:
            raise ValueError("action bounds must be non-empty and equal length")
        if any(low > high for low, high in zip(self.action_low, self.action_high, strict=True)):
            raise ValueError("action_low must not exceed action_high")
        if self.history_length <= 0:
            raise ValueError("history_length must be positive")
        if self.batch_size <= 0 or self.buffer_size <= 0:
            raise ValueError("batch_size and buffer_size must be positive")
        if self.learning_starts < 0:
            raise ValueError("learning_starts must be non-negative")
        if self.activation.strip().lower() not in {"relu", "leaky_relu", "leakyrelu", "silu"}:
            raise ValueError("activation must be relu, leaky_relu, or silu")
        for name in ("actor_activation", "critic_activation"):
            value = getattr(self, name)
            if value is not None and value.strip().lower() not in {
                "relu",
                "leaky_relu",
                "leakyrelu",
                "silu",
            }:
                raise ValueError(f"{name} must be relu, leaky_relu, or silu")
        if self.policy_delay <= 0:
            raise ValueError("policy_delay must be positive")
        if self.actor_warmup_updates < 0:
            raise ValueError("actor_warmup_updates must be non-negative")
        if self.cuda_graph_update and self.actor_warmup_updates > 0:
            raise ValueError("cuda_graph_update does not support actor_warmup_updates")
        if not math.isfinite(self.actor_output_sigma_init) or self.actor_output_sigma_init <= 0.0:
            raise ValueError("actor_output_sigma_init must be finite and positive")
        if not math.isfinite(self.actor_hidden_sigma_init) or self.actor_hidden_sigma_init <= 0.0:
            raise ValueError("actor_hidden_sigma_init must be finite and positive")
        if not isinstance(self.actor_noise_sigma_trainable, bool):
            raise TypeError("actor_noise_sigma_trainable must be a bool")
        if self.actor_output_weight_init_bound is not None and (
            not math.isfinite(self.actor_output_weight_init_bound)
            or self.actor_output_weight_init_bound <= 0.0
        ):
            raise ValueError("actor_output_weight_init_bound must be finite and positive")
        if not math.isfinite(self.actor_output_scale) or not 0.0 < self.actor_output_scale <= 1.0:
            raise ValueError("actor_output_scale must be finite and in (0, 1]")
        if str(self.actor_output_squash).strip().lower() not in {"tanh", "softsign"}:
            raise ValueError("actor_output_squash must be 'tanh' or 'softsign'")
        if self.target_policy_noise_sigma < 0.0:
            raise ValueError("target_policy_noise_sigma must be non-negative")
        if self.target_policy_noise_clip < 0.0:
            raise ValueError("target_policy_noise_clip must be non-negative")
        if self.saturation_penalty < 0.0:
            raise ValueError("saturation_penalty must be non-negative")
        if not 0.0 < self.saturation_threshold_fraction < 1.0:
            raise ValueError("saturation_threshold_fraction must be in (0, 1)")
        if self.privileged_dim < 0:
            raise ValueError("privileged_dim must be non-negative")
        if self.bc_auxiliary_weight < 0.0:
            raise ValueError("bc_auxiliary_weight must be non-negative")
        if self.bc_auxiliary_decay_updates < 0:
            raise ValueError("bc_auxiliary_decay_updates must be non-negative")
        if not 0.0 < self.gamma <= 1.0:
            raise ValueError("gamma must be in (0, 1]")
        if not 0.0 < self.tau <= 1.0:
            raise ValueError("tau must be in (0, 1]")
        if self.actor_lr <= 0.0 or self.critic_lr <= 0.0:
            raise ValueError("learning rates must be positive")
        if not 0.0 < self.per_alpha <= 1.0:
            raise ValueError("per_alpha must be in (0, 1]")
        if not 0.0 <= self.per_beta_start <= 1.0:
            raise ValueError("per_beta_start must be in [0, 1]")
        if self.per_beta_steps <= 0:
            raise ValueError("per_beta_steps must be positive")
        if self.per_epsilon <= 0.0:
            raise ValueError("per_epsilon must be positive")
        if not isinstance(self.cuda_graph_update, bool):
            raise TypeError("cuda_graph_update must be a bool")

    @property
    def action_dim(self) -> int:
        return len(self.action_low)

    @property
    def resolved_actor_activation(self) -> str:
        return self.actor_activation or self.activation

    @property
    def resolved_critic_activation(self) -> str:
        return self.critic_activation or self.activation


@dataclass
class UpdateMetrics:
    critic_loss: float
    actor_loss: float
    mean_q: float
    mean_td_error: float
    beta: float


@dataclass
class DDPGAgent:
    config: DDPGConfig
    seed: int = 0
    _device: torch.device = field(init=False, repr=False)
    actor: PolicyActor = field(init=False, repr=False)
    critic: Critic = field(init=False, repr=False)
    critic2: Critic | None = field(init=False, repr=False)
    target_actor: PolicyActor = field(init=False, repr=False)
    target_critic: Critic = field(init=False, repr=False)
    target_critic2: Critic | None = field(init=False, repr=False)
    actor_optimizer: torch.optim.Optimizer = field(init=False, repr=False)
    critic_optimizer: torch.optim.Optimizer = field(init=False, repr=False)
    critic2_optimizer: torch.optim.Optimizer | None = field(init=False, repr=False)
    replay: PrioritizedSequenceReplayBuffer = field(init=False, repr=False)
    _episode_id: int = field(default=-1, init=False, repr=False)
    _episode_step: int = field(default=0, init=False, repr=False)
    _history: list[np.ndarray] = field(default_factory=list, init=False, repr=False)
    _env_episode_ids: dict[int, int] = field(default_factory=dict, init=False, repr=False)
    _env_episode_steps: dict[int, int] = field(default_factory=dict, init=False, repr=False)
    _env_histories: dict[int, list[np.ndarray]] = field(
        default_factory=dict, init=False, repr=False
    )
    _env_hidden: dict[int, torch.Tensor] = field(default_factory=dict, init=False, repr=False)
    _bc_observations: torch.Tensor | None = field(default=None, init=False, repr=False)
    _bc_actions: torch.Tensor | None = field(default=None, init=False, repr=False)
    _gradient_updates: int = field(default=0, init=False, repr=False)
    last_replay_sample_time_s: float = field(default=0.0, init=False, repr=False)
    last_pper_update_time_s: float = field(default=0.0, init=False, repr=False)
    _cuda_graph: Any | None = field(default=None, init=False, repr=False)
    _cuda_graph_inputs: tuple[torch.Tensor, ...] | None = field(
        default=None, init=False, repr=False
    )
    _cuda_graph_outputs: tuple[torch.Tensor, ...] | None = field(
        default=None, init=False, repr=False
    )
    _action_half_range: torch.Tensor = field(init=False, repr=False)
    _rng: np.random.Generator = field(init=False, repr=False)
    _observation_spec: dict[str, Any] | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.config, DDPGConfig):
            raise TypeError("config must be a DDPGConfig")
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        self._device = torch.device(self.config.device)
        self._rng = np.random.default_rng(self.seed)
        self._action_half_range = torch.as_tensor(
            [
                0.5 * (high - low)
                for low, high in zip(
                    self.config.action_low,
                    self.config.action_high,
                    strict=True,
                )
            ],
            dtype=torch.float32,
            device=self._device,
        )
        self.actor = PolicyActor(
            self.config.observation_dim,
            self.config.action_low,
            self.config.action_high,
            hidden_sizes=self.config.hidden_sizes,
            use_gru=self.config.use_gru,
            gru_hidden_size=self.config.gru_hidden_size,
            use_noisy=self.config.use_noisy,
            activation=self.config.resolved_actor_activation,
            layer_norm=self.config.layer_norm,
            actor_output_noise=self.config.actor_output_noise,
            actor_output_sigma_init=self.config.actor_output_sigma_init,
            actor_hidden_sigma_init=self.config.actor_hidden_sigma_init,
            actor_output_weight_init_bound=self.config.actor_output_weight_init_bound,
            actor_output_scale=self.config.actor_output_scale,
            actor_output_squash=self.config.actor_output_squash,
        ).to(self._device)
        self.critic = Critic(
            self.config.observation_dim + self.config.privileged_dim,
            self.config.action_dim,
            hidden_sizes=self.config.hidden_sizes,
            use_gru=self.config.use_gru,
            gru_hidden_size=self.config.gru_hidden_size,
            use_noisy=self.config.use_noisy,
            activation=self.config.resolved_critic_activation,
            layer_norm=self.config.layer_norm,
        ).to(self._device)
        self.critic2 = (
            Critic(
                self.config.observation_dim + self.config.privileged_dim,
                self.config.action_dim,
                hidden_sizes=self.config.hidden_sizes,
                use_gru=self.config.use_gru,
                gru_hidden_size=self.config.gru_hidden_size,
                use_noisy=self.config.use_noisy,
                activation=self.config.resolved_critic_activation,
                layer_norm=self.config.layer_norm,
            ).to(self._device)
            if self.config.twin_critics
            else None
        )
        if not self.config.actor_noise_sigma_trainable:
            for module in self.actor.modules():
                if isinstance(module, NoisyLinear):
                    module.weight_rho.requires_grad_(False)
                    module.bias_rho.requires_grad_(False)
        self.target_actor = copy.deepcopy(self.actor).eval()
        self.target_critic = copy.deepcopy(self.critic).eval()
        self.target_critic2 = (
            copy.deepcopy(self.critic2).eval() if self.critic2 is not None else None
        )
        capturable = self.config.cuda_graph_update and self._device.type == "cuda"
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(),
            lr=self.config.actor_lr,
            capturable=capturable,
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(),
            lr=self.config.critic_lr,
            capturable=capturable,
        )
        self.critic2_optimizer = (
            torch.optim.Adam(
                self.critic2.parameters(),
                lr=self.config.critic_lr,
                capturable=capturable,
            )
            if self.critic2 is not None
            else None
        )
        self.replay = PrioritizedSequenceReplayBuffer(
            self.config.buffer_size,
            self.config.observation_dim,
            self.config.action_dim,
            history_length=self.config.history_length,
            use_pper=self.config.use_pper,
            alpha=self.config.per_alpha,
            epsilon=self.config.per_epsilon,
            seed=self.seed,
            privileged_dim=self.config.privileged_dim,
        )

    @property
    def device(self) -> str:
        return str(self._device)

    @property
    def cuda_graph_active(self) -> bool:
        return self._cuda_graph is not None

    def set_observation_spec(self, spec: dict[str, Any]) -> None:
        """Attach the observation contract to future checkpoints."""

        name = spec.get("name")
        version = spec.get("version")
        shape = spec.get("shape")
        if not isinstance(name, str) or not name:
            raise ValueError("observation spec name must be a non-empty string")
        if not isinstance(version, str) or not version:
            raise ValueError("observation spec version must be a non-empty string")
        if not isinstance(shape, (list, tuple)) or not shape:
            raise ValueError("observation spec shape must be a non-empty sequence")
        if tuple(int(value) for value in shape) != (self.config.observation_dim,):
            raise ValueError("observation spec shape does not match the agent")
        self._observation_spec = {
            "name": name,
            "version": version,
            "shape": [int(value) for value in shape],
        }

    def _check_observation_version(self, expected: str | None) -> None:
        if expected is None:
            return
        spec = self._observation_spec
        version = None if spec is None else spec.get("version")
        if version != expected:
            raise ValueError(
                "checkpoint observation version does not match the environment: "
                f"checkpoint={version!r}, environment={expected!r}"
            )

    def begin_episode(self, episode_id: int) -> None:
        if episode_id < 0:
            raise ValueError("episode_id must be non-negative")
        self._episode_id = int(episode_id)
        self._episode_step = 0
        self._history.clear()

    def begin_episode_for_env(self, env_id: int, episode_id: int) -> None:
        """Reset one environment's episode state without touching other envs."""

        if env_id < 0:
            raise ValueError("env_id must be non-negative")
        if episode_id < 0:
            raise ValueError("episode_id must be non-negative")
        self._env_episode_ids[int(env_id)] = int(episode_id)
        self._env_episode_steps[int(env_id)] = 0
        self._env_histories[int(env_id)] = []
        self._env_hidden.pop(int(env_id), None)

    def _require_env_episode(self, env_id: int) -> tuple[int, int]:
        if env_id not in self._env_episode_ids:
            raise RuntimeError(f"begin_episode_for_env() must be called for env {env_id}")
        return self._env_episode_ids[env_id], self._env_episode_steps[env_id]

    def _append_env_history(self, env_id: int, observation: np.ndarray) -> None:
        history = self._env_histories.setdefault(env_id, [])
        history.append(observation.copy())
        if len(history) > self.config.history_length:
            del history[0]

    def env_history(self, env_id: int) -> tuple[np.ndarray, ...]:
        return tuple(item.copy() for item in self._env_histories.get(env_id, ()))

    def env_episode_state(self, env_id: int) -> tuple[int, int]:
        return self._require_env_episode(env_id)

    def _observation_tensor(self, observation: np.ndarray) -> torch.Tensor:
        obs = np.asarray(observation, dtype=np.float32)
        if obs.shape != (self.config.observation_dim,):
            raise ValueError("observation shape does not match agent config")
        if not np.all(np.isfinite(obs)):
            raise ValueError("observation must contain only finite values")
        return torch.as_tensor(obs, device=self._device)

    def _critic_observation(
        self,
        observation: torch.Tensor,
        privileged: torch.Tensor,
    ) -> torch.Tensor:
        if self.config.privileged_dim == 0:
            return observation
        if privileged.shape[:-1] != observation.shape[:-1]:
            raise ValueError("privileged and observation leading dimensions must match")
        if privileged.shape[-1] != self.config.privileged_dim:
            raise ValueError("privileged feature dimension does not match config")
        return torch.cat((observation, privileged), dim=-1)

    def act(
        self,
        observation: np.ndarray,
        *,
        evaluate: bool = False,
    ) -> np.ndarray:
        obs_tensor = self._observation_tensor(observation)
        self._history.append(obs_tensor.cpu().numpy())
        if len(self._history) > self.config.history_length:
            del self._history[0]
        sequence = torch.as_tensor(
            np.stack(self._history, axis=0),
            device=self._device,
        ).unsqueeze(0)
        self.actor.eval() if evaluate else self.actor.train()
        with torch.no_grad():
            action_tensor, _ = self.actor(
                sequence,
                sample_noise=not evaluate and self.config.use_noisy,
            )
        return action_tensor[0, -1].detach().cpu().numpy().astype(np.float64)

    def act_for_env(
        self,
        env_id: int,
        observation: np.ndarray,
        *,
        evaluate: bool = False,
    ) -> np.ndarray:
        return self.act_batch([env_id], np.asarray(observation)[None, :], evaluate=evaluate)[0]

    def act_batch(
        self,
        env_ids: list[int] | tuple[int, ...],
        observations: np.ndarray,
        *,
        evaluate: bool = False,
    ) -> np.ndarray:
        """Act for multiple envs with isolated history, GRU hidden and noise.

        The legacy :meth:`act` rebuilds the actor sequence from the episode
        history on every call and starts the GRU from a zero hidden state.  A
        single-step forward with a persistent hidden state is not equivalent
        once the history window is truncated, so this method groups envs by
        history length and forwards the actual episode-local sequence.
        """

        ids = tuple(int(env_id) for env_id in env_ids)
        if not ids or len(set(ids)) != len(ids):
            raise ValueError("env_ids must be non-empty and unique")
        obs = np.asarray(observations, dtype=np.float32)
        if obs.shape != (len(ids), self.config.observation_dim):
            raise ValueError("observations shape does not match env_ids and agent config")
        if not np.all(np.isfinite(obs)):
            raise ValueError("observations must contain only finite values")
        for row, env_id in enumerate(ids):
            self._require_env_episode(env_id)
            self._append_env_history(env_id, obs[row])

        lengths = {len(self._env_histories[env_id]) for env_id in ids}
        actions = np.empty((len(ids), self.config.action_dim), dtype=np.float64)
        self.actor.eval() if evaluate else self.actor.train()
        with torch.no_grad():
            for length in sorted(lengths):
                rows = [
                    row
                    for row, env_id in enumerate(ids)
                    if len(self._env_histories[env_id]) == length
                ]
                sequence = torch.as_tensor(
                    np.stack(
                        [np.stack(self._env_histories[ids[row]], axis=0) for row in rows], axis=0
                    ),
                    device=self._device,
                )
                per_sample_noise = not evaluate and self.config.use_noisy and len(rows) > 1
                action_tensor, next_hidden = self.actor(
                    sequence,
                    sample_noise=not evaluate and self.config.use_noisy,
                    per_sample_noise=per_sample_noise,
                )
                actions[rows] = action_tensor[:, -1].detach().cpu().numpy().astype(np.float64)
                if next_hidden is not None:
                    for column, row in enumerate(rows):
                        self._env_hidden[ids[row]] = (
                            next_hidden[:, column : column + 1, :].detach().clone()
                        )
        return actions

    def observe(
        self,
        observation: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_observation: np.ndarray,
        done: bool,
        privileged: np.ndarray | None = None,
        next_privileged: np.ndarray | None = None,
    ) -> None:
        if self._episode_id < 0:
            raise RuntimeError("begin_episode() must be called before observe()")
        self.replay.add(
            observation,
            action,
            reward,
            next_observation,
            done,
            episode_id=self._episode_id,
            episode_step=self._episode_step,
            privileged=privileged,
            next_privileged=next_privileged,
        )
        self._episode_step += 1

    def observe_for_env(
        self,
        env_id: int,
        observation: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_observation: np.ndarray,
        done: bool,
        privileged: np.ndarray | None = None,
        next_privileged: np.ndarray | None = None,
    ) -> None:
        self.observe_batch(
            [env_id],
            np.asarray(observation, dtype=np.float32)[None, :],
            np.asarray(action, dtype=np.float32)[None, :],
            np.asarray([reward], dtype=np.float32),
            np.asarray(next_observation, dtype=np.float32)[None, :],
            np.asarray([done], dtype=np.bool_),
            privileged=None if privileged is None else np.asarray(privileged)[None, :],
            next_privileged=(
                None if next_privileged is None else np.asarray(next_privileged)[None, :]
            ),
        )

    def observe_batch(
        self,
        env_ids: list[int] | tuple[int, ...],
        observations: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_observations: np.ndarray,
        dones: np.ndarray,
        privileged: np.ndarray | None = None,
        next_privileged: np.ndarray | None = None,
    ) -> None:
        ids = tuple(int(env_id) for env_id in env_ids)
        if not ids or len(set(ids)) != len(ids):
            raise ValueError("env_ids must be non-empty and unique")
        obs = np.asarray(observations, dtype=np.float32)
        act = np.asarray(actions, dtype=np.float32)
        next_obs = np.asarray(next_observations, dtype=np.float32)
        reward_array = np.asarray(rewards, dtype=np.float32)
        done_array = np.asarray(dones, dtype=np.bool_)
        batch_size = len(ids)
        if obs.shape != (batch_size, self.config.observation_dim):
            raise ValueError("observations shape does not match env_ids and agent config")
        if act.shape != (batch_size, self.config.action_dim):
            raise ValueError("actions shape does not match env_ids and agent config")
        if next_obs.shape != obs.shape or reward_array.shape != (batch_size,):
            raise ValueError("next observations and rewards have incompatible shapes")
        if done_array.shape != (batch_size,):
            raise ValueError("dones must have shape (B,)")
        episode_ids: list[int] = []
        episode_steps: list[int] = []
        for env_id in ids:
            episode_id, episode_step = self._require_env_episode(env_id)
            episode_ids.append(episode_id)
            episode_steps.append(episode_step)
        self.replay.add_batch(
            obs,
            act,
            reward_array,
            next_obs,
            done_array,
            episode_ids=np.asarray(episode_ids, dtype=np.int64),
            episode_steps=np.asarray(episode_steps, dtype=np.int64),
            privileged=privileged,
            next_privileged=next_privileged,
            env_ids=np.asarray(ids, dtype=np.int64),
        )
        for env_id in ids:
            self._env_episode_steps[env_id] += 1

    def _beta(self, gradient_step: int) -> float:
        fraction = min(float(gradient_step) / float(self.config.per_beta_steps), 1.0)
        return float(self.config.per_beta_start + fraction * (1.0 - self.config.per_beta_start))

    def _batch_tensors(self, batch: ReplayBatch) -> tuple[torch.Tensor, ...]:
        return (
            torch.as_tensor(batch.observations, device=self._device),
            torch.as_tensor(batch.privileged, device=self._device),
            torch.as_tensor(batch.actions, device=self._device),
            torch.as_tensor(batch.rewards, device=self._device),
            torch.as_tensor(batch.next_observations, device=self._device),
            torch.as_tensor(batch.next_privileged, device=self._device),
            torch.as_tensor(batch.dones, device=self._device),
            torch.as_tensor(batch.valid_masks, device=self._device),
            torch.as_tensor(batch.weights, device=self._device),
        )

    def _can_capture_cuda_graph(self) -> bool:
        return bool(
            self.config.cuda_graph_update
            and self._device.type == "cuda"
            and self.config.policy_delay == 1
            and self.config.bc_auxiliary_weight == 0.0
            and self._bc_observations is None
            and self.actor_optimizer.state
            and self.critic_optimizer.state
        )

    def _update_core(
        self,
        obs: torch.Tensor,
        privileged: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_obs: torch.Tensor,
        next_privileged: torch.Tensor,
        dones: torch.Tensor,
        masks: torch.Tensor,
        weights: torch.Tensor,
        *,
        update_actor: bool,
    ) -> torch.Tensor:
        critic_obs = self._critic_observation(obs, privileged)
        next_critic_obs = self._critic_observation(next_obs, next_privileged)
        self.critic.train()
        self.actor.train()
        with torch.no_grad():
            target_actions, _ = self.target_actor(
                next_obs,
                sample_noise=False,
            )
            if self.config.target_policy_noise_sigma > 0.0:
                noise = torch.randn_like(target_actions) * self.config.target_policy_noise_sigma
                noise = noise.clamp(
                    -self.config.target_policy_noise_clip,
                    self.config.target_policy_noise_clip,
                )
                low = torch.as_tensor(self.config.action_low, device=self._device)
                high = torch.as_tensor(self.config.action_high, device=self._device)
                target_actions = (target_actions + noise).clamp(low, high)
            target_q, _ = self.target_critic(
                next_critic_obs,
                target_actions,
                sample_noise=False,
            )
            target_q = target_q.squeeze(-1)
            if self.target_critic2 is not None:
                target_q2, _ = self.target_critic2(
                    next_critic_obs,
                    target_actions,
                    sample_noise=False,
                )
                target_q = torch.minimum(target_q, target_q2.squeeze(-1))
            target_values = rewards + self.config.gamma * (1.0 - dones) * target_q

        current_q, _ = self.critic(
            critic_obs,
            actions,
            sample_noise=self.config.use_noisy,
        )
        current_q = current_q.squeeze(-1)
        td_errors = current_q - target_values
        critic_loss = (td_errors.pow(2) * weights.unsqueeze(1) * masks).sum() / (
            masks.sum().clamp_min(1.0)
        )
        critic_loss_metric = critic_loss.detach()
        if (
            self.critic2 is not None
            and self.critic2_optimizer is not None
            and self.target_critic2 is not None
        ):
            current_q2, _ = self.critic2(
                critic_obs,
                actions,
                sample_noise=self.config.use_noisy,
            )
            current_q2 = current_q2.squeeze(-1)
            td_errors2 = current_q2 - target_values
            critic_loss2 = (td_errors2.pow(2) * weights.unsqueeze(1) * masks).sum() / (
                masks.sum().clamp_min(1.0)
            )
            critic_loss_metric = critic_loss_metric + critic_loss2.detach()
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        if self.config.grad_clip_norm > 0.0:
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.config.grad_clip_norm)
        self.critic_optimizer.step()
        if (
            self.critic2 is not None
            and self.critic2_optimizer is not None
            and self.target_critic2 is not None
        ):
            self.critic2_optimizer.zero_grad(set_to_none=True)
            critic_loss2.backward()
            if self.config.grad_clip_norm > 0.0:
                nn.utils.clip_grad_norm_(
                    self.critic2.parameters(),
                    self.config.grad_clip_norm,
                )
            self.critic2_optimizer.step()

        if update_actor:
            actor_actions, _ = self.actor(obs, sample_noise=self.config.use_noisy)
            actor_q, _ = self.critic(
                critic_obs,
                actor_actions,
                sample_noise=self.config.use_noisy,
            )
            actor_loss = -(actor_q.squeeze(-1) * masks).sum() / masks.sum().clamp_min(1.0)
            if self.config.saturation_penalty > 0.0:
                normalized = actor_actions.abs() / self._action_half_range.clamp_min(1e-9)
                excess = torch.relu(normalized - self.config.saturation_threshold_fraction)
                actor_loss = actor_loss + self.config.saturation_penalty * excess.mean()
            if self._bc_observations is not None and self._bc_actions is not None:
                bc_batch_size = min(self.config.batch_size, len(self._bc_observations))
                bc_indices = torch.randint(
                    0,
                    len(self._bc_observations),
                    (bc_batch_size,),
                    device=self._device,
                )
                bc_prediction, _ = self.actor(
                    self._bc_observations[bc_indices],
                    sample_noise=False,
                )
                bc_loss = nn.functional.mse_loss(
                    bc_prediction[:, -1],
                    self._bc_actions[bc_indices],
                )
                actor_loss = actor_loss + self.bc_auxiliary_beta * bc_loss
            self.actor_optimizer.zero_grad(set_to_none=True)
            actor_loss.backward()
            if self.config.grad_clip_norm > 0.0:
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.config.grad_clip_norm)
            self.actor_optimizer.step()
            actor_loss_value = actor_loss.detach()
            actor_q_value = (actor_q.detach() * masks.unsqueeze(-1)).sum() / masks.sum()
        else:
            with torch.no_grad():
                actor_actions, _ = self.actor(obs, sample_noise=False)
                actor_q, _ = self.critic(critic_obs, actor_actions, sample_noise=False)
                actor_loss_value = torch.zeros((), device=self._device)
                actor_q_value = (actor_q * masks.unsqueeze(-1)).sum() / masks.sum()

        priority_errors = (td_errors.detach().abs() * masks).sum(dim=1) / masks.sum(
            dim=1
        ).clamp_min(1.0)
        if self.critic2 is not None:
            priority_errors = torch.maximum(priority_errors, td_errors2.detach().abs().mean(dim=1))
        if update_actor:
            self._soft_update(self.actor, self.target_actor)
        self._soft_update(self.critic, self.target_critic)
        if self.critic2 is not None and self.target_critic2 is not None:
            self._soft_update(self.critic2, self.target_critic2)
        return torch.cat(
            (
                priority_errors.detach(),
                torch.stack(
                    (
                        critic_loss_metric.reshape(1),
                        actor_loss_value.detach().reshape(1),
                        actor_q_value.detach().reshape(1),
                        priority_errors.mean().detach().reshape(1),
                    )
                ).reshape(4),
            )
        )

    def _capture_cuda_graph(self, inputs: tuple[torch.Tensor, ...]) -> None:
        module_states = {
            name: {key: value.detach().clone() for key, value in module.state_dict().items()}
            for name, module in (
                ("actor", self.actor),
                ("critic", self.critic),
                ("target_actor", self.target_actor),
                ("target_critic", self.target_critic),
            )
        }
        optimizer_states = {
            "actor": copy.deepcopy(self.actor_optimizer.state_dict()),
            "critic": copy.deepcopy(self.critic_optimizer.state_dict()),
        }
        if (
            self.critic2 is not None
            and self.critic2_optimizer is not None
            and self.target_critic2 is not None
        ):
            module_states["critic2"] = {
                key: value.detach().clone() for key, value in self.critic2.state_dict().items()
            }
            module_states["target_critic2"] = {
                key: value.detach().clone()
                for key, value in self.target_critic2.state_dict().items()
            }
            optimizer_states["critic2"] = copy.deepcopy(self.critic2_optimizer.state_dict())
        cpu_rng = torch.get_rng_state()
        cuda_rng = torch.cuda.get_rng_state_all()

        warmup_stream: Any = torch.cuda.Stream()
        warmup_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(warmup_stream):
            for _ in range(3):
                self._update_core(*inputs, update_actor=True)
        torch.cuda.current_stream().wait_stream(warmup_stream)
        torch.cuda.synchronize()

        self.actor.load_state_dict(module_states["actor"])
        self.critic.load_state_dict(module_states["critic"])
        self.target_actor.load_state_dict(module_states["target_actor"])
        self.target_critic.load_state_dict(module_states["target_critic"])
        self.actor_optimizer.load_state_dict(optimizer_states["actor"])
        self.critic_optimizer.load_state_dict(optimizer_states["critic"])
        if (
            self.critic2 is not None
            and self.critic2_optimizer is not None
            and self.target_critic2 is not None
        ):
            self.critic2.load_state_dict(module_states["critic2"])
            self.target_critic2.load_state_dict(module_states["target_critic2"])
            self.critic2_optimizer.load_state_dict(optimizer_states["critic2"])
        torch.set_rng_state(cpu_rng)
        torch.cuda.set_rng_state_all(cuda_rng)
        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = self._update_core(*inputs, update_actor=True)
        torch.cuda.synchronize()
        torch.set_rng_state(cpu_rng)
        torch.cuda.set_rng_state_all(cuda_rng)
        self._cuda_graph = graph
        self._cuda_graph_inputs = inputs
        self._cuda_graph_outputs = (output,)

    def _run_cuda_graph_update(
        self,
        inputs: tuple[torch.Tensor, ...],
        *,
        indices: np.ndarray,
        beta: float,
    ) -> UpdateMetrics:
        if self._cuda_graph is None or self._cuda_graph_inputs is None:
            raise RuntimeError("CUDA graph update requested before capture")
        if self._cuda_graph_outputs is None:
            raise RuntimeError("CUDA graph outputs are missing")
        if len(inputs) != len(self._cuda_graph_inputs):
            raise ValueError("CUDA graph input count mismatch")
        for source, destination in zip(inputs, self._cuda_graph_inputs, strict=True):
            destination.copy_(source)
        self._cuda_graph.replay()
        combined = self._cuda_graph_outputs[0]
        values = combined.cpu().numpy()
        priorities = values[: self.config.batch_size]
        critic_loss, actor_loss, mean_q, mean_td_error = values[-4:].tolist()
        priority_started = time.perf_counter()
        self.replay.update_priorities(indices, priorities)
        self.last_pper_update_time_s = time.perf_counter() - priority_started
        return UpdateMetrics(
            critic_loss=float(critic_loss),
            actor_loss=float(actor_loss),
            mean_q=float(mean_q),
            mean_td_error=float(mean_td_error),
            beta=beta,
        )

    def update(self, gradient_step: int) -> UpdateMetrics | None:
        if len(self.replay) < self.config.learning_starts:
            return None
        self._gradient_updates += 1
        beta = self._beta(gradient_step) if self.config.use_pper else 1.0
        sample_started = time.perf_counter()
        batch = self.replay.sample(self.config.batch_size, beta=beta)
        self.last_replay_sample_time_s = time.perf_counter() - sample_started
        inputs = self._batch_tensors(batch)
        if self._cuda_graph is None and self._can_capture_cuda_graph():
            self._capture_cuda_graph(inputs)
        if self._cuda_graph is not None:
            return self._run_cuda_graph_update(inputs, indices=batch.indices, beta=beta)
        update_actor = (gradient_step + 1) % self.config.policy_delay == 0
        if self._gradient_updates <= self.config.actor_warmup_updates:
            update_actor = False
        combined = self._update_core(
            *inputs,
            update_actor=update_actor,
        )
        values = combined.cpu().numpy()
        priorities = values[: self.config.batch_size]
        critic_loss, actor_loss, mean_q, mean_td_error = values[-4:].tolist()
        priority_started = time.perf_counter()
        self.replay.update_priorities(batch.indices, priorities)
        self.last_pper_update_time_s = time.perf_counter() - priority_started
        return UpdateMetrics(
            critic_loss=float(critic_loss),
            actor_loss=float(actor_loss),
            mean_q=float(mean_q),
            mean_td_error=float(mean_td_error),
            beta=beta,
        )

    def _soft_update(self, source: nn.Module, target: nn.Module) -> None:
        with torch.no_grad():
            for source_parameter, target_parameter in zip(
                source.parameters(),
                target.parameters(),
                strict=True,
            ):
                target_parameter.mul_(1.0 - self.config.tau)
                target_parameter.add_(self.config.tau * source_parameter)

    @property
    def bc_auxiliary_beta(self) -> float:
        if self._bc_observations is None or self.config.bc_auxiliary_weight <= 0.0:
            return 0.0
        if self.config.bc_auxiliary_decay_updates <= 0:
            return self.config.bc_auxiliary_weight
        fraction = min(
            float(getattr(self, "_gradient_updates", 0))
            / float(self.config.bc_auxiliary_decay_updates),
            1.0,
        )
        return self.config.bc_auxiliary_weight * (1.0 - fraction)

    def configure_behavior_cloning(
        self,
        observations: np.ndarray,
        actions: np.ndarray,
    ) -> None:
        obs = np.asarray(observations, dtype=np.float32)
        act = np.asarray(actions, dtype=np.float32)
        if obs.ndim != 3 or obs.shape[2] != self.config.observation_dim:
            raise ValueError("BC observations must have shape (N,T,D)")
        if act.shape != (obs.shape[0], self.config.action_dim):
            raise ValueError("BC actions must have shape (N,A)")
        if not np.all(np.isfinite(obs)) or not np.all(np.isfinite(act)):
            raise ValueError("BC data must contain only finite values")
        self._bc_observations = torch.as_tensor(obs, device=self._device)
        self._bc_actions = torch.as_tensor(act, device=self._device)

    def reset_actor_noise_sigmas(self) -> None:
        """Reset NoisyNet sigma after loading deterministic BC actor weights."""

        self.actor.reset_noise_sigmas()
        self.target_actor.load_state_dict(self.actor.state_dict())

    def load_actor_checkpoint(
        self,
        path: Path | str,
        *,
        expected_observation_version: str | None = None,
    ) -> None:
        payload = torch.load(Path(path), map_location="cpu", weights_only=False)
        if not isinstance(payload, dict) or "actor" not in payload:
            raise ValueError("actor checkpoint does not contain an actor state dict")
        if expected_observation_version is not None:
            spec = payload.get("observation_spec")
            version = None if not isinstance(spec, dict) else spec.get("version")
            if version != expected_observation_version:
                raise ValueError(
                    "actor observation version does not match the environment: "
                    f"checkpoint={version!r}, environment={expected_observation_version!r}"
                )
            if not isinstance(spec, dict):
                raise ValueError("actor checkpoint observation spec must be an object")
            self._observation_spec = {
                "name": str(spec.get("name")),
                "version": str(version),
                "shape": list(spec.get("shape", (self.config.observation_dim,))),
            }
        self.actor.load_state_dict(payload["actor"])
        self.target_actor.load_state_dict(self.actor.state_dict())

    def save(self, path: Path | str) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "format": "auv_rl.ddpg.v1",
                "config": asdict(self.config),
                "seed": self.seed,
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
                "critic2": self.critic2.state_dict() if self.critic2 is not None else None,
                "target_actor": self.target_actor.state_dict(),
                "target_critic": self.target_critic.state_dict(),
                "target_critic2": (
                    self.target_critic2.state_dict() if self.target_critic2 is not None else None
                ),
                "actor_optimizer": self.actor_optimizer.state_dict(),
                "critic_optimizer": self.critic_optimizer.state_dict(),
                "critic2_optimizer": (
                    self.critic2_optimizer.state_dict()
                    if self.critic2_optimizer is not None
                    else None
                ),
                "observation_spec": self._observation_spec,
            },
            destination,
        )

    @classmethod
    def load(
        cls,
        path: Path | str,
        *,
        device: str | None = None,
        expected_observation_version: str | None = None,
    ) -> DDPGAgent:
        payload = torch.load(Path(path), map_location="cpu", weights_only=False)
        if not isinstance(payload, dict) or payload.get("format") != "auv_rl.ddpg.v1":
            raise ValueError("not an auv_rl DDPG checkpoint")
        spec = payload.get("observation_spec")
        if expected_observation_version is not None:
            version = None if not isinstance(spec, dict) else spec.get("version")
            if version != expected_observation_version:
                raise ValueError(
                    "checkpoint observation version does not match the environment: "
                    f"checkpoint={version!r}, environment={expected_observation_version!r}"
                )
        raw_config = dict(payload["config"])
        if device is not None:
            raw_config["device"] = device
        for key in ("action_low", "action_high", "hidden_sizes"):
            raw_config[key] = tuple(raw_config[key])
        config = DDPGConfig(**raw_config)
        agent = cls(config, seed=int(payload.get("seed", 0)))
        if isinstance(spec, dict):
            agent._observation_spec = {
                "name": str(spec.get("name")),
                "version": str(spec.get("version")),
                "shape": [int(value) for value in spec.get("shape", ())],
            }
        agent._check_observation_version(expected_observation_version)
        agent.actor.load_state_dict(payload["actor"])
        agent.critic.load_state_dict(payload["critic"])
        if agent.critic2 is not None and payload.get("critic2") is not None:
            agent.critic2.load_state_dict(payload["critic2"])
        agent.target_actor.load_state_dict(payload["target_actor"])
        agent.target_critic.load_state_dict(payload["target_critic"])
        if agent.target_critic2 is not None and payload.get("target_critic2") is not None:
            agent.target_critic2.load_state_dict(payload["target_critic2"])
        agent.actor_optimizer.load_state_dict(payload["actor_optimizer"])
        agent.critic_optimizer.load_state_dict(payload["critic_optimizer"])
        if agent.critic2_optimizer is not None and payload.get("critic2_optimizer") is not None:
            agent.critic2_optimizer.load_state_dict(payload["critic2_optimizer"])
        return agent

    def training_state_dict(self) -> dict[str, Any]:
        """Capture optimizer, replay, RNG and per-environment episode state."""

        module_states: dict[str, Any] = {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "target_actor": self.target_actor.state_dict(),
            "target_critic": self.target_critic.state_dict(),
        }
        optimizer_states: dict[str, Any] = {
            "actor": copy.deepcopy(self.actor_optimizer.state_dict()),
            "critic": copy.deepcopy(self.critic_optimizer.state_dict()),
        }
        if (
            self.critic2 is not None
            and self.critic2_optimizer is not None
            and self.target_critic2 is not None
        ):
            module_states["critic2"] = self.critic2.state_dict()
            module_states["target_critic2"] = self.target_critic2.state_dict()
            optimizer_states["critic2"] = copy.deepcopy(self.critic2_optimizer.state_dict())

        cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
        return {
            "format": "auv_rl.ddpg.training_state.v1",
            "config": asdict(self.config),
            "seed": self.seed,
            "observation_spec": self._observation_spec,
            "modules": module_states,
            "optimizers": optimizer_states,
            "rng": {
                "numpy": copy.deepcopy(self._rng.bit_generator.state),
                "torch": torch.get_rng_state().clone(),
                "cuda": [state.clone() for state in cuda_rng],
            },
            "episodes": {
                "episode_id": int(self._episode_id),
                "episode_step": int(self._episode_step),
                "history": [item.copy() for item in self._history],
                "env_episode_ids": dict(self._env_episode_ids),
                "env_episode_steps": dict(self._env_episode_steps),
                "env_histories": {
                    int(env_id): [item.copy() for item in history]
                    for env_id, history in self._env_histories.items()
                },
                "env_hidden": {
                    int(env_id): hidden.detach().cpu().clone()
                    for env_id, hidden in self._env_hidden.items()
                },
            },
            "replay": self.replay.state_dict(),
            "gradient_updates": int(self._gradient_updates),
        }

    def load_training_state_dict(
        self,
        state: dict[str, Any],
        *,
        expected_observation_version: str | None = None,
    ) -> None:
        """Restore a training snapshot created by :meth:`training_state_dict`."""

        if not isinstance(state, dict) or state.get("format") != "auv_rl.ddpg.training_state.v1":
            raise ValueError("not an auv_rl DDPG training-state checkpoint")
        state_spec = state.get("observation_spec")
        state_version = None if not isinstance(state_spec, dict) else state_spec.get("version")
        expected = expected_observation_version
        if expected is None and self._observation_spec is not None:
            expected = self._observation_spec.get("version")
        if expected is not None and state_version != expected:
            raise ValueError(
                "training-state observation version does not match the environment: "
                f"checkpoint={state_version!r}, environment={expected!r}"
            )
        raw_config = dict(state["config"])
        current_config = asdict(self.config)
        for key, value in raw_config.items():
            if key == "device":
                continue
            if key in {"action_low", "action_high", "hidden_sizes"}:
                value = tuple(value)
            if current_config.get(key) != value:
                raise ValueError(f"training-state config field {key!r} does not match agent")

        modules = state["modules"]
        self.actor.load_state_dict(modules["actor"])
        self.critic.load_state_dict(modules["critic"])
        self.target_actor.load_state_dict(modules["target_actor"])
        self.target_critic.load_state_dict(modules["target_critic"])
        if self.critic2 is not None and modules.get("critic2") is not None:
            self.critic2.load_state_dict(modules["critic2"])
        if self.target_critic2 is not None and modules.get("target_critic2") is not None:
            self.target_critic2.load_state_dict(modules["target_critic2"])

        optimizers = state["optimizers"]
        self.actor_optimizer.load_state_dict(optimizers["actor"])
        self.critic_optimizer.load_state_dict(optimizers["critic"])
        if self.critic2_optimizer is not None and optimizers.get("critic2") is not None:
            self.critic2_optimizer.load_state_dict(optimizers["critic2"])

        episodes = state["episodes"]
        self._episode_id = int(episodes["episode_id"])
        self._episode_step = int(episodes["episode_step"])
        self._history = [np.asarray(item, dtype=np.float32).copy() for item in episodes["history"]]
        self._env_episode_ids = {
            int(env_id): int(episode_id)
            for env_id, episode_id in dict(episodes["env_episode_ids"]).items()
        }
        self._env_episode_steps = {
            int(env_id): int(step) for env_id, step in dict(episodes["env_episode_steps"]).items()
        }
        self._env_histories = {
            int(env_id): [np.asarray(item, dtype=np.float32).copy() for item in history]
            for env_id, history in dict(episodes["env_histories"]).items()
        }
        self._env_hidden = {
            int(env_id): hidden.to(self._device).detach().clone()
            for env_id, hidden in dict(episodes["env_hidden"]).items()
        }
        self.replay.load_state_dict(state["replay"])
        self._gradient_updates = int(state.get("gradient_updates", 0))

        rng = state["rng"]
        self._rng.bit_generator.state = copy.deepcopy(rng["numpy"])
        torch.set_rng_state(rng["torch"].cpu())
        cuda_states = list(rng.get("cuda", ()))
        if (
            self._device.type == "cuda"
            and torch.cuda.is_available()
            and cuda_states
            and len(cuda_states) == torch.cuda.device_count()
        ):
            torch.cuda.set_rng_state_all([item.cpu() for item in cuda_states])

        self._cuda_graph = None
        self._cuda_graph_inputs = None
        self._cuda_graph_outputs = None


__all__ = ["DDPGAgent", "DDPGConfig", "UpdateMetrics"]
