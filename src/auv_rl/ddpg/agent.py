from __future__ import annotations

import copy
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch  # type: ignore[import-not-found, unused-ignore]
from torch import nn  # type: ignore[import-not-found, unused-ignore]

from .networks import Critic, PolicyActor
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
    use_pper: bool = True
    per_alpha: float = 0.6
    per_beta_start: float = 0.4
    per_beta_steps: int = 100000
    per_epsilon: float = 1e-6
    grad_clip_norm: float = 1.0
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

    @property
    def action_dim(self) -> int:
        return len(self.action_low)


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
    target_actor: PolicyActor = field(init=False, repr=False)
    target_critic: Critic = field(init=False, repr=False)
    actor_optimizer: torch.optim.Optimizer = field(init=False, repr=False)
    critic_optimizer: torch.optim.Optimizer = field(init=False, repr=False)
    replay: PrioritizedSequenceReplayBuffer = field(init=False, repr=False)
    _episode_id: int = field(default=-1, init=False, repr=False)
    _episode_step: int = field(default=0, init=False, repr=False)
    _history: list[np.ndarray] = field(default_factory=list, init=False, repr=False)
    _rng: np.random.Generator = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.config, DDPGConfig):
            raise TypeError("config must be a DDPGConfig")
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        self._device = torch.device(self.config.device)
        self._rng = np.random.default_rng(self.seed)
        self.actor = PolicyActor(
            self.config.observation_dim,
            self.config.action_low,
            self.config.action_high,
            hidden_sizes=self.config.hidden_sizes,
            use_gru=self.config.use_gru,
            gru_hidden_size=self.config.gru_hidden_size,
            use_noisy=self.config.use_noisy,
        ).to(self._device)
        self.critic = Critic(
            self.config.observation_dim,
            self.config.action_dim,
            hidden_sizes=self.config.hidden_sizes,
            use_gru=self.config.use_gru,
            gru_hidden_size=self.config.gru_hidden_size,
            use_noisy=self.config.use_noisy,
        ).to(self._device)
        self.target_actor = copy.deepcopy(self.actor).eval()
        self.target_critic = copy.deepcopy(self.critic).eval()
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(),
            lr=self.config.actor_lr,
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(),
            lr=self.config.critic_lr,
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
        )

    @property
    def device(self) -> str:
        return str(self._device)

    def begin_episode(self, episode_id: int) -> None:
        if episode_id < 0:
            raise ValueError("episode_id must be non-negative")
        self._episode_id = int(episode_id)
        self._episode_step = 0
        self._history.clear()

    def _observation_tensor(self, observation: np.ndarray) -> torch.Tensor:
        obs = np.asarray(observation, dtype=np.float32)
        if obs.shape != (self.config.observation_dim,):
            raise ValueError("observation shape does not match agent config")
        if not np.all(np.isfinite(obs)):
            raise ValueError("observation must contain only finite values")
        return torch.as_tensor(obs, device=self._device)

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

    def observe(
        self,
        observation: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_observation: np.ndarray,
        done: bool,
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
        )
        self._episode_step += 1

    def _beta(self, gradient_step: int) -> float:
        fraction = min(float(gradient_step) / float(self.config.per_beta_steps), 1.0)
        return float(self.config.per_beta_start + fraction * (1.0 - self.config.per_beta_start))

    def _batch_tensors(self, batch: ReplayBatch) -> tuple[torch.Tensor, ...]:
        return (
            torch.as_tensor(batch.observations, device=self._device),
            torch.as_tensor(batch.actions, device=self._device),
            torch.as_tensor(batch.rewards, device=self._device),
            torch.as_tensor(batch.next_observations, device=self._device),
            torch.as_tensor(batch.dones, device=self._device),
            torch.as_tensor(batch.valid_masks, device=self._device),
            torch.as_tensor(batch.weights, device=self._device),
        )

    def update(self, gradient_step: int) -> UpdateMetrics | None:
        if len(self.replay) < self.config.learning_starts:
            return None
        beta = self._beta(gradient_step) if self.config.use_pper else 1.0
        batch = self.replay.sample(self.config.batch_size, beta=beta)
        obs, actions, rewards, next_obs, dones, masks, weights = self._batch_tensors(batch)
        self.critic.train()
        self.actor.train()
        with torch.no_grad():
            target_actions, _ = self.target_actor(
                next_obs,
                sample_noise=False,
            )
            target_q, _ = self.target_critic(
                next_obs,
                target_actions,
                sample_noise=False,
            )
            target_q = target_q.squeeze(-1)
            target_values = rewards + self.config.gamma * (1.0 - dones) * target_q

        current_q, _ = self.critic(
            obs,
            actions,
            sample_noise=self.config.use_noisy,
        )
        current_q = current_q.squeeze(-1)
        td_errors = current_q - target_values
        critic_loss = (td_errors.pow(2) * weights.unsqueeze(1) * masks).sum() / (
            masks.sum().clamp_min(1.0)
        )
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        if self.config.grad_clip_norm > 0.0:
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.config.grad_clip_norm)
        self.critic_optimizer.step()

        actor_actions, _ = self.actor(obs, sample_noise=self.config.use_noisy)
        actor_q, _ = self.critic(
            obs,
            actor_actions,
            sample_noise=self.config.use_noisy,
        )
        actor_loss = -(actor_q.squeeze(-1) * masks).sum() / masks.sum().clamp_min(1.0)
        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        if self.config.grad_clip_norm > 0.0:
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.config.grad_clip_norm)
        self.actor_optimizer.step()

        priority_errors = (td_errors.detach().abs() * masks).sum(dim=1) / masks.sum(
            dim=1
        ).clamp_min(1.0)
        self.replay.update_priorities(batch.indices, priority_errors.cpu().numpy())
        self._soft_update(self.actor, self.target_actor)
        self._soft_update(self.critic, self.target_critic)
        return UpdateMetrics(
            critic_loss=float(critic_loss.detach().cpu()),
            actor_loss=float(actor_loss.detach().cpu()),
            mean_q=float((actor_q.detach() * masks.unsqueeze(-1)).sum().cpu() / masks.sum().cpu()),
            mean_td_error=float(priority_errors.mean().detach().cpu()),
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
                "target_actor": self.target_actor.state_dict(),
                "target_critic": self.target_critic.state_dict(),
                "actor_optimizer": self.actor_optimizer.state_dict(),
                "critic_optimizer": self.critic_optimizer.state_dict(),
            },
            destination,
        )

    @classmethod
    def load(cls, path: Path | str, *, device: str | None = None) -> DDPGAgent:
        payload = torch.load(Path(path), map_location="cpu", weights_only=False)
        if not isinstance(payload, dict) or payload.get("format") != "auv_rl.ddpg.v1":
            raise ValueError("not an auv_rl DDPG checkpoint")
        raw_config = dict(payload["config"])
        if device is not None:
            raw_config["device"] = device
        for key in ("action_low", "action_high", "hidden_sizes"):
            raw_config[key] = tuple(raw_config[key])
        config = DDPGConfig(**raw_config)
        agent = cls(config, seed=int(payload.get("seed", 0)))
        agent.actor.load_state_dict(payload["actor"])
        agent.critic.load_state_dict(payload["critic"])
        agent.target_actor.load_state_dict(payload["target_actor"])
        agent.target_critic.load_state_dict(payload["target_critic"])
        agent.actor_optimizer.load_state_dict(payload["actor_optimizer"])
        agent.critic_optimizer.load_state_dict(payload["critic_optimizer"])
        return agent


__all__ = ["DDPGAgent", "DDPGConfig", "UpdateMetrics"]
