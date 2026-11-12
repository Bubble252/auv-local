from __future__ import annotations

from pathlib import Path

import numpy as np
import torch  # type: ignore[import-not-found, unused-ignore]

from ..ddpg.agent import DDPGConfig
from ..ddpg.networks import PolicyActor


class BCActorPolicy:
    """Inference-only actor loaded from a behavior-cloning checkpoint."""

    def __init__(self, actor: PolicyActor, config: DDPGConfig, *, seed: int = 0) -> None:
        self.actor = actor.eval()
        self.config = config
        self._device = torch.device(config.device)
        self._history: list[np.ndarray] = []
        torch.manual_seed(seed)

    @property
    def device(self) -> str:
        return str(self._device)

    def begin_episode(self, episode_id: int) -> None:
        if episode_id < 0:
            raise ValueError("episode_id must be non-negative")
        self._history.clear()

    def act(self, observation: np.ndarray, *, evaluate: bool = True) -> np.ndarray:
        obs = np.asarray(observation, dtype=np.float32)
        if obs.shape != (self.config.observation_dim,):
            raise ValueError("observation shape does not match BC actor config")
        if not np.all(np.isfinite(obs)):
            raise ValueError("observation must contain only finite values")
        self._history.append(obs.copy())
        if len(self._history) > self.config.history_length:
            del self._history[0]
        history = list(self._history)
        if len(history) < self.config.history_length:
            history = [history[0]] * (self.config.history_length - len(history)) + history
        sequence = torch.as_tensor(
            np.stack(history, axis=0),
            device=self._device,
        ).unsqueeze(0)
        self.actor.eval() if evaluate else self.actor.train()
        with torch.no_grad():
            action, _ = self.actor(sequence, sample_noise=False)
        return action[0, -1].detach().cpu().numpy().astype(np.float64)

    @classmethod
    def load(
        cls,
        path: Path | str,
        *,
        device: str | None = None,
        expected_observation_version: str | None = None,
    ) -> BCActorPolicy:
        payload = torch.load(Path(path), map_location="cpu", weights_only=False)
        if not isinstance(payload, dict) or payload.get("format") != "auv_rl.bc_actor.v1":
            raise ValueError("not an auv_rl BC actor checkpoint")
        if expected_observation_version is not None:
            spec = payload.get("observation_spec")
            version = None if not isinstance(spec, dict) else spec.get("version")
            if version != expected_observation_version:
                raise ValueError(
                    "BC actor observation version does not match the evaluation environment: "
                    f"checkpoint={version!r}, environment={expected_observation_version!r}"
                )
        raw_config = dict(payload["config"])
        if device is not None:
            raw_config["device"] = device
        for key in ("action_low", "action_high", "hidden_sizes"):
            raw_config[key] = tuple(raw_config[key])
        config = DDPGConfig(**raw_config)
        actor = PolicyActor(
            config.observation_dim,
            config.action_low,
            config.action_high,
            hidden_sizes=config.hidden_sizes,
            use_gru=config.use_gru,
            gru_hidden_size=config.gru_hidden_size,
            use_noisy=config.use_noisy,
            activation=config.resolved_actor_activation,
            layer_norm=config.layer_norm,
        ).to(torch.device(config.device))
        actor.load_state_dict(payload["actor"])
        return cls(actor, config, seed=int(payload.get("seed", 0)))


__all__ = ["BCActorPolicy"]
