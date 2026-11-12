"""SAC with a training-side hinge penalty on deterministic residual authority."""

from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np
import torch as th
from torch.nn import functional as F

from stable_baselines3.common.utils import polyak_update
from stable_baselines3.sac.sac import SAC

AUTHORITY_FORMS = ("hinge_rms",)
RMS_EPS = 1e-8


def parse_authority_regularization(training: Mapping[str, Any]) -> dict[str, Any]:
    """Parse and validate the optional ``training.authority_regularization`` block."""

    raw = training.get("authority_regularization", {})
    if raw is None:
        raw = {}
    if not isinstance(raw, Mapping):
        raise ValueError("training.authority_regularization must be an object")
    enabled = bool(raw.get("enabled", False))
    form = str(raw.get("form", "hinge_rms"))
    if form not in AUTHORITY_FORMS:
        raise ValueError(f"unsupported authority regularization form: {form}")
    rho = float(raw.get("rho", 0.0))
    beta = float(raw.get("beta", 0.0))
    if not math.isfinite(rho) or rho < 0.0:
        raise ValueError("authority_regularization.rho must be finite and non-negative")
    if not math.isfinite(beta) or beta < 0.0:
        raise ValueError("authority_regularization.beta must be finite and non-negative")
    if enabled and (rho <= 0.0 or beta <= 0.0):
        raise ValueError("enabled authority regularization requires rho > 0 and beta > 0")
    if not enabled:
        beta = 0.0
    return {
        "enabled": enabled,
        "form": form,
        "rho": rho,
        "beta": beta,
        "target": "deterministic_mean",
    }


def authority_hinge_penalty(
    actions: th.Tensor,
    *,
    rho: float,
    beta: float,
) -> th.Tensor:
    """Return ``beta * mean(ReLU(RMS(action) - rho)^2)`` for a batch of actions."""

    if actions.ndim != 2:
        raise ValueError("actions must have shape (batch, action_dim)")
    if not math.isfinite(float(rho)) or rho < 0.0:
        raise ValueError("rho must be finite and non-negative")
    if not math.isfinite(float(beta)) or beta < 0.0:
        raise ValueError("beta must be finite and non-negative")
    if beta == 0.0:
        return actions.new_zeros(())
    action_rms = th.sqrt(th.mean(actions.square(), dim=1) + RMS_EPS)
    excess = th.relu(action_rms - float(rho))
    return float(beta) * th.mean(excess.square())


class AuthorityRegularizedSAC(SAC):
    """SAC whose actor loss includes a soft deterministic-action RMS constraint.

    The critic target, entropy objective, and environment contract are unchanged.
    With ``authority_beta=0`` the training path is the stock SB3 SAC path.
    """

    def __init__(
        self,
        *args: Any,
        authority_rho: float = 0.0,
        authority_beta: float = 0.0,
        authority_form: str = "hinge_rms",
        **kwargs: Any,
    ) -> None:
        if authority_form not in AUTHORITY_FORMS:
            raise ValueError(f"unsupported authority regularization form: {authority_form}")
        if not math.isfinite(float(authority_rho)) or authority_rho < 0.0:
            raise ValueError("authority_rho must be finite and non-negative")
        if not math.isfinite(float(authority_beta)) or authority_beta < 0.0:
            raise ValueError("authority_beta must be finite and non-negative")
        if authority_beta > 0.0 and authority_rho <= 0.0:
            raise ValueError("authority_rho must be positive when authority_beta > 0")
        self.authority_rho = float(authority_rho)
        self.authority_beta = float(authority_beta)
        self.authority_form = str(authority_form)
        self.authority_metrics: dict[str, Any] = {
            "enabled": self.authority_beta > 0.0,
            "form": self.authority_form,
            "rho": self.authority_rho,
            "beta": self.authority_beta,
            "target": "deterministic_mean",
            "updates": 0,
            "rms_last": None,
            "rms_mean": None,
            "penalty_mean": None,
        }
        super().__init__(*args, **kwargs)

    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        # This mirrors stable-baselines3 2.x SAC.train() with one added actor-loss
        # term. Keeping the stock branch inline makes beta=0 behavior auditable.
        self.policy.set_training_mode(True)
        optimizers = [self.actor.optimizer, self.critic.optimizer]
        if self.ent_coef_optimizer is not None:
            optimizers += [self.ent_coef_optimizer]
        self._update_learning_rate(optimizers)

        ent_coef_losses: list[float] = []
        ent_coefs: list[float] = []
        actor_losses: list[float] = []
        critic_losses: list[float] = []
        authority_rms_values: list[float] = []
        authority_penalty_values: list[float] = []

        for gradient_step in range(gradient_steps):
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)  # type: ignore[union-attr]
            discounts = replay_data.discounts if replay_data.discounts is not None else self.gamma

            if self.use_sde:
                self.actor.reset_noise()

            actions_pi, log_prob = self.actor.action_log_prob(replay_data.observations)
            log_prob = log_prob.reshape(-1, 1)

            ent_coef_loss = None
            if self.ent_coef_optimizer is not None and self.log_ent_coef is not None:
                ent_coef = th.exp(self.log_ent_coef.detach())
                assert isinstance(self.target_entropy, float)
                ent_coef_loss = -(self.log_ent_coef * (log_prob + self.target_entropy).detach()).mean()
                ent_coef_losses.append(ent_coef_loss.item())
            else:
                ent_coef = self.ent_coef_tensor
            ent_coefs.append(ent_coef.item())

            if ent_coef_loss is not None and self.ent_coef_optimizer is not None:
                self.ent_coef_optimizer.zero_grad()
                ent_coef_loss.backward()
                self.ent_coef_optimizer.step()

            with th.no_grad():
                next_actions, next_log_prob = self.actor.action_log_prob(replay_data.next_observations)
                next_q_values = th.cat(self.critic_target(replay_data.next_observations, next_actions), dim=1)
                next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
                next_q_values = next_q_values - ent_coef * next_log_prob.reshape(-1, 1)
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * discounts * next_q_values

            current_q_values = self.critic(replay_data.observations, replay_data.actions)
            critic_loss = 0.5 * sum(F.mse_loss(current_q, target_q_values) for current_q in current_q_values)
            assert isinstance(critic_loss, th.Tensor)
            critic_losses.append(critic_loss.item())

            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

            q_values_pi = th.cat(self.critic(replay_data.observations, actions_pi), dim=1)
            min_qf_pi, _ = th.min(q_values_pi, dim=1, keepdim=True)
            sac_actor_loss = (ent_coef * log_prob - min_qf_pi).mean()

            authority_penalty = sac_actor_loss.new_zeros(())
            if self.authority_beta > 0.0:
                mean_actions, _, _ = self.actor.get_action_dist_params(replay_data.observations)
                deterministic_actions = th.tanh(mean_actions)
                action_rms = th.sqrt(th.mean(deterministic_actions.square(), dim=1) + RMS_EPS)
                authority_penalty = authority_hinge_penalty(
                    deterministic_actions,
                    rho=self.authority_rho,
                    beta=self.authority_beta,
                )
                authority_rms_values.append(float(action_rms.detach().mean().cpu()))
                authority_penalty_values.append(float(authority_penalty.detach().cpu()))

            actor_loss = sac_actor_loss + authority_penalty
            actor_losses.append(float(actor_loss.item()))

            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            self.actor.optimizer.step()

            if gradient_step % self.target_update_interval == 0:
                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
                polyak_update(self.batch_norm_stats, self.batch_norm_stats_target, 1.0)

        self._n_updates += gradient_steps
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/ent_coef", np.mean(ent_coefs))
        self.logger.record("train/actor_loss", np.mean(actor_losses))
        self.logger.record("train/critic_loss", np.mean(critic_losses))
        if ent_coef_losses:
            self.logger.record("train/ent_coef_loss", np.mean(ent_coef_losses))

        if authority_rms_values:
            previous_updates = int(self.authority_metrics.get("updates", 0))
            previous_rms = self.authority_metrics.get("rms_mean")
            previous_penalty = self.authority_metrics.get("penalty_mean")
            count = previous_updates + len(authority_rms_values)
            current_rms = float(np.mean(authority_rms_values))
            current_penalty = float(np.mean(authority_penalty_values))
            self.authority_metrics.update(
                {
                    "updates": count,
                    "rms_last": float(authority_rms_values[-1]),
                    "rms_mean": (
                        current_rms
                        if previous_rms is None
                        else float(
                            (float(previous_rms) * previous_updates + current_rms * len(authority_rms_values))
                            / count
                        )
                    ),
                    "penalty_mean": (
                        current_penalty
                        if previous_penalty is None
                        else float(
                            (float(previous_penalty) * previous_updates + current_penalty * len(authority_penalty_values))
                            / count
                        )
                    ),
                }
            )
            self.logger.record("train/authority_rms", current_rms)
            self.logger.record("train/authority_penalty", current_penalty)
