"""Translate the JSON PPO section into an rsl_rl runner config.

This module intentionally does not import rsl_rl.  G5 will turn the returned
dictionary into the installed version's runner config class, while the local
configuration tests remain runnable without Omniverse.
"""

from __future__ import annotations

from typing import Any

from auv_isaaclab.config import PpoSection


def build_rsl_rl_runner_cfg(ppo: PpoSection) -> dict[str, Any]:
    """Return the standard rsl_rl PPO runner fields for this experiment."""

    if not isinstance(ppo, PpoSection):
        raise TypeError("ppo must be a PpoSection")
    return {
        "num_steps_per_env": ppo.num_steps_per_env,
        "max_iterations": ppo.max_iterations,
        "save_interval": ppo.save_interval,
        "empirical_normalization": False,
        "clip_actions": 1.0,
        "policy": {
            "class_name": "ActorCritic",
            "actor_hidden_dims": list(ppo.actor_hidden_dims),
            "critic_hidden_dims": list(ppo.critic_hidden_dims),
            "activation": "elu",
            "init_noise_std": 1.0,
        },
        "algorithm": {
            "class_name": "PPO",
            "value_loss_coef": ppo.value_loss_coef,
            "use_clipped_value_loss": True,
            "clip_param": ppo.clip_param,
            "entropy_coef": ppo.entropy_coef,
            "num_learning_epochs": ppo.num_learning_epochs,
            "num_mini_batches": ppo.num_mini_batches,
            "learning_rate": ppo.learning_rate,
            "schedule": "fixed",
            "gamma": ppo.gamma,
            "lam": ppo.lam,
            "max_grad_norm": ppo.max_grad_norm,
        },
    }


__all__ = ["build_rsl_rl_runner_cfg"]
