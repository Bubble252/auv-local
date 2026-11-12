from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch  # type: ignore[import-not-found, unused-ignore]

from .agent import DDPGAgent


def action_grid(agent: DDPGAgent) -> torch.Tensor:
    """Return the 3x3 diagnostic action grid in physical units."""

    low = torch.as_tensor(agent.config.action_low, dtype=torch.float32)
    high = torch.as_tensor(agent.config.action_high, dtype=torch.float32)
    axes = [torch.linspace(float(low[index]), float(high[index]), 3) for index in range(2)]
    first, second = torch.meshgrid(axes[0], axes[1], indexing="ij")
    return torch.stack((first.reshape(-1), second.reshape(-1)), dim=-1)


def diagnose_critic(
    agent: DDPGAgent,
    observations: np.ndarray,
    privileged: np.ndarray | None = None,
) -> dict[str, Any]:
    """Measure whether Q and its action gradient vary on real P1 observations.

    The function intentionally uses no exploration noise. A collapsed critic
    therefore remains observable instead of being hidden by NoisyNet noise.
    """

    obs = np.asarray(observations, dtype=np.float32)
    if obs.ndim == 2:
        obs = obs[:, None, :]
    if obs.ndim != 3 or obs.shape[0] == 0 or obs.shape[2] != agent.config.observation_dim:
        raise ValueError("observations must have shape (N,D) or (N,T,D)")
    if not np.all(np.isfinite(obs)):
        raise ValueError("observations must contain only finite values")

    # cuDNN GRUs reject backward in eval mode. The critic has no dropout and
    # NoisyNet sampling is disabled, so train mode is deterministic here while
    # still allowing the action-gradient probe.
    agent.critic.train()
    agent.actor.eval()
    device = torch.device(agent.device)
    observation_tensor = torch.as_tensor(obs, device=device)
    if agent.config.privileged_dim > 0:
        privileged_array = (
            np.zeros(
                (*obs.shape[:-1], agent.config.privileged_dim),
                dtype=np.float32,
            )
            if privileged is None
            else np.asarray(privileged, dtype=np.float32)
        )
        if privileged_array.shape != (*obs.shape[:-1], agent.config.privileged_dim):
            raise ValueError("privileged features do not match observation batch")
        privileged_tensor = torch.as_tensor(privileged_array, device=device)
        critic_observation = torch.cat((observation_tensor, privileged_tensor), dim=-1)
    else:
        critic_observation = observation_tensor
    # The diagnostic grid sweeps the first two action dimensions, matching the
    # original P1 2-D action space.  PC command residuals have a third
    # (speed) axis; keep it at the neutral zero command so the probe still
    # evaluates the real critic without building an exponential 3-D grid.
    grid = action_grid(agent).to(device)
    action_grid_tensor = torch.zeros(
        (grid.shape[0], agent.config.action_dim),
        dtype=grid.dtype,
        device=device,
    )
    action_grid_tensor[:, : min(2, agent.config.action_dim)] = grid
    sample_count, time_count, observation_dim = observation_tensor.shape
    repeated_obs = critic_observation.repeat_interleave(action_grid_tensor.shape[0], dim=0)
    repeated_actions = action_grid_tensor.unsqueeze(0).expand(sample_count, -1, -1)
    repeated_actions = repeated_actions.reshape(
        -1, 1, agent.config.action_dim
    ).expand(-1, time_count, -1)
    repeated_actions = repeated_actions.clone().requires_grad_(True)

    captured: list[torch.Tensor] = []

    def capture_activation(_module: torch.nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
        assert isinstance(output, torch.Tensor)
        captured.append(output.detach())

    handle = agent.critic.first_activation().register_forward_hook(capture_activation)
    try:
        q_values, _ = agent.critic(
            repeated_obs,
            repeated_actions,
            sample_noise=False,
        )
        q_last = q_values[:, -1, 0]
        q_by_state = q_last.reshape(sample_count, action_grid_tensor.shape[0])
        q_range = q_by_state.max(dim=1).values - q_by_state.min(dim=1).values
        q_std = q_by_state.std(dim=1, unbiased=False)
        action_gradient = torch.autograd.grad(
            outputs=q_values.sum(),
            inputs=repeated_actions,
            retain_graph=False,
            create_graph=False,
        )[0]
        dq_da = action_gradient.abs().mean()
    finally:
        handle.remove()

    if not captured:
        raise RuntimeError("critic activation hook did not run")
    activations = captured[0]
    active = (activations.abs() > 1e-8).to(torch.float64).mean()

    with torch.no_grad():
        raw, _ = agent.actor.forward_raw(observation_tensor, sample_noise=False)
        squash = agent.actor.squash(raw)
        squash_derivative = agent.actor.squash_derivative(raw)
        raw_abs = raw.abs()
        bounded_action = agent.actor.actor_output_scale * squash
        raw_threshold = 9.0 if agent.actor.actor_output_squash == "softsign" else 3.0

    return {
        "schema": "auv_p1_critic_health.v1",
        "sample_count": int(sample_count),
        "time_count": int(time_count),
        "q_range_mean": float(q_range.mean().detach().cpu()),
        "q_std_mean": float(q_std.mean().detach().cpu()),
        "dq_da_mean": float(dq_da.detach().cpu()),
        "activation_active_fraction": float(active.detach().cpu()),
        "first_layer_dead_fraction": float(1.0 - active.detach().cpu()),
        "actor_raw_abs_max": float(raw.abs().max().detach().cpu()),
        "actor_raw_abs_mean": float(raw_abs.mean().detach().cpu()),
        "actor_raw_abs_p95": float(torch.quantile(raw_abs.flatten(), 0.95).detach().cpu()),
        "raw_saturation_fraction": float(
            (raw_abs >= raw_threshold).to(torch.float64).mean().detach().cpu()
        ),
        "actor_output_scale": float(agent.actor.actor_output_scale),
        "actor_output_squash": str(agent.actor.actor_output_squash),
        "bounded_action_abs_mean": float(bounded_action.abs().mean().detach().cpu()),
        "bounded_action_abs_max": float(bounded_action.abs().max().detach().cpu()),
        "bounded_saturation_fraction": float(
            (squash.abs() >= 0.9).to(torch.float64).mean().detach().cpu()
        ),
        "squash_grad_norm": float(squash_derivative.mean().detach().cpu()),
        "tanh_grad_norm": float(squash_derivative.mean().detach().cpu()),
        "q_min": float(q_by_state.min().detach().cpu()),
        "q_max": float(q_by_state.max().detach().cpu()),
    }


def critic_health_passes(health: dict[str, Any]) -> bool:
    return bool(
        float(health["q_range_mean"]) > 1e-3
        and float(health["dq_da_mean"]) > 1e-5
        and float(health["activation_active_fraction"]) > 0.05
    )


def reset_tanh_gate_passes(health: dict[str, Any]) -> bool:
    """Check the reset-state tanh slope gate.

    This gate is necessary but not sufficient: an evaluation can still show
    sustained action saturation even when the reset-state tanh slope is above
    the threshold.
    """

    return bool(float(health["tanh_grad_norm"]) > 1e-4)


def eval_saturation_gate_passes(rows: list[dict[str, Any]]) -> bool:
    """Check action saturation on actual evaluation rows."""

    if not rows:
        return False
    u_dot = float(np.mean([float(row.get("action_u_dot_saturation_ratio", 0.0)) for row in rows]))
    yaw_rate = float(
        np.mean([float(row.get("action_yaw_rate_saturation_ratio", 0.0)) for row in rows])
    )
    return bool(u_dot < 0.5 and yaw_rate < 0.5)


def saturation_gate_passes(
    health: dict[str, Any],
    eval_rows: list[dict[str, Any]] | None = None,
) -> bool:
    """Combined saturation gate.

    Without evaluation rows the gate is deliberately false. The reset-state
    tanh slope alone must not be reported as a saturation pass.
    """

    return bool(
        reset_tanh_gate_passes(health)
        and eval_rows is not None
        and eval_saturation_gate_passes(eval_rows)
    )


def write_critic_health(
    path: Path | str,
    health: dict[str, Any],
    eval_rows: list[dict[str, Any]] | None = None,
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(health)
    payload["critic_health_gate_passed"] = critic_health_passes(health)
    payload["reset_tanh_gate_passed"] = reset_tanh_gate_passes(health)
    if eval_rows is None:
        payload["eval_saturation_gate_passed"] = None
        payload["eval_u_dot_saturation_mean"] = None
        payload["eval_yaw_rate_saturation_mean"] = None
    else:
        payload["eval_saturation_gate_passed"] = eval_saturation_gate_passes(eval_rows)
        payload["eval_u_dot_saturation_mean"] = float(
            np.mean([float(row.get("action_u_dot_saturation_ratio", 0.0)) for row in eval_rows])
        )
        payload["eval_yaw_rate_saturation_mean"] = float(
            np.mean([float(row.get("action_yaw_rate_saturation_ratio", 0.0)) for row in eval_rows])
        )
    payload["saturation_gate_passed"] = saturation_gate_passes(health, eval_rows)
    destination.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


__all__ = [
    "action_grid",
    "critic_health_passes",
    "diagnose_critic",
    "eval_saturation_gate_passes",
    "reset_tanh_gate_passes",
    "saturation_gate_passes",
    "write_critic_health",
]
