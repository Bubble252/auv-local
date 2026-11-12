"""Small, dependency-light metrics helpers for batch PPO diagnostics."""

from __future__ import annotations

import math
import statistics
from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import Tensor

_COUNT_KEYS = {
    "/env/reset_count",
    "/env/nonfinite_observation_count",
    "/env/nonfinite_action_count",
    "/env/nonfinite_reward_count",
}

_EPISODE_KEYS = (
    "/env/reset_count",
    "/env/nonfinite_observation_count",
    "/env/nonfinite_action_count",
    "/env/nonfinite_reward_count",
    "/episode/success_rate",
    "/episode/timeout_rate",
    "/episode/out_of_bounds_rate",
    "/episode/invalid_rate",
    "/episode/return_mean",
    "/episode/length_mean",
    "/episode/three_d_rmse_mean",
    "/episode/horizontal_rmse_mean",
    "/episode/vertical_rmse_mean",
    "/episode/residual_abs_mean",
    "/episode/material_saturation_rate",
    "/episode/constraint_violations_mean",
    "/episode/base_at_limit_rate",
)


def summarize_latencies_ms(samples_ms: Sequence[float]) -> dict[str, float | int]:
    """Return stable p50/p95 latency statistics for one reporting window."""

    if not samples_ms:
        return {
            "count": 0,
            "mean_ms": 0.0,
            "p50_ms": 0.0,
            "p95_ms": 0.0,
            "max_ms": 0.0,
        }
    values = sorted(float(value) for value in samples_ms)
    p95_index = min(len(values) - 1, int(math.ceil(0.95 * len(values))) - 1)
    return {
        "count": len(values),
        "mean_ms": statistics.fmean(values),
        "p50_ms": statistics.median(values),
        "p95_ms": values[p95_index],
        "max_ms": values[-1],
    }


def _scalar_tensor(value: Any, *, device: torch.device) -> Tensor:
    if torch.is_tensor(value):
        return value.detach().reshape(-1)[0].to(device=device, dtype=torch.float64)
    return torch.tensor(float(value), device=device, dtype=torch.float64)


def aggregate_episode_logs(entries: Sequence[Mapping[str, Any]]) -> dict[str, float | None]:
    """Aggregate rsl_rl-style per-step ``infos['log']`` dictionaries.

    Rates and RMSE values are weighted by the number of completed episodes in
    each step. Counts are summed. The aggregation is done once per iteration,
    after all rollout steps have been collected.
    """

    if not entries:
        return {}
    first_tensor = next(
        (value for entry in entries for value in entry.values() if torch.is_tensor(value)),
        None,
    )
    device = first_tensor.device if isinstance(first_tensor, Tensor) else torch.device("cpu")
    reset_values = [
        _scalar_tensor(entry["/env/reset_count"], device=device)
        for entry in entries
        if "/env/reset_count" in entry
    ]
    weights = (
        torch.stack(reset_values)
        if reset_values
        else torch.zeros(len(entries), device=device, dtype=torch.float64)
    )
    result: dict[str, float | None] = {}
    for key in _EPISODE_KEYS:
        values = [_scalar_tensor(entry[key], device=device) for entry in entries if key in entry]
        if not values:
            continue
        stacked = torch.stack(values)
        if key in _COUNT_KEYS:
            result[key] = float(stacked.sum().item())
            continue
        if len(values) != len(weights):
            result[key] = float(stacked.mean().item())
            continue
        denominator = weights.sum()
        if float(denominator.item()) <= 0.0:
            result[key] = None
            continue
        result[key] = float((stacked * weights).sum().div(denominator).item())
    return result


def post_update_policy_metrics(
    policy: Any,
    observations: Tensor,
    actions: Tensor,
    old_log_probs: Tensor,
    *,
    clip_param: float,
    max_samples: int,
) -> dict[str, float | int | str]:
    """Estimate PPO KL/clip statistics on a deterministic rollout subset.

    rsl_rl 2.3.3 does not expose per-mini-batch KL or clip fraction. This
    post-update estimate compares the updated policy with the log-probabilities
    stored during rollout. It is labelled as an estimate in the manifest.
    """

    if max_samples <= 0:
        raise ValueError("max_samples must be positive")
    flat_observations = observations.flatten(0, 1)
    flat_actions = actions.flatten(0, 1)
    flat_old_log_probs = old_log_probs.flatten(0, 1).reshape(-1)
    sample_count = min(max_samples, int(flat_observations.shape[0]))
    indices = (
        torch.linspace(
            0,
            flat_observations.shape[0] - 1,
            steps=sample_count,
            device=flat_observations.device,
            dtype=torch.float64,
        )
        .round()
        .to(dtype=torch.long)
    )
    sampled_observations = flat_observations.index_select(0, indices)
    sampled_actions = flat_actions.index_select(0, indices)
    sampled_old_log_probs = flat_old_log_probs.index_select(0, indices)
    with torch.inference_mode():
        policy.act(sampled_observations)
        new_log_probs = policy.get_actions_log_prob(sampled_actions).reshape(-1)
        log_ratio = new_log_probs - sampled_old_log_probs
        ratio = torch.exp(log_ratio)
        approx_kl = torch.mean((ratio - 1.0) - log_ratio)
        clipped = (ratio < 1.0 - clip_param) | (ratio > 1.0 + clip_param)
    return {
        "method": "post_update_vs_rollout_old_log_prob",
        "sample_count": sample_count,
        "approx_kl": float(approx_kl.item()),
        "clip_fraction": float(clipped.to(dtype=torch.float32).mean().item()),
        "ratio_mean": float(ratio.mean().item()),
        "ratio_max": float(ratio.max().item()),
    }


__all__ = [
    "aggregate_episode_logs",
    "post_update_policy_metrics",
    "summarize_latencies_ms",
]
