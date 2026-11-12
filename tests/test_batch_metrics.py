from __future__ import annotations

import math

import pytest
import torch

from auv_lowlevel.batch.metrics import (
    aggregate_episode_logs,
    post_update_policy_metrics,
    summarize_latencies_ms,
)


class _FixedNormalPolicy:
    def __init__(self, mean: float) -> None:
        self._mean = mean

    def act(self, observations: torch.Tensor) -> torch.Tensor:
        mean = torch.full((observations.shape[0], 3), self._mean)
        self.distribution = torch.distributions.Normal(mean, torch.ones_like(mean))
        return mean

    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        return self.distribution.log_prob(actions).sum(dim=-1)


def test_latency_summary_is_stable_and_uses_expected_percentiles() -> None:
    summary = summarize_latencies_ms([5.0, 1.0, 3.0, 2.0, 4.0])

    assert summary == {
        "count": 5,
        "mean_ms": 3.0,
        "p50_ms": 3.0,
        "p95_ms": 5.0,
        "max_ms": 5.0,
    }
    assert summarize_latencies_ms(())["count"] == 0


def test_episode_log_aggregation_weights_rates_by_reset_count() -> None:
    entries = [
        {
            "/env/reset_count": torch.tensor(2.0),
            "/env/nonfinite_observation_count": torch.tensor(0.0),
            "/episode/success_rate": torch.tensor(1.0),
        },
        {
            "/env/reset_count": torch.tensor(1.0),
            "/env/nonfinite_observation_count": torch.tensor(1.0),
            "/episode/success_rate": torch.tensor(0.0),
        },
        {
            "/env/reset_count": torch.tensor(0.0),
            "/env/nonfinite_observation_count": torch.tensor(0.0),
            "/episode/success_rate": torch.tensor(0.0),
        },
    ]

    summary = aggregate_episode_logs(entries)

    assert summary["/env/reset_count"] == pytest.approx(3.0)
    assert summary["/env/nonfinite_observation_count"] == pytest.approx(1.0)
    assert summary["/episode/success_rate"] == pytest.approx(2.0 / 3.0)


def test_post_update_policy_metrics_detects_unchanged_and_changed_policies() -> None:
    observations = torch.zeros((8, 4, 26), dtype=torch.float32)
    actions = torch.zeros((8, 4, 3), dtype=torch.float32)
    unchanged = _FixedNormalPolicy(mean=0.0)
    unchanged.act(observations.flatten(0, 1))
    old_log_probs = unchanged.get_actions_log_prob(actions.flatten(0, 1)).reshape(8, 4, 1)

    metrics = post_update_policy_metrics(
        unchanged,
        observations,
        actions,
        old_log_probs,
        clip_param=0.2,
        max_samples=64,
    )

    assert metrics["sample_count"] == 32
    assert metrics["approx_kl"] == pytest.approx(0.0)
    assert metrics["clip_fraction"] == pytest.approx(0.0)
    assert math.isfinite(float(metrics["ratio_mean"]))

    changed = _FixedNormalPolicy(mean=0.5)
    changed_metrics = post_update_policy_metrics(
        changed,
        observations,
        actions,
        old_log_probs,
        clip_param=0.2,
        max_samples=16,
    )
    assert float(changed_metrics["approx_kl"]) > 0.0
    assert changed_metrics["clip_fraction"] == pytest.approx(1.0)
