"""Pass/fail gates for BC+RL fine-tuning diagnostics.

The gates are computed from ``training_summary.json`` produced by
``scripts/summarize_p1_bc_rl_run.py``. They intentionally mirror the B2 gate in
``doc/goals/goal6_120m_50hz_bc.md`` plus the stricter diagnostic checks in
``doc/goals/goal7_bc_ddpg_stabilization.md``.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

GATE_NAMES = (
    "final_s1",
    "final_s2",
    "recent_s1",
    "recent_s2",
    "action_saturation",
    "actor_raw_abs_max",
    "tanh_grad_norm",
    "critic_finite",
    "critic_scale",
)


@dataclass(frozen=True)
class BcRlGateThresholds:
    min_final_s1_successes: int = 3
    min_final_s2_successes: int = 3
    min_recent_successes: int = 1
    max_saturation: float = 0.5
    max_actor_raw_abs: float = 1.0
    min_tanh_grad_norm: float = 1e-4
    max_abs_mean_q: float = 50.0

    def __post_init__(self) -> None:
        if self.min_final_s1_successes < 0 or self.min_final_s2_successes < 0:
            raise ValueError("minimum final successes must be non-negative")
        if self.min_recent_successes < 0:
            raise ValueError("min_recent_successes must be non-negative")
        if not 0.0 < self.max_saturation <= 1.0:
            raise ValueError("max_saturation must be in (0, 1]")
        if self.max_actor_raw_abs <= 0.0:
            raise ValueError("max_actor_raw_abs must be positive")
        if self.min_tanh_grad_norm < 0.0:
            raise ValueError("min_tanh_grad_norm must be non-negative")
        if self.max_abs_mean_q <= 0.0:
            raise ValueError("max_abs_mean_q must be positive")


def _finite(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def evaluate_bc_rl_gates(
    summary: Mapping[str, Any],
    thresholds: BcRlGateThresholds | None = None,
) -> dict[str, bool]:
    """Return one boolean per gate plus ``overall`` for a run summary."""

    limits = BcRlGateThresholds() if thresholds is None else thresholds
    final = summary.get("final_evaluation") or {}
    scenario_successes = final.get("scenario_successes") or {}
    recent = summary.get("recent_episodes") or {}
    recent_by_scenario = recent.get("by_scenario") or {}
    health = final.get("health") or {}
    critic = summary.get("critic") or {}

    def _recent_successes(scenario: str) -> int:
        entry = recent_by_scenario.get(scenario) or {}
        return int(entry.get("successes", 0))

    saturation = max(
        _finite(final.get("mean_u_dot_saturation")) or 0.0,
        _finite(final.get("mean_yaw_rate_saturation")) or 0.0,
    )
    actor_raw = _finite(health.get("actor_raw_abs_max"))
    tanh_grad = _finite(health.get("tanh_grad_norm"))
    head = critic.get("head") or {}
    tail = critic.get("tail") or {}
    head_q = _finite(head.get("mean_q"))
    tail_q = _finite(tail.get("mean_q"))

    gates = {
        "final_s1": int(scenario_successes.get("s1", 0)) >= limits.min_final_s1_successes,
        "final_s2": int(scenario_successes.get("s2", 0)) >= limits.min_final_s2_successes,
        "recent_s1": _recent_successes("s1") >= limits.min_recent_successes,
        "recent_s2": _recent_successes("s2") >= limits.min_recent_successes,
        "action_saturation": saturation < limits.max_saturation,
        "actor_raw_abs_max": actor_raw is not None and actor_raw < limits.max_actor_raw_abs,
        "tanh_grad_norm": tanh_grad is not None and tanh_grad > limits.min_tanh_grad_norm,
        "critic_finite": bool(critic.get("available", False))
        and int(critic.get("nonfinite_rows", 0)) == 0
        and head_q is not None
        and tail_q is not None,
        "critic_scale": head_q is not None
        and tail_q is not None
        and abs(head_q) <= limits.max_abs_mean_q
        and abs(tail_q) <= limits.max_abs_mean_q,
    }
    gates["overall"] = all(gates[name] for name in GATE_NAMES)
    return gates


__all__ = ["BcRlGateThresholds", "GATE_NAMES", "evaluate_bc_rl_gates"]
