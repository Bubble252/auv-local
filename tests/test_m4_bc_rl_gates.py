from __future__ import annotations

import pytest

from auv_rl.ddpg.gates import BcRlGateThresholds, evaluate_bc_rl_gates


def _summary(*, actor_raw_abs_max: float = 0.3, tanh_grad_norm: float = 0.9) -> dict[str, object]:
    return {
        "final_evaluation": {
            "scenario_successes": {"s1": 3, "s2": 3},
            "mean_u_dot_saturation": 0.02,
            "mean_yaw_rate_saturation": 0.04,
            "health": {
                "actor_raw_abs_max": actor_raw_abs_max,
                "tanh_grad_norm": tanh_grad_norm,
            },
        },
        "recent_episodes": {
            "by_scenario": {
                "s1": {"successes": 2, "episodes": 5},
                "s2": {"successes": 1, "episodes": 3},
            }
        },
        "critic": {
            "available": True,
            "nonfinite_rows": 0,
            "head": {"mean_q": 0.5},
            "tail": {"mean_q": 2.0},
        },
    }


def test_all_gates_pass_for_healthy_summary() -> None:
    gates = evaluate_bc_rl_gates(_summary())
    assert gates["overall"] is True
    assert all(value for name, value in gates.items() if name != "overall")


@pytest.mark.parametrize(
    ("name", "summary"),
    [
        ("actor_raw_abs_max", _summary(actor_raw_abs_max=23.9)),
        ("tanh_grad_norm", _summary(tanh_grad_norm=0.0)),
    ],
)
def test_saturation_gates_fail_on_broken_actor(name: str, summary: dict[str, object]) -> None:
    gates = evaluate_bc_rl_gates(summary)
    assert gates[name] is False
    assert gates["overall"] is False


def test_critic_scale_gate_rejects_exploded_q() -> None:
    summary = _summary()
    summary["critic"] = {
        "available": True,
        "nonfinite_rows": 0,
        "head": {"mean_q": 0.5},
        "tail": {"mean_q": 180.0},
    }
    gates = evaluate_bc_rl_gates(summary)
    assert gates["critic_scale"] is False
    assert gates["overall"] is False


def test_missing_recent_success_fails_gate() -> None:
    summary = _summary()
    summary["recent_episodes"] = {"by_scenario": {"s1": {"successes": 2, "episodes": 5}}}
    gates = evaluate_bc_rl_gates(summary)
    assert gates["recent_s2"] is False
    assert gates["overall"] is False


def test_thresholds_validate_ranges() -> None:
    with pytest.raises(ValueError, match="max_saturation"):
        BcRlGateThresholds(max_saturation=0.0)
