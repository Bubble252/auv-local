from __future__ import annotations

import importlib.util
from pathlib import Path

SCRIPT = Path("scripts/summarize_p1_bc_rl_run.py")
SPEC = importlib.util.spec_from_file_location("summarize_p1_bc_rl_run", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _row(*, scenario: str, success: bool, rl_step: int, env_transitions: int) -> dict[str, str]:
    return {
        "scenario_key": scenario,
        "success": "True" if success else "False",
        "collision": "False",
        "out_of_bounds": "False" if success else "True",
        "timeout": "False",
        "failure_reason": "" if success else "out_of_bounds",
        "rl_step": str(rl_step),
        "env_transitions": str(env_transitions),
        "min_clearance_m": "1.0" if success else "-0.1",
        "action_u_dot_saturation_ratio": "0.0",
        "action_yaw_rate_saturation_ratio": "0.0",
    }


def test_episode_window_uses_completion_order_not_episode_length() -> None:
    rows = [
        _row(scenario="s1", success=True, rl_step=2000, env_transitions=10_000 + index * 1000)
        for index in range(8)
    ]
    rows.extend(
        [
            _row(scenario="s1", success=False, rl_step=500, env_transitions=90_000),
            _row(scenario="s2", success=False, rl_step=500, env_transitions=100_000),
        ]
    )

    window = MODULE._episode_window(rows)

    assert window["episodes"] == 2
    assert window["first_env_transitions"] == 90_000
    assert window["last_env_transitions"] == 100_000
    assert window["by_scenario"]["s1"]["episodes"] == 1
    assert window["by_scenario"]["s1"]["successes"] == 0
    assert window["by_scenario"]["s2"]["episodes"] == 1
    assert window["by_scenario"]["s2"]["successes"] == 0


def test_episode_window_empty_input() -> None:
    window = MODULE._episode_window([])
    assert window["episodes"] == 0
    assert window["first_env_transitions"] is None
    assert window["last_env_transitions"] is None
