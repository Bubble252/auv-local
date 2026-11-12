from __future__ import annotations

import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).parents[1] / "scripts" / "summarize_isaaclab_smoke.py"
SPEC = importlib.util.spec_from_file_location("summarize_isaaclab_smoke", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _run(repeat: int, digest: str, status: str = "PASS") -> dict[str, object]:
    return {
        "_repeat": repeat,
        "_path": f"/tmp/num_envs_1_repeat_{repeat}.json",
        "status": status,
        "steps_completed": 200,
        "all_observation_finite": True,
        "num_envs_actual": 1,
        "observation_shape": [1, 4],
        "action_shape": [1, 1],
        "reset_s": 0.1,
        "first_step_s": 0.01,
        "time_to_first_step_s": 1.0,
        "reset_observation_sha256": digest,
    }


def test_smoke_passes_when_seeded_resets_match() -> None:
    payload = MODULE.build_smoke_result([_run(0, "abc"), _run(1, "abc"), _run(2, "abc")], 3, 200)

    assert payload["status"] == "PASS"
    assert payload["reset_observation_deterministic_across_repeats"] is True
    assert payload["repeats_passed"] == 3


def test_smoke_fails_when_seeded_resets_differ() -> None:
    payload = MODULE.build_smoke_result([_run(0, "abc"), _run(1, "def"), _run(2, "abc")], 3, 200)

    assert payload["status"] == "FAIL_SMOKE"
    assert payload["reset_observation_deterministic_across_repeats"] is False


def test_smoke_fails_with_insufficient_repeats() -> None:
    payload = MODULE.build_smoke_result([_run(0, "abc")], 2, 200)

    assert payload["status"] == "FAIL_SMOKE"
    assert payload["repeats_passed"] == 1
