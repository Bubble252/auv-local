from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any


def _module() -> Any:
    spec = importlib.util.spec_from_file_location(
        "write_p1_ppo_final_evidence",
        Path("scripts/write_p1_ppo_final_evidence.py"),
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _gates() -> dict[str, Any]:
    return json.loads(Path("configs/ppo/isaaclab/final_gates.json").read_text(encoding="utf-8"))


def _scenario(successes: int, episodes: int, *, time_s: float = 10.0) -> dict[str, Any]:
    return {
        "episodes": episodes,
        "successes": successes,
        "mean_time_to_goal_s": time_s,
        "mean_min_clearance_m": 1.0,
    }


def _eval_summary(
    *,
    stage: str,
    stage_successes: int,
    stage_episodes: int,
    gate_successes: int | None = None,
) -> dict[str, Any]:
    scenarios = {
        "s1": _scenario(3, 3),
        "s2": _scenario(3, 3),
        stage: _scenario(stage_successes, stage_episodes),
    }
    instances = {
        "s1/s1": _scenario(3, 3),
        "s2/s2": _scenario(3, 3),
    }
    if stage == "c3":
        instances["c3/c3_gate_00"] = _scenario(2 if gate_successes is None else gate_successes, 3)
    return {
        "status": "PASS",
        "checkpoint_sha256": "abc",
        "gates": {
            "pass": True,
            "action_saturation_yaw": 0.1,
            "both_action_saturation": 0.02,
        },
        "scenarios": scenarios,
        "scenario_instances": instances,
    }


def _training() -> dict[str, Any]:
    return {
        "status": "PASS",
        "iterations_completed": 5000,
        "max_iterations": 5000,
        "eval_errors": [],
        "best_checkpoint_sha256": "abc",
        "best_checkpoint": "/tmp/best.pt",
        "seed": 0,
        "config_sha256": "config",
        "total_timesteps": 1000,
        "training_s": 10.0,
        "num_envs": 8,
        "num_steps_per_env": 32,
    }


def _ocean() -> dict[str, Any]:
    return {
        "main_success_gate": True,
        "latency_gate": {
            "gate": {"passed": True},
            "step_latency": {"median_ms": 10.0, "p95_ms": 12.0},
        },
        "scenario_summary": {"s1": {"successes": 3}, "s2": {"successes": 3}},
        "observation_source": "oceansim_simplified_pcl_120m",
        "policy_kind": "ppo",
        "jitter": False,
        "checkpoint_sha256": "abc",
    }


def _latency(median_ms: float, p95_ms: float, passed: bool) -> dict[str, Any]:
    return {
        "gate": {"passed": passed},
        "step_latency": {"median_ms": median_ms, "p95_ms": p95_ms},
    }


def test_final_evidence_passes_with_complete_consistent_inputs() -> None:
    module = _module()
    evaluation = module.build_evaluation(
        training=_training(),
        analytic=_eval_summary(stage="c1", stage_successes=3, stage_episodes=3),
        curriculum={
            "c1": _eval_summary(stage="c1", stage_successes=22, stage_episodes=24),
            "c2": _eval_summary(stage="c2", stage_successes=21, stage_episodes=24),
            "c3": _eval_summary(stage="c3", stage_successes=19, stage_episodes=21),
        },
        oceansim=_ocean(),
        latency_gate=None,
        gates=_gates(),
    )

    assert evaluation["status"] == "PASS"
    assert evaluation["checkpoint_consistent"] is True
    assert evaluation["curriculum"]["c3"]["gate_passed"] is True
    assert "OceanSim P1-A" in module.render_report(evaluation)


def test_final_evidence_fails_when_any_repeated_latency_run_fails() -> None:
    module = _module()
    evaluation = module.build_evaluation(
        training=_training(),
        analytic=_eval_summary(stage="c1", stage_successes=3, stage_episodes=3),
        curriculum={
            "c1": _eval_summary(stage="c1", stage_successes=22, stage_episodes=24),
            "c2": _eval_summary(stage="c2", stage_successes=21, stage_episodes=24),
            "c3": _eval_summary(stage="c3", stage_successes=19, stage_episodes=21),
        },
        oceansim=_ocean(),
        latency_gate=None,
        latency_gates=[
            _latency(18.0, 19.0, True),
            _latency(20.0, 27.0, False),
            _latency(19.0, 22.0, False),
        ],
        required_latency_runs=3,
        gates=_gates(),
    )

    assert evaluation["status"] == "FAIL"
    assert evaluation["oceansim"]["latency_stability"]["runs_provided"] == 3
    assert evaluation["oceansim"]["latency_stability"]["all_passed"] is False
    assert evaluation["oceansim"]["latency_ms"]["p95"] == 27.0
    assert "延迟门重复" in module.render_report(evaluation)


def test_final_evidence_is_partial_without_oceansim() -> None:
    module = _module()
    evaluation = module.build_evaluation(
        training=_training(),
        analytic=_eval_summary(stage="c1", stage_successes=3, stage_episodes=3),
        curriculum={
            "c1": _eval_summary(stage="c1", stage_successes=22, stage_episodes=24),
            "c2": _eval_summary(stage="c2", stage_successes=21, stage_episodes=24),
            "c3": _eval_summary(stage="c3", stage_successes=19, stage_episodes=21),
        },
        oceansim=None,
        latency_gate=None,
        gates=_gates(),
    )

    assert evaluation["status"] == "PARTIAL"
    assert evaluation["oceansim"]["status"] == "NOT_RUN"


def test_final_evidence_fails_on_checkpoint_mismatch() -> None:
    module = _module()
    analytic = _eval_summary(stage="c1", stage_successes=3, stage_episodes=3)
    analytic["checkpoint_sha256"] = "other"
    evaluation = module.build_evaluation(
        training=_training(),
        analytic=analytic,
        curriculum={
            "c1": _eval_summary(stage="c1", stage_successes=22, stage_episodes=24),
            "c2": _eval_summary(stage="c2", stage_successes=21, stage_episodes=24),
            "c3": _eval_summary(stage="c3", stage_successes=19, stage_episodes=21),
        },
        oceansim=_ocean(),
        latency_gate=None,
        gates=_gates(),
    )

    assert evaluation["checkpoint_consistent"] is False
    assert evaluation["status"] == "FAIL"


def test_final_evidence_fails_when_c3_gate_variant_regresses() -> None:
    module = _module()
    evaluation = module.build_evaluation(
        training=_training(),
        analytic=_eval_summary(stage="c1", stage_successes=3, stage_episodes=3),
        curriculum={
            "c1": _eval_summary(stage="c1", stage_successes=22, stage_episodes=24),
            "c2": _eval_summary(stage="c2", stage_successes=21, stage_episodes=24),
            "c3": _eval_summary(
                stage="c3",
                stage_successes=19,
                stage_episodes=21,
                gate_successes=1,
            ),
        },
        oceansim=_ocean(),
        latency_gate=None,
        gates=_gates(),
    )

    assert evaluation["curriculum"]["c3"]["gate_passed"] is False
    assert evaluation["status"] == "FAIL"
