from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

SCRIPT = Path("scripts/analyze_pc_goal30_traces.py")
SPEC = importlib.util.spec_from_file_location("analyze_pc_goal30_traces", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _record(
    seed: int,
    *,
    success_protocol: bool,
    success_eta: bool,
    outcome: str,
    horizontal: float,
) -> dict[str, object]:
    return {
        "seed": seed,
        "success_protocol": success_protocol,
        "success_eta": success_eta,
        "terminal_eta_capture": success_eta,
        "protocol_and_terminal_eta_capture": success_protocol and success_eta,
        "pnav_only_success": success_protocol and not success_eta,
        "outcome": outcome,
        "eta_distance_m": horizontal,
        "min_eta_distance_m": horizontal,
        "horizontal_normal_eta_m": horizontal,
        "horizontal_normal_pnav_m": horizontal,
        "longitudinal_eta_m": 0.0,
        "vertical_eta_m": 0.0,
        "nav_error_m": 0.1,
        "terminal_yaw_error_abs_mean_rad": 0.2,
        "terminal_speed_ratio": 1.0,
        "terminal_material_saturation_rate": 0.1,
        "terminal_rudder_abs_mean_rad": 0.1,
        "terminal_stern_abs_mean_rad": 0.1,
    }


def test_paired_reports_eta_capture_and_horizontal_gain() -> None:
    baseline = [
        _record(20, success_protocol=True, success_eta=True, outcome="success", horizontal=0.5),
        _record(21, success_protocol=False, success_eta=False, outcome="plane_miss", horizontal=2.5),
    ]
    candidate = [
        _record(20, success_protocol=True, success_eta=True, outcome="success", horizontal=0.4),
        _record(21, success_protocol=True, success_eta=True, outcome="success", horizontal=1.0),
    ]

    paired = MODULE._paired(baseline, candidate)

    assert paired["rate_delta"]["success_eta"] == 0.5
    assert paired["rate_delta"]["success_protocol"] == 0.5
    assert paired["outcome_delta"]["plane_miss"] == -0.5
    assert paired["numeric_delta"]["horizontal_normal_eta_m"]["median"] == -0.8


def _raw_trace(
    *,
    seeds: list[int],
    actuator_value: float,
    eta_value: float,
) -> dict[str, object]:
    steps_per_seed = 4
    episode_seed = np.repeat(np.asarray(seeds, dtype=np.int64), steps_per_seed)
    step = np.tile(np.arange(steps_per_seed, dtype=np.int64), len(seeds))
    rows = episode_seed.size
    return {
        "episode_seed": episode_seed,
        "step": step,
        "time_s": step.astype(np.float64) * 0.1,
        "eta": np.tile(np.asarray([0.0, eta_value, 0.0]), (rows, 1)),
        "nu": np.zeros((rows, 6)),
        "actual_actuator": np.tile(
            np.asarray([actuator_value, 0.2, 0.3]), (rows, 1)
        ),
        "command": np.tile(np.asarray([0.5, 0.0, 1.2]), (rows, 1)),
    }


def test_chain_audit_flags_identical_actuator_and_eta() -> None:
    baseline = {"trace": _raw_trace(seeds=[41], actuator_value=0.1, eta_value=1.0)}
    candidate = {"trace": _raw_trace(seeds=[41], actuator_value=0.1, eta_value=1.0)}

    audit = MODULE._chain_audit(baseline, candidate)

    assert audit["chain_valid"] is False
    assert audit["all_seeds_actuator_identical"] is True
    assert audit["all_seeds_eta_identical"] is True
    assert audit["max_abs_actuator_diff"] == 0.0


def test_chain_audit_accepts_executed_guidance_change() -> None:
    baseline = {"trace": _raw_trace(seeds=[41], actuator_value=0.1, eta_value=1.0)}
    candidate = {"trace": _raw_trace(seeds=[41], actuator_value=0.2, eta_value=1.2)}

    audit = MODULE._chain_audit(baseline, candidate)

    assert audit["chain_valid"] is True
    assert audit["seeds_with_actuator_change"] == 1
    assert audit["per_seed"][0]["first_actuator_change_step"] == 0


def _write_case(
    directory: Path,
    *,
    seeds: list[int],
    episodes: list[dict[str, object]],
    actuator_value: float,
    eta_value: float,
) -> None:
    directory.mkdir(parents=True)
    (directory / "summary.json").write_text(
        json.dumps(
            {
                "schema_version": MODULE.SUMMARY_SCHEMA,
                "endpoint_stop_enabled": True,
                "seeds": seeds,
                "episodes": episodes,
            }
        ),
        encoding="utf-8",
    )
    raw = _raw_trace(seeds=seeds, actuator_value=actuator_value, eta_value=eta_value)
    raw.update(
        {
            "p_nav": raw["eta"],
            "goal_ned_m": np.zeros_like(raw["eta"]),
            "goal_tangent_ned": np.tile(
                np.asarray([1.0, 0.0, 0.0]), (raw["eta"].shape[0], 1)
            ),
            "terminal_kind": np.repeat(
                np.asarray(
                    [1 if episode["success"] else 9 for episode in episodes],
                    dtype=np.int64,
                ),
                4,
            ),
            "guidance_terminal_speed_ratio": np.ones(raw["eta"].shape[0]),
            "heading_error_rad": np.full(raw["eta"].shape[0], 0.2),
            "pitch_error_rad": np.full(raw["eta"].shape[0], 0.1),
            "material_saturated": np.zeros(raw["eta"].shape[0]),
        }
    )
    np.savez(directory / "trace.npz", **raw)


def test_analyze_reads_two_scenarios_and_pairs(tmp_path: Path) -> None:
    seeds = [41]
    baseline_episode = {
        "seed": 41,
        "success": False,
        "goal_capture_eta": False,
        "goal_plane_miss": True,
        "timeout": False,
        "safety": False,
    }
    candidate_episode = {
        "seed": 41,
        "success": True,
        "goal_capture_eta": True,
        "goal_plane_miss": False,
        "timeout": False,
        "safety": False,
    }
    inputs: dict[tuple[str, str], Path] = {}
    for scenario in MODULE.SCENARIOS:
        baseline = tmp_path / scenario / "baseline"
        candidate = tmp_path / scenario / "candidate"
        _write_case(
            baseline,
            seeds=seeds,
            episodes=[baseline_episode],
            actuator_value=0.1,
            eta_value=2.5,
        )
        _write_case(
            candidate,
            seeds=seeds,
            episodes=[candidate_episode],
            actuator_value=0.2,
            eta_value=1.0,
        )
        inputs[(scenario, "baseline")] = baseline
        inputs[(scenario, "candidate")] = candidate

    result = MODULE.analyze(inputs)

    for scenario in MODULE.SCENARIOS:
        block = result["scenarios"][scenario]
        assert block["chain_audit"]["chain_valid"] is True
        assert block["paired"]["rate_delta"]["success_eta"] == 1.0
        assert block["paired"]["rate_delta"]["terminal_eta_capture"] == 1.0
        assert block["paired"]["numeric_delta"]["horizontal_normal_eta_m"]["median"] == -1.5
