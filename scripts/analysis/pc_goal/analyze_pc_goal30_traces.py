#!/usr/bin/env python3
"""Read-only paired analysis for Goal30 executed-chain terminal capture.

The script consumes ``evaluate_pc_ppo.py --trace-output`` artifacts for one
baseline and one candidate in each scenario.  It reports protocol success,
true-state ``eta`` capture, terminal horizontal-normal offset, terminal yaw,
speed, saturation, and an explicit control-chain audit.  It never loads a
policy, steps the environment, or trains.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np

SCHEMA_VERSION = "auv_pc_goal30_trace_analysis.v1"
SUMMARY_SCHEMA = "auv_pc_ppo_evaluation.v1"
SCENARIOS = ("default", "strict125")
MODES = ("baseline", "candidate")
TERMINAL_WINDOW_S = 2.0
BOOTSTRAP_SAMPLES = 10_000
BOOTSTRAP_SEED = 0
CHAIN_ATOL = 1e-9
GOAL_RADIUS_M = {"default": 2.0, "strict125": 1.25}


def _load_goal28_helpers() -> Any:
    path = Path(__file__).resolve().parent / "analyze_pc_goal28_traces.py"
    spec = importlib.util.spec_from_file_location("_goal28_trace_helpers", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import helpers from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_GOAL28 = _load_goal28_helpers()
_decompose = _GOAL28._decompose
_term_stats = _GOAL28._term_stats
_outcome = _GOAL28._outcome


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trace",
        action="append",
        required=True,
        metavar="SCENARIO:MODE=DIR",
        help="repeat for baseline/candidate in default and strict125",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _parse_inputs(values: Iterable[str]) -> dict[tuple[str, str], Path]:
    inputs: dict[tuple[str, str], Path] = {}
    for value in values:
        try:
            key, path_text = value.split("=", 1)
            scenario, mode = key.split(":", 1)
        except ValueError as exc:
            raise ValueError(
                f"invalid --trace {value!r}; expected SCENARIO:MODE=DIR"
            ) from exc
        if scenario not in SCENARIOS:
            raise ValueError(f"unsupported scenario {scenario!r}")
        if mode not in MODES:
            raise ValueError(f"unsupported mode {mode!r}")
        if (scenario, mode) in inputs:
            raise ValueError(f"duplicate trace for {scenario}:{mode}")
        inputs[(scenario, mode)] = Path(path_text)
    missing = [
        f"{scenario}:{mode}"
        for scenario in SCENARIOS
        for mode in MODES
        if (scenario, mode) not in inputs
    ]
    if missing:
        raise ValueError(f"missing traces: {', '.join(missing)}")
    return inputs


def _load_summary(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != SUMMARY_SCHEMA:
        raise ValueError(f"{path}: unexpected summary schema")
    if payload.get("endpoint_stop_enabled") is not True:
        raise ValueError(f"{path}: endpoint_stop is required")
    episodes = payload.get("episodes")
    if not isinstance(episodes, list) or not episodes:
        raise ValueError(f"{path}: expected non-empty episode records")
    seeds = [int(item["seed"]) for item in episodes]
    if seeds != list(payload.get("seeds", ())):
        raise ValueError(f"{path}: episode seeds do not match summary seeds")
    if len(set(seeds)) != len(seeds):
        raise ValueError(f"{path}: duplicate episode seeds")
    return payload


def _trace_records(
    scenario: str,
    mode: str,
    directory: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    trace_path = directory / "trace.npz"
    summary_path = directory / "summary.json"
    summary = _load_summary(summary_path)
    data = np.load(trace_path, allow_pickle=False)
    required = {
        "time_s",
        "episode_seed",
        "step",
        "eta",
        "p_nav",
        "goal_ned_m",
        "goal_tangent_ned",
        "terminal_kind",
        "guidance_terminal_speed_ratio",
        "heading_error_rad",
        "pitch_error_rad",
        "material_saturated",
        "actual_actuator",
        "command",
        "nu",
    }
    missing = sorted(required - set(data.files))
    if missing:
        raise ValueError(f"{trace_path}: missing trace fields {missing}")

    summary_by_seed = {int(item["seed"]): item for item in summary["episodes"]}
    seeds = data["episode_seed"].astype(np.int64)
    records: list[dict[str, Any]] = []
    for seed in sorted(summary_by_seed):
        indices = np.flatnonzero(seeds == seed)
        if indices.size == 0:
            raise ValueError(f"{trace_path}: no trace rows for seed {seed}")
        last = int(indices[-1])
        item = summary_by_seed[seed]
        terminal_kind = int(data["terminal_kind"][last])
        outcome = _outcome(item)
        expected_kind = {
            "success": 1,
            "timeout": 2,
            "plane_miss": 9,
        }.get(outcome)
        if expected_kind is not None and terminal_kind != expected_kind:
            raise ValueError(
                f"{trace_path}: seed {seed} terminal kind {terminal_kind} "
                f"does not match outcome {outcome}"
            )

        goal = np.asarray(data["goal_ned_m"][last], dtype=np.float64)
        tangent = np.asarray(data["goal_tangent_ned"][last], dtype=np.float64)
        p_nav = np.asarray(data["p_nav"][last], dtype=np.float64)
        eta = np.asarray(data["eta"][last, :3], dtype=np.float64)
        if not (
            np.all(np.isfinite(goal))
            and np.all(np.isfinite(tangent))
            and np.all(np.isfinite(p_nav))
            and np.all(np.isfinite(eta))
        ):
            raise ValueError(f"{trace_path}: seed {seed} has non-finite terminal state")
        longitudinal_pnav, horizontal_pnav, vertical_pnav = _decompose(
            p_nav - goal, tangent
        )
        longitudinal_eta, horizontal_eta, vertical_eta = _decompose(
            eta - goal, tangent
        )
        eta_distance = float(np.linalg.norm(eta - goal))
        terminal_eta_capture = eta_distance <= GOAL_RADIUS_M[scenario] + 1e-12
        success_protocol = bool(item["success"])
        eta_distances = np.linalg.norm(
            np.asarray(data["eta"][indices, :3], dtype=np.float64) - goal,
            axis=1,
        )

        last_time = float(data["time_s"][last])
        window = indices[data["time_s"][indices] >= last_time - TERMINAL_WINDOW_S]
        if window.size == 0:
            raise ValueError(f"{trace_path}: seed {seed} has an empty terminal window")
        yaw = np.abs(np.asarray(data["heading_error_rad"][window], dtype=np.float64))
        pitch = np.abs(np.asarray(data["pitch_error_rad"][window], dtype=np.float64))
        saturation = np.asarray(data["material_saturated"][window], dtype=np.float64)
        speed_ratio = np.asarray(
            data["guidance_terminal_speed_ratio"][window], dtype=np.float64
        )
        actuator = np.asarray(data["actual_actuator"][window], dtype=np.float64)
        if not (
            np.all(np.isfinite(yaw))
            and np.all(np.isfinite(pitch))
            and np.all(np.isfinite(saturation))
            and np.all(np.isfinite(speed_ratio))
            and np.all(np.isfinite(actuator))
        ):
            raise ValueError(f"{trace_path}: seed {seed} has non-finite terminal window")

        records.append(
            {
                "scenario": scenario,
                "mode": mode,
                "seed": seed,
                "outcome": outcome,
                "success_protocol": success_protocol,
                "success_eta": bool(item.get("goal_capture_eta", False)),
                "pnav_only_success": bool(item["success"])
                and not bool(item.get("goal_capture_eta", False)),
                "terminal_eta_capture": terminal_eta_capture,
                "protocol_and_terminal_eta_capture": (
                    success_protocol and terminal_eta_capture
                ),
                "terminal_kind": terminal_kind,
                "final_time_s": last_time,
                "pnav_distance_m": float(np.linalg.norm(p_nav - goal)),
                "eta_distance_m": eta_distance,
                "min_eta_distance_m": float(np.min(eta_distances)),
                "longitudinal_pnav_m": longitudinal_pnav,
                "horizontal_normal_pnav_m": horizontal_pnav,
                "vertical_pnav_m": vertical_pnav,
                "longitudinal_eta_m": longitudinal_eta,
                "horizontal_normal_eta_m": horizontal_eta,
                "vertical_eta_m": vertical_eta,
                "nav_error_m": float(np.linalg.norm(p_nav - eta)),
                "terminal_speed_ratio": float(np.mean(speed_ratio)),
                "terminal_yaw_error_abs_mean_rad": float(np.mean(yaw)),
                "terminal_pitch_error_abs_mean_rad": float(np.mean(pitch)),
                "terminal_material_saturation_rate": float(np.mean(saturation > 0.5)),
                "terminal_rudder_abs_mean_rad": float(np.mean(np.abs(actuator[:, 0]))),
                "terminal_stern_abs_mean_rad": float(np.mean(np.abs(actuator[:, 1]))),
            }
        )
    return summary, records, {
        "trace": data,
        "summary": summary,
        "directory": str(directory),
    }


def _aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    def rate(key: str) -> float:
        return float(np.mean([bool(row[key]) for row in records]))

    def stats(key: str) -> dict[str, Any]:
        return _term_stats(float(row[key]) for row in records)

    plane_miss = [row for row in records if row["outcome"] == "plane_miss"]
    return {
        "seed_count": len(records),
        "success_protocol_rate": rate("success_protocol"),
        "success_eta_rate": rate("success_eta"),
        "terminal_eta_capture_rate": rate("terminal_eta_capture"),
        "protocol_and_terminal_eta_capture_rate": rate(
            "protocol_and_terminal_eta_capture"
        ),
        "pnav_only_success_rate": rate("pnav_only_success"),
        "plane_miss_rate": float(np.mean([row["outcome"] == "plane_miss" for row in records])),
        "timeout_rate": float(np.mean([row["outcome"] == "timeout" for row in records])),
        "safety_rate": float(np.mean([row["outcome"] == "safety" for row in records])),
        "eta_distance_m": stats("eta_distance_m"),
        "min_eta_distance_m": stats("min_eta_distance_m"),
        "horizontal_normal_eta_m": stats("horizontal_normal_eta_m"),
        "horizontal_normal_pnav_m": stats("horizontal_normal_pnav_m"),
        "longitudinal_eta_m": stats("longitudinal_eta_m"),
        "vertical_eta_m": stats("vertical_eta_m"),
        "nav_error_m": stats("nav_error_m"),
        "terminal_yaw_error_abs_mean_rad": stats(
            "terminal_yaw_error_abs_mean_rad"
        ),
        "terminal_speed_ratio": stats("terminal_speed_ratio"),
        "terminal_material_saturation_rate": stats(
            "terminal_material_saturation_rate"
        ),
        "terminal_rudder_abs_mean_rad": stats("terminal_rudder_abs_mean_rad"),
        "terminal_stern_abs_mean_rad": stats("terminal_stern_abs_mean_rad"),
        "plane_miss_horizontal_normal_eta_m": _term_stats(
            row["horizontal_normal_eta_m"] for row in plane_miss
        ),
        "plane_miss_terminal_yaw_error_abs_mean_rad": _term_stats(
            row["terminal_yaw_error_abs_mean_rad"] for row in plane_miss
        ),
    }


def _bootstrap_ci(values: np.ndarray, *, samples: int = BOOTSTRAP_SAMPLES) -> list[float] | None:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size < 2:
        return None
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = rng.choice(values, size=(samples, values.size), replace=True)
    means = np.mean(draws, axis=1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def _paired(
    baseline: list[dict[str, Any]],
    candidate: list[dict[str, Any]],
) -> dict[str, Any]:
    by_seed_baseline = {row["seed"]: row for row in baseline}
    by_seed_candidate = {row["seed"]: row for row in candidate}
    if set(by_seed_baseline) != set(by_seed_candidate):
        raise ValueError("baseline and candidate seed sets differ")
    seeds = sorted(by_seed_baseline)
    rows = [
        (by_seed_baseline[seed], by_seed_candidate[seed]) for seed in seeds
    ]

    def rate_delta(key: str) -> float:
        return float(
            np.mean([bool(candidate_row[key]) for _, candidate_row in rows])
            - np.mean([bool(baseline_row[key]) for baseline_row, _ in rows])
        )

    numeric_keys = (
        "eta_distance_m",
        "min_eta_distance_m",
        "horizontal_normal_eta_m",
        "horizontal_normal_pnav_m",
        "longitudinal_eta_m",
        "vertical_eta_m",
        "nav_error_m",
        "terminal_yaw_error_abs_mean_rad",
        "terminal_speed_ratio",
        "terminal_material_saturation_rate",
        "terminal_rudder_abs_mean_rad",
        "terminal_stern_abs_mean_rad",
    )
    result: dict[str, Any] = {
        "seed_count": len(rows),
        "seeds": seeds,
        "rate_delta": {
            key: rate_delta(key)
            for key in (
                "success_protocol",
                "success_eta",
                "terminal_eta_capture",
                "protocol_and_terminal_eta_capture",
                "pnav_only_success",
            )
        },
        "outcome_delta": {
            "plane_miss": float(
                np.mean([candidate_row["outcome"] == "plane_miss" for _, candidate_row in rows])
                - np.mean([baseline_row["outcome"] == "plane_miss" for baseline_row, _ in rows])
            ),
            "timeout": float(
                np.mean([candidate_row["outcome"] == "timeout" for _, candidate_row in rows])
                - np.mean([baseline_row["outcome"] == "timeout" for baseline_row, _ in rows])
            ),
            "safety": float(
                np.mean([candidate_row["outcome"] == "safety" for _, candidate_row in rows])
                - np.mean([baseline_row["outcome"] == "safety" for baseline_row, _ in rows])
            ),
        },
        "numeric_delta": {},
    }
    for key in numeric_keys:
        delta = np.asarray(
            [
                float(candidate_row[key]) - float(baseline_row[key])
                for baseline_row, candidate_row in rows
            ],
            dtype=np.float64,
        )
        result["numeric_delta"][key] = {
            "mean": float(np.mean(delta)),
            "median": float(np.median(delta)),
            "p05": float(np.percentile(delta, 5)),
            "p95": float(np.percentile(delta, 95)),
            "bootstrap_95_ci": _bootstrap_ci(delta),
        }
    return result


def _chain_audit(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
) -> dict[str, Any]:
    base_trace = baseline["trace"]
    candidate_trace = candidate["trace"]
    base_seeds = base_trace["episode_seed"].astype(np.int64)
    candidate_seeds = candidate_trace["episode_seed"].astype(np.int64)
    common_seeds = sorted(set(base_seeds.tolist()) & set(candidate_seeds.tolist()))
    if not common_seeds:
        raise ValueError("no common seeds between baseline and candidate traces")
    per_seed: list[dict[str, Any]] = []
    for seed in common_seeds:
        base_idx = np.flatnonzero(base_seeds == seed)
        candidate_idx = np.flatnonzero(candidate_seeds == seed)
        base_steps = base_trace["step"][base_idx].astype(np.int64)
        candidate_steps = candidate_trace["step"][candidate_idx].astype(np.int64)
        base_by_step = {int(step): int(index) for step, index in zip(base_steps, base_idx, strict=True)}
        candidate_by_step = {
            int(step): int(index)
            for step, index in zip(candidate_steps, candidate_idx, strict=True)
        }
        common_steps = sorted(set(base_by_step) & set(candidate_by_step))
        if not common_steps:
            raise ValueError(f"seed {seed}: no common trace steps")
        base_rows = np.asarray([base_by_step[step] for step in common_steps], dtype=np.int64)
        candidate_rows = np.asarray(
            [candidate_by_step[step] for step in common_steps], dtype=np.int64
        )
        diffs: dict[str, float] = {}
        for key in ("eta", "nu", "actual_actuator", "command"):
            base_values = np.asarray(base_trace[key][base_rows], dtype=np.float64)
            candidate_values = np.asarray(
                candidate_trace[key][candidate_rows], dtype=np.float64
            )
            diffs[key] = float(np.max(np.abs(base_values - candidate_values)))
        actuator_diff = diffs["actual_actuator"]
        first_change = None
        if actuator_diff > CHAIN_ATOL:
            actuator_delta = np.max(
                np.abs(
                    np.asarray(base_trace["actual_actuator"][base_rows], dtype=np.float64)
                    - np.asarray(
                        candidate_trace["actual_actuator"][candidate_rows],
                        dtype=np.float64,
                    )
                ),
                axis=1,
            )
            first_change = int(common_steps[int(np.flatnonzero(actuator_delta > CHAIN_ATOL)[0])])
        per_seed.append(
            {
                "seed": int(seed),
                "common_steps": len(common_steps),
                "max_abs_eta_diff": diffs["eta"],
                "max_abs_nu_diff": diffs["nu"],
                "max_abs_actuator_diff": actuator_diff,
                "max_abs_command_diff": diffs["command"],
                "first_actuator_change_step": first_change,
            }
        )
    actuator_values = np.asarray(
        [row["max_abs_actuator_diff"] for row in per_seed], dtype=np.float64
    )
    eta_values = np.asarray([row["max_abs_eta_diff"] for row in per_seed], dtype=np.float64)
    return {
        "chain_valid": bool(
            np.any(actuator_values > CHAIN_ATOL) and np.any(eta_values > CHAIN_ATOL)
        ),
        "all_seeds_actuator_identical": bool(np.all(actuator_values <= CHAIN_ATOL)),
        "all_seeds_eta_identical": bool(np.all(eta_values <= CHAIN_ATOL)),
        "max_abs_actuator_diff": float(np.max(actuator_values)),
        "max_abs_eta_diff": float(np.max(eta_values)),
        "seeds_with_actuator_change": int(np.sum(actuator_values > CHAIN_ATOL)),
        "seed_count": len(per_seed),
        "per_seed": per_seed,
    }


def analyze(inputs: dict[tuple[str, str], Path]) -> dict[str, Any]:
    loaded: dict[tuple[str, str], dict[str, Any]] = {}
    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "terminal_window_s": TERMINAL_WINDOW_S,
        "chain_atol": CHAIN_ATOL,
        "scenarios": {},
    }
    for scenario in SCENARIOS:
        result["scenarios"][scenario] = {}
        for mode in MODES:
            directory = inputs[(scenario, mode)]
            summary, records, raw = _trace_records(scenario, mode, directory)
            loaded[(scenario, mode)] = raw
            result["scenarios"][scenario][mode] = {
                "source": str(directory),
                "summary_sha256": summary.get("summary_sha256"),
                "checkpoint_sha256": summary.get("checkpoint_sha256"),
                "aggregate": _aggregate(records),
                "records": records,
            }
        result["scenarios"][scenario]["paired"] = _paired(
            result["scenarios"][scenario]["baseline"]["records"],
            result["scenarios"][scenario]["candidate"]["records"],
        )
        result["scenarios"][scenario]["chain_audit"] = _chain_audit(
            loaded[(scenario, "baseline")],
            loaded[(scenario, "candidate")],
        )
    return result


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _markdown(result: dict[str, Any]) -> str:
    lines = [
        "# Goal30 executed-chain terminal analysis",
        "",
        "Read-only paired trace analysis; no training.",
        "",
        "| scenario | metric | baseline | candidate | delta | 95% CI |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for scenario in SCENARIOS:
        block = result["scenarios"][scenario]
        base = block["baseline"]["aggregate"]
        candidate = block["candidate"]["aggregate"]
        paired = block["paired"]
        for key in (
            "success_protocol_rate",
            "success_eta_rate",
            "terminal_eta_capture_rate",
            "protocol_and_terminal_eta_capture_rate",
            "plane_miss_rate",
            "timeout_rate",
            "safety_rate",
        ):
            delta_key = {
                "success_protocol_rate": ("rate_delta", "success_protocol"),
                "success_eta_rate": ("rate_delta", "success_eta"),
                "terminal_eta_capture_rate": ("rate_delta", "terminal_eta_capture"),
                "protocol_and_terminal_eta_capture_rate": (
                    "rate_delta",
                    "protocol_and_terminal_eta_capture",
                ),
                "plane_miss_rate": ("outcome_delta", "plane_miss"),
                "timeout_rate": ("outcome_delta", "timeout"),
                "safety_rate": ("outcome_delta", "safety"),
            }[key]
            lines.append(
                f"| {scenario} | {key} | {_fmt(base[key])} | {_fmt(candidate[key])} | "
                f"{_fmt(paired[delta_key[0]][delta_key[1]])} | n/a |"
            )
        for key in (
            "horizontal_normal_eta_m",
            "terminal_yaw_error_abs_mean_rad",
            "terminal_speed_ratio",
            "terminal_material_saturation_rate",
        ):
            delta = paired["numeric_delta"][key]
            ci = delta["bootstrap_95_ci"]
            ci_text = "n/a" if ci is None else f"[{ci[0]:.4f}, {ci[1]:.4f}]"
            lines.append(
                f"| {scenario} | {key} | {_fmt(base[key]['median'])} | "
                f"{_fmt(candidate[key]['median'])} | {_fmt(delta['median'])} | {ci_text} |"
            )
        audit = block["chain_audit"]
        lines.append(
            f"| {scenario} | chain actuator max diff | - | - | "
            f"{_fmt(audit['max_abs_actuator_diff'])} | n/a |"
        )
        lines.append(
            f"| {scenario} | chain eta max diff | - | - | "
            f"{_fmt(audit['max_abs_eta_diff'])} | n/a |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    inputs = _parse_inputs(args.trace)
    result = analyze(inputs)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "analysis.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.output / "analysis.md").write_text(_markdown(result), encoding="utf-8")
    for scenario in SCENARIOS:
        block = result["scenarios"][scenario]
        audit = block["chain_audit"]
        paired = block["paired"]
        print(
            f"{scenario}: chain_valid={audit['chain_valid']} "
            f"actuator_diff={_fmt(audit['max_abs_actuator_diff'])} "
            f"eta_diff={_fmt(audit['max_abs_eta_diff'])} "
            f"success_eta_delta={_fmt(paired['rate_delta']['success_eta'])} "
            f"horizontal_delta_med={_fmt(paired['numeric_delta']['horizontal_normal_eta_m']['median'])}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
