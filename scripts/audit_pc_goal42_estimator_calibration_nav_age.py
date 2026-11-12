#!/usr/bin/env python3
"""Audit terminal nav-age association and residual toward-goal bias."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA_VERSION = "auv_pc_goal42_terminal_age_bias_diagnostic.v1"
SCENARIOS = ("default", "strict125")
MODES = ("baseline", "candidate")
EXPECTED_SEEDS = tuple(range(20, 60))
EXPECTED_CHECKPOINT_SHA256 = (
    "0851496b15ffebbc0a19bafa5bf40c0493c0d9a3f485267b75fde5f811dde5d9"
)
MIN_DISCORDANT_SEEDS = 5
MIN_NON_DISCORDANT_SEEDS = 10
AGE_MEDIAN_DIFF_MIN_S = 0.10
RESIDUAL_TOWARD_GOAL_FRACTION_MIN = 0.80
BOOTSTRAP_SAMPLES = 5000
BOOTSTRAP_SEED = 20261109
TOWARD_GOAL_EPS_M = 1e-6
EPS = 1e-12

CSV_FIELDS = (
    "scenario",
    "mode",
    "seed",
    "success",
    "goal_capture_eta",
    "discordant",
    "nav_age_s",
    "usbl_age_s",
    "dvl_bottom_age_s",
    "raw_projection_m",
    "delay_projection_m",
    "residual_projection_m",
    "raw_toward_goal",
    "residual_toward_goal",
    "current_estimate_age_s",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--formal-root",
        type=Path,
        default=Path(
            "artifacts/lowlevel/goal42_estimator_calibration_nav_age_audit_20261109/formal"
        ),
    )
    parser.add_argument(
        "--reference-root",
        type=Path,
        default=Path(
            "artifacts/lowlevel/goal38_path_provenance_repair_20261109/formal"
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _finite_3vec(value: Any, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (3,):
        raise ValueError(f"{name} must be a 3-vector")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite")
    return array


def _interpolate_position(
    time_s: Any, eta_xyz: Any, query_time_s: float
) -> np.ndarray:
    times = np.asarray(time_s, dtype=np.float64)
    positions = np.asarray(eta_xyz, dtype=np.float64)
    if times.ndim != 1 or positions.shape != (times.size, 3):
        raise ValueError("trajectory time and eta shapes are inconsistent")
    if times.size == 0 or not np.all(np.isfinite(times)) or not np.all(np.isfinite(positions)):
        raise ValueError("trajectory must be non-empty and finite")
    if np.any(np.diff(times) <= 0.0):
        raise ValueError("trajectory time must be strictly increasing")
    query = float(query_time_s)
    if not math.isfinite(query):
        raise ValueError("query time must be finite")
    if query < float(times[0]) - EPS or query > float(times[-1]) + EPS:
        raise ValueError("query time is outside the trajectory")
    query = min(max(query, float(times[0])), float(times[-1]))
    return np.asarray(
        [np.interp(query, times, positions[:, index]) for index in range(3)],
        dtype=np.float64,
    )


def _terminal_age_metrics(
    *,
    p_nav: Any,
    eta: Any,
    goal: Any,
    nav_age_s: float,
    usbl_age_s: float,
    dvl_bottom_age_s: float,
    current_estimate_age_s: float,
    time_s: Any,
    eta_trajectory_xyz: Any,
    terminal_time_s: float,
) -> dict[str, Any]:
    p_nav_array = _finite_3vec(p_nav, "p_nav")
    eta_array = _finite_3vec(eta, "eta")
    goal_array = _finite_3vec(goal, "goal")
    ages = {
        "nav_age_s": float(nav_age_s),
        "usbl_age_s": float(usbl_age_s),
        "dvl_bottom_age_s": float(dvl_bottom_age_s),
        "current_estimate_age_s": float(current_estimate_age_s),
    }
    if any(not math.isfinite(value) or value < 0.0 for value in ages.values()):
        raise ValueError("terminal sensor ages must be finite and non-negative")
    terminal_time = float(terminal_time_s)
    if ages["nav_age_s"] > terminal_time + EPS:
        raise ValueError("terminal nav age exceeds terminal time")

    eta_lag = _interpolate_position(
        time_s,
        eta_trajectory_xyz,
        terminal_time - ages["nav_age_s"],
    )
    to_goal = goal_array - eta_array
    denominator = float(np.linalg.norm(to_goal))
    if denominator <= EPS:
        raise ValueError("terminal goal direction is undefined")
    goal_direction = to_goal / denominator

    raw_error_vec = p_nav_array - eta_array
    delay_vec = eta_lag - eta_array
    residual_error_vec = p_nav_array - eta_lag
    raw_projection_m = float(np.dot(raw_error_vec, goal_direction))
    delay_projection_m = float(np.dot(delay_vec, goal_direction))
    residual_projection_m = float(np.dot(residual_error_vec, goal_direction))
    return {
        **ages,
        "raw_projection_m": raw_projection_m,
        "delay_projection_m": delay_projection_m,
        "residual_projection_m": residual_projection_m,
        "raw_toward_goal": raw_projection_m > TOWARD_GOAL_EPS_M,
        "residual_toward_goal": residual_projection_m > TOWARD_GOAL_EPS_M,
    }


def _seed_row_indices(trace: Any) -> dict[int, np.ndarray]:
    required = (
        "episode_seed",
        "step",
        "time_s",
        "eta",
        "p_nav",
        "goal_ned_m",
        "nav_age_s",
        "usbl_age_s",
        "dvl_bottom_age_s",
        "current_estimate_age_s",
    )
    missing = [name for name in required if name not in trace.files]
    if missing:
        raise ValueError(f"trace is missing arrays: {missing}")
    seeds = np.asarray(trace["episode_seed"], dtype=np.int64)
    steps = np.asarray(trace["step"], dtype=np.int64)
    time_s = np.asarray(trace["time_s"], dtype=np.float64)
    if seeds.shape != steps.shape or seeds.shape != time_s.shape:
        raise ValueError("trace row arrays have inconsistent shapes")
    if seeds.size == 0 or not np.all(np.isfinite(time_s)):
        raise ValueError("trace is empty or has non-finite time")
    result: dict[int, np.ndarray] = {}
    for seed in EXPECTED_SEEDS:
        indices = np.flatnonzero(seeds == seed)
        if indices.size == 0:
            raise ValueError(f"trace has no rows for seed {seed}")
        if np.any(np.diff(steps[indices]) <= 0) or np.any(np.diff(time_s[indices]) <= 0):
            raise ValueError(f"trace for seed {seed} is not strictly increasing")
        result[seed] = indices
    return result


def _load_summary(path: Path) -> dict[str, Any]:
    summary = json.loads(path.read_text(encoding="utf-8"))
    seeds = sorted(int(seed) for seed in summary["seeds"])
    if seeds != list(EXPECTED_SEEDS):
        raise ValueError(f"{path}: seeds mismatch")
    if summary.get("checkpoint_sha256") != EXPECTED_CHECKPOINT_SHA256:
        raise ValueError(f"{path}: checkpoint SHA256 mismatch")
    episodes = {int(item["seed"]): item for item in summary["episodes"]}
    if sorted(episodes) != list(EXPECTED_SEEDS):
        raise ValueError(f"{path}: episode seeds mismatch")
    for item in episodes.values():
        for name in ("success", "goal_capture_eta", "goal_plane_miss"):
            if not isinstance(item.get(name), bool):
                raise ValueError(f"{path}: episode has invalid {name}")
    return {"summary": summary, "episodes": episodes}


def _record_for_seed(
    seed: int,
    rows: np.ndarray,
    trace: Any,
    episode: dict[str, Any],
    scenario: str,
    mode: str,
) -> dict[str, Any]:
    last = int(rows[-1])
    metrics = _terminal_age_metrics(
        p_nav=trace["p_nav"][last],
        eta=trace["eta"][last, :3],
        goal=trace["goal_ned_m"][last],
        nav_age_s=float(trace["nav_age_s"][last]),
        usbl_age_s=float(trace["usbl_age_s"][last]),
        dvl_bottom_age_s=float(trace["dvl_bottom_age_s"][last]),
        current_estimate_age_s=float(trace["current_estimate_age_s"][last]),
        time_s=trace["time_s"][rows],
        eta_trajectory_xyz=trace["eta"][rows, :3],
        terminal_time_s=float(trace["time_s"][last]),
    )
    success = bool(episode["success"])
    eta_capture = bool(episode["goal_capture_eta"])
    return {
        "scenario": scenario,
        "mode": mode,
        "seed": seed,
        "success": success,
        "goal_capture_eta": eta_capture,
        "discordant": success and not eta_capture,
        **metrics,
    }


def _median(values: list[float]) -> float:
    return float(np.median(np.asarray(values, dtype=np.float64)))


def _bootstrap_median_difference(
    discordant: list[float], non_discordant: list[float]
) -> tuple[float, float]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    discordant_array = np.asarray(discordant, dtype=np.float64)
    non_discordant_array = np.asarray(non_discordant, dtype=np.float64)
    differences = np.empty(BOOTSTRAP_SAMPLES, dtype=np.float64)
    for index in range(BOOTSTRAP_SAMPLES):
        discordant_sample = rng.choice(
            discordant_array, size=discordant_array.size, replace=True
        )
        non_discordant_sample = rng.choice(
            non_discordant_array, size=non_discordant_array.size, replace=True
        )
        differences[index] = np.median(discordant_sample) - np.median(
            non_discordant_sample
        )
    lower, upper = np.percentile(differences, [2.5, 97.5])
    return float(lower), float(upper)


def _aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    discordant = [row for row in records if row["discordant"]]
    non_discordant = [row for row in records if not row["discordant"]]
    if len(discordant) < MIN_DISCORDANT_SEEDS:
        raise ValueError(f"only {len(discordant)} discordant seeds")
    if len(non_discordant) < MIN_NON_DISCORDANT_SEEDS:
        raise ValueError(f"only {len(non_discordant)} non-discordant seeds")

    discordant_ages = [float(row["nav_age_s"]) for row in discordant]
    non_discordant_ages = [float(row["nav_age_s"]) for row in non_discordant]
    discordant_age_median = _median(discordant_ages)
    non_discordant_age_median = _median(non_discordant_ages)
    median_age_difference = discordant_age_median - non_discordant_age_median
    ci_low, ci_high = _bootstrap_median_difference(
        discordant_ages, non_discordant_ages
    )
    residual_toward_goal_count = sum(
        bool(row["residual_toward_goal"]) for row in discordant
    )
    residual_toward_goal_fraction = residual_toward_goal_count / len(discordant)
    raw_toward_goal_count = sum(bool(row["raw_toward_goal"]) for row in discordant)
    return {
        "seed_count": len(records),
        "discordant_seed_count": len(discordant),
        "non_discordant_seed_count": len(non_discordant),
        "discordant_nav_age_median_s": discordant_age_median,
        "non_discordant_nav_age_median_s": non_discordant_age_median,
        "nav_age_median_difference_s": median_age_difference,
        "nav_age_median_difference_ci95_s": [ci_low, ci_high],
        "median_raw_projection_m": _median(
            [float(row["raw_projection_m"]) for row in discordant]
        ),
        "median_delay_projection_m": _median(
            [float(row["delay_projection_m"]) for row in discordant]
        ),
        "median_residual_projection_m": _median(
            [float(row["residual_projection_m"]) for row in discordant]
        ),
        "median_usbl_age_s": _median(
            [float(row["usbl_age_s"]) for row in discordant]
        ),
        "raw_toward_goal_count": raw_toward_goal_count,
        "raw_toward_goal_fraction": raw_toward_goal_count / len(discordant),
        "residual_toward_goal_count": residual_toward_goal_count,
        "residual_toward_goal_fraction": residual_toward_goal_fraction,
    }


def _mode_verdict(pass_count: int) -> str:
    if pass_count == len(SCENARIOS) * len(MODES):
        return "SUPPORTED"
    if pass_count >= 1:
        return "PARTIAL"
    return "FALSIFIED"


def _overall_decision(
    age_verdict: str, residual_verdict: str, invalid: bool
) -> dict[str, str]:
    if invalid:
        return {
            "decision": "INVALID_REPLAY",
            "recommendation": "修 instrumentation 后只重跑一次，不解释机制。",
        }
    if age_verdict == "SUPPORTED" and residual_verdict == "SUPPORTED":
        return {
            "decision": "AGE_ASSOCIATED_WITH_RESIDUAL_BIAS",
            "recommendation": "预注册只读 calibration/estimator 单因素诊断；不训练。",
        }
    if residual_verdict == "SUPPORTED":
        return {
            "decision": "RESIDUAL_CALIBRATION_BIAS_SUPPORTED",
            "recommendation": "预注册只读 estimator bias 审计；不改控制器。",
        }
    if age_verdict == "SUPPORTED":
        return {
            "decision": "NAV_AGE_ASSOCIATION_ONLY",
            "recommendation": "只补最便宜的 age 分组验证，不训练。",
        }
    if "PARTIAL" in (age_verdict, residual_verdict):
        return {
            "decision": "PARTIAL_AGE_BIAS",
            "recommendation": "只补数据/分组验证，不训练。",
        }
    return {
        "decision": "NO_SINGLE_SOURCE_SUPPORTED",
        "recommendation": "停止 nav-age/calibration 单因素解释，回到 estimator 结构问题。",
    }


def _load_block(
    formal_root: Path,
    reference_root: Path,
    scenario: str,
    mode: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    replay_dir = formal_root / scenario / mode
    reference_dir = reference_root / scenario / mode
    replay = _load_summary(replay_dir / "summary.json")
    reference = _load_summary(reference_dir / "summary.json")
    for seed in EXPECTED_SEEDS:
        for name in ("success", "goal_capture_eta", "goal_plane_miss"):
            if replay["episodes"][seed][name] != reference["episodes"][seed][name]:
                raise ValueError(f"reference summary mismatch for seed {seed}: {name}")

    with np.load(replay_dir / "trace.npz", allow_pickle=False) as trace:
        seed_rows = _seed_row_indices(trace)
        records = []
        for seed in EXPECTED_SEEDS:
            episode = dict(replay["episodes"][seed])
            records.append(
                _record_for_seed(
                    seed,
                    seed_rows[seed],
                    trace,
                    episode,
                    scenario,
                    mode,
                )
            )

    aggregate = _aggregate(records)
    age_pass = (
        aggregate["nav_age_median_difference_s"]
        >= AGE_MEDIAN_DIFF_MIN_S - EPS
        and aggregate["nav_age_median_difference_ci95_s"][0] > 0.0
    )
    residual_pass = (
        aggregate["residual_toward_goal_fraction"]
        >= RESIDUAL_TOWARD_GOAL_FRACTION_MIN - EPS
    )
    block = {
        "scenario": scenario,
        "mode": mode,
        "source": str(replay_dir),
        "reference": str(reference_dir),
        "records": records,
        "aggregate": aggregate,
        "h_nav_age_association": {
            "pass": age_pass,
            "median_difference_pass": aggregate["nav_age_median_difference_s"]
            >= AGE_MEDIAN_DIFF_MIN_S - EPS,
            "ci_lower_positive": aggregate["nav_age_median_difference_ci95_s"][0]
            > 0.0,
        },
        "h_residual_calibration_bias": {
            "pass": residual_pass,
            "residual_toward_goal_fraction": aggregate[
                "residual_toward_goal_fraction"
            ],
        },
    }
    return block, records


def analyze(formal_root: Path, reference_root: Path) -> dict[str, Any]:
    scenarios: dict[str, Any] = {}
    all_blocks: list[dict[str, Any]] = []
    all_records: list[dict[str, Any]] = []
    invalid_reasons: list[str] = []
    for scenario in SCENARIOS:
        scenario_blocks: dict[str, Any] = {}
        for mode in MODES:
            try:
                block, records = _load_block(
                    formal_root, reference_root, scenario, mode
                )
            except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
                invalid_reasons.append(f"{scenario}/{mode}: {exc}")
                continue
            scenario_blocks[mode] = block
            all_blocks.append(block)
            all_records.extend(records)
        scenarios[scenario] = scenario_blocks

    invalid = bool(invalid_reasons) or len(all_blocks) != len(SCENARIOS) * len(MODES)
    block_by_key = {(block["scenario"], block["mode"]): block for block in all_blocks}
    if invalid:
        age_blocks = residual_blocks = 0
    else:
        age_blocks = sum(
            block_by_key[(scenario, mode)]["h_nav_age_association"]["pass"]
            for scenario in SCENARIOS
            for mode in MODES
        )
        residual_blocks = sum(
            block_by_key[(scenario, mode)]["h_residual_calibration_bias"]["pass"]
            for scenario in SCENARIOS
            for mode in MODES
        )
    age_verdict = _mode_verdict(age_blocks)
    residual_verdict = _mode_verdict(residual_blocks)
    overall = _overall_decision(age_verdict, residual_verdict, invalid)
    return {
        "schema_version": SCHEMA_VERSION,
        "validation": {
            "formal_root": str(formal_root),
            "reference_root": str(reference_root),
            "expected_seeds": list(EXPECTED_SEEDS),
            "expected_checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
            "heldout_opened": False,
            "block_count": len(all_blocks),
            "invalid": invalid,
            "invalid_reasons": invalid_reasons,
            "thresholds": {
                "min_discordant_seeds": MIN_DISCORDANT_SEEDS,
                "min_non_discordant_seeds": MIN_NON_DISCORDANT_SEEDS,
                "nav_age_median_difference_min_s": AGE_MEDIAN_DIFF_MIN_S,
                "residual_toward_goal_fraction_min": (
                    RESIDUAL_TOWARD_GOAL_FRACTION_MIN
                ),
                "toward_goal_eps_m": TOWARD_GOAL_EPS_M,
                "bootstrap_samples": BOOTSTRAP_SAMPLES,
                "bootstrap_seed": BOOTSTRAP_SEED,
            },
        },
        "scenarios": scenarios,
        "overall": {
            **overall,
            "h_nav_age_association": age_verdict,
            "h_residual_calibration_bias": residual_verdict,
            "age_passing_blocks": age_blocks,
            "residual_passing_blocks": residual_blocks,
            "block_count": len(all_blocks),
        },
        "records": all_records,
    }


def _markdown(result: dict[str, Any]) -> str:
    lines = [
        "# Goal42 terminal estimator calibration 与 nav-age 审计",
        "",
        f"Schema: `{result['schema_version']}`<br>",
        f"总体裁决：**{result['overall']['decision']}**",
        "",
        "## Block results",
        "",
        "| scenario | mode | discordant | age median diff s | age CI95 | "
        "residual toward frac | H-AGE | H-RESIDUAL |",
        "|---|---|---:|---:|---|---:|---|---|",
    ]
    for scenario in SCENARIOS:
        for mode in MODES:
            block = result["scenarios"].get(scenario, {}).get(mode)
            if block is None:
                lines.append(
                    f"| {scenario} | {mode} | n/a | n/a | n/a | n/a | n/a | n/a |"
                )
                continue
            aggregate = block["aggregate"]
            ci_low, ci_high = aggregate["nav_age_median_difference_ci95_s"]
            lines.append(
                f"| {scenario} | {mode} | {aggregate['discordant_seed_count']} | "
                f"{aggregate['nav_age_median_difference_s']:.4f} | "
                f"[{ci_low:.4f}, {ci_high:.4f}] | "
                f"{aggregate['residual_toward_goal_fraction']:.4f} | "
                f"{'PASS' if block['h_nav_age_association']['pass'] else 'FAIL'} | "
                f"{'PASS' if block['h_residual_calibration_bias']['pass'] else 'FAIL'} |"
            )
    lines.extend(
        [
            "",
            "## Overall",
            "",
            f"- H-NAV-AGE-ASSOCIATION: `{result['overall']['h_nav_age_association']}`",
            "- H-RESIDUAL-CALIBRATION-BIAS: "
            f"`{result['overall']['h_residual_calibration_bias']}`",
            f"- decision: `{result['overall']['decision']}`",
            f"- recommendation: {result['overall']['recommendation']}",
            "",
            "## Validation",
            "",
            f"- invalid: `{result['validation']['invalid']}`",
            f"- blocks: `{result['validation']['block_count']}/4`",
            f"- held-out opened: `{result['validation']['heldout_opened']}`",
        ]
    )
    for reason in result["validation"]["invalid_reasons"]:
        lines.append(f"- invalid reason: {reason}")
    return "\n".join(lines) + "\n"


def write_outputs(result: dict[str, Any], output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / "terminal_age_bias_diagnostic.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "terminal_age_bias_diagnostic.md").write_text(
        _markdown(result), encoding="utf-8"
    )
    with (output / "terminal_age_bias_per_seed.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for record in result["records"]:
            writer.writerow({key: record[key] for key in CSV_FIELDS})


def main() -> int:
    args = parse_args()
    result = analyze(args.formal_root, args.reference_root)
    write_outputs(result, args.output)
    print(
        json.dumps(
            {
                "H_NAV_AGE_ASSOCIATION": result["overall"][
                    "h_nav_age_association"
                ],
                "H_RESIDUAL_CALIBRATION_BIAS": result["overall"][
                    "h_residual_calibration_bias"
                ],
                "decision": result["overall"]["decision"],
                "recommendation": result["overall"]["recommendation"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
