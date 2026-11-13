#!/usr/bin/env python3
"""Audit Goal42 residual projections with exact p_nav queue-delay semantics."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA_VERSION = "auv_pc_goal43_nav_age_semantics.v1"
SCENARIOS = ("default", "strict125")
MODES = ("baseline", "candidate")
EXPECTED_SEEDS = tuple(range(20, 60))
EXPECTED_CHECKPOINT_SHA256 = (
    "0851496b15ffebbc0a19bafa5bf40c0493c0d9a3f485267b75fde5f811dde5d9"
)
DT_S = 0.02
MIN_DISCORDANT_SEEDS = 5
MIN_NON_DISCORDANT_SEEDS = 10
EXACT_TOWARD_GOAL_FRACTION_MIN = 0.80
EXCESS_AGE_MEDIAN_MIN_S = 0.10
PROJECTION_DELTA_MEDIAN_MIN_M = 0.10
TOWARD_GOAL_EPS_M = 1e-6
CSV_PROJECTION_TOLERANCE_M = 1e-4
EPS = 1e-12
AGE_GRID_TOLERANCE_S = 1e-9

INPUT_SHA256 = {
    "replay_gate.json": (
        "d95dadc48aab316733d286163923b9724d97932ced085412556edab1bc502901"
    ),
    "terminal_age_bias_diagnostic.json": (
        "9fcdd27dd2842ccda211f8709fa22807694982303adab78a56438a497b4601f0"
    ),
    "terminal_age_bias_per_seed.csv": (
        "4649a1d2a7aa77f103d5191c45ef2bb072d2efd7d162ec8069d7391fdce8d58a"
    ),
}
TRACE_SHA256 = {
    ("default", "baseline"): (
        "fe199b283f248fd94b5c416a4ab6639b3bf43f3f6824672f21966daf2ceb52d8"
    ),
    ("default", "candidate"): (
        "dc52bed06ad8fdddc52a58bb0fb73dbf9cb4ba049bbcce919344ae6e622e0548"
    ),
    ("strict125", "baseline"): (
        "aef6238f0996e5843242cb8ef3ac82e3bd966e369e63dcd91a82008644195387"
    ),
    ("strict125", "candidate"): (
        "b471fe1bd08c1540e10926dbb15a4736997f3787a1f9668831dc42e95b369a02"
    ),
}

CSV_FIELDS = (
    "scenario",
    "mode",
    "seed",
    "success",
    "goal_capture_eta",
    "discordant",
    "queue_age_s",
    "terminal_nav_age_s",
    "excess_age_s",
    "goal42_projection_m",
    "exact_projection_m",
    "projection_delta_m",
    "goal42_toward_goal",
    "exact_toward_goal",
    "exact_error_norm_m",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--formal-root",
        type=Path,
        default=Path(
            "artifacts/lowlevel/goal42_estimator_calibration_nav_age_audit_20261109/formal"
        ),
    )
    parser.add_argument(
        "--goal42-root",
        type=Path,
        default=Path(
            "artifacts/lowlevel/goal42_estimator_calibration_nav_age_audit_20261109"
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite_3vec(value: Any, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (3,):
        raise ValueError(f"{name} must be a 3-vector")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite")
    return array


def _queue_age_from_nav_age(nav_age_s: Any) -> float:
    ages = np.asarray(nav_age_s, dtype=np.float64)
    if ages.ndim != 1 or ages.size == 0:
        raise ValueError("nav_age_s must be a non-empty vector")
    if not np.all(np.isfinite(ages)):
        raise ValueError("nav_age_s must be finite")
    if np.any(ages < -EPS):
        raise ValueError("nav_age_s must be non-negative")
    queue_age = float(np.min(ages))
    steps = queue_age / DT_S
    if abs(steps - round(steps)) * DT_S > AGE_GRID_TOLERANCE_S:
        raise ValueError("queue_age_s is not on the control dt grid")
    return queue_age


def _validate_age_semantics(nav_age_s: Any, usbl_age_s: Any, queue_age_s: float) -> None:
    nav_age = np.asarray(nav_age_s, dtype=np.float64)
    usbl_age = np.asarray(usbl_age_s, dtype=np.float64)
    if nav_age.shape != usbl_age.shape:
        raise ValueError("nav_age_s and usbl_age_s shapes differ")
    if not np.all(np.isfinite(usbl_age)) or np.any(usbl_age < -EPS):
        raise ValueError("usbl_age_s must be finite and non-negative")
    hold_age = nav_age - queue_age_s
    if np.any(hold_age < -AGE_GRID_TOLERANCE_S):
        raise ValueError("nav_age_s falls below the inferred queue age")
    if not np.any(hold_age < AGE_GRID_TOLERANCE_S):
        raise ValueError("nav_age_s never reaches the inferred queue age")
    if np.any(usbl_age + AGE_GRID_TOLERANCE_S < hold_age):
        raise ValueError("USBL age is below the nav hold age")


def _interpolate_position(time_s: Any, eta_xyz: Any, query_time_s: float) -> np.ndarray:
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


def _terminal_metrics(
    *,
    p_nav: Any,
    eta: Any,
    goal: Any,
    time_s: Any,
    eta_trajectory_xyz: Any,
    terminal_time_s: float,
    terminal_nav_age_s: float,
    queue_age_s: float,
) -> dict[str, float | bool]:
    p_nav_array = _finite_3vec(p_nav, "p_nav")
    eta_array = _finite_3vec(eta, "eta")
    goal_array = _finite_3vec(goal, "goal")
    terminal_time = float(terminal_time_s)
    terminal_age = float(terminal_nav_age_s)
    queue_age = float(queue_age_s)
    if any(not math.isfinite(value) or value < 0.0 for value in (terminal_age, queue_age)):
        raise ValueError("terminal ages must be finite and non-negative")
    if queue_age > terminal_age + AGE_GRID_TOLERANCE_S:
        raise ValueError("queue age exceeds terminal nav age")
    if terminal_age > terminal_time + EPS or queue_age > terminal_time + EPS:
        raise ValueError("terminal age exceeds terminal time")

    to_goal = goal_array - eta_array
    denominator = float(np.linalg.norm(to_goal))
    if denominator <= EPS:
        raise ValueError("terminal goal direction is undefined")
    goal_direction = to_goal / denominator

    eta_queue = _interpolate_position(
        time_s, eta_trajectory_xyz, terminal_time - queue_age
    )
    eta_goal42 = _interpolate_position(
        time_s, eta_trajectory_xyz, terminal_time - terminal_age
    )
    exact_error_vec = p_nav_array - eta_queue
    goal42_error_vec = p_nav_array - eta_goal42
    exact_projection = float(np.dot(exact_error_vec, goal_direction))
    goal42_projection = float(np.dot(goal42_error_vec, goal_direction))
    return {
        "queue_age_s": queue_age,
        "terminal_nav_age_s": terminal_age,
        "excess_age_s": terminal_age - queue_age,
        "goal42_projection_m": goal42_projection,
        "exact_projection_m": exact_projection,
        "projection_delta_m": goal42_projection - exact_projection,
        "goal42_toward_goal": goal42_projection > TOWARD_GOAL_EPS_M,
        "exact_toward_goal": exact_projection > TOWARD_GOAL_EPS_M,
        "exact_error_norm_m": float(np.linalg.norm(exact_error_vec)),
    }


def _load_goal42_csv(path: Path) -> dict[tuple[str, str, int], float]:
    rows: dict[tuple[str, str, int], float] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = (row["scenario"], row["mode"], int(row["seed"]))
            if key in rows:
                raise ValueError(f"{path}: duplicate row {key}")
            rows[key] = float(row["residual_projection_m"])
    expected = len(SCENARIOS) * len(MODES) * len(EXPECTED_SEEDS)
    if len(rows) != expected:
        raise ValueError(f"{path}: expected {expected} rows, found {len(rows)}")
    return rows


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


def _load_summary(path: Path) -> dict[int, dict[str, Any]]:
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
    return episodes


def _record_for_seed(
    *,
    scenario: str,
    mode: str,
    seed: int,
    rows: np.ndarray,
    trace: Any,
    episode: dict[str, Any],
    goal42_projection_reference: float,
) -> dict[str, Any]:
    nav_age = np.asarray(trace["nav_age_s"][rows], dtype=np.float64)
    usbl_age = np.asarray(trace["usbl_age_s"][rows], dtype=np.float64)
    queue_age = _queue_age_from_nav_age(nav_age)
    _validate_age_semantics(nav_age, usbl_age, queue_age)
    terminal = int(rows[-1])
    metrics = _terminal_metrics(
        p_nav=trace["p_nav"][terminal],
        eta=trace["eta"][terminal, :3],
        goal=trace["goal_ned_m"][terminal],
        time_s=trace["time_s"][rows],
        eta_trajectory_xyz=trace["eta"][rows, :3],
        terminal_time_s=float(trace["time_s"][terminal]),
        terminal_nav_age_s=float(nav_age[-1]),
        queue_age_s=queue_age,
    )
    if abs(float(metrics["goal42_projection_m"]) - goal42_projection_reference) > (
        CSV_PROJECTION_TOLERANCE_M
    ):
        raise ValueError(f"Goal42 CSV projection mismatch for {scenario}/{mode}/{seed}")
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


def _aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    discordant = [row for row in records if row["discordant"]]
    non_discordant = [row for row in records if not row["discordant"]]
    if len(discordant) < MIN_DISCORDANT_SEEDS:
        raise ValueError(f"only {len(discordant)} discordant seeds")
    if len(non_discordant) < MIN_NON_DISCORDANT_SEEDS:
        raise ValueError(f"only {len(non_discordant)} non-discordant seeds")
    exact_toward_count = sum(bool(row["exact_toward_goal"]) for row in discordant)
    goal42_toward_count = sum(bool(row["goal42_toward_goal"]) for row in discordant)
    exact_fraction = exact_toward_count / len(discordant)
    median_excess_age = _median([float(row["excess_age_s"]) for row in discordant])
    median_delta = _median([float(row["projection_delta_m"]) for row in discordant])
    true_delay_pass = exact_fraction >= EXACT_TOWARD_GOAL_FRACTION_MIN
    age_inflation_pass = (
        median_excess_age >= EXCESS_AGE_MEDIAN_MIN_S
        and median_delta >= PROJECTION_DELTA_MEDIAN_MIN_M
    )
    return {
        "seed_count": len(records),
        "discordant_seed_count": len(discordant),
        "non_discordant_seed_count": len(non_discordant),
        "queue_age_median_s": _median([float(row["queue_age_s"]) for row in discordant]),
        "terminal_nav_age_median_s": _median(
            [float(row["terminal_nav_age_s"]) for row in discordant]
        ),
        "excess_age_median_s": median_excess_age,
        "goal42_projection_median_m": _median(
            [float(row["goal42_projection_m"]) for row in discordant]
        ),
        "exact_projection_median_m": _median(
            [float(row["exact_projection_m"]) for row in discordant]
        ),
        "projection_delta_median_m": median_delta,
        "exact_error_norm_median_m": _median(
            [float(row["exact_error_norm_m"]) for row in discordant]
        ),
        "goal42_toward_goal_count": goal42_toward_count,
        "goal42_toward_goal_fraction": goal42_toward_count / len(discordant),
        "exact_toward_goal_count": exact_toward_count,
        "exact_toward_goal_fraction": exact_fraction,
        "h_true_delay_bias_pass": true_delay_pass,
        "h_age_inflation_material_pass": age_inflation_pass,
    }


def _verdict(pass_count: int) -> str:
    if pass_count == len(SCENARIOS) * len(MODES):
        return "SUPPORTED"
    if pass_count >= 1:
        return "PARTIAL"
    return "FALSIFIED"


def _overall_decision(
    true_delay_verdict: str, age_inflation_verdict: str, *, invalid: bool
) -> dict[str, str]:
    if invalid:
        return {
            "decision": "INVALID_DIAGNOSTIC",
            "recommendation": "修输入后只重跑一次；不解释机制。",
        }
    supported = {"SUPPORTED"}
    if (
        true_delay_verdict in supported
        and age_inflation_verdict in supported
    ):
        return {
            "decision": "TRUE_BIAS_WITH_INFLATED_AGE",
            "recommendation": "保留方向结论、降级 magnitude；下一项只读组件审计。",
        }
    if true_delay_verdict in supported:
        return {
            "decision": "TRUE_BIAS_NO_MATERIAL_INFLATION",
            "recommendation": "保留 Goal42 结论；下一项只读组件审计。",
        }
    if age_inflation_verdict in supported:
        return {
            "decision": "GOAL42_RESIDUAL_MAGNITUDE_ARTIFACT",
            "recommendation": "撤回 residual-magnitude 解释；先修 age 定义，不进入组件审计。",
        }
    if "PARTIAL" in (true_delay_verdict, age_inflation_verdict):
        return {
            "decision": "PARTIAL_AGE_SEMANTICS_EVIDENCE",
            "recommendation": "只补数据/输入验证，不训练。",
        }
    return {
        "decision": "NO_EXACT_DELAY_BIAS",
        "recommendation": "停止 calibration-bias 单因素解释，回到 protocol/estimator 结构。",
    }


def _validate_inputs(formal_root: Path, goal42_root: Path) -> dict[str, Any]:
    for name, expected in INPUT_SHA256.items():
        path = goal42_root / name
        if not path.is_file():
            raise ValueError(f"missing input: {path}")
        if _sha256(path) != expected:
            raise ValueError(f"input SHA256 mismatch: {path}")
    for scenario, mode in TRACE_SHA256:
        path = formal_root / scenario / mode / "trace.npz"
        if not path.is_file():
            raise ValueError(f"missing trace: {path}")
        if _sha256(path) != TRACE_SHA256[(scenario, mode)]:
            raise ValueError(f"trace SHA256 mismatch: {path}")
    gate = json.loads((goal42_root / "replay_gate.json").read_text(encoding="utf-8"))
    validation = gate.get("validation", {})
    if validation.get("gate") != "PASS" or not validation.get("valid"):
        raise ValueError("Goal42 replay gate is not PASS")
    if int(validation.get("block_count", 0)) != len(SCENARIOS) * len(MODES):
        raise ValueError("Goal42 replay gate block count mismatch")
    if bool(validation.get("heldout_opened")):
        raise ValueError("Goal42 replay gate reports held-out access")
    return gate


def analyze(formal_root: Path, goal42_root: Path) -> dict[str, Any]:
    gate = _validate_inputs(formal_root, goal42_root)
    csv_reference = _load_goal42_csv(goal42_root / "terminal_age_bias_per_seed.csv")
    scenarios: dict[str, dict[str, Any]] = {}
    records: list[dict[str, Any]] = []
    for scenario in SCENARIOS:
        scenarios[scenario] = {}
        for mode in MODES:
            block_root = formal_root / scenario / mode
            episodes = _load_summary(block_root / "summary.json")
            with np.load(block_root / "trace.npz", allow_pickle=False) as trace:
                rows_by_seed = _seed_row_indices(trace)
                block_records = [
                    _record_for_seed(
                        scenario=scenario,
                        mode=mode,
                        seed=seed,
                        rows=rows_by_seed[seed],
                        trace=trace,
                        episode=episodes[seed],
                        goal42_projection_reference=csv_reference[
                            (scenario, mode, seed)
                        ],
                    )
                    for seed in EXPECTED_SEEDS
                ]
            aggregate = _aggregate(block_records)
            scenarios[scenario][mode] = {
                "aggregate": aggregate,
                "h_true_delay_bias": {
                    "pass": aggregate["h_true_delay_bias_pass"],
                    "threshold_fraction": EXACT_TOWARD_GOAL_FRACTION_MIN,
                },
                "h_age_inflation_material": {
                    "pass": aggregate["h_age_inflation_material_pass"],
                    "threshold_age_s": EXCESS_AGE_MEDIAN_MIN_S,
                    "threshold_projection_m": PROJECTION_DELTA_MEDIAN_MIN_M,
                },
            }
            records.extend(block_records)

    true_delay_blocks = sum(
        scenarios[scenario][mode]["h_true_delay_bias"]["pass"]
        for scenario in SCENARIOS
        for mode in MODES
    )
    age_inflation_blocks = sum(
        scenarios[scenario][mode]["h_age_inflation_material"]["pass"]
        for scenario in SCENARIOS
        for mode in MODES
    )
    true_delay_verdict = _verdict(true_delay_blocks)
    age_inflation_verdict = _verdict(age_inflation_blocks)
    overall = _overall_decision(
        true_delay_verdict, age_inflation_verdict, invalid=False
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "validation": {
            "formal_root": str(formal_root),
            "goal42_root": str(goal42_root),
            "expected_seeds": list(EXPECTED_SEEDS),
            "expected_checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
            "heldout_opened": False,
            "block_count": len(SCENARIOS) * len(MODES),
            "invalid": False,
            "invalid_reasons": [],
            "goal42_replay_gate": gate["validation"]["gate"],
            "input_sha256": {
                **INPUT_SHA256,
                **{
                    f"{scenario}/{mode}/trace.npz": trace_sha
                    for (scenario, mode), trace_sha in TRACE_SHA256.items()
                },
            },
            "thresholds": {
                "dt_s": DT_S,
                "min_discordant_seeds": MIN_DISCORDANT_SEEDS,
                "min_non_discordant_seeds": MIN_NON_DISCORDANT_SEEDS,
                "exact_toward_goal_fraction_min": EXACT_TOWARD_GOAL_FRACTION_MIN,
                "excess_age_median_min_s": EXCESS_AGE_MEDIAN_MIN_S,
                "projection_delta_median_min_m": PROJECTION_DELTA_MEDIAN_MIN_M,
                "toward_goal_eps_m": TOWARD_GOAL_EPS_M,
                "csv_projection_tolerance_m": CSV_PROJECTION_TOLERANCE_M,
            },
        },
        "scenarios": scenarios,
        "overall": {
            **overall,
            "h_true_delay_bias": true_delay_verdict,
            "h_age_inflation_material": age_inflation_verdict,
            "true_delay_passing_blocks": true_delay_blocks,
            "age_inflation_passing_blocks": age_inflation_blocks,
            "block_count": len(SCENARIOS) * len(MODES),
        },
        "records": records,
    }


def _markdown(result: dict[str, Any]) -> str:
    lines = [
        "# Goal43 nav-age semantics 与 exact queue-delay 审计",
        "",
        f"Schema: `{result['schema_version']}`<br>",
        f"总体裁决：**{result['overall']['decision']}**",
        "",
        "## Block results",
        "",
        "| scenario | mode | discordant | queue age s | nav age s | excess age s | "
        "goal42 proj m | exact proj m | delta m | exact toward frac | H-TRUE | H-INFLATE |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for scenario in SCENARIOS:
        for mode in MODES:
            block = result["scenarios"][scenario][mode]
            aggregate = block["aggregate"]
            lines.append(
                f"| {scenario} | {mode} | {aggregate['discordant_seed_count']} | "
                f"{aggregate['queue_age_median_s']:.4f} | "
                f"{aggregate['terminal_nav_age_median_s']:.4f} | "
                f"{aggregate['excess_age_median_s']:.4f} | "
                f"{aggregate['goal42_projection_median_m']:.4f} | "
                f"{aggregate['exact_projection_median_m']:.4f} | "
                f"{aggregate['projection_delta_median_m']:.4f} | "
                f"{aggregate['exact_toward_goal_fraction']:.4f} | "
                f"{'PASS' if block['h_true_delay_bias']['pass'] else 'FAIL'} | "
                f"{'PASS' if block['h_age_inflation_material']['pass'] else 'FAIL'} |"
            )
    lines.extend(
        [
            "",
            "## Overall",
            "",
            f"- H-TRUE-DELAY-BIAS: `{result['overall']['h_true_delay_bias']}`",
            "- H-AGE-INFLATION-MATERIAL: "
            f"`{result['overall']['h_age_inflation_material']}`",
            f"- decision: `{result['overall']['decision']}`",
            f"- recommendation: {result['overall']['recommendation']}",
            "",
            "## Validation",
            "",
            f"- invalid: `{result['validation']['invalid']}`",
            f"- blocks: `{result['validation']['block_count']}/4`",
            f"- Goal42 replay gate: `{result['validation']['goal42_replay_gate']}`",
            f"- held-out opened: `{result['validation']['heldout_opened']}`",
        ]
    )
    return "\n".join(lines) + "\n"


def write_outputs(result: dict[str, Any], output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / "nav_age_semantics_diagnostic.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "nav_age_semantics_diagnostic.md").write_text(
        _markdown(result), encoding="utf-8"
    )
    with (output / "nav_age_semantics_per_seed.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(result["records"])


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = analyze(args.formal_root, args.goal42_root)
    write_outputs(result, args.output)
    print(json.dumps(result["overall"], indent=2, sort_keys=True))
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
