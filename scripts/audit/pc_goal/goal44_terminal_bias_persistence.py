#!/usr/bin/env python3
"""Audit whether terminal exact queue-delay bias persists before the terminal."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA_VERSION = "auv_pc_goal44_terminal_bias_persistence.v1"
SCENARIOS = ("default", "strict125")
MODES = ("baseline", "candidate")
EXPECTED_SEEDS = tuple(range(20, 60))
EXPECTED_CHECKPOINT_SHA256 = (
    "0851496b15ffebbc0a19bafa5bf40c0493c0d9a3f485267b75fde5f811dde5d9"
)
DT_S = 0.02
LATE_WINDOW_S = 5.0
TERMINAL_EXCLUSION_S = 0.5
MIN_WINDOW_ROWS = 100
MIN_DISCORDANT_SEEDS = 5
MIN_NON_DISCORDANT_SEEDS = 10
PERSISTENT_TOWARD_GOAL_FRACTION_MIN = 0.80
PERSISTENT_LATE_SHARE_MEDIAN_MIN = 0.50
TERMINAL_DELTA_MEDIAN_MIN_M = 0.10
TOWARD_GOAL_EPS_M = 1e-6
CSV_PROJECTION_TOLERANCE_M = 1e-4
AGE_GRID_TOLERANCE_S = 1e-9
EPS = 1e-12

INPUT_SHA256 = {
    "default/baseline/trace.npz": (
        "fe199b283f248fd94b5c416a4ab6639b3bf43f3f6824672f21966daf2ceb52d8"
    ),
    "default/candidate/trace.npz": (
        "dc52bed06ad8fdddc52a58bb0fb73dbf9cb4ba049bbcce919344ae6e622e0548"
    ),
    "strict125/baseline/trace.npz": (
        "aef6238f0996e5843242cb8ef3ac82e3bd966e369e63dcd91a82008644195387"
    ),
    "strict125/candidate/trace.npz": (
        "b471fe1bd08c1540e10926dbb15a4736997f3787a1f9668831dc42e95b369a02"
    ),
}
GOAL43_SHA256 = {
    "nav_age_semantics_diagnostic.json": (
        "a14e1b008bb86ebb3474fda04e6f6fafd825f00bca858548f397099652083691"
    ),
    "nav_age_semantics_per_seed.csv": (
        "e2db51bc7f5b0a06a3720b4fca377303cf2712b087be8a5d0706493f8bcaddbf"
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
    "window_rows",
    "late_projection_m",
    "terminal_projection_m",
    "terminal_delta_m",
    "late_share",
    "late_toward_goal",
    "terminal_toward_goal",
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
        "--goal43-root",
        type=Path,
        default=Path(
            "artifacts/lowlevel/goal43_nav_age_semantics_audit_20261109"
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
    if not np.all(np.isfinite(ages)) or np.any(ages < -EPS):
        raise ValueError("nav_age_s must be finite and non-negative")
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


def _exact_error_trajectory(
    *, p_nav: Any, time_s: Any, eta_trajectory_xyz: Any, queue_age_s: float
) -> tuple[np.ndarray, np.ndarray]:
    p_nav_array = np.asarray(p_nav, dtype=np.float64)
    times = np.asarray(time_s, dtype=np.float64)
    if p_nav_array.shape != (times.size, 3):
        raise ValueError("p_nav and time_s shapes are inconsistent")
    source_times = times - float(queue_age_s)
    valid = source_times >= float(times[0]) - EPS
    if not np.any(valid):
        raise ValueError("no row has a valid exact queue-delay source time")
    errors = np.full_like(p_nav_array, np.nan)
    for row in np.flatnonzero(valid):
        eta_source = _interpolate_position(times, eta_trajectory_xyz, source_times[row])
        errors[row] = p_nav_array[row] - eta_source
    if not np.all(np.isfinite(errors[valid])):
        raise ValueError("exact error trajectory must be finite")
    return errors, valid


def _late_window_metrics(
    *,
    exact_error: np.ndarray,
    valid_rows: np.ndarray,
    time_s: Any,
    eta_terminal: Any,
    goal: Any,
) -> dict[str, Any]:
    times = np.asarray(time_s, dtype=np.float64)
    terminal_time = float(times[-1])
    window_mask = (
        valid_rows
        & (times >= terminal_time - LATE_WINDOW_S - EPS)
        & (times <= terminal_time - TERMINAL_EXCLUSION_S + EPS)
    )
    window_rows = int(window_mask.sum())
    if window_rows < MIN_WINDOW_ROWS:
        raise ValueError(f"late window has only {window_rows} rows")
    eta_array = _finite_3vec(eta_terminal, "eta_terminal")
    goal_array = _finite_3vec(goal, "goal")
    to_goal = goal_array - eta_array
    denominator = float(np.linalg.norm(to_goal))
    if denominator <= EPS:
        raise ValueError("terminal goal direction is undefined")
    goal_direction = to_goal / denominator
    late_error_vec = np.median(exact_error[window_mask], axis=0)
    terminal_error_vec = exact_error[-1]
    late_projection = float(np.dot(late_error_vec, goal_direction))
    terminal_projection = float(np.dot(terminal_error_vec, goal_direction))
    if abs(terminal_projection) <= EPS:
        late_share = 0.0
    else:
        late_share = late_projection / terminal_projection
    return {
        "window_rows": window_rows,
        "late_projection_m": late_projection,
        "terminal_projection_m": terminal_projection,
        "terminal_delta_m": terminal_projection - late_projection,
        "late_share": late_share,
        "late_toward_goal": late_projection > TOWARD_GOAL_EPS_M,
        "terminal_toward_goal": terminal_projection > TOWARD_GOAL_EPS_M,
    }


def _load_goal43_csv(path: Path) -> dict[tuple[str, str, int], dict[str, float]]:
    rows: dict[tuple[str, str, int], dict[str, float]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = (row["scenario"], row["mode"], int(row["seed"]))
            if key in rows:
                raise ValueError(f"{path}: duplicate row {key}")
            rows[key] = {
                "queue_age_s": float(row["queue_age_s"]),
                "exact_projection_m": float(row["exact_projection_m"]),
            }
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
    reference: dict[str, float],
) -> dict[str, Any]:
    nav_age = np.asarray(trace["nav_age_s"][rows], dtype=np.float64)
    usbl_age = np.asarray(trace["usbl_age_s"][rows], dtype=np.float64)
    queue_age = _queue_age_from_nav_age(nav_age)
    _validate_age_semantics(nav_age, usbl_age, queue_age)
    if abs(queue_age - reference["queue_age_s"]) > AGE_GRID_TOLERANCE_S:
        raise ValueError(f"Goal43 queue age mismatch for {scenario}/{mode}/{seed}")
    exact_error, valid_rows = _exact_error_trajectory(
        p_nav=trace["p_nav"][rows],
        time_s=trace["time_s"][rows],
        eta_trajectory_xyz=trace["eta"][rows, :3],
        queue_age_s=queue_age,
    )
    metrics = _late_window_metrics(
        exact_error=exact_error,
        valid_rows=valid_rows,
        time_s=trace["time_s"][rows],
        eta_terminal=trace["eta"][rows[-1], :3],
        goal=trace["goal_ned_m"][rows[-1]],
    )
    if abs(metrics["terminal_projection_m"] - reference["exact_projection_m"]) > (
        CSV_PROJECTION_TOLERANCE_M
    ):
        raise ValueError(f"Goal43 exact projection mismatch for {scenario}/{mode}/{seed}")
    success = bool(episode["success"])
    eta_capture = bool(episode["goal_capture_eta"])
    discordant = success and not eta_capture
    if discordant and metrics["terminal_projection_m"] <= TOWARD_GOAL_EPS_M:
        raise ValueError(f"discordant terminal projection is not positive: {seed}")
    return {
        "scenario": scenario,
        "mode": mode,
        "seed": seed,
        "success": success,
        "goal_capture_eta": eta_capture,
        "discordant": discordant,
        "queue_age_s": queue_age,
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
    late_toward_count = sum(bool(row["late_toward_goal"]) for row in discordant)
    late_toward_fraction = late_toward_count / len(discordant)
    median_late_share = _median([float(row["late_share"]) for row in discordant])
    median_terminal_delta = _median(
        [float(row["terminal_delta_m"]) for row in discordant]
    )
    persistent_pass = (
        late_toward_fraction >= PERSISTENT_TOWARD_GOAL_FRACTION_MIN
        and median_late_share >= PERSISTENT_LATE_SHARE_MEDIAN_MIN
    )
    amplification_pass = median_terminal_delta >= TERMINAL_DELTA_MEDIAN_MIN_M
    return {
        "seed_count": len(records),
        "discordant_seed_count": len(discordant),
        "non_discordant_seed_count": len(non_discordant),
        "window_rows_median": int(
            np.median([int(row["window_rows"]) for row in discordant])
        ),
        "late_projection_median_m": _median(
            [float(row["late_projection_m"]) for row in discordant]
        ),
        "terminal_projection_median_m": _median(
            [float(row["terminal_projection_m"]) for row in discordant]
        ),
        "terminal_delta_median_m": median_terminal_delta,
        "late_share_median": median_late_share,
        "late_toward_goal_count": late_toward_count,
        "late_toward_goal_fraction": late_toward_fraction,
        "h_persistent_late_bias_pass": persistent_pass,
        "h_terminal_amplification_pass": amplification_pass,
    }


def _verdict(pass_count: int) -> str:
    if pass_count == len(SCENARIOS) * len(MODES):
        return "SUPPORTED"
    if pass_count >= 1:
        return "PARTIAL"
    return "FALSIFIED"


def _overall_decision(
    persistent_verdict: str, amplification_verdict: str, *, invalid: bool
) -> dict[str, str]:
    if invalid:
        return {
            "decision": "INVALID_DIAGNOSTIC",
            "recommendation": "修输入后只重跑一次；不解释机制。",
        }
    if persistent_verdict == "SUPPORTED" and amplification_verdict == "SUPPORTED":
        return {
            "decision": "PERSISTENT_BIAS_WITH_TERMINAL_AMPLIFICATION",
            "recommendation": "预注册只读 late/terminal 分量分解；不训练。",
        }
    if persistent_verdict == "SUPPORTED":
        return {
            "decision": "PERSISTENT_LATE_BIAS",
            "recommendation": "预注册只读 persistent offset 审计；不训练。",
        }
    if amplification_verdict == "SUPPORTED":
        return {
            "decision": "TERMINAL_TRANSIENT_BIAS",
            "recommendation": "预注册只读 terminal event 审计；不训练。",
        }
    if "PARTIAL" in (persistent_verdict, amplification_verdict):
        return {
            "decision": "PARTIAL_PERSISTENCE_EVIDENCE",
            "recommendation": "只补数据/窗口验证，不训练。",
        }
    return {
        "decision": "NO_STABLE_LATE_SIGNATURE",
        "recommendation": "停止单因素 persistence 解释，回到 estimator 结构。",
    }


def _validate_inputs(formal_root: Path, goal43_root: Path) -> dict[str, Any]:
    for relative, expected in INPUT_SHA256.items():
        path = formal_root / relative
        if not path.is_file():
            raise ValueError(f"missing input: {path}")
        if _sha256(path) != expected:
            raise ValueError(f"input SHA256 mismatch: {path}")
    for name, expected in GOAL43_SHA256.items():
        path = goal43_root / name
        if not path.is_file():
            raise ValueError(f"missing input: {path}")
        if _sha256(path) != expected:
            raise ValueError(f"Goal43 output SHA256 mismatch: {path}")
    diagnostic = json.loads(
        (goal43_root / "nav_age_semantics_diagnostic.json").read_text(encoding="utf-8")
    )
    if diagnostic.get("overall", {}).get("decision") != "TRUE_BIAS_WITH_INFLATED_AGE":
        raise ValueError("Goal43 decision is not TRUE_BIAS_WITH_INFLATED_AGE")
    if bool(diagnostic.get("validation", {}).get("heldout_opened")):
        raise ValueError("Goal43 diagnostic reports held-out access")
    return diagnostic


def analyze(formal_root: Path, goal43_root: Path) -> dict[str, Any]:
    goal43 = _validate_inputs(formal_root, goal43_root)
    references = _load_goal43_csv(goal43_root / "nav_age_semantics_per_seed.csv")
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
                        reference=references[(scenario, mode, seed)],
                    )
                    for seed in EXPECTED_SEEDS
                ]
            aggregate = _aggregate(block_records)
            scenarios[scenario][mode] = {
                "aggregate": aggregate,
                "h_persistent_late_bias": {
                    "pass": aggregate["h_persistent_late_bias_pass"],
                    "threshold_fraction": PERSISTENT_TOWARD_GOAL_FRACTION_MIN,
                    "threshold_late_share": PERSISTENT_LATE_SHARE_MEDIAN_MIN,
                },
                "h_terminal_amplification": {
                    "pass": aggregate["h_terminal_amplification_pass"],
                    "threshold_delta_m": TERMINAL_DELTA_MEDIAN_MIN_M,
                },
            }
            records.extend(block_records)

    persistent_blocks = sum(
        scenarios[scenario][mode]["h_persistent_late_bias"]["pass"]
        for scenario in SCENARIOS
        for mode in MODES
    )
    amplification_blocks = sum(
        scenarios[scenario][mode]["h_terminal_amplification"]["pass"]
        for scenario in SCENARIOS
        for mode in MODES
    )
    persistent_verdict = _verdict(persistent_blocks)
    amplification_verdict = _verdict(amplification_blocks)
    overall = _overall_decision(
        persistent_verdict, amplification_verdict, invalid=False
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "validation": {
            "formal_root": str(formal_root),
            "goal43_root": str(goal43_root),
            "expected_seeds": list(EXPECTED_SEEDS),
            "expected_checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
            "heldout_opened": False,
            "block_count": len(SCENARIOS) * len(MODES),
            "invalid": False,
            "invalid_reasons": [],
            "goal43_decision": goal43["overall"]["decision"],
            "input_sha256": {**INPUT_SHA256, **GOAL43_SHA256},
            "thresholds": {
                "dt_s": DT_S,
                "late_window_s": LATE_WINDOW_S,
                "terminal_exclusion_s": TERMINAL_EXCLUSION_S,
                "min_window_rows": MIN_WINDOW_ROWS,
                "min_discordant_seeds": MIN_DISCORDANT_SEEDS,
                "min_non_discordant_seeds": MIN_NON_DISCORDANT_SEEDS,
                "persistent_toward_goal_fraction_min": (
                    PERSISTENT_TOWARD_GOAL_FRACTION_MIN
                ),
                "persistent_late_share_median_min": (
                    PERSISTENT_LATE_SHARE_MEDIAN_MIN
                ),
                "terminal_delta_median_min_m": TERMINAL_DELTA_MEDIAN_MIN_M,
                "toward_goal_eps_m": TOWARD_GOAL_EPS_M,
                "csv_projection_tolerance_m": CSV_PROJECTION_TOLERANCE_M,
            },
        },
        "scenarios": scenarios,
        "overall": {
            **overall,
            "h_persistent_late_bias": persistent_verdict,
            "h_terminal_amplification": amplification_verdict,
            "persistent_passing_blocks": persistent_blocks,
            "amplification_passing_blocks": amplification_blocks,
            "block_count": len(SCENARIOS) * len(MODES),
        },
        "records": records,
    }


def _markdown(result: dict[str, Any]) -> str:
    lines = [
        "# Goal44 terminal estimator bias persistence 审计",
        "",
        f"Schema: `{result['schema_version']}`<br>",
        f"总体裁决：**{result['overall']['decision']}**",
        "",
        "## Block results",
        "",
        "| scenario | mode | discordant | window rows | late proj m | terminal proj m | "
        "delta m | late share | late toward frac | H-PERSIST | H-AMPLIFY |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for scenario in SCENARIOS:
        for mode in MODES:
            block = result["scenarios"][scenario][mode]
            aggregate = block["aggregate"]
            lines.append(
                f"| {scenario} | {mode} | {aggregate['discordant_seed_count']} | "
                f"{aggregate['window_rows_median']} | "
                f"{aggregate['late_projection_median_m']:.4f} | "
                f"{aggregate['terminal_projection_median_m']:.4f} | "
                f"{aggregate['terminal_delta_median_m']:.4f} | "
                f"{aggregate['late_share_median']:.4f} | "
                f"{aggregate['late_toward_goal_fraction']:.4f} | "
                f"{'PASS' if block['h_persistent_late_bias']['pass'] else 'FAIL'} | "
                f"{'PASS' if block['h_terminal_amplification']['pass'] else 'FAIL'} |"
            )
    lines.extend(
        [
            "",
            "## Overall",
            "",
            f"- H-PERSISTENT-LATE-BIAS: `{result['overall']['h_persistent_late_bias']}`",
            "- H-TERMINAL-AMPLIFICATION: "
            f"`{result['overall']['h_terminal_amplification']}`",
            f"- decision: `{result['overall']['decision']}`",
            f"- recommendation: {result['overall']['recommendation']}",
            "",
            "## Validation",
            "",
            f"- invalid: `{result['validation']['invalid']}`",
            f"- blocks: `{result['validation']['block_count']}/4`",
            f"- Goal43 decision: `{result['validation']['goal43_decision']}`",
            f"- held-out opened: `{result['validation']['heldout_opened']}`",
        ]
    )
    return "\n".join(lines) + "\n"


def write_outputs(result: dict[str, Any], output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / "terminal_bias_persistence_diagnostic.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "terminal_bias_persistence_diagnostic.md").write_text(
        _markdown(result), encoding="utf-8"
    )
    with (output / "terminal_bias_persistence_per_seed.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(result["records"])


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = analyze(args.formal_root, args.goal43_root)
    write_outputs(result, args.output)
    print(json.dumps(result["overall"], indent=2, sort_keys=True))
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
