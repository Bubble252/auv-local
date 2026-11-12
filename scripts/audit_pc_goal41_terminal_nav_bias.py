#!/usr/bin/env python3
"""Audit terminal p_nav-versus-eta bias on the Goal40 discordant subgroup.

The script consumes the four Goal38 formal trace directories. It never loads a
policy, steps an environment, trains, or reads held-out seeds.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA_VERSION = "auv_pc_goal41_terminal_nav_bias_diagnostic.v1"
SCENARIOS = ("default", "strict125")
MODES = ("baseline", "candidate")
EXPECTED_SEEDS = tuple(range(20, 60))
EXPECTED_CHECKPOINT_SHA256 = (
    "0851496b15ffebbc0a19bafa5bf40c0493c0d9a3f485267b75fde5f811dde5d9"
)
EXPECTED_TRACE_SHA256 = {
    ("default", "baseline"): (
        "2ffba9597bf37acc9286f994555fae4c5170e1494103572c31fc939bda4a1528"
    ),
    ("default", "candidate"): (
        "bb0c8ed0722604cad763d89eac33f93bc0cdac3ac313f9dcdce6672a4a0f1985"
    ),
    ("strict125", "baseline"): (
        "fe33817f9a17c7514decc25870c67190fb649063e9e3e611f977211f3f8d8aed"
    ),
    ("strict125", "candidate"): (
        "78d0b88eda4d1e8a202562db7d0bff80c2212b432dd6df466f13400a43652cfa"
    ),
}
MIN_DISCORDANT_SEEDS = 5
PNAV_CLOSER_FRACTION_MIN = 0.80
TOWARD_GOAL_FRACTION_MIN = 0.80
DISTANCE_TOLERANCE_M = 1e-6
EPS = 1e-12

CSV_FIELDS = (
    "scenario",
    "mode",
    "seed",
    "success",
    "goal_capture_eta",
    "discordant",
    "pnav_goal_distance_m",
    "eta_goal_distance_m",
    "distance_gap_m",
    "p_nav_closer",
    "nav_error_m",
    "direction_cos",
    "toward_goal",
    "current_estimate_age_s",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--formal-root",
        type=Path,
        default=Path(
            "artifacts/lowlevel/goal38_path_provenance_repair_20261109/formal"
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _sha256_file(path: Path) -> str:
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


def _terminal_nav_metrics(
    p_nav: Any,
    eta: Any,
    goal: Any,
    pnav_goal_distance_m: float,
    eta_goal_distance_m: float,
) -> dict[str, Any]:
    p_nav_array = _finite_3vec(p_nav, "p_nav")
    eta_array = _finite_3vec(eta, "eta")
    goal_array = _finite_3vec(goal, "goal")
    pnav_distance = float(pnav_goal_distance_m)
    eta_distance = float(eta_goal_distance_m)
    if not math.isfinite(pnav_distance) or not math.isfinite(eta_distance):
        raise ValueError("terminal goal distances must be finite")

    recomputed_pnav_distance = float(np.linalg.norm(p_nav_array - goal_array))
    recomputed_eta_distance = float(np.linalg.norm(eta_array - goal_array))
    if abs(recomputed_pnav_distance - pnav_distance) > DISTANCE_TOLERANCE_M:
        raise ValueError("p_nav goal distance is inconsistent with trace positions")
    if abs(recomputed_eta_distance - eta_distance) > DISTANCE_TOLERANCE_M:
        raise ValueError("eta goal distance is inconsistent with trace positions")

    nav_error_vec = p_nav_array - eta_array
    nav_error_m = float(np.linalg.norm(nav_error_vec))
    to_goal_vec = goal_array - eta_array
    to_goal_norm = float(np.linalg.norm(to_goal_vec))
    denominator = nav_error_m * to_goal_norm
    if denominator <= EPS:
        raise ValueError("terminal nav direction cosine is undefined")
    direction_cos = float(np.dot(nav_error_vec, to_goal_vec) / denominator)
    direction_cos = float(np.clip(direction_cos, -1.0, 1.0))
    distance_gap_m = eta_distance - pnav_distance
    return {
        "pnav_goal_distance_m": pnav_distance,
        "eta_goal_distance_m": eta_distance,
        "distance_gap_m": distance_gap_m,
        "p_nav_closer": distance_gap_m > DISTANCE_TOLERANCE_M,
        "nav_error_m": nav_error_m,
        "direction_cos": direction_cos,
        "toward_goal": direction_cos > 0.0,
    }


def _last_row_indices(trace: Any) -> dict[int, int]:
    required = (
        "episode_seed",
        "step",
        "time_s",
        "p_nav",
        "eta",
        "goal_ned_m",
        "goal_distance_pnav_m",
        "goal_distance_eta_m",
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
    if seeds.size == 0:
        raise ValueError("trace is empty")
    if not np.all(np.isfinite(time_s)):
        raise ValueError("trace time contains non-finite values")
    result: dict[int, int] = {}
    for seed in EXPECTED_SEEDS:
        indices = np.flatnonzero(seeds == seed)
        if indices.size == 0:
            raise ValueError(f"trace has no rows for seed {seed}")
        if np.any(np.diff(steps[indices]) <= 0) or np.any(np.diff(time_s[indices]) <= 0):
            raise ValueError(f"trace for seed {seed} is not strictly increasing")
        result[seed] = int(indices[-1])
    return result


def _record_for_seed(
    seed: int,
    row_index: int,
    trace: Any,
    episode: dict[str, Any],
    scenario: str,
    mode: str,
) -> dict[str, Any]:
    success = bool(episode["success"])
    eta_capture = bool(episode["goal_capture_eta"])
    metrics = _terminal_nav_metrics(
        trace["p_nav"][row_index],
        trace["eta"][row_index, :3],
        trace["goal_ned_m"][row_index],
        float(trace["goal_distance_pnav_m"][row_index]),
        float(trace["goal_distance_eta_m"][row_index]),
    )
    current_estimate_age_s = float(trace["current_estimate_age_s"][row_index])
    if not math.isfinite(current_estimate_age_s):
        raise ValueError(f"non-finite current estimate age for seed {seed}")
    record = {
        "scenario": scenario,
        "mode": mode,
        "seed": seed,
        "success": success,
        "goal_capture_eta": eta_capture,
        "discordant": success and not eta_capture,
        "current_estimate_age_s": current_estimate_age_s,
        **metrics,
    }
    for key in (
        "pnav_goal_distance_m",
        "eta_goal_distance_m",
        "distance_gap_m",
        "nav_error_m",
        "direction_cos",
        "current_estimate_age_s",
    ):
        if not math.isfinite(float(record[key])):
            raise ValueError(f"non-finite {key} for seed {seed}")
    return record


def _median(values: list[float]) -> float:
    return float(np.median(np.asarray(values, dtype=np.float64)))


def _percentile(values: list[float], percentile: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), percentile))


def _aggregate(discordant: list[dict[str, Any]]) -> dict[str, Any]:
    if not discordant:
        raise ValueError("block has no protocol-success-without-eta-capture seeds")
    count = len(discordant)
    p_nav_closer_count = sum(bool(row["p_nav_closer"]) for row in discordant)
    toward_goal_count = sum(bool(row["toward_goal"]) for row in discordant)
    ages = [float(row["current_estimate_age_s"]) for row in discordant]
    return {
        "discordant_seed_count": count,
        "p_nav_closer_count": p_nav_closer_count,
        "p_nav_closer_fraction": p_nav_closer_count / count,
        "toward_goal_count": toward_goal_count,
        "toward_goal_fraction": toward_goal_count / count,
        "median_pnav_goal_distance_m": _median(
            [float(row["pnav_goal_distance_m"]) for row in discordant]
        ),
        "median_eta_goal_distance_m": _median(
            [float(row["eta_goal_distance_m"]) for row in discordant]
        ),
        "median_distance_gap_m": _median(
            [float(row["distance_gap_m"]) for row in discordant]
        ),
        "median_nav_error_m": _median(
            [float(row["nav_error_m"]) for row in discordant]
        ),
        "median_direction_cos": _median(
            [float(row["direction_cos"]) for row in discordant]
        ),
        "median_current_estimate_age_s": _median(ages),
        "p95_current_estimate_age_s": _percentile(ages, 95.0),
    }


def _mode_verdict(pass_count: int) -> str:
    if pass_count == len(SCENARIOS) * len(MODES):
        return "SUPPORTED"
    if pass_count >= 1:
        return "PARTIAL"
    return "FALSIFIED"


def _block_hypothesis_passes(aggregate: dict[str, Any]) -> dict[str, bool]:
    return {
        "distance": aggregate["p_nav_closer_fraction"]
        >= PNAV_CLOSER_FRACTION_MIN - EPS,
        "direction": aggregate["toward_goal_fraction"]
        >= TOWARD_GOAL_FRACTION_MIN - EPS,
    }


def _overall_decision(
    distance_verdict: str, direction_verdict: str, invalid: bool
) -> dict[str, str]:
    if invalid:
        return {
            "decision": "INVALID_DIAGNOSTIC",
            "recommendation": "修复输入、seed、checkpoint、discordant 或距离一致性后重跑。",
        }
    if distance_verdict == "SUPPORTED" and direction_verdict == "SUPPORTED":
        return {
            "decision": "NAV_BIAS_MECHANISM_SUPPORTED",
            "recommendation": "只允许预注册 estimator calibration/nav-age 审计；不训练控制器。",
        }
    if distance_verdict == "SUPPORTED":
        return {
            "decision": "DISTANCE_METRIC_BIAS_SUPPORTED",
            "recommendation": "只允许预注册 protocol metric/termination 审计；不改控制器。",
        }
    if direction_verdict == "SUPPORTED":
        return {
            "decision": "DIRECTION_BIAS_SUPPORTED",
            "recommendation": "只允许预注册 nav error 方向/外推诊断；不改控制器。",
        }
    if "PARTIAL" in (distance_verdict, direction_verdict):
        return {
            "decision": "PARTIAL_NAV_BIAS",
            "recommendation": "只补最便宜的分组验证，不训练。",
        }
    return {
        "decision": "NO_SUPPORTED_NAV_BIAS",
        "recommendation": "停止该 nav-bias 解释，回到分量/控制器问题。",
    }


def _load_block(
    formal_root: Path, scenario: str, mode: str
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    directory = formal_root / scenario / mode
    trace_path = directory / "trace.npz"
    actual_trace_sha256 = _sha256_file(trace_path)
    expected_trace_sha256 = EXPECTED_TRACE_SHA256[(scenario, mode)]
    if actual_trace_sha256 != expected_trace_sha256:
        raise ValueError(
            f"{scenario}/{mode}: trace SHA256 mismatch: {actual_trace_sha256}"
        )

    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    seeds = sorted(int(seed) for seed in summary["seeds"])
    if seeds != list(EXPECTED_SEEDS):
        raise ValueError(f"{scenario}/{mode}: seeds mismatch: {seeds}")
    if summary.get("checkpoint_sha256") != EXPECTED_CHECKPOINT_SHA256:
        raise ValueError(f"{scenario}/{mode}: checkpoint SHA256 mismatch")

    episode_items = summary["episodes"]
    if len(episode_items) != len(EXPECTED_SEEDS):
        raise ValueError(f"{scenario}/{mode}: episode count mismatch")
    episodes = {int(item["seed"]): item for item in episode_items}
    if sorted(episodes) != list(EXPECTED_SEEDS):
        raise ValueError(f"{scenario}/{mode}: episode seeds mismatch")
    for item in episodes.values():
        for key in ("success", "goal_capture_eta"):
            if not isinstance(item.get(key), bool):
                raise ValueError(f"{scenario}/{mode}: episode has invalid {key}")

    with np.load(trace_path, allow_pickle=False) as trace:
        last_indices = _last_row_indices(trace)
        records = []
        for seed in EXPECTED_SEEDS:
            episode = dict(episodes[seed])
            record = _record_for_seed(
                seed,
                last_indices[seed],
                trace,
                episode,
                scenario,
                mode,
            )
            records.append(record)

    discordant = [row for row in records if row["discordant"]]
    if len(discordant) < MIN_DISCORDANT_SEEDS:
        raise ValueError(
            f"{scenario}/{mode}: only {len(discordant)} discordant seeds"
        )
    aggregate = _aggregate(discordant)
    hypothesis_passes = _block_hypothesis_passes(aggregate)
    block = {
        "scenario": scenario,
        "mode": mode,
        "source": str(directory),
        "trace_sha256": actual_trace_sha256,
        "checkpoint_sha256": summary.get("checkpoint_sha256"),
        "records": records,
        "aggregate": aggregate,
        "h_nav_distance_bias": {"pass": hypothesis_passes["distance"]},
        "h_nav_direction_bias": {"pass": hypothesis_passes["direction"]},
    }
    return block, records


def analyze(formal_root: Path) -> dict[str, Any]:
    scenarios: dict[str, Any] = {}
    all_blocks: list[dict[str, Any]] = []
    all_records: list[dict[str, Any]] = []
    invalid_reasons: list[str] = []
    for scenario in SCENARIOS:
        scenario_blocks: dict[str, Any] = {}
        for mode in MODES:
            try:
                block, records = _load_block(formal_root, scenario, mode)
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
        distance_blocks = direction_blocks = 0
    else:
        distance_blocks = sum(
            block_by_key[(scenario, mode)]["h_nav_distance_bias"]["pass"]
            for scenario in SCENARIOS
            for mode in MODES
        )
        direction_blocks = sum(
            block_by_key[(scenario, mode)]["h_nav_direction_bias"]["pass"]
            for scenario in SCENARIOS
            for mode in MODES
        )
    distance_verdict = _mode_verdict(distance_blocks)
    direction_verdict = _mode_verdict(direction_blocks)
    overall = _overall_decision(distance_verdict, direction_verdict, invalid)
    return {
        "schema_version": SCHEMA_VERSION,
        "validation": {
            "formal_root": str(formal_root),
            "expected_seeds": list(EXPECTED_SEEDS),
            "expected_checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
            "expected_trace_sha256": {
                f"{scenario}/{mode}": EXPECTED_TRACE_SHA256[(scenario, mode)]
                for scenario in SCENARIOS
                for mode in MODES
            },
            "heldout_opened": False,
            "block_count": len(all_blocks),
            "invalid": invalid,
            "invalid_reasons": invalid_reasons,
            "thresholds": {
                "min_discordant_seeds": MIN_DISCORDANT_SEEDS,
                "p_nav_closer_fraction_min": PNAV_CLOSER_FRACTION_MIN,
                "toward_goal_fraction_min": TOWARD_GOAL_FRACTION_MIN,
                "distance_tolerance_m": DISTANCE_TOLERANCE_M,
            },
        },
        "scenarios": scenarios,
        "overall": {
            **overall,
            "h_nav_distance_bias": distance_verdict,
            "h_nav_direction_bias": direction_verdict,
            "distance_passing_blocks": distance_blocks,
            "direction_passing_blocks": direction_blocks,
            "block_count": len(all_blocks),
        },
        "records": all_records,
    }


def _markdown(result: dict[str, Any]) -> str:
    lines = [
        "# Goal41 terminal p_nav/eta 偏差方向审计",
        "",
        f"Schema: `{result['schema_version']}`<br>",
        f"总体裁决：**{result['overall']['decision']}**",
        "",
        "## Block results",
        "",
        "| scenario | mode | discordant n | p_nav closer frac | toward goal frac | "
        "median cos | median age s | H-DISTANCE | H-DIRECTION |",
        "|---|---|---:|---:|---:|---:|---:|---|---|",
    ]
    for scenario in SCENARIOS:
        for mode in MODES:
            block = result["scenarios"].get(scenario, {}).get(mode)
            if block is None:
                lines.append(
                    f"| {scenario} | {mode} | n/a | n/a | n/a | n/a | n/a | n/a | n/a |"
                )
                continue
            aggregate = block["aggregate"]
            lines.append(
                f"| {scenario} | {mode} | {aggregate['discordant_seed_count']} | "
                f"{aggregate['p_nav_closer_fraction']:.4f} | "
                f"{aggregate['toward_goal_fraction']:.4f} | "
                f"{aggregate['median_direction_cos']:.4f} | "
                f"{aggregate['median_current_estimate_age_s']:.4f} | "
                f"{'PASS' if block['h_nav_distance_bias']['pass'] else 'FAIL'} | "
                f"{'PASS' if block['h_nav_direction_bias']['pass'] else 'FAIL'} |"
            )
    lines.extend(
        [
            "",
            "## Overall",
            "",
            f"- H-NAV-DISTANCE-BIAS: `{result['overall']['h_nav_distance_bias']}`",
            f"- H-NAV-DIRECTION-BIAS: `{result['overall']['h_nav_direction_bias']}`",
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
    (output / "terminal_nav_bias_diagnostic.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "terminal_nav_bias_diagnostic.md").write_text(
        _markdown(result), encoding="utf-8"
    )
    with (output / "terminal_nav_bias_per_seed.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for record in result["records"]:
            writer.writerow({key: record[key] for key in CSV_FIELDS})


def main() -> int:
    args = parse_args()
    result = analyze(args.formal_root)
    write_outputs(result, args.output)
    print(
        json.dumps(
            {
                "H_NAV_DISTANCE_BIAS": result["overall"]["h_nav_distance_bias"],
                "H_NAV_DIRECTION_BIAS": result["overall"]["h_nav_direction_bias"],
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
