#!/usr/bin/env python3
"""Analyze all-seed terminal miss components and protocol/eta discordance.

The script consumes the four Goal38 formal trace directories. It never loads a
policy, steps an environment, trains, or reads held-out seeds.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA_VERSION = "auv_pc_goal40_terminal_component_diagnostic.v1"
SCENARIOS = ("default", "strict125")
MODES = ("baseline", "candidate")
EXPECTED_SEEDS = tuple(range(20, 60))
EXPECTED_CHECKPOINT_SHA256 = (
    "0851496b15ffebbc0a19bafa5bf40c0493c0d9a3f485267b75fde5f811dde5d9"
)
GOAL_RADIUS_M = {"default": 2.0, "strict125": 1.25}
MIN_PLANE_MISS_SEEDS = 5
MIN_PROTOCOL_SUCCESS_SEEDS = 10
VERTICAL_DOMINANT_FRACTION_MIN = 0.60
PROTOCOL_SUCCESS_WITHOUT_ETA_FRACTION_MIN = 0.30
EPS = 1e-12

CSV_FIELDS = (
    "scenario",
    "mode",
    "seed",
    "success",
    "goal_capture_eta",
    "goal_plane_miss",
    "longitudinal_m",
    "horizontal_m",
    "vertical_m",
    "abs_horizontal_over_radius",
    "abs_vertical_over_radius",
    "dominant_component",
    "protocol_success_without_eta_capture",
    "eta_capture_without_protocol_success",
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


def _decompose(
    offset: np.ndarray, goal_tangent: np.ndarray
) -> tuple[float, float, float]:
    offset_array = np.asarray(offset, dtype=np.float64)
    tangent = np.asarray(goal_tangent, dtype=np.float64)
    if offset_array.shape != (3,) or tangent.shape != (3,):
        raise ValueError("offset and goal tangent must be 3-vectors")
    if not np.all(np.isfinite(offset_array)) or not np.all(np.isfinite(tangent)):
        raise ValueError("offset and goal tangent must be finite")
    norm = float(np.linalg.norm(tangent))
    if norm <= EPS:
        raise ValueError("goal tangent must be non-zero")
    unit = tangent / norm
    longitudinal = float(np.dot(offset_array, unit))
    normal = offset_array - longitudinal * unit
    horizontal = float(np.hypot(normal[0], normal[1]))
    vertical = float(normal[2])
    if not all(math.isfinite(value) for value in (longitudinal, horizontal, vertical)):
        raise ValueError("decomposition produced a non-finite value")
    return longitudinal, horizontal, vertical


def _dominant_component(horizontal: float, vertical: float) -> str:
    return "horizontal" if abs(horizontal) >= abs(vertical) else "vertical"


def _last_row_indices(trace: Any) -> dict[int, int]:
    required = ("episode_seed", "step", "time_s", "eta", "goal_ned_m", "goal_tangent_ned")
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
) -> dict[str, Any]:
    eta = np.asarray(trace["eta"][row_index], dtype=np.float64)
    goal = np.asarray(trace["goal_ned_m"][row_index], dtype=np.float64)
    tangent = np.asarray(trace["goal_tangent_ned"][row_index], dtype=np.float64)
    longitudinal, horizontal, vertical = _decompose(eta[:3] - goal, tangent)
    radius = GOAL_RADIUS_M[scenario]
    success = bool(episode["success"])
    eta_capture = bool(episode["goal_capture_eta"])
    plane_miss = bool(episode["goal_plane_miss"])
    return {
        "scenario": scenario,
        "mode": episode["mode"],
        "seed": seed,
        "success": success,
        "goal_capture_eta": eta_capture,
        "goal_plane_miss": plane_miss,
        "longitudinal_m": longitudinal,
        "horizontal_m": horizontal,
        "vertical_m": vertical,
        "abs_horizontal_over_radius": abs(horizontal) / radius,
        "abs_vertical_over_radius": abs(vertical) / radius,
        "dominant_component": _dominant_component(horizontal, vertical),
        "protocol_success_without_eta_capture": success and not eta_capture,
        "eta_capture_without_protocol_success": eta_capture and not success,
    }


def _median(values: list[float]) -> float | None:
    return float(np.median(np.asarray(values, dtype=np.float64))) if values else None


def _aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    plane_misses = [row for row in records if row["goal_plane_miss"]]
    successes = [row for row in records if row["success"]]
    if not plane_misses:
        raise ValueError("block has no plane-miss episodes")
    if not successes:
        raise ValueError("block has no protocol-success episodes")
    vertical_dominant = sum(row["dominant_component"] == "vertical" for row in plane_misses)
    success_without_eta = sum(
        bool(row["protocol_success_without_eta_capture"]) for row in successes
    )
    eta_without_success = sum(
        bool(row["eta_capture_without_protocol_success"]) for row in records
    )
    return {
        "seed_count": len(records),
        "plane_miss_count": len(plane_misses),
        "protocol_success_count": len(successes),
        "plane_miss_vertical_dominant_count": vertical_dominant,
        "plane_miss_vertical_dominant_fraction": vertical_dominant / len(plane_misses),
        "plane_miss_horizontal_dominant_fraction": 1.0
        - vertical_dominant / len(plane_misses),
        "plane_miss_abs_horizontal_over_radius_median": _median(
            [row["abs_horizontal_over_radius"] for row in plane_misses]
        ),
        "plane_miss_abs_vertical_over_radius_median": _median(
            [row["abs_vertical_over_radius"] for row in plane_misses]
        ),
        "protocol_success_without_eta_capture_count": success_without_eta,
        "protocol_success_without_eta_capture_fraction": success_without_eta
        / len(successes),
        "eta_capture_without_protocol_success_count": eta_without_success,
    }


def _mode_verdict(pass_count: int) -> str:
    if pass_count == 4:
        return "SUPPORTED"
    if pass_count >= 1:
        return "PARTIAL"
    return "FALSIFIED"


def _overall_decision(
    vertical_verdict: str, protocol_verdict: str, invalid: bool
) -> dict[str, str]:
    if invalid:
        return {
            "decision": "INVALID_DIAGNOSTIC",
            "recommendation": "修复输入、seed、checkpoint 或 block 完整性后重跑，不解释机制。",
        }
    if vertical_verdict == "SUPPORTED" and protocol_verdict == "SUPPORTED":
        return {
            "decision": "MULTI_FACTOR",
            "recommendation": "先预注册 read-only p_nav-vs-eta nav error 分解；不直接改控制器。",
        }
    if vertical_verdict == "SUPPORTED":
        return {
            "decision": "VERTICAL_COMPONENT_SUPPORTED",
            "recommendation": "只允许预注册全 seed 的 vertical reference/state 单因素诊断。",
        }
    if protocol_verdict == "SUPPORTED":
        return {
            "decision": "PROTOCOL_ETA_DISCORDANCE_SUPPORTED",
            "recommendation": "只允许预注册 terminal estimator/protocol 审计，不训练。",
        }
    if "PARTIAL" in (vertical_verdict, protocol_verdict):
        return {
            "decision": "PARTIAL_ONLY",
            "recommendation": "只补最便宜的数据/分组验证，不训练。",
        }
    return {
        "decision": "NO_SUPPORTED_COMPONENT_SIGNAL",
        "recommendation": "停止全 seed 终端分量线，换新假设。",
    }


def _load_block(
    formal_root: Path, scenario: str, mode: str
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    directory = formal_root / scenario / mode
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    seeds = sorted(int(seed) for seed in summary["seeds"])
    if seeds != list(EXPECTED_SEEDS):
        raise ValueError(f"{scenario}/{mode}: seeds mismatch: {seeds}")
    if summary.get("checkpoint_sha256") != EXPECTED_CHECKPOINT_SHA256:
        raise ValueError(f"{scenario}/{mode}: checkpoint SHA256 mismatch")

    episodes = {int(item["seed"]): item for item in summary["episodes"]}
    if sorted(episodes) != list(EXPECTED_SEEDS):
        raise ValueError(f"{scenario}/{mode}: episode seeds mismatch")
    for item in episodes.values():
        for key in ("success", "goal_capture_eta", "goal_plane_miss"):
            if key not in item:
                raise ValueError(f"{scenario}/{mode}: episode missing {key}")

    with np.load(directory / "trace.npz", allow_pickle=False) as trace:
        last_indices = _last_row_indices(trace)
        records = []
        for seed in EXPECTED_SEEDS:
            episode = dict(episodes[seed])
            episode["mode"] = mode
            record = _record_for_seed(
                seed, last_indices[seed], trace, episode, scenario
            )
            for key in (
                "longitudinal_m",
                "horizontal_m",
                "vertical_m",
                "abs_horizontal_over_radius",
                "abs_vertical_over_radius",
            ):
                if not math.isfinite(float(record[key])):
                    raise ValueError(f"{scenario}/{mode}: non-finite {key} for seed {seed}")
            records.append(record)

    aggregate = _aggregate(records)
    if aggregate["plane_miss_count"] < MIN_PLANE_MISS_SEEDS:
        raise ValueError(
            f"{scenario}/{mode}: only {aggregate['plane_miss_count']} plane misses"
        )
    if aggregate["protocol_success_count"] < MIN_PROTOCOL_SUCCESS_SEEDS:
        raise ValueError(
            f"{scenario}/{mode}: only {aggregate['protocol_success_count']} protocol successes"
        )
    block = {
        "scenario": scenario,
        "mode": mode,
        "source": str(directory),
        "checkpoint_sha256": summary.get("checkpoint_sha256"),
        "records": records,
        "aggregate": aggregate,
        "h_vertical_concentration": {
            "pass": aggregate["plane_miss_vertical_dominant_fraction"]
            >= VERTICAL_DOMINANT_FRACTION_MIN - EPS,
        },
        "h_protocol_eta_discordance": {
            "pass": aggregate["protocol_success_without_eta_capture_fraction"]
            >= PROTOCOL_SUCCESS_WITHOUT_ETA_FRACTION_MIN - EPS,
        },
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
        vertical_blocks = protocol_blocks = 0
    else:
        vertical_blocks = sum(
            block_by_key[(scenario, mode)]["h_vertical_concentration"]["pass"]
            for scenario in SCENARIOS
            for mode in MODES
        )
        protocol_blocks = sum(
            block_by_key[(scenario, mode)]["h_protocol_eta_discordance"]["pass"]
            for scenario in SCENARIOS
            for mode in MODES
        )
    vertical_verdict = _mode_verdict(vertical_blocks)
    protocol_verdict = _mode_verdict(protocol_blocks)
    overall = _overall_decision(vertical_verdict, protocol_verdict, invalid)
    return {
        "schema_version": SCHEMA_VERSION,
        "validation": {
            "formal_root": str(formal_root),
            "expected_seeds": list(EXPECTED_SEEDS),
            "expected_checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
            "heldout_opened": False,
            "block_count": len(all_blocks),
            "invalid": invalid,
            "invalid_reasons": invalid_reasons,
            "thresholds": {
                "min_plane_miss_seeds": MIN_PLANE_MISS_SEEDS,
                "min_protocol_success_seeds": MIN_PROTOCOL_SUCCESS_SEEDS,
                "vertical_dominant_fraction_min": VERTICAL_DOMINANT_FRACTION_MIN,
                "protocol_success_without_eta_fraction_min": (
                    PROTOCOL_SUCCESS_WITHOUT_ETA_FRACTION_MIN
                ),
            },
        },
        "scenarios": scenarios,
        "overall": {
            **overall,
            "h_vertical_concentration": vertical_verdict,
            "h_protocol_eta_discordance": protocol_verdict,
            "vertical_passing_blocks": vertical_blocks,
            "protocol_eta_passing_blocks": protocol_blocks,
            "block_count": len(all_blocks),
        },
        "records": all_records,
    }


def _fmt(value: float | None, digits: int = 4) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def _markdown(result: dict[str, Any]) -> str:
    lines = [
        "# Goal40 全 seed 终端分量与 p_nav/eta 不一致诊断",
        "",
        f"Schema: `{result['schema_version']}`<br>",
        f"总体裁决：**{result['overall']['decision']}**",
        "",
        "## Block results",
        "",
        "| scenario | mode | plane miss n | vertical dominant frac | success-without-eta frac | H-VERTICAL | H-DISCORDANCE |",
        "|---|---|---:|---:|---:|---|---|",
    ]
    for scenario in SCENARIOS:
        for mode in MODES:
            block = result["scenarios"].get(scenario, {}).get(mode)
            if block is None:
                lines.append(f"| {scenario} | {mode} | n/a | n/a | n/a | n/a | n/a |")
                continue
            agg = block["aggregate"]
            lines.append(
                f"| {scenario} | {mode} | {agg['plane_miss_count']} | "
                f"{agg['plane_miss_vertical_dominant_fraction']:.4f} | "
                f"{agg['protocol_success_without_eta_capture_fraction']:.4f} | "
                f"{'PASS' if block['h_vertical_concentration']['pass'] else 'FAIL'} | "
                f"{'PASS' if block['h_protocol_eta_discordance']['pass'] else 'FAIL'} |"
            )
    lines.extend(
        [
            "",
            "## Overall",
            "",
            f"- H-VERTICAL-CONCENTRATION: `{result['overall']['h_vertical_concentration']}`",
            f"- H-PNAV-ETA-DISCORDANCE: `{result['overall']['h_protocol_eta_discordance']}`",
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
    (output / "terminal_component_diagnostic.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "terminal_component_diagnostic.md").write_text(
        _markdown(result), encoding="utf-8"
    )
    with (output / "terminal_component_per_seed.csv").open(
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
                "H_VERTICAL_CONCENTRATION": result["overall"][
                    "h_vertical_concentration"
                ],
                "H_PNAV_ETA_DISCORDANCE": result["overall"][
                    "h_protocol_eta_discordance"
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
