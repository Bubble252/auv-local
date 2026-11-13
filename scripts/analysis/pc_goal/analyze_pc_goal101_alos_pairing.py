#!/usr/bin/env python3
"""Audit Goal101's same-checkpoint ALOS / BC-only pairing.

The script is read-only with respect to rollouts and policies: it consumes
Goal100 B20 summaries and Goal101 ALOS summaries, validates the frozen
protocol, and applies the pre-registered Gate.
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "auv_pc_goal101_alos_pairing.v1"
ARMS = ("P_D3", "P_BC15")
SCENARIOS = ("default", "strict125")
EXPECTED_SEEDS = set(range(20, 60))
EXPECTED_B20_SUCCESS = {
    ("P_D3", "default"): 33,
    ("P_BC15", "default"): 33,
    ("P_D3", "strict125"): 20,
    ("P_BC15", "strict125"): 20,
}
EXPECTED_CHECKPOINT_SHA = {
    "P_D3": "2d57745a811268553041c561f2ae1ad590dced472b712efd90cdd9c011c1a616",
    "P_BC15": "1a6a153d0d2e9548c2e14ebbe1b0fd33b65188a51be5958565b59885f494fe43",
}


class AuditError(ValueError):
    """Raised when a run is incomplete or violates the frozen protocol."""


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--alos-root",
        type=Path,
        default=Path("artifacts/lowlevel/goal101_alos_pairing_20261111"),
    )
    parser.add_argument(
        "--goal100-root",
        type=Path,
        default=Path("artifacts/lowlevel/goal100_terminal_prior_20261111"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "artifacts/lowlevel/goal101_alos_pairing_20261111/phase0_summary.json"
        ),
    )
    parser.add_argument(
        "--markdown",
        type=Path,
        default=Path(
            "artifacts/lowlevel/goal101_alos_pairing_20261111/phase0_summary.md"
        ),
    )
    return parser.parse_args(argv)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise AuditError(f"{path}: expected a JSON object")
    return payload


def _count_success(episodes: list[dict[str, Any]], key: str = "success") -> int:
    return sum(1 for row in episodes if bool(row.get(key, False)))


def _episode_outcomes(path: Path, summary: dict[str, Any]) -> dict[str, Any]:
    episodes = summary.get("episodes")
    if not isinstance(episodes, list) or len(episodes) != 40:
        raise AuditError(f"{path}: expected 40 dev episodes")
    by_seed: dict[int, bool] = {}
    for row in episodes:
        if not isinstance(row, dict) or "seed" not in row or "success" not in row:
            raise AuditError(f"{path}: malformed episode row")
        by_seed[int(row["seed"])] = bool(row["success"])
    if set(by_seed) != EXPECTED_SEEDS:
        raise AuditError(f"{path}: episodes do not cover dev seeds 20-59")
    return {
        "success": _count_success(episodes),
        "success_by_seed": by_seed,
        "success_pnav": _count_success(episodes, "success_pnav_tight"),
        "success_eta_ever": _count_success(episodes, "success_eta_tight_ever"),
        "success_eta_done": _count_success(episodes, "success_eta_tight_at_done"),
        "timeout": sum(1 for row in episodes if bool(row.get("timeout", False))),
        "safety": sum(1 for row in episodes if bool(row.get("safety", False))),
    }


def _validate_common(path: Path, summary: dict[str, Any], *, arm: str) -> None:
    expected_sha = EXPECTED_CHECKPOINT_SHA[arm]
    actual_sha = summary.get("checkpoint_sha256")
    if actual_sha != expected_sha:
        raise AuditError(f"{path}: checkpoint SHA {actual_sha!r} != {expected_sha}")
    recorded_scale = summary.get("actuator_residual_scale_override")
    if recorded_scale is None or abs(float(recorded_scale) - 2.0) > 1e-12:
        raise AuditError(f"{path}: scale override {recorded_scale!r} != 2.0")


def _record(
    path: Path,
    *,
    arm: str,
    scenario: str,
    action_mode: str,
) -> dict[str, Any]:
    summary = _read_json(path)
    if summary.get("action_mode") != action_mode:
        raise AuditError(
            f"{path}: action_mode {summary.get('action_mode')!r} != {action_mode!r}"
        )
    if bool(summary.get("ppo_residual_attenuation_enabled", False)):
        raise AuditError(f"{path}: attenuation must be disabled")
    _validate_common(path, summary, arm=arm)
    episodes = summary.get("episodes")
    metrics = summary.get("metrics")
    if not isinstance(episodes, list) or not isinstance(metrics, dict):
        raise AuditError(f"{path}: missing episodes or metrics")
    record = _episode_outcomes(path, summary)
    record.update(
        {
            "arm": arm,
            "scenario": scenario,
            "action_mode": action_mode,
            "path_three_d_rmse_m": float(metrics.get("path_three_d_rmse_m", math.nan)),
            "goal_plane_miss_rate": float(metrics.get("goal_plane_miss_rate", math.nan)),
            "run": path.parent.name,
        }
    )
    if not math.isfinite(record["path_three_d_rmse_m"]):
        raise AuditError(f"{path}: non-finite path 3D RMSE")
    return record


def _transition(baseline: dict[int, bool], other: dict[int, bool]) -> dict[str, list[int]]:
    gained = sorted(seed for seed, value in other.items() if value and not baseline[seed])
    lost = sorted(seed for seed in baseline if baseline[seed] and not other[seed])
    return {"gained": gained, "lost": lost}


def _load_records(
    *,
    alos_root: Path,
    goal100_root: Path,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    if not alos_root.is_dir():
        raise AuditError(f"{alos_root}: missing Goal101 root")
    if not goal100_root.is_dir():
        raise AuditError(f"{goal100_root}: missing Goal100 root")
    alos: dict[str, dict[str, Any]] = {}
    b20: dict[str, dict[str, Any]] = {}
    for arm in ARMS:
        for scenario in SCENARIOS:
            key = f"{arm}_{scenario}"
            alos_path = alos_root / f"{key}_ALOS" / "summary.json"
            b20_path = goal100_root / f"{key}_B20" / "summary.json"
            if not alos_path.is_file():
                raise AuditError(f"{alos_path}: missing ALOS summary")
            if not b20_path.is_file():
                raise AuditError(f"{b20_path}: missing B20 summary")
            alos[key] = _record(
                alos_path, arm=arm, scenario=scenario, action_mode="alos"
            )
            b20[key] = _record(
                b20_path, arm=arm, scenario=scenario, action_mode="frozen-base"
            )
    return alos, b20


def _arm_invariance(alos: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """ALOS bypasses the policy, so both checkpoint arms must be identical."""

    result: dict[str, Any] = {"identical": True, "groups": {}}
    for scenario in SCENARIOS:
        left = alos[f"P_D3_{scenario}"]
        right = alos[f"P_BC15_{scenario}"]
        path_delta = abs(left["path_three_d_rmse_m"] - right["path_three_d_rmse_m"])
        by_seed_equal = left["success_by_seed"] == right["success_by_seed"]
        identical = path_delta <= 1e-9 and by_seed_equal
        result["groups"][scenario] = {
            "path_rmse_abs_delta_m": path_delta,
            "success_by_seed_equal": by_seed_equal,
            "identical": identical,
        }
        result["identical"] = result["identical"] and identical
    return result


def _gate(
    alos: dict[str, dict[str, Any]], b20: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    groups: dict[str, Any] = {}
    path_ratios: list[float] = []
    timeout_increase = False
    safety_increase = False
    beats = True
    parity = True
    below = False
    b20_reproduces = True

    for arm in ARMS:
        for scenario in SCENARIOS:
            key = f"{arm}_{scenario}"
            baseline = alos[key]
            candidate = b20[key]
            expected = EXPECTED_B20_SUCCESS[(arm, scenario)]
            if candidate["success"] != expected:
                b20_reproduces = False
            ratio = candidate["path_three_d_rmse_m"] / baseline["path_three_d_rmse_m"]
            path_ratios.append(ratio)
            delta = candidate["success"] - baseline["success"]
            timeout_increase = timeout_increase or candidate["timeout"] > baseline["timeout"]
            safety_increase = safety_increase or candidate["safety"] > baseline["safety"]
            beats = beats and delta > 0
            parity = parity and abs(delta) <= 1 and ratio <= 1.05
            below = below or delta < -1
            groups[key] = {
                "alos_success": baseline["success"],
                "b20_success": candidate["success"],
                "success_delta_b20_minus_alos": delta,
                "alos_path_three_d_rmse_m": baseline["path_three_d_rmse_m"],
                "b20_path_three_d_rmse_m": candidate["path_three_d_rmse_m"],
                "path_ratio_b20_over_alos": ratio,
                "alos_timeout": baseline["timeout"],
                "b20_timeout": candidate["timeout"],
                "alos_safety": baseline["safety"],
                "b20_safety": candidate["safety"],
                "transition_b20_vs_alos": _transition(
                    baseline["success_by_seed"], candidate["success_by_seed"]
                ),
            }

    path_ok = max(path_ratios) <= 1.05
    no_resource_increase = not timeout_increase and not safety_increase
    if not b20_reproduces:
        status = "INVALID"
    elif below:
        status = "BC_PRIOR_BELOW_ALOS"
    elif beats and path_ok and no_resource_increase:
        status = "BC_PRIOR_BEATS_ALOS"
    elif parity:
        status = "BC_PRIOR_PARITY"
    else:
        status = "INCONCLUSIVE"
    return {
        "status": status,
        "b20_reproduces": b20_reproduces,
        "beats_condition": beats,
        "parity_condition": parity,
        "below_condition": below,
        "path_within_5pct": path_ok,
        "no_timeout_or_safety_increase": no_resource_increase,
        "path_ratio_max": max(path_ratios),
        "path_ratio_mean": sum(path_ratios) / len(path_ratios),
        "groups": groups,
    }


def _markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Goal101 ALOS / BC-only 配对结果",
        "",
        f"- 裁决：`{payload['gate']['status']}`",
        f"- ALOS 两臂一致：`{payload['arm_invariance']['identical']}`",
        f"- B20 复现 33/20：`{payload['gate']['b20_reproduces']}`",
        f"- path ratio max：`{payload['gate']['path_ratio_max']:.4f}`",
        f"- path ratio mean：`{payload['gate']['path_ratio_mean']:.4f}`",
        "",
        "| group | ALOS | B20 | delta | path ratio | timeout | safety |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for key, row in payload["gate"]["groups"].items():
        lines.append(
            f"| `{key}` | {row['alos_success']}/40 | {row['b20_success']}/40 | "
            f"{row['success_delta_b20_minus_alos']:+d} | "
            f"{row['path_ratio_b20_over_alos']:.4f} | "
            f"{row['alos_timeout']}/{row['b20_timeout']} | "
            f"{row['alos_safety']}/{row['b20_safety']} |"
        )
    lines.extend(
        [
            "",
            "B20 相对 ALOS 的 seed 变化：",
            "",
            "| group | gained | lost |",
            "|---|---|---|",
        ]
    )
    for key, row in payload["gate"]["groups"].items():
        transition = row["transition_b20_vs_alos"]
        lines.append(
            f"| `{key}` | {transition['gained']} | {transition['lost']} |"
        )
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    alos, b20 = _load_records(
        alos_root=args.alos_root,
        goal100_root=args.goal100_root,
    )
    invariance = _arm_invariance(alos)
    if not invariance["identical"]:
        raise AuditError(
            "ALOS arms are not identical; the zero-residual baseline violated "
            "the policy-independence protocol"
        )
    gate = _gate(alos, b20)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "seeds": sorted(EXPECTED_SEEDS),
        "success_frame": "both",
        "arm_invariance": invariance,
        "gate": gate,
        "alos": alos,
        "b20": b20,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.write_text(_markdown(payload), encoding="utf-8")
    print(json.dumps(gate, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
