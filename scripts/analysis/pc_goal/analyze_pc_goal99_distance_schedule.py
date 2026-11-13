#!/usr/bin/env python3
"""Audit Goal99's distance-scheduled residual scale screen.

The script is read-only: it consumes Goal98 constant-scale results and Goal99
schedule summaries/traces.  It never loads a policy, steps an environment,
trains, changes success semantics, or reads held-out seeds.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA_VERSION = "auv_pc_goal99_distance_schedule.v1"
ARMS = ("P_D3", "P_BC15")
SCENARIOS = ("default", "strict125")
SCHEDULES = {
    "S20": {"near": 1.50, "far": 2.00, "switch_distance_m": 2.0},
    "S25": {"near": 1.50, "far": 2.00, "switch_distance_m": 2.5},
    "S20-N175": {"near": 1.75, "far": 2.00, "switch_distance_m": 2.0},
}
CONSTANT_SCALES = (1.50, 2.00)
FROZEN_SUCCESS = {"default": 31, "strict125": 20}
EXPECTED_CONSTANT_SUCCESS = {
    (1.50, "P_D3", "default"): 30,
    (1.50, "P_D3", "strict125"): 19,
    (1.50, "P_BC15", "default"): 30,
    (1.50, "P_BC15", "strict125"): 19,
    (2.00, "P_D3", "default"): 31,
    (2.00, "P_D3", "strict125"): 18,
    (2.00, "P_BC15", "default"): 32,
    (2.00, "P_BC15", "strict125"): 18,
}
EXPECTED_CHECKPOINT_SHA = {
    "P_D3": "2d57745a811268553041c561f2ae1ad590dced472b712efd90cdd9c011c1a616",
    "P_BC15": "1a6a153d0d2e9548c2e14ebbe1b0fd33b65188a51be5958565b59885f494fe43",
}
GOAL98_RUN_RE = re.compile(r"^(P_D3|P_BC15)_(default|strict125)_s(\d{3})$")
GOAL99_RUN_RE = re.compile(
    r"^(P_D3|P_BC15)_(default|strict125)_(S20|S25|S20-N175)$"
)


class AuditError(ValueError):
    """Raised when a run is incomplete or violates the frozen protocol."""


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("artifacts/lowlevel/goal99_distance_schedule_20261111"),
    )
    parser.add_argument(
        "--goal98-root",
        type=Path,
        default=Path("artifacts/lowlevel/goal98_residual_scale_grid_20261111"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "artifacts/lowlevel/goal99_distance_schedule_20261111/phase0_summary.json"
        ),
    )
    parser.add_argument(
        "--markdown",
        type=Path,
        default=Path(
            "artifacts/lowlevel/goal99_distance_schedule_20261111/phase0_summary.md"
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


def _episode_outcomes(
    path: Path, episodes: list[dict[str, Any]], *, expected_seeds: set[int]
) -> dict[str, Any]:
    if not isinstance(episodes, list) or len(episodes) != 40:
        raise AuditError(f"{path}: expected 40 dev episodes")
    by_seed: dict[int, bool] = {}
    for row in episodes:
        if not isinstance(row, dict) or "seed" not in row or "success" not in row:
            raise AuditError(f"{path}: malformed episode row")
        by_seed[int(row["seed"])] = bool(row["success"])
    if set(by_seed) != expected_seeds:
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


def _validate_common(
    path: Path, summary: dict[str, Any], *, arm: str
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    expected_sha = EXPECTED_CHECKPOINT_SHA[arm]
    actual_sha = summary.get("checkpoint_sha256")
    if actual_sha != expected_sha:
        raise AuditError(f"{path}: checkpoint SHA {actual_sha!r} != {expected_sha}")
    episodes = summary.get("episodes")
    if not isinstance(episodes, list):
        raise AuditError(f"{path}: missing episode list")
    metrics = summary.get("metrics")
    if not isinstance(metrics, dict):
        raise AuditError(f"{path}: missing metrics object")
    return episodes, metrics


def _load_goal98_constant(
    root: Path,
    *,
    arm: str,
    scenario: str,
    scale: float,
    expected_seeds: set[int],
) -> dict[str, Any]:
    run_dir = root / f"{arm}_{scenario}_s{int(round(scale * 100)):03d}"
    path = run_dir / "summary.json"
    if not path.is_file():
        raise AuditError(f"{path}: missing Goal98 constant-scale summary")
    summary = _read_json(path)
    recorded_scale = summary.get("actuator_residual_scale_override")
    if recorded_scale is None or abs(float(recorded_scale) - scale) > 1e-12:
        raise AuditError(f"{path}: scale {recorded_scale!r} != constant {scale:.2f}")
    episodes, metrics = _validate_common(path, summary, arm=arm)
    record = _episode_outcomes(path, episodes, expected_seeds=expected_seeds)
    record.update(
        {
            "path_three_d_rmse_m": float(metrics.get("path_three_d_rmse_m", math.nan)),
            "goal_plane_miss_rate": float(metrics.get("goal_plane_miss_rate", math.nan)),
        }
    )
    expected = EXPECTED_CONSTANT_SUCCESS[(scale, arm, scenario)]
    if record["success"] != expected:
        raise AuditError(
            f"{path}: {record['success']}/40 does not reproduce Goal98 {expected}/40"
        )
    return record


def _load_goal98_constants(
    root: Path, *, expected_seeds: set[int]
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for arm in ARMS:
        for scenario in SCENARIOS:
            group = f"{arm}_{scenario}"
            result[group] = {
                scale: _load_goal98_constant(
                    root,
                    arm=arm,
                    scenario=scenario,
                    scale=scale,
                    expected_seeds=expected_seeds,
                )
                for scale in CONSTANT_SCALES
            }
    return result


def _validate_schedule_trace(path: Path, spec: dict[str, float]) -> dict[str, float]:
    if not path.is_file():
        raise AuditError(f"{path}: missing trace")
    required = (
        "step",
        "episode_seed",
        "goal_distance_eta_m",
        "terminal_kind",
        "actuator_residual_applied",
        "actuator_residual_scale",
        "actuator_residual_schedule_distance_m",
    )
    with np.load(path) as data:
        missing = [name for name in required if name not in data]
        if missing:
            raise AuditError(f"{path}: missing trace fields {missing}")
        for name in required:
            array = np.asarray(data[name])
            if array.size == 0:
                raise AuditError(f"{path}: empty trace field {name}")
            if np.issubdtype(array.dtype, np.number) and not np.all(
                np.isfinite(array)
            ):
                raise AuditError(f"{path}: non-finite trace field {name}")
        scale = np.asarray(data["actuator_residual_scale"], dtype=np.float64)
        distance = np.asarray(
            data["actuator_residual_schedule_distance_m"], dtype=np.float64
        )
        if scale.shape != distance.shape:
            raise AuditError(f"{path}: scale/distance shape mismatch")
        expected = np.where(
            distance > spec["switch_distance_m"], spec["far"], spec["near"]
        )
        if not np.allclose(scale, expected, rtol=0.0, atol=1e-12):
            raise AuditError(f"{path}: trace scale does not match schedule boundary")
        return {
            "far_fraction": float(np.mean(scale == spec["far"])),
            "near_fraction": float(np.mean(scale == spec["near"])),
            "steps": int(scale.size),
        }


def _load_schedule_run(
    run_dir: Path,
    *,
    arm: str,
    scenario: str,
    schedule_id: str,
    spec: dict[str, float],
    expected_seeds: set[int],
) -> dict[str, Any]:
    path = run_dir / "summary.json"
    if not path.is_file():
        raise AuditError(f"{path}: missing schedule summary")
    summary = _read_json(path)
    expected_fields = {
        "actuator_residual_scale_near": spec["near"],
        "actuator_residual_scale_far": spec["far"],
        "actuator_residual_scale_switch_distance_m": spec["switch_distance_m"],
    }
    for name, expected in expected_fields.items():
        recorded = summary.get(name)
        if recorded is None or abs(float(recorded) - expected) > 1e-12:
            raise AuditError(f"{path}: {name} {recorded!r} != {expected}")
    if summary.get("actuator_residual_scale_schedule_enabled") is not True:
        raise AuditError(f"{path}: schedule not marked enabled")
    if summary.get("actuator_residual_scale_override") is not None:
        raise AuditError(f"{path}: constant scale override leaked into schedule run")
    episodes, metrics = _validate_common(path, summary, arm=arm)
    record = _episode_outcomes(path, episodes, expected_seeds=expected_seeds)
    record.update(
        {
            "run": run_dir.name,
            "arm": arm,
            "scenario": scenario,
            "schedule_id": schedule_id,
            "path_three_d_rmse_m": float(metrics.get("path_three_d_rmse_m", math.nan)),
            "goal_plane_miss_rate": float(metrics.get("goal_plane_miss_rate", math.nan)),
            "residual_magnitude_clip_rate": float(
                metrics.get("actuator_residual_magnitude_clip_rate", math.nan)
            ),
            "residual_rate_clip_rate": float(
                metrics.get("actuator_residual_rate_clip_rate", math.nan)
            ),
        }
    )
    record.update(_validate_schedule_trace(run_dir / "trace.npz", spec))
    return record


def _scan_schedule_runs(root: Path) -> dict[str, dict[str, Path]]:
    if not root.is_dir():
        raise AuditError(f"{root}: missing Goal99 root")
    result: dict[str, dict[str, Path]] = {}
    for run_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        match = GOAL99_RUN_RE.fullmatch(run_dir.name)
        if match is None:
            continue
        arm, scenario, schedule_id = match.groups()
        result.setdefault(schedule_id, {})[f"{arm}_{scenario}"] = run_dir
    expected_groups = {f"{arm}_{scenario}" for arm in ARMS for scenario in SCENARIOS}
    if set(result) != set(SCHEDULES):
        raise AuditError(f"missing schedules: {sorted(set(SCHEDULES) - set(result))}")
    for schedule_id, runs in result.items():
        missing = sorted(expected_groups - set(runs))
        if missing:
            raise AuditError(f"{schedule_id}: missing groups {missing}")
    return result


def _transition(
    baseline: dict[int, bool], other: dict[int, bool]
) -> dict[str, Any]:
    gained = sorted(seed for seed, value in other.items() if value and not baseline[seed])
    lost = sorted(seed for seed, value in baseline.items() if value and not other[seed])
    return {"gained": gained, "lost": lost, "net": len(gained) - len(lost)}


def _path_ratio(record: dict[str, Any], baseline: dict[str, Any]) -> float | None:
    value = record["path_three_d_rmse_m"]
    reference = baseline["path_three_d_rmse_m"]
    if not math.isfinite(value) or not math.isfinite(reference) or reference <= 0.0:
        return None
    return value / reference


def _summarize_schedule(
    schedule_id: str,
    runs: dict[str, Path],
    constants: dict[str, dict[str, Any]],
    *,
    expected_seeds: set[int],
) -> dict[str, Any]:
    spec = SCHEDULES[schedule_id]
    groups: list[dict[str, Any]] = []
    for arm in ARMS:
        for scenario in SCENARIOS:
            group = f"{arm}_{scenario}"
            record = _load_schedule_run(
                runs[group],
                arm=arm,
                scenario=scenario,
                schedule_id=schedule_id,
                spec=spec,
                expected_seeds=expected_seeds,
            )
            baseline = constants[group][1.50]
            constant2 = constants[group][2.00]
            groups.append(
                {
                    "group": group,
                    "success": record["success"],
                    "success_pnav": record["success_pnav"],
                    "success_eta_ever": record["success_eta_ever"],
                    "success_eta_done": record["success_eta_done"],
                    "timeout": record["timeout"],
                    "safety": record["safety"],
                    "path_rmse_ratio_vs_1.5": _path_ratio(record, baseline),
                    "goal_plane_miss_rate": record["goal_plane_miss_rate"],
                    "far_fraction": record["far_fraction"],
                    "near_fraction": record["near_fraction"],
                    "transition_vs_1.5": _transition(
                        baseline["success_by_seed"], record["success_by_seed"]
                    ),
                    "transition_vs_2.0": _transition(
                        constant2["success_by_seed"], record["success_by_seed"]
                    ),
                }
            )
    return {"schedule_id": schedule_id, "definition": spec, "groups": groups}


def _gate(
    roots: Path,
    goal98_root: Path,
    *,
    expected_seeds: set[int],
) -> dict[str, Any]:
    constants = _load_goal98_constants(goal98_root, expected_seeds=expected_seeds)
    runs = _scan_schedule_runs(roots)
    schedules = {
        schedule_id: _summarize_schedule(
            schedule_id,
            runs[schedule_id],
            constants,
            expected_seeds=expected_seeds,
        )
        for schedule_id in SCHEDULES
    }
    supported: list[dict[str, Any]] = []
    partial: list[dict[str, Any]] = []
    geometry_candidates: list[dict[str, Any]] = []
    for schedule_id, summary in schedules.items():
        rows = summary["groups"]
        success_ok = all(
            row["success"] >= FROZEN_SUCCESS[row["group"].rsplit("_", 1)[1]]
            for row in rows
        )
        path_ratios = [
            row["path_rmse_ratio_vs_1.5"]
            for row in rows
            if row["path_rmse_ratio_vs_1.5"] is not None
        ]
        path_ok = len(path_ratios) == len(rows) and float(
            sum(path_ratios) / len(path_ratios)
        ) <= 1.05
        risk_ok = all(row["timeout"] == 0 and row["safety"] == 0 for row in rows)
        improved = any(
            row["success"] > max(
                constants[row["group"]][1.50]["success"],
                constants[row["group"]][2.00]["success"],
            )
            for row in rows
        )
        strict_rows = [
            row for row in rows if row["group"].endswith("_strict125")
        ]
        strict_plane_miss = float(
            sum(row["goal_plane_miss_rate"] for row in strict_rows)
            / len(strict_rows)
        )
        record = {
            "schedule_id": schedule_id,
            "success_ok": success_ok,
            "path_ok": path_ok,
            "risk_ok": risk_ok,
            "improved_vs_constants": improved,
            "success": {row["group"]: row["success"] for row in rows},
            "mean_path_rmse_ratio": (
                float(sum(path_ratios) / len(path_ratios)) if path_ratios else None
            ),
            "strict_plane_miss_rate_mean": strict_plane_miss,
        }
        if success_ok and path_ok and risk_ok:
            supported.append(record)
        elif improved:
            partial.append(record)
        if not success_ok and improved and strict_plane_miss >= 0.5:
            geometry_candidates.append(record)

    if supported:
        status = "SCHEDULE_SUPPORTED"
        qualifying = supported[0]["schedule_id"]
    elif partial:
        status = "SCHEDULE_PARTIAL"
        qualifying = partial[0]["schedule_id"]
    else:
        status = "SCHEDULE_NOT_SUPPORTED"
        qualifying = None
    return {
        "status": status,
        "qualifying_schedule": qualifying,
        "supported": supported,
        "partial": partial,
        "geometry_limited_candidates": geometry_candidates,
        "frozen_success": FROZEN_SUCCESS,
        "constant_success": {
            f"{scale:.2f}": {
                group: constants[group][scale]["success"] for group in constants
            }
            for scale in CONSTANT_SCALES
        },
        "schedules": schedules,
        "baseline_failures": [],
    }


def _fmt(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def write_markdown(payload: dict[str, Any], path: Path) -> None:
    lines = [
        "# Goal99：距离调度的 PPO residual scale",
        "",
        "本文件由 `scripts/analyze_pc_goal99_distance_schedule.py` 生成；只读 summary/trace，不训练。",
        "",
        f"- Gate: `{payload['gate']['status']}`",
        f"- qualifying schedule: `{payload['gate']['qualifying_schedule']}`",
        f"- frozen success: `{payload['gate']['frozen_success']}`",
        "",
    ]
    if payload["gate"].get("reason"):
        lines.extend(["## INVALID 原因", "", str(payload["gate"]["reason"]), ""])
    schedules = payload["gate"].get("schedules", {})
    if isinstance(schedules, dict):
        schedules = schedules.values()
    for summary in schedules:
        definition = summary["definition"]
        lines.extend(
            [
                f"## {summary['schedule_id']}",
                "",
                f"near={definition['near']:.2f}, far={definition['far']:.2f}, "
                f"switch={definition['switch_distance_m']:.2f} m",
                "",
                "| group | success | pnav | eta-ever | path ratio | far % | strict plane miss | gained vs 1.5 | lost vs 1.5 |",
                "|---|---:|---:|---:|---:|---:|---:|---|---|",
            ]
        )
        for row in summary["groups"]:
            lines.append(
                f"| {row['group']} | {row['success']}/40 | {row['success_pnav']} | "
                f"{row['success_eta_ever']} | {_fmt(row['path_rmse_ratio_vs_1.5'])} | "
                f"{100.0 * row['far_fraction']:.1f}% | {_fmt(row['goal_plane_miss_rate'])} | "
                f"{row['transition_vs_1.5']['gained']} | {row['transition_vs_1.5']['lost']} |"
            )
        lines.append("")
    lines.extend(["## Gate 明细", ""])
    for key in ("supported", "partial", "geometry_limited_candidates"):
        for row in payload["gate"][key]:
            lines.append(
                f"- {key} {row['schedule_id']}: success={row['success']}, "
                f"mean_path_rmse_ratio={_fmt(row['mean_path_rmse_ratio'])}, "
                f"strict_plane_miss={_fmt(row['strict_plane_miss_rate_mean'])}, "
                f"success_ok={row['success_ok']}, path_ok={row['path_ok']}, "
                f"risk_ok={row['risk_ok']}"
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    expected_seeds = set(range(20, 60))
    try:
        gate = _gate(args.root, args.goal98_root, expected_seeds=expected_seeds)
    except AuditError as exc:
        gate = {
            "status": "INVALID",
            "reason": str(exc),
            "qualifying_schedule": None,
            "supported": [],
            "partial": [],
            "geometry_limited_candidates": [],
            "frozen_success": FROZEN_SUCCESS,
            "constant_success": {},
            "schedules": [],
            "baseline_failures": [],
        }
    payload = {"schema_version": SCHEMA_VERSION, "gate": gate}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    write_markdown(payload, args.markdown)
    print(f"wrote {args.output}")
    print(f"wrote {args.markdown}")
    print(f"gate={gate['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
