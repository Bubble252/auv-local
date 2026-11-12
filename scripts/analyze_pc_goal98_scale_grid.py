#!/usr/bin/env python3
"""Summarize Goal98's corrected PPO residual scale grid.

The script reads dev40 summaries and traces only.  It does not load a policy,
step an environment, train, change success semantics, or read held-out seeds.
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

SCHEMA_VERSION = "auv_pc_goal98_scale_grid.v1"
SCALES = (0.75, 1.00, 1.25, 1.50, 1.75, 2.00)
BASELINE_SCALE = 1.50
TRAINING_SCALE = 1.50
FROZEN_SUCCESS = {"default": 31, "strict125": 20}
EXPECTED_BASELINE_SUCCESS = {"default": 30, "strict125": 19}
ARMS = ("P_D3", "P_BC15")
SCENARIOS = ("default", "strict125")
EXPECTED_CHECKPOINT_SHA = {
    "P_D3": "2d57745a811268553041c561f2ae1ad590dced472b712efd90cdd9c011c1a616",
    "P_BC15": "1a6a153d0d2e9548c2e14ebbe1b0fd33b65188a51be5958565b59885f494fe43",
}
EXPECTED_GROUPS = {f"{arm}_{scenario}" for arm in ARMS for scenario in SCENARIOS}
RUN_RE = re.compile(r"^(P_D3|P_BC15)_(default|strict125)_s(\d{3})$")


class AuditError(ValueError):
    """Raised when the scale grid is incomplete or malformed."""


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("artifacts/lowlevel/goal98_residual_scale_grid_20261111"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "artifacts/lowlevel/goal98_residual_scale_grid_20261111/phase0_summary.json"
        ),
    )
    parser.add_argument(
        "--markdown",
        type=Path,
        default=Path(
            "artifacts/lowlevel/goal98_residual_scale_grid_20261111/phase0_summary.md"
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


def _validate_trace(path: Path) -> None:
    if not path.is_file():
        raise AuditError(f"{path}: missing trace")
    required = (
        "step",
        "episode_seed",
        "goal_distance_eta_m",
        "terminal_kind",
        "actuator_residual_applied",
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


def _run_record(run_dir: Path, scale: float) -> dict[str, Any]:
    path = run_dir / "summary.json"
    summary = _read_json(path)
    episodes = summary.get("episodes")
    if not isinstance(episodes, list) or not episodes:
        raise AuditError(f"{path}: missing episode list")
    if len(episodes) != 40:
        raise AuditError(f"{path}: expected 40 dev episodes, got {len(episodes)}")
    recorded_scale = summary.get("actuator_residual_scale_override")
    if recorded_scale is None or abs(float(recorded_scale) - scale) > 1e-12:
        raise AuditError(
            f"{path}: recorded scale {recorded_scale!r} != expected {scale}"
        )
    match = RUN_RE.fullmatch(run_dir.name)
    if match is None:
        raise AuditError(f"{run_dir.name}: unexpected run directory")
    arm = match.group(1)
    expected_sha = EXPECTED_CHECKPOINT_SHA[arm]
    actual_sha = summary.get("checkpoint_sha256")
    if actual_sha != expected_sha:
        raise AuditError(
            f"{path}: checkpoint SHA {actual_sha!r} != expected {expected_sha}"
        )
    _validate_trace(run_dir / "trace.npz")
    by_seed: dict[int, bool] = {}
    for row in episodes:
        if not isinstance(row, dict) or "seed" not in row or "success" not in row:
            raise AuditError(f"{path}: malformed episode row")
        by_seed[int(row["seed"])] = bool(row["success"])
    metrics = summary.get("metrics", {})
    if not isinstance(metrics, dict):
        raise AuditError(f"{path}: metrics must be an object")
    return {
        "run": run_dir.name,
        "episodes": len(episodes),
        "success": _count_success(episodes),
        "success_by_seed": by_seed,
        "success_pnav": _count_success(episodes, "success_pnav_tight"),
        "success_eta_ever": _count_success(episodes, "success_eta_tight_ever"),
        "success_eta_done": _count_success(episodes, "success_eta_tight_at_done"),
        "timeout": sum(1 for row in episodes if bool(row.get("timeout", False))),
        "safety": sum(1 for row in episodes if bool(row.get("safety", False))),
        "path_three_d_rmse_m": float(metrics.get("path_three_d_rmse_m", math.nan)),
        "goal_plane_miss_rate": float(metrics.get("goal_plane_miss_rate", math.nan)),
        "residual_magnitude_clip_rate": float(
            metrics.get("actuator_residual_magnitude_clip_rate", math.nan)
        ),
        "residual_rate_clip_rate": float(
            metrics.get("actuator_residual_rate_clip_rate", math.nan)
        ),
    }


def _scan(root: Path) -> dict[str, dict[float, Path]]:
    if not root.is_dir():
        raise AuditError(f"{root}: missing scale grid root")
    groups: dict[str, dict[float, Path]] = {}
    for run_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        match = RUN_RE.fullmatch(run_dir.name)
        if match is None:
            continue
        arm, scenario, raw_scale = match.groups()
        scale = int(raw_scale) / 100.0
        if scale not in SCALES:
            raise AuditError(f"{run_dir.name}: unexpected scale {scale}")
        groups.setdefault(f"{arm}_{scenario}", {})[scale] = run_dir
    if set(groups) != EXPECTED_GROUPS:
        raise AuditError(f"missing groups: {sorted(EXPECTED_GROUPS - set(groups))}")
    for group, runs in groups.items():
        missing = sorted(set(SCALES) - set(runs))
        if missing:
            raise AuditError(f"{group}: missing scales {missing}")
    return groups


def _transition(
    baseline: dict[int, bool], other: dict[int, bool]
) -> dict[str, Any]:
    gained = sorted(seed for seed, value in other.items() if value and not baseline[seed])
    lost = sorted(seed for seed, value in baseline.items() if value and not other[seed])
    return {"gained": gained, "lost": lost, "net": len(gained) - len(lost)}


def _summarize_group(name: str, runs: dict[float, Path]) -> dict[str, Any]:
    records = {scale: _run_record(path, scale) for scale, path in runs.items()}
    baseline = records[BASELINE_SCALE]
    rows: list[dict[str, Any]] = []
    for scale in SCALES:
        record = records[scale]
        path_ratio = None
        if (
            math.isfinite(record["path_three_d_rmse_m"])
            and math.isfinite(baseline["path_three_d_rmse_m"])
            and baseline["path_three_d_rmse_m"] > 0.0
        ):
            path_ratio = record["path_three_d_rmse_m"] / baseline["path_three_d_rmse_m"]
        rows.append(
            {
                "scale": scale,
                "success": record["success"],
                "success_pnav": record["success_pnav"],
                "success_eta_ever": record["success_eta_ever"],
                "success_eta_done": record["success_eta_done"],
                "timeout": record["timeout"],
                "safety": record["safety"],
                "path_three_d_rmse_m": record["path_three_d_rmse_m"],
                "path_rmse_ratio_vs_1.5": path_ratio,
                "residual_magnitude_clip_rate": record[
                    "residual_magnitude_clip_rate"
                ],
                "residual_rate_clip_rate": record["residual_rate_clip_rate"],
                "transition_vs_1.5": _transition(
                    baseline["success_by_seed"], record["success_by_seed"]
                ),
            }
        )
    return {"group": name, "rows": rows}


def _gate(groups: dict[str, dict[float, Path]]) -> dict[str, Any]:
    summaries = {name: _summarize_group(name, runs) for name, runs in groups.items()}
    by_group_scale = {
        name: {row["scale"]: row for row in group["rows"]}
        for name, group in summaries.items()
    }
    baseline_failures: list[str] = []
    for name in sorted(EXPECTED_GROUPS):
        scenario = name.rsplit("_", 1)[1]
        baseline = by_group_scale[name][BASELINE_SCALE]
        expected = EXPECTED_BASELINE_SUCCESS[scenario]
        if baseline["success"] != expected:
            baseline_failures.append(
                f"{name}: baseline {baseline['success']}/40 != expected {expected}/40"
            )
    if baseline_failures:
        return {
            "status": "INVALID",
            "reason": "baseline_1.50_does_not_reproduce_goal95",
            "baseline_failures": baseline_failures,
            "qualifying_scale": None,
            "supported": [],
            "partial": [],
            "frozen_success": FROZEN_SUCCESS,
            "baseline_scale": BASELINE_SCALE,
        }

    supported: list[dict[str, Any]] = []
    partial: list[dict[str, Any]] = []
    expected_groups = [f"{arm}_{scenario}" for arm in ARMS for scenario in SCENARIOS]
    for scale in SCALES:
        rows = [by_group_scale[name][scale] for name in expected_groups]
        baselines = [by_group_scale[name][BASELINE_SCALE] for name in expected_groups]
        scenarios = [name.rsplit("_", 1)[1] for name in expected_groups]
        success_ok = all(
            row["success"] >= FROZEN_SUCCESS[scenario]
            for row, scenario in zip(rows, scenarios, strict=True)
        )
        path_ratios = [
            row["path_rmse_ratio_vs_1.5"]
            for row in rows
            if row["path_rmse_ratio_vs_1.5"] is not None
        ]
        path_ok = len(path_ratios) == len(rows) and float(
            sum(path_ratios) / len(path_ratios)
        ) <= 1.05
        risk_ok = all(
            row["timeout"] <= baseline["timeout"] and row["safety"] <= baseline["safety"]
            for row, baseline in zip(rows, baselines, strict=True)
        )
        improved = any(
            row["success"] > baseline["success"]
            for row, baseline in zip(rows, baselines, strict=True)
        )
        record = {
            "scale": scale,
            "success_ok": success_ok,
            "path_ok": path_ok,
            "risk_ok": risk_ok,
            "success": {
                name: row["success"] for name, row in zip(expected_groups, rows, strict=True)
            },
            "mean_path_rmse_ratio": (
                float(sum(path_ratios) / len(path_ratios)) if path_ratios else None
            ),
        }
        if success_ok and path_ok and risk_ok:
            supported.append(record)
        elif success_ok or improved:
            partial.append(record)

    if supported:
        status = "SCALE_SUPPORTED"
        qualifying_scale = supported[0]["scale"]
    elif partial:
        status = "SCALE_PARTIAL"
        qualifying_scale = partial[0]["scale"]
    else:
        status = "SCALE_NOT_SUPPORTED"
        qualifying_scale = None
    return {
        "status": status,
        "qualifying_scale": qualifying_scale,
        "supported": supported,
        "partial": partial,
        "baseline_failures": [],
        "frozen_success": FROZEN_SUCCESS,
        "baseline_scale": BASELINE_SCALE,
    }


def _fmt(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def write_markdown(payload: dict[str, Any], path: Path) -> None:
    lines = [
        "# Goal98 Phase 0：修正锚点后的 PPO residual scale grid",
        "",
        "本文件由 `scripts/analyze_pc_goal98_scale_grid.py` 生成；只读 evaluation summary/trace，不训练。",
        "",
        f"- Gate: `{payload['gate']['status']}`",
        f"- qualifying scale: `{payload['gate']['qualifying_scale']}`",
        f"- baseline scale: `{payload['gate']['baseline_scale']}`",
        "",
    ]
    if payload["gate"].get("baseline_failures"):
        lines.append("## INVALID 原因")
        lines.append("")
        for failure in payload["gate"]["baseline_failures"]:
            lines.append(f"- {failure}")
        lines.append("")
    for group in payload["groups"]:
        lines.append(f"## {group['group']}")
        lines.append("")
        lines.append(
            "| scale | success | pnav | eta-ever | path RMSE ratio | timeout | safety | gained | lost |"
        )
        lines.append("|---:|---:|---:|---:|---:|---:|---:|---|---|")
        for row in group["rows"]:
            lines.append(
                f"| {row['scale']:.2f} | {row['success']}/40 | {row['success_pnav']} | "
                f"{row['success_eta_ever']} | {_fmt(row['path_rmse_ratio_vs_1.5'])} | "
                f"{row['timeout']} | {row['safety']} | "
                f"{row['transition_vs_1.5']['gained']} | {row['transition_vs_1.5']['lost']} |"
            )
        lines.append("")
    lines.append("## Gate 明细")
    lines.append("")
    for key in ("supported", "partial"):
        for row in payload["gate"][key]:
            lines.append(
                f"- {key} scale={row['scale']}: success={row['success']}, "
                f"mean_path_rmse_ratio={_fmt(row['mean_path_rmse_ratio'])}, "
                f"success_ok={row['success_ok']}, path_ok={row['path_ok']}, risk_ok={row['risk_ok']}"
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        groups = _scan(args.root)
        payload = {
            "schema_version": SCHEMA_VERSION,
            "scales": list(SCALES),
            "groups": [
                _summarize_group(name, runs) for name, runs in sorted(groups.items())
            ],
            "gate": _gate(groups),
        }
    except AuditError as exc:
        payload = {
            "schema_version": SCHEMA_VERSION,
            "scales": list(SCALES),
            "groups": [],
            "gate": {
                "status": "INVALID",
                "reason": str(exc),
                "qualifying_scale": None,
                "supported": [],
                "partial": [],
                "baseline_failures": [],
                "frozen_success": FROZEN_SUCCESS,
                "baseline_scale": BASELINE_SCALE,
            },
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    write_markdown(payload, args.markdown)
    print(f"wrote {args.output}")
    print(f"wrote {args.markdown}")
    print(f"gate={payload['gate']['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
