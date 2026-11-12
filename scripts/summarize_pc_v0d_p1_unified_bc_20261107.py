"""Summarize the frozen P1 unified BC evaluation set."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

BC_ORDER = (
    "g19_dagger",
    "g19_satw",
    "g18_s0",
    "g17_a00",
    "g17_a10",
    "g17_a01",
    "g17_a11",
)

CHECKPOINT_PATHS = {
    "g19_dagger": "artifacts/lowlevel/pc_v0d_r2b_weighted_bc_dagger_20261107/bc_satw_dagger2/pc_bc_actor.pt",
    "g19_satw": "artifacts/lowlevel/pc_v0d_r2b_saturation_bc_20261107/bc_satw/pc_bc_actor.pt",
    "g18_s0": "artifacts/lowlevel/pc_v0d_sign_magnitude_bc_20261107/s0_continuous/pc_bc_actor.pt",
    "g17_a00": "artifacts/lowlevel/pc_v0d_current_crab_factorial_20261107/bc_A00/pc_bc_actor.pt",
    "g17_a10": "artifacts/lowlevel/pc_v0d_current_crab_factorial_20261107/bc_A10/pc_bc_actor.pt",
    "g17_a01": "artifacts/lowlevel/pc_v0d_current_crab_factorial_20261107/bc_A01/pc_bc_actor.pt",
    "g17_a11": "artifacts/lowlevel/pc_v0d_current_crab_factorial_20261107/bc_A11/pc_bc_actor.pt",
}

SOURCE_LABELS = {
    "g19_dagger": "Goal19 targeted-DAgger weighted BC",
    "g19_satw": "Goal19 saturation-weighted BC baseline",
    "g18_s0": "Goal18 continuous-action BC control",
    "g17_a00": "Goal17 factor cell A00: current off, crab off",
    "g17_a10": "Goal17 factor cell A10: current on, crab off",
    "g17_a01": "Goal17 factor cell A01: current off, crab on",
    "g17_a11": "Goal17 factor cell A11: current on, crab on",
}

SCENARIOS = ("default", "strict125", "path_zero")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize P1 unified BC evaluation")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_summary(path: Path, *, path_zero: bool, config_suffix: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("seeds") != list(range(20, 40)):
        raise ValueError(f"{path}: seeds are not dev 20-39")
    if payload.get("path_dim") != 8:
        raise ValueError(f"{path}: path_dim is not 8")
    if payload.get("path_zero") is not path_zero:
        raise ValueError(f"{path}: path_zero mismatch")
    if payload.get("current_ablation") != "none":
        raise ValueError(f"{path}: unexpected current ablation")
    if not str(payload.get("config_path", "")).endswith(config_suffix):
        raise ValueError(f"{path}: unexpected config")
    return payload


def _scenario_record(
    payload: dict[str, Any],
) -> dict[str, float | int | bool | str]:
    student = payload["student"]
    episodes = payload["student_episodes"]
    return {
        "path_three_d_rmse_m": float(student["path_three_d_rmse_m"]),
        "path_horizontal_rmse_m": float(student["path_horizontal_rmse_m"]),
        "path_vertical_rmse_m": float(student["path_vertical_rmse_m"]),
        "success_rate": float(student["success_rate"]),
        "timeout_rate": float(student["timeout_rate"]),
        "safety_termination_rate": float(student["safety_termination_rate"]),
        "material_saturation_rate": float(student["material_saturation_rate"]),
        "speed_rmse_mps": float(student["speed_rmse_mps"]),
        "yaw_rmse_rad": float(student["yaw_rmse_rad"]),
        "path_completion_ratio_mean": float(student["path_completion_ratio_mean"]),
        "episode_count": int(len(episodes)),
        "seed_min": int(min(record["seed"] for record in episodes)),
        "seed_max": int(max(record["seed"] for record in episodes)),
    }


def _fmt(value: float, digits: int = 4) -> str:
    return f"{value:.{digits}f}"


def main() -> int:
    args = _parse_args()
    root = args.root.resolve()
    repo_root = Path(__file__).resolve().parents[1]
    rows: list[dict[str, Any]] = []
    problems: list[str] = []

    for bc in BC_ORDER:
        row: dict[str, Any] = {
            "id": bc,
            "source": SOURCE_LABELS[bc],
            "checkpoint_path": str((repo_root / CHECKPOINT_PATHS[bc]).resolve()),
        }
        checkpoint = (repo_root / CHECKPOINT_PATHS[bc]).resolve()
        row["checkpoint_exists"] = checkpoint.is_file()
        row["checkpoint_sha256"] = _sha256(checkpoint) if checkpoint.is_file() else None
        for scenario in SCENARIOS:
            summary_path = root / bc / scenario / "summary.json"
            if not summary_path.is_file():
                problems.append(f"missing summary: {summary_path}")
                row[scenario] = None
                continue
            config_suffix = (
                "odarrl_control_pc_c3_random_realistic_v0d_teacher_r2b_current_strict125.json"
                if scenario == "strict125"
                else "odarrl_control_pc_c3_random_realistic_v0d_teacher_r2b_current.json"
            )
            payload = _load_summary(
                summary_path,
                path_zero=scenario == "path_zero",
                config_suffix=config_suffix,
            )
            row[scenario] = _scenario_record(payload)
            row[f"{scenario}_summary_path"] = str(summary_path)
            reported_checkpoint = Path(payload["checkpoint_path"]).resolve()
            if reported_checkpoint != checkpoint:
                problems.append(
                    f"checkpoint mismatch: {summary_path} reports {reported_checkpoint}"
                )
        if all(row.get(scenario) is not None for scenario in SCENARIOS):
            default = row["default"]
            path_zero = row["path_zero"]
            strict = row["strict125"]
            row["path_zero_rmse_ratio"] = float(
                path_zero["path_three_d_rmse_m"] / default["path_three_d_rmse_m"]
            )
            row["strict125_rmse_delta_vs_default_m"] = float(
                strict["path_three_d_rmse_m"] - default["path_three_d_rmse_m"]
            )
            row["complete"] = True
        else:
            row["complete"] = False
        rows.append(row)

    complete_rows = [row for row in rows if row["complete"]]
    complete_rows.sort(
        key=lambda row: (
            float(row["default"]["path_three_d_rmse_m"]),
            -float(row["default"]["success_rate"]),
        )
    )
    if complete_rows:
        best_rmse = float(complete_rows[0]["default"]["path_three_d_rmse_m"])
        for rank, row in enumerate(complete_rows, start=1):
            row["default_rank"] = rank
            row["default_rmse_ratio_vs_best"] = float(
                row["default"]["path_three_d_rmse_m"] / best_rmse
            )

    report = {
        "schema_version": "auv_pc_v0d_p1_unified_bc_summary.v1",
        "root": str(root),
        "frozen_bcs": list(BC_ORDER),
        "eval_seeds": list(range(20, 40)),
        "scenarios": list(SCENARIOS),
        "ranking_metric": "default path_three_d_rmse_m ascending; success is a guardrail",
        "complete_count": len(complete_rows),
        "expected_count": len(BC_ORDER),
        "problems": problems,
        "rows": rows,
        "leaderboard": complete_rows,
    }
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    lines = [
        "# P1 unified BC evaluation",
        "",
        f"- eval seeds: `20-39` ({len(range(20, 40))} episodes per cell)",
        "- primary ranking: default 3D RMSE; success/timeout/safety are guardrails",
        "- path-zero is the same checkpoint with the actor path feature zeroed",
        "- historical `0-19` results are not mixed into this table",
        "",
        "## Default leaderboard",
        "",
        "| rank | BC | 3D RMSE | horizontal | vertical | success | timeout | safety | saturation |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in complete_rows:
        default = row["default"]
        lines.append(
            f"| {row['default_rank']} | `{row['id']}` | "
            f"{_fmt(default['path_three_d_rmse_m'])} | "
            f"{_fmt(default['path_horizontal_rmse_m'])} | "
            f"{_fmt(default['path_vertical_rmse_m'])} | "
            f"{_fmt(default['success_rate'])} | "
            f"{_fmt(default['timeout_rate'])} | "
            f"{_fmt(default['safety_termination_rate'])} | "
            f"{_fmt(default['material_saturation_rate'])} |"
        )
    lines.extend(
        [
            "",
            "## strict125 and path-zero",
            "",
            "| BC | strict125 RMSE | strict125 success | strict125 timeout | path-zero/default |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        if not row["complete"]:
            continue
        strict = row["strict125"]
        lines.append(
            f"| `{row['id']}` | {_fmt(strict['path_three_d_rmse_m'])} | "
            f"{_fmt(strict['success_rate'])} | {_fmt(strict['timeout_rate'])} | "
            f"{_fmt(float(row['path_zero_rmse_ratio']), 3)} |"
        )
    lines.extend(
        [
            "",
            "## Goal17 A matrix cells",
            "",
            "| cell | 3D RMSE | success | timeout | path-zero/default |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for bc in ("g17_a00", "g17_a10", "g17_a01", "g17_a11"):
        row = next((item for item in rows if item["id"] == bc), None)
        if row is None or not row["complete"]:
            lines.append(f"| `{bc}` | missing | - | - | - |")
            continue
        default = row["default"]
        lines.append(
            f"| `{bc}` | {_fmt(default['path_three_d_rmse_m'])} | "
            f"{_fmt(default['success_rate'])} | {_fmt(default['timeout_rate'])} | "
            f"{_fmt(float(row['path_zero_rmse_ratio']), 3)} |"
        )
    lines.extend(
        [
            "",
            "## Scope notes",
            "",
            "- The Goal17 rows use the single BC training seed `91430`; the historical",
            "  three-training-seed Goal17 aggregate is not substituted into this table.",
            "- No new model was trained and no held-out seed was used.",
            "- A missing or invalid input is reported as a problem, not silently replaced.",
            "",
        ]
    )
    if problems:
        lines.extend(["## Problems", ""])
        lines.extend(f"- {problem}" for problem in problems)
        lines.append("")
    (output / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(output / "summary.json")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
