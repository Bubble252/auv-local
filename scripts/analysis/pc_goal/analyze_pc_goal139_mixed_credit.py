#!/usr/bin/env python3
"""Analyze Goal139 mixed-horizon SAC and PPO credit-horizon screens."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

TRAIN_SEEDS = (86201, 86202, 86203)
DEV_SEEDS = tuple(range(20, 60))
LEVELS = ("nominal", "current15", "current20", "plant20", "combined20")
SCENARIOS = ("strict", "default")
SAC_ARMS = ("M50", "M75")
PPO_ARMS = ("G995", "G980", "G950")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algorithm", choices=("SAC", "PPO"), required=True)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("artifacts/lowlevel/goal139_mixed_horizon_credit_20261111"),
    )
    parser.add_argument(
        "--control-root",
        type=Path,
        default=None,
        help="Goal134 SAC N0 root; ignored for PPO (G995 is internal control)",
    )
    parser.add_argument(
        "--zero-root",
        type=Path,
        default=Path("artifacts/lowlevel/goal134_frozen_sac_stress_20261111"),
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--markdown", type=Path, default=None)
    return parser.parse_args()


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_episodes(path: Path) -> dict[int, dict[str, Any]]:
    if path.is_file():
        rows = _load_json(path)
    else:
        summary_path = path.with_name("summary.json")
        if not summary_path.is_file():
            raise FileNotFoundError(path)
        summary = _load_json(summary_path)
        rows = summary.get("episodes") if isinstance(summary, dict) else None
    if not isinstance(rows, list):
        raise ValueError(f"{path}: episodes must be a list")
    result = {int(row["seed"]): row for row in rows}
    missing = set(DEV_SEEDS) - set(result)
    if missing:
        raise ValueError(f"{path}: missing dev seeds {sorted(missing)}")
    return {seed: result[seed] for seed in DEV_SEEDS}


def _success(row: dict[str, Any]) -> bool:
    if "success" in row:
        return bool(row["success"])
    return bool(row["success_pnav_tight"])


def _finite(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, (bool, int, str)):
        return True
    return math.isfinite(float(value))


def _stats(rows: dict[int, dict[str, Any]]) -> dict[str, Any]:
    path_values = [
        float(row["path_three_d_rmse_m"])
        for row in rows.values()
        if row.get("path_three_d_rmse_m") is not None
    ]
    authority_values = []
    for row in rows.values():
        for key in (
            "residual_l2_ratio_gate_active",
            "residual_to_base_l2_ratio_mean",
            "residual_l2_ratio_mean",
        ):
            if row.get(key) is not None:
                authority_values.append(float(row[key]))
                break
    return {
        "episodes": len(rows),
        "success": sum(1 for row in rows.values() if _success(row)),
        "timeout": sum(1 for row in rows.values() if bool(row.get("timeout", False))),
        "safety": sum(1 for row in rows.values() if bool(row.get("safety", False))),
        "plane_miss": sum(
            1 for row in rows.values() if bool(row.get("goal_plane_miss", False))
        ),
        "path_three_d_rmse_m": float(np.mean(path_values)) if path_values else None,
        "residual_l2_ratio": (
            float(np.mean(authority_values)) if authority_values else None
        ),
        "finite": all(
            _finite(row.get(key))
            for row in rows.values()
            for key in (
                "path_three_d_rmse_m",
                "residual_l2_ratio_gate_active",
                "residual_to_base_l2_ratio_mean",
                "residual_l2_ratio_mean",
                "terminal_eta_distance_m",
            )
        ),
    }


def _ratio(value: float | None, reference: float | None) -> float | None:
    if value is None or reference is None or abs(reference) <= 1e-12:
        return None
    return float(value / reference)


def _sac_status(
    *,
    helped_seeds: int,
    zero_seeds: int,
    control_nominal_guard: bool,
    zero_nominal_guard: bool,
    safety_bad: bool,
    path_bad: bool,
    authority_bad: bool,
) -> tuple[str, str]:
    if safety_bad:
        return "MIXED_SAFETY_STOP", "timeout/safety exceeded the paired control guardrail"
    if helped_seeds >= 2 and (not control_nominal_guard or path_bad or authority_bad):
        return "MIXED_NOMINAL_TRADEOFF", "stress gain came with a nominal/path/authority loss"
    if (
        helped_seeds >= 2
        and zero_seeds >= 2
        and control_nominal_guard
        and zero_nominal_guard
    ):
        return "MIXED_ZERO_ADVANTAGE", "at least 2 seeds beat traditional zero in at least 2 levels"
    if helped_seeds >= 2 and control_nominal_guard:
        return "MIXED_ANCHOR_SIGNAL", "N0 anchor restored and stress signal present, but zero gap remains"
    if helped_seeds == 1:
        return "MIXED_SINGLE_SEED_ONLY", "only one training seed improved in at least 2 levels"
    return "NO_MIXED_EFFECT", "no training seed improved in at least 2 levels"


def _ppo_status(
    *,
    helped_seeds: int,
    zero_seeds: int,
    control_nominal_guard: bool,
    zero_nominal_guard: bool,
    safety_bad: bool,
    path_bad: bool,
    residual_collapse: bool,
) -> tuple[str, str]:
    if residual_collapse:
        return "PPO_RESIDUAL_COLLAPSE_PERSISTS", "gate-active residual L2 remained below 1%"
    if safety_bad:
        return "PPO_CREDIT_SAFETY_STOP", "timeout/safety exceeded the G995 guardrail"
    if helped_seeds >= 2 and (not control_nominal_guard or path_bad):
        return "PPO_CREDIT_TRADEOFF", "credit signal came with nominal/path loss"
    if (
        helped_seeds >= 2
        and zero_seeds >= 2
        and control_nominal_guard
        and zero_nominal_guard
    ):
        return "PPO_CREDIT_ZERO_ADVANTAGE", "at least 2 seeds beat traditional zero in at least 2 levels"
    if helped_seeds >= 2 and control_nominal_guard:
        return "PPO_CREDIT_SIGNAL", "credit-horizon signal present, but zero gap remains"
    if helped_seeds == 1:
        return "PPO_CREDIT_SINGLE_SEED_ONLY", "only one training seed improved in at least 2 levels"
    return "NO_CREDIT_HORIZON_EFFECT", "no training seed improved in at least 2 levels"


def _load_control_root(root: Path, algorithm: str, level: str, scenario: str, seed: int) -> Path:
    if algorithm == "PPO":
        return root / "eval" / "PPO" / "G995" / level / scenario / f"S_s{seed}" / "episodes.json"
    return root / level / scenario / f"S_s{seed}" / "episodes.json"


def analyze(root: Path, control_root: Path, zero_root: Path, algorithm: str) -> dict[str, Any]:
    arms = SAC_ARMS if algorithm == "SAC" else PPO_ARMS
    control_name = "N0" if algorithm == "SAC" else "G995"
    primary = "M50" if algorithm == "SAC" else "G980"
    eval_root = root / "eval" / algorithm
    zero: dict[tuple[str, str], dict[str, Any]] = {}
    control: dict[tuple[str, str, int], dict[int, dict[str, Any]]] = {}
    for level in LEVELS:
        for scenario in SCENARIOS:
            zero[(level, scenario)] = _stats(
                _load_episodes(zero_root / level / scenario / "zero" / "episodes.json")
            )
            for seed in TRAIN_SEEDS:
                control[(level, scenario, seed)] = _load_episodes(
                    _load_control_root(control_root, algorithm, level, scenario, seed)
                )

    result_arms: dict[str, Any] = {}
    validity = {"runs": 0, "finite": True, "heldout_opened": False}
    for arm in arms:
        per_seed: dict[int, Any] = {}
        for seed in TRAIN_SEEDS:
            levels_for_seed: dict[str, Any] = {}
            for level in LEVELS:
                levels_for_seed[level] = {}
                for scenario in SCENARIOS:
                    path = eval_root / arm / level / scenario / f"S_s{seed}" / "episodes.json"
                    policy_rows = _load_episodes(path)
                    policy = _stats(policy_rows)
                    reference = _stats(control[(level, scenario, seed)])
                    traditional = zero[(level, scenario)]
                    validity["runs"] += 1
                    validity["finite"] = validity["finite"] and bool(policy["finite"])
                    levels_for_seed[level][scenario] = {
                        "policy": policy,
                        "control": reference,
                        "zero": traditional,
                        "delta_vs_control": int(policy["success"] - reference["success"]),
                        "delta_vs_zero": int(policy["success"] - traditional["success"]),
                        "timeout_delta_vs_control": int(policy["timeout"] - reference["timeout"]),
                        "safety_delta_vs_control": int(policy["safety"] - reference["safety"]),
                    }
            helped_levels = [
                level
                for level in LEVELS
                if levels_for_seed[level]["strict"]["delta_vs_control"] >= 2
                and levels_for_seed[level]["default"]["delta_vs_control"] >= 0
            ]
            zero_levels = [
                level
                for level in LEVELS
                if levels_for_seed[level]["strict"]["delta_vs_zero"] >= 2
                and levels_for_seed[level]["default"]["delta_vs_zero"] >= 0
            ]
            nominal = levels_for_seed["nominal"]
            per_seed[seed] = {
                "levels": levels_for_seed,
                "helped_levels_vs_control": helped_levels,
                "helped_levels_vs_zero": zero_levels,
                "helped": len(helped_levels) >= 2,
                "zero_helped": len(zero_levels) >= 2,
                "control_nominal_guard": (
                    nominal["strict"]["delta_vs_control"] >= -1
                    and nominal["default"]["delta_vs_control"] >= -1
                ),
                "zero_nominal_guard": (
                    nominal["strict"]["delta_vs_zero"] >= -1
                    and nominal["default"]["delta_vs_zero"] >= -1
                ),
                "nominal_strict_delta_vs_control": nominal["strict"]["delta_vs_control"],
                "nominal_default_delta_vs_control": nominal["default"]["delta_vs_control"],
                "nominal_strict_delta_vs_zero": nominal["strict"]["delta_vs_zero"],
                "nominal_default_delta_vs_zero": nominal["default"]["delta_vs_zero"],
            }

        level_summary: dict[str, Any] = {}
        safety_bad = False
        path_bad = False
        authority_bad = False
        for level in LEVELS:
            level_summary[level] = {}
            for scenario in SCENARIOS:
                policy_success = sum(
                    per_seed[seed]["levels"][level][scenario]["policy"]["success"]
                    for seed in TRAIN_SEEDS
                )
                control_success = sum(
                    per_seed[seed]["levels"][level][scenario]["control"]["success"]
                    for seed in TRAIN_SEEDS
                )
                zero_success_one_seed = float(
                    per_seed[TRAIN_SEEDS[0]]["levels"][level][scenario]["zero"]["success"]
                )
                policy_timeout = sum(
                    per_seed[seed]["levels"][level][scenario]["policy"]["timeout"]
                    for seed in TRAIN_SEEDS
                )
                control_timeout = sum(
                    per_seed[seed]["levels"][level][scenario]["control"]["timeout"]
                    for seed in TRAIN_SEEDS
                )
                policy_safety = sum(
                    per_seed[seed]["levels"][level][scenario]["policy"]["safety"]
                    for seed in TRAIN_SEEDS
                )
                control_safety = sum(
                    per_seed[seed]["levels"][level][scenario]["control"]["safety"]
                    for seed in TRAIN_SEEDS
                )
                policy_path = float(
                    np.mean(
                        [
                            per_seed[seed]["levels"][level][scenario]["policy"]["path_three_d_rmse_m"]
                            for seed in TRAIN_SEEDS
                        ]
                    )
                )
                control_path = float(
                    np.mean(
                        [
                            per_seed[seed]["levels"][level][scenario]["control"]["path_three_d_rmse_m"]
                            for seed in TRAIN_SEEDS
                        ]
                    )
                )
                policy_rms = float(
                    np.mean(
                        [
                            per_seed[seed]["levels"][level][scenario]["policy"]["residual_l2_ratio"]
                            for seed in TRAIN_SEEDS
                        ]
                    )
                )
                control_rms = float(
                    np.mean(
                        [
                            per_seed[seed]["levels"][level][scenario]["control"]["residual_l2_ratio"]
                            for seed in TRAIN_SEEDS
                        ]
                    )
                )
                safety_bad = safety_bad or (
                    policy_timeout + policy_safety > control_timeout + control_safety + 2
                )
                path_bad = path_bad or (policy_path > 1.05 * control_path + 1e-9)
                authority_bad = authority_bad or (policy_rms > 1.5 * control_rms + 1e-9)
                level_summary[level][scenario] = {
                    "policy_success_mean40": policy_success / len(TRAIN_SEEDS),
                    "control_success_mean40": control_success / len(TRAIN_SEEDS),
                    "zero_success_mean40": zero_success_one_seed,
                    "delta_vs_control_total": int(policy_success - control_success),
                    "delta_vs_zero_total": int(
                        policy_success - zero_success_one_seed * len(TRAIN_SEEDS)
                    ),
                    "timeout": int(policy_timeout),
                    "safety": int(policy_safety),
                    "control_timeout": int(control_timeout),
                    "control_safety": int(control_safety),
                    "path_ratio_vs_control": _ratio(policy_path, control_path),
                    "authority_ratio_vs_control": _ratio(policy_rms, control_rms),
                    "policy_residual_l2_ratio": policy_rms,
                    "control_residual_l2_ratio": control_rms,
                }

        helped_seeds = sum(1 for seed in TRAIN_SEEDS if per_seed[seed]["helped"])
        zero_seeds = sum(1 for seed in TRAIN_SEEDS if per_seed[seed]["zero_helped"])
        control_nominal_guard = all(
            per_seed[seed]["control_nominal_guard"] for seed in TRAIN_SEEDS
        )
        zero_nominal_guard = all(per_seed[seed]["zero_nominal_guard"] for seed in TRAIN_SEEDS)
        residual_values = [
            level_summary[level][scenario]["policy_residual_l2_ratio"]
            for level in LEVELS
            for scenario in SCENARIOS
        ]
        residual_collapse = bool(algorithm == "PPO" and np.max(residual_values) < 0.01)
        if algorithm == "SAC":
            status, reason = _sac_status(
                helped_seeds=helped_seeds,
                zero_seeds=zero_seeds,
                control_nominal_guard=control_nominal_guard,
                zero_nominal_guard=zero_nominal_guard,
                safety_bad=safety_bad,
                path_bad=path_bad,
                authority_bad=authority_bad,
            )
        else:
            status, reason = _ppo_status(
                helped_seeds=helped_seeds,
                zero_seeds=zero_seeds,
                control_nominal_guard=control_nominal_guard,
                zero_nominal_guard=zero_nominal_guard,
                safety_bad=safety_bad,
                path_bad=path_bad,
                residual_collapse=residual_collapse,
            )
        result_arms[arm] = {
            "status": status,
            "reason": reason,
            "role": (
                "primary" if arm == primary else ("control" if arm == "G995" else "dose_diagnostic_only")
            ),
            "helped_seeds": helped_seeds,
            "zero_advantage_seeds": zero_seeds,
            "control_nominal_guard": control_nominal_guard,
            "zero_nominal_guard": zero_nominal_guard,
            "safety_bad": safety_bad,
            "path_bad": path_bad,
            "authority_bad": authority_bad,
            "ppo_residual_collapse": residual_collapse,
            "max_residual_l2_ratio": float(np.max(residual_values)),
            "levels": level_summary,
            "per_seed": per_seed,
        }

    primary_status = result_arms[primary]["status"]
    return {
        "schema": "auv_pc_goal139_mixed_credit.v1",
        "algorithm": algorithm,
        "status": primary_status,
        "primary_arm": primary,
        "control_arm": control_name,
        "arms": result_arms,
        "root": str(root),
        "control_root": str(control_root),
        "zero_root": str(zero_root),
        "train_seeds": list(TRAIN_SEEDS),
        "dev_seeds": [min(DEV_SEEDS), max(DEV_SEEDS)],
        "validity": validity,
    }


def _fmt(value: Any) -> str:
    if value is None:
        return "?"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def render_markdown(payload: dict[str, Any]) -> str:
    algorithm = payload["algorithm"]
    arms = SAC_ARMS if algorithm == "SAC" else PPO_ARMS
    title = "mixed-horizon" if algorithm == "SAC" else "credit-horizon"
    lines = [
        f"# Goal139 {algorithm} {title} screen",
        "",
        f"**主臂 {payload['primary_arm']} 裁决**：`{payload['status']}`",
        f"**control**：`{payload['control_arm']}`，**traditional zero** 来自 Goal134。",
        "",
        "| arm | helped seeds vs control | zero seeds | nominal guard | safety | path | residual | status |",
        "|---|---:|---:|---|---|---|---|---|",
    ]
    for arm in arms:
        row = payload["arms"][arm]
        lines.append(
            f"| {arm} | {row['helped_seeds']}/3 | {row['zero_advantage_seeds']}/3 | "
            f"{'OK' if row['control_nominal_guard'] else 'BAD'} | "
            f"{'BAD' if row['safety_bad'] else 'OK'} | "
            f"{'BAD' if row['path_bad'] else 'OK'} | "
            f"{_fmt(row['max_residual_l2_ratio'])} | `{row['status']}` |"
        )
    lines.extend(
        [
            "",
            "成功数为每个训练 seed 的 40 个 dev seeds 汇总；Δ 与同一训练 seed 的 control 配对。",
            "",
            "## Level summary",
            "",
            "| arm | level | strict P/control/zero | Δ vs control | Δ vs zero | default P/control/zero | Δ default | timeout/safety | path ratio |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for arm in arms:
        for level in LEVELS:
            strict = payload["arms"][arm]["levels"][level]["strict"]
            default = payload["arms"][arm]["levels"][level]["default"]
            lines.append(
                f"| {arm} | {level} | "
                f"{strict['policy_success_mean40']:.1f}/{strict['control_success_mean40']:.1f}/{strict['zero_success_mean40']:.1f} | "
                f"{strict['delta_vs_control_total']:+d} | {strict['delta_vs_zero_total']:+d} | "
                f"{default['policy_success_mean40']:.1f}/{default['control_success_mean40']:.1f}/{default['zero_success_mean40']:.1f} | "
                f"{default['delta_vs_control_total']:+d} | "
                f"{strict['timeout'] + strict['safety']}/{strict['control_timeout'] + strict['control_safety']} | "
                f"{_fmt(strict['path_ratio_vs_control'])} |"
            )
    lines.extend(["", "## Per-seed helped levels", ""])
    for arm in arms:
        lines.extend(
            [
                f"### {arm}",
                "",
                "| train seed | helped vs control | helped vs zero | nominal Δ vs control | nominal Δ vs zero |",
                "|---:|---|---|---:|---:|",
            ]
        )
        for seed in TRAIN_SEEDS:
            per_seed = payload["arms"][arm]["per_seed"]
            row = per_seed.get(str(seed), per_seed[seed])
            lines.append(
                f"| {seed} | {', '.join(row['helped_levels_vs_control']) or '-'} | "
                f"{', '.join(row['helped_levels_vs_zero']) or '-'} | "
                f"{row['nominal_strict_delta_vs_control']:+d}/{row['nominal_default_delta_vs_control']:+d} | "
                f"{row['nominal_strict_delta_vs_zero']:+d}/{row['nominal_default_delta_vs_zero']:+d} |"
            )
        lines.append("")
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    control_root = args.control_root
    if args.algorithm == "SAC" and control_root is None:
        control_root = Path("artifacts/lowlevel/goal134_frozen_sac_stress_20261111")
    if control_root is None:
        control_root = args.root
    output = args.output or args.root / "analysis" / f"goal139_{args.algorithm.lower()}.json"
    markdown = args.markdown or args.root / "analysis" / f"goal139_{args.algorithm.lower()}.md"
    payload = analyze(args.root, control_root, args.zero_root, args.algorithm)
    output.parent.mkdir(parents=True, exist_ok=True)
    markdown.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    markdown.write_text(render_markdown(payload), encoding="utf-8")
    print(json.dumps({"algorithm": args.algorithm, "status": payload["status"], "output": str(output)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
