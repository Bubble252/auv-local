#!/usr/bin/env python3
"""Analyze Goal138 near-terminal curriculum screens against paired N0 controls."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

ARMS = ("T15", "T10")
TRAIN_SEEDS = (86201, 86202, 86203)
DEV_SEEDS = tuple(range(20, 60))
LEVELS = ("nominal", "current15", "current20", "plant20", "combined20")
SCENARIOS = ("strict", "default")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algorithm", choices=("SAC", "PPO"), required=True)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("artifacts/lowlevel/goal138_terminal_density_20261111"),
    )
    parser.add_argument(
        "--n0-root",
        type=Path,
        default=None,
        help="Goal134 root for SAC; Goal136 PPO root for PPO",
    )
    parser.add_argument(
        "--zero-root",
        type=Path,
        default=None,
        help="Goal134 root containing zero-residual episodes",
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


def _arm_status(
    *,
    helped_seeds: int,
    nominal_strict_delta: int,
    nominal_default_delta: int,
    safety_bad: bool,
    path_bad: bool,
    authority_bad: bool,
    residual_collapse: bool,
) -> tuple[str, str]:
    if safety_bad:
        return "TERMINAL_SAFETY_STOP", "timeout/safety exceeded the paired N0 guardrail"
    if residual_collapse:
        return (
            "PPO_RESIDUAL_COLLAPSE_PERSISTS",
            "gate-active residual L2 remained below 1% of frozen-BC base",
        )
    if helped_seeds >= 2 and authority_bad:
        return "AUTHORITY_STOP", "success signal present but residual authority exceeded 1.5x N0"
    if helped_seeds >= 2 and (nominal_strict_delta < -1 or nominal_default_delta < -1 or path_bad):
        return (
            "TERMINAL_CURRICULUM_TRADEOFF",
            "stress gain came with a nominal or path guardrail loss",
        )
    if helped_seeds >= 2:
        return "TERMINAL_DENSITY_SIGNAL", "at least 2 training seeds improved in at least 2 levels"
    if helped_seeds == 1:
        return "TERMINAL_SINGLE_SEED_ONLY", "only one training seed improved"
    return "NO_TERMINAL_DENSITY_EFFECT", "no training seed improved in at least 2 levels"


def analyze(root: Path, n0_root: Path, zero_root: Path, algorithm: str) -> dict[str, Any]:
    eval_root = root / "eval" / algorithm
    n0_eval_root = n0_root if algorithm == "SAC" else n0_root / "eval" / "N0"
    zero: dict[tuple[str, str], dict[str, Any]] = {}
    n0: dict[tuple[str, str, int], dict[int, dict[str, Any]]] = {}
    for level in LEVELS:
        for scenario in SCENARIOS:
            zero[(level, scenario)] = _stats(
                _load_episodes(zero_root / level / scenario / "zero" / "episodes.json")
            )
            for seed in TRAIN_SEEDS:
                n0[(level, scenario, seed)] = _load_episodes(
                    n0_eval_root / level / scenario / f"S_s{seed}" / "episodes.json"
                )

    arms: dict[str, Any] = {}
    validity = {"runs": 0, "finite": True, "heldout_opened": False}
    for arm in ARMS:
        per_seed: dict[int, Any] = {}
        for seed in TRAIN_SEEDS:
            per_level: dict[str, Any] = {}
            for level in LEVELS:
                per_level[level] = {}
                for scenario in SCENARIOS:
                    path = eval_root / arm / level / scenario / f"S_s{seed}" / "episodes.json"
                    policy_rows = _load_episodes(path)
                    policy = _stats(policy_rows)
                    reference = _stats(n0[(level, scenario, seed)])
                    control = zero[(level, scenario)]
                    validity["runs"] += 1
                    validity["finite"] = validity["finite"] and bool(policy["finite"])
                    per_level[level][scenario] = {
                        "policy": policy,
                        "n0": reference,
                        "zero": control,
                        "delta_vs_n0": int(policy["success"] - reference["success"]),
                        "delta_vs_zero": int(policy["success"] - control["success"]),
                        "timeout_delta_vs_n0": int(policy["timeout"] - reference["timeout"]),
                        "safety_delta_vs_n0": int(policy["safety"] - reference["safety"]),
                    }
            helped_levels = [
                level
                for level in LEVELS
                if per_level[level]["strict"]["delta_vs_n0"] >= 2
                and per_level[level]["default"]["delta_vs_n0"] >= 0
            ]
            per_seed[seed] = {
                "levels": per_level,
                "helped_levels": helped_levels,
                "helped": len(helped_levels) >= 2,
                "nominal_strict_delta_vs_n0": per_level["nominal"]["strict"]["delta_vs_n0"],
                "nominal_default_delta_vs_n0": per_level["nominal"]["default"][
                    "delta_vs_n0"
                ],
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
                n0_success = sum(
                    per_seed[seed]["levels"][level][scenario]["n0"]["success"]
                    for seed in TRAIN_SEEDS
                )
                zero_success = float(
                    per_seed[TRAIN_SEEDS[0]]["levels"][level][scenario]["zero"]["success"]
                )
                policy_timeout = sum(
                    per_seed[seed]["levels"][level][scenario]["policy"]["timeout"]
                    for seed in TRAIN_SEEDS
                )
                n0_timeout = sum(
                    per_seed[seed]["levels"][level][scenario]["n0"]["timeout"]
                    for seed in TRAIN_SEEDS
                )
                policy_safety = sum(
                    per_seed[seed]["levels"][level][scenario]["policy"]["safety"]
                    for seed in TRAIN_SEEDS
                )
                n0_safety = sum(
                    per_seed[seed]["levels"][level][scenario]["n0"]["safety"]
                    for seed in TRAIN_SEEDS
                )
                policy_path = float(
                    np.mean(
                        [
                            per_seed[seed]["levels"][level][scenario]["policy"][
                                "path_three_d_rmse_m"
                            ]
                            for seed in TRAIN_SEEDS
                        ]
                    )
                )
                n0_path = float(
                    np.mean(
                        [
                            per_seed[seed]["levels"][level][scenario]["n0"][
                                "path_three_d_rmse_m"
                            ]
                            for seed in TRAIN_SEEDS
                        ]
                    )
                )
                policy_rms = float(
                    np.mean(
                        [
                            per_seed[seed]["levels"][level][scenario]["policy"][
                                "residual_l2_ratio"
                            ]
                            for seed in TRAIN_SEEDS
                        ]
                    )
                )
                n0_rms = float(
                    np.mean(
                        [
                            per_seed[seed]["levels"][level][scenario]["n0"][
                                "residual_l2_ratio"
                            ]
                            for seed in TRAIN_SEEDS
                        ]
                    )
                )
                safety_bad = safety_bad or (
                    policy_timeout + policy_safety
                    > n0_timeout + n0_safety + 2
                )
                path_bad = path_bad or (policy_path > 1.05 * n0_path + 1e-9)
                authority_bad = authority_bad or (policy_rms > 1.5 * n0_rms + 1e-9)
                level_summary[level][scenario] = {
                    "policy_success_mean40": policy_success / len(TRAIN_SEEDS),
                    "n0_success_mean40": n0_success / len(TRAIN_SEEDS),
                    "zero_success_mean40": zero_success,
                    "delta_vs_n0_total": int(policy_success - n0_success),
                    "delta_vs_zero_total": int(policy_success - zero_success * len(TRAIN_SEEDS)),
                    "timeout": int(policy_timeout),
                    "safety": int(policy_safety),
                    "n0_timeout": int(n0_timeout),
                    "n0_safety": int(n0_safety),
                    "path_ratio_vs_n0": _ratio(policy_path, n0_path),
                    "authority_ratio_vs_n0": _ratio(policy_rms, n0_rms),
                    "policy_residual_l2_ratio": policy_rms,
                    "n0_residual_l2_ratio": n0_rms,
                }

        helped_seeds = sum(1 for seed in TRAIN_SEEDS if per_seed[seed]["helped"])
        nominal_strict_delta = int(
            level_summary["nominal"]["strict"]["delta_vs_n0_total"]
        )
        nominal_default_delta = int(
            level_summary["nominal"]["default"]["delta_vs_n0_total"]
        )
        residual_values = [
            level_summary[level][scenario]["policy_residual_l2_ratio"]
            for level in LEVELS
            for scenario in SCENARIOS
        ]
        residual_collapse = bool(
            algorithm == "PPO" and np.max(residual_values) < 0.01
        )
        status, reason = _arm_status(
            helped_seeds=helped_seeds,
            nominal_strict_delta=nominal_strict_delta,
            nominal_default_delta=nominal_default_delta,
            safety_bad=safety_bad,
            path_bad=path_bad,
            authority_bad=authority_bad,
            residual_collapse=residual_collapse,
        )
        arms[arm] = {
            "status": status,
            "reason": reason,
            "role": "primary" if arm == "T15" else "dose_diagnostic_only",
            "helped_seeds": helped_seeds,
            "nominal_strict_delta_vs_n0": nominal_strict_delta,
            "nominal_default_delta_vs_n0": nominal_default_delta,
            "safety_bad": safety_bad,
            "path_bad": path_bad,
            "authority_bad": authority_bad,
            "ppo_residual_collapse": residual_collapse,
            "max_residual_l2_ratio": float(np.max(residual_values)),
            "levels": level_summary,
            "per_seed": per_seed,
        }

    primary_status = arms["T15"]["status"]
    return {
        "schema": "auv_pc_goal138_terminal_density.v1",
        "algorithm": algorithm,
        "status": primary_status,
        "t15_status": primary_status,
        "t10_status": arms["T10"]["status"],
        "root": str(root),
        "n0_root": str(n0_root),
        "zero_root": str(zero_root),
        "train_seeds": list(TRAIN_SEEDS),
        "dev_seeds": [min(DEV_SEEDS), max(DEV_SEEDS)],
        "validity": validity,
        "arms": arms,
    }


def _fmt(value: Any) -> str:
    if value is None:
        return "?"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        f"# Goal138 {payload['algorithm']} terminal-density screen",
        "",
        f"**主臂 T15 裁决**：`{payload['status']}`",
        f"**T10 剂量诊断**：`{payload['t10_status']}`",
        "",
        "| arm | helped seeds | nominal strict Δ vs N0 | nominal default Δ vs N0 | safety | path | authority | status |",
        "|---|---:|---:|---:|---|---|---|---|",
    ]
    for arm in ARMS:
        row = payload["arms"][arm]
        lines.append(
            f"| {arm} | {row['helped_seeds']}/3 | {row['nominal_strict_delta_vs_n0']:+d} | "
            f"{row['nominal_default_delta_vs_n0']:+d} | "
            f"{'BAD' if row['safety_bad'] else 'OK'} | "
            f"{'BAD' if row['path_bad'] else 'OK'} | "
            f"{'BAD' if row['authority_bad'] else 'OK'} | `{row['status']}` |"
        )
    lines.extend(
        [
            "",
            "成功数为每个训练 seed 的 40 个 dev seeds 汇总；Δ vs N0 是同一训练 seed 的配对总计。",
            "",
            "## Level summary",
            "",
            "| arm | level | strict D/N0/zero | Δ strict | default D/N0/zero | Δ default | timeout/safety | path ratio | authority ratio |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for arm in ARMS:
        for level in LEVELS:
            strict = payload["arms"][arm]["levels"][level]["strict"]
            default = payload["arms"][arm]["levels"][level]["default"]
            lines.append(
                f"| {arm} | {level} | "
                f"{strict['policy_success_mean40']:.1f}/{strict['n0_success_mean40']:.1f}/{strict['zero_success_mean40']:.1f} | "
                f"{strict['delta_vs_n0_total']:+d} | "
                f"{default['policy_success_mean40']:.1f}/{default['n0_success_mean40']:.1f}/{default['zero_success_mean40']:.1f} | "
                f"{default['delta_vs_n0_total']:+d} | "
                f"{strict['timeout'] + strict['safety']}/{strict['n0_timeout'] + strict['n0_safety']} | "
                f"{_fmt(strict['path_ratio_vs_n0'])} | "
                f"{_fmt(strict['authority_ratio_vs_n0'])} |"
            )
    lines.extend(["", "## Per-seed helped levels", ""])
    for arm in ARMS:
        lines.extend(
            [
                f"### {arm}",
                "",
                "| train seed | helped levels | nominal strict Δ | nominal default Δ |",
                "|---:|---|---:|---:|",
            ]
        )
        for seed in TRAIN_SEEDS:
            per_seed = payload["arms"][arm]["per_seed"]
            row = per_seed.get(str(seed), per_seed[seed])
            lines.append(
                f"| {seed} | {', '.join(row['helped_levels']) or '-'} | "
                f"{row['nominal_strict_delta_vs_n0']:+d} | "
                f"{row['nominal_default_delta_vs_n0']:+d} |"
            )
        lines.append("")
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    n0_root = args.n0_root
    if n0_root is None:
        n0_root = (
            Path("artifacts/lowlevel/goal134_frozen_sac_stress_20261111")
            if args.algorithm == "SAC"
            else Path("artifacts/lowlevel/goal136_dline_ppo_20261111")
        )
    zero_root = args.zero_root or Path(
        "artifacts/lowlevel/goal134_frozen_sac_stress_20261111"
    )
    output = args.output or args.root / "analysis" / f"goal138_{args.algorithm.lower()}.json"
    markdown = args.markdown or args.root / "analysis" / f"goal138_{args.algorithm.lower()}.md"
    payload = analyze(args.root, n0_root, zero_root, args.algorithm)
    output.parent.mkdir(parents=True, exist_ok=True)
    markdown.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    markdown.write_text(render_markdown(payload), encoding="utf-8")
    print(json.dumps({"algorithm": args.algorithm, "status": payload["status"], "output": str(output)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
