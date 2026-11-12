from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

THRESHOLD_EPS = 1e-12


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pair Goal59 policy and zero-residual evaluations by training seed"
    )
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[240, 368, 496, 624, 752, 880, 1008, 1136],
    )
    parser.add_argument(
        "--scenarios",
        nargs="+",
        default=["default", "strict125"],
    )
    parser.add_argument(
        "--policy-template",
        default="artifacts/formal_s{seed}/eval_{scenario}_policy/summary.json",
    )
    parser.add_argument(
        "--zero-template",
        default="artifacts/zero_baseline_{scenario}/summary.json",
    )
    parser.add_argument(
        "--probe-template",
        default="artifacts/formal_s{seed}/noisynet_probe.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown", type=Path)
    parser.add_argument("--bootstrap-draws", type=int, default=10000)
    return parser.parse_args()


def _resolve(root: Path, template: str, **values: Any) -> Path:
    path = Path(template.format(**values))
    return path if path.is_absolute() else root / path


def _load_summary(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    summary = payload.get("summary", payload)
    if not isinstance(summary, dict):
        raise ValueError(f"summary must be an object: {path}")
    return summary


def _bootstrap_ci(values: list[float], *, draws: int, seed: int = 20261109) -> list[float]:
    if not values:
        return [float("nan"), float("nan")]
    array = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, array.size, size=(draws, array.size))
    means = array[indices].mean(axis=1)
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def _scenario_row(policy: dict[str, Any], zero: dict[str, Any]) -> dict[str, Any]:
    return {
        "policy_success": float(policy["success_rate"]),
        "zero_success": float(zero["success_rate"]),
        "delta_success": float(policy["success_rate"] - zero["success_rate"]),
        "policy_capture": float(policy["terminal_eta_capture_rate"]),
        "zero_capture": float(zero["terminal_eta_capture_rate"]),
        "delta_capture": float(
            policy["terminal_eta_capture_rate"] - zero["terminal_eta_capture_rate"]
        ),
        "policy_plane_miss": float(policy["goal_plane_miss_rate"]),
        "zero_plane_miss": float(zero["goal_plane_miss_rate"]),
        "delta_plane_miss": float(
            policy["goal_plane_miss_rate"] - zero["goal_plane_miss_rate"]
        ),
        "policy_pre_goal_rmse": policy["pre_goal_3d_rmse_m"],
        "zero_pre_goal_rmse": zero["pre_goal_3d_rmse_m"],
        "delta_pre_goal_rmse": (
            None
            if policy["pre_goal_3d_rmse_m"] is None or zero["pre_goal_3d_rmse_m"] is None
            else float(policy["pre_goal_3d_rmse_m"] - zero["pre_goal_3d_rmse_m"])
        ),
        "delta_safety": float(
            policy["safety_termination_rate"] - zero["safety_termination_rate"]
        ),
        "delta_timeout": float(policy["timeout_rate"] - zero["timeout_rate"]),
        "gate_active_rate": float(policy["gate_active_rate"]),
        "residual_l2_ratio_gate_active": float(
            policy["residual_l2_ratio_gate_active"]
        ),
        "action_saturation_fraction": float(policy["action_saturation_fraction"]),
        "finite": bool(policy["finite"]),
    }


def _seed_row(
    seed: int,
    scenarios: dict[str, dict[str, Any]],
    probe: dict[str, Any] | None,
) -> dict[str, Any]:
    capture_deltas = [float(row["delta_capture"]) for row in scenarios.values()]
    main_effect = (
        max(capture_deltas) >= 0.05 - THRESHOLD_EPS
        and min(capture_deltas) >= -THRESHOLD_EPS
    )
    success_pass = all(
        float(row["delta_success"]) >= -0.025 - THRESHOLD_EPS
        for row in scenarios.values()
    )
    plane_pass = all(
        float(row["delta_plane_miss"]) <= 0.05 + THRESHOLD_EPS
        for row in scenarios.values()
    )
    safety_pass = all(
        float(row["delta_safety"]) <= 0.02 + THRESHOLD_EPS
        for row in scenarios.values()
    )
    timeout_pass = all(
        float(row["delta_timeout"]) <= 0.05 + THRESHOLD_EPS
        for row in scenarios.values()
    )
    path_pass = all(
        row["delta_pre_goal_rmse"] is None
        or float(row["delta_pre_goal_rmse"]) <= 0.10 + THRESHOLD_EPS
        for row in scenarios.values()
    )
    residual_pass = all(
        float(row["residual_l2_ratio_gate_active"]) >= 0.01 - THRESHOLD_EPS
        for row in scenarios.values()
    )
    gate_pass = all(float(row["gate_active_rate"]) > 0.0 for row in scenarios.values())
    finite_pass = all(bool(row["finite"]) for row in scenarios.values())
    bounded_pass = all(
        float(row["action_saturation_fraction"]) <= 0.5 + THRESHOLD_EPS
        for row in scenarios.values()
    )
    raw_pass = None
    if probe is not None:
        raw_pass = (
            float(probe.get("raw_saturation_fraction_any_dim", 1.0))
            <= 0.5 + THRESHOLD_EPS
        )
    checks = {
        "main_effect": main_effect,
        "success": success_pass,
        "plane_miss": plane_pass,
        "safety": safety_pass,
        "timeout": timeout_pass,
        "path_risk": path_pass,
        "residual_nonzero": residual_pass,
        "gate_active": gate_pass,
        "finite": finite_pass,
        "bounded_saturation": bounded_pass,
        "raw_saturation": raw_pass,
    }
    seed_pass = all(value for value in checks.values() if value is not None)
    return {
        "seed": int(seed),
        "scenarios": scenarios,
        "probe": probe,
        "checks": checks,
        "seed_pass": bool(seed_pass),
    }


def _scenario_summary(rows: list[dict[str, Any]], scenario: str, draws: int) -> dict[str, Any]:
    values = [float(row["scenarios"][scenario]["delta_capture"]) for row in rows]
    success = [float(row["scenarios"][scenario]["delta_success"]) for row in rows]
    plane = [float(row["scenarios"][scenario]["delta_plane_miss"]) for row in rows]
    return {
        "delta_capture_mean": float(np.mean(values)),
        "delta_capture_ci95": _bootstrap_ci(values, draws=draws),
        "delta_success_mean": float(np.mean(success)),
        "delta_plane_miss_mean": float(np.mean(plane)),
    }


def _overall(rows: list[dict[str, Any]], scenarios: list[str], draws: int) -> dict[str, Any]:
    per_scenario = {
        scenario: _scenario_summary(rows, scenario, draws) for scenario in scenarios
    }
    seed_pass_count = sum(1 for row in rows if row["seed_pass"])
    overall_pass = False
    for summary in per_scenario.values():
        if (
            summary["delta_capture_mean"] >= 0.05
            and summary["delta_capture_ci95"][0] > 0.0
            and summary["delta_success_mean"] >= -0.025
            and summary["delta_plane_miss_mean"] <= 0.05
            and seed_pass_count >= 6
        ):
            overall_pass = True
    main_effect_seeds = [
        row for row in rows if bool(row["checks"]["main_effect"])
    ]
    success_failed_main = [
        row
        for row in main_effect_seeds
        if not bool(row["checks"]["success"])
    ]
    if overall_pass:
        status = "PASS"
    elif success_failed_main:
        status = "FAIL"
    elif main_effect_seeds:
        status = "AMBIGUOUS"
    else:
        status = "FAIL"
    return {
        "status": status,
        "seed_pass_count": int(seed_pass_count),
        "seed_count": len(rows),
        "main_effect_seed_count": len(main_effect_seeds),
        "per_scenario": per_scenario,
    }


def _markdown(result: dict[str, Any], scenarios: list[str]) -> str:
    lines = [
        "# Goal59 配对结果：policy vs zero-residual",
        "",
        f"总体裁决：**{result['overall']['status']}**",
        "",
        "| seed | " + " | ".join(f"{s} Δcapture / Δsuccess / Δplane" for s in scenarios) + " | seed通过 |",
        "|---:|" + "---:|" * len(scenarios) + "---|",
    ]
    for row in result["rows"]:
        cells = []
        for scenario in scenarios:
            item = row["scenarios"][scenario]
            cells.append(
                f"{item['delta_capture']:+.3f} / {item['delta_success']:+.3f} / "
                f"{item['delta_plane_miss']:+.3f}"
            )
        lines.append(
            f"| {row['seed']} | " + " | ".join(cells) + f" | {row['seed_pass']} |"
        )
    lines.extend(["", "## 场景均值", "", "| 场景 | Δcapture 均值 | 95% CI | Δsuccess | Δplane-miss |", "|---|---:|---|---:|---:|"])
    for scenario in scenarios:
        item = result["overall"]["per_scenario"][scenario]
        lines.append(
            f"| {scenario} | {item['delta_capture_mean']:+.4f} | "
            f"[{item['delta_capture_ci95'][0]:+.4f}, {item['delta_capture_ci95'][1]:+.4f}] | "
            f"{item['delta_success_mean']:+.4f} | {item['delta_plane_miss_mean']:+.4f} |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    args = _parse_args()
    rows: list[dict[str, Any]] = []
    for seed in args.seeds:
        scenario_rows = {}
        for scenario in args.scenarios:
            policy_path = _resolve(
                args.root,
                args.policy_template,
                seed=seed,
                scenario=scenario,
            )
            zero_path = _resolve(
                args.root,
                args.zero_template,
                seed=seed,
                scenario=scenario,
            )
            scenario_rows[scenario] = _scenario_row(
                _load_summary(policy_path),
                _load_summary(zero_path),
            )
        probe_path = _resolve(args.root, args.probe_template, seed=seed)
        probe = (
            json.loads(probe_path.read_text(encoding="utf-8"))
            if probe_path.is_file()
            else None
        )
        rows.append(_seed_row(seed, scenario_rows, probe))
    result = {
        "schema": "auv_goal59_paired.v1",
        "seeds": args.seeds,
        "scenarios": args.scenarios,
        "rows": rows,
        "overall": _overall(rows, args.scenarios, args.bootstrap_draws),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    if args.markdown is not None:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(_markdown(result, args.scenarios), encoding="utf-8")
    print(json.dumps(result["overall"], indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
