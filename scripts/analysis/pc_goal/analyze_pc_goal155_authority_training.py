#!/usr/bin/env python3
"""Analyze the Goal155 residual-authority 3-seed training screen."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
GOAL145_ANALYZER = REPO_ROOT / "scripts/analyze_pc_goal145_capture_path_audit.py"
SCENARIOS = ("default", "strict125")
SEEDS = (86201, 86202, 86203)
ARMS = ("full", "near10_2", "mid_only", "far_off_only")
GATED_ARMS = ("near10_2", "mid_only", "far_off_only")
SUCCESS_MIN = {"default": 6.0 / 120.0, "strict125": 4.0 / 120.0}
PATH_MAX = 0.05
VERTICAL_MAX = 0.03


def _load_helpers() -> Any:
    spec = importlib.util.spec_from_file_location("goal145_audit", GOAL145_ANALYZER)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {GOAL145_ANALYZER}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


HELPERS = _load_helpers()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=REPO_ROOT / "artifacts/lowlevel/goal155_authority_training_screen_20261111",
    )
    parser.add_argument(
        "--eval-root",
        type=Path,
        default=None,
        help="evaluation root; default: <root>/eval. Use <root>/eval_cpu for the CPU/float64 gate.",
    )
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=155)
    return parser.parse_args()


def _run_dir(eval_root: Path, arm: str, seed: int, scenario: str) -> Path:
    return eval_root / f"{arm}_{seed}" / scenario


def _residual_bins(run_dirs: list[Path]) -> dict[str, Any]:
    values: dict[str, list[np.ndarray]] = {"le2m": [], "2to10m": [], "gt10m": []}
    channel_sq: dict[str, list[np.ndarray]] = {"le2m": [], "2to10m": [], "gt10m": []}
    channel_sum: dict[str, list[np.ndarray]] = {"le2m": [], "2to10m": [], "gt10m": []}
    lag1_values: dict[str, list[float]] = {"le2m": [], "2to10m": [], "gt10m": []}
    raw_saturation: list[np.ndarray] = []
    effective_saturation: list[np.ndarray] = []
    actuator_saturation: list[np.ndarray] = []
    material_saturation: list[np.ndarray] = []
    for run_dir in run_dirs:
        path = run_dir / "trace.npz"
        if not path.is_file():
            continue
        trace = np.load(path, allow_pickle=True)
        if "goal_distance_eta" not in trace.files:
            continue
        distance = np.asarray(trace["goal_distance_eta"], dtype=np.float64)
        episode = (
            np.asarray(trace["episode_id"], dtype=np.int64)
            if "episode_id" in trace.files
            else np.zeros_like(distance, dtype=np.int64)
        )
        residual_key = None
        for candidate in ("learned_residual_action", "residual"):
            if candidate in trace.files:
                residual_key = candidate
                break
        if residual_key is None:
            continue
        residual = np.asarray(trace[residual_key], dtype=np.float64)
        norm = np.linalg.norm(residual, axis=1)
        finite = np.isfinite(distance) & np.isfinite(norm)
        masks = {
            "le2m": finite & (distance <= 2.0),
            "2to10m": finite & (distance > 2.0) & (distance <= 10.0),
            "gt10m": finite & (distance > 10.0),
        }
        for name, mask in masks.items():
            if bool(mask.any()):
                chunk = residual[mask]
                values[name].append(norm[mask])
                channel_sq[name].append((chunk**2).sum(axis=0))
                channel_sum[name].append(chunk.sum(axis=0))
                # Lag-1 must be computed within each episode. Pooling across
                # episode boundaries dilutes persistence and can falsely look
                # like white noise.
                for episode_id in np.unique(episode[mask]):
                    episode_chunk = chunk[episode[mask] == episode_id]
                    if episode_chunk.shape[0] < 4:
                        continue
                    for channel in range(episode_chunk.shape[1]):
                        centered = episode_chunk[:, channel] - episode_chunk[:, channel].mean()
                        denominator = float((centered**2).sum())
                        if denominator > 1e-12:
                            lag1_values[name].append(
                                float((centered[:-1] * centered[1:]).sum() / denominator)
                            )
        if "policy_action_raw" in trace.files:
            raw = np.asarray(trace["policy_action_raw"], dtype=np.float64)
            raw_saturation.append(np.any(np.abs(raw) > 0.95, axis=1))
        if "policy_action_effective" in trace.files:
            effective = np.asarray(trace["policy_action_effective"], dtype=np.float64)
            effective_saturation.append(np.any(np.abs(effective) > 0.95, axis=1))
        for key, target in (
            ("actuator_saturated", actuator_saturation),
            ("material_saturated", material_saturation),
        ):
            if key in trace.files:
                target.append(np.asarray(trace[key], dtype=bool))
    output: dict[str, Any] = {}
    for name, chunks in values.items():
        if not chunks:
            output[name] = {"steps": 0, "rms": None, "mean": None}
            continue
        array = np.concatenate(chunks)
        total_steps = int(array.size)
        channel_rms = (
            np.sqrt(np.sum(np.stack(channel_sq[name]), axis=0) / total_steps)
            if channel_sq[name]
            else None
        )
        mean_vector = (
            np.sum(np.stack(channel_sum[name]), axis=0) / total_steps
            if channel_sum[name]
            else None
        )
        rms = float(np.sqrt(np.mean(array**2)))
        output[name] = {
            "steps": int(array.size),
            "rms": rms,
            "mean": float(np.mean(array)),
            "channel_rms": None if channel_rms is None else channel_rms.tolist(),
            "mean_vector": None if mean_vector is None else mean_vector.tolist(),
            "bias_fraction": (
                None
                if mean_vector is None or rms <= 1e-12
                else float(np.linalg.norm(mean_vector) / rms)
            ),
            "lag1_mean": (
                None if not lag1_values[name] else float(np.mean(lag1_values[name]))
            ),
            "lag1_p50": (
                None if not lag1_values[name] else float(np.median(lag1_values[name]))
            ),
        }
    for name, chunks in (
        ("policy_head_saturation", raw_saturation),
        ("policy_effective_saturation", effective_saturation),
        ("actuator_saturation", actuator_saturation),
        ("material_saturation", material_saturation),
    ):
        if chunks:
            array = np.concatenate(chunks)
            output[name] = {"steps": int(array.size), "fraction": float(array.mean())}
        else:
            output[name] = {"steps": 0, "fraction": None}
    return output


def _pooled_summary(rows: list[dict[str, Any]], *, samples: int, seed: int) -> dict[str, Any]:
    success = [float(row["policy_success"]) - float(row["zero_success"]) for row in rows]
    path = [
        float(row["policy_path_three_d_rmse_m"]) - float(row["zero_path_three_d_rmse_m"])
        for row in rows
        if "policy_path_three_d_rmse_m" in row and "zero_path_three_d_rmse_m" in row
    ]
    vertical = [
        float(row["policy_path_vertical_rmse_m"]) - float(row["zero_path_vertical_rmse_m"])
        for row in rows
        if "policy_path_vertical_rmse_m" in row and "zero_path_vertical_rmse_m" in row
    ]
    return {
        "n_pairs": len(rows),
        "policy_success_count": sum(bool(row["policy_success"]) for row in rows),
        "zero_success_count": sum(bool(row["zero_success"]) for row in rows),
        "policy_plane_miss_count": sum(bool(row["policy_plane_miss"]) for row in rows),
        "zero_plane_miss_count": sum(bool(row["zero_plane_miss"]) for row in rows),
        "policy_timeout_count": sum(bool(row["policy_timeout"]) for row in rows),
        "zero_timeout_count": sum(bool(row["zero_timeout"]) for row in rows),
        "policy_safety_count": sum(bool(row["policy_safety"]) for row in rows),
        "zero_safety_count": sum(bool(row["zero_safety"]) for row in rows),
        "deltas": {
            "success": HELPERS._summarize_delta(
                "policy_minus_zero_success", success, samples=samples, seed=seed
            ),
            "path_three_d_rmse_m": HELPERS._summarize_delta(
                "policy_minus_zero_path", path, samples=samples, seed=seed + 1
            ),
            "path_vertical_rmse_m": HELPERS._summarize_delta(
                "policy_minus_zero_vertical", vertical, samples=samples, seed=seed + 2
            ),
        },
    }


def _fmt(value: Any) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _activity_check(arm: str, bins: dict[str, Any]) -> bool:
    le2 = bins.get("le2m", {}).get("rms") or 0.0
    mid = bins.get("2to10m", {}).get("rms") or 0.0
    far = bins.get("gt10m", {}).get("rms") or 0.0
    if arm == "near10_2":
        return le2 >= 0.5 and far <= 0.05
    if arm == "mid_only":
        return mid >= 0.5 and le2 <= 0.05 and far <= 0.05
    if arm == "far_off_only":
        return le2 >= 0.5 and mid >= 0.5 and far <= 0.05
    return True


def _terminal_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    distances = [
        float(row["policy_final_goal_distance_m"])
        for row in rows
        if "policy_final_goal_distance_m" in row
    ]
    if not distances:
        return {"n": 0, "median_m": None, "p90_m": None, "within_2p5": None, "within_3p0": None}
    array = np.asarray(distances, dtype=np.float64)
    return {
        "n": int(array.size),
        "median_m": float(np.median(array)),
        "p90_m": float(np.percentile(array, 90.0)),
        "within_2p5": float(np.mean(array <= 2.5)),
        "within_3p0": float(np.mean(array <= 3.0)),
    }


def _markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Goal155 residual-authority training screen",
        "",
        f"Verdict: `{summary['verdict']}`",
        "",
        "## Pooled vs zero",
        "",
        "| arm | scenario | policy/zero | Δsuccess | path P/Z (m) | Δpath (m) | Δvertical (m) | gate |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for arm in ARMS:
        for scenario in SCENARIOS:
            result = summary["arms"][arm]["scenarios"][scenario]
            paired = result["paired"]
            path_p = float(
                np.mean([row["policy_path_three_d_rmse_m"] for row in result["rows"]])
            )
            path_z = float(
                np.mean([row["zero_path_three_d_rmse_m"] for row in result["rows"]])
            )
            checks = result["gate"]["checks"]
            failed = [name for name, ok in checks.items() if not ok]
            lines.append(
                "| {arm} | {scenario} | {ps}/{zs} | {ds} | {pp:.4f}/{zp:.4f} | {dp} | {dv} | {gate} |".format(
                    arm=arm,
                    scenario=scenario,
                    ps=paired["policy_success_count"],
                    zs=paired["zero_success_count"],
                    ds=_fmt(paired["deltas"]["success"]["mean"]),
                    pp=path_p,
                    zp=path_z,
                    dp=_fmt(paired["deltas"]["path_three_d_rmse_m"]["mean"]),
                    dv=_fmt(paired["deltas"]["path_vertical_rmse_m"]["mean"]),
                    gate="PASS" if not failed else "FAIL " + ",".join(failed),
                )
            )
    lines.extend(
        [
            "",
            "## Per-seed default success",
            "",
            "| arm | seed | policy/zero | Δsuccess | Δpath (m) |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for arm in ARMS:
        for seed in SEEDS:
            result = summary["arms"][arm]["per_seed"]["default"][str(seed)]
            paired = result["paired"]
            lines.append(
                "| {arm} | {seed} | {ps}/{zs} | {ds} | {dp} |".format(
                    arm=arm,
                    seed=seed,
                    ps=paired["policy_success_count"],
                    zs=paired["zero_success_count"],
                    ds=_fmt(paired["deltas"]["success"]["mean"]),
                    dp=_fmt(paired["deltas"]["path_three_d_rmse_m"]["mean"]),
                )
            )
    lines.extend(["", "## Residual / saturation", ""])
    for arm in ARMS:
        bins = summary["arms"][arm]["residual_bins"]["default"]
        lines.append(
            f"- `{arm}` le2m `{_fmt(bins.get('le2m', {}).get('rms'))}`, "
            f"2to10m `{_fmt(bins.get('2to10m', {}).get('rms'))}`, "
            f"gt10m `{_fmt(bins.get('gt10m', {}).get('rms'))}`, "
            f"bias(le2/2-10/gt10) `{_fmt(bins.get('le2m', {}).get('bias_fraction'))}/"
            f"{_fmt(bins.get('2to10m', {}).get('bias_fraction'))}/"
            f"{_fmt(bins.get('gt10m', {}).get('bias_fraction'))}`, "
            f"lag1(le2/2-10/gt10) `{_fmt(bins.get('le2m', {}).get('lag1_mean'))}/"
            f"{_fmt(bins.get('2to10m', {}).get('lag1_mean'))}/"
            f"{_fmt(bins.get('gt10m', {}).get('lag1_mean'))}`, "
            f"head_sat `{_fmt(bins.get('policy_head_saturation', {}).get('fraction'))}`, "
            f"effective_sat `{_fmt(bins.get('policy_effective_saturation', {}).get('fraction'))}`"
        )
    lines.extend(["", "## Terminal distribution", ""])
    for arm in ARMS:
        terminal = summary["arms"][arm]["terminal"]["default"]
        lines.append(
            f"- `{arm}` n `{terminal['n']}`, median `{_fmt(terminal['median_m'])}`, "
            f"p90 `{_fmt(terminal['p90_m'])}`, within2.5 `{_fmt(terminal['within_2p5'])}`, "
            f"within3.0 `{_fmt(terminal['within_3p0'])}`"
        )
    lines.extend(["", "## Gate", ""])
    for arm in ARMS:
        lines.append(f"- `{arm}`: `{summary['arms'][arm]['verdict']}`")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    args = _parse_args()
    root = args.root.resolve()
    eval_root = (args.eval_root or (root / "eval")).resolve()
    zero_runs: dict[str, dict[str, Any]] = {}
    arm_runs: dict[str, dict[str, dict[int, dict[str, Any]]]] = {}
    for scenario in SCENARIOS:
        zero_runs[scenario] = HELPERS._load_run(
            eval_root / "zero" / scenario, interface="t2"
        )
        arm_runs[scenario] = {}
        for arm in ARMS:
            arm_runs[scenario][arm] = {
                seed: HELPERS._load_run(
                    _run_dir(eval_root, arm, seed, scenario), interface="t2"
                )
                for seed in SEEDS
            }

    arms: dict[str, Any] = {}
    for arm in ARMS:
        per_seed: dict[str, dict[str, Any]] = {}
        scenarios: dict[str, Any] = {}
        for scenario in SCENARIOS:
            per_seed[scenario] = {}
            for seed in SEEDS:
                paired = HELPERS._paired(
                    arm_runs[scenario][arm][seed],
                    zero_runs[scenario],
                    samples=args.bootstrap_samples,
                    seed=args.seed + seed + 1000 * ARMS.index(arm),
                )
                per_seed[scenario][str(seed)] = {"paired": paired, "rows": paired["rows"]}
            rows = [
                row
                for seed in SEEDS
                for row in per_seed[scenario][str(seed)]["rows"]
            ]
            paired = _pooled_summary(rows, samples=args.bootstrap_samples, seed=args.seed)
            bins = _residual_bins([_run_dir(eval_root, arm, seed, scenario) for seed in SEEDS])
            terminal = _terminal_summary(rows)
            seed_nonneg = sum(
                1
                for seed in SEEDS
                if per_seed[scenario][str(seed)]["paired"]["deltas"]["success"]["mean"] >= 0.0
            )
            checks = {
                "success_effect": paired["deltas"]["success"]["mean"] >= SUCCESS_MIN[scenario],
                "path_cost": paired["deltas"]["path_three_d_rmse_m"]["mean"] <= PATH_MAX,
                "seed_consistency": seed_nonneg >= 2,
                "residual_activity": _activity_check(arm, bins),
                "vertical_guard": paired["deltas"]["path_vertical_rmse_m"]["mean"] <= VERTICAL_MAX,
                "safety": (
                    paired["policy_timeout_count"] <= paired["zero_timeout_count"]
                    and paired["policy_safety_count"] <= paired["zero_safety_count"]
                ),
            }
            scenarios[scenario] = {
                "rows": rows,
                "paired": paired,
                "residual_bins": bins,
                "terminal": terminal,
                "seed_nonnegative": seed_nonneg,
                "gate": {"checks": checks, "pass": all(checks.values())},
            }
        arms[arm] = {
            "per_seed": per_seed,
            "scenarios": scenarios,
            "residual_bins": {s: scenarios[s]["residual_bins"] for s in SCENARIOS},
            "terminal": {s: scenarios[s]["terminal"] for s in SCENARIOS},
        }

    for arm in ARMS:
        all_pass = all(arms[arm]["scenarios"][s]["gate"]["pass"] for s in SCENARIOS)
        checks = {
            s: arms[arm]["scenarios"][s]["gate"]["checks"] for s in SCENARIOS
        }
        residual_invalid = any(
            not checks[s]["residual_activity"] for s in SCENARIOS
        )
        success_fail = any(not checks[s]["success_effect"] for s in SCENARIOS)
        path_fail = any(not checks[s]["path_cost"] for s in SCENARIOS)
        seed_fail = any(not checks[s]["seed_consistency"] for s in SCENARIOS)
        vertical_fail = any(not checks[s]["vertical_guard"] for s in SCENARIOS)
        if residual_invalid:
            verdict = "INVALID_NO_RESIDUAL"
        elif all_pass:
            verdict = "SCREEN_PASS_READY_FOR_5M_FORMAL"
        elif vertical_fail:
            verdict = "VERTICAL_DOMINANT"
        elif success_fail and not path_fail:
            verdict = "CAPTURE_TRADEOFF"
        elif path_fail and not success_fail:
            verdict = "PATH_TRADEOFF_REMAINS"
        elif seed_fail and not success_fail:
            verdict = "SEED_CONCENTRATED"
        else:
            verdict = "TRAIN_GATE_FAIL"
        arms[arm]["verdict"] = verdict

    passing = [arm for arm in GATED_ARMS if arms[arm]["verdict"].startswith("SCREEN_PASS")]
    if passing:
        verdict = "AUTHORITY_TRAIN_PASS"
    elif all(arms[arm]["verdict"] == "CAPTURE_TRADEOFF" for arm in GATED_ARMS):
        verdict = "CAPTURE_TRADEOFF"
    elif all(arms[arm]["verdict"] == "PATH_TRADEOFF_REMAINS" for arm in GATED_ARMS):
        verdict = "PATH_TRADEOFF_REMAINS"
    elif all(arms[arm]["verdict"] in {"TRAIN_GATE_FAIL", "INVALID_NO_RESIDUAL"} for arm in GATED_ARMS):
        verdict = "AUTHORITY_NOT_BOTTLENECK"
    else:
        verdict = "MIXED_FAILURE"

    summary = {
        "root": str(root),
        "eval_root": str(eval_root),
        "verdict": verdict,
        "passing_arms": passing,
        "arms": arms,
        "note": "full is a frozen Goal86 3-seed control; gated arms are new 1M runs.",
    }
    summary_root = eval_root if args.eval_root is not None else root
    json_path = summary_root / "audit_summary.json"
    md_path = summary_root / "audit_summary.md"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    md_path.write_text(_markdown(summary), encoding="utf-8")
    print(json.dumps({"json": str(json_path), "md": str(md_path), "verdict": verdict}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
