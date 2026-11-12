#!/usr/bin/env python3
"""Analyze Goal160 action-parameterization evaluations against H3 zero.

Layout: <root>/<arm>/<radius>/<level>/<mode>_seed<seed>/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

ARMS = ("A_actuator_near10_2", "B_command_near10_2", "C_command_terminal_train")
RADII = ("default", "strict125")
LEVELS = ("nominal", "current07", "current10", "current15")
SEEDS = (86201, 86202, 86203)
MODES = ("zero", "policy")
PRIMARY_LEVEL = "current10"
OTHER_LEVELS = ("nominal", "current07", "current15")


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "NA"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "NA"


def _episode_rows(run_dir: Path, episodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    trace_path = run_dir / "trace.npz"
    if not trace_path.is_file():
        return []
    trace = np.load(trace_path, allow_pickle=False)
    required = {"seed", "eta_ned", "goal_ned", "goal_distance_eta", "done"}
    if not required.issubset(trace.files):
        return []
    seeds = np.asarray(trace["seed"], dtype=np.int64)
    eta = np.asarray(trace["eta_ned"], dtype=np.float64)
    goal = np.asarray(trace["goal_ned"], dtype=np.float64)
    distance = np.asarray(trace["goal_distance_eta"], dtype=np.float64)
    done = np.asarray(trace["done"], dtype=bool)
    residual = (
        np.asarray(trace["residual"], dtype=np.float64)
        if "residual" in trace.files
        else np.asarray(trace["learned_residual_action"], dtype=np.float64)
        if "learned_residual_action" in trace.files
        else np.zeros((len(seeds), 3))
    )
    command_residual = (
        np.asarray(trace["command_residual_applied"], dtype=np.float64)
        if "command_residual_applied" in trace.files
        else None
    )
    vertical_path = (
        np.asarray(trace["path_vertical_error"], dtype=np.float64)
        if "path_vertical_error" in trace.files
        else np.full(len(seeds), np.nan)
    )
    horizontal_path = (
        np.asarray(trace["path_horizontal_error"], dtype=np.float64)
        if "path_horizontal_error" in trace.files
        else np.full(len(seeds), np.nan)
    )
    progress = (
        np.asarray(trace["path_progress"], dtype=np.float64)
        if "path_progress" in trace.files
        else np.full(len(seeds), np.nan)
    )
    success_by_seed = {
        int(item.get("seed", -1)): bool(item.get("success_eta_tight_at_done", False))
        for item in episodes
    }
    rows: list[dict[str, Any]] = []
    for seed in np.unique(seeds):
        index = np.flatnonzero(seeds == seed)
        if not index.size:
            continue
        end = index[-1]
        done_index = index[done[index]]
        if done_index.size:
            end = int(done_index[0])
        miss = eta[end, :3] - goal[end, :3]
        entry_index = index[distance[index] <= 2.0]
        entry_vertical = float("nan")
        entry_horizontal = float("nan")
        if entry_index.size:
            entry = int(entry_index[0])
            entry_vertical = float(abs(eta[entry, 2] - goal[entry, 2]))
            entry_horizontal = float(np.linalg.norm(eta[entry, :2] - goal[entry, :2]))
        finite_distance = distance[index][np.isfinite(distance[index])]
        closest_distance = (
            float(np.min(finite_distance)) if finite_distance.size else float("nan")
        )
        near_fraction = {
            f"le{str(radius).replace('.', 'p')}": float(
                np.mean(distance[index] <= radius)
            )
            for radius in (1.25, 1.5, 2.0, 3.0)
        }
        progress_delta = float("nan")
        progress_values = progress[index]
        if np.isfinite(progress_values).any():
            progress_delta = float(np.nanmax(progress_values) - np.nanmin(progress_values))
        local_residual = residual[index]
        bins = {
            "le2": residual[index[distance[index] <= 2.0]],
            "2to10": residual[index[(distance[index] > 2.0) & (distance[index] <= 10.0)]],
            "gt10": residual[index[distance[index] > 10.0]],
        }
        command_bins = None
        if command_residual is not None:
            command_bins = {
                "le2": command_residual[index[distance[index] <= 2.0]],
                "2to10": command_residual[index[(distance[index] > 2.0) & (distance[index] <= 10.0)]],
                "gt10": command_residual[index[distance[index] > 10.0]],
            }
        rows.append(
            {
                "seed": int(seed),
                "success": success_by_seed.get(int(seed), False),
                "final_distance": float(distance[end]),
                "final_vertical": float(abs(miss[2])),
                "final_horizontal": float(np.linalg.norm(miss[:2])),
                "entry_vertical": entry_vertical,
                "entry_horizontal": entry_horizontal,
                "closest_distance": closest_distance,
                "near_fraction": near_fraction,
                "path_progress_delta": progress_delta,
                "vertical_path_rmse": float(np.sqrt(np.nanmean(np.square(vertical_path[index])))),
                "horizontal_path_rmse": float(np.sqrt(np.nanmean(np.square(horizontal_path[index])))),
                "residual_abs_mean": float(np.mean(np.abs(local_residual))),
                "residual_channel_abs": [
                    float(np.mean(np.abs(local_residual[:, channel]))) for channel in range(3)
                ],
                "residual_bin_rms": {
                    name: (float(np.sqrt(np.mean(np.square(values)))) if values.size else 0.0)
                    for name, values in bins.items()
                },
                "command_residual_bin_rms": (
                    None
                    if command_bins is None
                    else {
                        name: (float(np.sqrt(np.mean(np.square(values)))) if values.size else 0.0)
                        for name, values in command_bins.items()
                    }
                ),
            }
        )
    return rows


def _run(run_dir: Path) -> dict[str, Any] | None:
    summary_path = run_dir / "summary.json"
    if not summary_path.is_file():
        return None
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    episodes_path = run_dir / "episodes.json"
    episodes = (
        json.loads(episodes_path.read_text(encoding="utf-8"))
        if episodes_path.is_file()
        else summary.get("episodes", [])
    )
    rows = _episode_rows(run_dir, episodes)
    result: dict[str, Any] = {
        "run_dir": str(run_dir),
        "n": len(episodes),
        "success": sum(bool(item.get("success_eta_tight_at_done", False)) for item in episodes),
        "plane_miss": sum(bool(item.get("goal_plane_miss", False)) for item in episodes),
        "timeout": sum(bool(item.get("timeout", False)) for item in episodes),
        "safety": sum(bool(item.get("safety", False)) for item in episodes),
        "finite": bool(summary.get("summary", {}).get("finite", True)),
        "summary": summary.get("summary", {}),
    }
    if rows:
        result.update(
            {
                "final_distance_median": float(np.median([row["final_distance"] for row in rows])),
                "final_distance_p90": float(np.nanpercentile([row["final_distance"] for row in rows], 90)),
                "final_vertical_median": float(np.median([row["final_vertical"] for row in rows])),
                "final_horizontal_median": float(np.median([row["final_horizontal"] for row in rows])),
                "entry_vertical_p90": float(np.nanpercentile([row["entry_vertical"] for row in rows], 90)),
                "entry_horizontal_p90": float(np.nanpercentile([row["entry_horizontal"] for row in rows], 90)),
                "closest_distance_median": float(np.nanmedian([row["closest_distance"] for row in rows])),
                "path_progress_delta_median": float(np.nanmedian([row["path_progress_delta"] for row in rows])),
                "final_distance_bins": {
                    "le1p25": sum(row["final_distance"] <= 1.25 for row in rows),
                    "1p25to1p5": sum(1.25 < row["final_distance"] <= 1.5 for row in rows),
                    "1p5to2": sum(1.5 < row["final_distance"] <= 2.0 for row in rows),
                    "2to3": sum(2.0 < row["final_distance"] <= 3.0 for row in rows),
                    "gt3": sum(row["final_distance"] > 3.0 for row in rows),
                },
                "near_fraction": {
                    key: float(np.mean([row["near_fraction"][key] for row in rows]))
                    for key in rows[0]["near_fraction"]
                },
                "vertical_path_rmse": float(np.mean([row["vertical_path_rmse"] for row in rows])),
                "horizontal_path_rmse": float(np.mean([row["horizontal_path_rmse"] for row in rows])),
                "residual_abs_mean": float(np.mean([row["residual_abs_mean"] for row in rows])),
                "residual_channel_abs": np.mean(
                    [row["residual_channel_abs"] for row in rows], axis=0
                ).tolist(),
            }
        )
        for name in ("le2", "2to10", "gt10"):
            result[f"residual_rms_{name}"] = float(
                np.mean([row["residual_bin_rms"][name] for row in rows])
            )
        command_rows = [row["command_residual_bin_rms"] for row in rows if row["command_residual_bin_rms"] is not None]
        if command_rows:
            result["command_residual_rms"] = {
                name: float(np.mean([row[name] for row in command_rows]))
                for name in ("le2", "2to10", "gt10")
            }
    return result


def _pool(runs: list[dict[str, Any]]) -> dict[str, Any]:
    if not runs:
        return {}
    return {
        "n": sum(run["n"] for run in runs),
        "success": sum(run["success"] for run in runs),
        "plane_miss": sum(run["plane_miss"] for run in runs),
        "timeout": sum(run["timeout"] for run in runs),
        "safety": sum(run["safety"] for run in runs),
        "finite": all(run.get("finite", True) for run in runs),
        "final_distance_bins": {
            key: sum(run.get("final_distance_bins", {}).get(key, 0) for run in runs)
            for key in ("le1p25", "1p25to1p5", "1p5to2", "2to3", "gt3")
        },
        "final_distance_median": float(np.mean([run.get("final_distance_median", np.nan) for run in runs])),
        "final_distance_p90": float(np.mean([run.get("final_distance_p90", np.nan) for run in runs])),
        "final_vertical_median": float(np.nanmean([run.get("final_vertical_median", np.nan) for run in runs])),
        "final_horizontal_median": float(np.nanmean([run.get("final_horizontal_median", np.nan) for run in runs])),
        "entry_vertical_p90": float(np.nanmean([run.get("entry_vertical_p90", np.nan) for run in runs])),
        "entry_horizontal_p90": float(np.nanmean([run.get("entry_horizontal_p90", np.nan) for run in runs])),
        "closest_distance_median": float(np.nanmean([run.get("closest_distance_median", np.nan) for run in runs])),
        "path_progress_delta_median": float(np.nanmean([run.get("path_progress_delta_median", np.nan) for run in runs])),
        "vertical_path_rmse": float(np.nanmean([run.get("vertical_path_rmse", np.nan) for run in runs])),
        "horizontal_path_rmse": float(np.nanmean([run.get("horizontal_path_rmse", np.nan) for run in runs])),
        "residual_abs_mean": float(np.nanmean([run.get("residual_abs_mean", np.nan) for run in runs])),
        "residual_rms_le2": float(np.nanmean([run.get("residual_rms_le2", np.nan) for run in runs])),
        "residual_rms_2to10": float(np.nanmean([run.get("residual_rms_2to10", np.nan) for run in runs])),
        "residual_rms_gt10": float(np.nanmean([run.get("residual_rms_gt10", np.nan) for run in runs])),
        "policy_action_saturation_fraction": float(
            np.nanmean(
                [
                    run.get("summary", {}).get("policy_action_saturation_fraction_mean", np.nan)
                    for run in runs
                ]
            )
        ),
        "material_saturation_fraction": float(
            np.nanmean(
                [
                    run.get("summary", {}).get("material_saturation_fraction_mean", np.nan)
                    for run in runs
                ]
            )
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("artifacts/lowlevel/goal160_action_parameterization_20261112/eval"),
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    root = args.root
    output_dir = args.output or root

    runs: dict[tuple[str, str, str, int, str], dict[str, Any]] = {}
    for arm in ARMS:
        for radius in RADII:
            for level in LEVELS:
                for seed in SEEDS:
                    for mode in MODES:
                        result = _run(root / arm / radius / level / f"{mode}_seed{seed}")
                        if result is not None:
                            runs[(arm, radius, level, seed, mode)] = result

    paired: list[dict[str, Any]] = []
    for arm in ARMS:
        for radius in RADII:
            for level in LEVELS:
                for seed in SEEDS:
                    zero = runs.get((arm, radius, level, seed, "zero"))
                    policy = runs.get((arm, radius, level, seed, "policy"))
                    if zero is None or policy is None:
                        continue
                    paired.append(
                        {
                            "arm": arm,
                            "radius": radius,
                            "level": level,
                            "seed": seed,
                            "zero_success": zero["success"],
                            "policy_success": policy["success"],
                            "delta_success": policy["success"] - zero["success"],
                            "delta_vertical": policy.get("final_vertical_median", np.nan)
                            - zero.get("final_vertical_median", np.nan),
                            "delta_horizontal": policy.get("final_horizontal_median", np.nan)
                            - zero.get("final_horizontal_median", np.nan),
                            "delta_entry_vertical_p90": policy.get("entry_vertical_p90", np.nan)
                            - zero.get("entry_vertical_p90", np.nan),
                            "delta_entry_horizontal_p90": policy.get("entry_horizontal_p90", np.nan)
                            - zero.get("entry_horizontal_p90", np.nan),
                            "policy_residual_rms_le2": policy.get("residual_rms_le2"),
                            "policy_residual_rms_2to10": policy.get("residual_rms_2to10"),
                            "policy_residual_rms_gt10": policy.get("residual_rms_gt10"),
                        }
                    )

    pooled: dict[tuple[str, str, str], dict[str, Any]] = {}
    for arm in ARMS:
        for radius in RADII:
            for level in LEVELS:
                zero_runs = [
                    runs[(arm, radius, level, seed, "zero")]
                    for seed in SEEDS
                    if (arm, radius, level, seed, "zero") in runs
                ]
                policy_runs = [
                    runs[(arm, radius, level, seed, "policy")]
                    for seed in SEEDS
                    if (arm, radius, level, seed, "policy") in runs
                ]
                if not zero_runs or not policy_runs:
                    continue
                zero = _pool(zero_runs)
                policy = _pool(policy_runs)
                pooled[(arm, radius, level)] = {
                    "zero": zero,
                    "policy": policy,
                    "delta_success": policy["success"] - zero["success"],
                    "seeds_nonnegative": sum(
                        row["delta_success"] >= 0
                        for row in paired
                        if row["arm"] == arm
                        and row["radius"] == radius
                        and row["level"] == level
                    ),
                    "delta_path3d": policy["vertical_path_rmse"] - zero["vertical_path_rmse"],
                    "delta_path_horizontal": policy["horizontal_path_rmse"] - zero["horizontal_path_rmse"],
                    "delta_vertical": policy["final_vertical_median"] - zero["final_vertical_median"],
                    "delta_horizontal": policy["final_horizontal_median"] - zero["final_horizontal_median"],
                    "delta_entry_vertical_p90": policy["entry_vertical_p90"] - zero["entry_vertical_p90"],
                    "delta_entry_horizontal_p90": policy["entry_horizontal_p90"] - zero["entry_horizontal_p90"],
                }

    gate: dict[str, Any] = {}
    for radius in RADII:
        baseline = pooled.get(("A_actuator_near10_2", radius, PRIMARY_LEVEL))
        if not baseline:
            gate[radius] = {"status": "INCOMPLETE"}
            continue
        candidates = []
        for arm in ("B_command_near10_2", "C_command_terminal_train"):
            item = pooled.get((arm, radius, PRIMARY_LEVEL))
            if item:
                candidates.append((arm, item))
        if not candidates:
            gate[radius] = {"status": "INCOMPLETE"}
            continue
        best_arm, best = max(candidates, key=lambda pair: pair[1]["policy"]["success"])
        other_ok = all(
            pooled.get((best_arm, radius, level), {}).get("delta_success", -10**9) >= -2
            for level in OTHER_LEVELS
        )
        path_ok = (
            best["delta_path3d"] <= 0.05 + 1e-12
            and best["delta_vertical"] <= 0.05 + 1e-12
        )
        entry_ok = (
            best["delta_entry_vertical_p90"] < 0.0
            or best["delta_entry_horizontal_p90"] < 0.0
        )
        policy_bins = best["policy"].get("final_distance_bins", {})
        zero_bins = best["zero"].get("final_distance_bins", {})
        policy_le2 = sum(
            policy_bins.get(key, 0) for key in ("le1p25", "1p25to1p5", "1p5to2")
        )
        zero_le2 = sum(
            zero_bins.get(key, 0) for key in ("le1p25", "1p25to1p5", "1p5to2")
        )
        bins_ok = (
            policy_le2 >= zero_le2
            and policy_bins.get("gt3", 0) <= zero_bins.get("gt3", 0)
        )
        risk_ok = (
            best["policy"].get("timeout", 0) <= best["zero"].get("timeout", 0)
            and best["policy"].get("safety", 0) <= best["zero"].get("safety", 0)
            and best["policy"].get("finite", False)
        )
        residual_le2 = float(best["policy"].get("residual_rms_le2", np.nan))
        residual_gt10 = float(best["policy"].get("residual_rms_gt10", np.nan))
        residual_ok = bool(
            np.isfinite(residual_le2)
            and np.isfinite(residual_gt10)
            and residual_le2 > 1e-6
            and residual_gt10 <= 0.1 * residual_le2 + 1e-6
        )
        gate[radius] = {
            "best_command_arm": best_arm,
            "delta_vs_zero": best["delta_success"],
            "delta_vs_actuator": best["policy"]["success"] - baseline["policy"]["success"],
            "seeds_nonnegative_vs_zero": best["seeds_nonnegative"],
            "other_levels_guard": bool(other_ok),
            "path_guard": bool(path_ok),
            "entry_improved": bool(entry_ok),
            "final_distance_bins_ok": bool(bins_ok),
            "residual_active": bool(residual_ok),
            "risk_ok": bool(risk_ok),
            "policy_le2": policy_le2,
            "zero_le2": zero_le2,
            "policy_gt3": policy_bins.get("gt3", 0),
            "zero_gt3": zero_bins.get("gt3", 0),
            "policy_timeout": best["policy"].get("timeout", 0),
            "zero_timeout": best["zero"].get("timeout", 0),
            "policy_safety": best["policy"].get("safety", 0),
            "zero_safety": best["zero"].get("safety", 0),
            "residual_rms_le2": residual_le2,
            "residual_rms_gt10": residual_gt10,
            "delta_path3d": best["delta_path3d"],
            "delta_vertical": best["delta_vertical"],
            "delta_entry_vertical_p90": best["delta_entry_vertical_p90"],
            "delta_entry_horizontal_p90": best["delta_entry_horizontal_p90"],
        }
        threshold_default = 4
        threshold_strict = 2
        threshold = threshold_default if radius == "default" else threshold_strict
        gate[radius]["pass"] = bool(
            best["delta_success"] >= threshold
            and (best["policy"]["success"] - baseline["policy"]["success"]) >= threshold
            and best["seeds_nonnegative"] >= 2
            and other_ok
            and path_ok
            and entry_ok
            and bins_ok
            and residual_ok
            and risk_ok
        )

    payload = {
        "root": str(root),
        "arms": ARMS,
        "radii": RADII,
        "levels": LEVELS,
        "seeds": SEEDS,
        "runs": [
            {"arm": key[0], "radius": key[1], "level": key[2], "seed": key[3], "mode": key[4], **value}
            for key, value in runs.items()
        ],
        "paired": paired,
        "pooled": [
            {"arm": key[0], "radius": key[1], "level": key[2], **value}
            for key, value in pooled.items()
        ],
        "gate": gate,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "audit_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    lines = [
        "# Goal160 action-parameterization screen",
        "",
        "H3 controller, dev seeds 20–59, C3/ALOS, `direct_commands_active=false`.",
        "",
        "| arm | radius | level | zero | policy | Δ | seeds nonneg | path v Δ | path h Δ | entry v p90 Δ | entry h p90 Δ | residual <=2 / 2-10 / >10 |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for arm in ARMS:
        for radius in RADII:
            for level in LEVELS:
                item = pooled.get((arm, radius, level))
                if not item:
                    continue
                lines.append(
                    "| {arm} | {radius} | {level} | {z}/{n} | {p}/{n} | {d} | {nn}/3 | {dv} | {dh} | {ev} | {eh} | {r1} / {r2} / {r3} |".format(
                        arm=arm,
                        radius=radius,
                        level=level,
                        z=item["zero"]["success"],
                        p=item["policy"]["success"],
                        n=item["zero"]["n"],
                        d=item["delta_success"],
                        nn=item["seeds_nonnegative"],
                        dv=_fmt(item["delta_path3d"]),
                        dh=_fmt(item["delta_path_horizontal"]),
                        ev=_fmt(item["delta_entry_vertical_p90"]),
                        eh=_fmt(item["delta_entry_horizontal_p90"]),
                        r1=_fmt(item["policy"].get("residual_rms_le2")),
                        r2=_fmt(item["policy"].get("residual_rms_2to10")),
                        r3=_fmt(item["policy"].get("residual_rms_gt10")),
                    )
                )

    lines += ["", "## Gate", "", "| radius | best arm | Δ vs zero | Δ vs actuator | seeds | path guard | entry improved | bins ok | residual active | risk ok | pass |", "|---|---|---:|---:|---:|---|---|---|---|---|---|"]
    for radius, item in gate.items():
        lines.append(
            "| {radius} | {arm} | {dz} | {da} | {s}/3 | {pg} | {eg} | {bg} | {rg} | {rk} | {pass_} |".format(
                radius=radius,
                arm=item.get("best_command_arm", "NA"),
                dz=item.get("delta_vs_zero", "NA"),
                da=item.get("delta_vs_actuator", "NA"),
                s=item.get("seeds_nonnegative_vs_zero", "NA"),
                pg=item.get("path_guard", "NA"),
                eg=item.get("entry_improved", "NA"),
                bg=item.get("final_distance_bins_ok", "NA"),
                rg=item.get("residual_active", "NA"),
                rk=item.get("risk_ok", "NA"),
                pass_=item.get("pass", "NA"),
            )
        )
    lines += [
        "",
        "## Per-seed",
        "",
        "| arm | radius | level | seed | zero | policy | Δ | Δv | Δh | Δentry v | Δentry h |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in paired:
        lines.append(
            "| {arm} | {radius} | {level} | {seed} | {z} | {p} | {d} | {dv} | {dh} | {ev} | {eh} |".format(
                arm=row["arm"],
                radius=row["radius"],
                level=row["level"],
                seed=row["seed"],
                z=row["zero_success"],
                p=row["policy_success"],
                d=row["delta_success"],
                dv=_fmt(row["delta_vertical"]),
                dh=_fmt(row["delta_horizontal"]),
                ev=_fmt(row["delta_entry_vertical_p90"]),
                eh=_fmt(row["delta_entry_horizontal_p90"]),
            )
        )
    (output_dir / "audit_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print((output_dir / "audit_summary.md").read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
