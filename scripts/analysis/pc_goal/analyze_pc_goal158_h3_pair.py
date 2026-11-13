#!/usr/bin/env python3
"""Analyze Goal158 H3 zero vs Goal86 1M near10_2 paired evaluations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


LEVELS = ("nominal", "current07", "current10", "current15")
SEEDS = (86201, 86202, 86203)
MODES = ("zero", "policy")


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "NA"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    return f"{float(value):.{digits}f}"


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
        np.asarray(trace["learned_residual_action"], dtype=np.float64)
        if "learned_residual_action" in trace.files
        else np.zeros((len(seeds), 3))
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
        # Prefer the first done step; fall back to the last sampled step.
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
            progress_delta = float(
                np.nanmax(progress_values) - np.nanmin(progress_values)
            )
        local_residual = residual[index]
        bins = {
            "le2": residual[index[distance[index] <= 2.0]],
            "2to10": residual[index[(distance[index] > 2.0) & (distance[index] <= 10.0)]],
            "gt10": residual[index[distance[index] > 10.0]],
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
                "vertical_path_rmse": float(
                    np.sqrt(np.nanmean(np.square(vertical_path[index])))
                ),
                "horizontal_path_rmse": float(
                    np.sqrt(np.nanmean(np.square(horizontal_path[index])))
                ),
                "residual_abs_mean": float(np.mean(np.abs(local_residual))),
                "residual_channel_abs": [
                    float(np.mean(np.abs(local_residual[:, channel])))
                    for channel in range(3)
                ],
                "residual_bin_rms": {
                    name: (float(np.sqrt(np.mean(np.square(values)))) if values.size else 0.0)
                    for name, values in bins.items()
                },
            }
        )
    return rows


def _run(root: Path, level: str, seed: int, mode: str) -> dict[str, Any] | None:
    run_dir = root / level / f"{mode}_seed{seed}"
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
        "level": level,
        "seed": seed,
        "mode": mode,
        "n": len(episodes),
        "success": sum(
            bool(item.get("success_eta_tight_at_done", False)) for item in episodes
        ),
        "plane_miss": sum(bool(item.get("goal_plane_miss", False)) for item in episodes),
        "timeout": sum(bool(item.get("timeout", False)) for item in episodes),
        "safety": sum(bool(item.get("safety", False)) for item in episodes),
    }
    if rows:
        result["final_distance_median"] = float(
            np.median([row["final_distance"] for row in rows])
        )
        result["final_vertical_median"] = float(
            np.median([row["final_vertical"] for row in rows])
        )
        result["final_horizontal_median"] = float(
            np.median([row["final_horizontal"] for row in rows])
        )
        result["entry_vertical_p90"] = float(
            np.nanpercentile([row["entry_vertical"] for row in rows], 90)
        )
        result["entry_horizontal_p90"] = float(
            np.nanpercentile([row["entry_horizontal"] for row in rows], 90)
        )
        result["final_distance_p90"] = float(
            np.nanpercentile([row["final_distance"] for row in rows], 90)
        )
        result["closest_distance_median"] = float(
            np.nanmedian([row["closest_distance"] for row in rows])
        )
        result["path_progress_delta_median"] = float(
            np.nanmedian([row["path_progress_delta"] for row in rows])
        )
        result["final_distance_bins"] = {
            "le1p25": sum(row["final_distance"] <= 1.25 for row in rows),
            "1p25to1p5": sum(1.25 < row["final_distance"] <= 1.5 for row in rows),
            "1p5to2": sum(1.5 < row["final_distance"] <= 2.0 for row in rows),
            "2to3": sum(2.0 < row["final_distance"] <= 3.0 for row in rows),
            "gt3": sum(row["final_distance"] > 3.0 for row in rows),
        }
        result["near_fraction"] = {
            key: float(np.mean([row["near_fraction"][key] for row in rows]))
            for key in rows[0]["near_fraction"]
        }
        result["vertical_path_rmse"] = float(
            np.mean([row["vertical_path_rmse"] for row in rows])
        )
        result["horizontal_path_rmse"] = float(
            np.mean([row["horizontal_path_rmse"] for row in rows])
        )
        result["residual_abs_mean"] = float(
            np.mean([row["residual_abs_mean"] for row in rows])
        )
        result["residual_channel_abs"] = np.mean(
            [row["residual_channel_abs"] for row in rows], axis=0
        ).tolist()
        for name in ("le2", "2to10", "gt10"):
            result[f"residual_rms_{name}"] = float(
                np.mean([row["residual_bin_rms"][name] for row in rows])
            )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("artifacts/lowlevel/goal158_h3_neural_pair_20261112"),
    )
    args = parser.parse_args()
    root = args.root

    runs: dict[tuple[str, int, str], dict[str, Any]] = {}
    for level in LEVELS:
        for seed in SEEDS:
            for mode in MODES:
                result = _run(root, level, seed, mode)
                if result is not None:
                    runs[(level, seed, mode)] = result

    paired: list[dict[str, Any]] = []
    for level in LEVELS:
        for seed in SEEDS:
            zero = runs.get((level, seed, "zero"))
            policy = runs.get((level, seed, "policy"))
            if zero is None or policy is None:
                continue
            paired.append(
                {
                    "level": level,
                    "seed": seed,
                    "zero_success": zero["success"],
                    "policy_success": policy["success"],
                    "delta_success": policy["success"] - zero["success"],
                    "delta_vertical": (
                        policy.get("final_vertical_median", float("nan"))
                        - zero.get("final_vertical_median", float("nan"))
                    ),
                    "delta_horizontal": (
                        policy.get("final_horizontal_median", float("nan"))
                        - zero.get("final_horizontal_median", float("nan"))
                    ),
                    "policy_residual_abs_mean": policy.get("residual_abs_mean"),
                    "policy_residual_rms_le2": policy.get("residual_rms_le2"),
                    "policy_residual_rms_2to10": policy.get("residual_rms_2to10"),
                    "policy_residual_rms_gt10": policy.get("residual_rms_gt10"),
                }
            )

    pooled: dict[str, Any] = {}
    for level in LEVELS:
        level_rows = [row for row in paired if row["level"] == level]
        if not level_rows:
            continue
        pooled[level] = {
            "zero_success": sum(row["zero_success"] for row in level_rows),
            "policy_success": sum(row["policy_success"] for row in level_rows),
            "delta_success": sum(row["delta_success"] for row in level_rows),
            "seeds_nonnegative": sum(row["delta_success"] >= 0 for row in level_rows),
            "n_seeds": len(level_rows),
            "mean_delta_vertical": float(
                np.nanmean([row["delta_vertical"] for row in level_rows])
            ),
            "mean_delta_horizontal": float(
                np.nanmean([row["delta_horizontal"] for row in level_rows])
            ),
        }

    output = {"root": str(root), "runs": list(runs.values()), "paired": paired, "pooled": pooled}
    root.mkdir(parents=True, exist_ok=True)
    (root / "audit_summary.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    lines = [
        "# Goal158 H3 strong baseline vs Goal86 1M near10_2",
        "",
        "C1, H3 controller, dev seeds 20–59, strict125, success frame `eta_tight_at_done`.",
        "",
        "| level | zero | policy | Δsuccess | seeds nonneg | zero v med | policy v med | Δv | zero h med | policy h med | Δh | residual le2 / 2-10 / >10 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for level in LEVELS:
        rows = [row for row in paired if row["level"] == level]
        if not rows:
            continue
        zero = runs[(level, 86201, "zero")] if (level, 86201, "zero") in runs else {}
        policy = runs[(level, 86201, "policy")] if (level, 86201, "policy") in runs else {}
        pool = pooled[level]
        lines.append(
            "| {level} | {zs} | {ps} | {ds} | {nn}/{ns} | {zv} | {pv} | {dv} | {zh} | {ph} | {dh} | {r1} / {r2} / {r3} |".format(
                level=level,
                zs=f"{pool['zero_success']}/{pool['n_seeds']*40}",
                ps=f"{pool['policy_success']}/{pool['n_seeds']*40}",
                ds=pool["delta_success"],
                nn=pool["seeds_nonnegative"],
                ns=pool["n_seeds"],
                zv=_fmt(zero.get("final_vertical_median")),
                pv=_fmt(policy.get("final_vertical_median")),
                dv=_fmt(pool["mean_delta_vertical"]),
                zh=_fmt(zero.get("final_horizontal_median")),
                ph=_fmt(policy.get("final_horizontal_median")),
                dh=_fmt(pool["mean_delta_horizontal"]),
                r1=_fmt(policy.get("residual_rms_le2")),
                r2=_fmt(policy.get("residual_rms_2to10")),
                r3=_fmt(policy.get("residual_rms_gt10")),
            )
        )

    lines.extend(
        [
            "",
            "## Route / terminal health",
            "",
            "Final distance bins count episodes, not steps. `entry` is the first step with eta distance <= 2 m; `progress` is the trace path-progress range.",
            "",
            "| level | mode | seed | final d p50/p90 | <=1.25/1.25-1.5/1.5-2/2-3/>3 | entry v p90 | entry h p90 | closest p50 | progress | near<=1.25/2/3 | path v/h RMSE |",
            "|---|---|---:|---|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for level in LEVELS:
        for seed in SEEDS:
            for mode in MODES:
                run = runs.get((level, seed, mode))
                if run is None:
                    continue
                bins = run.get("final_distance_bins", {})
                near = run.get("near_fraction", {})
                lines.append(
                    "| {level} | {mode} | {seed} | {p50}/{p90} | {b1}/{b2}/{b3}/{b4}/{b5} | {ev} | {eh} | {cl} | {prog} | {n1}/{n2}/{n3} | {pv}/{ph} |".format(
                        level=level,
                        mode=mode,
                        seed=seed,
                        p50=_fmt(run.get("final_distance_median")),
                        p90=_fmt(run.get("final_distance_p90")),
                        b1=bins.get("le1p25", 0),
                        b2=bins.get("1p25to1p5", 0),
                        b3=bins.get("1p5to2", 0),
                        b4=bins.get("2to3", 0),
                        b5=bins.get("gt3", 0),
                        ev=_fmt(run.get("entry_vertical_p90")),
                        eh=_fmt(run.get("entry_horizontal_p90")),
                        cl=_fmt(run.get("closest_distance_median")),
                        prog=_fmt(run.get("path_progress_delta_median")),
                        n1=_fmt(near.get("le1p25")),
                        n2=_fmt(near.get("le2p0")),
                        n3=_fmt(near.get("le3p0")),
                        pv=_fmt(run.get("vertical_path_rmse")),
                        ph=_fmt(run.get("horizontal_path_rmse")),
                    )
                )
    lines.append("")

    lines += [
        "",
        "## Per-seed",
        "",
        "| level | seed | zero | policy | Δ | Δv | Δh | residual abs |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in paired:
        lines.append(
            "| {level} | {seed} | {z} | {p} | {d} | {dv} | {dh} | {r} |".format(
                level=row["level"],
                seed=row["seed"],
                z=row["zero_success"],
                p=row["policy_success"],
                d=row["delta_success"],
                dv=_fmt(row["delta_vertical"]),
                dh=_fmt(row["delta_horizontal"]),
                r=_fmt(row["policy_residual_abs_mean"]),
            )
        )
    (root / "audit_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print((root / "audit_summary.md").read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
