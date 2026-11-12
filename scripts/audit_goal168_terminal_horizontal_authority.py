#!/usr/bin/env python3
"""Read-only audit of Goal167 terminal horizontal goal-plane misses."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

RUDDER_LIMIT_RAD = math.radians(20.0)
CRAB_LIMIT_RAD = math.radians(18.0)


def _finite(value: Any) -> float | None:
    value = float(value)
    if not math.isfinite(value):
        return None
    return value


def _p90(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan")
    return float(np.percentile(values, 90))


def _mean(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan")
    return float(np.mean(values))


def _wrap(angle: np.ndarray) -> np.ndarray:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def _episode_rows(trace: dict[str, np.ndarray], seed: int) -> dict[str, np.ndarray]:
    index = np.flatnonzero(np.asarray(trace["seed"], dtype=np.int64) == int(seed))
    index = index[np.argsort(np.asarray(trace["step"])[index], kind="stable")]
    return {name: np.asarray(value)[index] for name, value in trace.items()}


def _audit_episode(rows: dict[str, np.ndarray], seed: int) -> dict[str, Any]:
    time_s = np.asarray(rows["time_s"], dtype=np.float64)
    distance = np.asarray(rows["goal_distance_eta"], dtype=np.float64)
    horizontal = np.asarray(rows["path_horizontal_error"], dtype=np.float64)
    vertical = np.asarray(rows["path_vertical_error"], dtype=np.float64)
    progress = np.asarray(rows["path_progress"], dtype=np.float64)
    ground_speed = np.linalg.norm(
        np.asarray(rows["ground_velocity_body_mps"], dtype=np.float64), axis=1
    )
    ground_velocity = np.asarray(rows["ground_velocity_body_mps"], dtype=np.float64)
    water_velocity = np.asarray(rows["water_velocity_body_mps"], dtype=np.float64)
    current_speed = np.linalg.norm(ground_velocity - water_velocity, axis=1)
    eta = np.asarray(rows["eta_ned"], dtype=np.float64)
    goal = np.asarray(rows["goal_ned"], dtype=np.float64)
    yaw = np.asarray(rows["eta_rpy"], dtype=np.float64)[:, 2]
    command = np.asarray(rows["command_physical"], dtype=np.float64)
    rudder = np.asarray(rows["actual_actuator"], dtype=np.float64)[:, 0]
    terminal_phase = np.asarray(rows["guidance_terminal_phase"], dtype=bool)
    crab = np.asarray(rows["guidance_crab_angle_rad"], dtype=np.float64)

    yaw_rate = np.zeros_like(yaw)
    if yaw.size >= 2:
        dt = np.diff(time_s)
        dt = np.where(dt > 1e-9, dt, np.nan)
        yaw_rate[:-1] = _wrap(np.diff(np.unwrap(yaw))) / dt

    goal_bearing = np.arctan2(goal[:, 1] - eta[:, 1], goal[:, 0] - eta[:, 0])
    heading_goal_error = np.abs(_wrap(goal_bearing - yaw))
    command_goal_error = np.abs(_wrap(goal_bearing - command[:, 0]))

    mask = (distance >= 2.0) & (distance <= 20.0)
    if int(mask.sum()) < 5:
        mask = distance <= 20.0
    along = np.sqrt(np.maximum(distance * distance - horizontal * horizontal - vertical * vertical, 0.0))
    time_to_go = along / np.maximum(ground_speed, 0.1)
    omega_req = np.arctan2(np.abs(horizontal), np.maximum(along, 1e-6)) / np.maximum(
        time_to_go, 1e-6
    )
    accel_req = 2.0 * np.abs(horizontal) / np.maximum(time_to_go * time_to_go, 1e-6)

    omega_available = _p90(np.abs(yaw_rate[mask]))
    speed_available = _p90(ground_speed[mask])
    accel_available = speed_available * omega_available
    omega_required = _p90(omega_req[mask])
    accel_required = _p90(accel_req[mask])
    omega_margin = omega_available / omega_required if omega_required > 1e-9 else float("inf")
    accel_margin = accel_available / accel_required if accel_required > 1e-9 else float("inf")
    margin = min(omega_margin, accel_margin)

    terminal_fraction = _mean(terminal_phase[mask].astype(np.float64))
    projection_jump = 0.0
    if int(mask.sum()) >= 2:
        indices = np.flatnonzero(mask)
        projection_jump = float(np.max(np.abs(np.diff(progress[indices]))))
    rudder_sat = _mean((np.abs(rudder[mask]) >= 0.90 * RUDDER_LIMIT_RAD).astype(np.float64))
    crab_sat = _mean((np.abs(crab[mask]) >= 0.90 * CRAB_LIMIT_RAD).astype(np.float64))

    entries: dict[str, Any] = {}
    for threshold in (20.0, 15.0, 10.0, 5.0):
        candidates = np.flatnonzero(distance <= threshold)
        if candidates.size == 0:
            entries[f"entry_{int(threshold)}m"] = None
            continue
        index = int(candidates[0])
        entry_omega = float(omega_req[index])
        entry_margin = (
            omega_available / entry_omega if entry_omega > 1e-9 else float("inf")
        )
        entry_time = float(time_to_go[index])
        rotate_time = (
            float(command_goal_error[index]) / max(omega_available, 1e-9)
            if omega_available > 1e-9
            else float("inf")
        )
        entries[f"entry_{int(threshold)}m"] = {
            "horizontal_error_m": _finite(abs(horizontal[index])),
            "along_track_m": _finite(along[index]),
            "time_to_plane_s": _finite(entry_time),
            "omega_required_rad_s": _finite(entry_omega),
            "authority_margin": _finite(entry_margin),
            "terminal_phase": bool(terminal_phase[index]),
            "command_to_goal_error_rad": _finite(command_goal_error[index]),
            "rotate_time_ratio": _finite(
                rotate_time / entry_time if entry_time > 1e-9 else float("inf")
            ),
            "rudder_rad": _finite(rudder[index]),
        }

    band_20_15 = (distance <= 20.0) & (distance >= 15.0)
    phase_duty_20_15 = _mean(terminal_phase[band_20_15].astype(np.float64))
    phase_transitions = int(np.sum(terminal_phase[1:] != terminal_phase[:-1]))
    entry_20 = entries.get("entry_20m") or {}
    entry_20_margin = entry_20.get("authority_margin")
    entry_20_command_error = entry_20.get("command_to_goal_error_rad")
    entry_20_rotate_ratio = entry_20.get("rotate_time_ratio")

    if projection_jump > 5.0:
        classification = "PROJECTION_JUMP"
    elif entry_20_margin is not None and entry_20_margin < 1.0:
        classification = "AUTHORITY_INSUFFICIENT_AT_20M"
    elif (
        entry_20_command_error is not None
        and entry_20_command_error > 0.35
        and phase_duty_20_15 >= 0.80
        and entry_20_rotate_ratio is not None
        and entry_20_rotate_ratio > 0.80
    ):
        classification = "TERMINAL_HEADING_LAG"
    elif terminal_fraction < 0.90:
        classification = "GUIDANCE_NOT_ACTIVE"
    elif entry_20_margin is not None and entry_20_margin >= 1.0:
        classification = "AUTHORITY_SUFFICIENT"
    else:
        classification = "UNCLASSIFIED"

    return {
        "seed": int(seed),
        "steps_le20m": int(mask.sum()),
        "terminal_phase_fraction": _finite(terminal_fraction),
        "projection_jump_max_m": _finite(projection_jump),
        "horizontal_p90_m": _finite(_p90(np.abs(horizontal[mask]))),
        "vertical_p90_m": _finite(_p90(np.abs(vertical[mask]))),
        "ground_speed_p90_mps": _finite(speed_available),
        "yaw_rate_available_p90_rad_s": _finite(omega_available),
        "yaw_rate_required_p90_rad_s": _finite(omega_required),
        "yaw_rate_margin": _finite(omega_margin),
        "lateral_accel_available_p90_mps2": _finite(accel_available),
        "lateral_accel_required_p90_mps2": _finite(accel_required),
        "lateral_accel_margin": _finite(accel_margin),
        "authority_margin": _finite(margin),
        "rudder_saturation_fraction": _finite(rudder_sat),
        "crab_saturation_fraction": _finite(crab_sat),
        "current_speed_mean_mps": _finite(_mean(current_speed[mask])),
        "heading_to_goal_error_p90_rad": _finite(_p90(heading_goal_error[mask])),
        "command_to_goal_error_p90_rad": _finite(_p90(command_goal_error[mask])),
        "phase_duty_20_15": _finite(phase_duty_20_15),
        "phase_transitions": phase_transitions,
        "entries": entries,
        "classification": classification,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arms", nargs="+", default=["A_actuator_terminal", "B_command_terminal"])
    parser.add_argument("--radius", default="default")
    parser.add_argument("--level", default="current10")
    parser.add_argument("--seeds", nargs="+", type=int, default=[86201, 86202, 86203])
    args = parser.parse_args()

    result: dict[str, Any] = {
        "schema": "goal168_terminal_horizontal_authority.v1",
        "eval_root": str(args.eval_root),
        "radius": args.radius,
        "level": args.level,
        "arms": {},
    }
    for arm in args.arms:
        arm_rows: list[dict[str, Any]] = []
        for seed in args.seeds:
            run_dir = args.eval_root / arm / args.radius / args.level / f"policy_seed{seed}"
            episodes = json.loads((run_dir / "episodes.json").read_text(encoding="utf-8"))
            trace = np.load(run_dir / "trace.npz", allow_pickle=False)
            failures = [row for row in episodes if not bool(row.get("success", False))]
            for episode in failures:
                episode_seed = int(episode["seed"])
                rows = _episode_rows(dict(trace), episode_seed)
                audit = _audit_episode(rows, episode_seed)
                audit.update(
                    {
                        "arm": arm,
                        "policy_seed": int(seed),
                        "terminal_kind": int(episode.get("terminal_kind", -1)),
                        "final_distance_m": _finite(episode.get("terminal_eta_distance_m", math.nan)),
                        "success": bool(episode.get("success", False)),
                    }
                )
                arm_rows.append(audit)
        counts: dict[str, int] = {}
        for row in arm_rows:
            counts[row["classification"]] = counts.get(row["classification"], 0) + 1
        result["arms"][arm] = {"episodes": arm_rows, "classification_counts": counts}

    args.output.mkdir(parents=True, exist_ok=True)
    json_path = args.output / "audit.json"
    json_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "# Goal168 terminal horizontal capture authority audit",
        "",
        f"- eval root: `{args.eval_root}`",
        f"- radius/level: `{args.radius}` / `{args.level}`",
        "",
    ]
    for arm, payload in result["arms"].items():
        lines.extend(
            [
                f"## {arm}",
                "",
                "| policy seed | episode seed | final d | h p90 | v p90 | terminal frac | phase duty 20-15 | proj jump | entry20 margin | entry20 cmd err | rotate ratio | omega avail | rudder sat | class |",
                "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
            ]
        )
        for row in payload["episodes"]:
            entry = row.get("entries", {}).get("entry_20m") or {}
            lines.append(
                "| {policy_seed} | {seed} | {final:.3f} | {h:.3f} | {v:.3f} | "
                "{term:.3f} | {duty:.3f} | {jump:.3f} | {margin:.2f} | {cmd:.2f} | "
                "{rot:.2f} | {oa:.3f} | {rs:.3f} | {cls} |".format(
                    policy_seed=row["policy_seed"],
                    seed=row["seed"],
                    final=row["final_distance_m"] if row["final_distance_m"] is not None else float("nan"),
                    h=row["horizontal_p90_m"] if row["horizontal_p90_m"] is not None else float("nan"),
                    v=row["vertical_p90_m"] if row["vertical_p90_m"] is not None else float("nan"),
                    term=row["terminal_phase_fraction"] if row["terminal_phase_fraction"] is not None else float("nan"),
                    duty=row["phase_duty_20_15"] if row["phase_duty_20_15"] is not None else float("nan"),
                    jump=row["projection_jump_max_m"] if row["projection_jump_max_m"] is not None else float("nan"),
                    margin=entry.get("authority_margin", float("nan")),
                    cmd=entry.get("command_to_goal_error_rad", float("nan")),
                    rot=entry.get("rotate_time_ratio", float("nan")),
                    oa=row["yaw_rate_available_p90_rad_s"] if row["yaw_rate_available_p90_rad_s"] is not None else float("nan"),
                    rs=row["rudder_saturation_fraction"] if row["rudder_saturation_fraction"] is not None else float("nan"),
                    cls=row["classification"],
                )
            )
        lines.extend(["", f"Classification counts: `{payload['classification_counts']}`", ""])
    (args.output / "audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json_path)
    print(args.output / "audit.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
