from __future__ import annotations

import argparse
import json
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from auv_lowlevel.config import load_config
from auv_lowlevel.evaluate import Rollout, run_rollout
from auv_lowlevel.path import generate_sinusoidal_3d_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the ODARRL low-level B0 baseline")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/lowlevel/odarrl_control_base.json"),
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--seeds",
        default=None,
        help="Comma/range seed list, for example 0-49 or 0,2,4-6; defaults to the config pool",
    )
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--no-plot", action="store_true")
    return parser.parse_args()


def _parse_seeds(text: str | None, default_start: int, default_count: int) -> tuple[int, ...]:
    if text is None:
        return tuple(range(default_start, default_start + default_count))
    seeds: list[int] = []
    for token in text.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            start_text, end_text = token.split("-", maxsplit=1)
            start, end = int(start_text), int(end_text)
            if end < start:
                raise ValueError("seed range end must not precede start")
            seeds.extend(range(start, end + 1))
        else:
            seeds.append(int(token))
    if not seeds or len(set(seeds)) != len(seeds) or min(seeds) < 0:
        raise ValueError("seeds must be non-empty, unique, and non-negative")
    return tuple(seeds)


def _run_worker(payload: tuple[str, int | None, int, str]) -> dict[str, Any]:
    config_path, max_steps, seed, output_text = payload
    config = load_config(config_path)
    if max_steps is not None:
        config = replace(
            config,
            plant=replace(config.plant, max_episode_steps=max_steps),
        )
    rollout = run_rollout(config, seed)
    output = Path(output_text) / "per_seed"
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / f"seed_{seed:05d}.npz", **_rollout_arrays(rollout))
    summary = rollout.summary()
    (output / f"seed_{seed:05d}.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return summary


def _rollout_arrays(rollout: Rollout) -> dict[str, np.ndarray]:
    return {
        "t_s": rollout.t_s,
        "position_ned_m": rollout.position_ned_m,
        "reference_point_ned_m": rollout.reference_point_ned_m,
        "horizontal_error_m": rollout.horizontal_error_m,
        "vertical_error_m": rollout.vertical_error_m,
        "error_3d_m": rollout.error_3d_m,
        "progress_m": rollout.progress_m,
        "yaw_error_rad": rollout.yaw_error_rad,
        "water_surge_mps": rollout.water_surge_mps,
        "reference_speed_mps": rollout.reference_speed_mps,
        "current_ned_mps": rollout.current_ned_mps,
        "observation": rollout.observation,
        "final_observation": rollout.final_observation,
        "base_command": rollout.base_command,
        "residual_normalized": rollout.residual_normalized,
        "delta_command": rollout.delta_command,
        "requested_total": rollout.requested_total,
        "limited_total": rollout.limited_total,
        "actual_actuator": rollout.actual_actuator,
        "reward_total": rollout.reward_total,
        "reward_ppo_scaled": rollout.reward_ppo_scaled,
        "terminated": rollout.terminated,
        "truncated": rollout.truncated,
        "saturated": rollout.saturated,
        "constraint_violation_count": rollout.constraint_violation_count,
        "pvs_step_latency_s": rollout.pvs_step_latency_s,
        "control_loop_latency_s": rollout.control_loop_latency_s,
        "shadow_teacher_margin_rpm": rollout.shadow_teacher_margin_rpm,
    }


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[1],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _write_markdown(summary: dict[str, Any], output: Path, max_steps: int | None) -> None:
    aggregate = summary["aggregate"]
    lines = [
        "# ODARRL low-level baseline",
        "",
        f"- config: `{summary['config_path']}`",
        f"- config_sha256: `{summary['config_sha256']}`",
        f"- git_commit: `{summary['git_commit']}`",
        f"- seeds: `{summary['seeds']}`",
        f"- max_steps_override: `{max_steps}`",
        "",
        "## Aggregate",
        "",
        "| metric | value |",
        "|---|---:|",
    ]
    for key in (
        "success_rate",
        "three_d_rmse_m_mean",
        "horizontal_rmse_m_mean",
        "vertical_rmse_m_mean",
        "yaw_rmse_rad_mean",
        "speed_error_rmse_mps_mean",
        "arrival_time_s_mean",
        "saturation_rate_mean",
        "constraint_violations_mean",
        "pvs_step_latency_p95_ms_p95",
        "control_loop_latency_p95_ms_p95",
    ):
        value = aggregate[key]
        lines.append(f"| `{key}` | {value:.6g} |" if isinstance(value, float) else f"| `{key}` | {value} |")
    lines += [
        "",
        "## Per seed",
        "",
        "| seed | success | termination | arrival_s | 3D RMSE | horizontal RMSE | vertical RMSE | saturation |",
        "|---:|:---:|---|---:|---:|---:|---:|---:|",
    ]
    for item in summary["episodes"]:
        arrival = "" if item["arrival_time_s"] is None else f"{item['arrival_time_s']:.2f}"
        lines.append(
            f"| {item['seed']} | {item['success']} | {item['termination']} | {arrival} | "
            f"{item['three_d_rmse_m']:.4f} | {item['horizontal_rmse_m']:.4f} | "
            f"{item['vertical_rmse_m']:.4f} | {item['saturation_rate']:.4f} |"
        )
    (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _plot(output: Path, config_path: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is unavailable; skipping plots", flush=True)
        return

    path = generate_sinusoidal_3d_path(load_config(config_path).reference)
    files = sorted((output / "per_seed").glob("seed_*.npz"))
    if not files:
        return
    plots = output / "plots"
    plots.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(path.points_ned_m[:, 1], path.points_ned_m[:, 0], "k--", lw=2, label="reference")
    for file in files:
        data = np.load(file)
        ax.plot(data["position_ned_m"][:, 1], data["position_ned_m"][:, 0], lw=0.8, alpha=0.45)
    ax.set_xlabel("east / m")
    ax.set_ylabel("north / m")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots / "trajectories_xy.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(path.points_ned_m[:, 0], path.points_ned_m[:, 2], "k--", lw=2, label="reference")
    for file in files:
        data = np.load(file)
        ax.plot(data["position_ned_m"][:, 0], data["position_ned_m"][:, 2], lw=0.8, alpha=0.45)
    ax.invert_yaxis()
    ax.set_xlabel("north / m")
    ax.set_ylabel("down / m")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots / "trajectories_xz.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    for file in files[:5]:
        data = np.load(file)
        axes[0].plot(data["t_s"], data["horizontal_error_m"], lw=0.8)
        axes[1].plot(data["t_s"], data["vertical_error_m"], lw=0.8)
        axes[2].plot(data["t_s"], data["error_3d_m"], lw=0.8)
    for axis, label in zip(
        axes,
        ("horizontal error / m", "vertical error / m", "3D error / m"),
        strict=True,
    ):
        axis.set_ylabel(label)
        axis.grid(True, alpha=0.3)
    axes[-1].set_xlabel("time / s")
    fig.tight_layout()
    fig.savefig(plots / "tracking_errors.png", dpi=180)
    plt.close(fig)


def main() -> int:
    args = _parse_args()
    config = load_config(args.config)
    seeds = _parse_seeds(
        args.seeds,
        config.evaluation.seed_start,
        config.evaluation.seed_count,
    )
    output = args.output or Path(config.output)
    output.mkdir(parents=True, exist_ok=True)
    if args.workers <= 0:
        raise ValueError("--workers must be positive")

    payloads = [
        (str(args.config.resolve()), args.max_steps, seed, str(output.resolve()))
        for seed in seeds
    ]
    if args.workers == 1:
        episodes = [_run_worker(payload) for payload in payloads]
    else:
        episodes_by_seed: dict[int, dict[str, Any]] = {}
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(_run_worker, payload): payload[2] for payload in payloads}
            for future in as_completed(futures):
                seed = futures[future]
                episodes_by_seed[seed] = future.result()
                print(f"finished seed {seed}", flush=True)
        episodes = [episodes_by_seed[seed] for seed in seeds]

    order = {seed: index for index, seed in enumerate(seeds)}
    episodes.sort(key=lambda item: order[int(item["seed"])])
    summary_payload = {
        "config_path": str(args.config.resolve()),
        "config_sha256": config.config_sha256,
        "git_commit": _git_commit(),
        "seeds": list(seeds),
        "max_steps_override": args.max_steps,
        "aggregate": _aggregate_from_episodes(episodes),
        "episodes": episodes,
    }
    (output / "summary.json").write_text(
        json.dumps(summary_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    _write_markdown(summary_payload, output, args.max_steps)
    if not args.no_plot:
        _plot(output, args.config)
    print(json.dumps(summary_payload["aggregate"], indent=2, ensure_ascii=False))
    return 0


def _aggregate_from_episodes(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    scalar_keys = [
        "steps",
        "duration_s",
        "return_sum",
        "ppo_return_sum",
        "horizontal_rmse_m",
        "vertical_rmse_m",
        "three_d_rmse_m",
        "horizontal_mean_abs_error_m",
        "vertical_mean_abs_error_m",
        "three_d_mean_error_m",
        "yaw_rmse_rad",
        "water_surge_mean_mps",
        "water_surge_std_mps",
        "speed_error_rmse_mps",
        "saturation_rate",
        "constraint_violations",
        "command_smoothness_mean_sq",
        "pvs_step_latency_p50_ms",
        "pvs_step_latency_p95_ms",
        "control_loop_latency_p50_ms",
        "control_loop_latency_p95_ms",
        "control_loop_p95_rate_hz",
        "pvs_step_rate_hz",
    ]
    aggregate: dict[str, Any] = {
        "episode_count": len(episodes),
        "success_count": sum(bool(item["success"]) for item in episodes),
        "success_rate": float(np.mean([bool(item["success"]) for item in episodes])),
        "termination_counts": {
            str(kind): sum(item["termination"] == kind for item in episodes)
            for kind in sorted({str(item["termination"]) for item in episodes})
        },
    }
    for key in scalar_keys:
        values = np.asarray([item[key] for item in episodes], dtype=np.float64)
        aggregate[f"{key}_mean"] = float(np.mean(values))
        aggregate[f"{key}_std"] = float(np.std(values))
        aggregate[f"{key}_median"] = float(np.median(values))
        aggregate[f"{key}_p95"] = float(np.percentile(values, 95))
    arrivals = [item["arrival_time_s"] for item in episodes if item["arrival_time_s"] is not None]
    aggregate["arrival_time_s_mean"] = float(np.mean(arrivals)) if arrivals else None
    aggregate["arrival_time_s_median"] = float(np.median(arrivals)) if arrivals else None
    return aggregate


if __name__ == "__main__":
    raise SystemExit(main())
