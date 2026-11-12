"""Render one P1 analytic rollout as a top-down 2D GIF.

The script uses the same 17-D observation contract and analytic 13-sector
perception as training.  It is intended for qualitative inspection only; the
quantitative gates still come from ``training_summary.json``.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import replace
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from auv_core.contracts import Obstacle  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402
from matplotlib.artist import Artist  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402
from matplotlib.patches import Circle, Polygon, Rectangle  # noqa: E402

from auv_isaaclab.tasks.p1_avoidance.scenarios import build_scenario_groups  # noqa: E402
from auv_rl.bc.expert_scenarios import (  # noqa: E402
    EXPERT_SCENARIO_FAMILIES,
    ExpertScenarioPoolConfig,
    build_expert_scenario_pool,
)
from auv_rl.ddpg.agent import DDPGAgent  # noqa: E402
from auv_rl.envs.analytic import build_analytic_perception  # noqa: E402
from auv_rl.envs.planar_avoidance import PlanarAvoidanceEnv  # noqa: E402
from auv_rl.ppo_policy import PPOActorPolicy  # noqa: E402
from auv_rl.scenarios import PlanarScenarioCase, default_planar_scenarios  # noqa: E402
from auv_rl.training import env_config_from_payload, load_training_payload  # noqa: E402

_BACKGROUND = "#f4f1e8"
_OBSTACLE = "#c44e52"
_OBSTACLE_EDGE = "#7f1d1d"
_TRAIL = "#1f77b4"
_VEHICLE = "#0b3d91"
_GOAL = "#2ca02c"
_TEXT = "#1f2937"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a P1 rollout as a 2D GIF")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--policy-kind", choices=("ddpg", "ppo"), default="ddpg")
    parser.add_argument("--history-length", type=int, default=16)
    parser.add_argument(
        "--scenario-key",
        required=True,
        help="s1/s2, a C1-C3 variant such as c3_gate_00, or an expert pool key",
    )
    parser.add_argument("--mode", choices=("eval", "explore"), default="eval")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--dpi", type=int, default=110)
    parser.add_argument("--title", default=None)
    parser.add_argument("--static-png", action="store_true")
    parser.add_argument("--max-steps", type=int, default=None)
    return parser.parse_args()


def _resolve_cases(scenario_key: str) -> tuple[tuple[PlanarScenarioCase, ...], str]:
    curriculum_cases = tuple(case for group in build_scenario_groups().values() for case in group)
    if scenario_key in {case.key for case in curriculum_cases}:
        return curriculum_cases, scenario_key
    if scenario_key.startswith("expert_"):
        pool = build_expert_scenario_pool(
            ExpertScenarioPoolConfig(
                seed=0,
                instances_per_family=5,
                families=EXPERT_SCENARIO_FAMILIES,
            )
        )
        keys = {case.key for case in pool.cases}
        if scenario_key not in keys:
            raise ValueError(f"unknown expert scenario {scenario_key!r}; available: {sorted(keys)}")
        return pool.cases, scenario_key
    cases = default_planar_scenarios()
    keys = {case.key for case in cases}
    if scenario_key not in keys:
        raise ValueError(f"unknown scenario {scenario_key!r}; available: {sorted(keys)}")
    return cases, scenario_key


def _case_by_key(cases: tuple[PlanarScenarioCase, ...], key: str) -> PlanarScenarioCase:
    for case in cases:
        if case.key == key:
            return case
    raise ValueError(f"unknown scenario key: {key}")


def _rollout(
    env: PlanarAvoidanceEnv,
    policy: Any,
    *,
    policy_kind: str,
    scenario_key: str,
    evaluate: bool,
    seed: int,
    max_steps: int | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    observation, info = env.reset(
        seed=seed,
        options={"scenario_key": scenario_key, "jitter": False},
    )
    if policy_kind == "ddpg":
        policy.begin_episode_for_env(0, 0)
    else:
        policy.begin_episode(0)
    torch.manual_seed(seed)
    np.random.seed(seed)
    state = env._state
    assert state is not None
    frames: list[dict[str, Any]] = [
        {
            "x": float(state.position_ned_m[0]),
            "y": float(state.position_ned_m[1]),
            "yaw": float(math.atan2(state.quat_xyzw[2], state.quat_xyzw[3]) * 2.0),
            "t": float(state.t_s),
            "surge": float(state.linear_velocity_body_mps[0]),
            "action_u_dot": 0.0,
            "action_yaw_rate": 0.0,
            "min_clearance": float(info["min_clearance_m"]),
            "goal_distance": float(info["goal_distance_m"]),
        }
    ]
    steps = int(env.config.max_episode_steps if max_steps is None else max_steps)
    for _ in range(steps):
        if policy_kind == "ddpg":
            action = np.asarray(policy.act_for_env(0, observation, evaluate=evaluate))
        else:
            action = np.asarray(policy.act(observation, evaluate=True))
        action = action.astype(np.float64, copy=False)
        observation, _, terminated, truncated, info = env.step(action)
        state = env._state
        assert state is not None
        frames.append(
            {
                "x": float(state.position_ned_m[0]),
                "y": float(state.position_ned_m[1]),
                "yaw": float(math.atan2(state.quat_xyzw[2], state.quat_xyzw[3]) * 2.0),
                "t": float(state.t_s),
                "surge": float(state.linear_velocity_body_mps[0]),
                "action_u_dot": float(action[0]),
                "action_yaw_rate": float(action[1]),
                "min_clearance": float(info["min_clearance_m"]),
                "goal_distance": float(info["goal_distance_m"]),
            }
        )
        if bool(terminated) or bool(truncated):
            break
    return frames, dict(info)


def _draw_obstacle(ax: Axes, obstacle: Obstacle) -> None:
    center = (float(obstacle.center_ned_m[0]), float(obstacle.center_ned_m[1]))
    if obstacle.kind in {"cylinder", "sphere"}:
        assert obstacle.radius_m is not None
        ax.add_patch(
            Circle(
                center,
                obstacle.radius_m,
                facecolor=_OBSTACLE,
                edgecolor=_OBSTACLE_EDGE,
                alpha=0.55,
                linewidth=1.2,
                zorder=2,
            )
        )
    elif obstacle.kind == "box":
        assert obstacle.half_extents_m is not None
        half = (float(obstacle.half_extents_m[0]), float(obstacle.half_extents_m[1]))
        ax.add_patch(
            Rectangle(
                (center[0] - half[0], center[1] - half[1]),
                2.0 * half[0],
                2.0 * half[1],
                facecolor=_OBSTACLE,
                edgecolor=_OBSTACLE_EDGE,
                alpha=0.55,
                linewidth=1.2,
                zorder=2,
            )
        )


def _vehicle_polygon(x: float, y: float, yaw: float) -> np.ndarray:
    length = 1.6
    width = 0.55
    local = np.array(
        [
            [0.55 * length, 0.0],
            [-0.45 * length, 0.5 * width],
            [-0.45 * length, -0.5 * width],
        ],
        dtype=np.float64,
    )
    rotation = np.array(
        [[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]],
        dtype=np.float64,
    )
    return local @ rotation.T + np.array([x, y], dtype=np.float64)


def _setup_axes(ax: Axes, case: PlanarScenarioCase) -> None:
    bounds = case.scenario.world.obstacle_map.bounds_ned_m
    x_min, x_max = float(bounds[0, 0]), float(bounds[0, 1])
    y_min, y_max = float(bounds[1, 0]), float(bounds[1, 1])
    ax.set_xlim(x_min - 0.5, x_max + 0.5)
    ax.set_ylim(y_min - 0.5, y_max + 0.5)
    ax.set_aspect("equal", adjustable="box")
    ax.set_facecolor(_BACKGROUND)
    ax.grid(color="#d6d3c8", linewidth=0.6, alpha=0.7)
    ax.set_xlabel("NED x [m]")
    ax.set_ylabel("NED y [m]")
    ax.add_patch(
        Rectangle(
            (x_min, y_min),
            x_max - x_min,
            y_max - y_min,
            fill=False,
            edgecolor="#4b5563",
            linewidth=1.0,
            linestyle="--",
            zorder=1,
        )
    )
    for obstacle in case.scenario.world.obstacle_map.obstacles:
        _draw_obstacle(ax, obstacle)
    scenario = case.scenario
    start = scenario.start.position_ned_m
    ax.plot(
        start[0],
        start[1],
        marker="o",
        markersize=6,
        color="#4b5563",
        markeredgecolor="white",
        zorder=4,
    )
    ax.plot(
        scenario.goal_ned_m[0],
        scenario.goal_ned_m[1],
        marker="*",
        markersize=18,
        color=_GOAL,
        markeredgecolor="#14532d",
        zorder=4,
    )
    ax.add_patch(
        Circle(
            (float(scenario.goal_ned_m[0]), float(scenario.goal_ned_m[1])),
            scenario.goal_radius_m,
            fill=False,
            edgecolor=_GOAL,
            linestyle=":",
            linewidth=1.2,
            zorder=3,
        )
    )


def _draw_frame(
    ax: Axes,
    case: PlanarScenarioCase,
    frames: list[dict[str, Any]],
    index: int,
    *,
    title: str,
) -> None:
    ax.clear()
    _setup_axes(ax, case)
    visible = frames[: index + 1]
    xs = [frame["x"] for frame in visible]
    ys = [frame["y"] for frame in visible]
    ax.plot(xs, ys, color=_TRAIL, linewidth=2.2, alpha=0.9, zorder=3)
    current = frames[index]
    polygon = _vehicle_polygon(current["x"], current["y"], current["yaw"])
    ax.add_patch(
        Polygon(
            polygon,
            closed=True,
            facecolor=_VEHICLE,
            edgecolor="white",
            linewidth=1.0,
            zorder=5,
        )
    )
    ax.plot(current["x"], current["y"], marker=".", color="#f59e0b", markersize=5, zorder=6)
    status = (
        f"t={current['t']:5.2f}s   v={current['surge']:.2f} m/s   "
        f"min clear={current['min_clearance']:.2f} m\n"
        f"goal dist={current['goal_distance']:5.2f} m   "
        f"u_dot={current['action_u_dot']:+.2f} m/s^2   "
        f"yaw rate={math.degrees(current['action_yaw_rate']):+.1f} deg/s"
    )
    ax.text(
        0.02,
        0.98,
        status,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=8.5,
        color=_TEXT,
        bbox={"facecolor": "white", "edgecolor": "#d1d5db", "alpha": 0.88, "pad": 4},
        zorder=10,
    )
    ax.set_title(title, fontsize=11, loc="left", color=_TEXT, pad=8)


def _save_gif(
    case: PlanarScenarioCase,
    frames: list[dict[str, Any]],
    *,
    output: Path,
    title: str,
    stride: int,
    fps: int,
    dpi: int,
) -> None:
    indices = list(range(0, len(frames), max(stride, 1)))
    if indices[-1] != len(frames) - 1:
        indices.append(len(frames) - 1)
    figure, ax = plt.subplots(figsize=(8.4, 4.7), dpi=dpi)
    figure.patch.set_facecolor("white")

    def update(frame_index: int) -> tuple[Artist, ...]:
        _draw_frame(ax, case, frames, frame_index, title=title)
        return (ax,)

    animation = FuncAnimation(
        figure,
        update,
        frames=indices,
        interval=1000.0 / max(fps, 1),
        repeat=False,
        blit=False,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    animation.save(output, writer=PillowWriter(fps=fps))
    plt.close(figure)


def _save_static(
    case: PlanarScenarioCase,
    frames: list[dict[str, Any]],
    *,
    output: Path,
    title: str,
    dpi: int,
) -> None:
    figure, ax = plt.subplots(figsize=(8.4, 4.7), dpi=dpi)
    figure.patch.set_facecolor("white")
    _draw_frame(ax, case, frames, len(frames) - 1, title=title)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, bbox_inches="tight")
    plt.close(figure)


def _save_data(
    output: Path,
    frames: list[dict[str, Any]],
    info: dict[str, Any],
    *,
    scenario_key: str,
    scenario_name: str,
    mode: str,
    seed: int,
    config: Path,
    checkpoint: Path,
    policy_kind: str,
    observation_source: str,
) -> None:
    stem = output.with_suffix("")
    actions = np.array(
        [[frame["action_u_dot"], frame["action_yaw_rate"]] for frame in frames],
        dtype=np.float64,
    )
    np.savez_compressed(
        stem.with_suffix(".npz"),
        x=np.array([frame["x"] for frame in frames]),
        y=np.array([frame["y"] for frame in frames]),
        yaw=np.array([frame["yaw"] for frame in frames]),
        t=np.array([frame["t"] for frame in frames]),
        actions=actions,
        min_clearance=np.array([frame["min_clearance"] for frame in frames]),
        goal_distance=np.array([frame["goal_distance"] for frame in frames]),
    )
    metadata = {
        "scenario_key": scenario_key,
        "scenario_name": scenario_name,
        "mode": mode,
        "seed": seed,
        "success": bool(info.get("success", False)),
        "failure_reason": info.get("failure_reason"),
        "steps": int(info.get("rl_step", len(frames) - 1)),
        "simulation_time_s": float(info.get("simulation_time_s", frames[-1]["t"])),
        "min_clearance_m": float(info.get("min_clearance_m", float("nan"))),
        "path_length_m": float(info.get("path_length_m", float("nan"))),
        "final_goal_distance_m": float(frames[-1]["goal_distance"]),
        "action_u_dot_saturation_ratio": float(
            info.get("action_u_dot_saturation_ratio", float("nan"))
        ),
        "action_yaw_rate_saturation_ratio": float(
            info.get("action_yaw_rate_saturation_ratio", float("nan"))
        ),
        "config": str(config),
        "checkpoint": str(checkpoint),
        "policy_kind": policy_kind,
        "observation_source": observation_source,
    }
    stem.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def main() -> int:
    args = _parse_args()
    if args.stride <= 0:
        raise ValueError("stride must be positive")
    if args.fps <= 0:
        raise ValueError("fps must be positive")
    payload = load_training_payload(args.config)
    env_config, _ = env_config_from_payload(payload)
    cases, scenario_key = _resolve_cases(args.scenario_key)
    env_config = replace(
        env_config,
        active_scenario_keys=tuple(case.key for case in cases),
    )
    case = _case_by_key(cases, scenario_key)
    env = PlanarAvoidanceEnv(
        env_config,
        scenarios=cases,
        perception=build_analytic_perception(env_config),
    )
    if args.policy_kind == "ppo":
        if args.mode != "eval":
            raise ValueError("PPO rendering is deterministic; --mode explore is unsupported")
        policy: Any = PPOActorPolicy.load(
            args.checkpoint,
            frame_dim=env.builder.spec.shape[0],
            history_length=args.history_length,
            action_scale=(env.limits.max_accel_mps2, env.limits.max_yaw_rate_rps),
            device="cpu",
        )
    else:
        policy = DDPGAgent.load(args.checkpoint, device="cpu")
    frames, info = _rollout(
        env,
        policy,
        policy_kind=args.policy_kind,
        scenario_key=scenario_key,
        evaluate=args.mode == "eval",
        seed=args.seed,
        max_steps=args.max_steps,
    )
    checkpoint_label = args.checkpoint.stem.removesuffix("_checkpoint_latest").upper()
    title = args.title or (
        f"{case.name} | {checkpoint_label} {args.policy_kind.upper()} {args.mode} | "
        f"analytic sensor | seed {args.seed}"
        + ("" if bool(info.get("success", False)) else " | FAILED")
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.suffix.lower() == ".gif":
        _save_gif(
            case,
            frames,
            output=args.output,
            title=title,
            stride=args.stride,
            fps=args.fps,
            dpi=args.dpi,
        )
    else:
        _save_static(case, frames, output=args.output, title=title, dpi=args.dpi)
    if args.static_png and args.output.suffix.lower() == ".gif":
        _save_static(
            case,
            frames,
            output=args.output.with_name(f"{args.output.stem}_trajectory.png"),
            title=title,
            dpi=args.dpi,
        )
    _save_data(
        args.output,
        frames,
        info,
        scenario_key=scenario_key,
        scenario_name=case.name,
        mode=args.mode,
        seed=args.seed,
        config=args.config,
        checkpoint=args.checkpoint,
        policy_kind=args.policy_kind,
        observation_source="analytic_13sector_120m",
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "scenario": scenario_key,
                "mode": args.mode,
                "success": bool(info.get("success", False)),
                "failure_reason": info.get("failure_reason"),
                "steps": int(info.get("rl_step", len(frames) - 1)),
                "min_clearance_m": float(info.get("min_clearance_m", float("nan"))),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
