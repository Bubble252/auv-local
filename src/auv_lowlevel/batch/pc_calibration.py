"""Evidence helpers for the PC20 high-speed command envelope."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import torch

from auv_lowlevel.config import LowLevelConfig

from .dynamics import BatchedRemus100Dynamics, Remus100BatchState


def worst_case_uniform_current_mps(config: LowLevelConfig) -> float:
    """Return the maximum horizontal current magnitude used by the config gate."""

    if not config.current.enabled:
        return 0.0
    bound = float(config.current.max_speed_mps)
    if config.current.time_variation is not None:
        bound += float(config.current.time_variation.speed_amplitude_mps)
    return bound


def _new_state(
    count: int,
    *,
    surge_mps: float,
    rpm: float,
    dtype: torch.dtype,
    device: torch.device,
) -> Remus100BatchState:
    eta = torch.zeros((count, 6), dtype=dtype, device=device)
    nu = torch.zeros((count, 6), dtype=dtype, device=device)
    nu[:, 0] = surge_mps
    actuator = torch.zeros((count, 3), dtype=dtype, device=device)
    actuator[:, 2] = rpm
    return Remus100BatchState(eta=eta, nu=nu, actuator=actuator)


def _steady_ground_speeds(
    dynamics: BatchedRemus100Dynamics,
    *,
    rpms: Iterable[float],
    currents_mps: Iterable[tuple[float, float]],
    settle_seconds: float,
    window_seconds: float,
) -> np.ndarray:
    displacement = _steady_ground_velocity_ned(
        dynamics,
        rpms=rpms,
        currents_mps=currents_mps,
        settle_seconds=settle_seconds,
        window_seconds=window_seconds,
    )
    return np.linalg.norm(displacement, axis=1)


def _steady_ground_velocity_ned(
    dynamics: BatchedRemus100Dynamics,
    *,
    rpms: Iterable[float],
    currents_mps: Iterable[tuple[float, float]],
    settle_seconds: float,
    window_seconds: float,
) -> np.ndarray:
    rpm_values = tuple(float(value) for value in rpms)
    current_values = tuple((float(north), float(east)) for north, east in currents_mps)
    if len(current_values) != 1 and len(current_values) != len(rpm_values):
        raise ValueError("currents must contain one row or one row per rpm")
    if settle_seconds <= 0.0 or window_seconds <= 0.0:
        raise ValueError("settle and window durations must be positive")

    count = len(rpm_values)
    if len(current_values) == 1:
        current_values = current_values * count
    dtype = dynamics.dtype
    device = dynamics.device
    state = _new_state(
        count,
        surge_mps=0.5,
        rpm=0.0,
        dtype=dtype,
        device=device,
    )
    rpm_tensor = torch.tensor(rpm_values, dtype=dtype, device=device)
    command = torch.zeros((count, 3), dtype=dtype, device=device)
    command[:, 2] = rpm_tensor
    current = torch.tensor(current_values, dtype=dtype, device=device)
    state = Remus100BatchState(
        eta=state.eta,
        nu=state.nu,
        actuator=torch.zeros_like(state.actuator),
    )

    settle_steps = int(round(settle_seconds / dynamics.sample_time_s))
    window_steps = max(1, int(round(window_seconds / dynamics.sample_time_s)))
    for _ in range(settle_steps):
        state = dynamics.step(state, command, current).state
    start_xy = state.eta[:, :2].clone()
    for _ in range(window_steps):
        state = dynamics.step(state, command, current).state
    displacement = state.eta[:, :2] - start_xy
    return displacement.div(window_seconds).detach().cpu().numpy()


def _current_reachability_sweep(
    dynamics: BatchedRemus100Dynamics,
    *,
    command_speeds_mps: Iterable[float],
    max_rpm: float,
    current_magnitudes_mps: Iterable[float],
    directions_deg: Iterable[float],
    settle_seconds: float,
    window_seconds: float,
) -> list[dict[str, Any]]:
    command_speeds = tuple(float(value) for value in command_speeds_mps)
    magnitudes = tuple(float(value) for value in current_magnitudes_mps)
    directions = tuple(float(value) for value in directions_deg)
    if any(value < 0.0 for value in magnitudes):
        raise ValueError("current sweep magnitudes must be non-negative")
    combinations = [
        (magnitude, direction, direction) for magnitude in magnitudes for direction in directions
    ]
    currents = tuple(
        (
            magnitude * np.cos(np.deg2rad(direction)),
            magnitude * np.sin(np.deg2rad(direction)),
        )
        for magnitude, direction, _ in combinations
    )
    rpms = (max_rpm,) * len(combinations)
    velocities = _steady_ground_velocity_ned(
        dynamics,
        rpms=rpms,
        currents_mps=currents,
        settle_seconds=settle_seconds,
        window_seconds=window_seconds,
    )
    rows: list[dict[str, Any]] = []
    for command_speed in command_speeds:
        for index, (magnitude, direction, _) in enumerate(combinations):
            forward_speed = float(velocities[index, 0])
            crossflow_speed = float(velocities[index, 1])
            margin = forward_speed - command_speed
            rows.append(
                {
                    "command_speed_mps": command_speed,
                    "current_magnitude_mps": magnitude,
                    "current_direction_deg": direction,
                    "max_rpm_forward_ground_speed_mps": forward_speed,
                    "max_rpm_crossflow_speed_mps": crossflow_speed,
                    "max_rpm_ground_speed_mps": float(np.hypot(forward_speed, crossflow_speed)),
                    "forward_margin_mps": margin,
                    "reachable_at_max_rpm": bool(margin >= -1e-6),
                }
            )
    return rows


def _coastdown_response(
    dynamics: BatchedRemus100Dynamics,
    *,
    speeds_mps: Iterable[float],
    rpm_values: Iterable[float],
    settle_seconds: float,
    max_seconds: float,
    speed_threshold_mps: float,
) -> list[dict[str, Any]]:
    speeds = tuple(float(value) for value in speeds_mps)
    rpms = tuple(float(value) for value in rpm_values)
    if len(speeds) != len(rpms):
        raise ValueError("speeds and rpm values must have equal length")
    if max_seconds <= 0.0 or speed_threshold_mps < 0.0:
        raise ValueError("coastdown duration must be positive and threshold non-negative")
    count = len(speeds)
    dtype = dynamics.dtype
    device = dynamics.device
    state = _new_state(count, surge_mps=0.0, rpm=0.0, dtype=dtype, device=device)
    state.nu[:, 0] = torch.tensor(speeds, dtype=dtype, device=device)
    command = torch.zeros((count, 3), dtype=dtype, device=device)
    command[:, 2] = torch.tensor(rpms, dtype=dtype, device=device)
    current = torch.zeros((count, 2), dtype=dtype, device=device)
    settle_steps = max(1, int(round(settle_seconds / dynamics.sample_time_s)))
    for _ in range(settle_steps):
        state = dynamics.step(state, command, current).state

    initial_speed = torch.linalg.vector_norm(state.nu[:, :3], dim=1)
    start_xy = state.eta[:, :2].clone()
    command.zero_()
    steps = max(1, int(round(max_seconds / dynamics.sample_time_s)))
    speed_history = torch.empty((steps, count), dtype=dtype, device=device)
    distance_history = torch.empty((steps, count), dtype=dtype, device=device)
    for step in range(steps):
        state = dynamics.step(state, command, current).state
        speed_history[step] = torch.linalg.vector_norm(state.nu[:, :3], dim=1)
        distance_history[step] = torch.linalg.vector_norm(state.eta[:, :2] - start_xy, dim=1)

    speeds_np = speed_history.detach().cpu().numpy()
    distances_np = distance_history.detach().cpu().numpy()
    rows: list[dict[str, Any]] = []
    for index, (speed, rpm) in enumerate(zip(speeds, rpms, strict=True)):
        stopped = np.flatnonzero(speeds_np[:, index] <= speed_threshold_mps)
        stopped_step = int(stopped[0]) if stopped.size else steps - 1
        rows.append(
            {
                "target_speed_mps": speed,
                "rpm": rpm,
                "initial_ground_speed_mps": float(initial_speed[index]),
                "coastdown_time_to_threshold_s": (stopped_step + 1) * dynamics.sample_time_s,
                "coastdown_distance_to_threshold_m": float(distances_np[stopped_step, index]),
                "final_ground_speed_mps": float(speeds_np[-1, index]),
                "threshold_reached": bool(stopped.size),
                "speed_threshold_mps": speed_threshold_mps,
            }
        )
    return rows


def _step_response(
    dynamics: BatchedRemus100Dynamics,
    *,
    speeds_mps: Iterable[float],
    rpm_values: Iterable[float],
    fin_command_rad: float,
    axis: str,
    duration_seconds: float,
) -> list[dict[str, Any]]:
    if axis not in {"rudder", "stern"}:
        raise ValueError("axis must be rudder or stern")
    speeds = tuple(float(value) for value in speeds_mps)
    rpms = tuple(float(value) for value in rpm_values)
    if len(speeds) != len(rpms):
        raise ValueError("speeds and rpm values must have equal length")
    directions = (1.0, -1.0)
    count = len(speeds) * len(directions)
    repeated_speeds = tuple(value for value in speeds for _ in directions)
    repeated_rpms = tuple(value for value in rpms for _ in directions)
    signs = torch.tensor(
        [direction for _ in speeds for direction in directions],
        dtype=dynamics.dtype,
        device=dynamics.device,
    )
    state = _new_state(
        count,
        surge_mps=0.0,
        rpm=0.0,
        dtype=dynamics.dtype,
        device=dynamics.device,
    )
    state.nu[:, 0] = torch.tensor(repeated_speeds, dtype=dynamics.dtype, device=dynamics.device)
    command = torch.zeros((count, 3), dtype=dynamics.dtype, device=dynamics.device)
    command[:, 2] = torch.tensor(repeated_rpms, dtype=dynamics.dtype, device=dynamics.device)
    channel = 0 if axis == "rudder" else 1
    command[:, channel] = signs * fin_command_rad
    current = torch.zeros((count, 2), dtype=dynamics.dtype, device=dynamics.device)
    steps = max(1, int(round(duration_seconds / dynamics.sample_time_s)))
    for _ in range(steps):
        state = dynamics.step(state, command, current).state

    yaw_rate = state.nu[:, 5].detach().cpu().numpy()
    pitch_rate = state.nu[:, 4].detach().cpu().numpy()
    surge = state.nu[:, 0].detach().cpu().numpy()
    depth = state.eta[:, 2].detach().cpu().numpy()
    rows: list[dict[str, Any]] = []
    index = 0
    for speed, rpm in zip(speeds, rpms, strict=True):
        signs_for_speed = []
        for _ in directions:
            signs_for_speed.append(
                {
                    "speed_mps": speed,
                    "rpm": rpm,
                    "final_surge_mps": float(surge[index]),
                    "final_yaw_rate_rad_s": float(yaw_rate[index]),
                    "final_pitch_rate_rad_s": float(pitch_rate[index]),
                    "depth_change_m": float(depth[index]),
                    "turn_radius_m": (
                        float(surge[index] / abs(yaw_rate[index]))
                        if abs(yaw_rate[index]) > 1e-9
                        else None
                    ),
                }
            )
            index += 1
        rows.append(
            {
                "target_speed_mps": speed,
                "rpm": rpm,
                "positive_fin": signs_for_speed[0],
                "negative_fin": signs_for_speed[1],
            }
        )
    return rows


def calibrate_pc20_high_speed(
    config: LowLevelConfig,
    *,
    config_path: Path | str | None = None,
    settle_seconds: float = 60.0,
    window_seconds: float = 10.0,
    step_seconds: float = 5.0,
    rpm_points: int = 9,
    current_sweep_mps: Iterable[float] | None = None,
    current_directions_deg: Iterable[float] = tuple(range(0, 360, 45)),
    coastdown_seconds: float = 60.0,
    coastdown_speed_threshold_mps: float = 0.2,
    device: str = "cpu",
    dtype: torch.dtype = torch.float64,
) -> dict[str, Any]:
    """Measure steady speed, current margins and 20-degree fin response."""

    limits = config.actuator
    max_rpm = float(limits.policy_max_propeller_rpm)
    max_fin = max(
        float(limits.policy_max_rudder_angle_rad),
        float(limits.policy_max_stern_plane_angle_rad),
    )
    dynamics = BatchedRemus100Dynamics(
        sample_time_s=float(config.plant.dt_s),
        max_rudder_angle_rad=float(limits.policy_max_rudder_angle_rad),
        max_stern_plane_angle_rad=float(limits.policy_max_stern_plane_angle_rad),
        max_propeller_rpm=max_rpm,
        device=device,
        dtype=dtype,
    )
    rpms = np.linspace(max_rpm / max(1, rpm_points - 1), max_rpm, rpm_points)
    static_speeds = _steady_ground_speeds(
        dynamics,
        rpms=rpms,
        currents_mps=((0.0, 0.0),),
        settle_seconds=settle_seconds,
        window_seconds=window_seconds,
    )
    current_bound = worst_case_uniform_current_mps(config)
    head_speed, tail_speed = _steady_ground_speeds(
        dynamics,
        rpms=(max_rpm, max_rpm),
        currents_mps=((-current_bound, 0.0), (current_bound, 0.0)),
        settle_seconds=settle_seconds,
        window_seconds=window_seconds,
    )
    command_range = config.reference.command_speed_range_mps
    command_high = (
        float(command_range[1]) if command_range is not None else float(config.reference.speed_mps)
    )
    static_max = float(static_speeds[-1])
    reachability_margin = float(head_speed) - command_high
    target_speeds = tuple(
        sorted(
            {
                float(command_range[0]) if command_range is not None else command_high,
                command_high,
            }
        )
    )
    target_rpms = np.interp(target_speeds, static_speeds, rpms).tolist()
    sweep_magnitudes = (
        (0.0, current_bound)
        if current_sweep_mps is None
        else tuple(float(value) for value in current_sweep_mps)
    )
    current_sweep = _current_reachability_sweep(
        dynamics,
        command_speeds_mps=target_speeds,
        max_rpm=max_rpm,
        current_magnitudes_mps=sweep_magnitudes,
        directions_deg=current_directions_deg,
        settle_seconds=settle_seconds,
        window_seconds=window_seconds,
    )
    coastdown = _coastdown_response(
        dynamics,
        speeds_mps=target_speeds,
        rpm_values=target_rpms,
        settle_seconds=settle_seconds,
        max_seconds=coastdown_seconds,
        speed_threshold_mps=coastdown_speed_threshold_mps,
    )
    rudder_response = _step_response(
        dynamics,
        speeds_mps=target_speeds,
        rpm_values=target_rpms,
        fin_command_rad=max_fin,
        axis="rudder",
        duration_seconds=step_seconds,
    )
    stern_response = _step_response(
        dynamics,
        speeds_mps=target_speeds,
        rpm_values=target_rpms,
        fin_command_rad=max_fin,
        axis="stern",
        duration_seconds=step_seconds,
    )
    passed = reachability_margin >= -1e-6 and command_high <= static_max + 1e-6
    return {
        "schema_version": "auv_pc20_high_speed_calibration.v2",
        "config_path": None if config_path is None else str(Path(config_path).resolve()),
        "experiment_name": config.name,
        "command_speed_range_mps": (None if command_range is None else list(command_range)),
        "current_envelope": {
            "worst_case_uniform_current_mps": current_bound,
            "head_current_max_ground_speed_mps": float(head_speed),
            "tail_current_max_ground_speed_mps": float(tail_speed),
        },
        "actuator_envelope": {
            "max_rudder_angle_rad": float(limits.policy_max_rudder_angle_rad),
            "max_stern_plane_angle_rad": float(limits.policy_max_stern_plane_angle_rad),
            "max_propeller_rpm": max_rpm,
        },
        "rpm_steady_speed_curve": [
            {"rpm": float(rpm), "ground_speed_mps": float(speed)}
            for rpm, speed in zip(rpms, static_speeds, strict=True)
        ],
        "reachability": {
            "command_high_mps": command_high,
            "static_max_ground_speed_mps": static_max,
            "margin_mps": reachability_margin,
            "passed": bool(passed),
        },
        "rudder_step_response": rudder_response,
        "stern_step_response": stern_response,
        "current_reachability_sweep": current_sweep,
        "coastdown_response": coastdown,
        "current_sweep_definition": {
            "direction_deg": "world north; 0=tail current, 180=head current",
            "rpm": max_rpm,
            "margin": "forward ground speed minus command speed",
        },
        "simulation": {
            "sample_time_s": float(config.plant.dt_s),
            "settle_seconds": float(settle_seconds),
            "window_seconds": float(window_seconds),
            "step_seconds": float(step_seconds),
            "coastdown_seconds": float(coastdown_seconds),
            "coastdown_speed_threshold_mps": float(coastdown_speed_threshold_mps),
        },
    }


__all__ = [
    "calibrate_pc20_high_speed",
    "worst_case_uniform_current_mps",
]
