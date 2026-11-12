from __future__ import annotations

import numpy as np
import torch
from auv_core.attitude import quaternion_from_euler
from auv_core.contracts import State, UniformCurrent
from auv_core.guidance import ThreeDAlosGuidance

from auv_lowlevel.batch.guidance import BatchedThreeDAlosGuidance
from auv_lowlevel.batch.path import BatchedReferencePath
from auv_lowlevel.config import load_config
from auv_lowlevel.guidance import TerminalCaptureGuidance
from auv_lowlevel.path import generate_sinusoidal_3d_path


def _states_and_current(
    count: int, *, near_goal: bool, seed: int
) -> tuple[np.ndarray, np.ndarray, list[State], UniformCurrent]:
    config = load_config("configs/lowlevel/odarrl_control_base.json")
    path = generate_sinusoidal_3d_path(config.reference)
    rng = np.random.default_rng(seed)
    if near_goal:
        positions = path.goal_ned_m + rng.uniform(-8.0, 8.0, size=(count, 3))
    else:
        progress = rng.uniform(0.0, path.total_length_m, size=count)
        positions = np.stack([path.point_at_s(float(value)) for value in progress])
        positions += rng.uniform(-2.0, 2.0, size=(count, 3))
    yaw = rng.uniform(-np.pi, np.pi, size=count)
    pitch = rng.uniform(-0.4, 0.4, size=count)
    quaternions = np.stack(
        [quaternion_from_euler(0.0, float(pitch[i]), float(yaw[i])) for i in range(count)]
    )
    current_velocity = np.array([0.25, -0.16, 0.0])
    current = UniformCurrent(velocity_ned_mps=current_velocity)
    states = [
        State(
            t_s=0.0,
            position_ned_m=positions[index],
            quat_xyzw=quaternions[index],
            linear_velocity_body_mps=np.array([1.2, 0.0, 0.0]),
            angular_velocity_body_rps=np.zeros(3),
        )
        for index in range(count)
    ]
    return positions, quaternions, states, current


def _scalar_solution(config, state: State, current: UniformCurrent):
    path = generate_sinusoidal_3d_path(config.reference)
    inner = ThreeDAlosGuidance(
        config.guidance.horizontal_lookahead_m,
        config.guidance.vertical_lookahead_m,
        config.reference.speed_mps,
        min_lookahead_m=config.guidance.min_lookahead_m,
        max_lookahead_m=config.guidance.max_lookahead_m,
        horizontal_gain=config.guidance.horizontal_gain,
        vertical_gain=config.guidance.vertical_gain,
        lookahead_error_gain=config.guidance.lookahead_error_gain,
        current_field=current if config.guidance.compensate_current else None,
        compensate_current=config.guidance.compensate_current,
        validity_period_s=config.guidance.validity_period_s,
    )
    guidance = TerminalCaptureGuidance(
        inner,
        path.goal_ned_m,
        config.guidance.terminal_capture_distance_m,
    )
    return path, guidance.solve(state, path.trajectory(), state.t_s)


def test_alos_matches_scalar_in_free_space() -> None:
    config = load_config("configs/lowlevel/odarrl_control_base.json")
    path = generate_sinusoidal_3d_path(config.reference)
    batched_path = BatchedReferencePath.from_reference_path(path)
    positions, _, states, current = _states_and_current(80, near_goal=False, seed=7)
    positions_t = torch.from_numpy(positions)
    projection = batched_path.project(positions_t)
    guidance = BatchedThreeDAlosGuidance(config.guidance, config.reference)
    actual = guidance.solve(
        path=batched_path,
        projection=projection,
        position_ned_m=positions_t,
        current_ned_mps=torch.from_numpy(current.velocity_ned_mps).expand(80, 3),
    )
    expected = [_scalar_solution(config, state, current)[1] for state in states]

    np.testing.assert_allclose(
        actual.horizontal_error_m.numpy(), [item.horizontal_error_m for item in expected]
    )
    np.testing.assert_allclose(
        actual.vertical_error_m.numpy(), [item.vertical_error_m for item in expected]
    )
    np.testing.assert_allclose(
        actual.desired_heading_rad.numpy(), [item.desired_heading_rad for item in expected]
    )
    np.testing.assert_allclose(
        actual.desired_pitch_rad.numpy(), [item.desired_pitch_rad for item in expected]
    )
    np.testing.assert_allclose(
        actual.command_speed_mps.numpy(), [item.command_speed_mps for item in expected]
    )
    np.testing.assert_allclose(
        actual.desired_body_direction_ned.numpy(),
        np.stack([item.desired_body_direction_ned for item in expected]),
        atol=1e-12,
    )
    np.testing.assert_allclose(
        actual.reference_position_ned_m.numpy(),
        np.stack([item.reference.position_ned_m for item in expected]),
        atol=1e-12,
    )
    np.testing.assert_allclose(
        actual.reference_quat_xyzw.numpy(),
        np.stack([item.reference.quat_xyzw for item in expected]),
        atol=1e-12,
    )


def test_terminal_capture_matches_scalar_near_goal() -> None:
    config = load_config("configs/lowlevel/odarrl_control_base.json")
    path = generate_sinusoidal_3d_path(config.reference)
    batched_path = BatchedReferencePath.from_reference_path(path)
    positions, _, states, current = _states_and_current(40, near_goal=True, seed=9)
    positions_t = torch.from_numpy(positions)
    projection = batched_path.project(positions_t)
    guidance = BatchedThreeDAlosGuidance(config.guidance, config.reference)
    actual = guidance.solve(
        path=batched_path,
        projection=projection,
        position_ned_m=positions_t,
        current_ned_mps=torch.from_numpy(current.velocity_ned_mps).expand(40, 3),
    )
    expected = [_scalar_solution(config, state, current)[1] for state in states]

    np.testing.assert_allclose(
        actual.command_speed_mps.numpy(), [item.command_speed_mps for item in expected]
    )
    np.testing.assert_allclose(
        actual.reference_position_ned_m.numpy(),
        np.stack([item.reference.position_ned_m for item in expected]),
        atol=1e-12,
    )
    np.testing.assert_allclose(
        actual.desired_body_direction_ned.numpy(),
        np.stack([item.desired_body_direction_ned for item in expected]),
        atol=1e-12,
    )
