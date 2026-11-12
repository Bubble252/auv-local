from __future__ import annotations

from dataclasses import replace

import numpy as np
import torch
from auv_core.contracts import State, UniformCurrent
from auv_core.guidance import ThreeDAlosGuidance

from auv_lowlevel.batch.guidance import BatchedThreeDAlosGuidance
from auv_lowlevel.batch.observation import BatchedObservationBuilder
from auv_lowlevel.batch.path import BatchedReferencePath
from auv_lowlevel.config import LowLevelConfig, load_config
from auv_lowlevel.guidance import TerminalCaptureGuidance
from auv_lowlevel.observation import ObservationBuilder, world_to_body
from auv_lowlevel.path import generate_sinusoidal_3d_path


def _hidden_config() -> LowLevelConfig:
    config = load_config("configs/lowlevel/odarrl_control_v2_26d.json")
    return replace(
        config,
        observation=replace(
            config.observation,
            expose_current=False,
            speed_feedback="ground",
            current_contract="hidden",
        ),
        guidance=replace(config.guidance, compensate_current=False),
        controller=replace(config.controller, use_current_field=False),
    )


def _oracle_config() -> LowLevelConfig:
    config = load_config("configs/lowlevel/odarrl_control_v2_26d.json")
    return replace(
        config,
        observation=replace(
            config.observation,
            expose_current=True,
            speed_feedback="water_relative",
            current_contract="oracle",
        ),
    )


def _build(
    config: LowLevelConfig,
    current: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    path = generate_sinusoidal_3d_path(config.reference)
    trajectory = path.trajectory()
    batched_path = BatchedReferencePath.from_reference_path(path)
    batched_guidance = BatchedThreeDAlosGuidance(config.guidance, config.reference)
    scalar_builder = ObservationBuilder(config)
    batched_builder = BatchedObservationBuilder(config)

    eta = np.zeros((3, 6), dtype=np.float64)
    eta[:, 0] = [5.0, 12.0, 20.0]
    eta[:, 2] = [3.0, 4.0, 2.0]
    eta[:, 5] = [0.05, -0.08, 0.12]
    nu = np.zeros((3, 6), dtype=np.float64)
    nu[:, 0] = [1.1, 1.2, 1.3]
    nu[:, 1] = [0.02, -0.03, 0.01]
    nu[:, 5] = [0.01, -0.02, 0.03]

    eta_t = torch.from_numpy(eta)
    nu_t = torch.from_numpy(nu)
    current_t = torch.from_numpy(current)
    projection = batched_path.project(eta_t[:, :3])
    solution = batched_guidance.solve(
        path=batched_path,
        projection=projection,
        position_ned_m=eta_t[:, :3],
        current_ned_mps=current_t,
    )
    previous_actuator = np.column_stack(
        (
            np.array([0.02, -0.01, 0.03]),
            np.array([-0.01, 0.02, -0.02]),
            np.array([900.0, 1000.0, 1100.0]),
        )
    )
    previous_residual = np.array(
        [[0.1, 0.2, 0.3], [-0.1, -0.2, -0.3], [0.0, 0.05, -0.05]],
        dtype=np.float64,
    )
    batched = batched_builder.build(
        eta=eta_t,
        nu=nu_t,
        path=batched_path,
        projection=projection,
        solution=solution,
        current_ned_mps=current_t,
        previous_actuator=torch.from_numpy(previous_actuator),
        previous_residual=torch.from_numpy(previous_residual),
        speed_integral_rpm=None,
    ).numpy()

    scalar = []
    for index in range(eta.shape[0]):
        state = State(
            t_s=0.0,
            position_ned_m=eta[index, :3],
            quat_xyzw=np.array([0.0, 0.0, np.sin(eta[index, 5] / 2.0), np.cos(eta[index, 5] / 2.0)]),
            linear_velocity_body_mps=nu[index, :3],
            angular_velocity_body_rps=nu[index, 3:],
        )
        current_field = UniformCurrent(velocity_ned_mps=current[index])
        inner = ThreeDAlosGuidance(
            config.guidance.horizontal_lookahead_m,
            config.guidance.vertical_lookahead_m,
            config.reference.speed_mps,
            min_lookahead_m=config.guidance.min_lookahead_m,
            max_lookahead_m=config.guidance.max_lookahead_m,
            horizontal_gain=config.guidance.horizontal_gain,
            vertical_gain=config.guidance.vertical_gain,
            lookahead_error_gain=config.guidance.lookahead_error_gain,
            current_field=(
                current_field if config.guidance.compensate_current else None
            ),
            compensate_current=config.guidance.compensate_current,
            validity_period_s=config.guidance.validity_period_s,
        )
        guidance = TerminalCaptureGuidance(
            inner,
            path.goal_ned_m,
            config.guidance.terminal_capture_distance_m,
        )
        scalar_solution = guidance.solve(state, trajectory, 0.0)
        scalar.append(
            scalar_builder.build(
                state=state,
                reference=scalar_solution.reference,
                solution=scalar_solution,
                path=path,
                current_ned_mps=current[index],
                previous_actuator=previous_actuator[index],
                previous_residual=previous_residual[index],
                speed_integral_rpm=0.0,
            )
        )
    return batched, np.stack(scalar)


def test_hidden_current_observation_is_invariant_to_true_current() -> None:
    config = _hidden_config()
    first_current = np.array(
        [[0.35, 0.10, 0.0], [0.10, -0.20, 0.05], [-0.15, 0.08, 0.0]]
    )
    second_current = np.array(
        [[-0.25, -0.05, 0.0], [0.30, 0.15, -0.04], [0.05, -0.12, 0.0]]
    )

    first_batch, first_scalar = _build(config, first_current)
    second_batch, second_scalar = _build(config, second_current)

    np.testing.assert_allclose(first_batch, second_batch, atol=0.0, rtol=0.0)
    np.testing.assert_allclose(first_scalar, second_scalar, atol=0.0, rtol=0.0)
    np.testing.assert_allclose(first_batch, first_scalar, atol=1e-12, rtol=1e-12)
    assert np.all(first_batch[:, 17:20] == 0.0)


def test_hidden_current_speed_feedback_uses_ground_speed_only() -> None:
    config = _hidden_config()
    current = np.array(
        [[0.35, 0.10, 0.0], [0.10, -0.20, 0.05], [-0.15, 0.08, 0.0]]
    )
    linear_x = np.array([1.1, 1.2, 1.3])
    expected_speed_error = config.reference.speed_mps - linear_x

    batch, scalar = _build(config, current)

    np.testing.assert_allclose(batch[:, 11], linear_x / 2.5, atol=1e-12)
    np.testing.assert_allclose(batch[:, 12], expected_speed_error, atol=1e-12)
    np.testing.assert_allclose(scalar[:, 11], linear_x / 2.5, atol=1e-12)
    np.testing.assert_allclose(scalar[:, 12], expected_speed_error, atol=1e-12)


def test_oracle_current_still_exposes_current_and_water_relative_feedback() -> None:
    config = _oracle_config()
    current = np.array(
        [[0.35, 0.10, 0.0], [0.10, -0.20, 0.05], [-0.15, 0.08, 0.0]]
    )
    batch, scalar = _build(config, current)

    assert np.any(np.abs(batch[:, 17:20]) > 1e-12)
    np.testing.assert_allclose(batch, scalar, atol=1e-12, rtol=1e-12)
    yaw = np.array([0.05, -0.08, 0.12])
    current_body_x = np.array(
        [
            world_to_body(
                current[index],
                np.array([0.0, 0.0, np.sin(yaw[index] / 2.0), np.cos(yaw[index] / 2.0)]),
            )[0]
            for index in range(current.shape[0])
        ]
    )
    np.testing.assert_allclose(
        batch[:, 11],
        (np.array([1.1, 1.2, 1.3]) - current_body_x) / 2.5,
        atol=1e-12,
    )
