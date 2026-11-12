from __future__ import annotations

import numpy as np
import torch
from auv_core.attitude import quaternion_from_euler
from auv_core.contracts import State, UniformCurrent
from auv_core.guidance import ThreeDAlosGuidance

from auv_lowlevel.actions import ActionPipeline, ActuatorLimits
from auv_lowlevel.batch.actions import BatchedActionPipeline
from auv_lowlevel.batch.guidance import BatchedThreeDAlosGuidance
from auv_lowlevel.batch.observation import (
    OBSERVATION_SCHEMA_V2,
    BatchedObservationBuilder,
)
from auv_lowlevel.batch.path import BatchedReferencePath
from auv_lowlevel.batch.reward import (
    BatchedRewardComputer,
    TerminalKind,
    batched_terminal_kind,
)
from auv_lowlevel.config import load_config
from auv_lowlevel.guidance import TerminalCaptureGuidance
from auv_lowlevel.observation import ObservationBuilder
from auv_lowlevel.path import generate_sinusoidal_3d_path
from auv_lowlevel.reward import RewardComputer


def _limits() -> ActuatorLimits:
    return ActuatorLimits(
        max_rudder_angle_rad=np.deg2rad(15.0),
        max_stern_plane_angle_rad=np.deg2rad(15.0),
        max_rudder_rate_rad_s=np.deg2rad(15.0) / 0.1,
        max_stern_plane_rate_rad_s=np.deg2rad(15.0) / 0.1,
        max_propeller_rpm=1525.0,
    )


def test_batched_observation_matches_scalar_v1() -> None:
    config = load_config("configs/lowlevel/odarrl_control_base.json")
    path = generate_sinusoidal_3d_path(config.reference)
    trajectory = path.trajectory()
    batched_path = BatchedReferencePath.from_reference_path(path)
    guidance = BatchedThreeDAlosGuidance(config.guidance, config.reference)
    builder = BatchedObservationBuilder(config)
    scalar_builder = ObservationBuilder(config)
    rng = np.random.default_rng(31)
    count = 80
    progress = rng.uniform(0.0, path.total_length_m, size=count)
    positions = np.stack([path.point_at_s(float(value)) for value in progress])
    positions += rng.uniform(-1.5, 1.5, size=(count, 3))
    yaw = rng.uniform(-np.pi, np.pi, size=count)
    pitch = rng.uniform(-0.5, 0.5, size=count)
    roll = rng.uniform(-0.2, 0.2, size=count)
    quaternions = np.stack(
        [
            quaternion_from_euler(float(roll[i]), float(pitch[i]), float(yaw[i]))
            for i in range(count)
        ]
    )
    nu = np.column_stack(
        (
            rng.uniform(0.8, 1.6, size=count),
            rng.uniform(-0.2, 0.2, size=(count, 2)),
            rng.uniform(-0.1, 0.1, size=(count, 3)),
        )
    )
    current = np.column_stack((rng.uniform(-0.4, 0.4, size=(count, 2)), np.zeros(count)))
    previous_actuator = np.column_stack(
        (
            rng.uniform(-0.2, 0.2, size=(count, 2)),
            rng.uniform(0.0, 1525.0, size=count),
        )
    )
    previous_residual = rng.uniform(-1.0, 1.0, size=(count, 3))
    integral = rng.uniform(-400.0, 400.0, size=count)
    positions_t = torch.from_numpy(positions)
    projection = batched_path.project(positions_t)
    solution = guidance.solve(
        path=batched_path,
        projection=projection,
        position_ned_m=positions_t,
        current_ned_mps=torch.from_numpy(current),
    )
    actual = builder.build(
        eta=torch.from_numpy(
            np.column_stack((positions, np.stack((roll, pitch, yaw), axis=1)))
        ),
        nu=torch.from_numpy(nu),
        path=batched_path,
        projection=projection,
        solution=solution,
        current_ned_mps=torch.from_numpy(current),
        previous_actuator=torch.from_numpy(previous_actuator),
        previous_residual=torch.from_numpy(previous_residual),
        speed_integral_rpm=torch.from_numpy(integral),
    )

    expected = []
    for index in range(count):
        state = State(
            t_s=0.0,
            position_ned_m=positions[index],
            quat_xyzw=quaternions[index],
            linear_velocity_body_mps=nu[index, :3],
            angular_velocity_body_rps=nu[index, 3:],
        )
        field = UniformCurrent(velocity_ned_mps=current[index])
        scalar_inner = ThreeDAlosGuidance(
            config.guidance.horizontal_lookahead_m,
            config.guidance.vertical_lookahead_m,
            config.reference.speed_mps,
            min_lookahead_m=config.guidance.min_lookahead_m,
            max_lookahead_m=config.guidance.max_lookahead_m,
            horizontal_gain=config.guidance.horizontal_gain,
            vertical_gain=config.guidance.vertical_gain,
            lookahead_error_gain=config.guidance.lookahead_error_gain,
            current_field=field if config.guidance.compensate_current else None,
            compensate_current=config.guidance.compensate_current,
            validity_period_s=config.guidance.validity_period_s,
        )
        scalar_guidance = TerminalCaptureGuidance(
            scalar_inner,
            path.goal_ned_m,
            config.guidance.terminal_capture_distance_m,
        )
        scalar_solution = scalar_guidance.solve(state, trajectory, 0.0)
        expected.append(
            scalar_builder.build(
                state=state,
                reference=scalar_solution.reference,
                solution=scalar_solution,
                path=path,
                current_ned_mps=current[index],
                previous_actuator=previous_actuator[index],
                previous_residual=previous_residual[index],
                speed_integral_rpm=float(integral[index]),
                expose_current=True,
            )
        )
    np.testing.assert_allclose(actual.numpy(), np.stack(expected), atol=1e-12, rtol=1e-12)


def test_v2_schema_is_exact_prefix_of_v1() -> None:
    config = load_config("configs/lowlevel/odarrl_control_base.json")
    path = generate_sinusoidal_3d_path(config.reference)
    batched_path = BatchedReferencePath.from_reference_path(path)
    guidance = BatchedThreeDAlosGuidance(config.guidance, config.reference)
    eta = torch.zeros((3, 6), dtype=torch.float64)
    eta[:, 0] = torch.tensor([1.0, 5.0, 9.0])
    eta[:, 2] = 10.0
    nu = torch.zeros((3, 6), dtype=torch.float64)
    nu[:, 0] = 1.2
    current = torch.tensor([[0.2, 0.1, 0.0]]).expand(3, 3)
    projection = batched_path.project(eta[:, :3])
    solution = guidance.solve(
        path=batched_path,
        projection=projection,
        position_ned_m=eta[:, :3],
        current_ned_mps=current,
    )
    common = {
        "eta": eta,
        "nu": nu,
        "path": batched_path,
        "projection": projection,
        "solution": solution,
        "current_ned_mps": current,
        "previous_actuator": torch.zeros((3, 3), dtype=torch.float64),
        "previous_residual": torch.zeros((3, 3), dtype=torch.float64),
    }
    v1 = BatchedObservationBuilder(config).build(**common, speed_integral_rpm=torch.ones(3))
    v2 = BatchedObservationBuilder(config, schema=OBSERVATION_SCHEMA_V2).build(
        **common, speed_integral_rpm=None
    )
    assert v1.shape == (3, 27)
    assert v2.shape == (3, 26)
    assert torch.equal(v2, v1[:, :26])


def test_batched_action_pipeline_matches_scalar() -> None:
    config = load_config("configs/lowlevel/odarrl_control_base.json")
    scalar = ActionPipeline(config, _limits())
    batched = BatchedActionPipeline(config, _limits())
    rng = np.random.default_rng(41)
    count = 200
    base = np.column_stack(
        (
            rng.uniform(-0.4, 0.4, size=(count, 2)),
            rng.uniform(-100.0, 1700.0, size=count),
        )
    )
    residual = rng.uniform(-1.5, 1.5, size=(count, 3))
    previous = np.column_stack(
        (
            rng.uniform(-0.26, 0.26, size=(count, 2)),
            rng.uniform(0.0, 1525.0, size=count),
        )
    )
    actual = batched.apply(
        torch.from_numpy(base),
        torch.from_numpy(residual),
        torch.from_numpy(previous),
        dt_s=0.02,
    )
    expected = [scalar.apply(base[i], residual[i], previous[i], 0.02) for i in range(count)]

    np.testing.assert_allclose(actual.delta.numpy(), [item.delta for item in expected], atol=1e-12)
    np.testing.assert_allclose(actual.total.numpy(), [item.total for item in expected], atol=1e-12)
    assert actual.magnitude_clipped.numpy().tolist() == [
        item.magnitude_clipped.tolist() for item in expected
    ]
    assert actual.rate_clipped.numpy().tolist() == [item.rate_clipped.tolist() for item in expected]
    assert actual.constraint_violation_count.numpy().tolist() == [
        item.constraint_violation_count for item in expected
    ]
    assert actual.saturated.numpy().tolist() == [item.saturated for item in expected]


def test_batched_reward_matches_scalar() -> None:
    config = load_config("configs/lowlevel/odarrl_control_base.json")
    scalar = RewardComputer(config)
    batched = BatchedRewardComputer(config)
    rng = np.random.default_rng(51)
    count = 50
    values = {
        "previous_progress_m": rng.uniform(0.0, 30.0, size=count),
        "progress_m": rng.uniform(0.0, 35.0, size=count),
        "previous_goal_distance_m": rng.uniform(0.0, 40.0, size=count),
        "goal_distance_m": rng.uniform(0.0, 40.0, size=count),
        "error_3d_m": rng.uniform(0.0, 5.0, size=count),
        "yaw_error_rad": rng.uniform(-np.pi, np.pi, size=count),
        "water_surge_mps": rng.uniform(0.5, 1.8, size=count),
        "reference_speed_mps": np.full(count, 1.2),
        "residual_normalized": rng.uniform(-1.0, 1.0, size=(count, 3)),
        "previous_residual_normalized": rng.uniform(-1.0, 1.0, size=(count, 3)),
        "saturated": rng.integers(0, 2, size=count).astype(bool),
        "terminal_kind": rng.integers(0, 5, size=count),
    }
    actual = batched.compute(**{key: torch.from_numpy(value) for key, value in values.items()})
    names = {0: None, 1: "goal", 2: "timeout", 3: "out_of_bounds", 4: "invalid"}
    expected = [
        scalar.compute(
            previous_progress_m=values["previous_progress_m"][index],
            progress_m=values["progress_m"][index],
            previous_goal_distance_m=values["previous_goal_distance_m"][index],
            goal_distance_m=values["goal_distance_m"][index],
            error_3d_m=values["error_3d_m"][index],
            yaw_error_rad=values["yaw_error_rad"][index],
            water_surge_mps=values["water_surge_mps"][index],
            reference_speed_mps=values["reference_speed_mps"][index],
            residual_normalized=values["residual_normalized"][index],
            previous_residual_normalized=values["previous_residual_normalized"][index],
            saturated=bool(values["saturated"][index]),
            terminal_kind=names[int(values["terminal_kind"][index])],
        )
        for index in range(count)
    ]
    np.testing.assert_allclose(actual.total.numpy(), [item.total for item in expected], atol=1e-12)
    np.testing.assert_allclose(
        actual.ppo_scaled.numpy(), [item.ppo_scaled for item in expected], atol=1e-12
    )


def test_terminal_priority_matches_scalar_rules() -> None:
    position = torch.tensor(
        [
            [0.0, 0.0, 10.0],
            [0.0, 0.0, 10.0],
            [500.0, 0.0, 10.0],
            [float("nan"), 0.0, 10.0],
            [5.0, 0.0, 10.0],
        ],
        dtype=torch.float64,
    )
    kind = batched_terminal_kind(
        position_ned_m=position,
        goal_ned_m=torch.tensor([0.0, 0.0, 10.0], dtype=torch.float64),
        bounds_ned_m=torch.tensor(
            [[-200.0, 200.0], [-200.0, 200.0], [0.0, 200.0]], dtype=torch.float64
        ),
        goal_radius_m=2.0,
        step_count=torch.tensor([0, 2500, 2500, 0, 30]),
        max_episode_steps=2500,
    )
    assert kind.numpy().tolist() == [
        TerminalKind.GOAL,
        TerminalKind.GOAL,
        TerminalKind.OUT_OF_BOUNDS,
        TerminalKind.INVALID,
        TerminalKind.NONE,
    ]
