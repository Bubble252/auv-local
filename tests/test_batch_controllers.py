from __future__ import annotations

import numpy as np
import torch
from auv_core.attitude import quaternion_from_euler
from auv_core.contracts import Reference, State, UniformCurrent
from auv_core.control.remus100 import Remus100TrackingController
from auv_core.scenarios import remus100_vehicle

from auv_lowlevel.batch.controllers import (
    BatchedBCPolicy,
    BatchedControlContext,
    BatchedRemus100Teacher,
    BatchedShadowSpeedIntegrator,
)
from auv_lowlevel.batch.dynamics import (
    BatchedRemus100Dynamics,
    Remus100BatchState,
    _device_matches,
)
from auv_lowlevel.bc import BCPolicy
from auv_lowlevel.config import load_config


def _teacher_context(
    count: int,
    eta: np.ndarray,
    nu: np.ndarray,
    reference_quaternion: np.ndarray,
    speed: np.ndarray,
    current: np.ndarray,
    observation: np.ndarray,
) -> BatchedControlContext:
    quaternion = np.stack(
        [
            quaternion_from_euler(float(eta[i, 3]), float(eta[i, 4]), float(eta[i, 5]))
            for i in range(count)
        ]
    )
    return BatchedControlContext(
        eta=torch.from_numpy(eta),
        nu=torch.from_numpy(nu),
        quat_xyzw=torch.from_numpy(quaternion),
        current_ned_mps=torch.from_numpy(current),
        reference_quat_xyzw=torch.from_numpy(reference_quaternion),
        command_speed_mps=torch.from_numpy(speed),
        observation=torch.from_numpy(observation),
        dt_s=0.02,
    )


def test_device_matches_unindexed_cuda_and_explicit_index() -> None:
    assert _device_matches(torch.device("cuda:0"), torch.device("cuda"))
    assert _device_matches(torch.device("cuda:1"), torch.device("cuda:1"))
    assert not _device_matches(torch.device("cuda:1"), torch.device("cuda:0"))
    assert not _device_matches(torch.device("cpu"), torch.device("cuda"))
    assert _device_matches(torch.device("cpu"), torch.device("cpu"))


def test_batched_teacher_matches_scalar_controller_and_shadow_integral() -> None:
    config = load_config("configs/lowlevel/odarrl_control_base.json")
    vehicle = remus100_vehicle()
    teacher = BatchedRemus100Teacher(config.controller, vehicle)
    teacher.reset(count=4)
    shadow = BatchedShadowSpeedIntegrator(config.controller, 1525.0)
    shadow.reset(count=4)
    rng = np.random.default_rng(61)
    count = 4
    eta = np.column_stack(
        (
            np.zeros((count, 3)),
            rng.uniform(-0.1, 0.1, size=(count, 2)),
            rng.uniform(-0.4, 0.4, size=count),
        )
    )
    nu = np.concatenate(
        (
            rng.uniform(0.8, 1.5, size=(count, 1)),
            rng.uniform(-0.1, 0.1, size=(count, 5)),
        ),
        axis=1,
    )
    reference_quaternion = np.stack(
        [quaternion_from_euler(0.0, 0.1, 0.15 * (index + 1)) for index in range(count)]
    )
    speed = np.full(count, 1.2)
    current = np.tile(np.array([0.2, 0.1, 0.0]), (count, 1))
    observation = np.zeros((count, 27), dtype=np.float32)
    dynamics = BatchedRemus100Dynamics(dtype=torch.float64)
    scalar_teachers = [
        Remus100TrackingController(
            vehicle,
            current_field=UniformCurrent(velocity_ned_mps=np.array([0.2, 0.1, 0.0])),
            nominal_step_s=config.controller.nominal_step_s,
            heading_gain=config.controller.heading_gain,
            heading_rate_gain=config.controller.heading_rate_gain,
            pitch_gain=config.controller.pitch_gain,
            pitch_rate_gain=config.controller.pitch_rate_gain,
            speed_feedforward_rpm_per_mps=config.controller.speed_feedforward_rpm_per_mps,
            speed_gain_rpm_per_mps=config.controller.speed_gain_rpm_per_mps,
            speed_integral_gain_rpm_per_mps_s=(
                config.controller.speed_integral_gain_rpm_per_mps_s
            ),
            max_speed_integral_rpm=config.controller.max_speed_integral_rpm,
        )
        for _ in range(count)
    ]
    for step in range(5):
        context = _teacher_context(
            count,
            eta,
            nu,
            reference_quaternion,
            speed,
            current,
            observation,
        )
        expected = []
        for index, controller in enumerate(scalar_teachers):
            state = State(
                t_s=0.02 * step,
                position_ned_m=eta[index, :3],
                quat_xyzw=quaternion_from_euler(
                    float(eta[index, 3]), float(eta[index, 4]), float(eta[index, 5])
                ),
                linear_velocity_body_mps=nu[index, :3],
                angular_velocity_body_rps=nu[index, 3:],
            )
            reference = Reference(
                stamp_s=0.02 * step,
                position_ned_m=np.zeros(3),
                quat_xyzw=reference_quaternion[index],
                linear_velocity_body_mps=np.array([1.2, 0.0, 0.0]),
                angular_velocity_body_rps=np.zeros(3),
                valid_until_s=0.02 * step + 0.5,
            )
            command = controller.control(state, reference)
            expected.append(
                [command.rudder_angle_rad, command.stern_plane_angle_rad, command.propeller_rpm]
            )
        actual = teacher.control(context).numpy()
        np.testing.assert_allclose(actual, expected, atol=1e-12)
        diagnostics = [controller.last_diagnostics for controller in scalar_teachers]
        assert all(item is not None for item in diagnostics)
        shadow.advance(
            torch.tensor(
                [item.relative_surge_speed_mps for item in diagnostics if item is not None],
                dtype=torch.float64,
            ),
            torch.full((count,), 1.2, dtype=torch.float64),
            dt_s=0.02,
        )
        np.testing.assert_allclose(
            shadow.value_rpm.numpy(),
            [item.speed_integral_rpm for item in diagnostics if item is not None],
            atol=1e-12,
        )
        result = dynamics.step(
            Remus100BatchState(
                eta=torch.from_numpy(eta),
                nu=torch.from_numpy(nu),
                actuator=torch.from_numpy(np.zeros((count, 3))),
            ),
            torch.from_numpy(actual),
            torch.from_numpy(current[:, :2]),
        )
        eta = result.state.eta.numpy()
        nu = result.state.nu.numpy()


def test_batched_bc_matches_scalar_policy_on_smoke_checkpoint() -> None:
    checkpoint = "artifacts/lowlevel/i_bc_smoke_02/bc/pi0.pt"
    scalar = BCPolicy(checkpoint, device="cpu")
    batched = BatchedBCPolicy(checkpoint, device="cpu")
    rng = np.random.default_rng(71)
    observation = rng.normal(size=(32, 27)).astype(np.float32)
    context = BatchedControlContext(
        eta=torch.zeros((32, 6), dtype=torch.float64),
        nu=torch.zeros((32, 6), dtype=torch.float64),
        quat_xyzw=torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=torch.float64).expand(32, 4),
        current_ned_mps=torch.zeros((32, 3), dtype=torch.float64),
        reference_quat_xyzw=torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=torch.float64).expand(32, 4),
        command_speed_mps=torch.full((32,), 1.2, dtype=torch.float64),
        observation=torch.from_numpy(observation),
        dt_s=0.02,
    )
    actual = batched.control(context).numpy()
    expected = scalar.act(observation)
    np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-5)
