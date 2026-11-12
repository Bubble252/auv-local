from __future__ import annotations

from types import SimpleNamespace

import torch

from auv_lowlevel.batch.controllers import (
    BatchedControlContext,
    BatchedRemus100Teacher,
    controller_state_from_actor_state,
)
from auv_lowlevel.config import ControllerConfig


def test_controller_nu_reorders_angular_rate_and_linear_velocity() -> None:
    actor_state = torch.tensor(
        [[
            10.0,
            20.0,
            30.0,
            0.1,
            0.2,
            0.3,
            0.4,
            0.5,
            0.6,
            1.1,
            1.2,
            1.3,
        ]],
        dtype=torch.float64,
    )
    eta, nu = controller_state_from_actor_state(actor_state)
    torch.testing.assert_close(eta, actor_state[:, :6])
    torch.testing.assert_close(
        nu,
        torch.tensor([[1.1, 1.2, 1.3, 0.4, 0.5, 0.6]], dtype=torch.float64),
    )


def test_teacher_speed_loop_uses_explicit_water_surge() -> None:
    config = ControllerConfig(
        use_current_field=False,
        nominal_step_s=0.02,
        heading_gain=2.0,
        pitch_gain=2.0,
        heading_rate_gain=0.6,
        pitch_rate_gain=1.0,
        speed_feedforward_rpm_per_mps=600.0,
        speed_gain_rpm_per_mps=80.0,
        speed_integral_gain_rpm_per_mps_s=10.0,
        max_speed_integral_rpm=400.0,
    )
    limits = SimpleNamespace(
        max_rudder_angle_rad=0.35,
        max_stern_plane_angle_rad=0.35,
        max_rudder_rate_rad_s=2.6,
        max_stern_plane_rate_rad_s=2.6,
        max_propeller_rpm=1525.0,
    )
    controller = BatchedRemus100Teacher(
        config,
        limits,
        dtype=torch.float64,
    )
    controller.reset(count=1)
    context = BatchedControlContext(
        eta=torch.zeros((1, 6), dtype=torch.float64),
        nu=torch.tensor([[2.0, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=torch.float64),
        quat_xyzw=torch.tensor([[0.0, 0.0, 0.0, 1.0]], dtype=torch.float64),
        current_ned_mps=torch.zeros((1, 3), dtype=torch.float64),
        reference_quat_xyzw=torch.tensor(
            [[0.0, 0.0, 0.0, 1.0]], dtype=torch.float64
        ),
        command_speed_mps=torch.tensor([1.0], dtype=torch.float64),
        observation=torch.zeros((1, 12), dtype=torch.float64),
        dt_s=0.02,
        water_surge_mps=torch.tensor([1.0], dtype=torch.float64),
    )
    actuator = controller.control(context)
    # ground surge is 2.0, but water surge is explicitly 1.0, so there is no
    # speed error and the integrator must not wind up.
    torch.testing.assert_close(actuator[:, 2], torch.tensor([600.0], dtype=torch.float64))
    torch.testing.assert_close(
        controller.speed_integral_rpm,
        torch.zeros(1, dtype=torch.float64),
    )

