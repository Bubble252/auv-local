from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch

from auv_lowlevel.batch.controllers import BatchedControlContext, BatchedRemus100Teacher
from auv_lowlevel.config import load_config


def _actuator_limits() -> SimpleNamespace:
    return SimpleNamespace(
        max_rudder_angle_rad=float(np.deg2rad(15.0)),
        max_stern_plane_angle_rad=float(np.deg2rad(15.0)),
        max_rudder_rate_rad_s=float(np.deg2rad(15.0) / 0.1),
        max_stern_plane_rate_rad_s=float(np.deg2rad(15.0) / 0.1),
        max_propeller_rpm=1525.0,
    )


def _control_context(current_ned_mps: torch.Tensor) -> BatchedControlContext:
    batch = 2
    eta = torch.tensor(
        [[0.0, 0.0, 10.0, 0.0, 0.0, 0.05], [2.0, 1.0, 10.0, 0.0, 0.0, -0.08]],
        dtype=torch.float64,
    )
    nu = torch.tensor(
        [[1.1, 0.02, 0.0, 0.0, 0.0, 0.01], [1.2, -0.03, 0.0, 0.0, 0.0, -0.02]],
        dtype=torch.float64,
    )
    yaw = eta[:, 5]
    quat = torch.stack(
        [
            torch.zeros(batch, dtype=torch.float64),
            torch.zeros(batch, dtype=torch.float64),
            torch.sin(yaw / 2.0),
            torch.cos(yaw / 2.0),
        ],
        dim=1,
    )
    return BatchedControlContext(
        eta=eta,
        nu=nu,
        quat_xyzw=quat,
        current_ned_mps=current_ned_mps,
        reference_quat_xyzw=quat.clone(),
        command_speed_mps=torch.full((batch,), 1.2, dtype=torch.float64),
        observation=torch.zeros((batch, 26), dtype=torch.float64),
        dt_s=0.02,
    )


def test_hidden_teacher_command_is_invariant_to_true_current() -> None:
    config = load_config("configs/lowlevel/odarrl_control_i_residual_26d_rpm_only_hidden_timevarying.json")
    assert config.controller.use_current_field is False
    teacher = BatchedRemus100Teacher(config.controller, _actuator_limits(), dtype=torch.float64)

    current_a = torch.tensor([[0.35, 0.10, 0.0], [-0.25, 0.08, 0.0]], dtype=torch.float64)
    current_b = torch.tensor([[-0.40, -0.20, 0.0], [0.30, -0.15, 0.0]], dtype=torch.float64)

    teacher.reset(count=2)
    command_a = teacher.control(_control_context(current_a))
    teacher.reset(count=2)
    command_b = teacher.control(_control_context(current_b))

    torch.testing.assert_close(command_a, command_b, atol=0.0, rtol=0.0)


def test_teacher_current_feedforward_changes_command_when_enabled() -> None:
    config = load_config("configs/lowlevel/odarrl_control_v2_26d.json")
    assert config.controller.use_current_field is True
    teacher = BatchedRemus100Teacher(config.controller, _actuator_limits(), dtype=torch.float64)

    current_a = torch.tensor([[0.35, 0.10, 0.0], [-0.25, 0.08, 0.0]], dtype=torch.float64)
    current_b = torch.tensor([[-0.40, -0.20, 0.0], [0.30, -0.15, 0.0]], dtype=torch.float64)

    teacher.reset(count=2)
    command_a = teacher.control(_control_context(current_a))
    teacher.reset(count=2)
    command_b = teacher.control(_control_context(current_b))

    assert not torch.allclose(command_a, command_b)
