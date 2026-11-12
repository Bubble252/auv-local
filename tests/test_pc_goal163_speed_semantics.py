from __future__ import annotations

import torch

from auv_lowlevel.batch.geometry import (
    ground_velocity_body,
    legacy_ground_velocity_body,
    quaternion_from_euler,
    water_velocity_body,
)


def _straight_line_state() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    eta = torch.zeros((1, 6), dtype=torch.float64)
    nu = torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=torch.float64)
    current = torch.tensor([[0.5, 0.0, 0.0]], dtype=torch.float64)
    return eta, nu, current


def test_eta_forward_difference_equals_nu_for_ground_velocity_contract() -> None:
    """The simulated ``eta`` derivative identifies ``nu`` as ground velocity."""

    eta, nu, _current = _straight_line_state()
    dt = 0.1
    eta_next = eta.clone()
    eta_next[:, :3] = eta[:, :3] + dt * nu[:, :3]
    measured_ground = (eta_next[:, :3] - eta[:, :3]) / dt
    torch.testing.assert_close(measured_ground, nu[:, :3])


def test_ground_velocity_helper_matches_eta_forward_difference() -> None:
    eta, nu, current = _straight_line_state()
    quat = quaternion_from_euler(eta[:, 3], eta[:, 4], eta[:, 5])
    helper = ground_velocity_body(eta, nu, current)
    assert quat.shape == (1, 4)
    torch.testing.assert_close(helper, nu[:, :3])


def test_water_velocity_helper_removes_current() -> None:
    eta, nu, current = _straight_line_state()
    helper = water_velocity_body(eta, nu, current)
    torch.testing.assert_close(helper, nu[:, :3] - current)


def test_legacy_ground_helper_keeps_bit_compatible_current_addition() -> None:
    eta, nu, current = _straight_line_state()
    helper = legacy_ground_velocity_body(eta, nu, current)
    torch.testing.assert_close(helper, nu[:, :3] + current)
