from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import torch
from auv_core.dynamics.remus100 import _load_pvs

from auv_lowlevel.batch.dynamics import BatchedRemus100Dynamics, Remus100BatchState

PVS_SOURCE = Path(
    os.environ.get(
        "PVS_SOURCE_DIR",
        Path(__file__).resolve().parents[2] / "python" / "PythonVehicleSimulator-master" / "src",
    )
)


def _reference_step(
    model_type: type,
    attitude_euler: object,
    eta: np.ndarray,
    nu: np.ndarray,
    actuator: np.ndarray,
    command: np.ndarray,
    current: np.ndarray,
    dt_s: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model = model_type(controlSystem="stepInput")
    model.V_c = float(np.linalg.norm(current))
    model.beta_c = float(np.arctan2(current[1], current[0]))
    nu_next, actuator_next = model.dynamics(
        eta.copy(),
        nu.copy(),
        actuator.copy(),
        command.copy(),
        dt_s,
    )
    eta_next = attitude_euler(eta.copy(), nu_next.copy(), dt_s)  # type: ignore[operator]
    return eta_next, nu_next, actuator_next


def _random_batch(count: int, seed: int) -> tuple[np.ndarray, ...]:
    rng = np.random.default_rng(seed)
    eta = np.column_stack(
        (
            rng.uniform(-10.0, 10.0, size=(count, 3)),
            rng.uniform(-0.3, 0.3, size=(count, 2)),
            rng.uniform(-np.pi, np.pi, size=count),
        )
    )
    nu = np.column_stack(
        (
            rng.uniform(0.8, 1.5, size=count),
            rng.uniform(-0.2, 0.2, size=(count, 2)),
            rng.uniform(-0.1, 0.1, size=(count, 3)),
        )
    )
    actuator = np.column_stack(
        (
            rng.uniform(-0.2, 0.2, size=(count, 2)),
            rng.uniform(-100.0, 1600.0, size=count),
        )
    )
    command = np.column_stack(
        (
            rng.uniform(-0.2, 0.2, size=(count, 2)),
            rng.uniform(0.0, 1525.0, size=count),
        )
    )
    current = rng.uniform(-0.4, 0.4, size=(count, 2))
    return eta, nu, actuator, command, current


def test_single_step_matches_pvs_for_random_states() -> None:
    model_type, attitude_euler = _load_pvs(str(PVS_SOURCE))
    eta, nu, actuator, command, current = _random_batch(32, seed=11)
    dynamics = BatchedRemus100Dynamics(dtype=torch.float64)

    result = dynamics.step(
        Remus100BatchState(
            eta=torch.from_numpy(eta),
            nu=torch.from_numpy(nu),
            actuator=torch.from_numpy(actuator),
        ),
        torch.from_numpy(command),
        torch.from_numpy(current),
    )

    for index in range(eta.shape[0]):
        eta_ref, nu_ref, actuator_ref = _reference_step(
            model_type,
            attitude_euler,
            eta[index],
            nu[index],
            actuator[index],
            command[index],
            current[index],
            dynamics.sample_time_s,
        )
        np.testing.assert_allclose(result.state.eta[index].numpy(), eta_ref, atol=1e-10, rtol=1e-10)
        np.testing.assert_allclose(result.state.nu[index].numpy(), nu_ref, atol=1e-10, rtol=1e-10)
        np.testing.assert_allclose(
            result.state.actuator[index].numpy(),
            actuator_ref,
            atol=1e-10,
            rtol=1e-10,
        )


def test_float32_single_step_meets_p0_tolerance() -> None:
    model_type, attitude_euler = _load_pvs(str(PVS_SOURCE))
    eta, nu, actuator, command, current = _random_batch(8, seed=17)
    dynamics = BatchedRemus100Dynamics(dtype=torch.float32)

    result = dynamics.step(
        Remus100BatchState(
            eta=torch.from_numpy(eta).to(torch.float32),
            nu=torch.from_numpy(nu).to(torch.float32),
            actuator=torch.from_numpy(actuator).to(torch.float32),
        ),
        torch.from_numpy(command).to(torch.float32),
        torch.from_numpy(current).to(torch.float32),
    )

    for index in range(eta.shape[0]):
        eta_ref, nu_ref, actuator_ref = _reference_step(
            model_type,
            attitude_euler,
            eta[index],
            nu[index],
            actuator[index],
            command[index],
            current[index],
            dynamics.sample_time_s,
        )
        assert np.max(np.abs(result.state.eta[index].numpy() - eta_ref)) < 1e-4
        assert np.max(np.abs(result.state.nu[index].numpy() - nu_ref)) < 1e-4
        assert np.max(np.abs(result.state.actuator[index].numpy() - actuator_ref)) < 1e-4


def test_500_step_open_loop_matches_pvs() -> None:
    model_type, attitude_euler = _load_pvs(str(PVS_SOURCE))
    rng = np.random.default_rng(23)
    count = 3
    eta = np.column_stack(
        (
            rng.uniform(-2.0, 2.0, size=(count, 3)),
            rng.uniform(-0.1, 0.1, size=(count, 2)),
            rng.uniform(-1.0, 1.0, size=count),
        )
    )
    nu = np.column_stack(
        (
            np.full(count, 1.2),
            rng.uniform(-0.05, 0.05, size=(count, 2)),
            rng.uniform(-0.02, 0.02, size=(count, 3)),
        )
    )
    actuator = np.zeros((count, 3), dtype=np.float64)
    command = np.tile(np.array([0.03, -0.01, 800.0]), (count, 1))
    current = np.tile(np.array([0.2, 0.1]), (count, 1))
    dynamics = BatchedRemus100Dynamics(dtype=torch.float64)
    state = Remus100BatchState(
        eta=torch.from_numpy(eta),
        nu=torch.from_numpy(nu),
        actuator=torch.from_numpy(actuator),
    )
    command_tensor = torch.from_numpy(command)
    current_tensor = torch.from_numpy(current)

    reference_eta = eta.copy()
    reference_nu = nu.copy()
    reference_actuator = actuator.copy()
    for _ in range(500):
        state = dynamics.step(state, command_tensor, current_tensor).state
        for index in range(count):
            eta_ref, nu_ref, actuator_ref = _reference_step(
                model_type,
                attitude_euler,
                reference_eta[index],
                reference_nu[index],
                reference_actuator[index],
                command[index],
                current[index],
                dynamics.sample_time_s,
            )
            reference_eta[index] = eta_ref
            reference_nu[index] = nu_ref
            reference_actuator[index] = actuator_ref

    np.testing.assert_allclose(state.eta.numpy(), reference_eta, atol=1e-9, rtol=1e-9)
    np.testing.assert_allclose(state.nu.numpy(), reference_nu, atol=1e-9, rtol=1e-9)
    np.testing.assert_allclose(
        state.actuator.numpy(),
        reference_actuator,
        atol=1e-9,
        rtol=1e-9,
    )


def test_rejects_wrong_shapes_and_vertical_current_shape() -> None:
    dynamics = BatchedRemus100Dynamics(dtype=torch.float64)
    state = Remus100BatchState(
        eta=torch.zeros((2, 6), dtype=torch.float64),
        nu=torch.zeros((2, 6), dtype=torch.float64),
        actuator=torch.zeros((2, 3), dtype=torch.float64),
    )

    with pytest.raises(ValueError, match="current_velocity_ned_mps"):
        dynamics.step(
            state,
            torch.zeros((2, 3), dtype=torch.float64),
            torch.zeros((2, 3), dtype=torch.float64),
        )
    with pytest.raises(ValueError, match="state.nu"):
        dynamics.step(
            Remus100BatchState(
                eta=state.eta,
                nu=torch.zeros((2, 5), dtype=torch.float64),
                actuator=state.actuator,
            ),
            torch.zeros((2, 3), dtype=torch.float64),
            torch.zeros((2, 2), dtype=torch.float64),
        )
    with pytest.raises(ValueError, match="dtype"):
        dynamics.step(
            Remus100BatchState(
                eta=state.eta.to(torch.float32),
                nu=state.nu,
                actuator=state.actuator,
            ),
            torch.zeros((2, 3), dtype=torch.float64),
            torch.zeros((2, 2), dtype=torch.float64),
        )


def test_optional_finite_validation_rejects_nan() -> None:
    dynamics = BatchedRemus100Dynamics(dtype=torch.float64, validate_finite=True)
    state = Remus100BatchState(
        eta=torch.zeros((1, 6), dtype=torch.float64),
        nu=torch.zeros((1, 6), dtype=torch.float64),
        actuator=torch.zeros((1, 3), dtype=torch.float64),
    )
    bad_command = torch.zeros((1, 3), dtype=torch.float64)
    bad_command[0, 0] = torch.nan
    with pytest.raises(ValueError, match="NaN or Inf"):
        dynamics.step(state, bad_command, torch.zeros((1, 2), dtype=torch.float64))
