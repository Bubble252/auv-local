from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from auv_core.dynamics.remus100 import _load_pvs

from auv_lowlevel.batch.dynamics import BatchedRemus100Dynamics, Remus100BatchState


def _default_pvs_source() -> Path:
    return (
        Path(__file__).resolve().parents[2]
        / "python"
        / "PythonVehicleSimulator-master"
        / "src"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate batched REMUS100 against PVS.")
    parser.add_argument("--samples", type=int, default=10_000)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--open-loop-envs", type=int, default=3)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--pvs-source", type=Path, default=_default_pvs_source())
    parser.add_argument("--json-output")
    args = parser.parse_args()
    if args.samples <= 0 or args.steps <= 0 or args.open_loop_envs <= 0:
        parser.error("--samples, --steps and --open-loop-envs must be positive")

    model_type, attitude_euler = _load_pvs(str(args.pvs_source))
    device = torch.device(args.device)
    dtype = torch.float32 if args.dtype == "float32" else torch.float64
    dynamics = BatchedRemus100Dynamics(device=device, dtype=dtype)
    rng = np.random.default_rng(args.seed)

    sample_count = args.samples
    eta = np.column_stack(
        (
            rng.uniform(-10.0, 10.0, size=(sample_count, 3)),
            rng.uniform(-0.3, 0.3, size=(sample_count, 2)),
            rng.uniform(-np.pi, np.pi, size=sample_count),
        )
    )
    nu = np.column_stack(
        (
            rng.uniform(0.8, 1.5, size=sample_count),
            rng.uniform(-0.2, 0.2, size=(sample_count, 2)),
            rng.uniform(-0.1, 0.1, size=(sample_count, 3)),
        )
    )
    actuator = np.column_stack(
        (
            rng.uniform(-0.2, 0.2, size=(sample_count, 2)),
            rng.uniform(-100.0, 1600.0, size=sample_count),
        )
    )
    command = np.column_stack(
        (
            rng.uniform(-0.2, 0.2, size=(sample_count, 2)),
            rng.uniform(0.0, 1525.0, size=sample_count),
        )
    )
    current = rng.uniform(-0.4, 0.4, size=(sample_count, 2))
    result = dynamics.step(
        Remus100BatchState(
            eta=torch.as_tensor(eta, device=device, dtype=dtype),
            nu=torch.as_tensor(nu, device=device, dtype=dtype),
            actuator=torch.as_tensor(actuator, device=device, dtype=dtype),
        ),
        torch.as_tensor(command, device=device, dtype=dtype),
        torch.as_tensor(current, device=device, dtype=dtype),
    )
    result_np = {
        "eta": result.state.eta.detach().cpu().numpy(),
        "nu": result.state.nu.detach().cpu().numpy(),
        "actuator": result.state.actuator.detach().cpu().numpy(),
    }
    errors = {
        "eta_max_abs": 0.0,
        "nu_max_abs": 0.0,
        "actuator_max_abs": 0.0,
        "nu_dot_max_abs": 0.0,
        "nu_dot_max_relative": 0.0,
        "nu_dot_max_relative_significant": 0.0,
    }
    nu_dot = result.nu_dot.detach().cpu().numpy()
    for index in range(sample_count):
        model = model_type(controlSystem="stepInput")
        model.V_c = float(np.linalg.norm(current[index]))
        model.beta_c = float(np.arctan2(current[index, 1], current[index, 0]))
        nu_trial = nu[index].copy()
        model.dynamics(
            eta[index].copy(),
            nu_trial,
            actuator[index].copy(),
            actuator[index].copy(),
            1.0,
        )
        nu_dot_ref = nu_trial - nu[index]
        nu_ref, actuator_ref = model.dynamics(
            eta[index].copy(),
            nu[index].copy(),
            actuator[index].copy(),
            command[index].copy(),
            dynamics.sample_time_s,
        )
        eta_ref = attitude_euler(eta[index].copy(), nu_ref.copy(), dynamics.sample_time_s)
        errors["eta_max_abs"] = max(
            errors["eta_max_abs"],
            float(np.max(np.abs(result_np["eta"][index] - eta_ref))),
        )
        errors["nu_max_abs"] = max(
            errors["nu_max_abs"],
            float(np.max(np.abs(result_np["nu"][index] - nu_ref))),
        )
        errors["actuator_max_abs"] = max(
            errors["actuator_max_abs"],
            float(np.max(np.abs(result_np["actuator"][index] - actuator_ref))),
        )
        nu_dot_difference = np.abs(nu_dot[index] - nu_dot_ref)
        errors["nu_dot_max_abs"] = max(
            errors["nu_dot_max_abs"],
            float(np.max(nu_dot_difference)),
        )
        errors["nu_dot_max_relative"] = max(
            errors["nu_dot_max_relative"],
            float(np.max(nu_dot_difference / np.maximum(np.abs(nu_dot_ref), 1e-8))),
        )
        significant = np.abs(nu_dot_ref) >= 1e-3
        if bool(np.any(significant)):
            errors["nu_dot_max_relative_significant"] = max(
                errors["nu_dot_max_relative_significant"],
                float(
                    np.max(
                        nu_dot_difference[significant]
                        / np.abs(nu_dot_ref[significant])
                    )
                ),
            )

    open_loop_count = min(sample_count, args.open_loop_envs)
    open_eta = eta[:open_loop_count].copy()
    open_nu = nu[:open_loop_count].copy()
    open_actuator = actuator[:open_loop_count].copy()
    open_command = command[:open_loop_count].copy()
    open_current = current[:open_loop_count].copy()
    batch_state = Remus100BatchState(
        eta=torch.as_tensor(open_eta, device=device, dtype=dtype),
        nu=torch.as_tensor(open_nu, device=device, dtype=dtype),
        actuator=torch.as_tensor(open_actuator, device=device, dtype=dtype),
    )
    command_tensor = torch.as_tensor(open_command, device=device, dtype=dtype)
    current_tensor = torch.as_tensor(open_current, device=device, dtype=dtype)
    for _ in range(args.steps):
        batch_state = dynamics.step(batch_state, command_tensor, current_tensor).state
        for index in range(open_loop_count):
            model = model_type(controlSystem="stepInput")
            model.V_c = float(np.linalg.norm(open_current[index]))
            model.beta_c = float(np.arctan2(open_current[index, 1], open_current[index, 0]))
            nu_ref, actuator_ref = model.dynamics(
                open_eta[index].copy(),
                open_nu[index].copy(),
                open_actuator[index].copy(),
                open_command[index].copy(),
                dynamics.sample_time_s,
            )
            open_eta[index] = attitude_euler(
                open_eta[index].copy(),
                nu_ref.copy(),
                dynamics.sample_time_s,
            )
            open_nu[index] = nu_ref
            open_actuator[index] = actuator_ref

    errors["open_loop_eta_max_abs"] = float(
        np.max(np.abs(batch_state.eta.detach().cpu().numpy() - open_eta))
    )
    errors["open_loop_nu_max_abs"] = float(
        np.max(np.abs(batch_state.nu.detach().cpu().numpy() - open_nu))
    )
    errors["open_loop_actuator_max_abs"] = float(
        np.max(np.abs(batch_state.actuator.detach().cpu().numpy() - open_actuator))
    )
    payload = {
        "samples": sample_count,
        "open_loop_steps": args.steps,
        "open_loop_envs": open_loop_count,
        "dtype": args.dtype,
        "device": str(device),
        "errors": errors,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    if args.json_output:
        with open(args.json_output, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
