from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from auv_core.contracts import (
    Reference,
    Remus100ActuatorCommand,
    ScenarioConfig,
    SimulationSnapshot,
    State,
)
from auv_core.control.remus100 import Remus100TrackingController
from auv_core.dynamics.remus100 import PvsRemus100Backend


@dataclass(frozen=True)
class ExecutionStep:
    state: State
    command: Remus100ActuatorCommand
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.state, State):
            raise TypeError("state must be a State")
        if not isinstance(self.command, Remus100ActuatorCommand):
            raise TypeError("command must be a Remus100ActuatorCommand")
        object.__setattr__(self, "diagnostics", dict(self.diagnostics))


@runtime_checkable
class ReferenceExecutor(Protocol):
    def reset(self) -> State: ...

    def step(self, reference: Reference) -> ExecutionStep: ...


class PvsRemus100ReferenceExecutor:
    """PVS-backed adapter from ALOS Reference to REMUS actuator commands.

    This is the only local module that imports the global REMUS controller and
    PVS backend. The local planner and ALOS never see actuator-level types.
    """

    def __init__(
        self,
        scenario: ScenarioConfig,
        *,
        sample_time_s: float = 0.02,
        pvs_source_dir: Path | str | None = None,
    ) -> None:
        if not isinstance(scenario, ScenarioConfig):
            raise TypeError("scenario must be a ScenarioConfig")
        self.scenario = scenario
        self.sample_time_s = float(sample_time_s)
        self.controller = Remus100TrackingController(
            scenario.vehicle,
            current_field=scenario.world.current_field,
            nominal_step_s=self.sample_time_s,
        )
        self.backend = PvsRemus100Backend(
            sample_time_s=self.sample_time_s,
            pvs_source_dir=pvs_source_dir,
        )
        self._snapshot: SimulationSnapshot | None = None

    def reset(self) -> State:
        self.controller.reset()
        self._snapshot = self.backend.reset(self.scenario, seed=0)
        return self._snapshot.state

    def step(self, reference: Reference) -> ExecutionStep:
        if self._snapshot is None:
            raise RuntimeError("reset() must be called before step()")
        if not isinstance(reference, Reference):
            raise TypeError("reference must be a Reference")
        state = self._snapshot.state
        if abs(reference.stamp_s - state.t_s) > 1e-9:
            raise ValueError("reference stamp must match the current state timestamp")
        if reference.valid_until_s is None:
            raise ValueError("M2 requires a finite reference validity window")
        if state.t_s > reference.valid_until_s + 1e-9:
            raise ValueError("reference has expired before execution")

        command = self.controller.control(state, reference)
        if abs(command.stamp_s - state.t_s) > 1e-9:
            raise ValueError("controller command timestamp must match the current state")
        next_snapshot = self.backend.step(command)
        if next_snapshot.state.t_s <= state.t_s:
            raise ValueError("PVS state time must increase after a step")
        self._snapshot = next_snapshot
        diagnostics: dict[str, Any] = dict(next_snapshot.diagnostics)
        if self.controller.last_diagnostics is not None:
            diagnostics["heading_error_rad"] = self.controller.last_diagnostics.heading_error_rad
            diagnostics["pitch_error_rad"] = self.controller.last_diagnostics.pitch_error_rad
        return ExecutionStep(
            state=next_snapshot.state,
            command=command,
            diagnostics=diagnostics,
        )


__all__ = [
    "ExecutionStep",
    "PvsRemus100ReferenceExecutor",
    "ReferenceExecutor",
]
