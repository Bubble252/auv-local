from __future__ import annotations

from typing import Protocol, runtime_checkable

from auv_planner.contracts import (
    ActuatorCommand,
    CapabilitySet,
    ControlProposal,
    Reference,
    SafeCommand,
    SafetyContext,
    State,
)
from auv_planner.interfaces import Guidance

from .contracts import (
    LocalPathPlan,
    LocalPathRequest,
    Reaction,
    ReactiveRequest,
    SurgeYawAction,
)


@runtime_checkable
class LocalPathPlanner(Protocol):
    def reset(self, *, seed: int = 0) -> None: ...

    def plan_local(self, request: LocalPathRequest) -> LocalPathPlan: ...


@runtime_checkable
class ReactivePolicy(Protocol):
    def reset(self, *, seed: int = 0) -> None: ...

    def decide(self, request: ReactiveRequest) -> Reaction: ...


@runtime_checkable
class LocalMotionModel(Protocol):
    def step(self, state: State, action: SurgeYawAction, dt_s: float) -> State: ...


@runtime_checkable
class MotionArbiter(Protocol):
    def arbitrate(
        self,
        nominal: Reference,
        reaction: Reaction | None,
        request: ReactiveRequest,
    ) -> ControlProposal: ...


@runtime_checkable
class SafetyFilter(Protocol):
    def filter(self, proposal: ControlProposal, context: SafetyContext) -> SafeCommand: ...


@runtime_checkable
class MotionExecutor(Protocol):
    def execute(
        self,
        state: State,
        nominal_reference: Reference | None,
        command: SafeCommand,
        capability: CapabilitySet,
    ) -> ActuatorCommand: ...


__all__ = [
    "Guidance",
    "LocalMotionModel",
    "LocalPathPlanner",
    "MotionArbiter",
    "MotionExecutor",
    "ReactivePolicy",
    "SafetyFilter",
]
