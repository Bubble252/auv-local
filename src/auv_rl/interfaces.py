from __future__ import annotations

from typing import Protocol, runtime_checkable

from auv_core.contracts import FloatArray

from .contracts import ActionSpec, ObservationSpec


@runtime_checkable
class RLPolicy(Protocol):
    observation_spec: ObservationSpec
    action_spec: ActionSpec

    def reset(self, *, seed: int = 0) -> None: ...

    def act(self, observation: FloatArray) -> FloatArray: ...


__all__ = ["RLPolicy"]
