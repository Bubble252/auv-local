from __future__ import annotations

from auv_planner.contracts import ControlProposal, SafeCommand, SafetyContext

from .contracts import (
    ControlAuthority,
    LocalPathPlan,
    LocalPathRequest,
    LocalPlanStatus,
    Reaction,
    ReactiveRequest,
)


class NoLocalPathPlanner:
    """Explicit no-op planner used when a local path layer is disabled."""

    def __init__(self) -> None:
        self._seed = 0

    def reset(self, *, seed: int = 0) -> None:
        if not isinstance(seed, int):
            raise TypeError("seed must be an int")
        self._seed = seed

    def plan_local(self, request: LocalPathRequest) -> LocalPathPlan:
        if not isinstance(request, LocalPathRequest):
            raise TypeError("request must be a LocalPathRequest")
        return LocalPathPlan(
            status=LocalPlanStatus.NO_SOLUTION,
            decision_id=request.request_id,
            trajectory=None,
            reason="no local path planner configured",
            diagnostics={"seed": self._seed, "planner": type(self).__name__},
        )


class NoReactivePolicy:
    """Explicit no-op reactive policy used when the local layer is disabled."""

    def __init__(self) -> None:
        self._seed = 0

    def reset(self, *, seed: int = 0) -> None:
        if not isinstance(seed, int):
            raise TypeError("seed must be an int")
        self._seed = seed

    def decide(self, request: ReactiveRequest) -> Reaction:
        if not isinstance(request, ReactiveRequest):
            raise TypeError("request must be a ReactiveRequest")
        return Reaction(
            status=LocalPlanStatus.NO_SOLUTION,
            decision_id=request.request_id,
            motion=None,
            authority=ControlAuthority.NOMINAL,
            reason="no reactive policy configured",
            diagnostics={"seed": self._seed, "policy": type(self).__name__},
        )


class NoSafetyFilter:
    """Identity safety filter with an explicit and auditable pass-through."""

    def filter(self, proposal: ControlProposal, context: SafetyContext) -> SafeCommand:
        if not isinstance(proposal, ControlProposal):
            raise TypeError("proposal must be a ControlProposal")
        if not isinstance(context, SafetyContext):
            raise TypeError("context must be a SafetyContext")
        return SafeCommand(
            command=proposal,
            allowed=True,
            valid_until_s=proposal.valid_until_s,
        )


__all__ = ["NoLocalPathPlanner", "NoReactivePolicy", "NoSafetyFilter"]
