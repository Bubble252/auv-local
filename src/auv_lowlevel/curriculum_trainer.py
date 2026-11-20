"""Training-loop hooks for the C-line curriculum."""

from __future__ import annotations

import json
import subprocess
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from .curriculum import (
    CurriculumDecision,
    CurriculumMixRamp,
    CurriculumPromotionController,
    CurriculumStageSampler,
    RouteGateThresholds,
    RouteGateThresholdTable,
    RouteGateWindow,
    StageBudgetStatus,
    StageEvaluation,
    curriculum_stage_mix_target,
)


def parse_eval_summary(summary: Mapping[str, Any]) -> StageEvaluation:
    """Convert one frozen evaluation summary into a stage evaluation.

    Only safety, completion, and route RMSE feed the StageGate.  Success,
    timeout, plane-miss, 3D RMSE, and pre-goal 3D RMSE remain diagnostics.
    """

    episodes = summary.get("episodes")
    if not isinstance(episodes, list) or not episodes:
        raise ValueError("evaluation summary must contain a non-empty episodes list")

    def rate(name: str) -> float:
        values = [bool(row.get(name, False)) for row in episodes]
        return float(sum(values)) / float(len(values))

    def mean_metric(name: str) -> float | None:
        values = [float(row[name]) for row in episodes if row.get(name) is not None]
        if not values:
            return None
        return float(sum(values)) / float(len(values))

    default_success = rate("success")
    eta_tight_success_rate = rate("success_eta_tight_at_done")
    timeout_rate = rate("timeout")
    if "goal_plane_miss" in episodes[0]:
        plane_miss_rate = rate("goal_plane_miss")
    else:
        plane_miss_rate = rate("plane_miss")
    safety_rate = rate("safety")
    completion = mean_metric("path_completion_ratio")
    if completion is None:
        fallback = summary.get("path_completion_ratio_mean")
        completion = None if fallback is None else float(fallback)
    if completion is None:
        raise ValueError(
            "evaluation summary is missing path_completion_ratio; the StageGate "
            "cannot run without its completion guard"
        )
    return StageEvaluation(
        safety_rate=safety_rate,
        path_completion_ratio=completion,
        cross_track_rmse_m=mean_metric("path_horizontal_rmse_m"),
        vertical_rmse_m=mean_metric("path_vertical_rmse_m"),
        default_success=default_success,
        eta_tight_success_rate=eta_tight_success_rate,
        timeout_rate=timeout_rate,
        plane_miss_rate=plane_miss_rate,
        path_three_d_rmse_m=mean_metric("path_three_d_rmse_m"),
        pre_goal_3d_rmse_m=mean_metric("pre_goal_3d_rmse_m"),
    )


def run_eval_command(
    command: Sequence[str],
    *,
    output_dir: Path,
    timeout_s: float | None = None,
) -> dict[str, Any]:
    """Run one external evaluator and return its ``summary.json``."""

    output_dir.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(
        [str(part) for part in command],
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout_s,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "curriculum evaluation failed with exit code "
            f"{completed.returncode}: {completed.stderr[-2000:]}"
        )
    summary_path = output_dir / "summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(f"evaluator did not write {summary_path}")
    return json.loads(summary_path.read_text())


class CurriculumTrainerHook:
    """Drive promotion/rollback and stage-mix changes from frozen dev evals."""

    def __init__(
        self,
        *,
        controller: CurriculumPromotionController,
        sampler: CurriculumStageSampler,
        output_dir: Path,
        eval_interval_transitions: int,
        evaluate: Callable[[str, int], StageEvaluation],
        mix_ramp_transitions: int = 0,
        preview_enabled: bool = True,
        force_current_encoder_alpha_zero: bool = False,
        promotion_enabled: bool = True,
        routine_gate: RouteGateWindow | None = None,
        route_gate_thresholds: RouteGateThresholds | RouteGateThresholdTable | None = None,
        stage_boundary_evals: bool = False,
        stage_boundary_retry_iterations: int | None = None,
    ) -> None:
        if isinstance(eval_interval_transitions, bool) or eval_interval_transitions <= 0:
            raise ValueError("eval_interval_transitions must be a positive integer")
        if isinstance(mix_ramp_transitions, bool) or mix_ramp_transitions < 0:
            raise ValueError("mix_ramp_transitions must be a non-negative integer")
        self.controller = controller
        self.sampler = sampler
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.eval_interval_transitions = int(eval_interval_transitions)
        self.evaluate = evaluate
        self.mix_ramp_transitions = int(mix_ramp_transitions)
        self.preview_enabled = bool(preview_enabled)
        self.force_current_encoder_alpha_zero = bool(force_current_encoder_alpha_zero)
        self.promotion_enabled = bool(promotion_enabled)
        self.routine_gate = routine_gate
        self.route_gate_thresholds = route_gate_thresholds
        self.stage_boundary_evals = bool(stage_boundary_evals)
        self.stage_boundary_retry_iterations: int | None = None
        self._next_stage_eval_iteration: int | None = None
        if self.stage_boundary_evals:
            retry_iterations = stage_boundary_retry_iterations
            if retry_iterations is None and routine_gate is not None:
                retry_iterations = int(routine_gate.window_size)
            if retry_iterations is None:
                raise ValueError(
                    "stage-boundary evaluation requires a retry interval or routine gate"
                )
            if retry_iterations <= 0:
                raise ValueError("stage boundary retry iterations must be positive")
            self.stage_boundary_retry_iterations = int(retry_iterations)
            self._next_stage_eval_iteration = self._stage_boundary_target(0)
        if (
            routine_gate is not None
            and routine_gate.mode == "enforce"
            and route_gate_thresholds is None
        ):
            raise ValueError("route gate enforce mode requires frozen route thresholds")
        self._next_eval_transition = int(eval_interval_transitions)
        self._mix_ramp: CurriculumMixRamp | None = None
        self._stage_entered_at = 0
        self.decisions_path = self.output_dir / "curriculum_decisions.jsonl"
        self.budget_path = self.output_dir / "curriculum_stage_budget.jsonl"
        self.route_gate_path = self.output_dir / "curriculum_route_gate.jsonl"
        self.state_path = self.output_dir / "curriculum_state.json"
        self._last_state_payload: dict[str, Any] | None = None
        self._write_state(0, None)

    def _stage_boundary_target(self, entry_iteration: int) -> int:
        spec = self.controller.stages[self.controller.stage]
        budget = spec.soft_budget_iterations
        if budget is None:
            if self.stage_boundary_retry_iterations is None:
                raise RuntimeError("stage boundary retry interval is not configured")
            budget = self.stage_boundary_retry_iterations
        return int(entry_iteration) + int(budget)

    def _write_state(
        self,
        completed_transitions: int,
        decision: CurriculumDecision | None,
        *,
        completed_iterations: int | None = None,
        finalized: bool = False,
    ) -> None:
        payload = {
            "schema_version": "auv_pc_curriculum_trainer_state.v1",
            "completed_transitions": int(completed_transitions),
            "completed_iterations": (
                None if completed_iterations is None else int(completed_iterations)
            ),
            "stage": self.controller.stage,
            "stage_mix": [[key, float(weight)] for key, weight in self.sampler.stage_mix],
            "promotion_enabled": self.promotion_enabled,
            "frozen_stage": None if self.promotion_enabled else self.controller.stage,
            "route_gate": (
                None if self.routine_gate is None else dict(self.routine_gate.last_status)
            ),
            "route_gate_window": (
                None if self.routine_gate is None else self.routine_gate.state_dict()
            ),
            "last_decision": None if decision is None else decision.value,
            "rollbacks_used": self.controller.rollbacks_used,
            "force_current_encoder_alpha_zero": self.force_current_encoder_alpha_zero,
            "eval_schedule": ("stage_boundary" if self.stage_boundary_evals else "fixed_interval"),
            "next_eval_transition": (
                None if self.stage_boundary_evals else self._next_eval_transition
            ),
            "next_eval_iteration": self._next_stage_eval_iteration,
            "stage_entered_at": int(self._stage_entered_at),
            "controller_state": self.controller.state_dict(),
            "mix_ramp": (
                None if self._mix_ramp is None else self._mix_ramp.as_dict()
            ),
            "finalized": bool(finalized),
        }
        self._last_state_payload = payload
        self.state_path.write_text(json.dumps(payload, indent=1) + "\n")

    def state_dict(self) -> dict[str, Any]:
        """Return the latest complete state for checkpoint embedding."""

        if self._last_state_payload is None:
            raise RuntimeError("curriculum state has not been initialized")
        return json.loads(json.dumps(self._last_state_payload))

    def load_state(self, payload: Mapping[str, Any]) -> None:
        """Restore a state written by :meth:`_write_state` or :meth:`state_dict`."""

        stage = str(payload.get("stage", self.controller.stage))
        stage_mix = payload.get("stage_mix")
        if stage_mix is not None:
            self.sampler.set_stage_mix(
                [(str(key), float(weight)) for key, weight in stage_mix]
            )
        controller_state = payload.get("controller_state")
        if controller_state is not None:
            self.controller.load_state_dict(controller_state)
        else:
            consumed_transitions = dict(payload.get("stage_consumed", {}) or {})
            consumed_iterations = dict(payload.get("stage_consumed_iterations", {}) or {})
            self.controller.load_state_dict(
                {
                    "stage": stage,
                    "stage_consumed": {
                        stage: int(consumed_transitions.get(stage, 0))
                    },
                    "stage_consumed_iterations": {
                        stage: int(consumed_iterations.get(stage, 0))
                    },
                    "stage_entry_transition": 0,
                    "stage_entry_iteration": 0,
                }
            )
        if self.promotion_enabled and self.controller.stage != stage:
            raise ValueError(
                f"curriculum state stage mismatch: {self.controller.stage!r} != {stage!r}"
            )
        self._stage_entered_at = int(
            payload.get(
                "stage_entered_at",
                self.controller._stage_entry_transition or 0,
            )
        )
        if self.stage_boundary_evals:
            self._next_eval_transition = int(self.eval_interval_transitions)
            next_iteration = payload.get("next_eval_iteration")
            self._next_stage_eval_iteration = (
                None if next_iteration is None else int(next_iteration)
            )
            if self._next_stage_eval_iteration is None:
                completed_iterations = int(payload.get("completed_iterations", 0) or 0)
                self._next_stage_eval_iteration = self._stage_boundary_target(
                    completed_iterations
                )
        else:
            next_transition = payload.get("next_eval_transition")
            self._next_eval_transition = int(
                self.eval_interval_transitions
                if next_transition is None
                else next_transition
            )
            self._next_stage_eval_iteration = None
        mix_ramp = payload.get("mix_ramp")
        self._mix_ramp = (
            None if mix_ramp is None else CurriculumMixRamp.from_dict(mix_ramp)
        )
        route_gate_state = payload.get("route_gate_window")
        if self.routine_gate is not None and route_gate_state is not None:
            self.routine_gate.load_state_dict(route_gate_state)
        restored = dict(payload)
        restored["stage"] = self.controller.stage
        restored["stage_mix"] = [
            [key, float(weight)] for key, weight in self.sampler.stage_mix
        ]
        restored["controller_state"] = self.controller.state_dict()
        self._last_state_payload = restored
        self.state_path.write_text(json.dumps(restored, indent=1) + "\n")

    def _apply_mix_ramp(self, completed_transitions: int) -> None:
        if self._mix_ramp is None:
            return
        self.sampler.set_stage_mix(self._mix_ramp.mix_at(completed_transitions))
        if completed_transitions >= (self._mix_ramp.start_transition + self._mix_ramp.transitions):
            self._mix_ramp = None

    def maybe_update(
        self,
        completed_transitions: int,
        completed_iterations: int | None = None,
        training_metrics: Mapping[str, Any] | None = None,
    ) -> CurriculumDecision | None:
        """Evaluate at the next scheduled boundary, then promote or roll back."""

        if isinstance(completed_transitions, bool) or completed_transitions < 0:
            raise ValueError("completed_transitions must be a non-negative integer")
        if completed_iterations is not None and (
            isinstance(completed_iterations, bool) or completed_iterations < 0
        ):
            raise ValueError("completed_iterations must be a non-negative integer")
        self._apply_mix_ramp(completed_transitions)
        route_status: dict[str, Any] | None = None
        if self.routine_gate is not None and training_metrics is not None:
            self.controller.consume_iterations(completed_iterations)
            spec = self.controller.stages[self.controller.stage]
            route_status = self.routine_gate.observe(
                stage=self.controller.stage,
                iteration=(
                    0
                    if completed_iterations is None
                    else self.controller.stage_consumed_iterations
                ),
                metrics=training_metrics,
                min_iterations=int(spec.soft_budget_iterations or 0),
                thresholds=self.route_gate_thresholds,
            )
            with self.route_gate_path.open("a") as handle:
                handle.write(json.dumps(route_status, sort_keys=True) + "\n")
        if not self.promotion_enabled:
            if (
                not self.stage_boundary_evals
                and completed_transitions >= self._next_eval_transition
            ):
                while completed_transitions >= self._next_eval_transition:
                    self._next_eval_transition += self.eval_interval_transitions
            self._write_state(
                completed_transitions,
                None,
                completed_iterations=completed_iterations,
            )
            return None
        if self.routine_gate is not None and self.routine_gate.mode == "enforce":
            return self._maybe_update_enforce(
                completed_transitions,
                completed_iterations,
                route_status,
            )
        if self.stage_boundary_evals:
            if completed_iterations is None:
                raise ValueError("stage-boundary evaluation requires completed_iterations")
            assert self._next_stage_eval_iteration is not None
            if completed_iterations < self._next_stage_eval_iteration:
                self._write_state(
                    completed_transitions,
                    None,
                    completed_iterations=completed_iterations,
                )
                return None
        else:
            if completed_transitions < self._next_eval_transition:
                self._write_state(
                    completed_transitions,
                    None,
                    completed_iterations=completed_iterations,
                )
                return None
            while completed_transitions >= self._next_eval_transition:
                self._next_eval_transition += self.eval_interval_transitions
        evaluation = self.evaluate(self.controller.stage, completed_transitions)
        previous_stage = self.controller.stage
        decision = self.controller.observe(
            evaluation,
            completed_transitions=completed_transitions,
            completed_iterations=completed_iterations,
        )
        if self.controller.stage != previous_stage:
            self._stage_entered_at = int(completed_transitions)
            if self.stage_boundary_evals:
                assert completed_iterations is not None
                self._next_stage_eval_iteration = self._stage_boundary_target(completed_iterations)
        elif self.stage_boundary_evals:
            assert completed_iterations is not None
            assert self.stage_boundary_retry_iterations is not None
            self._next_stage_eval_iteration = (
                int(completed_iterations) + self.stage_boundary_retry_iterations
            )
        record = self.controller.history[-1].as_dict()
        with self.decisions_path.open("a") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        if decision in {
            CurriculumDecision.PROMOTE,
            CurriculumDecision.ROLLBACK,
        }:
            target_mix = curriculum_stage_mix_target(
                self.controller.stage,
                preview_enabled=self.preview_enabled,
            )
            if self.mix_ramp_transitions > 0:
                self._mix_ramp = CurriculumMixRamp(
                    start_mix=self.sampler.stage_mix,
                    target_mix=target_mix,
                    start_transition=int(completed_transitions),
                    transitions=self.mix_ramp_transitions,
                )
                self._apply_mix_ramp(completed_transitions)
            else:
                self.sampler.set_stage_mix(target_mix)
        self._write_state(
            completed_transitions,
            decision,
            completed_iterations=completed_iterations,
        )
        return decision

    def check_stage_budget(
        self,
        completed_transitions: int,
        completed_iterations: int | None,
    ) -> StageBudgetStatus | None:
        """Fail-closed per-iteration hard-cap check.

        Unlike ``maybe_update`` this never runs a frozen-dev evaluation, so the
        training loop can call it every iteration without changing the
        evaluation schedule.
        """

        if isinstance(completed_transitions, bool) or completed_transitions < 0:
            raise ValueError("completed_transitions must be a non-negative integer")
        if not self.promotion_enabled:
            return None
        if completed_iterations is None:
            return None
        status = self.controller.check_stage_budget(int(completed_iterations))
        if not status.exhausted:
            return None
        record = {
            "schema_version": "auv_pc_curriculum_stage_budget.v1",
            "decision": CurriculumDecision.STAGE_BUDGET_EXHAUSTED.value,
            "reason": "stage iteration hard-cap exhausted before promotion",
            "completed_transitions": int(completed_transitions),
            **status.as_dict(),
        }
        with self.budget_path.open("a") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._write_state(
            completed_transitions,
            CurriculumDecision.STAGE_BUDGET_EXHAUSTED,
            completed_iterations=int(completed_iterations),
        )
        return status

    def _maybe_update_enforce(
        self,
        completed_transitions: int,
        completed_iterations: int | None,
        route_status: Mapping[str, Any] | None,
    ) -> CurriculumDecision | None:
        """Use the training-rollout route window as the routine promotion gate."""

        if route_status is None or route_status.get("should_promote") is not True:
            self._write_state(
                completed_transitions,
                None,
                completed_iterations=completed_iterations,
            )
            return None

        previous_stage = self.controller.stage
        evaluation = self.evaluate(previous_stage, completed_transitions)
        spec = self.controller.stages[previous_stage]
        if evaluation.safety_rate > spec.gate.safety_max:
            self.routine_gate.reset()
            decision = self.controller.veto_after_route_gate(
                evaluation,
                completed_transitions=completed_transitions,
                completed_iterations=completed_iterations,
                route_gate=route_status,
                reason=(
                    "frozen-dev boundary safety veto: "
                    f"{evaluation.safety_rate:.6f} > {spec.gate.safety_max:.6f}"
                ),
            )
        else:
            decision = self.controller.promote_after_route_gate(
                evaluation,
                completed_transitions=completed_transitions,
                completed_iterations=completed_iterations,
                route_gate=route_status,
            )
            self.routine_gate.reset()

        if self.controller.stage != previous_stage:
            self._stage_entered_at = int(completed_transitions)
        record = self.controller.history[-1].as_dict()
        with self.decisions_path.open("a") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        if decision in {
            CurriculumDecision.PROMOTE,
            CurriculumDecision.ROLLBACK,
        }:
            target_mix = curriculum_stage_mix_target(
                self.controller.stage,
                preview_enabled=self.preview_enabled,
            )
            if self.mix_ramp_transitions > 0:
                self._mix_ramp = CurriculumMixRamp(
                    start_mix=self.sampler.stage_mix,
                    target_mix=target_mix,
                    start_transition=int(completed_transitions),
                    transitions=self.mix_ramp_transitions,
                )
                self._apply_mix_ramp(completed_transitions)
            else:
                self.sampler.set_stage_mix(target_mix)
        self._write_state(
            completed_transitions,
            decision,
            completed_iterations=completed_iterations,
        )
        return decision

    def finalize(self, completed_transitions: int) -> None:
        """Persist the final curriculum state without changing the active mix."""

        if isinstance(completed_transitions, bool) or completed_transitions < 0:
            raise ValueError("completed_transitions must be a non-negative integer")
        completed_iterations = (
            None
            if self._last_state_payload is None
            else self._last_state_payload.get("completed_iterations")
        )
        self._write_state(
            completed_transitions,
            None,
            completed_iterations=completed_iterations,
            finalized=True,
        )

    def current_encoder_alpha(self, completed_transitions: int) -> float:
        """Return the stage-aware actor current-encoder alpha."""

        if isinstance(completed_transitions, bool) or completed_transitions < 0:
            raise ValueError("completed_transitions must be a non-negative integer")
        if self.force_current_encoder_alpha_zero:
            return 0.0
        spec = self.controller.stages[self.controller.stage]
        if not spec.actor_current_encoder:
            return 0.0
        if spec.current_encoder_alpha_ramp_transitions <= 0:
            return float(spec.current_encoder_alpha_end)
        elapsed = max(0, int(completed_transitions) - int(self._stage_entered_at))
        fraction = min(elapsed / float(spec.current_encoder_alpha_ramp_transitions), 1.0)
        return float(
            spec.current_encoder_alpha_start
            + (spec.current_encoder_alpha_end - spec.current_encoder_alpha_start) * fraction
        )

    @property
    def stage(self) -> str:
        return self.controller.stage

    @property
    def stage_mix(self) -> tuple[tuple[str, float], ...]:
        return tuple(self.sampler.stage_mix)
