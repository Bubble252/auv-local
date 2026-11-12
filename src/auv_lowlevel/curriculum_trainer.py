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
    StageEvaluation,
    StageGate,
    curriculum_stage_mix_target,
)


def parse_eval_summary(
    summary: Mapping[str, Any],
    *,
    require_stratified: bool = False,
) -> StageEvaluation:
    """Convert one frozen evaluation summary into a stage evaluation."""

    episodes = summary.get("episodes")
    if not isinstance(episodes, list) or not episodes:
        raise ValueError("evaluation summary must contain a non-empty episodes list")

    def rate(name: str) -> float:
        values = [bool(row.get(name, False)) for row in episodes]
        return float(sum(values)) / float(len(values))

    default_success = rate("success")
    strict_success = rate("success_eta_tight_at_done")
    timeout_rate = rate("timeout")
    if "goal_plane_miss" in episodes[0]:
        plane_miss_rate = rate("goal_plane_miss")
    else:
        plane_miss_rate = rate("plane_miss")
    safety_rate = rate("safety")
    stratified_default_success: dict[str, float] | None = None
    if any(isinstance(row.get("curriculum"), Mapping) for row in episodes):
        groups: dict[str, list[bool]] = {}
        for index, row in enumerate(episodes):
            metadata = row.get("curriculum")
            stratum = metadata.get("stratum") if isinstance(metadata, Mapping) else None
            if not isinstance(stratum, str) or not stratum:
                if require_stratified:
                    raise ValueError(
                        "evaluation episode "
                        f"{index} is missing curriculum.stratum; refusing to "
                        "evaluate a stratified Gate with unstratified data"
                    )
                groups = {}
                break
            groups.setdefault(stratum, []).append(bool(row.get("success", False)))
        if groups:
            stratified_default_success = {
                stratum: float(sum(values)) / float(len(values))
                for stratum, values in groups.items()
            }
    elif require_stratified:
        raise ValueError(
            "evaluation summary has no curriculum.stratum metadata; refusing to "
            "evaluate a stratified Gate with unstratified data"
        )
    return StageEvaluation(
        default_success=default_success,
        strict_success=strict_success,
        timeout_rate=timeout_rate,
        plane_miss_rate=plane_miss_rate,
        safety_rate=safety_rate,
        stratified_default_success=stratified_default_success,
    )


def combine_eval_summaries(
    default_summary: Mapping[str, Any],
    strict_summary: Mapping[str, Any],
    *,
    gate: StageGate | None = None,
) -> StageEvaluation:
    """Combine default-radius and strict-radius summaries into one evaluation."""

    require_stratified = (
        gate is not None and gate.stratified_default_success_min is not None
    )
    default_eval = parse_eval_summary(
        default_summary,
        require_stratified=require_stratified,
    )
    strict_eval = parse_eval_summary(strict_summary)
    return StageEvaluation(
        default_success=default_eval.default_success,
        strict_success=strict_eval.default_success,
        timeout_rate=max(default_eval.timeout_rate, strict_eval.timeout_rate),
        plane_miss_rate=max(default_eval.plane_miss_rate, strict_eval.plane_miss_rate),
        safety_rate=max(default_eval.safety_rate, strict_eval.safety_rate),
        stratified_default_success=default_eval.stratified_default_success,
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
        self.force_current_encoder_alpha_zero = bool(
            force_current_encoder_alpha_zero
        )
        self.promotion_enabled = bool(promotion_enabled)
        self._next_eval_transition = int(eval_interval_transitions)
        self._mix_ramp: CurriculumMixRamp | None = None
        self._stage_entered_at = 0
        self.decisions_path = self.output_dir / "curriculum_decisions.jsonl"
        self.state_path = self.output_dir / "curriculum_state.json"
        self._write_state(0, None)

    def _write_state(
        self,
        completed_transitions: int,
        decision: CurriculumDecision | None,
        *,
        finalized: bool = False,
    ) -> None:
        payload = {
            "schema_version": "auv_pc_curriculum_trainer_state.v1",
            "completed_transitions": int(completed_transitions),
            "stage": self.controller.stage,
            "stage_mix": [[key, float(weight)] for key, weight in self.sampler.stage_mix],
            "promotion_enabled": self.promotion_enabled,
            "frozen_stage": None if self.promotion_enabled else self.controller.stage,
            "last_decision": None if decision is None else decision.value,
            "rollbacks_used": self.controller.rollbacks_used,
            "force_current_encoder_alpha_zero": self.force_current_encoder_alpha_zero,
            "finalized": bool(finalized),
        }
        self.state_path.write_text(json.dumps(payload, indent=1) + "\n")

    def _apply_mix_ramp(self, completed_transitions: int) -> None:
        if self._mix_ramp is None:
            return
        self.sampler.set_stage_mix(self._mix_ramp.mix_at(completed_transitions))
        if completed_transitions >= (
            self._mix_ramp.start_transition + self._mix_ramp.transitions
        ):
            self._mix_ramp = None

    def maybe_update(self, completed_transitions: int) -> CurriculumDecision | None:
        """Evaluate at the next scheduled boundary, then promote or roll back."""

        if isinstance(completed_transitions, bool) or completed_transitions < 0:
            raise ValueError("completed_transitions must be a non-negative integer")
        self._apply_mix_ramp(completed_transitions)
        if not self.promotion_enabled:
            if completed_transitions >= self._next_eval_transition:
                while completed_transitions >= self._next_eval_transition:
                    self._next_eval_transition += self.eval_interval_transitions
                self._write_state(completed_transitions, None)
            return None
        if completed_transitions < self._next_eval_transition:
            return None
        while completed_transitions >= self._next_eval_transition:
            self._next_eval_transition += self.eval_interval_transitions
        evaluation = self.evaluate(self.controller.stage, completed_transitions)
        previous_stage = self.controller.stage
        decision = self.controller.observe(
            evaluation,
            completed_transitions=completed_transitions,
        )
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
        self._write_state(completed_transitions, decision)
        return decision

    def finalize(self, completed_transitions: int) -> None:
        """Persist the final curriculum state without changing the active mix."""

        if isinstance(completed_transitions, bool) or completed_transitions < 0:
            raise ValueError("completed_transitions must be a non-negative integer")
        self._write_state(completed_transitions, None, finalized=True)

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
        fraction = min(
            elapsed / float(spec.current_encoder_alpha_ramp_transitions), 1.0
        )
        return float(
            spec.current_encoder_alpha_start
            + (spec.current_encoder_alpha_end - spec.current_encoder_alpha_start)
            * fraction
        )

    @property
    def stage(self) -> str:
        return self.controller.stage

    @property
    def stage_mix(self) -> tuple[tuple[str, float], ...]:
        return tuple(self.sampler.stage_mix)
