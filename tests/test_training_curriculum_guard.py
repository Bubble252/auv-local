from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import json

import pytest

from auv_lowlevel.config import load_config
from auv_lowlevel.curriculum import (
    CurriculumPromotionController,
    CurriculumStageSampler,
)
from auv_lowlevel.curriculum_trainer import CurriculumTrainerHook
from auv_lowlevel.experiment_contract import require_training_curriculum


ROOT = Path(__file__).resolve().parents[1]
GOAL173_TRAIN_ENV = ROOT / "configs" / "lowlevel" / "goal173" / "env_route_health_train_nu_ground.json"


def _config(*, enabled: bool, stage_mix: tuple[tuple[str, float], ...]) -> SimpleNamespace:
    return SimpleNamespace(
        curriculum=SimpleNamespace(
            enabled=enabled,
            stage_mix=stage_mix,
            manifest_filename="curriculum_episodes.jsonl",
        )
    )


def test_goal173_training_environment_passes_curriculum_guard() -> None:
    config = load_config(GOAL173_TRAIN_ENV)
    require_training_curriculum(config, source=GOAL173_TRAIN_ENV)
    assert config.curriculum.promotion_enabled is False


def test_training_without_curriculum_fails_closed() -> None:
    with pytest.raises(ValueError, match="curriculum.enabled=true"):
        require_training_curriculum(
            _config(enabled=False, stage_mix=(("C3", 1.0),)),
            source="bad-config.json",
        )


def test_training_with_empty_curriculum_mix_fails_closed() -> None:
    with pytest.raises(ValueError, match="non-empty curriculum.stage_mix"):
        require_training_curriculum(
            _config(enabled=True, stage_mix=()),
            source="bad-config.json",
        )


def test_frozen_curriculum_never_evaluates_or_promotes(tmp_path: Path) -> None:
    mix = (("C3", 0.70), ("C2", 0.15), ("C0", 0.10), ("C4", 0.05))
    sampler = CurriculumStageSampler(stage_mix=mix)
    evaluations: list[tuple[str, int]] = []
    hook = CurriculumTrainerHook(
        controller=CurriculumPromotionController(start_stage="C3"),
        sampler=sampler,
        output_dir=tmp_path,
        eval_interval_transitions=10,
        evaluate=lambda stage, transitions: evaluations.append((stage, transitions)),
        promotion_enabled=False,
    )
    assert hook.maybe_update(100) is None
    hook.finalize(100)
    state = json.loads((tmp_path / "curriculum_state.json").read_text())
    assert evaluations == []
    assert state["stage"] == "C3"
    assert state["frozen_stage"] == "C3"
    assert state["promotion_enabled"] is False
    assert state["finalized"] is True
    assert tuple(tuple(item) for item in state["stage_mix"]) == mix
