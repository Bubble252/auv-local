from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import audit_goal173_artifacts as goal173_audit  # noqa: E402


FROZEN_MIX = (("C3", 0.70), ("C2", 0.15), ("C0", 0.10), ("C4", 0.05))


def test_goal173_audit_rejects_autopromoted_curriculum_state(tmp_path: Path) -> None:
    audit = goal173_audit._load_audit()
    state_path = tmp_path / "curriculum_state.json"
    state_path.write_text(
        json.dumps(
            {
                "completed_transitions": 1_024_000,
                "stage": "C5",
                "stage_mix": [
                    ["C5", 0.70],
                    ["C4", 0.15],
                    ["C0", 0.10],
                    ["C6", 0.05],
                ],
                "promotion_enabled": True,
                "frozen_stage": None,
                "finalized": True,
            }
        ),
        encoding="utf-8",
    )
    errors: list[str] = []
    audit._check_curriculum_state(
        state_path,
        initial_stage="C3",
        stage_mix=FROZEN_MIX,
        expected_promotion_enabled=False,
        errors=errors,
    )
    assert any("promotion is not disabled" in error for error in errors)
    assert any("runtime curriculum stage is 'C5'" in error for error in errors)
    assert any("runtime stage_mix drifted" in error for error in errors)
