from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_goal156_standing_arms.py"
SPEC = importlib.util.spec_from_file_location("goal156_standing_arms", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def _touch_checkpoints(root: Path, seeds: tuple[int, ...]) -> None:
    for seed in seeds:
        path = root / "train" / f"full_seed{seed}" / "final.zip"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"checkpoint")


def test_build_arms_reuses_one_checkpoint_for_all_gates(tmp_path: Path) -> None:
    _touch_checkpoints(tmp_path, (86201,))
    arms = module.build_arms(
        tmp_path, seeds=(86201,), variants=("full", "near10_2", "mid_only")
    )
    by_tag = {arm["tag"]: arm for arm in arms}
    assert set(by_tag) == {"zero", "full_86201", "near10_2_86201", "mid_only_86201"}
    assert by_tag["zero"]["action_mode"] == "zero-residual"
    for tag in ("full_86201", "near10_2_86201", "mid_only_86201"):
        assert by_tag[tag]["checkpoint"] == str(
            (tmp_path / "train" / "full_seed86201" / "final.zip").resolve()
        )
    assert by_tag["full_86201"]["learner_terminal_gate_mode"] == "none"
    assert by_tag["mid_only_86201"]["learner_terminal_gate_mode"] == "shell"


def test_build_arms_rejects_missing_checkpoint(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="final.zip"):
        module.build_arms(tmp_path, seeds=(86201,), variants=("full",))


def test_zero_only_does_not_require_checkpoints(tmp_path: Path) -> None:
    arms = module.build_arms(tmp_path, zero_only=True)
    assert [arm["tag"] for arm in arms] == ["zero"]
    assert arms[0]["action_mode"] == "zero-residual"
