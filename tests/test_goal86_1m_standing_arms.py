from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/build_goal86_1m_standing_arms.py"
SPEC = importlib.util.spec_from_file_location("goal86_standing_arms", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_goal86_checkpoint_layout_and_arm_count(tmp_path: Path) -> None:
    paths = {
        86201: tmp_path / "t2_sac/final.zip",
        86202: tmp_path / "t2_sac_seed86202/final.zip",
        86203: tmp_path / "t2_sac_seed86203/final.zip",
    }
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"checkpoint")
    arms = MODULE.build_arms(tmp_path)
    assert len(arms) == 13
    assert {arm["tag"] for arm in arms} == {
        "zero",
        *{f"{variant}_{seed}" for variant in MODULE.VARIANTS for seed in MODULE.SEEDS},
    }
    checkpoint_arms = [arm for arm in arms if arm["tag"] != "zero"]
    assert all(Path(arm["checkpoint"]).is_file() for arm in checkpoint_arms)
    assert all(arm["evaluator"] == "sac" for arm in arms)


def test_goal86_arm_builder_rejects_missing_checkpoint(tmp_path: Path) -> None:
    try:
        MODULE.build_arms(tmp_path)
    except FileNotFoundError as exc:
        assert "final.zip" in str(exc)
    else:
        raise AssertionError("missing checkpoint must fail")
