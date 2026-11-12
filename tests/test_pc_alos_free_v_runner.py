from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_pc_alos_free_v_20261105.py"


def _load_module() -> ModuleType:
    sys.path.insert(0, str(ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location("run_pc_alos_free_v", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_v_plan_matches_pre_registered_budgets() -> None:
    module = _load_module()

    smoke = module.build_plan("smoke")
    screen = module.build_plan("screen")

    assert (smoke.iterations, smoke.num_envs, smoke.num_steps) == (1, 8, 16)
    assert (smoke.eval_seed_start, smoke.eval_seed_count) == (0, 8)
    assert (screen.iterations, screen.num_envs, screen.num_steps) == (32, 128, 256)
    assert (screen.eval_seed_start, screen.eval_seed_count) == (1000, 20)
    assert smoke.tags == screen.tags == ("default", "strict125", "grid_hs")


@pytest.mark.parametrize("stage_key", ("V0", "V1"))
def test_v_config_contracts_are_valid(stage_key: str) -> None:
    module = _load_module()

    module.validate_config(module.STAGES[stage_key])


def _write_manifest(root: Path, *, status: str, passed: bool | None = None) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "run_manifest.json").write_text(
        json.dumps({"status": status}), encoding="utf-8"
    )
    if passed is not None:
        report = root / "report"
        report.mkdir(parents=True, exist_ok=True)
        (report / "summary.json").write_text(
            json.dumps({"decision": {"passed": passed}}), encoding="utf-8"
        )


def test_v1_is_blocked_without_completed_v0(tmp_path: Path) -> None:
    module = _load_module()

    with pytest.raises(RuntimeError, match="missing"):
        module.require_v0_passed(tmp_path, mode="screen")


def test_v1_is_blocked_when_v0_screen_fails(tmp_path: Path) -> None:
    module = _load_module()
    _write_manifest(tmp_path / "V0_fixed_speed", status="completed", passed=False)

    with pytest.raises(RuntimeError, match="did not pass"):
        module.require_v0_passed(tmp_path, mode="screen")


def test_v1_is_allowed_after_v0_screen_passes(tmp_path: Path) -> None:
    module = _load_module()
    _write_manifest(tmp_path / "V0_fixed_speed", status="completed", passed=True)

    module.require_v0_passed(tmp_path, mode="screen")


def test_v1_smoke_only_requires_completed_v0_smoke(tmp_path: Path) -> None:
    module = _load_module()
    _write_manifest(tmp_path / "smoke" / "V0_fixed_speed", status="completed")

    module.require_v0_passed(tmp_path, mode="smoke")
