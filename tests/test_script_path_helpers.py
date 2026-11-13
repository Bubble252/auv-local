from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

from auv_lowlevel._paths import resolve_from_root

REPO_ROOT = Path(__file__).resolve().parents[1]
P1A_SCRIPTS = (
    "audit_goal86_rl_backends.py",
    "audit_pc_goal45_fixed_calibration_residual.py",
    "audit_pc_goal46_dynamic_residual_association.py",
    "audit_pc_goal47_estimator_state.py",
    "audit_pc_goal48_strict125_sigma_residual.py",
    "audit_pc_goal49_controller_feedback.py",
    "audit_pc_goal50_guidance_timing.py",
    "audit_pc_goal51_unrecorded_state.py",
    "audit_pc_goal52_sigma_outcome_relevance.py",
    "audit_pc_goal53_recorded_feature_outcome.py",
    "audit_pc_goal57_heading_error_replication.py",
    "audit_pc_goal86_pa_residual_labels.py",
    "audit_pc_terminal_contract.py",
    "build_pc_goal95_residual_supervision.py",
    "build_pc_standing_eval_configs.py",
    "diagnose_pc_dade_history.py",
    "evaluate_pc_ppo.py",
    "evaluate_pc_sac.py",
    "pretrain_odarrl_control_residual.py",
    "pretrain_pc_goal86_pa_residual.py",
)
P1B_SCRIPTS = (
    "audit_goal86_bc_interfaces.py",
    "train_pc_command_residual_sac.py",
)


def _load_script(script_name: str) -> ModuleType:
    path = REPO_ROOT / "scripts" / script_name
    spec = importlib.util.spec_from_file_location(f"path_probe_{path.stem}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_resolve_from_root_preserves_absolute_relative_missing_and_symlink(tmp_path: Path) -> None:
    absolute = tmp_path / "absolute" / "run.json"
    missing = tmp_path / "missing" / "run.json"
    (tmp_path / "target").mkdir()
    (tmp_path / "link").symlink_to(tmp_path / "target", target_is_directory=True)

    assert resolve_from_root(tmp_path, absolute) == absolute
    assert resolve_from_root(tmp_path, "configs/run.json") == tmp_path / "configs/run.json"
    assert resolve_from_root(tmp_path, missing) == missing
    linked = resolve_from_root(tmp_path, "link/run.json")
    assert linked == tmp_path / "link/run.json"
    assert linked.resolve() == (tmp_path / "target" / "run.json").resolve()


@pytest.mark.parametrize("script_name", P1A_SCRIPTS + P1B_SCRIPTS)
def test_p1_resolver_wrappers_keep_repo_root(script_name: str) -> None:
    module = _load_script(script_name)

    absolute = module.REPO_ROOT / "scripts" / "run.json"
    assert module._resolve(absolute) == absolute
    assert module._resolve(Path("configs/run.json")) == module.REPO_ROOT / "configs/run.json"
