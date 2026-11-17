from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest

from auv_lowlevel.eval_pairing import BASELINE, PAIRED, UNPAIRED, audit_paired_zero

ROOT = Path(__file__).resolve().parents[1]
BUILDER = ROOT / "scripts" / "build_goal182_eval_arms.py"
GOAL = ROOT / "configs" / "lowlevel" / "goal182"


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("build_goal182_eval_arms", BUILDER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {BUILDER}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fake_run(root: Path, name: str, *, seed: int) -> Path:
    run_dir = root / "formal" / name
    run_dir.mkdir(parents=True)
    final = run_dir / "final.pt"
    final.write_bytes(f"{name}-final".encode())
    initial = run_dir / "model_initial.pt"
    initial.write_bytes(f"{name}-initial".encode())
    control = "CC_PE2E" if name.startswith("pe2e") else "CC_ALOS_BODY"
    manifest = {
        "status": "completed",
        "curriculum_final_stage": "C6",
        "final_checkpoint_sha256": _sha256(final),
        "initial_checkpoint_sha256": _sha256(initial),
        "control_contract": control,
    }
    if "residual" in name:
        stem = name.rsplit("_seed", 1)[0]
        config_path = GOAL / f"train_formal_mod4_{stem}_5d_m_seed{seed}.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        manifest["frozen_bc_checkpoint_sha256"] = config["base_policy"]["checkpoint_sha256"]
    (run_dir / "training_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return run_dir


def _build_payload(tmp_path: Path) -> dict:
    module = _load()
    stems = (
        "pe2e_finetune",
        "pe2e_residual",
        "alosb_finetune",
        "alosb_residual",
        "alosb_trres_cmd",
        "alosb_trres_act",
    )
    for stem in stems:
        _fake_run(tmp_path, f"{stem}_seed86201", seed=86201)
    return module.build_arms(run_root=tmp_path, seeds=(86201,))


def test_goal182_arms_have_paired_alosb_zeros_and_unpaired_pe2e(tmp_path: Path) -> None:
    payload = _build_payload(tmp_path)
    assert payload["policy_arms"] == 6
    assert payload["zero_arms"] == 4
    arms = payload["arms"]
    statuses, errors = audit_paired_zero(arms)
    assert errors == []
    assert statuses["pe2e_finetune_seed86201"] == UNPAIRED
    assert statuses["pe2e_residual_seed86201"] == UNPAIRED
    assert statuses["alosb_finetune_seed86201"] == PAIRED
    assert statuses["alosb_residual_seed86201"] == PAIRED
    assert statuses["alosb_trres_cmd_seed86201"] == PAIRED
    assert statuses["alosb_trres_act_seed86201"] == PAIRED
    assert all(
        statuses[zero] == BASELINE
        for zero in (
            "zero_bcft",
            "zero_bcres",
            "zero_trres_cmd",
            "zero_trres_act",
        )
    )
    by_tag = {arm["tag"]: arm for arm in arms}
    assert "paired_zero_tag" not in by_tag["pe2e_finetune_seed86201"]
    assert by_tag["pe2e_finetune_seed86201"]["zero_baseline_status"] == UNPAIRED
    assert by_tag["alosb_finetune_seed86201"]["paired_zero_tag"] == "zero_bcft"
    assert by_tag["zero_bcft"]["checkpoint"].endswith("model_initial.pt")
    assert by_tag["zero_bcft"]["action_mode"] == "policy"
    assert by_tag["zero_bcres"]["checkpoint"].endswith("final.pt")
    assert by_tag["zero_bcres"]["action_mode"] == "frozen-base"
    assert by_tag["zero_trres_cmd"]["action_mode"] == "alos"
    assert by_tag["zero_trres_act"]["action_mode"] == "alos"


def test_goal182_arms_fail_when_bcft_zero_update_checkpoint_is_missing(tmp_path: Path) -> None:
    payload = _build_payload(tmp_path)
    assert payload["policy_arms"] == 6
    (tmp_path / "formal" / "alosb_finetune_seed86201" / "model_initial.pt").unlink()
    with pytest.raises(FileNotFoundError, match="model_initial.pt"):
        _load().build_arms(run_root=tmp_path, seeds=(86201,))


def test_goal182_arms_reject_incomplete_training_run(tmp_path: Path) -> None:
    _build_payload(tmp_path)
    manifest = tmp_path / "formal" / "pe2e_finetune_seed86201" / "training_manifest.json"
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["status"] = "running"
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="status must be completed"):
        _load().build_arms(run_root=tmp_path, seeds=(86201,))
