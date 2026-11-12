from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np


def _load(name: str, relative: str):
    script = Path(__file__).resolve().parents[1] / relative
    spec = importlib.util.spec_from_file_location(name, script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


evaluate = _load("goal101_evaluate", "scripts/evaluate_pc_ppo.py")
runner = _load("goal101_runner", "scripts/run_pc_goal101_alos_pairing.py")
audit = _load("goal101_analyzer", "scripts/analyze_pc_goal101_alos_pairing.py")


def test_alos_bypasses_frozen_residual_wrapper() -> None:
    assert not evaluate._uses_frozen_residual_wrapper(
        base_mode="frozen_pc_bc_residual",
        base_output_mode="residual",
        action_mode="alos",
    )
    for mode in ("frozen-base", "policy", "gated-residual", "attenuated-residual"):
        assert evaluate._uses_frozen_residual_wrapper(
            base_mode="frozen_pc_bc_residual",
            base_output_mode="residual",
            action_mode=mode,
        )


def test_alos_action_mode_parses_without_attenuation_flags() -> None:
    args = evaluate._parse_args(
        [
            "--training-config",
            "training.json",
            "--checkpoint",
            "final.pt",
            "--output",
            "out",
            "--action-mode",
            "alos",
        ]
    )
    assert args.action_mode == "alos"
    assert args.ppo_residual_attenuation_full_m == 1.25
    assert args.ppo_residual_attenuation_enter_m == 3.0


def test_build_jobs_covers_four_alos_runs(tmp_path: Path) -> None:
    jobs = runner.build_jobs(tmp_path)
    assert len(jobs) == 4
    assert {job.key for job in jobs} == {
        f"{arm}_{scenario}_ALOS"
        for arm in runner.ARMS
        for scenario in runner.SCENARIOS
    }
    job = next(j for j in jobs if j.key == "P_BC15_strict125_ALOS")
    command = runner.command_for(
        job, python="/python", seed_start=20, seed_count=40, trace=True
    )
    assert command[command.index("--action-mode") + 1] == "alos"
    assert command[command.index("--success-frame") + 1] == "both"
    assert command[command.index("--seed-start") + 1] == "20"
    assert command[command.index("--seed-count") + 1] == "40"
    assert command[command.index("--actuator-residual-scale-override") + 1] == "2.00"
    assert command[command.index("--trace-output") + 1].endswith("trace.npz")


def _episodes(successes: int) -> list[dict[str, object]]:
    return [
        {
            "seed": seed,
            "success": offset < successes,
            "success_pnav_tight": offset < successes,
            "success_eta_tight_ever": offset < successes,
            "success_eta_tight_at_done": offset < successes,
            "timeout": False,
            "safety": False,
        }
        for offset, seed in enumerate(range(20, 60))
    ]


def _write_summary(
    path: Path,
    *,
    arm: str,
    action_mode: str,
    successes: int,
    path_rmse: float = 2.70,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "episodes": _episodes(successes),
        "checkpoint_sha256": audit.EXPECTED_CHECKPOINT_SHA[arm],
        "action_mode": action_mode,
        "actuator_residual_scale_override": 2.0,
        "ppo_residual_attenuation_enabled": False,
        "metrics": {
            "path_three_d_rmse_m": path_rmse,
            "goal_plane_miss_rate": 0.0,
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_alos_runner_validation_does_not_require_attenuation_trace(tmp_path: Path) -> None:
    job = next(
        job
        for job in runner.build_jobs(tmp_path)
        if job.key == "P_D3_default_ALOS"
    )
    job.output.mkdir(parents=True, exist_ok=True)
    _write_summary(
        job.output / "summary.json",
        arm=job.arm,
        action_mode="alos",
        successes=20,
    )
    np.savez(job.output / "trace.npz", episode_seed=np.arange(20, 60))
    summary = json.loads((job.output / "summary.json").read_text(encoding="utf-8"))
    runner.validate_summary(job, summary, trace=True)


def _write_pair(
    root: Path,
    *,
    alos_success: dict[tuple[str, str], int],
    b20_success: dict[tuple[str, str], int],
    b20_path: float = 2.70,
    alos_path: float = 2.70,
) -> tuple[Path, Path]:
    alos_root = root / "alos"
    b20_root = root / "b20"
    for arm in audit.ARMS:
        for scenario in audit.SCENARIOS:
            key = (arm, scenario)
            _write_summary(
                alos_root / f"{arm}_{scenario}_ALOS" / "summary.json",
                arm=arm,
                action_mode="alos",
                successes=alos_success[key],
                path_rmse=alos_path,
            )
            _write_summary(
                b20_root / f"{arm}_{scenario}_B20" / "summary.json",
                arm=arm,
                action_mode="frozen-base",
                successes=b20_success[key],
                path_rmse=b20_path,
            )
    return alos_root, b20_root


def test_gate_detects_b20_beating_alos(tmp_path: Path) -> None:
    alos_success = {
        ("P_D3", "default"): 20,
        ("P_BC15", "default"): 20,
        ("P_D3", "strict125"): 10,
        ("P_BC15", "strict125"): 10,
    }
    alos_root, b20_root = _write_pair(
        tmp_path,
        alos_success=alos_success,
        b20_success=audit.EXPECTED_B20_SUCCESS,
    )
    alos, b20 = audit._load_records(alos_root=alos_root, goal100_root=b20_root)
    assert audit._arm_invariance(alos)["identical"] is True
    gate = audit._gate(alos, b20)
    assert gate["status"] == "BC_PRIOR_BEATS_ALOS"


def test_gate_detects_b20_below_alos(tmp_path: Path) -> None:
    alos_success = {
        ("P_D3", "default"): 36,
        ("P_BC15", "default"): 36,
        ("P_D3", "strict125"): 23,
        ("P_BC15", "strict125"): 23,
    }
    alos_root, b20_root = _write_pair(
        tmp_path,
        alos_success=alos_success,
        b20_success=audit.EXPECTED_B20_SUCCESS,
    )
    alos, b20 = audit._load_records(alos_root=alos_root, goal100_root=b20_root)
    gate = audit._gate(alos, b20)
    assert gate["status"] == "BC_PRIOR_BELOW_ALOS"


def test_gate_detects_parity(tmp_path: Path) -> None:
    alos_success = {
        ("P_D3", "default"): 33,
        ("P_BC15", "default"): 33,
        ("P_D3", "strict125"): 20,
        ("P_BC15", "strict125"): 20,
    }
    alos_root, b20_root = _write_pair(
        tmp_path,
        alos_success=alos_success,
        b20_success=audit.EXPECTED_B20_SUCCESS,
    )
    alos, b20 = audit._load_records(alos_root=alos_root, goal100_root=b20_root)
    gate = audit._gate(alos, b20)
    assert gate["status"] == "BC_PRIOR_PARITY"


def test_b20_reproduction_failure_is_invalid(tmp_path: Path) -> None:
    alos_success = {
        ("P_D3", "default"): 20,
        ("P_BC15", "default"): 20,
        ("P_D3", "strict125"): 10,
        ("P_BC15", "strict125"): 10,
    }
    b20_success = dict(audit.EXPECTED_B20_SUCCESS)
    b20_success[("P_D3", "default")] = 32
    alos_root, b20_root = _write_pair(
        tmp_path,
        alos_success=alos_success,
        b20_success=b20_success,
    )
    alos, b20 = audit._load_records(alos_root=alos_root, goal100_root=b20_root)
    gate = audit._gate(alos, b20)
    assert gate["status"] == "INVALID"


def test_alos_arm_mismatch_is_detected(tmp_path: Path) -> None:
    alos_success = {
        ("P_D3", "default"): 20,
        ("P_BC15", "default"): 21,
        ("P_D3", "strict125"): 10,
        ("P_BC15", "strict125"): 10,
    }
    alos_root, b20_root = _write_pair(
        tmp_path,
        alos_success=alos_success,
        b20_success=audit.EXPECTED_B20_SUCCESS,
    )
    alos, _ = audit._load_records(alos_root=alos_root, goal100_root=b20_root)
    invariance = audit._arm_invariance(alos)
    assert invariance["identical"] is False
    assert invariance["groups"]["default"]["success_by_seed_equal"] is False
