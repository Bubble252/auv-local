from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


def _load(name: str, relative: str):
    script = Path(__file__).resolve().parents[1] / relative
    spec = importlib.util.spec_from_file_location(name, script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


runner = _load("goal102_runner", "scripts/run_pc_goal102_ppo_dose_response.py")
audit = _load("goal102_analyzer", "scripts/analyze_pc_goal102_ppo_dose_response.py")


def test_build_jobs_covers_all_doses(tmp_path: Path) -> None:
    jobs = runner.build_jobs(tmp_path)
    assert len(jobs) == 20
    assert {job.dose for job in jobs} == set(runner.DOSES)
    assert {job.arm for job in jobs} == set(runner.ARMS)
    assert len({job.key for job in jobs}) == 20
    job = next(j for j in jobs if j.key == "P_BC15_strict125_D50")
    command = runner.command_for(
        job, python="/python", seed_start=20, seed_count=40, trace=True
    )
    assert command[command.index("--action-mode") + 1] == "policy"
    assert command[command.index("--actuator-residual-scale-override") + 1] == "2.00"
    assert command[command.index("--seed-start") + 1] == "20"
    assert command[command.index("--seed-count") + 1] == "40"
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
    dose: str | None,
    action_mode: str,
    successes: int,
    path_rmse: float = 2.75,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "episodes": _episodes(successes),
        "checkpoint_sha256": audit.EXPECTED_CHECKPOINT_SHA[(arm, dose or "D100")],
        "action_mode": action_mode,
        "actuator_residual_scale_override": 2.0,
        "ppo_residual_attenuation_enabled": False,
        "metrics": {
            "path_three_d_rmse_m": path_rmse,
            "goal_plane_miss_rate": 0.0,
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_all(
    root: Path,
    *,
    success: dict[tuple[str, str, str], int] | None = None,
) -> tuple[Path, Path, Path]:
    success = success or {}
    g100 = root / "g100"
    g101 = root / "g101"
    g102 = root / "g102"
    for arm in audit.ARMS:
        for scenario in audit.SCENARIOS:
            _write_summary(
                g100 / f"{arm}_{scenario}_B20" / "summary.json",
                arm=arm,
                dose=None,
                action_mode="frozen-base",
                successes=audit.EXPECTED_B20_SUCCESS[(arm, scenario)],
            )
            _write_summary(
                g101 / f"{arm}_{scenario}_ALOS" / "summary.json",
                arm=arm,
                dose=None,
                action_mode="alos",
                successes=audit.EXPECTED_ALOS_SUCCESS[(arm, scenario)],
            )
            for dose in audit.DOSE_IDS:
                key = (dose, arm, scenario)
                default = 20 if scenario == "strict125" else 33
                _write_summary(
                    g102 / f"{arm}_{scenario}_{dose}" / "summary.json",
                    arm=arm,
                    dose=dose,
                    action_mode="policy",
                    successes=success.get(key, default),
                )
    return g100, g101, g102


def _gate(root: Path, success=None) -> dict[str, object]:
    g100, g101, g102 = _write_all(root, success=success)
    b20, alos = audit._load_references(g100, g101)
    doses = audit._load_doses(g102)
    return audit._gate(doses, b20, alos)


def test_bounded_window_is_detected(tmp_path: Path) -> None:
    success = {}
    for arm in audit.ARMS:
        success[("D50", arm, "default")] = 33
        success[("D50", arm, "strict125")] = 21
    gate = _gate(tmp_path, success)
    assert gate["status"] == "BOUNDED_PPO_SUPPORTED"
    assert gate["both_arm_doses"] == ["D50"]


def test_single_arm_signal_is_detected(tmp_path: Path) -> None:
    success = {
        ("D25", "P_D3", "default"): 33,
        ("D25", "P_D3", "strict125"): 21,
    }
    gate = _gate(tmp_path, success)
    assert gate["status"] == "SINGLE_ARM_SIGNAL"
    assert gate["single_arm_doses"] == ["D25"]


def test_neutral_dose_is_detected(tmp_path: Path) -> None:
    gate = _gate(tmp_path)
    assert gate["status"] == "PPO_DOSE_NEUTRAL"
    assert gate["both_arm_doses"] == []


def test_harmful_dose_is_detected(tmp_path: Path) -> None:
    success = {}
    for dose in audit.DOSE_IDS:
        for arm in audit.ARMS:
            success[(dose, arm, "default")] = 31
            success[(dose, arm, "strict125")] = 18
    gate = _gate(tmp_path, success)
    assert gate["status"] == "PPO_DOSE_HARM"


def test_bad_checkpoint_sha_is_invalid(tmp_path: Path) -> None:
    g100, g101, g102 = _write_all(tmp_path)
    path = g102 / "P_D3_default_D50" / "summary.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["checkpoint_sha256"] = "0" * 64
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(audit.AuditError, match="checkpoint SHA"):
        audit._load_doses(g102)
