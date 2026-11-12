from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch


def _load(name: str, relative: str):
    script = Path(__file__).resolve().parents[1] / relative
    spec = importlib.util.spec_from_file_location(name, script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


evaluate = _load("goal100_evaluate", "scripts/evaluate_pc_ppo.py")
runner = _load("goal100_runner", "scripts/run_pc_goal100_terminal_prior.py")
audit = _load("goal100_analyzer", "scripts/analyze_pc_goal100_terminal_prior.py")


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
    attenuation: tuple[float, float] | None = None,
    scale_override: float | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, object] = {
        "episodes": _episodes(successes),
        "checkpoint_sha256": audit.EXPECTED_CHECKPOINT_SHA[arm],
        "action_mode": action_mode,
        "actuator_residual_scale_override": scale_override,
        "ppo_residual_attenuation_enabled": attenuation is not None,
        "ppo_residual_attenuation_full_m": None if attenuation is None else attenuation[0],
        "ppo_residual_attenuation_enter_m": None if attenuation is None else attenuation[1],
        "metrics": {
            "path_three_d_rmse_m": 2.70,
            "goal_plane_miss_rate": 0.0,
            "actuator_residual_magnitude_clip_rate": 0.1,
            "actuator_residual_rate_clip_rate": 0.1,
            "ppo_residual_attenuation_steps": 5 if attenuation is not None else 0,
            "ppo_residual_attenuation_mean": 0.5 if attenuation is not None else None,
            "ppo_residual_attenuation_zero_fraction": 0.4 if attenuation is not None else None,
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_attenuation_trace(path: Path, full_m: float, enter_m: float) -> None:
    distance = np.array([0.5, full_m, (full_m + enter_m) / 2.0, enter_m, 10.0])
    attenuation = 1.0 - audit._smooth_terminal_gate(
        distance, enter_m=enter_m, full_m=full_m
    )
    np.savez(
        path,
        step=np.arange(distance.size, dtype=np.int64),
        episode_seed=np.full(distance.size, 20, dtype=np.int64),
        goal_distance_eta_m=distance,
        ppo_residual_attenuation=attenuation,
        ppo_residual_attenuation_distance_m=distance,
    )


def _write_references(goal95: Path, goal98: Path) -> None:
    for arm in audit.ARMS:
        for scenario in audit.SCENARIOS:
            for profile, mode in (("P0_15", "policy"), ("B15", "frozen-base")):
                _write_summary(
                    goal95 / f"{arm}_s86201_{scenario}_{mode}" / "summary.json",
                    arm=arm,
                    action_mode=mode,
                    successes=audit.EXPECTED_REFERENCE_SUCCESS[
                        (profile, arm, scenario)
                    ],
                )
            _write_summary(
                goal98 / f"{arm}_{scenario}_s200" / "summary.json",
                arm=arm,
                action_mode="policy",
                successes=audit.EXPECTED_REFERENCE_SUCCESS[
                    ("P0_20", arm, scenario)
                ],
                scale_override=2.0,
            )


def _write_goal100(
    root: Path,
    *,
    successes: dict[str, int] | None = None,
) -> None:
    successes = successes or {}
    for arm in audit.ARMS:
        for scenario in audit.SCENARIOS:
            target = 31 if scenario == "default" else 20
            for profile, spec in audit.PROFILES.items():
                key = f"{arm}_{scenario}_{profile}"
                run = root / key
                attenuation = None
                if profile != "B20":
                    attenuation = (
                        float(spec["attenuation_full_m"]),
                        float(spec["attenuation_enter_m"]),
                    )
                _write_summary(
                    run / "summary.json",
                    arm=arm,
                    action_mode=str(spec["action_mode"]),
                    successes=successes.get(key, target),
                    attenuation=attenuation,
                    scale_override=2.0,
                )
                if attenuation is not None:
                    _write_attenuation_trace(run / "trace.npz", *attenuation)
                else:
                    np.savez(run / "trace.npz")


def test_attenuation_is_zero_near_and_one_far() -> None:
    distance = torch.tensor([0.5, 1.25, 2.125, 3.0, 10.0])
    attenuation = evaluate._ppo_residual_attenuation(
        distance, full_m=1.25, enter_m=3.0
    )
    assert attenuation[0].item() == pytest.approx(0.0)
    assert attenuation[1].item() == pytest.approx(0.0)
    assert 0.0 < attenuation[2].item() < 1.0
    assert attenuation[3].item() == pytest.approx(1.0)
    assert attenuation[4].item() == pytest.approx(1.0)
    assert torch.all(attenuation[1:] >= attenuation[:-1])


def test_build_jobs_covers_all_profiles(tmp_path: Path) -> None:
    jobs = runner.build_jobs(tmp_path)
    assert len(jobs) == 16
    assert {job.profile for job in jobs} == {"B20", "A3", "A2", "A125"}
    assert len({job.key for job in jobs}) == 16
    job = next(j for j in jobs if j.key == "P_BC15_strict125_A3")
    command = runner.command_for(
        job, python="/python", seed_start=20, seed_count=40, trace=True
    )
    assert command[command.index("--action-mode") + 1] == "attenuated-residual"
    assert command[command.index("--actuator-residual-scale-override") + 1] == "2.00"
    assert command[command.index("--ppo-residual-attenuation-full-m") + 1] == "3.00"
    assert command[command.index("--ppo-residual-attenuation-enter-m") + 1] == "5.00"


def test_bc_scale_only_is_detected(tmp_path: Path) -> None:
    goal95, goal98, goal100 = tmp_path / "g95", tmp_path / "g98", tmp_path / "g100"
    _write_references(goal95, goal98)
    _write_goal100(goal100)
    payload = audit._gate(
        goal100, goal95, goal98, expected_seeds=set(range(20, 60))
    )
    assert payload["status"] == "BC_SCALE_ONLY_SUPPORTED"
    assert payload["b20_qualifies"] is True


def test_attenuation_is_detected_when_b20_fails(tmp_path: Path) -> None:
    goal95, goal98, goal100 = tmp_path / "g95", tmp_path / "g98", tmp_path / "g100"
    _write_references(goal95, goal98)
    successes = {
        f"{arm}_strict125_B20": 18 for arm in audit.ARMS
    }
    _write_goal100(goal100, successes=successes)
    payload = audit._gate(
        goal100, goal95, goal98, expected_seeds=set(range(20, 60))
    )
    assert payload["status"] == "ATTENUATION_SUPPORTED"
    assert payload["qualifying_attenuation_profiles"] == ["A3", "A2", "A125"]


def test_bad_attenuation_trace_is_invalid(tmp_path: Path) -> None:
    goal95, goal98, goal100 = tmp_path / "g95", tmp_path / "g98", tmp_path / "g100"
    _write_references(goal95, goal98)
    _write_goal100(goal100)
    run = goal100 / "P_D3_default_A3"
    _write_attenuation_trace(run / "trace.npz", 3.0, 5.0)
    with np.load(run / "trace.npz") as data:
        payload = dict(data)
    payload["ppo_residual_attenuation"] = np.zeros_like(
        payload["ppo_residual_attenuation"]
    )
    np.savez(run / "trace.npz", **payload)
    with pytest.raises(audit.AuditError, match="attenuation does not match"):
        audit._gate(goal100, goal95, goal98, expected_seeds=set(range(20, 60)))
