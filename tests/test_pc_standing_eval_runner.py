from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_pc_standing_eval_suite.py"
SPEC = importlib.util.spec_from_file_location("standing_eval_runner", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def test_resolve_keeps_absolute_path_and_uses_repo_root(tmp_path: Path) -> None:
    absolute = tmp_path / "run.json"

    assert runner._resolve(absolute) == absolute
    assert runner._resolve("configs/run.json") == runner.REPO_ROOT / "configs/run.json"


def _manifest(tmp_path: Path) -> dict:
    scenario_a = tmp_path / "a.json"
    scenario_b = tmp_path / "b.json"
    scenario_a.write_text("{}\n", encoding="utf-8")
    scenario_b.write_text("{}\n", encoding="utf-8")
    return {
        "scenarios": [
            {"id": "a", "config": str(scenario_a)},
            {"id": "b", "config": str(scenario_b)},
        ]
    }


def _arms() -> list[dict]:
    return [
        {
            "tag": "zero",
            "paradigm": "BC-SAC",
            "evaluator": "sac",
            "training_config": "configs/zero.json",
            "action_mode": "zero-residual",
            "disable_curriculum_direct_command": True,
        },
        {
            "tag": "sac",
            "paradigm": "BC-SAC",
            "evaluator": "sac",
            "training_config": "configs/sac.json",
            "checkpoint": "artifacts/sac/final.zip",
            "action_mode": "policy",
            "residual_scale": 0.25,
            "learner_terminal_gate_mode": "none",
        },
        {
            "tag": "ppo",
            "paradigm": "BC-PPO",
            "evaluator": "ppo",
            "training_config": "configs/ppo.json",
            "checkpoint": "artifacts/ppo/final.pt",
            "action_mode": "policy",
            "curriculum_stage": "C1",
            "disable_curriculum_direct_command": True,
        },
        {
            "tag": "bc",
            "paradigm": "BC",
            "evaluator": "bc_residual",
            "training_config": "configs/bc.json",
            "checkpoint": "artifacts/bc/pc_bc_actor.pt",
            "action_mode": "policy",
            "extra_args": ["--path-zero"],
        },
    ]


def _jobs(tmp_path: Path, arms: list[dict] | None = None, scenarios: list[str] | None = None):
    return runner.build_jobs(
        manifest=_manifest(tmp_path),
        arms=_arms() if arms is None else arms,
        output_root=tmp_path / "eval",
        python="/python",
        seed_start=20,
        seed_count=40,
        device="cpu",
        dtype="float64",
        overwrite=True,
        scenarios=scenarios,
    )


def test_sac_zero_and_policy_commands_are_distinct(tmp_path: Path) -> None:
    jobs = _jobs(tmp_path, scenarios=["a"])
    assert len(jobs) == 4
    by_tag = {job["run_tag"]: job for job in jobs}
    zero = by_tag["zero"]["command"]
    assert zero[1].endswith("evaluate_pc_sac.py")
    assert "--checkpoint" not in zero
    assert zero[zero.index("--action-mode") + 1] == "zero-residual"
    assert "--disable-curriculum-direct-command" in zero

    sac = by_tag["sac"]["command"]
    assert sac[sac.index("--checkpoint") + 1].endswith("artifacts/sac/final.zip")
    assert sac[sac.index("--residual-scale") + 1] == "0.25"
    assert sac[sac.index("--trace-output") + 1] == by_tag["sac"]["output"]
    assert "--disable-curriculum-direct-command" in sac


def test_ppo_and_bc_keep_evaluator_specific_contracts(tmp_path: Path) -> None:
    jobs = _jobs(tmp_path, scenarios=["a"])
    by_tag = {job["run_tag"]: job for job in jobs}
    ppo = by_tag["ppo"]["command"]
    assert ppo[1].endswith("evaluate_pc_ppo.py")
    assert ppo[ppo.index("--curriculum-stage") + 1] == "C1"
    assert ppo[ppo.index("--trace-output") + 1].endswith("trace.npz")
    assert "--disable-curriculum-direct-command" in ppo

    bc = by_tag["bc"]["command"]
    assert bc[1].endswith("evaluate_pc_bc_residual.py")
    assert bc[bc.index("--config") + 1].endswith("configs/bc.json")
    assert "--disable-curriculum-direct-command" in bc
    assert bc[-1] == "--path-zero"


def test_scenario_filter_and_training_seed_tag(tmp_path: Path) -> None:
    arms = _arms()[:1]
    arms[0]["training_seed"] = 86201
    jobs = _jobs(tmp_path, arms=arms, scenarios=["b"])
    assert len(jobs) == 1
    assert jobs[0]["run_tag"] == "zero_s86201"
    assert jobs[0]["output"].endswith("zero_s86201/b")


def test_heldout_seed_window_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="held-out"):
        runner.build_jobs(
            manifest=_manifest(tmp_path),
            arms=_arms()[:1],
            output_root=tmp_path / "eval",
            python="/python",
            seed_start=90,
            seed_count=20,
            device="cpu",
            dtype="float64",
            overwrite=True,
        )


def test_arm_validation_rejects_policy_without_checkpoint(tmp_path: Path) -> None:
    path = tmp_path / "arms.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": runner.ARMS_SCHEMA,
                "arms": [
                    {
                        "tag": "bad",
                        "evaluator": "sac",
                        "training_config": "configs/x.json",
                        "action_mode": "policy",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="requires checkpoint"):
        runner.load_arms(path)


def test_zero_only_selection_skips_paired_analyzer() -> None:
    jobs = [{"arm_tag": "zero"}, {"arm_tag": "zero"}]
    assert runner.has_paired_arms(jobs, zero_tag="zero") is False
    jobs.append({"arm_tag": "mid_only_86201"})
    assert runner.has_paired_arms(jobs, zero_tag="zero") is True
