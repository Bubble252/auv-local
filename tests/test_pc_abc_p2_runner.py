from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_pc_abc_p2_20261105.py"


def _load_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("run_pc_abc_p2_20261105", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_p2_plans_use_fixed_seeds_budgets_and_history_ablations() -> None:
    module = _load_module()

    smoke = module.build_plan("smoke")
    screen = module.build_plan("screen")

    assert smoke.iterations == 32
    assert smoke.num_envs == 8
    assert smoke.num_steps == 16
    assert smoke.eval_seed_start == 0
    assert smoke.eval_seed_count == 8
    assert screen.iterations == 32
    assert screen.num_envs == 128
    assert screen.num_steps == 256
    assert screen.eval_seed_start == 1000
    assert screen.eval_seed_count == 20
    assert smoke.tags == ("default", "strict125")
    assert smoke.history_ablations == ("none", "history-zero")
    assert screen.history_ablations == ("none", "history-zero")


def test_p2_smoke_and_screen_write_separate_manifests_and_reports(tmp_path: Path) -> None:
    module = _load_module()
    smoke = module.build_plan("smoke")
    screen = module.build_plan("screen")

    assert module.run_manifest_path(tmp_path, smoke) == tmp_path / "smoke/run_manifest.json"
    assert module.report_root(tmp_path, smoke) == tmp_path / "smoke/report"
    assert module.run_manifest_path(tmp_path, screen) == tmp_path / "run_manifest.json"
    assert module.report_root(tmp_path, screen) == tmp_path / "report"
    assert module.stage_root(tmp_path, smoke).name == "P2b_encoded"
    assert module.stage_root(tmp_path, screen).name == "P2b_encoded_screen"


def test_p2_eval_command_uses_history_ablation_without_path_ablation() -> None:
    module = _load_module()
    plan = module.build_plan("screen")
    job = module.EvalJob(
        tag="default",
        history_ablation="history-zero",
        checkpoint=Path("/tmp/final.pt"),
        output=Path("/tmp/eval"),
        environment_config=Path("/tmp/environment.json"),
    )

    command = module.build_eval_command("python", job=job, plan=plan, device="cpu")

    assert command[command.index("--history-ablation") + 1] == "history-zero"
    assert "--path-ablation" not in command


def test_p2_configs_are_encoded_history_with_zero_path_dimension() -> None:
    module = _load_module()

    module._validate_configs()


def test_p2_training_audit_accepts_encoded_history_contract(tmp_path: Path) -> None:
    module = _load_module()
    from auv_lowlevel.batch.pc_policy import pc_schema_hash

    plan = module.build_plan("screen")
    output = module.training_output(tmp_path, plan, plan.train_seed)
    output.mkdir(parents=True)
    bc = tmp_path / "bc.pt"
    bc.write_bytes(b"bc-checkpoint")
    final = output / "final.pt"
    final.write_bytes(b"final-checkpoint")
    manifest = {
        "status": "completed",
        "seed": plan.train_seed,
        "num_envs": plan.num_envs,
        "num_steps_per_env": plan.num_steps,
        "iterations": plan.iterations,
        "requested_total_env_steps": plan.num_envs * plan.num_steps * plan.iterations,
        "total_env_steps": plan.num_envs * plan.num_steps * plan.iterations,
        "warmstart_actor_sha256": module._sha256(bc),
        "final_checkpoint_sha256": module._sha256(final),
        "history": module.EXPECTED_HISTORY,
        "final_checkpoint_metadata": {
            "pc": {
                "schema_hash": pc_schema_hash(path_dim=0, dade_history_mode="encoded"),
                "dade_enabled": True,
                "dade_history_mode": "encoded",
                "dade_state_mode": "full",
                "actor_dade_observation_dim": 67,
                "path": {"dim": 0, "blocks": [], "schema_hash": None},
            }
        },
    }
    (output / "training_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    (output / "iteration_records.json").write_text(
        json.dumps(
            {
                "records": [
                    {"iteration": index + 1, "losses_finite": True}
                    for index in range(plan.iterations)
                ]
            }
        )
        + "\n",
        encoding="utf-8",
    )

    audit = module._audit_training(tmp_path, plan, bc_checkpoint=bc)

    assert audit["passed"] is True
    assert audit["failures"] == []
    assert audit["pc_metadata"]["actor_dade_observation_dim"] == 67
