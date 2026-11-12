from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_pc_abc_prework_20261105.py"


def _load_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("run_pc_abc_prework", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_build_plan_matches_declared_smoke_and_screen_budgets() -> None:
    module = _load_module()

    smoke = module.build_plan("smoke")
    assert smoke.iterations == 1
    assert smoke.num_envs == 8
    assert smoke.num_steps == 16
    assert smoke.eval_seed_start == 0
    assert smoke.eval_seed_count == 8
    assert smoke.ablations == ("none",)

    screen = module.build_plan("screen")
    assert screen.iterations == 32
    assert screen.num_envs == 128
    assert screen.num_steps == 256
    assert screen.eval_seed_start == 1000
    assert screen.eval_seed_count == 20
    assert screen.ablations == ("none", "history-zero", "history-reverse")


def test_parse_args_accepts_isolated_formal_stage_name(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_module()
    monkeypatch.setattr(
        "sys.argv",
        [
            "runner",
            "--mode",
            "screen",
            "--stage-name",
            "A_screen",
        ],
    )
    args = module._parse_args()
    assert args.mode == "screen"
    assert args.stage_name == "A_screen"


def test_abc_configs_differ_only_in_declared_stride_and_identity() -> None:
    module = _load_module()
    payloads = {
        arm.name: json.loads((ROOT / arm.config).read_text(encoding="utf-8"))
        for arm in module.ARMS
    }

    def scrub(payload: dict, arm_name: str, stride: int) -> dict:
        copied = json.loads(json.dumps(payload))
        copied["experiment"]["name"] = "<name>"
        copied["experiment"]["output"] = "<output>"
        copied["history"]["short_stride"] = "<stride>"
        copied["history"]["long_stride"] = "<stride>"
        return copied

    baseline = scrub(payloads["A0_stride1"], "A0_stride1", 1)
    assert scrub(payloads["A1_stride5"], "A1_stride5", 5) == baseline
    assert scrub(payloads["A2_stride10"], "A2_stride10", 10) == baseline
    for arm in module.ARMS:
        payload = payloads[arm.name]
        assert payload["history"]["short_stride"] == arm.stride
        assert payload["history"]["long_stride"] == arm.stride
        assert payload["environment"]["config"].endswith(
            "odarrl_control_pc_c3_random_realistic_v1_rewardscale001.json"
        )
        assert payload["pc"]["dade_enabled"] is True
        assert payload["pc"]["d_z_total"] == 64
        assert payload["environment"]["num_envs"] == 128
        assert payload["training"]["iterations"] == 32
        assert payload["runner"]["num_steps_per_env"] == 256


def test_training_and_evaluation_commands_are_seed_paired(tmp_path: Path) -> None:
    module = _load_module()
    plan = module.build_plan("screen", train_seed=91430)
    arm = module.ARMS[1]
    run_root = tmp_path / "screen" / "P0_a_screen"
    train = module.build_train_command(
        "python",
        config=ROOT / arm.config,
        output=module.training_output(run_root, arm, plan.train_seed),
        warmstart=module.DEFAULT_BC,
        plan=plan,
        device="cuda",
    )

    assert train[:2] == ["python", "scripts/train_batch_lowlevel_ppo.py"]
    assert train[train.index("--iterations") + 1] == "32"
    assert train[train.index("--num-envs") + 1] == "128"
    assert train[train.index("--num-steps") + 1] == "256"
    assert train[train.index("--seed") + 1] == "91430"
    assert train[train.index("--warmstart-actor") + 1] == str(module.DEFAULT_BC)
    assert "--record-only" in train

    job = module.build_eval_job(
        root=ROOT,
        run_root=run_root,
        tag="strict125",
        ablation="history-zero",
        arm=arm,
        train_seed=plan.train_seed,
    )
    evaluate = module.build_eval_command(
        "python", root=ROOT, job=job, plan=plan, device="cpu"
    )
    assert evaluate[:2] == ["python", "scripts/evaluate_pc_ppo.py"]
    assert evaluate[evaluate.index("--checkpoint") + 1] == str(
        module.training_output(run_root, arm, plan.train_seed) / "final.pt"
    )
    assert evaluate[evaluate.index("--seed-start") + 1] == "1000"
    assert evaluate[evaluate.index("--seed-count") + 1] == "20"
    assert evaluate[evaluate.index("--history-ablation") + 1] == "history-zero"


def test_runner_refuses_to_overwrite_incomplete_outputs(tmp_path: Path) -> None:
    module = _load_module()
    plan = module.build_plan("smoke")
    arm = module.ARMS[0]
    run_root = tmp_path / "smoke" / "P0_a_screen"

    output = module.training_output(run_root, arm, plan.train_seed)
    output.mkdir(parents=True)
    (output / "partial.log").write_text("partial\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="refusing to overwrite"):
        module._prepare_training(run_root, arm, plan.train_seed)

    job = module.build_eval_job(
        root=ROOT,
        run_root=run_root,
        tag="default",
        ablation="none",
        arm=arm,
        train_seed=plan.train_seed,
    )
    job.output.mkdir(parents=True)
    (job.output / "partial.log").write_text("partial\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="refusing to overwrite"):
        module._prepare_eval(job)


def test_runner_resumes_only_completed_training(tmp_path: Path) -> None:
    module = _load_module()
    plan = module.build_plan("smoke")
    arm = module.ARMS[0]
    run_root = tmp_path / "smoke" / "P0_a_screen"
    output = module.training_output(run_root, arm, plan.train_seed)
    output.mkdir(parents=True)
    (output / "final.pt").write_bytes(b"checkpoint")
    (output / "training_manifest.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "final_checkpoint": str(output / "final.pt"),
            }
        ),
        encoding="utf-8",
    )

    assert module._prepare_training(run_root, arm, plan.train_seed) == "skip"


def test_cpu_binding_rejects_ranges_outside_host_cpus() -> None:
    module = _load_module()
    command = ["python", "scripts/evaluate_pc_ppo.py"]

    assert module._with_cpu_binding(
        command,
        device="cpu",
        cpu_base=0,
        slot=0,
        cpu_stride=1,
    )[:4] == ["taskset", "-c", "0-0", "nice"]
    with pytest.raises(ValueError, match="outside the available range"):
        module._with_cpu_binding(
            command,
            device="cpu",
            cpu_base=10**9,
            slot=0,
            cpu_stride=1,
        )
