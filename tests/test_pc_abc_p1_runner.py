from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_pc_abc_p1_20261105.py"


def _load_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("run_pc_abc_p1_20261105", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_p1_plans_use_fixed_evaluation_seeds_and_all_path_ablations() -> None:
    module = _load_module()

    smoke = module.build_plan("smoke")
    screen = module.build_plan("screen")

    assert smoke.eval_seed_start == 0
    assert smoke.eval_seed_count == 8
    assert screen.eval_seed_start == 1000
    assert screen.eval_seed_count == 20
    assert smoke.path_ablations == module.PATH_ABLATIONS
    assert screen.path_ablations == module.PATH_ABLATIONS


def test_p1_eval_command_passes_path_ablation() -> None:
    module = _load_module()
    plan = module.build_plan("screen")
    arm = module.ARMS["full"]
    job = module.EvalJob(
        tag="default",
        path_ablation="path-zero",
        arm=arm,
        stage_root=Path("/tmp/stage"),
        checkpoint=Path("/tmp/checkpoint.pt"),
        output=Path("/tmp/eval"),
        environment_config=Path("/tmp/environment.json"),
    )

    command = module.build_eval_command(
        "python",
        root=module.REPO_ROOT,
        job=job,
        plan=plan,
        device="cpu",
    )

    assert "--path-ablation" in command
    assert command[command.index("--path-ablation") + 1] == "path-zero"
    assert "--history-ablation" not in command


def test_p1_configs_finish_with_the_same_schema() -> None:
    module = _load_module()

    module._validate_configs()


def test_smoke_and_screen_write_separate_manifests_and_reports(tmp_path: Path) -> None:
    module = _load_module()
    smoke = module.build_plan("smoke")
    screen = module.build_plan("screen")

    assert module.run_manifest_path(tmp_path, smoke) == tmp_path / "smoke/run_manifest.json"
    assert module.report_root(tmp_path, smoke) == tmp_path / "smoke/report"
    assert module.run_manifest_path(tmp_path, screen) == tmp_path / "run_manifest.json"
    assert module.report_root(tmp_path, screen) == tmp_path / "report"


def test_smoke_and_screen_use_seed_matched_p0_baselines() -> None:
    module = _load_module()

    smoke = module.default_baseline_eval_root("smoke")
    screen = module.default_baseline_eval_root("screen")

    assert smoke.name == "eval"
    assert smoke.parent.name == "P0_a_screen"
    assert smoke.parent.parent.name == "smoke"
    assert screen.name == "eval"
    assert screen.parent.name == "P0_a_screen"
