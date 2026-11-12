from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_pc_eval_matrix.py"
MATRIX = ROOT / "configs" / "lowlevel" / "eval" / "pc_v1_nominal_eval_matrix_20261108.json"


def _load_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("run_pc_eval_matrix", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _matrix() -> dict:
    return json.loads(MATRIX.read_text(encoding="utf-8"))


def test_matrix_fixture_is_valid() -> None:
    module = _load_module()
    matrix = module.validate_matrix(_matrix())

    assert matrix["dev_seed_start"] == 20
    assert matrix["dev_seed_count"] == 40
    assert matrix["dagger_pool_seeds"] == list(range(200, 240))
    assert matrix["held_out_seeds"] == list(range(100, 150))


def test_validate_matrix_rejects_duplicate_dev_seeds() -> None:
    module = _load_module()
    matrix = _matrix()
    matrix["seeds"]["dev"][1] = matrix["seeds"]["dev"][0]

    with pytest.raises(ValueError, match="duplicate seeds"):
        module.validate_matrix(matrix)


def test_validate_matrix_rejects_held_out_overlap() -> None:
    module = _load_module()
    matrix = _matrix()
    matrix["seeds"]["dev"].append(100)

    with pytest.raises(ValueError, match="overlap held_out"):
        module.validate_matrix(matrix)


def test_validate_matrix_rejects_non_contiguous_dev_seeds() -> None:
    module = _load_module()
    matrix = _matrix()
    matrix["seeds"]["dev"][-1] = 60

    with pytest.raises(ValueError, match="contiguous"):
        module.validate_matrix(matrix)


def test_validate_environment_configs_enforces_scenarios() -> None:
    module = _load_module()
    matrix = module.validate_matrix(_matrix())
    hashes = module.validate_environment_configs(matrix)

    assert set(hashes) == {"default", "strict125"}
    assert all(len(value) == 64 for value in hashes.values())


def test_validate_environment_configs_rejects_wrong_goal_radius() -> None:
    module = _load_module()
    matrix = module.validate_matrix(_matrix())
    matrix = copy.deepcopy(matrix)
    matrix["base_environment_configs"]["strict125"] = matrix["base_environment_configs"]["default"]

    with pytest.raises(ValueError, match="goal_radius_m"):
        module.validate_environment_configs(matrix)


def test_build_eval_command_adds_path_zero_flag() -> None:
    module = _load_module()
    matrix = module.validate_matrix(_matrix())
    command = module.build_eval_command(
        matrix=matrix,
        scenario="strict125",
        path_mode="path_zero",
        checkpoint=Path("/tmp/checkpoint.pt"),
        output=Path("/tmp/out"),
    )

    assert command[-1] == "--path-zero"
    assert command[command.index("--seed-start") + 1] == "20"
    assert command[command.index("--seed-count") + 1] == "40"
    assert command[command.index("--termination-mode") + 1] == "endpoint_stop"
    assert command[command.index("--path-dim") + 1] == "8"


def test_success_only_rmse_averages_episode_mse() -> None:
    module = _load_module()
    episodes = [
        {"success": True, "steps": 10, "path_three_d_rmse_m": 2.0},
        {"success": True, "steps": 30, "path_three_d_rmse_m": 4.0},
        {"success": False, "steps": 100, "path_three_d_rmse_m": 100.0},
    ]

    value = module._success_only_rmse(episodes, "path_three_d_rmse_m")

    assert value == pytest.approx(((4.0 + 16.0) / 2) ** 0.5)


def test_success_only_rmse_returns_none_without_success() -> None:
    module = _load_module()
    assert module._success_only_rmse([], "path_three_d_rmse_m") is None
    assert (
        module._success_only_rmse(
            [{"success": False, "steps": 10, "path_three_d_rmse_m": 1.0}],
            "path_three_d_rmse_m",
        )
        is None
    )
