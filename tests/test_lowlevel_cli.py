from __future__ import annotations

import argparse
import ast
import hashlib
import os
import subprocess
import sys
from pathlib import Path

import pytest
from auv_lowlevel._cli import add_seed_range_arguments

REPO_ROOT = Path(__file__).resolve().parents[1]
SEED_RANGE_SCRIPTS = (
    "run_pc_goal97_scale_sweep.py",
    "run_pc_goal98_scale_grid.py",
    "run_pc_goal99_distance_schedule.py",
    "run_pc_goal100_terminal_prior.py",
    "run_pc_goal101_alos_pairing.py",
    "run_pc_goal102_ppo_dose_response.py",
    "run_pc_goal104_channel_parameterization.py",
    "run_pc_goal105_channel_interaction.py",
    "run_pc_goal106_near_terminal_rpm_schedule.py",
    "run_pc_goal107_terminal_reward_credit.py",
)
HELP_GOLDEN_SHA256 = {
    "run_pc_goal97_scale_sweep.py": "4035554d12f9aababc73b307f557126705a329d66b6eb67f0c493c0db7891736",
    "run_pc_goal98_scale_grid.py": "2ed50cc09433a411c934e7942ed6f88d7d52b97cb2e94ce36133b3239fe93cc3",
    "run_pc_goal99_distance_schedule.py": "f96216cea47778adafd0ffb39d65d708eaedcc45fedf4206d96bf844f25b5e9d",
    "run_pc_goal100_terminal_prior.py": "0236a8979dcbcefe68097309676d754870ee6c4d42b1a8c596aef1eb3a7aa2fe",
    "run_pc_goal101_alos_pairing.py": "f62b7a59dcea94af42c8a07e70edcdfe5a6eee4483b54d0aef72656b5057feee",
    "run_pc_goal102_ppo_dose_response.py": "f1714de6f9c9726df1317f258e7388049ca761cc0c2b8d19ca59219dfeeb3897",
    "run_pc_goal104_channel_parameterization.py": "b1f9096e3fff1fec4021a7463e8630df1dbb48dfd7a2c8afa377310eef9db457",
    "run_pc_goal105_channel_interaction.py": "84f4726c259b8d88ddbd80c45d5cab216d916adee6c8234192dde6118b05c19f",
    "run_pc_goal106_near_terminal_rpm_schedule.py": "e9067276ca4e1f6f0f2d3cc04216dd648c5ac0a287685699a0857e7a942d55b8",
    "run_pc_goal107_terminal_reward_credit.py": "5d30fe3eb1eda2568420b0776aab054a5bf4c5845902e400160434d1aff3c8b7",
}


def test_seed_range_arguments_preserve_defaults() -> None:
    parser = argparse.ArgumentParser()
    add_seed_range_arguments(parser)

    args = parser.parse_args([])

    assert args.seed_start == 20
    assert args.seed_count == 40


@pytest.mark.parametrize("script_name", SEED_RANGE_SCRIPTS)
def test_scripts_use_shared_seed_range_arguments(script_name: str) -> None:
    path = REPO_ROOT / "scripts" / script_name
    tree = ast.parse(path.read_text(encoding="utf-8"))
    shared_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "add_seed_range_arguments"
    ]
    direct_seed_args = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "add_argument"
        and node.args
        and isinstance(node.args[0], ast.Constant)
        and node.args[0].value in {"--seed-start", "--seed-count"}
    ]

    assert len(shared_calls) == 1
    assert direct_seed_args == []


@pytest.mark.parametrize("script_name", SEED_RANGE_SCRIPTS)
def test_script_help_matches_golden(script_name: str) -> None:
    env = os.environ.copy()
    env.update(
        {
            "COLUMNS": "100",
            "PYTHONHASHSEED": "0",
            "PYTHONPATH": str(REPO_ROOT / "src"),
        }
    )
    completed = subprocess.run(
        [sys.executable, f"scripts/{script_name}", "--help"],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr.decode(errors="replace")
    assert hashlib.sha256(completed.stdout).hexdigest() == HELP_GOLDEN_SHA256[script_name]
