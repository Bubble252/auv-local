from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "evaluate_pc_ppo.py"
SPEC = importlib.util.spec_from_file_location("evaluate_pc_ppo_goal96_reward", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
EVALUATOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EVALUATOR)


def test_reward_breakdown_trace_flag_is_opt_in() -> None:
    base_args = [
        "--training-config",
        "train.json",
        "--checkpoint",
        "model.pt",
        "--output",
        "out",
    ]
    assert EVALUATOR._parse_args(base_args).trace_reward_breakdown is False
    args = EVALUATOR._parse_args(
        [
            *base_args,
            "--trace-output",
            "trace.npz",
            "--trace-reward-breakdown",
        ]
    )
    assert args.trace_reward_breakdown is True


def test_reward_breakdown_trace_fields_are_complete_and_row_aligned() -> None:
    infos = {
        "reward_total": torch.zeros((2,)),
        "reward_ppo_scaled": torch.zeros((2,)),
        "reward_terminal": torch.zeros((2,)),
        "reward_terminal_progress": torch.zeros((2,)),
        "reward_path_cross": torch.zeros((2,)),
        "raw_reward": torch.zeros((2,)),
        "terminal_kind": torch.zeros((2,), dtype=torch.long),
    }
    fields = EVALUATOR._reward_breakdown_trace_fields(infos)
    assert set(fields) == {
        "reward_total",
        "reward_ppo_scaled",
        "reward_terminal",
        "reward_terminal_progress",
        "reward_path_cross",
    }
    assert all(value.shape == (2,) for value in fields.values())


def test_reward_breakdown_trace_requires_trace_output() -> None:
    EVALUATOR._validate_trace_reward_breakdown(Path("trace.npz"), True)
    with pytest.raises(ValueError, match="--trace-reward-breakdown requires"):
        EVALUATOR._validate_trace_reward_breakdown(None, True)
