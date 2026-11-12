from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
EVALUATOR_SCRIPT = REPO_ROOT / "scripts" / "evaluate_pc_command_residual_offpolicy.py"
EVALUATOR_SPEC = importlib.util.spec_from_file_location(
    "evaluate_pc_command_residual_offpolicy", EVALUATOR_SCRIPT
)
assert EVALUATOR_SPEC is not None and EVALUATOR_SPEC.loader is not None
EVALUATOR = importlib.util.module_from_spec(EVALUATOR_SPEC)
EVALUATOR_SPEC.loader.exec_module(EVALUATOR)

ANALYSIS_SCRIPT = REPO_ROOT / "scripts" / "analyze_pc_goal60_trace_audit.py"
ANALYSIS_SPEC = importlib.util.spec_from_file_location(
    "analyze_pc_goal60_trace_audit", ANALYSIS_SCRIPT
)
assert ANALYSIS_SPEC is not None and ANALYSIS_SPEC.loader is not None
ANALYSIS = importlib.util.module_from_spec(ANALYSIS_SPEC)
ANALYSIS_SPEC.loader.exec_module(ANALYSIS)


def test_trace_put_handles_scalar_on_subset_rows() -> None:
    trace: dict[str, list[np.ndarray]] = {}
    EVALUATOR._trace_put(trace, "step", np.asarray([1]), 7)
    assert trace["step"][0].tolist() == [7]


def test_trace_put_indexes_full_arrays_by_rows() -> None:
    trace: dict[str, list[np.ndarray]] = {}
    EVALUATOR._trace_put(trace, "seed", np.asarray([1, 3]), np.asarray([20, 21, 22, 23]))
    assert trace["seed"][0].tolist() == [21, 23]


def test_trace_put_preserves_action_dimension() -> None:
    trace: dict[str, list[np.ndarray]] = {}
    values = np.arange(6, dtype=np.float64).reshape(2, 3)
    EVALUATOR._trace_put(trace, "action", np.asarray([1]), values)
    assert trace["action"][0].shape == (1, 3)
    assert trace["action"][0].tolist() == [[3.0, 4.0, 5.0]]


def test_spearman_is_one_for_monotone_values() -> None:
    left = np.asarray([1.0, 2.0, 3.0, 4.0])
    right = np.asarray([10.0, 20.0, 30.0, 40.0])
    assert ANALYSIS._spearman(left, right) == pytest.approx(1.0)


def test_spearman_is_none_for_constant_values() -> None:
    left = np.ones(4)
    right = np.asarray([1.0, 2.0, 3.0, 4.0])
    assert ANALYSIS._spearman(left, right) is None
