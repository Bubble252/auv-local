from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

SCRIPT = (
    Path(__file__).resolve().parents[1] / "scripts" / "benchmark_odarrl_policy_latency.py"
)
SPEC = importlib.util.spec_from_file_location("benchmark_odarrl_policy_latency", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_percentile_interpolates_and_validates_probability() -> None:
    values = [4.0, 1.0, 3.0, 2.0]
    assert MODULE._percentile(values, 0.0) == 1.0
    assert MODULE._percentile(values, 0.5) == 2.5
    assert MODULE._percentile(values, 1.0) == 4.0
    with pytest.raises(ValueError, match="probability"):
        MODULE._percentile(values, 1.1)
    with pytest.raises(ValueError, match="sample"):
        MODULE._percentile([], 0.5)


def test_latency_summary_reports_expected_percentiles() -> None:
    summary = MODULE._summarize_ms([1.0, 2.0, 3.0, 4.0])
    assert summary["mean_ms"] == pytest.approx(2.5)
    assert summary["p50_ms"] == pytest.approx(2.5)
    assert summary["p95_ms"] == pytest.approx(3.85)
    assert summary["min_ms"] == 1.0
    assert summary["max_ms"] == 4.0
