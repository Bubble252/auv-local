from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_SCRIPT = REPO_ROOT / "scripts" / "analyze_pc_goal74_capture_margin.py"
SPEC = importlib.util.spec_from_file_location("analyze_pc_goal74_capture_margin", ANALYSIS_SCRIPT)
assert SPEC is not None and SPEC.loader is not None
ANALYSIS = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = ANALYSIS
SPEC.loader.exec_module(ANALYSIS)


def _record(
    *,
    seed: int,
    d_p: float,
    d_e: float,
    tight_radius: float = 1.75,
    protocol_radius: float = 2.0,
    signed: float = -1.0,
    bias: float = 0.0,
    actual_success: bool = True,
) -> dict[str, object]:
    record: dict[str, object] = {
        "scenario": "default",
        "run_key": "752",
        "action_mode": "policy",
        "seed": seed,
        "tight_radius_m": tight_radius,
        "protocol_radius_m": protocol_radius,
        "terminal_p_nav_distance_m": d_p,
        "terminal_eta_distance_m": d_e,
        "terminal_signed_offset_m": signed,
        "p_nav_eta_error_m": abs(d_e - d_p),
        "bias_projection_m": bias,
        "actual_success": actual_success,
        "actual_true_capture": d_e <= protocol_radius,
    }
    record.update(
        ANALYSIS._margin_fields(
            d_p=d_p,
            d_e=d_e,
            tight_radius=tight_radius,
            protocol_radius=protocol_radius,
            signed=signed,
        )
    )
    return record


def test_margin_tag_is_stable() -> None:
    assert ANALYSIS._margin_tag(0.0) == "0p00"
    assert ANALYSIS._margin_tag(0.25) == "0p25"


def test_true_capture_uses_protocol_radius_not_tight_radius() -> None:
    record = _record(seed=20, d_p=1.74, d_e=1.90)
    assert record["true_capture"] is True
    assert record["true_success_0p00"] is True
    assert record["false_success_0p00"] is False


def test_margin_sweep_reduces_false_success_and_detects_bias() -> None:
    records = [
        _record(seed=20, d_p=1.73, d_e=2.25, bias=0.20),
        _record(seed=21, d_p=1.74, d_e=2.30, bias=0.20),
        _record(seed=22, d_p=1.72, d_e=2.20, bias=0.20),
        _record(seed=23, d_p=1.25, d_e=1.35, bias=0.05),
        _record(seed=24, d_p=1.20, d_e=1.30, bias=0.05),
        _record(seed=25, d_p=1.30, d_e=1.40, bias=0.05),
    ]
    result = ANALYSIS._aggregate(records, "default")
    assert result["margin_counts"]["0.00"]["false_success"] == 3
    assert result["margin_counts"]["0.25"]["false_success"] == 0
    assert result["h74a"]["supported"] is True
    assert result["h74b"]["supported"] is True
    assert result["reclassification_mismatch"] == 0


def test_plane_crossing_is_diagnostic_only() -> None:
    record = _record(seed=20, d_p=3.0, d_e=3.1, signed=0.1, bias=0.0)
    assert record["plane_0p50"] is True
    assert record["success_0p50"] is False
    assert record["false_success_0p50"] is False


def test_bias_gap_requires_both_groups() -> None:
    records = [
        _record(seed=20, d_p=1.73, d_e=2.25, bias=0.20),
        _record(seed=21, d_p=1.74, d_e=2.30, bias=0.20),
        _record(seed=22, d_p=1.72, d_e=2.20, bias=0.20),
    ]
    result = ANALYSIS._aggregate(records, "default")
    assert result["h74b"]["supported"] is False


def test_margin_reduction_threshold() -> None:
    records = [
        _record(seed=20, d_p=1.73, d_e=2.25),
        _record(seed=21, d_p=1.24, d_e=1.30),
    ]
    result = ANALYSIS._aggregate(records, "default")
    assert result["h74a"]["false_success_reduction_at_0p25"] == 1
    assert result["h74a"]["supported"] is False


def test_margin_is_unusable_when_true_success_collapses() -> None:
    records = [
        _record(seed=20, d_p=1.73, d_e=2.25, bias=0.20),
        _record(seed=21, d_p=1.74, d_e=2.30, bias=0.20),
        _record(seed=22, d_p=1.72, d_e=2.20, bias=0.20),
        _record(seed=23, d_p=1.74, d_e=1.90, bias=0.05),
        _record(seed=24, d_p=1.73, d_e=1.95, bias=0.05),
        _record(seed=25, d_p=1.72, d_e=1.85, bias=0.05),
    ]
    result = ANALYSIS._aggregate(records, "default")
    assert result["margin_counts"]["0.00"]["false_success"] == 3
    assert result["margin_counts"]["0.25"]["false_success"] == 0
    assert result["margin_counts"]["0.25"]["true_success"] == 0
    assert result["h74a"]["margin_usable"] is False
    assert result["h74a"]["supported"] is False
    assert result["h74b"]["supported"] is True
