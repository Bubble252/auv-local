from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from summarize_pc_v0d_r2b_smooth_teacher_20261107 import _variant_checks  # noqa: E402


def _metrics(
    *,
    rmse: float = 5.0,
    success: float = 0.8,
    timeout: float = 0.2,
    safety: float = 0.0,
    rudder_sat: float = 0.3,
    stern_sat: float = 0.1,
) -> dict[str, object]:
    return {
        "path_three_d_rmse_m": rmse,
        "success_rate": success,
        "timeout_rate": timeout,
        "safety_termination_rate": safety,
        "action_channel_saturation_rate": [rudder_sat, stern_sat, 0.0],
    }


def test_smooth_teacher_gate_passes_valid_variant() -> None:
    checks = _variant_checks(
        _metrics(),
        _metrics(rmse=9.0, success=0.2, timeout=0.8),
        _metrics(success=0.85, timeout=0.15),
        _metrics(rmse=9.4, success=0.15, timeout=0.85),
    )
    assert all(checks.values())


def test_smooth_teacher_gate_rejects_excess_saturation_and_timeout() -> None:
    checks = _variant_checks(
        _metrics(rudder_sat=0.41, timeout=0.26),
        _metrics(rmse=9.0, success=0.2, timeout=0.96),
        _metrics(success=0.85, timeout=0.15),
        _metrics(rmse=9.4, success=0.15, timeout=0.85),
    )
    assert checks["default_rudder_sat_le_0p40"] is False
    assert checks["default_timeout_le_baseline_plus_0p10"] is False
    assert checks["strict125_timeout_le_baseline_plus_0p10"] is False
