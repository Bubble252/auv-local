from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

SCRIPT = Path("scripts/analyze_pc_goal31_gate.py")
SPEC = importlib.util.spec_from_file_location("analyze_pc_goal31_gate", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _block(
    *,
    capture: float,
    horizontal: float,
    horizontal_ci_high: float,
    vertical: float,
    saturation: float = 0.0,
    chain_seeds: int = 40,
) -> dict[str, object]:
    rates = {
        key: {"delta": 0.0}
        for key in (
            "success_protocol",
            "success_eta",
            "terminal_eta_capture",
            "protocol_and_terminal_eta_capture",
            "pnav_only_success",
        )
    }
    rates["terminal_eta_capture"] = {"delta": capture}
    numeric = {
        "horizontal_normal_eta_m": {
            "median": horizontal,
            "bootstrap_95_ci": [-1.0, horizontal_ci_high],
        },
        "vertical_eta_m": {
            "median": vertical,
            "bootstrap_95_ci": [-1.0, 1.0],
        },
        "terminal_speed_ratio": {"median": 0.0},
        "terminal_material_saturation_rate": {"median": saturation},
    }
    return {
        "rates": rates,
        "outcomes": {"plane_miss": 0.0, "timeout": 0.0, "safety": 0.0},
        "paired": {"numeric_delta": numeric},
        "path_metrics": {
            "path_three_d_rmse_m": {"median": 0.0},
            "path_vertical_rmse_m": {"median": 0.0},
        },
        "chain_audit": {
            "per_seed": [
                {"max_abs_actuator_diff": 1.0} for _ in range(chain_seeds)
            ]
        },
    }


def test_gate_passes_when_capture_and_offsets_clear_thresholds() -> None:
    blocks = {
        "default": _block(
            capture=0.05,
            horizontal=-0.10,
            horizontal_ci_high=-0.01,
            vertical=0.0,
        ),
        "strict125": _block(
            capture=0.0,
            horizontal=-0.08,
            horizontal_ci_high=0.02,
            vertical=0.01,
        ),
    }

    gate = MODULE.evaluate_gate(blocks)

    assert gate["verdict"] == "PASS"
    assert gate["primary"]["pass"] is True


def test_gate_fails_on_vertical_offset_even_if_capture_improves() -> None:
    blocks = {
        "default": _block(
            capture=0.05,
            horizontal=-0.10,
            horizontal_ci_high=-0.01,
            vertical=0.11,
        ),
        "strict125": _block(
            capture=0.0,
            horizontal=-0.08,
            horizontal_ci_high=0.02,
            vertical=0.01,
        ),
    }

    gate = MODULE.evaluate_gate(blocks)

    assert gate["verdict"] == "FAIL"
    assert gate["primary"]["vertical_offset"] is False


def test_gate_fails_on_terminal_saturation_guardrail() -> None:
    blocks = {
        "default": _block(
            capture=0.05,
            horizontal=-0.10,
            horizontal_ci_high=-0.01,
            vertical=0.0,
            saturation=0.04,
        ),
        "strict125": _block(
            capture=0.0,
            horizontal=-0.08,
            horizontal_ci_high=0.02,
            vertical=0.01,
        ),
    }

    gate = MODULE.evaluate_gate(blocks)

    assert gate["verdict"] == "FAIL"
    assert gate["guards"]["terminal_saturation"] is False


def test_gate_reports_invalid_chain_when_too_few_seeds_change() -> None:
    blocks = {
        "default": _block(
            capture=0.05,
            horizontal=-0.10,
            horizontal_ci_high=-0.01,
            vertical=0.0,
            chain_seeds=10,
        ),
        "strict125": _block(
            capture=0.0,
            horizontal=-0.08,
            horizontal_ci_high=0.02,
            vertical=0.01,
        ),
    }

    gate = MODULE.evaluate_gate(blocks)

    assert gate["verdict"] == "FAIL"
    assert gate["chain_valid"] is False
