from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

SCRIPT = Path("scripts/analyze_pc_goal33_terminal_authority.py")
SPEC = importlib.util.spec_from_file_location("analyze_pc_goal33_terminal_authority", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _record(
    seed: int,
    *,
    speed: float = 1.5,
    authority: float = 0.2,
    shortfall: float = 0.3,
    binding: float = 0.1,
    entry_excess: float = 4.0,
    required: float = 0.5,
    time_to_plane: float = 9.0,
    vertical_miss: bool = True,
) -> dict[str, object]:
    return {
        "seed": seed,
        "vertical_miss": vertical_miss,
        "horizontal_speed_median_mps": speed,
        "authority_p95_mps": authority,
        "required_minus_authority_mps": shortfall,
        "stern_binding_rate": binding,
        "stern_limit_bound_rate": binding,
        "stern_rate_bound_rate": 0.0,
        "entry_excess_m": entry_excess,
        "required_closure_mps": required,
        "time_to_plane_s": time_to_plane,
    }


def test_load_limits_validates_and_converts_config(tmp_path: Path) -> None:
    config = {
        "actuator": {
            "max_stern_plane_angle_deg": 20,
            "max_fin_rate_deg_s": 150,
        },
        "base_controller": {"pitch_gain": 2, "pitch_rate_gain": 1},
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    summary = {"environment_config_override": str(path)}

    limits = MODULE._load_limits(summary, tmp_path)

    assert limits["config_path"] == str(path)
    assert np.isclose(limits["max_stern_plane_angle_rad"], np.deg2rad(20.0))
    assert np.isclose(limits["max_fin_rate_rad_s"], np.deg2rad(150.0))


def test_load_limits_rejects_changed_controller_gain(tmp_path: Path) -> None:
    config = {
        "actuator": {
            "max_stern_plane_angle_deg": 20,
            "max_fin_rate_deg_s": 150,
        },
        "base_controller": {"pitch_gain": 3, "pitch_rate_gain": 1},
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    summary = {"environment_config": str(path)}

    try:
        MODULE._load_limits(summary, tmp_path)
    except ValueError as exc:
        assert "unexpected controller/actuator values" in str(exc)
    else:
        raise AssertionError("expected config mismatch to fail")


def test_stern_segment_metrics_detects_angle_limit_binding() -> None:
    count = 5
    trace = {
        "time_s": np.arange(count, dtype=np.float64) * 0.02,
        "pitch_error_rad": np.full(count, 0.2, dtype=np.float64),
        "nu": np.zeros((count, 6), dtype=np.float64),
        "actual_actuator": np.tile(
            np.asarray([[0.1, 0.2, 100.0]], dtype=np.float64), (count, 1)
        ),
        "material_saturated": np.zeros(count, dtype=bool),
    }
    limits = {
        "pitch_gain": 2.0,
        "pitch_rate_gain": 1.0,
        "max_stern_plane_angle_rad": np.deg2rad(20.0),
        "max_fin_rate_rad_s": np.deg2rad(150.0),
    }

    result = MODULE._stern_segment_metrics(
        trace,
        np.arange(count, dtype=np.int64),
        0,
        count - 1,
        limits,
    )

    assert result["stern_limit_bound_rate"] == 1.0
    assert result["stern_binding_rate"] == 1.0
    assert result["stern_rate_bound_rate"] == 0.0
    assert np.isclose(result["stern_demand_abs_median_rad"], 0.4)


def test_stern_segment_metrics_detects_rate_limit_binding() -> None:
    count = 4
    time_s = np.arange(count, dtype=np.float64) * 0.02
    pitch_error = np.asarray([0.0, 1.0, 1.0, 1.0], dtype=np.float64)
    max_step = np.deg2rad(150.0) * 0.02
    stern = -max_step * np.arange(count, dtype=np.float64)
    actuator = np.zeros((count, 3), dtype=np.float64)
    actuator[:, 0] = 0.1
    actuator[:, 1] = stern
    actuator[:, 2] = 100.0
    trace = {
        "time_s": time_s,
        "pitch_error_rad": pitch_error,
        "nu": np.zeros((count, 6), dtype=np.float64),
        "actual_actuator": actuator,
        "material_saturated": np.zeros(count, dtype=bool),
    }
    limits = {
        "pitch_gain": 2.0,
        "pitch_rate_gain": 1.0,
        "max_stern_plane_angle_rad": np.deg2rad(20.0),
        "max_fin_rate_rad_s": np.deg2rad(150.0),
    }

    result = MODULE._stern_segment_metrics(
        trace,
        np.arange(count, dtype=np.int64),
        0,
        count - 1,
        limits,
    )

    assert result["stern_rate_bound_rate"] == 1.0


def test_rank_halves_excludes_odd_middle_record() -> None:
    records = [_record(seed, speed=float(seed)) for seed in range(20, 25)]

    low, high = MODULE._rank_halves(records, "horizontal_speed_median_mps")

    assert [row["seed"] for row in low] == [20, 21]
    assert [row["seed"] for row in high] == [23, 24]


def test_h_speed_supports_speed_conditioned_authority() -> None:
    records = [
        _record(
            seed,
            speed=1.0 + 0.05 * index,
            authority=0.15 + 0.02 * index,
        )
        for index, seed in enumerate(range(20, 32))
    ]

    result = MODULE._h_speed(records)

    assert result["passes_all"] is True
    assert result["speed_authority_spearman"] == 1.0
    assert result["high_minus_low_authority_mps"] > 0.05


def test_h_speed_falsifies_flat_authority() -> None:
    records = [
        _record(seed, speed=1.0 + 0.05 * index, authority=0.2)
        for index, seed in enumerate(range(20, 32))
    ]

    result = MODULE._h_speed(records)

    assert result["passes_all"] is False
    assert result["high_minus_low_authority_mps"] == 0.0


def test_h_stern_supports_shortfall_binding_correlation() -> None:
    records = [
        _record(
            seed,
            shortfall=0.1 + 0.03 * index,
            binding=0.03 + 0.01 * index,
        )
        for index, seed in enumerate(range(20, 32))
    ]

    result = MODULE._h_stern(records)

    assert result["passes_all"] is True
    assert result["shortfall_binding_spearman"] == 1.0
    assert result["high_shortfall_binding_median"] >= 0.05


def test_h_stern_falsifies_low_binding() -> None:
    records = [
        _record(seed, shortfall=0.1 + 0.03 * index, binding=0.0)
        for index, seed in enumerate(range(20, 32))
    ]

    result = MODULE._h_stern(records)

    assert result["passes_all"] is False
    assert result["high_shortfall_binding_median"] == 0.0


def test_h_excess_supports_magnitude_dominant_required_closure() -> None:
    records = [
        _record(
            seed,
            entry_excess=2.0 + 0.2 * index,
            required=0.3 + 0.03 * index,
            time_to_plane=8.0 + 0.1 * index,
        )
        for index, seed in enumerate(range(20, 32))
    ]

    result = MODULE._h_excess(records)

    assert result["passes_all"] is True
    assert result["excess_required_spearman"] == 1.0
    assert result["time_required_spearman"] == 1.0


def test_h_excess_falsifies_short_time_drive() -> None:
    records = [
        _record(
            seed,
            entry_excess=2.0,
            required=0.5 + 0.05 * index,
            time_to_plane=10.0 - 0.2 * index,
        )
        for index, seed in enumerate(range(20, 32))
    ]

    result = MODULE._h_excess(records)

    assert result["passes_all"] is False
    assert result["time_required_spearman"] == -1.0


def test_overall_verdict_reports_single_speed_mechanism() -> None:
    blocks = {}
    for scenario in MODULE.SCENARIOS:
        blocks[scenario] = {}
        for mode in MODULE.MODES:
            blocks[scenario][mode] = {
                "h_speed": {"passes_all": True},
                "h_stern": {"passes_all": False},
                "h_excess": {"passes_all": False},
            }

    result = MODULE._overall_verdict(blocks)

    assert result["decision"] == "SPEED_DEPENDENT_AUTHORITY_SUPPORTED"


def test_overall_verdict_refuses_multi_mechanism_candidate() -> None:
    blocks = {}
    for scenario in MODULE.SCENARIOS:
        blocks[scenario] = {}
        for mode in MODULE.MODES:
            blocks[scenario][mode] = {
                "h_speed": {"passes_all": True},
                "h_stern": {"passes_all": True},
                "h_excess": {"passes_all": False},
            }

    result = MODULE._overall_verdict(blocks)

    assert result["decision"] == "MULTI_FACTOR"


def test_overall_verdict_refuses_unsupported_mechanism() -> None:
    blocks = {}
    for scenario in MODULE.SCENARIOS:
        blocks[scenario] = {}
        for mode in MODULE.MODES:
            blocks[scenario][mode] = {
                "h_speed": {"passes_all": False},
                "h_stern": {"passes_all": False},
                "h_excess": {"passes_all": False},
            }

    result = MODULE._overall_verdict(blocks)

    assert result["decision"] == "NO_SUPPORTED_MECHANISM"
