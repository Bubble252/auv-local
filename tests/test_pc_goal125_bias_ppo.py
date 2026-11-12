from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ANALYZER = ROOT / "scripts" / "analyze_pc_goal125_bias_ppo.py"


def _load_analyzer():
    spec = importlib.util.spec_from_file_location("goal125_analyzer", ANALYZER)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load Goal125 analyzer")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_MOD = _load_analyzer()


def _run(success: int, *, timeout: int = 0, safety: int = 0) -> dict:
    episodes = []
    for index, seed in enumerate(range(20, 60)):
        episodes.append(
            {
                "seed": seed,
                "success": index < success,
                "timeout": index < timeout,
                "safety": index < safety,
                "final_eta_error_l2": 1.0,
                "residual_to_base_l2_ratio_mean": 0.01,
            }
        )
    return {
        "success": success,
        "timeout": timeout,
        "safety": safety,
        "success_seeds": list(range(20, 20 + success)),
        "final_eta_error_l2_median": 1.0,
        "residual_to_base_l2_ratio_median": 0.01,
        "metrics": {},
        "episodes": episodes,
    }


def _controls(strict: int = 24, default: int = 33):
    return {
        seed: {"strict": _run(strict), "default": _run(default)}
        for seed in (86201, 86202)
    }


def _runs(strict_by_arm: dict[str, int], default_by_arm: dict[str, int]):
    return {
        arm: {
            "seeds": {
                seed: {
                    "strict": _run(strict_by_arm[arm]),
                    "default": _run(default_by_arm[arm]),
                }
                for seed in (86201, 86202)
            }
        }
        for arm in _MOD.ARMS
    }


def test_goal125_gate_gain() -> None:
    runs = _runs(
        {"Z15": 24, "B15": 24, "B20": 26, "BP3": 24},
        {"Z15": 33, "B15": 33, "B20": 33, "BP3": 33},
    )
    gate = _MOD._gate(runs, _controls())
    assert gate["status"] == "PPO_BIAS_GAIN"
    assert gate["arms"]["B20"]["strict_delta_mean"] == 2.0


def test_goal125_gate_fixed_bias_ceiling() -> None:
    runs = _runs(
        {"Z15": 24, "B15": 24, "B20": 24, "BP3": 24},
        {"Z15": 33, "B15": 33, "B20": 33, "BP3": 33},
    )
    gate = _MOD._gate(runs, _controls())
    assert gate["status"] == "BIAS_FIXED_IS_CEILING"


def test_goal125_gate_harm() -> None:
    runs = _runs(
        {"Z15": 22, "B15": 22, "B20": 22, "BP3": 22},
        {"Z15": 33, "B15": 33, "B20": 33, "BP3": 33},
    )
    gate = _MOD._gate(runs, _controls())
    assert gate["status"] == "PPO_BIAS_HARM"


def test_goal125_gate_invalid_when_control_not_reproduced() -> None:
    runs = _runs(
        {"Z15": 24, "B15": 24, "B20": 24, "BP3": 24},
        {"Z15": 33, "B15": 33, "B20": 33, "BP3": 33},
    )
    gate = _MOD._gate(runs, _controls(strict=20))
    assert gate["status"] == "INVALID"
