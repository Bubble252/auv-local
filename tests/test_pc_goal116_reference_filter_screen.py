from __future__ import annotations

import importlib.util
import json
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from auv_lowlevel.config import load_config

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


runner = _load("goal116_runner", SCRIPTS / "run_pc_goal116_reference_filter_screen.py")
analyzer = _load("goal116_analyzer", SCRIPTS / "analyze_pc_goal116_reference_filter_screen.py")
evaluator = _load("goal116_evaluator", SCRIPTS / "evaluate_pc_ppo.py")


def test_goal116_derived_configs_only_change_pitch_rate_limit(tmp_path: Path) -> None:
    base = json.loads(runner.BASE_CONFIG.read_text(encoding="utf-8"))
    jobs = runner.build_jobs(tmp_path)
    assert [job.arm for job in jobs] == ["R050", "R100", "R200", "ROFF"]
    for job in jobs:
        payload = json.loads(job.environment_config.read_text(encoding="utf-8"))
        expected = json.loads(json.dumps(base))
        expected["guidance"]["command_rate_limit_enabled"] = job.rate_limit_enabled
        expected["guidance"]["pitch_rate_limit_rad_s"] = job.pitch_rate_limit_rad_s
        assert payload == expected
        command = runner.command_for(job, python="python3")
        assert command[command.index("--environment-config") + 1] == str(job.environment_config)
        assert command[command.index("--action-mode") + 1] == "frozen-base"


def _arm(success: int, *, timeout: int = 0, safety: int = 0, subset_success: int = 0) -> dict[str, object]:
    return {
        "success_count": success,
        "timeout_count": timeout,
        "safety_count": safety,
        "original_failure_success_count": subset_success,
    }


def _decide(**arms: dict[str, object]) -> dict[str, object]:
    payload = {
        "R050": _arm(20),
        "R100": _arm(20),
        "R200": _arm(20),
        "ROFF": _arm(20),
    }
    payload.update(arms)
    return analyzer._decide(payload)


def test_goal116_higher_limit_requires_efficiency_safety_and_subset_gain() -> None:
    gate = _decide(R200=_arm(23, subset_success=2))
    assert gate["status"] == "FILTER_CAUSAL_UP"
    gate = _decide(R200=_arm(23, timeout=2, subset_success=2))
    assert gate["status"] == "HARM"


def test_goal116_low_limit_claims_noise_dominance() -> None:
    gate = _decide(R050=_arm(23, subset_success=2))
    assert gate["status"] == "REFERENCE_NOISE_DOMINANT"


def test_goal116_flat_result_is_non_causal() -> None:
    gate = _decide(R050=_arm(22, subset_success=1), R200=_arm(21, subset_success=1))
    assert gate["status"] == "FILTER_NONCAUSAL"


def test_goal116_both_directions_qualify_is_non_monotonic() -> None:
    gate = _decide(R050=_arm(23, subset_success=2), R200=_arm(23, subset_success=2))
    assert gate["status"] == "NONMONOTONIC_FILTER_RESPONSE"


def test_goal116_guidance_override_is_narrowly_allowed() -> None:
    base = load_config(runner.BASE_CONFIG)
    allowed = replace(
        base,
        guidance=replace(
            base.guidance,
            command_rate_limit_enabled=False,
            pitch_rate_limit_rad_s=0.5235987756,
        ),
    )
    evaluator._validate_environment_override(
        base, allowed, allow_guidance_rate_limit_override=True
    )
    with pytest.raises(ValueError, match="guidance contract"):
        evaluator._validate_environment_override(base, allowed)

    forbidden = replace(
        base,
        guidance=replace(base.guidance, heading_rate_limit_rad_s=0.9),
    )
    with pytest.raises(ValueError, match="guidance contract"):
        evaluator._validate_environment_override(
            base, forbidden, allow_guidance_rate_limit_override=True
        )
