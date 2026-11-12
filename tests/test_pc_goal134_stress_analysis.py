from __future__ import annotations

from scripts.analyze_pc_goal134_stress import _classify_level, _overall_status


def test_classify_level_prioritizes_advantage() -> None:
    level = {
        "zero_strict_drop": 2,
        "advantaged": ["S_s86201", "S_s86202", "S_s86203", "S_s86204"],
        "harm": [],
    }
    assert _classify_level(level, level["zero_strict_drop"]) == "STRESS_ADVANTAGE"


def test_classify_level_requires_stress_discrimination() -> None:
    level = {
        "zero_strict_drop": 1,
        "advantaged": ["S_s86201", "S_s86202", "S_s86203", "S_s86204"],
        "harm": [],
    }
    assert (
        _classify_level(level, level["zero_strict_drop"])
        == "STRESS_NOT_DISCRIMINATIVE"
    )


def test_classify_level_single_seed_and_harm() -> None:
    single = {
        "zero_strict_drop": 2,
        "advantaged": ["S_s86201"],
        "harm": [],
    }
    harm = {
        "zero_strict_drop": 2,
        "advantaged": [],
        "harm": ["S_s86201", "S_s86202", "S_s86203", "S_s86204"],
    }
    assert _classify_level(single, single["zero_strict_drop"]) == "SINGLE_SEED_ONLY"
    assert _classify_level(harm, harm["zero_strict_drop"]) == "SAC_STRESS_HARM"


def test_overall_status_does_not_claim_no_advantage_without_stress() -> None:
    levels = {
        "nominal": {"zero_strict_drop": 0, "status": "STRESS_NOT_DISCRIMINATIVE"},
        "current20": {"zero_strict_drop": 1, "status": "STRESS_NOT_DISCRIMINATIVE"},
    }
    assert _overall_status(levels)[0] == "STRESS_NOT_DISCRIMINATIVE"
