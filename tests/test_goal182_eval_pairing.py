from __future__ import annotations

from auv_lowlevel.eval_pairing import (
    BASELINE,
    PAIRED,
    UNPAIRED,
    audit_paired_zero,
)


def _zero(tag: str, *, zero_for: list[str] | None = None) -> dict:
    return {
        "tag": tag,
        "evaluator": "ppo",
        "training_config": "configs/zero.json",
        "action_mode": "frozen-base",
        "is_zero_baseline": True,
        "zero_for": zero_for or [],
    }


def _policy(tag: str, composition: str, *, paired: str | None = None) -> dict:
    arm = {
        "tag": tag,
        "evaluator": "ppo",
        "training_config": "configs/policy.json",
        "checkpoint": "artifacts/policy.pt",
        "action_mode": "policy",
        "learning_composition": composition,
    }
    if paired is not None:
        arm["paired_zero_tag"] = paired
    return arm


def test_bcres_policy_requires_explicit_paired_zero() -> None:
    arms = [_policy("bcres", "BCRES")]
    statuses, errors = audit_paired_zero(arms, strict=True)
    assert statuses["bcres"] == "INVALID_MISSING_PAIRED_ZERO"
    assert any("missing paired_zero_tag" in error for error in errors)


def test_paired_zero_must_be_selected_and_declared_by_zero() -> None:
    arms = [
        _policy("bcres", "BCRES", paired="zero_bcres"),
        _zero("zero_bcres", zero_for=["bcres"]),
    ]
    statuses, errors = audit_paired_zero(arms, selected_tags=["bcres"])
    assert statuses["bcres"] == "INVALID_MISSING_PAIRED_ZERO"
    assert any("not selected" in error for error in errors)

    statuses, errors = audit_paired_zero(arms)
    assert statuses == {"bcres": PAIRED, "zero_bcres": BASELINE}
    assert errors == []


def test_zero_for_mismatch_is_invalid() -> None:
    arms = [
        _policy("trres", "TRRES_CMD", paired="zero_trres"),
        _zero("zero_trres", zero_for=["another_arm"]),
    ]
    statuses, errors = audit_paired_zero(arms)
    assert statuses["trres"] == "INVALID_ZERO_BASELINE_PROTOCOL"
    assert any("does not declare" in error for error in errors)


def test_pe2e_without_paired_zero_is_explicitly_unpaired() -> None:
    arms = [
        {
            "tag": "pe2e",
            "evaluator": "ppo",
            "training_config": "configs/policy.json",
            "checkpoint": "artifacts/policy.pt",
            "action_mode": "policy",
            "control_contract": "CC_PE2E",
        }
    ]
    statuses, errors = audit_paired_zero(arms, strict=True)
    assert statuses["pe2e"] == UNPAIRED
    assert errors == []


def test_pe2e_must_not_claim_cross_architecture_zero() -> None:
    arms = [
        {
            "tag": "pe2e",
            "evaluator": "ppo",
            "training_config": "configs/policy.json",
            "checkpoint": "artifacts/policy.pt",
            "action_mode": "policy",
            "control_contract": "CC_PE2E",
            "paired_zero_tag": "zero_bc",
        },
        _zero("zero_bc", zero_for=["pe2e"]),
    ]
    statuses, errors = audit_paired_zero(arms)
    assert statuses["pe2e"] == "INVALID_ZERO_BASELINE_PROTOCOL"
    assert any("must not claim" in error for error in errors)


def test_pe2e_bcft_is_unpaired_even_though_bcft_requires_pair_elsewhere() -> None:
    arm = _policy("pe2e_bcft", "BCFT")
    arm["control_contract"] = "CC_PE2E"
    statuses, errors = audit_paired_zero([arm], strict=True)
    assert statuses["pe2e_bcft"] == UNPAIRED
    assert errors == []


def test_pe2e_bcft_must_not_claim_a_bc_zero() -> None:
    arm = _policy("pe2e_bcft", "BCFT", paired="zero_bc")
    arm["control_contract"] = "CC_PE2E"
    zero = _zero("zero_bc", zero_for=["pe2e_bcft"])
    statuses, errors = audit_paired_zero([arm, zero])
    assert statuses["pe2e_bcft"] == "INVALID_ZERO_BASELINE_PROTOCOL"
    assert any("must not claim" in error for error in errors)
