"""Paired-zero baseline metadata and validation for standing evaluations."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

PAIR_REQUIRED_COMPOSITIONS = frozenset({"BCRES", "BCFT", "TRRES_CMD", "TRRES_ACT"})
PE2E_CONTROL_CONTRACT = "CC_PE2E"
PAIRED = "PAIRED"
UNPAIRED = "ZERO_BASELINE_UNPAIRED"
BASELINE = "ZERO_BASELINE"
UNKNOWN = "UNKNOWN"


def is_zero_arm(arm: dict[str, Any]) -> bool:
    """Return whether an arm describes a zero-residual baseline."""

    if bool(arm.get("is_zero_baseline", False)):
        return True
    zero_for = arm.get("zero_for")
    if isinstance(zero_for, str):
        return bool(zero_for)
    if isinstance(zero_for, list):
        return bool(zero_for)
    return str(arm.get("action_mode", "")) in {"zero-residual", "frozen-base", "alos"}


def _tag(arm: dict[str, Any]) -> str:
    return str(arm.get("tag", ""))


def _selected(arms: list[dict[str, Any]], selected_tags: Iterable[str] | None) -> list[str]:
    all_tags = [_tag(arm) for arm in arms]
    if selected_tags is None:
        return all_tags
    requested = set(str(tag) for tag in selected_tags)
    return [tag for tag in all_tags if tag in requested]


def audit_paired_zero(
    arms: list[dict[str, Any]],
    *,
    selected_tags: Iterable[str] | None = None,
    strict: bool = False,
) -> tuple[dict[str, str], list[str]]:
    """Validate paired-zero declarations and return per-arm statuses.

    BCRES/BCFT/TRRES_CMD/TRRES_ACT must name a zero arm from the same selected
    evaluation set.  A non-residual PE2E arm is explicitly unpaired instead of
    being forced to use a cross-architecture baseline.
    """

    by_tag = {_tag(arm): arm for arm in arms}
    selected = _selected(arms, selected_tags)
    statuses: dict[str, str] = {}
    errors: list[str] = []
    for tag in selected:
        arm = by_tag[tag]
        if is_zero_arm(arm):
            statuses[tag] = BASELINE
            continue
        composition = str(arm.get("learning_composition", "") or "")
        control_contract = str(arm.get("control_contract", "") or "")
        paired_tag = arm.get("paired_zero_tag")
        declared = str(arm.get("zero_baseline_status", "") or "")
        if control_contract == PE2E_CONTROL_CONTRACT or composition == "PE2E":
            if paired_tag or declared == PAIRED:
                errors.append(
                    f"INVALID_ZERO_BASELINE_PROTOCOL:PE2E arm {tag} must not claim a paired zero"
                )
                statuses[tag] = "INVALID_ZERO_BASELINE_PROTOCOL"
            else:
                statuses[tag] = UNPAIRED
            continue
        if composition in PAIR_REQUIRED_COMPOSITIONS:
            if declared == UNPAIRED:
                errors.append(
                    f"INVALID_ZERO_BASELINE_PROTOCOL:arm {tag} may not declare "
                    f"{UNPAIRED} for {composition}"
                )
            if not paired_tag:
                errors.append(f"INVALID_INCOMPLETE_EVAL:arm {tag} missing paired_zero_tag")
                statuses[tag] = "INVALID_MISSING_PAIRED_ZERO"
                continue
            zero_tag = str(paired_tag)
            zero = by_tag.get(zero_tag)
            if zero_tag not in selected or zero is None:
                errors.append(
                    f"INVALID_INCOMPLETE_EVAL:arm {tag} paired zero {zero_tag} is not selected"
                )
                statuses[tag] = "INVALID_MISSING_PAIRED_ZERO"
                continue
            if not is_zero_arm(zero):
                errors.append(
                    f"INVALID_ZERO_BASELINE_PROTOCOL:arm {tag} points to non-zero arm {zero_tag}"
                )
                statuses[tag] = "INVALID_ZERO_BASELINE_PROTOCOL"
                continue
            zero_for = zero.get("zero_for")
            if zero_for is not None:
                allowed = {zero_for} if isinstance(zero_for, str) else set(zero_for)
                if tag not in allowed:
                    errors.append(
                        f"INVALID_ZERO_BASELINE_PROTOCOL:zero {zero_tag} does not declare "
                        f"zero_for arm {tag}"
                    )
                    statuses[tag] = "INVALID_ZERO_BASELINE_PROTOCOL"
                    continue
            arm_set = arm.get("evaluation_set_id")
            zero_set = zero.get("evaluation_set_id")
            if arm_set is not None and zero_set is not None and str(arm_set) != str(zero_set):
                errors.append(
                    f"INVALID_ZERO_BASELINE_PROTOCOL:arm {tag} and zero {zero_tag} "
                    "use different evaluation_set_id"
                )
                statuses[tag] = "INVALID_ZERO_BASELINE_PROTOCOL"
                continue
            statuses[tag] = PAIRED
            continue
        if declared == UNPAIRED:
            statuses[tag] = UNPAIRED
        elif paired_tag:
            statuses[tag] = PAIRED
        else:
            statuses[tag] = UNKNOWN
            if strict:
                errors.append(
                    f"INVALID_ZERO_BASELINE_PROTOCOL:arm {tag} missing learning_composition"
                )
    return statuses, errors
