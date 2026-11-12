"""Evaluation-only success semantics for PC terminal outcomes.

The deployment success frame is the p_nav tight-radius GOAL termination.  The
research primary is whether true ``eta`` ever entered the tight radius.  The
``eta`` frame is reported offline only and must never affect rollout, reward,
termination, or checkpoint selection.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Literal, cast

SuccessFrame = Literal["p_nav", "eta", "both"]
SUCCESS_FRAMES: tuple[SuccessFrame, ...] = ("p_nav", "eta", "both")
DEFAULT_SUCCESS_FRAME: SuccessFrame = "both"


def validate_success_frame(frame: str) -> SuccessFrame:
    """Return a validated success frame or raise a clear error."""

    if frame not in SUCCESS_FRAMES:
        raise ValueError(
            f"unsupported success frame {frame!r}; expected one of {SUCCESS_FRAMES}"
        )
    return cast(SuccessFrame, frame)


def score_success_contract(
    *,
    p_nav_tight_success: bool,
    eta_tight_ever: bool,
    eta_tight_at_done: bool,
    success_frame: str = DEFAULT_SUCCESS_FRAME,
    eta_done_distance_m: float | None = None,
    p_nav_done_distance_m: float | None = None,
    goal_radius_m: float | None = None,
) -> dict[str, Any]:
    """Return the stable per-episode success fields for one evaluation."""

    frame = validate_success_frame(success_frame)
    p_nav = bool(p_nav_tight_success)
    eta_ever = bool(eta_tight_ever)
    eta_done = bool(eta_tight_at_done)
    selected: bool | None
    if frame == "p_nav":
        selected = p_nav
    elif frame == "eta":
        selected = eta_ever
    else:
        selected = None
    result = {
        "success_frame": frame,
        "success_pnav_tight": p_nav,
        "success_eta_tight_ever": eta_ever,
        "success_eta_tight_at_done": eta_done,
        "success_selected": selected,
        "success_frame_gap_eta_minus_pnav": float(eta_ever) - float(p_nav),
    }
    if eta_done_distance_m is not None:
        result["eta_done_distance_m"] = float(eta_done_distance_m)
    if p_nav_done_distance_m is not None:
        result["p_nav_done_distance_m"] = float(p_nav_done_distance_m)
    if goal_radius_m is not None and eta_done_distance_m is not None:
        result["success_margin_m"] = float(goal_radius_m) - float(eta_done_distance_m)
    return result


def summarize_success_contract(
    rows: Sequence[Mapping[str, Any]],
    *,
    success_frame: str = DEFAULT_SUCCESS_FRAME,
) -> dict[str, Any]:
    """Aggregate the frozen success frames without changing control semantics."""

    frame = validate_success_frame(success_frame)
    if not rows:
        raise ValueError("success contract summary requires at least one row")
    contract_rows: list[Mapping[str, Any]] = []
    for row in rows:
        if "success_pnav_tight" in row:
            contract_rows.append(row)
            continue
        enriched = dict(row)
        enriched.update(
            score_success_contract(
                p_nav_tight_success=bool(
                    row.get("success", row.get("goal_capture_pnav", False))
                ),
                eta_tight_ever=bool(
                    row.get("goal_capture_eta", row.get("terminal_eta_capture", False))
                ),
                eta_tight_at_done=bool(
                    row.get(
                        "goal_capture_eta_at_done",
                        row.get("terminal_eta_capture", False),
                    )
                ),
                success_frame=frame,
                eta_done_distance_m=row.get("terminal_eta_distance_m"),
                p_nav_done_distance_m=row.get("terminal_p_nav_distance_m"),
            )
        )
        contract_rows.append(enriched)
    rows = contract_rows
    p_nav = [bool(row["success_pnav_tight"]) for row in rows]
    eta_ever = [bool(row["success_eta_tight_ever"]) for row in rows]
    eta_done = [bool(row["success_eta_tight_at_done"]) for row in rows]
    selected_values = [
        bool(value)
        for row in rows
        if (value := row.get("success_selected")) is not None
    ]
    p_nav_count = sum(p_nav)
    eta_count = sum(eta_ever)
    done_count = sum(eta_done)
    episode_count = len(rows)
    summary = {
        "success_frame": frame,
        "success_rate": p_nav_count / episode_count,
        "success_count_pnav_tight": p_nav_count,
        "success_rate_pnav_tight": p_nav_count / episode_count,
        "success_count_eta_tight_ever": eta_count,
        "success_rate_eta_tight_ever": eta_count / episode_count,
        "success_count_eta_tight_at_done": done_count,
        "success_rate_eta_tight_at_done": done_count / episode_count,
        "success_frame_gap_eta_minus_pnav": (eta_count - p_nav_count)
        / episode_count,
        "success_rate_selected": (
            None if frame == "both" else sum(selected_values) / episode_count
        ),
    }
    for name in ("eta_done_distance_m", "p_nav_done_distance_m", "success_margin_m"):
        values = [float(row[name]) for row in rows if row.get(name) is not None]
        if values:
            summary[f"{name}_mean"] = float(sum(values) / len(values))
    return summary
