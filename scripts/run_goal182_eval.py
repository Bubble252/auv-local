#!/usr/bin/env python3
"""Run the Goal182 evaluation protocol, build canonical reports, and audit them.

Non-responsibility: this launcher does not alter evaluator semantics, matrix
definitions, pressure thresholds, or checkpoint selection.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

from auv_lowlevel.eval_pairing import audit_paired_zero

REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNER = REPO_ROOT / "scripts" / "run_pc_standing_eval_suite.py"
BUILDER = REPO_ROOT / "scripts" / "build_goal182_eval_report.py"
AUDITOR = REPO_ROOT / "scripts" / "audit_goal182_artifacts.py"
MATRIX_CHECKER = REPO_ROOT / "scripts" / "check_goal182_eval_matrix.py"
DEFAULT_MATRIX = REPO_ROOT / "configs" / "governance" / "goal182_eval_matrix.json"
DEFAULT_RADIUS_LABELS = {"default", "2p0", "2.0"}
DEFAULT_RADIUS_M = 2.0  # magic-number-ok: Goal182 active evaluation fixes plant.goal_radius_m


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--arms", type=Path, required=True)
    parser.add_argument("--eval-matrix", type=Path, default=DEFAULT_MATRIX)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--matrix-id", required=True)
    parser.add_argument("--tier", required=True)
    parser.add_argument("--phase", required=True)
    parser.add_argument("--goal-id", default="Goal182")
    parser.add_argument("--training-profile", default="PPO-FLAT-4096x48-E5-MB4")
    parser.add_argument("--tags", nargs="*", default=None)
    parser.add_argument("--scenarios", nargs="*", default=None)
    parser.add_argument("--seed-start", type=int, default=None)
    parser.add_argument("--seed-count", type=int, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument("--gpus", default="")
    parser.add_argument("--max-parallel", type=int, default=4)
    parser.add_argument("--timeout-s", type=float, default=None)
    parser.add_argument("--sb3-root", type=Path, default=None)
    parser.add_argument("--zero-tag", default="zero")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--legacy-analyze", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def _run(command: list[str]) -> int:
    print("+", " ".join(command), flush=True)
    return subprocess.run(command, cwd=REPO_ROOT, check=False).returncode


def _runner_command(args: argparse.Namespace) -> list[str]:
    command = [
        args.python,
        str(RUNNER),
        "--manifest",
        str(_resolve(args.manifest)),
        "--arms",
        str(_resolve(args.arms)),
        "--output-root",
        str(_resolve(args.output_root)),
        "--python",
        args.python,
        "--device",
        args.device,
        "--dtype",
        args.dtype,
        "--gpus",
        args.gpus,
        "--max-parallel",
        str(args.max_parallel),
        "--zero-tag",
        args.zero_tag,
    ]
    for flag, value in (
        ("--seed-start", args.seed_start),
        ("--seed-count", args.seed_count),
        ("--timeout-s", args.timeout_s),
        ("--sb3-root", None if args.sb3_root is None else _resolve(args.sb3_root)),
    ):
        if value is not None:
            command.extend([flag, str(value)])
    for flag, values in (("--tags", args.tags), ("--scenarios", args.scenarios)):
        if values:
            command.append(flag)
            command.extend(str(value) for value in values)
    if args.overwrite:
        command.append("--overwrite")
    if args.skip_existing:
        command.append("--skip-existing")
    if args.dry_run:
        command.append("--dry-run")
    if not args.legacy_analyze:
        command.append("--no-analyze")
    return command


def _build_report_command(args: argparse.Namespace) -> list[str]:
    return [
        args.python,
        str(BUILDER),
        "--run-manifest",
        str(_resolve(args.output_root) / "run_manifest.json"),
        "--output-root",
        str(_resolve(args.output_root)),
        "--goal-id",
        args.goal_id,
        "--matrix-id",
        args.matrix_id,
        "--tier",
        args.tier,
        "--phase",
        args.phase,
        "--training-profile",
        args.training_profile,
    ]


def _job_count(output_root: Path) -> int:
    payload = json.loads((output_root / "run_manifest.json").read_text(encoding="utf-8"))
    return int(payload["job_count"])


def _matrix_errors(args: argparse.Namespace) -> list[str]:
    payload = json.loads(_resolve(args.eval_matrix).read_text(encoding="utf-8"))
    errors: list[str] = []
    if payload.get("matrix_id") != args.matrix_id:
        errors.append("eval matrix id does not match --matrix-id")
    heldout = payload.get("heldout")
    if not isinstance(heldout, dict) or heldout.get("enabled") is not False:
        errors.append("held-out must remain closed")
    tiers = payload.get("tiers")
    if not isinstance(tiers, list):
        return errors + ["eval matrix tiers must be a list"]
    tier = next((item for item in tiers if item.get("tier_id") == args.tier), None)
    if tier is None:
        errors.append(f"tier {args.tier!r} is not registered")
    elif "formal" in args.phase.lower():
        if payload.get("matrix_status") != "FROZEN" or tier.get("status") != "FROZEN":
            errors.append("formal evaluation requires a FROZEN matrix and tier")
    return errors


def _manifest_radius_errors(path: Path) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    errors: list[str] = []
    axes = payload.get("axes")
    if isinstance(axes, dict):
        radii = axes.get("goal_radii")
        if isinstance(radii, list):
            for radius in radii:
                if str(radius) not in DEFAULT_RADIUS_LABELS:
                    errors.append(f"retired or unknown goal radius label in manifest: {radius!r}")
    scenarios = payload.get("scenarios")
    if isinstance(scenarios, list):
        for scenario in scenarios:
            if not isinstance(scenario, dict):
                continue
            scenario_axes = scenario.get("axes")
            axes = scenario_axes if isinstance(scenario_axes, dict) else {}
            label = axes.get("goal_radius", scenario.get("goal_radius"))
            radius_m = axes.get("goal_radius_m", scenario.get("goal_radius_m"))
            if label is None and radius_m is None:
                matched = re.search(r"(?:^|__)g-([^_]+)(?:__|$)", str(scenario.get("id", "")))
                if matched:
                    label = matched.group(1)
            if label is None and radius_m is None:
                errors.append(
                    f"scenario {scenario.get('id')} is missing goal radius; "
                    "Goal182 active evaluation requires 2.0 m"
                )
                continue
            if label is not None and str(label) not in DEFAULT_RADIUS_LABELS:
                errors.append(f"retired goal radius in scenario {scenario.get('id')}: {label!r}")
            if radius_m is not None:
                try:
                    radius_value = float(radius_m)
                except (TypeError, ValueError):
                    errors.append(
                        f"goal_radius_m is not numeric in {scenario.get('id')}: {radius_m!r}"
                    )
                else:
                    if abs(radius_value - DEFAULT_RADIUS_M) > 1e-9:
                        errors.append(
                            f"goal_radius_m must be 2.0 in {scenario.get('id')}; got {radius_m!r}"
                        )
    return errors


def _tier_axes(path: Path, tier_id: str) -> tuple[list[float], list[str]] | None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    tiers = payload.get("tiers")
    if not isinstance(tiers, list):
        return None
    tier = next((item for item in tiers if item.get("tier_id") == tier_id), None)
    if not isinstance(tier, dict):
        return None
    depths = [float(value) for value in tier.get("depths_m", [])]
    routes = [str(value) for value in tier.get("route_profiles", [])]
    return depths, routes


def _manifest_depth_errors(
    path: Path, *, eval_matrix: Path, tier_id: str, formal: bool
) -> list[str]:
    if tier_id != "DEPTH_GENERALIZATION":
        return []
    expected = _tier_axes(eval_matrix, tier_id)
    if expected is None:
        return ["DEPTH_GENERALIZATION axes are missing from the eval matrix"]
    expected_depths, expected_routes = expected
    payload = json.loads(path.read_text(encoding="utf-8"))
    errors: list[str] = []
    axes = payload.get("axes")
    if isinstance(axes, dict):
        manifest_depths = axes.get("depth_profiles_m")
        if manifest_depths is not None:
            actual = [float(value) for value in manifest_depths]
            if len(actual) != len(expected_depths) or any(
                abs(a - b) > 1e-9 for a, b in zip(actual, expected_depths, strict=True)
            ):
                errors.append(f"manifest depth_profiles_m mismatch: {actual} != {expected_depths}")
    scenarios = payload.get("scenarios")
    if not isinstance(scenarios, list) or not scenarios:
        return errors + ["manifest scenarios must be a non-empty list"]
    found_depths: set[float] = set()
    found_routes: set[str] = set()
    for scenario in scenarios:
        if not isinstance(scenario, dict):
            continue
        sid = str(scenario.get("id", ""))
        scenario_axes = scenario.get("axes")
        axes = scenario_axes if isinstance(scenario_axes, dict) else {}
        depth = axes.get("depth_m", scenario.get("depth_m"))
        route = axes.get("route_profile", scenario.get("route_profile"))
        if depth is None:
            matched = re.search(r"(?:^|__)d([0-9p.]+)(?:__|$)", sid)
            if matched:
                depth = matched.group(1).replace("p", ".")
        if route is None:
            matched = re.search(r"(?:^|__)r-(.+?)(?=__|$)", sid)
            if matched:
                route = matched.group(1)
        if depth is None:
            errors.append(f"scenario {sid} is missing depth_m")
            continue
        try:
            depth_value = float(depth)
        except (TypeError, ValueError):
            errors.append(f"scenario {sid} depth_m is not numeric: {depth!r}")
            continue
        if not any(abs(depth_value - value) <= 1e-9 for value in expected_depths):
            errors.append(f"scenario {sid} depth_m {depth_value} is not registered")
        found_depths.add(depth_value)
        if route is None:
            errors.append(f"scenario {sid} is missing route_profile")
        else:
            found_routes.add(str(route))
    if formal:
        missing_depths = [
            value
            for value in expected_depths
            if not any(abs(value - actual) <= 1e-9 for actual in found_depths)
        ]
        missing_routes = [value for value in expected_routes if value not in found_routes]
        if missing_depths:
            errors.append(f"manifest is missing depths {missing_depths}")
        if missing_routes:
            errors.append(f"manifest is missing routes {missing_routes}")
    return errors


def _arm_pairing_errors(path: Path, *, selected_tags: list[str] | None, strict: bool) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    arms = payload.get("arms")
    if not isinstance(arms, list) or not arms:
        return ["arms manifest must contain a non-empty arms list"]
    _statuses, errors = audit_paired_zero(
        arms,
        selected_tags=selected_tags,
        strict=strict,
    )
    return errors


def main() -> int:
    args = _parse_args()
    output_root = _resolve(args.output_root)
    errors = _matrix_errors(args)
    if errors:
        for error in errors:
            print(f"Goal182 eval launcher: {error}", file=sys.stderr)
        return 2
    radius_errors = _manifest_radius_errors(_resolve(args.manifest))
    if radius_errors:
        for error in radius_errors:
            print(f"Goal182 eval launcher: {error}", file=sys.stderr)
        return 2
    formal = "formal" in args.phase.lower()
    depth_errors = _manifest_depth_errors(
        _resolve(args.manifest),
        eval_matrix=_resolve(args.eval_matrix),
        tier_id=args.tier,
        formal=formal,
    )
    if depth_errors:
        for error in depth_errors:
            print(f"Goal182 eval launcher: {error}", file=sys.stderr)
        return 2
    pairing_errors = _arm_pairing_errors(
        _resolve(args.arms),
        selected_tags=args.tags,
        strict=formal,
    )
    if pairing_errors:
        for error in pairing_errors:
            print(f"Goal182 eval launcher: {error}", file=sys.stderr)
        return 2
    matrix_check = [
        args.python,
        str(MATRIX_CHECKER),
        "--matrix",
        str(_resolve(args.eval_matrix)),
    ]
    if "formal" in args.phase.lower():
        matrix_check.append("--require-frozen")
    returncode = _run(matrix_check)
    if returncode != 0:
        return returncode
    returncode = _run(_runner_command(args))
    if returncode != 0 or args.dry_run:
        return returncode
    report_returncode = _run(_build_report_command(args))
    audit_returncode = _run(
        [
            args.python,
            str(AUDITOR),
            "--eval-set-root",
            str(output_root),
            "--expected-runs",
            str(_job_count(output_root)),
        ]
    )
    if report_returncode != 0 or audit_returncode != 0:
        return 1
    print(f"Goal182 evaluation protocol complete: {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
