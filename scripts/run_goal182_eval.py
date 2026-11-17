#!/usr/bin/env python3
"""Run the Goal182 evaluation protocol, build canonical reports, and audit them.

Non-responsibility: this launcher does not alter evaluator semantics, matrix
definitions, pressure thresholds, or checkpoint selection.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNER = REPO_ROOT / "scripts" / "run_pc_standing_eval_suite.py"
BUILDER = REPO_ROOT / "scripts" / "build_goal182_eval_report.py"
AUDITOR = REPO_ROOT / "scripts" / "audit_goal182_artifacts.py"
MATRIX_CHECKER = REPO_ROOT / "scripts" / "check_goal182_eval_matrix.py"
DEFAULT_MATRIX = REPO_ROOT / "configs" / "governance" / "goal182_eval_matrix.json"


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


def main() -> int:
    args = _parse_args()
    output_root = _resolve(args.output_root)
    errors = _matrix_errors(args)
    if errors:
        for error in errors:
            print(f"Goal182 eval launcher: {error}", file=sys.stderr)
        return 2
    matrix_check = [
        args.python,
        str(MATRIX_CHECKER),
        "--matrix",
        str(_resolve(args.eval_matrix)),
    ]
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
