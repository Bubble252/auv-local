#!/usr/bin/env python3
"""Fail closed on missing Goal171 curriculum provenance or evaluation cells."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = REPO_ROOT / "configs" / "lowlevel" / "goal171" / "goal171_manifest.json"
CURRICULUM_SCHEMA = "auv_pc_curriculum_episode_manifest.v1"


class AuditError(RuntimeError):
    def __init__(self, errors: Iterable[str]) -> None:
        self.errors = list(errors)
        super().__init__("\n".join(self.errors))


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{line_number}: expected a JSON object")
        rows.append(row)
    return rows


def _protocol() -> tuple[dict[str, Any], tuple[tuple[str, float], ...], tuple[str, ...]]:
    manifest = _read_json(MANIFEST_PATH)
    protocol = manifest["protocol"]
    stage_mix = tuple(
        (str(item["stage"]), float(item["weight"]))
        for item in protocol["curriculum"]["stage_mix"]
    )
    arms = tuple(str(item["arm"]) for item in manifest["arms"])
    return protocol, stage_mix, arms


def _normalise_mix(value: Any) -> tuple[tuple[str, float], ...]:
    if not isinstance(value, list):
        raise ValueError("stage_mix must be a list")
    result: list[tuple[str, float]] = []
    for item in value:
        if isinstance(item, dict):
            result.append((str(item["stage"]), float(item["weight"])))
        elif isinstance(item, (list, tuple)) and len(item) == 2:
            result.append((str(item[0]), float(item[1])))
        else:
            raise ValueError(f"invalid stage_mix entry: {item!r}")
    return tuple(result)


def _check_training_manifest(
    path: Path,
    *,
    stage_mix: tuple[tuple[str, float], ...],
    expected_promotion_enabled: bool | None = None,
    errors: list[str],
) -> dict[str, Any] | None:
    try:
        manifest = _read_json(path)
    except Exception as exc:
        errors.append(f"{path}: cannot read training manifest: {exc}")
        return None
    curriculum = manifest.get("curriculum")
    if not isinstance(curriculum, dict):
        errors.append(f"{path}: missing curriculum block")
        return manifest
    if curriculum.get("enabled") is not True:
        errors.append(f"{path}: curriculum.enabled is not true")
    if expected_promotion_enabled is not None:
        if curriculum.get("promotion_enabled") is not expected_promotion_enabled:
            errors.append(
                f"{path}: curriculum.promotion_enabled is "
                f"{curriculum.get('promotion_enabled')!r}, expected "
                f"{expected_promotion_enabled!r}"
            )
        if manifest.get("curriculum_promotion_enabled") is not expected_promotion_enabled:
            errors.append(f"{path}: top-level curriculum_promotion_enabled drifted")
    if curriculum.get("initial_stage") != "C3":
        errors.append(f"{path}: curriculum.initial_stage != C3")
    try:
        observed_mix = _normalise_mix(curriculum.get("stage_mix"))
    except Exception as exc:
        errors.append(f"{path}: invalid curriculum.stage_mix: {exc}")
    else:
        if observed_mix != stage_mix:
            errors.append(
                f"{path}: curriculum.stage_mix drifted: {observed_mix!r} != {stage_mix!r}"
            )
    if curriculum.get("manifest_filename") != "curriculum_episodes.jsonl":
        errors.append(f"{path}: curriculum.manifest_filename drifted")
    status = manifest.get("status")
    if status is not None and status != "completed":
        errors.append(f"{path}: training status is {status!r}, expected 'completed'")
    return manifest


def _check_curriculum_state(
    path: Path,
    *,
    initial_stage: str,
    stage_mix: tuple[tuple[str, float], ...],
    expected_promotion_enabled: bool | None,
    errors: list[str],
) -> None:
    if expected_promotion_enabled is not False:
        return
    try:
        state = _read_json(path)
    except Exception as exc:
        errors.append(f"{path}: cannot read frozen curriculum state: {exc}")
        return
    if state.get("promotion_enabled") is not False:
        errors.append(f"{path}: curriculum promotion is not disabled")
    if state.get("frozen_stage") != initial_stage:
        errors.append(
            f"{path}: frozen_stage is {state.get('frozen_stage')!r}, "
            f"expected {initial_stage!r}"
        )
    if state.get("stage") != initial_stage:
        errors.append(
            f"{path}: runtime curriculum stage is {state.get('stage')!r}, "
            f"expected frozen {initial_stage!r}"
        )
    try:
        observed_mix = _normalise_mix(state.get("stage_mix"))
    except Exception as exc:
        errors.append(f"{path}: invalid runtime stage_mix: {exc}")
    else:
        if observed_mix != stage_mix:
            errors.append(
                f"{path}: runtime stage_mix drifted: {observed_mix!r} != {stage_mix!r}"
            )
    if state.get("finalized") is not True:
        errors.append(f"{path}: curriculum state was not finalized")
    completed = state.get("completed_transitions")
    if isinstance(completed, bool) or not isinstance(completed, int) or completed <= 0:
        errors.append(f"{path}: invalid completed_transitions {completed!r}")
    decisions = path.parent / "curriculum_decisions.jsonl"
    if decisions.is_file() and any(
        line.strip() for line in decisions.read_text(encoding="utf-8").splitlines()
    ):
        errors.append(f"{decisions}: frozen curriculum produced promotion decisions")


def _check_curriculum_episodes(
    path: Path,
    *,
    protocol: dict[str, Any],
    stage_mix: tuple[tuple[str, float], ...],
    min_episodes: int,
    errors: list[str],
) -> dict[str, Any] | None:
    try:
        rows = _read_jsonl(path)
    except Exception as exc:
        errors.append(f"{path}: cannot read curriculum episodes: {exc}")
        return None
    if len(rows) < min_episodes:
        errors.append(
            f"{path}: only {len(rows)} curriculum episodes, expected at least {min_episodes}"
        )
    if not rows:
        return None

    expected_stages = {stage for stage, _weight in stage_mix}
    counts = Counter(str(row.get("stage")) for row in rows)
    missing = sorted(expected_stages - set(counts))
    unknown = sorted(set(counts) - expected_stages)
    if missing:
        errors.append(f"{path}: missing curriculum stages {missing}")
    if unknown:
        errors.append(f"{path}: unexpected curriculum stages {unknown}")

    max_depth_m = float(protocol["max_depth_m"])
    max_speed_mps = float(protocol["max_ground_speed_mps"])
    worst_depth = 0.0
    worst_speed = 0.0
    for index, row in enumerate(rows, 1):
        if row.get("schema_version") != CURRICULUM_SCHEMA:
            errors.append(f"{path}:{index}: unexpected schema_version")
            break
        try:
            depth = float(row["depth_m"])
            speed = float(row["speed_mps"])
            current = float(row["current_speed_mps"])
        except (KeyError, TypeError, ValueError) as exc:
            errors.append(f"{path}:{index}: invalid depth/speed fields: {exc}")
            continue
        if not all(math.isfinite(value) for value in (depth, speed, current)):
            errors.append(f"{path}:{index}: non-finite depth/speed")
            continue
        worst_depth = max(worst_depth, depth)
        worst_speed = max(worst_speed, speed + current)
        if depth < 0.0 or depth > max_depth_m + 1e-9:
            errors.append(f"{path}:{index}: depth {depth:.3f} outside [0, {max_depth_m}]")
            break
        if speed + current > max_speed_mps + 1e-9:
            errors.append(
                f"{path}:{index}: speed+current {speed + current:.3f} exceeds "
                f"{max_speed_mps:.3f} m/s"
            )
            break

    total = len(rows)
    shares: dict[str, float] = {}
    for stage, target in stage_mix:
        observed = counts[stage] / total
        shares[stage] = observed
        tolerance = max(0.04, 4.0 * math.sqrt(target * (1.0 - target) / total))
        if abs(observed - target) > tolerance:
            errors.append(
                f"{path}: stage {stage} share {observed:.3f} drifted from "
                f"{target:.3f} (tolerance {tolerance:.3f})"
            )
    return {
        "episodes": total,
        "shares": shares,
        "worst_depth_m": worst_depth,
        "worst_speed_mps": worst_speed,
    }


def _check_training_log(path: Path, manifest: dict[str, Any] | None, errors: list[str]) -> None:
    if manifest is None:
        return
    records = manifest.get("iteration_records")
    if isinstance(records, list) and records:
        lengths = [
            float(record["mean_episode_length"])
            for record in records
            if isinstance(record, dict) and record.get("mean_episode_length") is not None
        ]
        if lengths and max(lengths) <= 2.0:
            errors.append(
                f"{path}: all mean_episode_length values are <=2.0; "
                "the curriculum is terminating immediately"
            )


def audit_train(
    *,
    root: Path,
    seeds: tuple[int, ...],
    min_episodes: int,
) -> list[str]:
    protocol, stage_mix, arms = _protocol()
    curriculum_protocol = protocol["curriculum"]
    expected_promotion_enabled = curriculum_protocol.get("promotion_enabled")
    initial_stage = str(curriculum_protocol["initial_stage"])
    errors: list[str] = []
    summaries: list[str] = []
    for arm in arms:
        for seed in seeds:
            run_dir = root / "train" / arm / f"seed{seed}"
            checkpoints = [run_dir / "final.pt", run_dir / "final.zip"]
            if not any(path.is_file() for path in checkpoints):
                errors.append(f"{run_dir}: missing final.pt/final.zip")
            manifest_path = run_dir / "training_manifest.json"
            manifest = _check_training_manifest(
                manifest_path,
                stage_mix=stage_mix,
                expected_promotion_enabled=(
                    None
                    if expected_promotion_enabled is None
                    else bool(expected_promotion_enabled)
                ),
                errors=errors,
            )
            curriculum_path = run_dir / protocol["curriculum"]["manifest_filename"]
            curriculum = _check_curriculum_episodes(
                curriculum_path,
                protocol=protocol,
                stage_mix=stage_mix,
                min_episodes=min_episodes,
                errors=errors,
            )
            _check_curriculum_state(
                run_dir / "curriculum_state.json",
                initial_stage=initial_stage,
                stage_mix=stage_mix,
                expected_promotion_enabled=(
                    None
                    if expected_promotion_enabled is None
                    else bool(expected_promotion_enabled)
                ),
                errors=errors,
            )
            _check_training_log(run_dir, manifest, errors)
            if curriculum is not None:
                summaries.append(
                    f"{arm}/seed{seed}: episodes={curriculum['episodes']} "
                    f"shares={curriculum['shares']} "
                    f"depth_max={curriculum['worst_depth_m']:.1f} "
                    f"speed_max={curriculum['worst_speed_mps']:.3f}"
                )
    if summaries:
        print("Goal171 curriculum summary:")
        for summary in summaries:
            print(f"  {summary}")
    return errors


def _episode_seeds(summary: dict[str, Any], episodes_path: Path) -> list[int]:
    episodes = summary.get("episodes")
    if not isinstance(episodes, list):
        if not episodes_path.is_file():
            raise ValueError("missing episodes.json and summary.episodes")
        episodes = _read_json(episodes_path)
    if not isinstance(episodes, list):
        raise ValueError("episodes payload is not a list")
    seeds: list[int] = []
    for episode in episodes:
        if not isinstance(episode, dict) or "seed" not in episode:
            raise ValueError("episode is missing seed")
        seeds.append(int(episode["seed"]))
    return seeds


def _check_eval_run(
    run_dir: Path,
    *,
    expected_seeds: tuple[int, ...],
    errors: list[str],
) -> bool:
    summary_path = run_dir / "summary.json"
    if not summary_path.is_file():
        errors.append(f"{run_dir}: missing summary.json")
        return False
    try:
        summary = _read_json(summary_path)
        seeds = _episode_seeds(summary, run_dir / "episodes.json")
    except Exception as exc:
        errors.append(f"{run_dir}: cannot read evaluation payload: {exc}")
        return False
    if sorted(seeds) != list(expected_seeds):
        errors.append(
            f"{run_dir}: seed set mismatch: {seeds[:3]}...{seeds[-3:]} != "
            f"{expected_seeds[0]}..{expected_seeds[-1]}"
        )
    trace_candidates = [run_dir / "trace.npz", run_dir.parent / f"{run_dir.name}.npz"]
    if not any(path.is_file() for path in trace_candidates):
        errors.append(f"{run_dir}: missing trace.npz")
    return True


def audit_eval(
    *,
    root: Path,
    seeds: tuple[int, ...],
    seed_start: int,
    seed_count: int,
) -> list[str]:
    protocol, _stage_mix, arms = _protocol()
    if seed_start != int(protocol["dev_seed_start"]) or seed_count != int(
        protocol["dev_seed_count"]
    ):
        raise AuditError(["evaluation dev seed range does not match the manifest"])
    expected_seeds = tuple(range(seed_start, seed_start + seed_count))
    expected_checkpoint_seeds = seeds or (int(_read_json(MANIFEST_PATH)["train_seed"]),)
    levels = tuple(str(level) for level in protocol["levels"])
    radii = tuple((str(name), float(value)) for name, value in protocol["radii"].items())
    zero_arms = ("zero_sac50", "zero_bc50", "zero_alos50")
    errors: list[str] = []
    completed = 0
    expected = 0
    for radius, _value in radii:
        for level in levels:
            for zero_arm in zero_arms:
                expected += 1
                run_dir = root / "eval" / zero_arm / radius / level / "zero_seed86201"
                if _check_eval_run(run_dir, expected_seeds=expected_seeds, errors=errors):
                    completed += 1
            for arm in arms:
                for seed in expected_checkpoint_seeds:
                    expected += 1
                    run_dir = (
                        root
                        / "eval"
                        / arm
                        / radius
                        / level
                        / f"policy_seed{seed}"
                    )
                    if _check_eval_run(
                        run_dir, expected_seeds=expected_seeds, errors=errors
                    ):
                        completed += 1
    if len(expected_checkpoint_seeds) == 1 and expected != int(
        protocol["expected_total_eval_runs"]
    ):
        errors.append(
            f"evaluation audit enumerated {expected} cells, expected "
            f"{protocol['expected_total_eval_runs']}"
        )
    print(f"Goal171 evaluation cells present: {completed}/{expected}")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("train", "eval"), required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--seeds", default="86201")
    parser.add_argument("--min-episodes", type=int, default=100)
    parser.add_argument("--seed-start", type=int, default=20)
    parser.add_argument("--seed-count", type=int, default=40)
    args = parser.parse_args()
    root = args.root if args.root.is_absolute() else REPO_ROOT / args.root
    seeds = tuple(int(item) for item in args.seeds.split())
    if args.phase == "train":
        errors = audit_train(
            root=root,
            seeds=seeds,
            min_episodes=args.min_episodes,
        )
    else:
        errors = audit_eval(
            root=root,
            seeds=seeds,
            seed_start=args.seed_start,
            seed_count=args.seed_count,
        )
    if errors:
        raise AuditError(errors)
    print(f"Goal171 {args.phase} artifact audit OK: {root}")


if __name__ == "__main__":
    try:
        main()
    except AuditError as exc:
        print("Goal171 artifact audit FAILED:", file=sys.stderr)
        for error in exc.errors:
            print(f"  - {error}", file=sys.stderr)
        raise SystemExit(1) from None
