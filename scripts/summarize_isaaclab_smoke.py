from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any

FILENAME_PATTERN = re.compile(r"num_envs_1_repeat_(\d+)\.json$")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the single-environment IsaacLab smoke result"
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--required-repeats", type=int, default=2)
    parser.add_argument("--required-steps", type=int, default=200)
    return parser.parse_args()


def _load_runs(directory: Path) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    for path in sorted(directory.glob("num_envs_1_repeat_*.json")):
        match = FILENAME_PATTERN.search(path.name)
        if not match:
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            payload = {"status": "FAIL_INVALID_JSON", "error": repr(exc)}
        payload["_path"] = str(path)
        payload["_repeat"] = int(match.group(1))
        runs.append(payload)
    return runs


def _is_pass(run: dict[str, Any], required_steps: int) -> bool:
    return (
        run.get("status") == "PASS"
        and int(run.get("steps_completed") or 0) >= required_steps
        and bool(run.get("all_observation_finite", False))
        and int(run.get("num_envs_actual") or 0) == 1
    )


def _summary(run: dict[str, Any]) -> dict[str, Any]:
    return {
        "repeat": run["_repeat"],
        "file": Path(run["_path"]).name,
        "status": run.get("status"),
        "steps_completed": run.get("steps_completed"),
        "observation_shape": run.get("observation_shape"),
        "action_shape": run.get("action_shape"),
        "reset_s": run.get("reset_s"),
        "first_step_s": run.get("first_step_s"),
        "time_to_first_step_s": run.get("time_to_first_step_s"),
        "all_observation_finite": run.get("all_observation_finite"),
        "reset_observation_sha256": run.get("reset_observation_sha256"),
        "num_envs_actual": run.get("num_envs_actual"),
    }


def build_smoke_result(
    runs: list[dict[str, Any]], required_repeats: int, required_steps: int
) -> dict[str, Any]:
    runs = sorted(runs, key=lambda run: run["_repeat"])
    if not runs:
        return {
            "schema": "auv_isaaclab_smoke.v1",
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "status": "FAIL_NO_DATA",
            "required_repeats": required_repeats,
            "required_steps": required_steps,
        }

    passed = [run for run in runs if _is_pass(run, required_steps)]
    hashes = [run.get("reset_observation_sha256") for run in runs]
    hash_available = all(isinstance(value, str) and value for value in hashes)
    deterministic = hash_available and len(set(hashes)) == 1
    gate = len(passed) >= required_repeats and deterministic and len(runs) >= required_repeats

    base = next((run for run in runs if run["_repeat"] == 0), runs[0])
    payload = dict(base)
    payload.pop("_path", None)
    payload.pop("_repeat", None)
    payload.update(
        {
            "schema": "auv_isaaclab_smoke.v1",
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "source_run": Path(base["_path"]).name,
            "status": "PASS" if gate else "FAIL_SMOKE",
            "base_probe_status": base.get("status"),
            "required_repeats": required_repeats,
            "required_steps": required_steps,
            "repeats_observed": len(runs),
            "repeats_passed": len(passed),
            "reset_observation_hash_available": hash_available,
            "reset_observation_deterministic_across_repeats": deterministic,
            "repeat_runs": [_summary(run) for run in runs],
        }
    )
    return payload


def main() -> int:
    args = _parse_args()
    runs = _load_runs(args.input)
    payload = build_smoke_result(runs, args.required_repeats, args.required_steps)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0 if payload["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
