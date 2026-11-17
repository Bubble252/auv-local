#!/usr/bin/env python3
"""Build the Goal182 standing-eval arm manifest from completed first-wave runs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
ARMS_SCHEMA = "auv_pc_standing_eval_arms.v1"
DEFAULT_SEEDS = (86201, 86202, 86203)
PE2E = "CC_PE2E"
ALOSB = "CC_ALOS_BODY"
RUN_SPECS: dict[str, dict[str, str]] = {
    "pe2e_finetune": {
        "control_contract": PE2E,
        "learning_composition": "BCFT",
        "config": "configs/lowlevel/goal182/train_formal_mod4_pe2e_finetune_5d_m_seed{seed}.json",
    },
    "pe2e_residual": {
        "control_contract": PE2E,
        "learning_composition": "BCRES",
        "config": "configs/lowlevel/goal182/train_formal_mod4_pe2e_residual_5d_m_seed{seed}.json",
    },
    "alosb_finetune": {
        "control_contract": ALOSB,
        "learning_composition": "BCFT",
        "config": "configs/lowlevel/goal182/train_formal_mod4_alosb_finetune_5d_m_seed{seed}.json",
    },
    "alosb_residual": {
        "control_contract": ALOSB,
        "learning_composition": "BCRES",
        "config": "configs/lowlevel/goal182/train_formal_mod4_alosb_residual_5d_m_seed{seed}.json",
    },
    "alosb_trres_cmd": {
        "control_contract": ALOSB,
        "learning_composition": "TRRES_CMD",
        "config": "configs/lowlevel/goal182/train_formal_mod4_alosb_trres_cmd_5d_m_seed{seed}.json",
    },
    "alosb_trres_act": {
        "control_contract": ALOSB,
        "learning_composition": "TRRES_ACT",
        "config": "configs/lowlevel/goal182/train_formal_mod4_alosb_trres_act_5d_m_seed{seed}.json",
    },
}
ZERO_SPECS = {
    "BCFT": ("alosb_finetune", "policy", "model_initial.pt", "zero_update"),
    "BCRES": ("alosb_residual", "frozen-base", "final.pt", "frozen_bc"),
    "TRRES_CMD": ("alosb_trres_cmd", "alos", "final.pt", "alos_zero_residual"),
    "TRRES_ACT": ("alosb_trres_act", "alos", "final.pt", "alos_zero_residual"),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def _relative(path: Path, repo_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(repo_root.resolve()))
    except ValueError:
        return str(path.resolve())


def _run_name(stem: str, seed: int) -> str:
    return f"{stem}_seed{seed}"


def _config_for(spec: dict[str, str], seed: int, repo_root: Path) -> Path:
    return repo_root / spec["config"].format(seed=seed)


def _checkpoint(run_dir: Path, filename: str) -> Path:
    path = run_dir / filename
    if not path.is_file():
        raise FileNotFoundError(f"missing checkpoint: {path}")
    return path


def _validate_run_manifest(run_dir: Path, checkpoint: Path, *, control: str) -> dict[str, Any]:
    manifest_path = run_dir / "training_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"missing training manifest: {manifest_path}")
    manifest = _load_json(manifest_path)
    if manifest.get("status") != "completed":
        raise ValueError(f"{manifest_path}: training status must be completed")
    if manifest.get("curriculum_final_stage") != "C6":
        raise ValueError(f"{manifest_path}: curriculum_final_stage must be C6")
    expected_sha = _sha256(checkpoint)
    recorded_sha = manifest.get("final_checkpoint_sha256")
    if recorded_sha is not None and str(recorded_sha) != expected_sha:
        raise ValueError(f"{manifest_path}: final checkpoint SHA does not match manifest")
    if control == PE2E and manifest.get("control_contract") not in {None, PE2E}:
        raise ValueError(f"{manifest_path}: control_contract does not match PE2E")
    return manifest


def build_arms(
    *,
    run_root: Path,
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
    repo_root: Path = REPO_ROOT,
    run_subdir: str = "formal",
) -> dict[str, Any]:
    run_root = run_root.resolve()
    repo_root = repo_root.resolve()
    if not seeds:
        raise ValueError("at least one training seed is required")
    formal_root = run_root / run_subdir
    policy_runs: dict[str, dict[str, Any]] = {}
    policy_tags: dict[str, list[str]] = {}
    for stem, spec in RUN_SPECS.items():
        composition = spec["learning_composition"]
        policy_tags[composition] = []
        for seed in seeds:
            name = _run_name(stem, seed)
            run_dir = formal_root / name
            config_path = _config_for(spec, seed, repo_root)
            if not config_path.is_file():
                raise FileNotFoundError(f"missing training config: {config_path}")
            config = _load_json(config_path)
            experiment = config.get("experiment")
            if not isinstance(experiment, dict):
                raise ValueError(f"{config_path}: experiment must be an object")
            if experiment.get("learning_composition") != composition:
                raise ValueError(f"{config_path}: learning_composition mismatch")
            checkpoint = _checkpoint(run_dir, "final.pt")
            manifest = _validate_run_manifest(run_dir, checkpoint, control=spec["control_contract"])
            if composition == "BCRES":
                base_policy = config.get("base_policy")
                if not isinstance(base_policy, dict):
                    raise ValueError(f"{config_path}: BCRES requires base_policy")
                base_sha = base_policy.get("checkpoint_sha256")
                recorded_base = manifest.get("frozen_bc_checkpoint_sha256")
                if recorded_base is not None and str(recorded_base) != str(base_sha):
                    raise ValueError(f"{config_path}: frozen BC SHA differs from manifest")
            tag = name
            policy_tags[composition].append(tag)
            policy_runs[tag] = {
                "tag": tag,
                "role": "policy",
                "paradigm": "PPO",
                "evaluator": "ppo",
                "training_config": _relative(config_path, repo_root),
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": _sha256(checkpoint),
                "action_mode": "policy",
                "learning_composition": composition,
                "control_contract": spec["control_contract"],
                "training_seed": seed,
                "source_run": name,
                "checkpoint_selection": "final",
                "evaluation_set_id": f"G182-MAIN-{spec['control_contract']}",
                "is_zero_baseline": False,
            }

    arms: list[dict[str, Any]] = list(policy_runs.values())
    zeros: dict[str, dict[str, Any]] = {}
    for composition, (stem, action_mode, filename, selection) in ZERO_SPECS.items():
        spec = RUN_SPECS[stem]
        zero_seed = seeds[0]
        zero_name = _run_name(stem, zero_seed)
        zero_dir = formal_root / zero_name
        zero_checkpoint = _checkpoint(zero_dir, filename)
        if composition == "BCFT":
            zero_manifest = _load_json(zero_dir / "training_manifest.json")
            recorded_initial = zero_manifest.get("initial_checkpoint_sha256")
            if recorded_initial is not None and str(recorded_initial) != _sha256(zero_checkpoint):
                raise ValueError(f"{zero_dir}: initial checkpoint SHA does not match manifest")
        config_path = _config_for(spec, zero_seed, repo_root)
        target_tags = policy_tags.get(composition, [])
        if not target_tags:
            continue
        zero_tag = f"zero_{composition.lower()}"
        zero = {
            "tag": zero_tag,
            "role": "zero_baseline",
            "paradigm": "PPO",
            "evaluator": "ppo",
            "training_config": _relative(config_path, repo_root),
            "checkpoint": str(zero_checkpoint),
            "checkpoint_sha256": _sha256(zero_checkpoint),
            "action_mode": action_mode,
            "learning_composition": composition,
            "control_contract": ALOSB,
            "training_seed": None,
            "source_run": zero_name,
            "checkpoint_selection": selection,
            "evaluation_set_id": f"G182-MAIN-{ALOSB}",
            "is_zero_baseline": True,
            "zero_for": target_tags,
        }
        zeros[zero_tag] = zero
        for tag in target_tags:
            policy_runs[tag]["paired_zero_tag"] = zero_tag
            policy_runs[tag]["zero_baseline_status"] = "PAIRED"

    for arm in arms:
        if arm["control_contract"] == PE2E:
            arm["zero_baseline_status"] = "ZERO_BASELINE_UNPAIRED"
            arm.pop("paired_zero_tag", None)
    arms = list(policy_runs.values()) + list(zeros.values())
    return {
        "schema_version": ARMS_SCHEMA,
        "goal_id": "Goal182",
        "phase": "MAIN_RANKING",
        "seeds": list(seeds),
        "policy_arms": len(policy_runs),
        "zero_arms": len(zeros),
        "arms": arms,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--run-subdir", default="formal")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    output = args.output or (args.run_root.resolve() / "eval_arms.json")
    payload = build_arms(
        run_root=args.run_root,
        seeds=tuple(args.seeds),
        repo_root=args.repo_root,
        run_subdir=args.run_subdir,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(output),
                "policy_arms": payload["policy_arms"],
                "zero_arms": payload["zero_arms"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
