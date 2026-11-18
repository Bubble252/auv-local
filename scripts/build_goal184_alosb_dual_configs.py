#!/usr/bin/env python3
"""Build Goal184 ALOSB COMP/NOCOMP formal and calibration configs."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = REPO_ROOT / "configs" / "lowlevel" / "goal184"
CAL_ROOT = CONFIG_ROOT / "calibration"
STAGES = ("C0", "C1", "C2", "C3", "C4", "C5", "C6")
RUN_ROOT = Path(
    "/data/yhx/underwater-robotics/runs/auv-local/goal184/20261118-goal184-64d8225"
)

LINES = {
    "comp": {
        "env": CONFIG_ROOT / "env_alos_formal_empirical_comp.json",
        "bc": RUN_ROOT
        / "bc/alos_comp_bc_stage_matched_v4_teacher_alos_truth/pc_bc_actor.pt",
        "bc_sha256": "7433802d2a6c43205941da179fcb08e501f556a4a1a8c5ebd638d2b849e6f236",
    },
    "nocomp": {
        "env": CONFIG_ROOT / "env_alos_formal_empirical.json",
        "bc": RUN_ROOT
        / "bc/alos_bc_stage_matched_v4_teacher_alos_nocomp/pc_bc_actor.pt",
        "bc_sha256": "f4ccb463d5e468ad4bbdf5b6de050cc7a1d00b40f16f83e8f49fc1ebb796093a",
    },
}

BCFT_SOURCE = CONFIG_ROOT / "train_formal_mod4_pe2e_bcft_env_empirical_5d_m_seed86201.json"
BCRES_SOURCE = CONFIG_ROOT / "train_formal_mod4_pe2e_bcres_env_empirical_5d_m_seed86201.json"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _formal(line: str, module: str) -> Path:
    spec = LINES[line]
    source = BCFT_SOURCE if module == "bcft" else BCRES_SOURCE
    payload = copy.deepcopy(_load(source))
    payload["environment"]["config"] = spec["env"].relative_to(REPO_ROOT).as_posix()
    payload["pc"]["actor_command_dim"] = 3
    payload["pc"]["critic_command_dim"] = 3
    payload["training"]["iterations"] = 2600
    payload["experiment"].update(
        {
            "name": (
                "goal184_ppo_formal_mod4_alosb_"
                f"{module}_{line}_env_empirical_5d_m_seed86201"
            ),
            "output": (
                "artifacts/lowlevel/goal184_alosb_dual_20261118/train/mod4/"
                f"alosb_{module}_{line}_empirical_seed86201"
            ),
            "seed": 86201,
        }
    )
    if module == "bcft":
        payload["experiment"]["warmstart_actor"] = str(spec["bc"])
        payload["experiment"]["warmstart_actor_sha256"] = spec["bc_sha256"]
    else:
        payload["base_policy"].update(
            {
                "checkpoint": str(spec["bc"]),
                "checkpoint_sha256": spec["bc_sha256"],
                "mode": "frozen_pc_bc_residual",
                "residual_action_mask": [1.0, 1.0, 1.0],
                "residual_scale": 1.0,
            }
        )
        for key in ("warmstart_actor", "warmstart_actor_sha256"):
            payload["experiment"].pop(key, None)
    destination = CONFIG_ROOT / (
        f"train_formal_mod4_alosb_{module}_{line}_env_empirical_5d_m_seed86201.json"
    )
    _write(destination, payload)
    return destination


def _calibration_env(line: str, stage: str) -> Path:
    payload = copy.deepcopy(_load(LINES[line]["env"]))
    payload["curriculum"]["initial_stage"] = stage
    payload["curriculum"]["promotion_enabled"] = False
    payload["curriculum"]["stage_mix"] = [{"stage": stage, "weight": 1.0}]
    payload["experiment"].update(
        {
            "name": f"goal184_route_gate_calibration_env_alosb_{line}_{stage}",
            "output": (
                "artifacts/lowlevel/goal184_alosb_dual_20261118/calibration/"
                f"alosb_{line}/env_{stage}"
            ),
        }
    )
    destination = CAL_ROOT / f"stage_matched_alosb_{line}_v4_100it_fixedlr" / f"env_{stage}.json"
    _write(destination, payload)
    return destination


def _calibration_train(line: str, stage: str, env_path: Path, formal: Path) -> Path:
    payload = copy.deepcopy(_load(formal))
    payload["environment"]["config"] = env_path.relative_to(REPO_ROOT).as_posix()
    payload["algorithm"]["learning_rate"] = 1e-12
    payload["algorithm"]["schedule"] = "fixed"
    for key in (
        "learning_rate_actor",
        "learning_rate_critic",
        "learning_rate_dade_head",
    ):
        payload["algorithm"].pop(key, None)
    payload["base_policy"]["residual_scale"] = 1e-6
    payload["training"]["iterations"] = 100
    payload["runner"]["save_interval"] = 100
    payload["experiment"].update(
        {
            "name": f"goal184_route_gate_calibration_alosb_{line}_{stage}",
            "output": (
                "artifacts/lowlevel/goal184_alosb_dual_20261118/calibration/"
                f"alosb_{line}/reference_{stage}"
            ),
        }
    )
    destination = CAL_ROOT / f"stage_matched_alosb_{line}_v4_100it_fixedlr" / f"train_{stage}.json"
    _write(destination, payload)
    return destination


def main() -> int:
    files: dict[str, dict[str, dict[str, str]]] = {}
    for line in LINES:
        formal = {
            module: _formal(line, module) for module in ("bcft", "bcres")
        }
        envs = {stage: _calibration_env(line, stage) for stage in STAGES}
        trains = {
            stage: _calibration_train(line, stage, envs[stage], formal["bcft"])
            for stage in STAGES
        }
        files[line] = {
            "formal": {
                module: path.relative_to(REPO_ROOT).as_posix()
                for module, path in formal.items()
            },
            "env": {
                stage: path.relative_to(REPO_ROOT).as_posix()
                for stage, path in envs.items()
            },
            "train": {
                stage: path.relative_to(REPO_ROOT).as_posix()
                for stage, path in trains.items()
            },
        }
        manifest = {
            "schema_version": "auv_goal184_alosb_dual_configs.v1",
            "line": line,
            "env_source": LINES[line]["env"].relative_to(REPO_ROOT).as_posix(),
            "env_source_sha256": _sha256(LINES[line]["env"]),
            "bc_path": str(LINES[line]["bc"]),
            "bc_sha256": LINES[line]["bc_sha256"],
            "stages": list(STAGES),
            "iterations": 100,
            "learning_rate": 1e-12,
            "schedule": "fixed",
            "normalization_mode": "empirical",
            "sensor_profile": "unbiased_small_noise_v3",
            "files": {
                "formal": files[line]["formal"],
                "env": files[line]["env"],
                "train": files[line]["train"],
            },
            "files_sha256": {
                path: _sha256(REPO_ROOT / path)
                for group in files[line].values()
                for path in group.values()
            },
        }
        _write(
            CAL_ROOT / f"stage_matched_alosb_{line}_v4_100it_fixedlr" / "manifest.json",
            manifest,
        )
        print(f"[{line}] " + json.dumps(files[line], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
