#!/usr/bin/env python3
"""Audit Goal86 PPO/SAC backends, action interfaces and runtime state."""

from __future__ import annotations

import argparse
import glob
import hashlib
import importlib
import json
import os
import platform
import shutil
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "auv_pc_goal86_rl_backend_audit.v1"
REPO_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = REPO_ROOT.parent


def _discover_sb3_root() -> Path:
    candidates: list[Path] = []
    ancestor = WORKSPACE_ROOT
    for _ in range(4):
        candidates.extend(
            (
                ancestor / "third_party" / "stable-baselines3",
                ancestor / "stable-baselines3",
                ancestor / "src" / "stable-baselines3",
            )
        )
        if ancestor.parent == ancestor:
            break
        ancestor = ancestor.parent
    for candidate in candidates:
        if (candidate / "stable_baselines3" / "sac" / "sac.py").is_file():
            return candidate
    return candidates[0]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--sb3-root",
        type=Path,
        default=None,
        help="stable-baselines3 checkout; auto-discovered when omitted",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        action="append",
        default=[],
        help="PPO checkpoint to hash/read; repeatable.",
    )
    return parser.parse_args(argv)


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def _display(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _grep_sources(patterns: Sequence[str]) -> list[dict[str, Any]]:
    hits: list[dict[str, Any]] = []
    roots = (REPO_ROOT / "src", REPO_ROOT / "scripts")
    for root in roots:
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*.py")):
            try:
                lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
            except OSError:
                continue
            for lineno, line in enumerate(lines, start=1):
                if any(pattern.lower() in line.lower() for pattern in patterns):
                    hits.append(
                        {
                            "path": _display(path),
                            "line": lineno,
                            "text": line.strip()[:180],
                        }
                    )
    return hits


def _checkpoint_record(path: Path) -> dict[str, Any]:
    path = _resolve(path)
    record: dict[str, Any] = {
        "path": str(path),
        "exists": path.is_file(),
        "size_bytes": path.stat().st_size if path.is_file() else None,
        "sha256": None,
        "load_ok": False,
        "error": None,
        "keys": [],
        "policy_kind": None,
        "iteration": None,
    }
    if not path.is_file():
        record["error"] = "missing checkpoint"
        return record
    record["sha256"] = _sha256(path)
    try:
        import torch

        payload = torch.load(path, map_location="cpu", weights_only=False)
        record["load_ok"] = True
        if isinstance(payload, dict):
            record["keys"] = sorted(str(key) for key in payload)
            infos = payload.get("infos")
            if isinstance(infos, dict):
                record["policy_kind"] = infos.get("policy_kind")
                record["checkpoint_schema"] = infos.get("schema_version")
            record["iteration"] = payload.get("iter")
    except Exception as error:  # noqa: BLE001 - audit records malformed files
        record["error"] = f"{type(error).__name__}: {error}"
    return record


def _ppo_audit() -> dict[str, Any]:
    entrypoint = REPO_ROOT / "scripts" / "train_batch_lowlevel_ppo.py"
    evaluation = REPO_ROOT / "scripts" / "evaluate_pc_ppo.py"
    entry_text = entrypoint.read_text(encoding="utf-8") if entrypoint.is_file() else ""
    residual_configs = sorted(
        _display(path)
        for path in (REPO_ROOT / "configs" / "lowlevel").glob("*ppo*.json")
    )
    command_residual_configs: list[str] = []
    actuator_residual_configs: list[str] = []
    for path in sorted((REPO_ROOT / "configs" / "lowlevel").glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        command_residual = payload.get("command_residual")
        if isinstance(command_residual, dict) and command_residual.get("enabled"):
            command_residual_configs.append(_display(path))
        residual = payload.get("residual")
        if isinstance(residual, dict) and any(
            float(residual.get(key, 0.0) or 0.0) > 0.0
            for key in ("scale_rudder", "scale_stern", "scale_rpm")
        ):
            actuator_residual_configs.append(_display(path))
    command_env = REPO_ROOT / "src" / "auv_lowlevel" / "batch" / "pc_command_residual.py"
    checks = {
        "entrypoint_exists": entrypoint.is_file(),
        "evaluation_exists": evaluation.is_file(),
        "supports_actor_warmstart": "--warmstart-actor" in entry_text,
        "supports_residual_warmstart": "--warmstart-residual" in entry_text,
        "imports_bc_actor_prior_loader": "load_pc_bc_actor_prior" in entry_text,
        "command_residual_env_exists": command_env.is_file(),
        "command_residual_configs_present": bool(command_residual_configs),
        "actuator_residual_configs_present": bool(actuator_residual_configs),
    }
    return {
        "entrypoint": _display(entrypoint),
        "evaluation": _display(evaluation),
        "command_residual_env": _display(command_env),
        "ppo_config_count": len(residual_configs),
        "ppo_config_examples": residual_configs[:12],
        "command_residual_config_count": len(command_residual_configs),
        "command_residual_config_examples": command_residual_configs[:12],
        "actuator_residual_config_count": len(actuator_residual_configs),
        "actuator_residual_config_examples": actuator_residual_configs[:12],
        "checks": checks,
        "status": "READY" if all(checks.values()) else "PPO_ENTRYPOINT_INCOMPLETE",
    }


def _sac_audit(sb3_root: Path) -> dict[str, Any]:
    sb3_root = _resolve(sb3_root)
    sac_source = sb3_root / "stable_baselines3" / "sac" / "sac.py"
    version_file = sb3_root / "stable_baselines3" / "version.txt"
    version = version_file.read_text(encoding="utf-8").strip() if version_file.is_file() else None
    import_error = None
    import_ok = False
    sac_class = None
    if sb3_root.is_dir() and str(sb3_root) not in sys.path:
        sys.path.insert(0, str(sb3_root))
    try:
        module = importlib.import_module("stable_baselines3")
        sac_module = importlib.import_module("stable_baselines3.sac")
        import_ok = True
        sac_class = getattr(sac_module, "SAC", None) is not None
        imported_version = getattr(module, "__version__", None)
    except Exception as error:  # noqa: BLE001 - audit records import failures
        import_error = f"{type(error).__name__}: {error}"
        imported_version = None
    project_imports = [
        hit
        for hit in _grep_sources(("stable_baselines3", "from sb3"))
        if hit["path"] != _display(Path(__file__).resolve())
    ]
    project_sac_imports = [
        hit
        for hit in _grep_sources(("import SAC", "stable_baselines3.sac", ".sac import"))
        if hit["path"] != _display(Path(__file__).resolve())
    ]
    project_sac_files = sorted(
        _display(path)
        for path in list((REPO_ROOT / "src").rglob("*sac*.py"))
        + list((REPO_ROOT / "scripts").rglob("*sac*.py"))
        if path.resolve() != Path(__file__).resolve()
    )
    adapter_candidate = REPO_ROOT / "src" / "auv_rl" / "ddpg" / "pc_command_residual_env.py"
    ddpg_pc_trainer = REPO_ROOT / "src" / "auv_rl" / "ddpg" / "pc_vector_trainer.py"
    checks = {
        "sb3_root_exists": sb3_root.is_dir(),
        "sac_source_exists": sac_source.is_file(),
        "sac_class_importable": import_ok and bool(sac_class),
        "project_imports_sb3": bool(project_imports),
        "project_imports_sac": bool(project_sac_imports),
        "pc_sac_source_exists": bool(project_sac_files),
        "pc_sac_trainer_exists": any("train" in name for name in project_sac_files),
        "pc_sac_eval_exists": any("eval" in name for name in project_sac_files),
        "offpolicy_gym_adapter_candidate_exists": adapter_candidate.is_file(),
        "custom_pc_ddpg_trainer_exists": ddpg_pc_trainer.is_file(),
    }
    missing = [
        name
        for name in (
            "pc_sac_source_exists",
            "pc_sac_trainer_exists",
            "pc_sac_eval_exists",
        )
        if not checks[name]
    ]
    return {
        "sb3_root": str(sb3_root),
        "sb3_version": version,
        "imported_version": imported_version,
        "sac_source": _display(sac_source),
        "import_error": import_error,
        "project_sb3_imports": project_imports[:20],
        "project_sac_imports": project_sac_imports[:20],
        "project_sac_files": project_sac_files,
        "offpolicy_gym_adapter_candidate": _display(adapter_candidate),
        "custom_pc_ddpg_trainer": _display(ddpg_pc_trainer),
        "integration_reference": (
            "The existing PC command-residual DDPG trainer is the closest "
            "off-policy integration reference, but it is not SAC and cannot be "
            "reported as a SAC result."
        ),
        "checks": checks,
        "missing_components": missing,
        "status": (
            "SAC_PC_ADAPTER_MISSING"
            if missing
            else "SAC_READY_SUBJECT_TO_SMOKE"
        ),
        "screen_allowed": not missing,
    }


def _goal85_contract_audit() -> dict[str, Any]:
    configs: list[dict[str, Any]] = []
    for path in sorted(
        (REPO_ROOT / "configs" / "lowlevel").glob("pc_goal85_*_20261110.json")
    ):
        payload = json.loads(path.read_text(encoding="utf-8"))
        sensors = payload.get("sensors") or {}
        termination = payload.get("termination") or {}
        configs.append(
            {
                "path": _display(path),
                "sha256": _sha256(path),
                "sensor_profile": sensors.get("profile"),
                "position_source": termination.get("position_source"),
                "unbiased_noise_v1": sensors.get("profile") == "unbiased_noise_v1",
                "eta_termination": termination.get("position_source") == "eta",
            }
        )
    test_files = [
        _display(path)
        for path in (
            REPO_ROOT / "tests" / "test_pc_goal85_unbiased_contract.py",
            REPO_ROOT / "tests" / "test_pc_terminal_proximity.py",
        )
        if path.is_file()
    ]
    passed = bool(configs) and all(
        item["unbiased_noise_v1"] and item["eta_termination"] for item in configs
    )
    return {
        "configs": configs,
        "test_files": test_files,
        "passed": passed,
        "status": "CONTRACT_IMPLEMENTED" if passed else "CONTRACT_INCOMPLETE",
    }


def _gpu_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "nvidia_smi_available": shutil.which("nvidia-smi") is not None,
        "gpus": [],
        "torch_cuda_available": None,
        "torch_device_count": None,
        "torch_version": None,
    }
    try:
        import torch

        state["torch_version"] = torch.__version__
        state["torch_cuda_available"] = bool(torch.cuda.is_available())
        state["torch_device_count"] = int(torch.cuda.device_count())
    except Exception as error:  # noqa: BLE001
        state["torch_error"] = f"{type(error).__name__}: {error}"
    if state["nvidia_smi_available"]:
        try:
            output = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=index,name,memory.total,memory.used,driver_version",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
                stderr=subprocess.DEVNULL,
            )
            for line in output.splitlines():
                parts = [item.strip() for item in line.split(",")]
                if len(parts) == 5:
                    state["gpus"].append(
                        {
                            "index": int(parts[0]),
                            "name": parts[1],
                            "memory_total_mib": int(parts[2]),
                            "memory_used_mib": int(parts[3]),
                            "driver_version": parts[4],
                        }
                    )
        except Exception as error:  # noqa: BLE001
            state["nvidia_smi_error"] = f"{type(error).__name__}: {error}"
    return state


def _disk_state() -> dict[str, Any]:
    paths = {
        "repo": REPO_ROOT,
        "artifacts": REPO_ROOT / "artifacts",
        "checkpoints": REPO_ROOT / "checkpoints",
    }
    result: dict[str, Any] = {}
    for name, path in paths.items():
        usage = shutil.disk_usage(path if path.exists() else REPO_ROOT)
        result[name] = {
            "path": str(path),
            "total_gib": round(usage.total / 1024**3, 2),
            "used_gib": round(usage.used / 1024**3, 2),
            "free_gib": round(usage.free / 1024**3, 2),
        }
    return result


def _runtime_state() -> dict[str, Any]:
    memory_total_gib = None
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("MemTotal:"):
                memory_total_gib = round(int(line.split()[1]) / 1024**2, 2)
                break
    except OSError:
        pass
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "memory_total_gib": memory_total_gib,
        "gpu": _gpu_state(),
        "disk": _disk_state(),
    }


def _discover_checkpoints(explicit: Sequence[Path]) -> list[Path]:
    if explicit:
        return [_resolve(path) for path in explicit]
    patterns = (
        "artifacts/goal59*/**/final.pt",
        "artifacts/**/formal*/final.pt",
        "checkpoints/goal59/**/final.pt",
    )
    found: set[Path] = set()
    for pattern in patterns:
        found.update(Path(match) for match in glob.glob(str(REPO_ROOT / pattern), recursive=True))
    return sorted(path for path in found if path.is_file())


def run_audit(*, sb3_root: Path, checkpoints: Sequence[Path]) -> dict[str, Any]:
    ppo = _ppo_audit()
    sac = _sac_audit(sb3_root)
    goal85 = _goal85_contract_audit()
    checkpoint_records = [
        _checkpoint_record(path) for path in _discover_checkpoints(checkpoints)
    ]
    ppo_ready = ppo["status"] == "READY"
    sac_ready = bool(sac["screen_allowed"])
    return {
        "schema_version": SCHEMA_VERSION,
        "repo_root": str(REPO_ROOT),
        "ppo": ppo,
        "sac": sac,
        "goal85_contract": goal85,
        "checkpoints": checkpoint_records,
        "runtime": _runtime_state(),
        "arm_decision": {
            "P-A-PPO": "READY" if ppo_ready else "PPO_BACKEND_NOT_READY",
            "T1-PPO": "READY" if ppo_ready else "PPO_BACKEND_NOT_READY",
            "T2-PPO": "READY" if ppo_ready else "PPO_BACKEND_NOT_READY",
            "U1-PPO": "READY" if ppo_ready else "PPO_BACKEND_NOT_READY",
            "P-C-PPO": "NOT_AVAILABLE",
            "U2-PPO": "NOT_AVAILABLE",
            "*-SAC": "BLOCKED_SAC_PC_ADAPTER_MISSING" if not sac_ready else "READY",
        },
        "ready_for_ppo_screen": ppo_ready and goal85["passed"],
        "ready_for_sac_screen": sac_ready and goal85["passed"],
        "passed": ppo_ready and goal85["passed"],
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    payload = run_audit(
        sb3_root=args.sb3_root or _discover_sb3_root(),
        checkpoints=args.checkpoint,
    )
    output = _resolve(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    summary = {
        "ppo": payload["ppo"]["status"],
        "sac": payload["sac"]["status"],
        "goal85": payload["goal85_contract"]["status"],
        "ready_for_ppo_screen": payload["ready_for_ppo_screen"],
        "ready_for_sac_screen": payload["ready_for_sac_screen"],
        "checkpoint_count": len(payload["checkpoints"]),
        "gpu_count": len(payload["runtime"]["gpu"]["gpus"]),
        "output": str(output),
    }
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
