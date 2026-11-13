from __future__ import annotations

import hashlib
import json
import math
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from auv_lowlevel.batch.pc_env import BatchedPCLowLevelEnv
from auv_lowlevel.config import load_config

ROOT = Path(__file__).resolve().parents[1]
ENV_CONFIG = ROOT / "configs" / "lowlevel" / "goal173" / "env_route_health_train_nu_ground.json"
TRACE_SHA256 = "f33a2ef188460ed68f8557e8ed22ce6f9de9bd84ac08b877652248092eb8a126"


def _canonical(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        array = value.detach().cpu().numpy()
        return {
            "kind": "tensor",
            "shape": list(array.shape),
            "dtype": str(array.dtype),
            "values": [float(item) for item in array.reshape(-1)],
        }
    if isinstance(value, np.ndarray):
        return {
            "kind": "ndarray",
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "values": value.reshape(-1).tolist(),
        }
    if is_dataclass(value) and not isinstance(value, type):
        return {
            "kind": type(value).__name__,
            "fields": {
                field.name: _canonical(getattr(value, field.name)) for field in fields(value)
            },
        }
    if isinstance(value, dict):
        return {str(key): _canonical(item) for key, item in sorted(value.items(), key=str)}
    if isinstance(value, (list, tuple)):
        return [_canonical(item) for item in value]
    if isinstance(value, (str, bool, int, float)) or value is None:
        return value
    return repr(value)


def _trace() -> tuple[str, list[dict[str, Any]]]:
    config = load_config(ENV_CONFIG)
    env = BatchedPCLowLevelEnv(config, num_envs=2, path_dim=12)
    observation = env.reset([31415, 27182])
    records: list[dict[str, Any]] = [{"step": -1, "observation": _canonical(observation)}]
    with torch.no_grad():
        for step in range(12):
            batch_index = torch.arange(2, dtype=torch.float64)
            action = torch.stack(
                [
                    torch.full_like(batch_index, 0.05 * math.sin(step + 1)),
                    torch.full_like(batch_index, -0.03 * math.cos(0.5 * step)),
                    torch.full_like(batch_index, 0.02 * math.sin(0.25 * step)),
                ],
                dim=1,
            )
            result = env.step(action)
            records.append(
                {
                    "step": step,
                    "action": _canonical(action),
                    "observation": _canonical(result.observation),
                    "reward": _canonical(result.reward),
                    "terminated": _canonical(result.terminated),
                    "truncated": _canonical(result.truncated),
                    "done": _canonical(result.done),
                    "info": _canonical(result.info),
                }
            )
    blob = json.dumps(records, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest(), records


def test_pc_env_step_trace_is_stable() -> None:
    digest, records = _trace()
    first = records[1]

    assert first["observation"]["fields"]["actor_state"]["shape"] == [2, 12]
    assert first["observation"]["fields"]["actor_path"]["shape"] == [2, 12]
    assert first["observation"]["fields"]["critic_state"]["shape"] == [2, 15]
    assert first["terminated"]["shape"] == [2]
    assert first["truncated"]["shape"] == [2]
    assert first["done"]["shape"] == [2]
    assert digest == TRACE_SHA256
