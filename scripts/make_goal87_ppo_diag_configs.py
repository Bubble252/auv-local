"""Generate the pre-registered Goal87 PPO diagnostic configs from the frozen T2/P-A screens."""

from __future__ import annotations

import copy
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = REPO_ROOT / "configs/lowlevel/goal86/ppo_diag_20261110"
BASE_T2 = REPO_ROOT / "configs/lowlevel/goal86/pc_goal86_t2_ppo_screen_20261110.json"
BASE_PA = REPO_ROOT / "configs/lowlevel/goal86/pc_goal86_p-a_ppo_screen_20261110.json"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _set_common(payload: dict, name: str, *, seed: int = 86201) -> None:
    payload["experiment"]["name"] = f"pc_goal87_ppo_diag_{name}_20261110"
    payload["experiment"]["output"] = f"artifacts/lowlevel/goal86/ppo_diag/{name}"
    payload["experiment"]["seed"] = seed


def _write(name: str, payload: dict) -> None:
    path = CONFIG_ROOT / f"{name}.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    CONFIG_ROOT.mkdir(parents=True, exist_ok=True)
    t2 = _load(BASE_T2)
    pa = _load(BASE_PA)

    payload = copy.deepcopy(t2)
    payload["runner"]["num_steps_per_env"] = 25
    payload["training"]["iterations"] = 40
    _set_common(payload, "upd25x40")
    _write("upd25x40", payload)

    payload = copy.deepcopy(t2)
    payload["runner"]["num_steps_per_env"] = 50
    payload["training"]["iterations"] = 20
    _set_common(payload, "upd50x20")
    _write("upd50x20", payload)

    payload = copy.deepcopy(t2)
    payload["training"]["initial_residual_zero"] = False
    _set_common(payload, "random_init")
    _write("random_init", payload)

    for name, noise in (("noise0.3", 0.3), ("noise0.5", 0.5)):
        payload = copy.deepcopy(t2)
        payload["policy"]["init_noise_std"] = noise
        _set_common(payload, name)
        _write(name, payload)

    payload = copy.deepcopy(t2)
    payload["algorithm"]["num_learning_epochs"] = 4
    _set_common(payload, "epochs4")
    _write("epochs4", payload)

    payload = copy.deepcopy(t2)
    payload["algorithm"]["critic_warmup_iterations"] = 0
    _set_common(payload, "critic0")
    _write("critic0", payload)

    payload = copy.deepcopy(t2)
    payload["algorithm"]["clip_param"] = 0.2
    _set_common(payload, "clip0.2")
    _write("clip0.2", payload)

    payload = copy.deepcopy(t2)
    payload["algorithm"]["learning_rate"] = 1e-3
    payload["algorithm"]["learning_rate_actor"] = 1e-3
    payload["algorithm"]["learning_rate_critic"] = 1e-3
    _set_common(payload, "lr1e-3")
    _write("lr1e-3", payload)

    payload = copy.deepcopy(t2)
    payload["algorithm"]["normalize_advantage_per_mini_batch"] = True
    _set_common(payload, "adv_norm")
    _write("adv_norm", payload)

    payload = copy.deepcopy(pa)
    payload["training"]["initial_residual_zero"] = False
    payload["pc"]["actuator_residual_scale"] = 1.0
    _set_common(payload, "pa_bc_head")
    _write("pa_bc_head", payload)

    payload = copy.deepcopy(pa)
    payload["training"]["initial_residual_zero"] = False
    payload["pc"]["actuator_residual_scale"] = 1.0
    payload["runner"]["num_steps_per_env"] = 25
    payload["training"]["iterations"] = 40
    _set_common(payload, "pa_bc_head_upd25x40")
    _write("pa_bc_head_upd25x40", payload)


if __name__ == "__main__":
    main()
