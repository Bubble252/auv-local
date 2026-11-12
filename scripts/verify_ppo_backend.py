from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _parse_args() -> tuple[argparse.Namespace, Any]:
    parser = argparse.ArgumentParser(description="Verify the isolated rsl_rl PPO backend")
    parser.add_argument("--output", type=Path, default=None)
    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    app_launcher = AppLauncher(args)
    return args, app_launcher.app


def main() -> int:
    args, simulation_app = _parse_args()
    result: dict[str, Any]
    try:
        import isaaclab_rl
        import rsl_rl
        import torch
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

        result = {
            "status": "PASS",
            "python_torch": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_device_count": int(torch.cuda.device_count()),
            "rsl_rl_version": getattr(rsl_rl, "__version__", "unknown"),
            "isaaclab_rl_version": getattr(isaaclab_rl, "__version__", "unknown"),
            "wrapper_class": RslRlVecEnvWrapper.__name__,
        }
    except Exception as exc:
        result = {
            "status": "FAIL",
            "error_type": type(exc).__name__,
            "error": str(exc),
        }

    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    simulation_app.close()
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
