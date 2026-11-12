from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from auv_lowlevel.batch.pc_calibration import calibrate_pc20_high_speed
from auv_lowlevel.config import load_config


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--settle-seconds", type=float, default=60.0)
    parser.add_argument("--window-seconds", type=float, default=10.0)
    parser.add_argument("--step-seconds", type=float, default=5.0)
    parser.add_argument("--rpm-points", type=int, default=9)
    parser.add_argument("--current-sweep-mps", type=float, nargs="+")
    parser.add_argument(
        "--current-directions-deg",
        type=float,
        nargs="+",
        default=list(range(0, 360, 45)),
    )
    parser.add_argument("--coastdown-seconds", type=float, default=60.0)
    parser.add_argument("--coastdown-speed-threshold-mps", type=float, default=0.2)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.output.exists():
        raise FileExistsError(f"calibration output already exists: {args.output}")
    dtype = torch.float32 if args.dtype == "float32" else torch.float64
    reports = []
    for path in args.config:
        config = load_config(path)
        if config.arm != "PC" or config.policy_mode != "pure_controller":
            raise ValueError(f"{path} is not a PC pure-controller config")
        reports.append(
            calibrate_pc20_high_speed(
                config,
                config_path=path,
                settle_seconds=args.settle_seconds,
                window_seconds=args.window_seconds,
                step_seconds=args.step_seconds,
                rpm_points=args.rpm_points,
                current_sweep_mps=args.current_sweep_mps,
                current_directions_deg=args.current_directions_deg,
                coastdown_seconds=args.coastdown_seconds,
                coastdown_speed_threshold_mps=args.coastdown_speed_threshold_mps,
                device=args.device,
                dtype=dtype,
            )
        )
    payload = {
        "schema_version": "auv_pc20_high_speed_calibration_set.v2",
        "configs": reports,
        "all_passed": all(report["reachability"]["passed"] for report in reports),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
