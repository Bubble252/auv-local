#!/usr/bin/env python3
"""Run a Python script with a hard CUDA allocator cap.

Usage:
    python _gpu_mem_cap.py <script.py> [args...]

The cap is read from AUV_GPU_MEM_LIMIT_MB. It limits this process' caching
allocator, so a job can never take more than the budget reserved at launch.
"""

from __future__ import annotations

import os
import runpy
import sys


def main() -> None:
    limit_mb = float(os.environ.get("AUV_GPU_MEM_LIMIT_MB", "0") or 0)
    if limit_mb > 0:
        import torch

        if torch.cuda.is_available():
            total_mb = torch.cuda.get_device_properties(0).total_memory / (1024.0**2)
            fraction = min(0.98, max(0.05, limit_mb / total_mb))
            torch.cuda.set_per_process_memory_fraction(fraction, 0)
            print(
                f"[gpu-cap] limit={limit_mb:.0f} MiB total={total_mb:.0f} MiB "
                f"fraction={fraction:.3f}",
                flush=True,
            )
    target = sys.argv[1]
    sys.argv = sys.argv[1:]
    sys.path[0] = os.path.dirname(os.path.abspath(target))
    runpy.run_path(target, run_name="__main__")


if __name__ == "__main__":
    main()
