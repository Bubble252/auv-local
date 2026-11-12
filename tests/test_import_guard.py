from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def _run_import_check(code: str) -> subprocess.CompletedProcess[str]:
    project = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = os.pathsep.join(
        part for part in (str(project / "src"), existing) if part
    )
    return subprocess.run(
        [sys.executable, "-c", code],
        cwd=project,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def _assert_clean(code: str) -> None:
    result = _run_import_check(code)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"


def test_local_core_import_does_not_load_rl_simulator_or_global_planners() -> None:
    _assert_clean(
        "import sys\n"
        "import auv_local\n"
        "forbidden = {'torch', 'holoocean', 'rclpy', 'python_vehicle_simulator', "
        "'auv_rl', 'auv_planner', 'auv_local.execution', "
        "'auv_local.closed_loop', 'matplotlib'}\n"
        "loaded = forbidden.intersection(sys.modules)\n"
        "if loaded:\n"
        "    raise SystemExit(sorted(loaded))\n"
        "print('ok')\n"
    )


def test_contract_import_does_not_load_implementations() -> None:
    _assert_clean(
        "import sys\n"
        "import auv_local.contracts\n"
        "forbidden = {'torch', 'holoocean', 'rclpy', 'python_vehicle_simulator', "
        "'auv_rl', 'auv_planner', 'auv_local.execution', "
        "'auv_local.closed_loop', 'matplotlib'}\n"
        "loaded = forbidden.intersection(sys.modules)\n"
        "if loaded:\n"
        "    raise SystemExit(sorted(loaded))\n"
        "print('ok')\n"
    )


def test_rl_contract_import_does_not_load_local_planner_or_torch() -> None:
    _assert_clean(
        "import sys\n"
        "import auv_rl\n"
        "forbidden = {'torch', 'holoocean', 'rclpy', 'auv_local', 'auv_planner'}\n"
        "loaded = forbidden.intersection(sys.modules)\n"
        "if loaded:\n"
        "    raise SystemExit(sorted(loaded))\n"
        "print('ok')\n"
    )


def test_public_imports_remain_available() -> None:
    _assert_clean(
        "import sys\n"
        "from auv_local import LocalObservation, RangeScan\n"
        "from auv_rl import ActionSpec, ObservationSpec\n"
        "assert LocalObservation.__name__ == 'LocalObservation'\n"
        "assert RangeScan.__name__ == 'RangeScan'\n"
        "assert ActionSpec.__name__ == 'ActionSpec'\n"
        "assert ObservationSpec.__name__ == 'ObservationSpec'\n"
        "assert 'auv_planner' not in sys.modules\n"
        "print('ok')\n"
    )
