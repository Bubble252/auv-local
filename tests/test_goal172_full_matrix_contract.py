from __future__ import annotations

import importlib.util
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

from goal172_matrix_contract import (  # noqa: E402
    LEVELS,
    RADII,
    SEED_COUNT,
    SEED_START,
    diagnostic_jobs,
)


def _load_checker():
    path = SCRIPTS / "check_goal172_protocol.py"
    spec = importlib.util.spec_from_file_location("check_goal172_protocol", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_goal172_matrix_is_full_matrix_not_optional() -> None:
    jobs = diagnostic_jobs()
    counts = Counter(job.kind for job in jobs)

    assert LEVELS == ("nominal", "current07", "current10", "current15")
    assert RADII == ("default", "strict125")
    assert SEED_START == 20
    assert SEED_COUNT == 40
    assert len(jobs) == 184
    assert counts == {
        "ppo": 48,
        "sac": 48,
        "channel": 48,
        "rudder_scale": 24,
        "reward": 16,
    }


def test_goal172_jobs_cover_every_level_and_radius() -> None:
    jobs = diagnostic_jobs()
    for job in jobs:
        assert job.level in LEVELS
        assert job.radius in RADII

    level_counts = Counter(job.level for job in jobs)
    radius_counts = Counter(job.radius for job in jobs)
    assert level_counts == {level: 46 for level in LEVELS}
    assert radius_counts == {radius: 92 for radius in RADII}


def test_goal172_output_path_is_level_aware(tmp_path: Path) -> None:
    jobs = diagnostic_jobs()
    paths = [job.output_path(tmp_path) for job in jobs]
    assert len(paths) == len(set(paths))
    assert any("nominal" in path.parts for path in paths)
    assert any("current15" in path.parts for path in paths)


def test_goal172_output_audit_rejects_empty_matrix(tmp_path: Path) -> None:
    checker = _load_checker()
    errors = checker.audit_output(tmp_path)
    assert len(errors) == 184
    assert all("missing summary.json" in error for error in errors)

