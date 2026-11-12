from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

SPEC = importlib.util.spec_from_file_location(
    "goal109_residual_direction",
    SCRIPTS / "analyze_pc_goal109_residual_direction.py",
)
assert SPEC is not None and SPEC.loader is not None
audit = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = audit
SPEC.loader.exec_module(audit)


def _write_run(
    root: Path,
    *,
    scenario: str,
    profile: str,
    success_seeds: set[int],
    mode: str,
) -> None:
    run = root / f"{audit.ARM}_{scenario}_{profile}"
    run.mkdir(parents=True)
    episodes = []
    steps = []
    seeds = []
    time_s = []
    distance = []
    eta = []
    goal = []
    terminal_kind = []
    heading_error = []
    pitch_error = []
    speed_error = []
    applied = []
    base = []
    saturated = []

    for seed in range(20, 60):
        success = seed in success_seeds
        episodes.append(
            {
                "seed": seed,
                "success": success,
                "timeout": False,
                "safety": False,
            }
        )

        if mode == "geometry":
            final_position = [1.2, 0.0, 0.1] if success else [1.8, 0.0, 0.8]
            heading = 0.1 if success else 0.8
            pitch = 0.1 if success else 0.8
        else:
            final_position = [1.5, 0.0, 0.5]
            heading = 0.2
            pitch = 0.2

        residual = np.asarray([0.1, 0.1, 0.1], dtype=np.float64)
        material = False
        if mode == "direction" and not success:
            residual = np.asarray([-1.0, -1.0, -1.0], dtype=np.float64)
            material = True
        elif mode == "magnitude" and not success:
            residual = np.asarray([2.0, 2.0, 2.0], dtype=np.float64)
            material = True

        for step, position in enumerate(([3.0, 0.0, 0.0], final_position)):
            steps.append(step)
            seeds.append(seed)
            time_s.append(float(step))
            distance.append(float(np.linalg.norm(position)))
            eta.append(position)
            goal.append([0.0, 0.0, 0.0])
            terminal_kind.append(1 if success and step == 1 else 0)
            heading_error.append(heading)
            pitch_error.append(pitch)
            speed_error.append(0.2)
            applied.append(residual)
            base.append([1.0, 1.0, 1.0])
            saturated.append(material)

    (run / "summary.json").write_text(
        json.dumps(
            {
                "episodes": episodes,
                "checkpoint_sha256": audit.EXPECTED_CHECKPOINT_SHA,
                "actuator_residual_scale_override": audit.EVAL_SCALE,
                "actuator_residual_channel_gains": list(audit.PROFILES[profile]),
            }
        ),
        encoding="utf-8",
    )
    np.savez(
        run / "trace.npz",
        step=np.asarray(steps),
        episode_seed=np.asarray(seeds),
        time_s=np.asarray(time_s),
        goal_distance_eta_m=np.asarray(distance),
        eta=np.asarray(eta),
        goal_ned_m=np.asarray(goal),
        terminal_kind=np.asarray(terminal_kind),
        heading_error_rad=np.asarray(heading_error),
        pitch_error_rad=np.asarray(pitch_error),
        speed_error_mps=np.asarray(speed_error),
        actuator_residual_applied=np.asarray(applied),
        actuator_base_physical=np.asarray(base),
        material_saturated=np.asarray(saturated),
    )


def _write_matrix(
    root: Path,
    *,
    mode: str,
    c0_default: int = 33,
    c0_strict: int = 20,
) -> None:
    counts = {
        "default": {"C0": c0_default, "U050": 32},
        "strict125": {"C0": c0_strict, "U050": 23},
    }
    for scenario in audit.SCENARIOS:
        for profile in audit.PROFILES:
            _write_run(
                root,
                scenario=scenario,
                profile=profile,
                success_seeds=set(range(20, 20 + counts[scenario][profile])),
                mode=mode,
            )


def test_audit_detects_direction_and_magnitude(tmp_path: Path) -> None:
    _write_matrix(tmp_path, mode="direction")
    summary = audit._summarize(audit._scan(tmp_path))
    gate = audit._gate(summary)
    assert gate["status"] == "RESIDUAL_DIRECTION_MISMATCH"
    assert gate["direction_max_abs_cliffs_delta"] >= audit.EFFECT_THRESHOLD
    assert gate["magnitude_max_abs_cliffs_delta"] >= audit.EFFECT_THRESHOLD


def test_audit_detects_magnitude_without_direction(tmp_path: Path) -> None:
    _write_matrix(tmp_path, mode="magnitude")
    summary = audit._summarize(audit._scan(tmp_path))
    gate = audit._gate(summary)
    assert gate["status"] == "RESIDUAL_MAGNITUDE_LIMITED"
    assert gate["direction_max_abs_cliffs_delta"] < audit.EFFECT_THRESHOLD
    assert gate["magnitude_max_abs_cliffs_delta"] >= audit.EFFECT_THRESHOLD


def test_audit_detects_terminal_geometry_without_residual_signal(
    tmp_path: Path,
) -> None:
    _write_matrix(tmp_path, mode="geometry")
    summary = audit._summarize(audit._scan(tmp_path))
    gate = audit._gate(summary)
    assert gate["status"] == "TERMINAL_GEOMETRY_LIMITED"
    assert gate["direction_max_abs_cliffs_delta"] < audit.EFFECT_THRESHOLD
    assert gate["magnitude_max_abs_cliffs_delta"] < audit.EFFECT_THRESHOLD
    assert len(gate["geometry_strong_features"]) >= 2


def test_audit_invalid_when_c0_does_not_reproduce(tmp_path: Path) -> None:
    _write_matrix(tmp_path, mode="none", c0_default=32)
    summary = audit._summarize(audit._scan(tmp_path))
    gate = audit._gate(summary)
    assert gate["status"] == "INVALID"
    assert "default" in gate["baseline_failures"][0]


def test_audit_invalid_when_trace_missing_speed_error(tmp_path: Path) -> None:
    _write_matrix(tmp_path, mode="none")
    run = tmp_path / f"{audit.ARM}_default_C0"
    with np.load(run / "trace.npz") as data:
        arrays = {name: np.asarray(data[name]) for name in data.files}
    del arrays["speed_error_mps"]
    np.savez(run / "trace.npz", **arrays)
    output = tmp_path / "summary.json"
    markdown = tmp_path / "summary.md"
    audit.main(
        [
            "--root",
            str(tmp_path),
            "--output",
            str(output),
            "--markdown",
            str(markdown),
        ]
    )
    gate = json.loads(output.read_text(encoding="utf-8"))["gate"]
    assert gate["status"] == "INVALID"
    assert "speed_error_mps" in gate["reason"]
