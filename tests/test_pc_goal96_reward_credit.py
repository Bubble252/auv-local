from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "analyze_pc_goal96_reward_credit.py"
)
SPEC = importlib.util.spec_from_file_location("goal96_reward_credit", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def _write_run(root: Path) -> Path:
    run = root / "P_D3_s86201_strict125_policy"
    run.mkdir(parents=True)
    distances = np.asarray([10.0, 5.0, 3.0, 2.0, 1.2, 10.0, 5.0, 3.0, 2.0, 1.8])
    seeds = np.asarray([20] * 5 + [21] * 5, dtype=np.int64)
    steps = np.asarray(list(range(5)) * 2, dtype=np.int64)
    terminal_kind = np.zeros(10, dtype=np.int64)
    terminal_kind[4] = 1
    terminal_kind[9] = 9
    total = np.asarray([1.0, 1.5, 1.0, 0.5, 300.0, 1.0, 0.5, 0.0, -1.0, -100.0])
    terminal = np.asarray([0.0, 0.0, 0.0, 0.0, 300.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    plane_miss = np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -100.0])
    terminal_progress = np.asarray([0.2, 0.2, 0.2, 0.2, 0.0, 0.1, 0.1, 0.1, 0.1, 0.0])
    command_yaw = total - terminal - plane_miss - terminal_progress
    np.savez_compressed(
        run / "trace.npz",
        episode_seed=seeds,
        step=steps,
        terminal_kind=terminal_kind,
        goal_distance_eta_m=distances,
        reward_total=total,
        reward_terminal=terminal,
        reward_goal_plane_miss=plane_miss,
        reward_terminal_progress=terminal_progress,
        reward_command_yaw=command_yaw,
    )
    (run / "summary.json").write_text(json.dumps({"dt_s": 0.5}), encoding="utf-8")
    (run / "terminal_proximity.json").write_text(
        json.dumps(
            {
                "episodes": [
                    {"seed": 20, "success": True},
                    {"seed": 21, "success": False},
                ]
            }
        ),
        encoding="utf-8",
    )
    return run


def test_phase_b_reports_reward_components_and_terminal_share(tmp_path: Path) -> None:
    run = _write_run(tmp_path)
    payload = audit.analyze_run(run, bootstrap_draws=20, bootstrap_seed=7)
    assert payload["successes"] == 1
    assert payload["episodes"] == 2
    assert payload["component_returns"]["reward_terminal"]["success_mean"] == 300.0
    assert payload["terminal_positive_share"]["success_mean"] == 300.0 / 304.0
    assert payload["success_terminal_positive_share_mean"] == 300.0 / 304.0
    assert payload["decision"]["terminal_positive_share_below_0.80"] is False
    assert payload["reward_to_go_at_entry"]["5m"]["success_mean"] == 303.0
    assert payload["reward_to_go_at_entry"]["5m"]["failure_mean"] == -100.5


def test_phase_b_cli_writes_json_and_markdown(tmp_path: Path) -> None:
    run = _write_run(tmp_path)
    output = tmp_path / "phaseB.json"
    markdown = tmp_path / "phaseB.md"
    audit.main(
        [
            "--trace-root",
            str(run.parent),
            "--output",
            str(output),
            "--markdown",
            str(markdown),
            "--bootstrap-draws",
            "20",
        ]
    )
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["schema_version"] == audit.SCHEMA_VERSION
    assert len(payload["runs"]) == 1
    assert "reward / credit" in markdown.read_text(encoding="utf-8")


def test_phase_b_reconstructs_raw_path_geometry_as_reward(tmp_path: Path) -> None:
    path = tmp_path / "trace.npz"
    command = np.asarray([1.0, 2.0])
    cross = np.asarray([3.0, 4.0])
    vertical = np.asarray([4.0, 3.0])
    total = command - 0.15 * np.sqrt(cross**2 + vertical**2) / 5.0
    np.savez_compressed(
        path,
        reward_total=total,
        reward_command_yaw=command,
        reward_path_cross=cross,
        reward_path_vertical=vertical,
    )
    weights = {
        "pc_control_mode": "alos",
        "w_path_3d": 0.15,
        "w_yaw": 0.0,
        "w_prog": 0.0,
        "w_path_speed_smooth": 0.0,
        "w_path_energy": 0.0,
        "w_u": 0.0,
    }
    with np.load(path) as data:
        contributions = audit._reward_contributions(data, weights)
    np.testing.assert_allclose(
        contributions["reward_path_track_3d"], -0.15 * 5.0 / 5.0
    )
    reconstructed = sum(contributions.values())
    np.testing.assert_allclose(reconstructed, total)
