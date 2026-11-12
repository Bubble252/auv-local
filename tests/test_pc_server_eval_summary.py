from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "summarize_pc_server_evals.py"
BASICS_SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "summarize_pc_server_evals_basics_20261103.py"
)
METRICS = (
    "success_rate",
    "timeout_rate",
    "safety_termination_rate",
    "material_saturation_rate",
    "path_three_d_rmse_m",
    "speed_rmse_mps",
    "yaw_rmse_rad",
    "command_hold_success_rate",
    "final_hold_fraction_mean",
    "path_completion_ratio_mean",
)


def _load_module(script: Path = SCRIPT) -> ModuleType:
    spec = importlib.util.spec_from_file_location("summarize_pc_server_evals", script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {script}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_summary(
    root: Path,
    *,
    tag: str,
    arm_run: str,
    training_config: str,
    value: float,
    checkpoint: str = "99",
    extra_metrics: dict[str, float] | None = None,
    suffix: str = "",
) -> None:
    output = (
        root
        / tag
        / f"{arm_run}_seed91430"
        / f"ck{checkpoint}_seed0_31{suffix}"
    )
    output.mkdir(parents=True)
    summary = {
        "schema_version": "auv_pc_ppo_evaluation.v1",
        "training_config": training_config,
        "checkpoint": str(output / f"model_{checkpoint}.pt"),
        "seeds": list(range(32)),
        "metrics": {name: value for name in METRICS}
        | ({} if extra_metrics is None else extra_metrics),
    }
    (output / "summary.json").write_text(json.dumps(summary), encoding="utf-8")


def test_baseline_by_tag_parsing_rejects_malformed_values() -> None:
    module = _load_module()

    assert module._parse_baseline_by_tag(["h1=a", "h2=b"]) == {"h1": "a", "h2": "b"}
    with pytest.raises(ValueError, match="expected TAG=ARM"):
        module._parse_baseline_by_tag(["h1"])
    with pytest.raises(ValueError, match="duplicate"):
        module._parse_baseline_by_tag(["h1=a", "h1=b"])


def test_summary_uses_tag_specific_baseline_for_paired_deltas(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_module()
    root = tmp_path / "evals"
    output = tmp_path / "summary"
    h1_config = "odarrl_control_ppo_pc_h1_speed_1p8_2p0_frozen_bc_residual_nodade.json"
    h2_config = "odarrl_control_ppo_pc_h2_speed_2p0_2p25_frozen_bc_residual_nodade.json"
    _write_summary(
        root,
        tag="h2",
        arm_run="frozen_bc_residual_nodade_h1",
        training_config=h1_config,
        value=0.6,
    )
    _write_summary(
        root,
        tag="h2",
        arm_run="frozen_bc_residual_nodade_h2",
        training_config=h2_config,
        value=0.4,
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "summarize_pc_server_evals.py",
            "--eval-root",
            str(root),
            "--output-dir",
            str(output),
            "--baseline-arm",
            "frozen_bc_residual_nodade",
            "--baseline-by-tag",
            "h2=frozen_bc_residual_nodade_h2",
        ],
    )
    assert module.main() == 0

    report = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert report["baseline_by_tag"] == {"h2": "frozen_bc_residual_nodade_h2"}
    assert len(report["paired"]) == 1
    paired = report["paired"][0]
    assert paired["arm"] == "frozen_bc_residual_nodade_h1"
    assert paired["baseline_arm"] == "frozen_bc_residual_nodade_h2"
    assert paired["delta_success_rate"] == pytest.approx(0.2)


def test_summary_keeps_corrected_frozen_residual_arms_separate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_module()
    root = tmp_path / "evals"
    output = tmp_path / "summary"
    baseline = "frozen_bc_residual_nodade_lr3e-4_scale1p0"
    isolated = "frozen_bc_residual_dade_head_isolated_lr3e-4_scale1p0"
    history_only = "frozen_bc_residual_dade_history_only_lr3e-4_scale1p0"

    for arm in (baseline, isolated, history_only):
        config = (
            "odarrl_control_ppo_pc_c3_random_realistic_"
            f"{arm}_dz64_gamma995_warmup5_critic3e-4_fixed_rewardscale001.json"
        )
        assert module._arm_from_config(config) == arm

    _write_summary(
        root,
        tag="default",
        arm_run=baseline,
        training_config=(
            "odarrl_control_ppo_pc_c3_random_realistic_"
            f"{baseline}_dz64_gamma995_warmup5_critic3e-4_fixed_rewardscale001.json"
        ),
        value=0.5,
    )
    _write_summary(
        root,
        tag="default",
        arm_run=isolated,
        training_config=(
            "odarrl_control_ppo_pc_c3_random_realistic_"
            f"{isolated}_dz64_gamma995_warmup5_critic3e-4_fixed_rewardscale001.json"
        ),
        value=0.625,
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "summarize_pc_server_evals.py",
            "--eval-root",
            str(root),
            "--output-dir",
            str(output),
            "--baseline-arm",
            baseline,
        ],
    )
    assert module.main() == 0

    report = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert len(report["paired"]) == 1
    paired = report["paired"][0]
    assert paired["arm"] == isolated
    assert paired["baseline_arm"] == baseline
    assert paired["delta_success_rate"] == pytest.approx(0.125)


def test_summary_keeps_c2_baselines_and_tags_separate() -> None:
    module = _load_module()

    assert (
        module._arm_from_config("odarrl_control_ppo_pc_c2_alos_nodade_stable.json")
        == "c2_nodade"
    )
    assert (
        module._arm_from_config("odarrl_control_ppo_pc_c2_alos_dade_stable.json")
        == "c2_dade"
    )
    root = Path("/evals")
    assert module._tag(root, root / "c2" / "a" / "ck0_seed0_63" / "summary.json") == "c2"
    assert (
        module._tag(root, root / "c2_clean" / "a" / "ck0_seed0_63" / "summary.json")
        == "c2_clean"
    )


def test_summary_preserves_speed_current_grid_tags() -> None:
    module = _load_module(BASICS_SCRIPT)
    root = Path("/evals")
    for tag in ("grid_lw", "grid_ls", "grid_hw", "grid_hs"):
        path = root / tag / "arm" / "ckfinal_seed0_31" / "summary.json"
        assert module._tag(root, path) == tag
    assert (
        module._arm_from_config(
            "odarrl_control_ppo_pc_grid_frozen_bc_residual_nodade_"
            "lr3e-4_scale1p0_dz64.json"
        )
        == "frozen_bc_residual_nodade_grid_lr3e-4_scale1p0"
    )
    assert (
        module._arm_from_config(
            "odarrl_control_ppo_pc_grid_dagger_frozen_bc_residual_nodade_"
            "lr3e-4_scale1p0_dz64.json"
        )
        == "frozen_bc_residual_nodade_grid_dagger_lr3e-4_scale1p0"
    )
    assert (
        module._arm_from_config(
            "odarrl_control_ppo_pc_grid_dade_dagger_frozen_bc_residual_dade_pose_"
            "lr3e-4_scale1p0_dz64.json"
        )
        == "frozen_bc_residual_dade_grid_dagger_pose_lr3e-4_scale1p0"
    )
    assert (
        module._arm_from_config(
            "odarrl_control_ppo_pc_grid_frozen_bc_residual_nodade_smooth1e-3_"
            "lr3e-4_scale1p0_dz64.json"
        )
        == "frozen_bc_residual_nodade_grid_smooth1e-3_lr3e-4_scale1p0"
    )
    assert (
        module._arm_from_config(
            "odarrl_control_ppo_pc_grid_frozen_bc_residual_nodade_smooth1e-3_"
            "l2_1e-2_lr3e-4_scale1p0_dz64.json"
        )
        == "frozen_bc_residual_nodade_grid_smooth1e-3_l2_1e-2_lr3e-4_scale1p0"
    )


@pytest.mark.parametrize("script", [SCRIPT, BASICS_SCRIPT])
def test_summary_keeps_multienv_supervised_residual_arms_separate(script: Path) -> None:
    module = _load_module(script)

    assert (
        module._arm_from_config(
            "odarrl_control_ppo_pc_multienv_frozen_bc_residual_nodade_"
            "lr3e-4_scale1p0_dz64.json"
        )
        == "frozen_bc_residual_nodade_multienv_lr3e-4_scale1p0"
    )
    assert (
        module._arm_from_config(
            "odarrl_control_ppo_pc_multienv_frozen_bc_residual_dade_pose_"
            "lr3e-4_scale1p0_dz64.json"
        )
        == "frozen_bc_residual_dade_multienv_pose_lr3e-4_scale1p0"
    )
    assert (
        module._arm_from_config(
            "odarrl_control_ppo_pc_mixed_domain_frozen_bc_residual_nodade_"
            "lr3e-4_scale1p0_dz64.json"
        )
        == "frozen_bc_residual_nodade_mixed_domain_lr3e-4_scale1p0"
    )


@pytest.mark.parametrize("script", [SCRIPT, BASICS_SCRIPT])
def test_summary_distinguishes_supervised_residual_from_ppo(script: Path) -> None:
    module = _load_module(script)
    base = (
        "odarrl_control_ppo_pc_h3_speed_2p25_2p4_"
        "frozen_bc_residual_nodade_lr3e-4_scale1p0_dz64.json"
    )

    assert (
        module._arm_from_summary(
            {"training_config": base, "checkpoint_metadata": {"policy_kind": "pc_actor_critic"}}
        )
        == "frozen_bc_residual_nodade_h3_lr3e-4_scale1p0"
    )
    assert (
        module._arm_from_summary(
            {
                "training_config": base,
                "checkpoint_metadata": {"policy_kind": "pc_supervised_residual"},
            }
        )
        == "supervised_frozen_bc_residual_nodade_h3_lr3e-4_scale1p0"
    )


@pytest.mark.parametrize(
    ("config_name", "arm"),
    [
        (
            "odarrl_control_ppo_pc_h3_speed_2p25_2p4_"
            "frozen_bc_residual_nodade_lr3e-4_scale1p0_dz64.json",
            "frozen_bc_residual_nodade_h3_lr3e-4_scale1p0",
        ),
        (
            "odarrl_control_ppo_pc_h1_speed_1p8_2p0_"
            "frozen_bc_residual_nodade_lr3e-4_scale1p0_dz64.json",
            "frozen_bc_residual_nodade_h1_lr3e-4_scale1p0",
        ),
        (
            "odarrl_control_ppo_pc_h2_speed_2p0_2p25_"
            "frozen_bc_residual_nodade_lr3e-4_scale1p0_dz64.json",
            "frozen_bc_residual_nodade_h2_lr3e-4_scale1p0",
        ),
        (
            "odarrl_control_ppo_pc_c4_stress_"
            "frozen_bc_residual_nodade_lr3e-4_scale1p0_dz64.json",
            "frozen_bc_residual_nodade_c4_lr3e-4_scale1p0",
        ),
    ],
)
def test_summary_keeps_corrected_hard_environment_arms_separate(
    config_name: str, arm: str
) -> None:
    module = _load_module()

    assert module._arm_from_config(config_name) == arm


def test_summary_pairs_checkpoint_with_its_initial_baseline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_module()
    root = tmp_path / "evals"
    output = tmp_path / "summary"
    h2_config = "odarrl_control_ppo_pc_h2_speed_2p0_2p25_frozen_bc_residual_nodade.json"
    _write_summary(
        root,
        tag="h2",
        arm_run="frozen_bc_residual_nodade_h2",
        training_config=h2_config,
        value=0.4,
        checkpoint="initial",
    )
    _write_summary(
        root,
        tag="h2",
        arm_run="frozen_bc_residual_nodade_h2",
        training_config=h2_config,
        value=0.55,
        checkpoint="99",
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "summarize_pc_server_evals.py",
            "--eval-root",
            str(root),
            "--output-dir",
            str(output),
            "--checkpoint-baseline",
            "99=initial",
        ],
    )
    assert module.main() == 0

    report = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert report["checkpoint_baseline"] == {"99": "initial"}
    assert len(report["checkpoint_paired"]) == 1
    row = report["checkpoint_paired"][0]
    assert row["checkpoint"] == "99"
    assert row["checkpoint_baseline"] == "initial"
    assert row["delta_success_rate"] == pytest.approx(0.15)


@pytest.mark.parametrize("script", [SCRIPT, BASICS_SCRIPT])
def test_summary_falls_back_to_first_checkpoint_for_c2_without_initial(
    script: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_module(script)
    root = tmp_path / "evals"
    output = tmp_path / "summary"
    config = "odarrl_control_ppo_pc_c2_alos_nodade_stable.json"
    _write_summary(
        root,
        tag="c2",
        arm_run="c2_nodade",
        training_config=config,
        value=0.5,
        checkpoint="0",
    )
    _write_summary(
        root,
        tag="c2",
        arm_run="c2_nodade",
        training_config=config,
        value=0.7,
        checkpoint="20",
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            script.name,
            "--eval-root",
            str(root),
            "--output-dir",
            str(output),
            "--checkpoint-baseline",
            "20=initial",
        ],
    )
    assert module.main() == 0

    report = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert len(report["checkpoint_paired"]) == 1
    row = report["checkpoint_paired"][0]
    assert row["checkpoint_baseline"] == "0"
    assert row["checkpoint_baseline_requested"] == "initial"
    assert row["delta_success_rate"] == pytest.approx(0.2)


@pytest.mark.parametrize("script", [SCRIPT, BASICS_SCRIPT])
def test_summary_excludes_history_ablation_from_normal_performance(
    script: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_module(script)
    root = tmp_path / "evals"
    output = tmp_path / "summary"
    config = (
        "odarrl_control_ppo_pc_c3_random_realistic_"
        "frozen_bc_residual_dade_head.json"
    )
    _write_summary(
        root,
        tag="default",
        arm_run="frozen_bc_residual_dade_head",
        training_config=config,
        value=0.5,
        checkpoint="99",
    )
    _write_summary(
        root,
        tag="default",
        arm_run="frozen_bc_residual_dade_head",
        training_config=config,
        value=0.4,
        checkpoint="99",
        suffix="_history-zero",
    )
    history_path = next(root.glob("**/*_history-zero/summary.json"))
    assert module._training_seed(history_path) == 91430
    assert module._history_ablation(history_path) == "history-zero"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            script.name,
            "--eval-root",
            str(root),
            "--output-dir",
            str(output),
        ],
    )
    assert module.main() == 0

    report = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert len(report["rows"]) == 1
    assert report["rows"][0]["history_ablation"] == "none"
    assert report["history_ablations"] == ["none"]


def test_summary_keeps_optional_residual_metrics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_module()
    root = tmp_path / "evals"
    output = tmp_path / "summary"
    config = (
        "odarrl_control_ppo_pc_c3_random_realistic_"
        "frozen_bc_residual_nodade.json"
    )
    _write_summary(
        root,
        tag="default",
        arm_run="frozen_bc_residual_nodade",
        training_config=config,
        value=0.5,
        checkpoint="initial",
        extra_metrics={"residual_action_abs_mean": 0.0},
    )
    _write_summary(
        root,
        tag="default",
        arm_run="frozen_bc_residual_nodade",
        training_config=config,
        value=0.5,
        checkpoint="99",
        extra_metrics={"residual_action_abs_mean": 0.002},
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "summarize_pc_server_evals.py",
            "--eval-root",
            str(root),
            "--output-dir",
            str(output),
            "--checkpoint-baseline",
            "99=initial",
        ],
    )
    assert module.main() == 0

    report = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    row = report["checkpoint_paired"][0]
    assert row["delta_residual_action_abs_mean"] == pytest.approx(0.002)
    aggregate = next(
        item
        for item in report["aggregates"]
        if item["checkpoint"] == "99" and item["arm"] == "frozen_bc_residual_nodade"
    )
    assert aggregate["raw"]["residual_action_abs_mean"]["mean"] == pytest.approx(0.002)


def test_summary_skips_runs_marked_invalid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_module()
    root = tmp_path / "evals"
    output = tmp_path / "summary"
    config = "odarrl_control_ppo_pc_c3_random_realistic_dade_head.json"
    _write_summary(
        root,
        tag="default",
        arm_run="dade_head",
        training_config=config,
        value=0.5,
    )
    summary_path = next(root.glob("**/summary.json"))
    checkpoint_parent = summary_path.parent
    (checkpoint_parent / "INVALID_FOR_DADE_CONCLUSIONS.json").write_text(
        json.dumps({"reason": "optimizer_group_defect"}), encoding="utf-8"
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "summarize_pc_server_evals.py",
            "--eval-root",
            str(root),
            "--output-dir",
            str(output),
        ],
    )
    assert module.main() == 0

    report = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert report["rows"] == []
    assert len(report["invalid_runs_skipped"]) == 1
    assert "INVALID_FOR_DADE_CONCLUSIONS" in report["invalid_runs_skipped"][0]["reason"]

