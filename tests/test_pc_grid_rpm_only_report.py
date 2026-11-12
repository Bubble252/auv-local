from __future__ import annotations

import importlib.util
import json
from pathlib import Path

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "summarize_pc_grid_rpm_only_20261104.py"
)
SPEC = importlib.util.spec_from_file_location("summarize_pc_grid_rpm_only", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

REFERENCE_ARM = MODULE.REFERENCE_ARM
TAGS = MODULE.TAGS
TREATMENT_ARM = MODULE.TREATMENT_ARM
build_report = MODULE.build_report
render_markdown = MODULE.render_markdown


def _row(arm: str, tag: str, seed: int, **overrides: float) -> dict[str, object]:
    values: dict[str, object] = {
        "arm": arm,
        "tag": tag,
        "train_seed": seed,
        "checkpoint": "final",
        "history_ablation": "none",
        "eval_seeds": 32,
        "success_rate": 0.90,
        "timeout_rate": 0.10,
        "path_three_d_rmse_m": 3.0,
        "material_saturation_rate": 0.30,
        "mean_abs_speed_error_mps": 0.20,
        "mean_abs_yaw_error_rad": 0.20,
        "ground_speed_ratio_min": 0.80,
        "ground_route_speed_ratio_min": 0.80,
        "ground_route_speed_ratio_episode_min_mean": 0.80,
        "ground_route_speed_ratio_mean": 0.90,
    }
    values.update(overrides)
    return values


def _summary(tmp_path: Path, *, drop_first_seed: bool = False) -> Path:
    rows: list[dict[str, object]] = []
    for seed in range(5):
        for tag in TAGS:
            if drop_first_seed and seed == 0 and tag == "default":
                continue
            overrides: dict[str, float] = {}
            if tag == "grid_hs":
                overrides = {"success_rate": 0.90, "path_three_d_rmse_m": 5.5}
            elif tag == "strict125":
                overrides = {
                    "success_rate": 0.55,
                    "path_three_d_rmse_m": 6.0,
                    "timeout_rate": 0.50,
                }
            elif tag == "default":
                overrides = {"success_rate": 0.75, "path_three_d_rmse_m": 4.5}
            elif tag == "h1":
                overrides = {
                    "success_rate": 0.96,
                    "path_three_d_rmse_m": 4.0,
                    "material_saturation_rate": 0.39,
                }
            elif tag == "h3":
                overrides = {
                    "success_rate": 0.96,
                    "path_three_d_rmse_m": 3.0,
                    "material_saturation_rate": 0.44,
                }
            rows.append(_row(TREATMENT_ARM, tag, seed, **overrides))
            reference_overrides = dict(overrides)
            reference_overrides["path_three_d_rmse_m"] = max(
                3.1, float(overrides.get("path_three_d_rmse_m", 3.0)) + 0.1
            )
            rows.append(_row(REFERENCE_ARM, tag, seed, **reference_overrides))
    path = tmp_path / "summary.json"
    path.write_text(json.dumps({"rows": rows}), encoding="utf-8")
    return path


def test_rpm_only_report_passes_matched_paired_gate(tmp_path: Path) -> None:
    report = build_report(
        summary_path=_summary(tmp_path),
        treatment_arm=TREATMENT_ARM,
        reference_arm=REFERENCE_ARM,
        train_seed_count=5,
        seed_count=32,
    )

    assert report["overall_status"] == "pass"
    assert report["stages"]["grid_hs"]["gate"]["status"] == "pass"
    assert "RPM-only" in render_markdown(report)


def test_rpm_only_report_fails_when_strict_timeout_or_speed_floor_fails(
    tmp_path: Path,
) -> None:
    path = _summary(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    for row in payload["rows"]:
        if row["arm"] == TREATMENT_ARM and row["tag"] == "strict125":
            row["timeout_rate"] = 0.60
            row["ground_route_speed_ratio_min"] = 0.60
            row["ground_route_speed_ratio_episode_min_mean"] = 0.60
    path.write_text(json.dumps(payload), encoding="utf-8")

    report = build_report(
        summary_path=path,
        treatment_arm=TREATMENT_ARM,
        reference_arm=REFERENCE_ARM,
        train_seed_count=5,
        seed_count=32,
    )

    assert report["overall_status"] == "fail"
    checks = report["stages"]["strict125"]["gate"]["checks"]
    assert checks["timeout_le_0.531"] is False
    assert checks["ground_route_speed_min_ge_0.75"] is False


def test_rpm_only_report_marks_missing_training_seed_incomplete(tmp_path: Path) -> None:
    report = build_report(
        summary_path=_summary(tmp_path, drop_first_seed=True),
        treatment_arm=TREATMENT_ARM,
        reference_arm=REFERENCE_ARM,
        train_seed_count=5,
        seed_count=32,
    )

    assert report["overall_status"] == "incomplete"
    assert report["stages"]["default"]["status"] == "incomplete"


def test_rpm_only_report_supports_smoke_tag_subset_without_legacy_ground_metric(
    tmp_path: Path,
) -> None:
    path = _summary(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    for row in payload["rows"]:
        row.pop("ground_speed_ratio_min", None)
    path.write_text(json.dumps(payload), encoding="utf-8")

    report = build_report(
        summary_path=path,
        treatment_arm=TREATMENT_ARM,
        reference_arm=REFERENCE_ARM,
        train_seed_count=5,
        seed_count=32,
        tags=("grid_hs", "default", "strict125"),
    )

    assert report["overall_status"] == "pass"
    assert report["requested_tags"] == ["grid_hs", "default", "strict125"]
    assert set(report["stages"]) == {"grid_hs", "default", "strict125"}
