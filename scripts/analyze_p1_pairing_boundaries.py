from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import calibrate_p1_analytic_sensor as calibration  # noqa: E402

from auv_rl.observation import ForwardSectorObservationBuilder  # noqa: E402
from auv_rl.perception import AnalyticSectorPerceptionBackend  # noqa: E402
from auv_rl.scenarios import default_planar_scenarios, scenario_by_key  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Classify sector-boundary outliers in an analytic/OceanSim pairing CSV"
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("artifacts/m4/p1/parallel/analytic_vs_oceansim.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/m4/p1/parallel/edge_boundary_analysis.json"),
    )
    parser.add_argument("--max-error-m", type=float, default=0.75)
    parser.add_argument("--local-range-m", type=float, default=20.0)
    return parser.parse_args()


def _close(value: float, target: float, tolerance: float = 1e-9) -> bool:
    return abs(value - target) <= tolerance


def main() -> int:
    args = _parse_args()
    rows = list(csv.DictReader(args.csv.open(encoding="utf-8")))
    failing = [row for row in rows if float(row["abs_error_m"]) > float(args.max_error_m) + 1e-12]
    poses = {
        pose.pose_id: pose
        for scenario_poses in calibration._all_poses(float(args.local_range_m)).values()
        for pose in scenario_poses
    }
    builder = ForwardSectorObservationBuilder(
        max_range_m=float(args.local_range_m),
        sector_count=13,
        field_of_view_rad=math.radians(130.0),
    )
    backend = AnalyticSectorPerceptionBackend()
    cases = default_planar_scenarios()
    scenario_by_key_value = {key: scenario_by_key(cases, key).scenario for key in ("s1", "s2")}
    analyses: list[dict[str, Any]] = []
    for row in failing:
        pose_id = str(row["pose_id"])
        pose = poses[pose_id]
        scenario = scenario_by_key_value[pose.scenario]
        backend.reset(scenario=scenario)
        observation = backend.observe(calibration._state(pose))
        scan = observation.scans[0]
        angles_deg = np.degrees(np.arctan2(scan.directions_body[:, 1], scan.directions_body[:, 0]))
        sector_index = int(row["sector_index"])
        lower_deg = math.degrees(builder.sector_edges_rad[sector_index])
        upper_deg = math.degrees(builder.sector_edges_rad[sector_index + 1])
        if sector_index == builder.sector_count - 1:
            in_sector = (angles_deg >= lower_deg - 1e-12) & (angles_deg <= upper_deg + 1e-12)
        else:
            in_sector = (angles_deg >= lower_deg - 1e-12) & (angles_deg < upper_deg - 1e-12)
        hit = in_sector & (scan.ranges_m < float(args.local_range_m) - 1e-9)
        hit_angles = [float(value) for value in angles_deg[hit]]
        margin = min(min(abs(angle - lower_deg), abs(angle - upper_deg)) for angle in hit_angles)
        analyses.append(
            {
                "scenario": row["scenario"],
                "pose_id": pose_id,
                "category": row["category"],
                "outside_nominal_bounds": row["outside_nominal_bounds"] == "True",
                "sector_index": sector_index,
                "sector_edges_deg": [lower_deg, upper_deg],
                "ocean_range_m": float(row["ocean_range_m"]),
                "analytic_range_m": float(row["analytic_range_m"]),
                "abs_error_m": float(row["abs_error_m"]),
                "dangerous_miss": row["dangerous_miss"] == "True",
                "analytic_hit_ray_count": len(hit_angles),
                "analytic_hit_angles_deg": hit_angles,
                "nearest_boundary_margin_deg": margin,
                "ocean_saturated": bool(
                    _close(float(row["ocean_range_m"]), float(args.local_range_m))
                ),
                "analytic_hit_within_one_ray_of_boundary": margin <= 0.05 + 1e-9,
            }
        )

    all_boundary = bool(analyses) and all(
        item["analytic_hit_within_one_ray_of_boundary"] for item in analyses
    )
    safe_direction_only = bool(analyses) and all(
        item["ocean_saturated"] and not item["dangerous_miss"] for item in analyses
    )
    payload = {
        "schema": "auv_p1_pairing_boundary_analysis.v1",
        "source_csv": str(args.csv),
        "threshold_m": float(args.max_error_m),
        "rows_total": len(rows),
        "failing_rows": len(analyses),
        "failing_poses": len({item["pose_id"] for item in analyses}),
        "all_failures_are_single_boundary_ray": all_boundary,
        "all_failures_are_safe_direction": safe_direction_only,
        "all_failure_sector_indices": sorted({item["sector_index"] for item in analyses}),
        "max_nearest_boundary_margin_deg": max(
            item["nearest_boundary_margin_deg"] for item in analyses
        ),
        "analyses": analyses,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
