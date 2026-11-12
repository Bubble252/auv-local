from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import calibrate_p1_analytic_sensor as calibration  # noqa: E402

from auv_rl.oceansim_pose import p1_state_to_oceansim_pose  # noqa: E402
from auv_rl.perception import (  # noqa: E402
    OceanSimPointCloudSource,
    PointCloudSonarAdapter,
    _as_numpy_array,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect raw OceanSim points for one pairing pose")
    parser.add_argument("--pose-id", default="s2_start")
    parser.add_argument("--pose-updates", type=int, default=1)
    parser.add_argument("--sequence", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("/tmp/p1_pairing_debug"))
    return parser.parse_args()


def _summarize_points(points: np.ndarray, *, origin: np.ndarray, half_fov: float) -> str:
    if points.size == 0:
        return "points=0"
    relative = points - origin.reshape(1, 3)
    horizontal = np.hypot(relative[:, 0], relative[:, 1])
    azimuth = np.degrees(np.arctan2(relative[:, 1], relative[:, 0]))
    bins = np.floor((np.radians(azimuth) + half_fov) / math.radians(0.1)).astype(int)
    bins = np.clip(bins, 0, 1299)
    counts = np.bincount(bins, minlength=1300)
    active = np.flatnonzero(counts)
    return (
        f"points={points.shape[0]} az_min={azimuth.min():.3f} az_max={azimuth.max():.3f} "
        f"range_min={horizontal.min():.3f} range_max={horizontal.max():.3f} "
        f"active_bins={active.size} active_span={active[0] if active.size else -1}.."
        f"{active[-1] if active.size else -1}"
    )


def main() -> int:
    args = _parse_args()
    selected = None
    for poses in calibration._all_poses().values():
        for pose in poses:
            if pose.pose_id == args.pose_id:
                selected = pose
                break
        if selected is not None:
            break
    if selected is None:
        raise SystemExit(f"unknown pose id: {args.pose_id}")

    target_position = np.array([selected.x_m, selected.y_m, -5.0], dtype=np.float64)
    original_scan = PointCloudSonarAdapter.scan_from_points
    original_capture = OceanSimPointCloudSource.capture
    printed_raw = [False]
    printed_body = [False]

    def debug_capture(self, state):
        if not printed_raw[0] and np.allclose(state.position_ned_m, target_position):
            frame = original_capture(self, state)
            raw = _as_numpy_array(self.sensor.scan_data["pcl"], "pcl")
            view = np.asarray(self.sensor.scan_data["viewTransform"], dtype=np.float64)
            position, orientation = p1_state_to_oceansim_pose(state)
            print(
                "DEBUG_RAW",
                args.pose_id,
                "state",
                np.asarray(state.position_ned_m).tolist(),
                "pose_writer_position",
                np.asarray(position).tolist(),
                "pose_writer_orientation",
                np.asarray(orientation).tolist(),
                "raw_shape",
                raw.shape,
                "raw_min",
                raw.min(axis=0).tolist() if raw.size else [],
                "raw_max",
                raw.max(axis=0).tolist() if raw.size else [],
                "view_translation",
                view[:3, 3].tolist(),
                "view_rotation",
                view[:3, :3].round(6).tolist(),
                flush=True,
            )
            printed_raw[0] = True
            return frame
        return original_capture(self, state)

    def debug_scan(self, points_body_m, *, stamp_s, state, **kwargs):
        if not printed_body[0] and np.allclose(state.position_ned_m, target_position):
            origin = self.config.origin_array
            print(
                "DEBUG_POSE",
                args.pose_id,
                "state",
                np.asarray(state.position_ned_m).tolist(),
                "origin",
                np.asarray(origin).tolist(),
                _summarize_points(
                    np.asarray(points_body_m, dtype=np.float64),
                    origin=origin,
                    half_fov=0.5 * self.config.field_of_view_rad,
                ),
                flush=True,
            )
            printed_body[0] = True
        return original_scan(self, points_body_m, stamp_s=stamp_s, state=state, **kwargs)

    PointCloudSonarAdapter.scan_from_points = debug_scan
    OceanSimPointCloudSource.capture = debug_capture
    if not args.sequence:
        calibration._all_poses = lambda: {"s2": [selected]}  # type: ignore[assignment]
    sys.argv = [
        sys.argv[0],
        "--output",
        str(args.output),
        "--pose-updates",
        str(args.pose_updates),
    ]
    return calibration.main()


if __name__ == "__main__":
    raise SystemExit(main())
