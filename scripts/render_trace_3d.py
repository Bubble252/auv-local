"""Render persisted M2 trace JSON files as interactive 3D review pages."""

from __future__ import annotations

import argparse
from pathlib import Path

from auv_core.contracts import Obstacle

from auv_local.scenarios import m2_scenarios
from auv_local.trace_viewer3d import PlotlyJSMode, render_trace_directory

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render M2 traces as interactive 3D HTML")
    parser.add_argument(
        "--trace-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "m2" / "traces",
        help="directory containing M2 trace JSON files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "m2" / "3d",
        help="directory for generated HTML files",
    )
    parser.add_argument(
        "--plotlyjs",
        choices=("directory", "inline", "cdn"),
        default="directory",
        help="directory writes one shared plotly.min.js; inline embeds it in every page; cdn needs network",
    )
    return parser.parse_args()


def _obstacles_by_scenario() -> dict[str, tuple[Obstacle, ...]]:
    return {
        case.scenario.name: case.scenario.world.obstacle_map.obstacles for case in m2_scenarios()
    }


def main() -> int:
    args = _parse_args()
    plotlyjs_mode: PlotlyJSMode = args.plotlyjs
    outputs = render_trace_directory(
        args.trace_dir,
        args.output_dir,
        obstacles_by_scenario=_obstacles_by_scenario(),
        plotlyjs_mode=plotlyjs_mode,
    )
    for path in outputs:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
