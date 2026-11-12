"""Standalone interactive 3D viewer for persisted M2 trace JSON files.

The viewer is deliberately not imported by :mod:`auv_local` and does not
participate in planning or execution. It reads trace logs, builds Plotly
figures, and writes self-contained HTML review pages.
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import cos, isfinite, pi, sin
from pathlib import Path
from typing import Any, Literal

import numpy as np
import plotly.graph_objects as go
from auv_core.contracts import Obstacle
from plotly.offline import get_plotlyjs

Point3 = tuple[float, float, float]
PlotlyJSMode = Literal["directory", "inline", "cdn"]

_SEED_COLORS = ("royalblue", "darkorange", "seagreen", "mediumpurple", "crimson")
_REFERENCE_COLOR = "rgba(90, 90, 90, 0.45)"
_LOCAL_PLAN_COLOR = "rgba(35, 120, 170, 0.28)"


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON object")
    return value


def _number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _point(value: object, name: str) -> Point3:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (3,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite shape-(3,) vector")
    return (float(array[0]), float(array[1]), float(array[2]))


def _display_point(point: Point3) -> Point3:
    """Render NED altitude as an up-positive display coordinate."""

    return (point[0], point[1], -point[2])


def _local_path(value: object, name: str) -> tuple[Point3, ...]:
    if not isinstance(value, list) or len(value) < 2:
        raise ValueError(f"{name} must contain at least two points")
    return tuple(_point(point, f"{name}[{index}]") for index, point in enumerate(value))


@dataclass(frozen=True)
class TraceSample:
    t_s: float
    position_ned_m: Point3
    reference_position_ned_m: Point3
    local_path_ned_m: tuple[Point3, ...]
    replanned: bool
    speed_mps: float
    rudder_angle_rad: float
    stern_plane_angle_rad: float
    propeller_rpm: float

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any], index: int) -> TraceSample:
        prefix = f"samples[{index}]"
        state = _mapping(payload.get("state"), f"{prefix}.state")
        reference = _mapping(payload.get("reference"), f"{prefix}.reference")
        command = _mapping(payload.get("command"), f"{prefix}.command")
        velocity = _point(
            state.get("linear_velocity_body_mps"),
            f"{prefix}.state.linear_velocity_body_mps",
        )
        return cls(
            t_s=_number(state.get("t_s"), f"{prefix}.state.t_s"),
            position_ned_m=_point(state.get("position_ned_m"), f"{prefix}.state.position"),
            reference_position_ned_m=_point(
                reference.get("position_ned_m"), f"{prefix}.reference.position"
            ),
            local_path_ned_m=_local_path(
                payload.get("local_path_ned_m"), f"{prefix}.local_path_ned_m"
            ),
            replanned=bool(payload.get("replanned", False)),
            speed_mps=velocity[0],
            rudder_angle_rad=_number(
                command.get("rudder_angle_rad"), f"{prefix}.command.rudder_angle_rad"
            ),
            stern_plane_angle_rad=_number(
                command.get("stern_plane_angle_rad"),
                f"{prefix}.command.stern_plane_angle_rad",
            ),
            propeller_rpm=_number(command.get("propeller_rpm"), f"{prefix}.command.propeller_rpm"),
        )


@dataclass(frozen=True)
class TraceDocument:
    case_name: str
    scenario_name: str
    seed: int
    termination_reason: str
    initial_position_ned_m: Point3
    goal_ned_m: Point3
    samples: tuple[TraceSample, ...]

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> TraceDocument:
        initial_state = _mapping(payload.get("initial_state"), "initial_state")
        raw_samples = payload.get("samples")
        if not isinstance(raw_samples, list) or not raw_samples:
            raise ValueError("samples must be a non-empty JSON array")
        seed_value = _number(payload.get("seed"), "seed")
        if not seed_value.is_integer():
            raise ValueError("seed must be an integer")
        return cls(
            case_name=_text(payload.get("case_name"), "case_name"),
            scenario_name=_text(payload.get("scenario_name"), "scenario_name"),
            seed=int(seed_value),
            termination_reason=_text(payload.get("termination_reason"), "termination_reason"),
            initial_position_ned_m=_point(
                initial_state.get("position_ned_m"), "initial_state.position_ned_m"
            ),
            goal_ned_m=_point(payload.get("goal_ned_m"), "goal_ned_m"),
            samples=tuple(
                TraceSample.from_dict(_mapping(sample, f"samples[{index}]"), index)
                for index, sample in enumerate(raw_samples)
            ),
        )

    @classmethod
    def from_path(cls, path: Path | str) -> TraceDocument:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(_mapping(payload, "trace"))


def _hover_label(sample: TraceSample) -> str:
    return (
        f"t={sample.t_s:.2f} s<br>"
        f"NED=({sample.position_ned_m[0]:.2f}, "
        f"{sample.position_ned_m[1]:.2f}, {sample.position_ned_m[2]:.2f})<br>"
        f"speed={sample.speed_mps:.2f} m/s<br>"
        f"rudder={sample.rudder_angle_rad:.3f} rad<br>"
        f"stern plane={sample.stern_plane_angle_rad:.3f} rad<br>"
        f"RPM={sample.propeller_rpm:.0f}"
    )


def _actual_trace(document: TraceDocument, color: str) -> go.Scatter3d:
    positions = (document.initial_position_ned_m,) + tuple(
        sample.position_ned_m for sample in document.samples
    )
    labels = ("initial state",) + tuple(_hover_label(sample) for sample in document.samples)
    display = tuple(_display_point(point) for point in positions)
    return go.Scatter3d(
        x=[point[0] for point in display],
        y=[point[1] for point in display],
        z=[point[2] for point in display],
        mode="lines",
        line={"color": color, "width": 6},
        name=f"seed {document.seed}: actual ({document.termination_reason})",
        legendgroup=document.case_name,
        text=list(labels),
        hovertemplate="%{text}<extra></extra>",
    )


def _reference_trace(document: TraceDocument, color: str) -> go.Scatter3d:
    display = tuple(_display_point(sample.reference_position_ned_m) for sample in document.samples)
    return go.Scatter3d(
        x=[point[0] for point in display],
        y=[point[1] for point in display],
        z=[point[2] for point in display],
        mode="lines",
        line={"color": color, "width": 2, "dash": "dot"},
        opacity=0.55,
        name=f"seed {document.seed}: ALOS reference",
        legendgroup=document.case_name,
        hoverinfo="skip",
    )


def _local_plans_trace(document: TraceDocument) -> go.Scatter3d | None:
    x: list[float | None] = []
    y: list[float | None] = []
    z: list[float | None] = []
    for sample in document.samples:
        if not sample.replanned:
            continue
        for point in sample.local_path_ned_m:
            display = _display_point(point)
            x.append(display[0])
            y.append(display[1])
            z.append(display[2])
        x.append(None)
        y.append(None)
        z.append(None)
    if not x:
        return None
    return go.Scatter3d(
        x=x,
        y=y,
        z=z,
        mode="lines",
        line={"color": _LOCAL_PLAN_COLOR, "width": 2},
        name=f"seed {document.seed}: local plans",
        legendgroup=document.case_name,
        visible="legendonly",
        hoverinfo="skip",
    )


def _final_marker(document: TraceDocument, color: str) -> go.Scatter3d:
    final = _display_point(document.samples[-1].position_ned_m)
    symbol = "circle" if document.termination_reason == "goal" else "x"
    return go.Scatter3d(
        x=[final[0]],
        y=[final[1]],
        z=[final[2]],
        mode="markers",
        marker={"color": color, "size": 6, "symbol": symbol},
        name=f"seed {document.seed}: final",
        legendgroup=document.case_name,
        text=[f"seed {document.seed}: {document.termination_reason}"],
        hovertemplate="%{text}<extra></extra>",
    )


def _mesh_trace(
    *,
    vertices: Sequence[Point3],
    faces: Sequence[tuple[int, int, int]],
    name: str,
    hover: str,
) -> go.Mesh3d:
    display = tuple(_display_point(point) for point in vertices)
    return go.Mesh3d(
        x=[point[0] for point in display],
        y=[point[1] for point in display],
        z=[point[2] for point in display],
        i=[face[0] for face in faces],
        j=[face[1] for face in faces],
        k=[face[2] for face in faces],
        color="rgb(150, 155, 160)",
        opacity=0.22,
        flatshading=True,
        name=name,
        text=[hover],
        hovertemplate="%{text}<extra></extra>",
        showlegend=False,
    )


def _box_geometry(
    obstacle: Obstacle,
) -> tuple[tuple[Point3, ...], tuple[tuple[int, int, int], ...]]:
    if obstacle.half_extents_m is None:
        raise ValueError(f"box obstacle {obstacle.obstacle_id} has no half_extents_m")
    center = np.asarray(obstacle.center_ned_m, dtype=np.float64)
    half = np.asarray(obstacle.half_extents_m, dtype=np.float64)
    if half.shape != (3,) or np.any(half <= 0.0):
        raise ValueError(f"box obstacle {obstacle.obstacle_id} has invalid half_extents_m")
    vertices = tuple(
        (
            float(center[0] + sx * half[0]),
            float(center[1] + sy * half[1]),
            float(center[2] + sz * half[2]),
        )
        for sx, sy, sz in (
            (-1, -1, -1),
            (1, -1, -1),
            (1, 1, -1),
            (-1, 1, -1),
            (-1, -1, 1),
            (1, -1, 1),
            (1, 1, 1),
            (-1, 1, 1),
        )
    )
    faces = (
        (0, 1, 2),
        (0, 2, 3),
        (4, 6, 5),
        (4, 7, 6),
        (0, 5, 1),
        (0, 4, 5),
        (1, 6, 2),
        (1, 5, 6),
        (2, 7, 3),
        (2, 6, 7),
        (3, 4, 0),
        (3, 7, 4),
    )
    return vertices, faces


def _cylinder_geometry(
    obstacle: Obstacle,
) -> tuple[tuple[Point3, ...], tuple[tuple[int, int, int], ...]]:
    if obstacle.radius_m is None or obstacle.height_m is None:
        raise ValueError(f"cylinder obstacle {obstacle.obstacle_id} has incomplete dimensions")
    center = np.asarray(obstacle.center_ned_m, dtype=np.float64)
    radius = float(obstacle.radius_m)
    height = float(obstacle.height_m)
    if radius <= 0.0 or height <= 0.0:
        raise ValueError(f"cylinder obstacle {obstacle.obstacle_id} has invalid dimensions")
    count = 32
    lower: list[Point3] = []
    upper: list[Point3] = []
    for index in range(count):
        angle = 2.0 * pi * index / count
        x = float(center[0] + radius * cos(angle))
        y = float(center[1] + radius * sin(angle))
        lower.append((x, y, float(center[2] - height / 2.0)))
        upper.append((x, y, float(center[2] + height / 2.0)))
    vertices = tuple(
        lower
        + upper
        + [
            (float(center[0]), float(center[1]), float(center[2] - height / 2.0)),
            (float(center[0]), float(center[1]), float(center[2] + height / 2.0)),
        ]
    )
    lower_center = 2 * count
    upper_center = lower_center + 1
    faces: list[tuple[int, int, int]] = []
    for index in range(count):
        next_index = (index + 1) % count
        faces.append((index, next_index, count + index))
        faces.append((count + index, next_index, count + next_index))
        faces.append((lower_center, next_index, index))
        faces.append((upper_center, count + index, count + next_index))
    return vertices, tuple(faces)


def _sphere_geometry(
    obstacle: Obstacle,
) -> tuple[tuple[Point3, ...], tuple[tuple[int, int, int], ...]]:
    if obstacle.radius_m is None:
        raise ValueError(f"sphere obstacle {obstacle.obstacle_id} has no radius_m")
    center = np.asarray(obstacle.center_ned_m, dtype=np.float64)
    radius = float(obstacle.radius_m)
    if radius <= 0.0:
        raise ValueError(f"sphere obstacle {obstacle.obstacle_id} has invalid radius")
    latitude_count = 12
    longitude_count = 24
    vertices: list[Point3] = []
    for latitude in range(latitude_count + 1):
        theta = pi * latitude / latitude_count
        for longitude in range(longitude_count):
            phi = 2.0 * pi * longitude / longitude_count
            vertices.append(
                (
                    float(center[0] + radius * sin(theta) * cos(phi)),
                    float(center[1] + radius * sin(theta) * sin(phi)),
                    float(center[2] + radius * cos(theta)),
                )
            )
    faces: list[tuple[int, int, int]] = []
    for latitude in range(latitude_count):
        for longitude in range(longitude_count):
            next_longitude = (longitude + 1) % longitude_count
            first = latitude * longitude_count + longitude
            second = latitude * longitude_count + next_longitude
            third = (latitude + 1) * longitude_count + longitude
            fourth = (latitude + 1) * longitude_count + next_longitude
            faces.extend(((first, second, third), (second, fourth, third)))
    return tuple(vertices), tuple(faces)


def _obstacle_trace(obstacle: Obstacle) -> go.Mesh3d:
    if obstacle.kind == "box":
        vertices, faces = _box_geometry(obstacle)
    elif obstacle.kind == "cylinder":
        vertices, faces = _cylinder_geometry(obstacle)
    elif obstacle.kind == "sphere":
        vertices, faces = _sphere_geometry(obstacle)
    else:
        raise ValueError(f"unsupported obstacle kind: {obstacle.kind}")
    return _mesh_trace(
        vertices=vertices,
        faces=faces,
        name=obstacle.obstacle_id,
        hover=(
            f"{obstacle.kind}: {obstacle.obstacle_id}<br>"
            f"center NED=({obstacle.center_ned_m[0]:.2f}, "
            f"{obstacle.center_ned_m[1]:.2f}, {obstacle.center_ned_m[2]:.2f})"
        ),
    )


def build_figure(
    documents: Sequence[TraceDocument],
    obstacles: Sequence[Obstacle] = (),
    *,
    title: str | None = None,
) -> go.Figure:
    if not documents:
        raise ValueError("at least one trace document is required")
    traces: list[Any] = [_obstacle_trace(obstacle) for obstacle in obstacles]
    for index, document in enumerate(sorted(documents, key=lambda item: item.seed)):
        color = _SEED_COLORS[index % len(_SEED_COLORS)]
        traces.append(_actual_trace(document, color))
        traces.append(_reference_trace(document, color))
        local_trace = _local_plans_trace(document)
        if local_trace is not None:
            traces.append(local_trace)
        traces.append(_final_marker(document, color))

    start = _display_point(documents[0].initial_position_ned_m)
    goal = _display_point(documents[0].goal_ned_m)
    traces.extend(
        (
            go.Scatter3d(
                x=[start[0]],
                y=[start[1]],
                z=[start[2]],
                mode="markers",
                marker={"color": "green", "size": 8, "symbol": "circle"},
                name="start",
                text=["start"],
                hovertemplate="%{text}<extra></extra>",
            ),
            go.Scatter3d(
                x=[goal[0]],
                y=[goal[1]],
                z=[goal[2]],
                mode="markers",
                marker={"color": "red", "size": 10, "symbol": "diamond"},
                name="goal",
                text=["goal"],
                hovertemplate="%{text}<extra></extra>",
            ),
        )
    )

    scenario = documents[0].scenario_name
    summary = "; ".join(
        f"seed {document.seed}: {document.termination_reason}" for document in documents
    )
    figure = go.Figure(data=traces)
    figure.update_layout(
        title={
            "text": (f"{title or scenario} · 3D trajectory review<br><sup>{summary}</sup>"),
            "x": 0.5,
        },
        template="plotly_white",
        height=820,
        margin={"l": 0, "r": 0, "t": 85, "b": 0},
        hovermode="closest",
        legend={"x": 0.01, "y": 0.99, "bgcolor": "rgba(255,255,255,0.75)"},
        scene={
            "xaxis": {
                "title": "NED x [m]",
                "backgroundcolor": "rgb(245,245,245)",
                "gridcolor": "white",
            },
            "yaxis": {
                "title": "NED y [m]",
                "backgroundcolor": "rgb(245,245,245)",
                "gridcolor": "white",
            },
            "zaxis": {
                "title": "Altitude [m] (up positive)",
                "backgroundcolor": "rgb(245,245,245)",
                "gridcolor": "white",
            },
            "aspectmode": "data",
            "dragmode": "orbit",
            "camera": {"eye": {"x": 1.7, "y": 1.7, "z": 1.2}},
        },
        annotations=[
            {
                "text": "Drag to rotate · wheel to zoom · right-drag to pan",
                "xref": "paper",
                "yref": "paper",
                "x": 0.01,
                "y": 0.01,
                "showarrow": False,
                "font": {"size": 11, "color": "rgb(90,90,90)"},
            }
        ],
    )
    return figure


def _ensure_plotlyjs(directory: Path) -> None:
    path = directory / "plotly.min.js"
    if not path.exists():
        path.write_text(get_plotlyjs(), encoding="utf-8")


def render_group_html(
    documents: Sequence[TraceDocument],
    obstacles: Sequence[Obstacle],
    output_path: Path | str,
    *,
    plotlyjs_mode: PlotlyJSMode = "directory",
) -> Path:
    if plotlyjs_mode not in {"directory", "inline", "cdn"}:
        raise ValueError(f"unsupported plotlyjs_mode: {plotlyjs_mode}")
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if plotlyjs_mode == "directory":
        _ensure_plotlyjs(destination.parent)
    figure = build_figure(documents, obstacles)
    figure.write_html(
        str(destination),
        include_plotlyjs=plotlyjs_mode,
        full_html=True,
        auto_open=False,
        config={
            "displayModeBar": True,
            "displaylogo": False,
            "scrollZoom": True,
            "showTips": True,
        },
    )
    return destination


def group_by_scenario(documents: Sequence[TraceDocument]) -> dict[str, tuple[TraceDocument, ...]]:
    grouped: defaultdict[str, list[TraceDocument]] = defaultdict(list)
    for document in documents:
        grouped[document.scenario_name].append(document)
    return {
        scenario: tuple(sorted(items, key=lambda item: item.seed))
        for scenario, items in grouped.items()
    }


def render_trace_directory(
    trace_dir: Path | str,
    output_dir: Path | str,
    *,
    obstacles_by_scenario: Mapping[str, Sequence[Obstacle]] | None = None,
    plotlyjs_mode: PlotlyJSMode = "directory",
) -> tuple[Path, ...]:
    trace_paths = sorted(Path(trace_dir).glob("*.json"))
    if not trace_paths:
        raise FileNotFoundError(f"no JSON trace files found in {trace_dir}")
    documents = tuple(TraceDocument.from_path(path) for path in trace_paths)
    groups = group_by_scenario(documents)
    obstacle_map = obstacles_by_scenario or {}
    output_paths: list[Path] = []
    for scenario_name, scenario_documents in sorted(groups.items()):
        output_path = Path(output_dir) / f"{scenario_name}.html"
        obstacles = obstacle_map.get(scenario_name, ())
        output_paths.append(
            render_group_html(
                scenario_documents,
                obstacles,
                output_path,
                plotlyjs_mode=plotlyjs_mode,
            )
        )
    return tuple(output_paths)


__all__ = [
    "PlotlyJSMode",
    "TraceDocument",
    "TraceSample",
    "build_figure",
    "group_by_scenario",
    "render_group_html",
    "render_trace_directory",
]
