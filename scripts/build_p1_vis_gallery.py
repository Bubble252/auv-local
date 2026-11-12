"""Build an HTML gallery and a static contact sheet from 2D rollout artifacts."""

from __future__ import annotations

import argparse
import base64
import html
import importlib.util
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Polygon  # noqa: E402

_RENDER_SCRIPT = Path(__file__).with_name("render_p1_rollout_2d.py")
_SPEC = importlib.util.spec_from_file_location("render_p1_rollout_2d", _RENDER_SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
RENDER = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(RENDER)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build P1 2D visualization gallery")
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--contact-sheet", type=Path, default=None)
    parser.add_argument("--html-output", type=Path, default=None)
    parser.add_argument("--standalone-output", type=Path, default=None)
    parser.add_argument("--dpi", type=int, default=110)
    parser.add_argument("--title", default="P1 2D 避障可视化")
    parser.add_argument(
        "--subtitle",
        default="最终 checkpoint · 解析 13 扇区感知 · 50 Hz · 无起点抖动。",
    )
    parser.add_argument(
        "--final-evaluation",
        type=Path,
        default=None,
        help="Optional final_evaluation.json used to render a result summary.",
    )
    return parser.parse_args()


def _load_runs(directory: Path) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    for metadata_path in sorted(directory.glob("*.json")):
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        gif_path = metadata_path.with_suffix(".gif")
        if not gif_path.is_file():
            continue
        stem = metadata_path.with_suffix("")
        metadata["_metadata_path"] = metadata_path
        metadata["_gif_path"] = gif_path
        metadata["_npz_path"] = stem.with_suffix(".npz")
        metadata["_png_path"] = gif_path.with_name(f"{gif_path.stem}_trajectory.png")
        runs.append(metadata)
    return runs


def _contact_sheet(runs: list[dict[str, Any]], output: Path, dpi: int) -> None:
    columns = 2
    rows = max((len(runs) + columns - 1) // columns, 1)
    figure, axes = plt.subplots(rows, columns, figsize=(11.0, 3.9 * rows), dpi=dpi)
    flat_axes = np.asarray(axes, dtype=object).reshape(-1)
    for ax, run in zip(flat_axes, runs, strict=False):
        cases, key = RENDER._resolve_cases(str(run["scenario_key"]))
        case = RENDER._case_by_key(cases, key)
        RENDER._setup_axes(ax, case)
        data = np.load(run["_npz_path"])
        color = "#2ca02c" if run["success"] else "#d62728"
        ax.plot(
            data["x"],
            data["y"],
            color=color,
            linewidth=2.4,
            alpha=0.95,
            zorder=4,
        )
        final = len(data["x"]) - 1
        polygon = RENDER._vehicle_polygon(
            float(data["x"][final]),
            float(data["y"][final]),
            float(data["yaw"][final]),
        )
        ax.add_patch(
            Polygon(
                polygon,
                closed=True,
                facecolor=RENDER._VEHICLE,
                edgecolor="white",
                linewidth=0.8,
                zorder=6,
            )
        )
        outcome = "SUCCESS" if run["success"] else str(run["failure_reason"]).upper()
        ax.set_title(
            f"{run['scenario_name']} | {run['mode']}\n"
            f"{outcome} | min clear={run['min_clearance_m']:.2f} m | "
            f"t={run['simulation_time_s']:.1f}s",
            fontsize=10,
            color=color,
            loc="left",
        )
        ax.tick_params(labelsize=7)
    for ax in flat_axes[len(runs) :]:
        ax.axis("off")
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, bbox_inches="tight")
    plt.close(figure)


def _data_uri(path: Path, mime_type: str) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def _stage_metric(
    label: str,
    row: dict[str, Any],
    *,
    minimum: int | None = None,
) -> str:
    episodes = int(row["episodes"])
    successes = int(row["successes"])
    requirement = f" / 门槛 {minimum}" if minimum is not None else ""
    return f"""
      <div class="metric">
        <span>{html.escape(label)}</span>
        <strong>{successes}/{episodes}</strong>
        <small>{row["mean_time_to_goal_s"]:.2f}s · clearance {row["mean_min_clearance_m"]:.2f}m{requirement}</small>
      </div>"""


def _evaluation_summary_html(path: Path) -> str:
    data = json.loads(path.read_text(encoding="utf-8"))
    analytic = data["analytic"]
    curriculum = data["curriculum"]
    status = str(analytic.get("status", "UNKNOWN"))
    status_class = "pass" if status == "PASS" else "fail"
    metrics = [
        _stage_metric("S1 无障碍", analytic["s1"]),
        _stage_metric("S2 单圆柱", analytic["s2"]),
        _stage_metric("C1 单圆柱课程", curriculum["c1"]["stage"], minimum=22),
        _stage_metric("C2 左右阻挡", curriculum["c2"]["stage"], minimum=21),
        _stage_metric("C3 连续避障", curriculum["c3"]["stage"], minimum=19),
    ]
    gate = curriculum["c3"]["extra_gate"]["c3_gate_00"]
    yaw_saturation = float(analytic["action_saturation_yaw"]) * 100.0
    both_saturation = float(analytic["both_action_saturation"]) * 100.0
    checkpoint = html.escape(str(analytic["checkpoint_sha256"])[:12])
    return f"""
  <section class="summary">
    <div class="summary-head">
      <div><span>解析最终评估</span><strong class="{status_class}">{html.escape(status)}</strong></div>
      <div><span>Checkpoint</span><strong>C3 best · {checkpoint}</strong></div>
      <div><span>动作饱和</span><strong>yaw {yaw_saturation:.1f}% · both {both_saturation:.1f}%</strong></div>
      <div><span>C3 offset gate</span><strong>{gate["successes"]}/{gate["episodes"]}</strong></div>
    </div>
    <div class="metrics">{"".join(metrics)}
    </div>
    <p class="summary-note">解析 13 扇区 · 120m 量程 · 50Hz · 二维运动学 · 固定评估集。所有场景零碰撞、零越界、零超时；C3 offset gate 的平均最小离障距离约 {gate["mean_min_clearance_m"]:.2f}m。页面中的 GIF 为 seed 0 代表轨迹，统计指标来自完整的固定评估集。</p>
    <p class="summary-note warn">边界：本页只展示解析环境成果，不代表 OceanSim 真实声呐评估；策略的 u_dot 通道基本长期饱和，表现为“全速前进 + 转向避障”。</p>
  </section>"""


def _html_gallery(
    runs: list[dict[str, Any]],
    output: Path,
    *,
    embed: bool = False,
    contact_sheet: Path | None = None,
    title: str = "P1 2D 避障可视化",
    subtitle: str = "最终 checkpoint · 解析 13 扇区感知 · 50 Hz · 无起点抖动。",
    summary_html: str = "",
) -> None:
    cards: list[str] = []
    for run in runs:
        status = "success" if run["success"] else "failure"
        outcome = "成功" if run["success"] else f"失败：{run['failure_reason']}"
        if embed:
            gif_source = _data_uri(run["_gif_path"], "image/gif")
            static_html = ""
            if run["_png_path"].is_file():
                static_source = _data_uri(run["_png_path"], "image/png")
                static_html = f"""
          <details>
            <summary>查看静态轨迹</summary>
            <img class="static" src="{static_source}" alt="{html.escape(run["scenario_name"])} 静态轨迹">
          </details>"""
        else:
            gif_source = html.escape(run["_gif_path"].name)
            static_html = f"""
          <p><a href="{html.escape(run["_png_path"].name)}">静态轨迹</a>
             · <a href="{html.escape(run["_npz_path"].name)}">轨迹数据</a></p>"""
        cards.append(
            f"""
      <article class="card {status}">
        <img src="{gif_source}" alt="{html.escape(run["scenario_name"])}">
        <div class="body">
          <h2>{html.escape(run["scenario_name"])}</h2>
          <p class="badge">{html.escape(outcome)}</p>
          <p>{html.escape(run["mode"])} · seed {run["seed"]} · {run["simulation_time_s"]:.2f}s
             · min clearance {run["min_clearance_m"]:.2f}m</p>
          {static_html}
        </div>
      </article>"""
        )
    if contact_sheet is not None and contact_sheet.is_file():
        source = _data_uri(contact_sheet, "image/png") if embed else html.escape(contact_sheet.name)
        contact_sheet_html = f"""
  <section class="contact">
    <h2>全部轨迹总览</h2>
    <img src="{source}" alt="全部轨迹静态拼图">
  </section>"""
        delivery_note = (
            "单文件内嵌版：GIF、静态轨迹和总览图均已写入 HTML，可单独发送。"
            if embed
            else "本地相册版：图片使用相对路径，需要和同目录 GIF/PNG 一起打开或分发。"
        )
    else:
        contact_sheet_html = ""
        delivery_note = (
            "单文件内嵌版：GIF 和静态轨迹均已写入 HTML，可单独发送。"
            if embed
            else "本地相册版：图片使用相对路径，需要和同目录 GIF/PNG 一起打开或分发。"
        )
    document = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(title)}</title>
  <style>
    body {{ margin: 0; padding: 28px; background: #f3f4f6; color: #111827;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
    h1 {{ margin: 0 0 6px; font-size: 26px; }}
    .sub {{ color: #4b5563; margin: 0 0 22px; }}
    .summary {{ max-width: 1180px; margin: 0 auto 22px; background: white; border-radius: 12px;
                padding: 16px 18px; box-shadow: 0 2px 10px rgba(0,0,0,.08); }}
    .summary-head {{ display: flex; flex-wrap: wrap; gap: 12px 28px; padding-bottom: 12px;
                     border-bottom: 1px solid #e5e7eb; }}
    .summary-head div {{ display: flex; flex-direction: column; gap: 3px; }}
    .summary-head span, .metric span {{ color: #6b7280; font-size: 12px; }}
    .summary-head strong {{ font-size: 15px; }}
    .summary-head .pass {{ color: #15803d; }}
    .summary-head .fail {{ color: #b91c1c; }}
    .metrics {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
                gap: 10px; margin-top: 14px; }}
    .metric {{ background: #f9fafb; border: 1px solid #e5e7eb; border-radius: 9px;
               padding: 10px 12px; display: flex; flex-direction: column; gap: 2px; }}
    .metric strong {{ font-size: 20px; color: #111827; }}
    .metric small {{ color: #6b7280; font-size: 11px; }}
    .summary-note {{ margin: 12px 0 0; color: #4b5563; font-size: 12px; line-height: 1.55; }}
    .summary-note.warn {{ color: #92400e; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(420px, 1fr));
             gap: 18px; }}
    .card {{ background: white; border-radius: 12px; overflow: hidden;
             box-shadow: 0 2px 10px rgba(0,0,0,.08); border-top: 5px solid #2ca02c; }}
    .card.failure {{ border-top-color: #d62728; }}
    .card img {{ display: block; width: 100%; height: auto; background: #f4f1e8; }}
    .body {{ padding: 12px 16px 16px; }}
    .contact {{ max-width: 1180px; margin: 0 auto 22px; background: white; border-radius: 12px;
                padding: 16px; box-shadow: 0 2px 10px rgba(0,0,0,.08); }}
    .contact img {{ display: block; width: 100%; height: auto; }}
    h2 {{ margin: 0 0 6px; font-size: 16px; }}
    p {{ margin: 4px 0; color: #4b5563; font-size: 13px; }}
    .badge {{ font-weight: 700; color: #15803d; }}
    .failure .badge {{ color: #b91c1c; }}
    a {{ color: #1d4ed8; }}
    details {{ margin-top: 8px; }}
    summary {{ cursor: pointer; color: #1d4ed8; font-size: 13px; }}
    img.static {{ margin-top: 10px; border-radius: 8px; }}
  </style>
</head>
<body>
  <h1>{html.escape(title)}</h1>
  <p class="sub">{html.escape(subtitle)}<br>{delivery_note}</p>
  {summary_html}
  {contact_sheet_html}
  <div class="grid">{"".join(cards)}
  </div>
</body>
</html>
"""
    output.write_text(document, encoding="utf-8")


def main() -> int:
    args = _parse_args()
    runs = _load_runs(args.directory)
    if not runs:
        raise SystemExit(f"no rollout JSON artifacts found in {args.directory}")
    contact_sheet = args.contact_sheet or args.directory / "contact_sheet.png"
    html_output = args.html_output or args.directory / "index.html"
    summary_html = (
        _evaluation_summary_html(args.final_evaluation) if args.final_evaluation is not None else ""
    )
    _contact_sheet(runs, contact_sheet, args.dpi)
    _html_gallery(
        runs,
        html_output,
        contact_sheet=contact_sheet,
        title=args.title,
        subtitle=args.subtitle,
        summary_html=summary_html,
    )
    if args.standalone_output is not None:
        _html_gallery(
            runs,
            args.standalone_output,
            embed=True,
            contact_sheet=contact_sheet,
            title=args.title,
            subtitle=args.subtitle,
            summary_html=summary_html,
        )
    print(
        json.dumps(
            {
                "runs": len(runs),
                "contact_sheet": str(contact_sheet),
                "html": str(html_output),
                "standalone_html": (
                    str(args.standalone_output) if args.standalone_output is not None else None
                ),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
