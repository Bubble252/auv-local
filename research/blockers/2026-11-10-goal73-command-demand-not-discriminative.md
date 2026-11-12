# Blocker：command-demand/曲率不能区分 newly-lost，false-success 主要是终端 p_nav 偏差

日期：2026-11-10<br>
关联 Goal：Goal73<br>
状态：`COMMAND_DEMAND_NOT_SUPPORTED`<br>
上游：Goal72 `LIMITER_GEOMETRY_TRADEOFF`；Goal70 `COMMAND_RATE_INFEASIBLE`

## 1. 现象

Goal73 Step 1 只读复用 Goal67/Goal71 rich trace，按 Goal72 的
`newly_lost` / `both_true` 分组，计算 pre-terminal 1 s 的 command demand。
结果没有支持 H73-A：

| scenario | group | n | heading rate P90 (rad/s) | pitch rate P90 (rad/s) | saturation | curvature (1/m) | speed (m/s) |
|---|---|---:|---:|---:|---:|---:|---:|
| default | newly_lost | 8 | 0.5236 | 0.2618 | 0.1765 | 0.0675 | 1.7270 |
| default | both_true | 59 | 0.5236 | 0.2618 | 0.2157 | 0.0460 | 1.3388 |
| strict125 | newly_lost | 3 | 0.4082 | 0.2618 | 0.0980 | 0.0107 | 1.1877 |
| strict125 | both_true | 23 | 0.5236 | 0.2618 | 0.3725 | 0.0352 | 1.6473 |

- heading/pitch command rate 在两组都顶到 limiter 的上下限（`0.5236` /
  `0.2618 rad/s`），已经不是区分变量；
- default 的曲率代理只高 `+0.0215 1/m`，刚过 `0.02` 阈值；strict125 反而是
  `-0.0245 1/m`，方向相反；
- material saturation 在 default/strict125 的 newly-lost 都**低于** both-true；
- mean command speed 在 default 更高（`+0.388 m/s`），但 strict125 更低
  （`-0.460 m/s`），不一致。

因此 H73-A 证伪：command-demand/曲率不是当前 newly-lost 的稳定区分变量。
按预注册，不实现 curvature/command-demand speed governor。

## 2. 终端捕获证据

同一次审计输出了 false-success 的终端明细。关键事实是：所有 false-success
都是 `sphere` 捕获，`p_nav` 距离刚好压在 tight radius 内，而 true `eta`
距离仍在半径外：

| scenario | group | n | terminal p_nav-eta error (m) | terminal eta dist (m) |
|---|---|---:|---:|---:|
| default | newly_lost | 8 | 0.4475 | 2.3464 |
| default | both_true | 59 | 0.4012 | 1.5366 |
| strict125 | newly_lost | 3 | 0.6131 | 1.4870 |
| strict125 | both_true | 23 | 0.4184 | 0.8226 |

- default 的 false-success `p_nav` 距离约 `1.71–1.75 m`（tight radius
  `1.75 m`），true `eta` 距离约 `2.01–2.88 m`；
- strict125 的 false-success `p_nav` 距离约 `0.94–0.99 m`（tight radius
  `1.00 m`），true `eta` 距离约 `1.25–1.62 m`；
- `p_nav-eta` 误差中位约 `0.45 m`（default）和 `0.61 m`（strict125），
  远大于 `p_nav` 距离到半径的余量（约 `0.01–0.04 m`）。

这不是“plane crossing 大横向 offset”，而是 **p_nav 估计误差在 capture
边界附近翻转标签**。控制器是按 `p_nav` 捕获的；true `eta` 未进半径时，
`success=true, terminal_eta_capture=false` 是协议与真实状态的差异。

## 3. 已排除项

| 解释 | 状态 | 理由 |
|---|---|---|
| command rate/曲率主导 | 排除 | 两组都顶到 limiter；strict125 方向相反 |
| material saturation 主导 | 排除 | newly-lost 饱和更低 |
| NoisyNet/residual 主导 | 排除 | Goal72 C 负控；deterministic mean；zero-residual 不恢复 |
| residual gate 关闭导致无权限 | 排除 | gate 在 5 m 内为 1；false-success 的 gate distance 约 1–1.8 m |
| plane crossing 大 offset | 排除 | 所有 false-success 都是 `sphere` 模式，p_nav 距离在半径内 |
| timeout/safety | 排除 | Goal72/73 均为 0 |

## 4. 可证伪下一步

### H74-A：capture margin 收紧会按 p_nav 边界余量减少 false-success

机制：false-success 的 `p_nav` 距离只比半径小 `0.01–0.04 m`；把 capture
条件改为 `p_nav distance <= radius - margin` 会让这些边界样本退出 success。

支持：在 `margin=0.10/0.25/0.50 m` 的只读重分类中，false-success 单调下降，
且下降幅度与 `p_nav` 距离分布一致；同时报告 true success 的损失。

证伪：margin 收紧不能减少 false-success，或减少的样本并不是边界样本。

### H74-B：p_nav-eta 误差是系统偏差而不是零均值噪声

机制：如果误差沿 goal 方向有稳定符号/幅度，可以做方向性 bias correction；
如果只是零均值噪声，margin 收紧会同时损失 true success。

支持：terminal `p_nav - eta` 在 goal 方向上的投影中位数显著非零，且
false-success 与 true-success 的偏差分布不同。

证伪：投影中位数跨零，或 false-success/true-success 分布无差异。

### H74-C：动作参数化不是当前主因

机制：gate 在 5 m 内为 1，residual 有权限；如果 margin 重分类已经解释
false-success，则不应继续加控制 authority。

支持：H74-A 成立且 residual/gate 指标在 false-success 与 true-success
之间不提供额外区分。

## 5. 最便宜 falsification

只读重分类现有 Goal71 episodes，不新增 rollout、不训练：

1. 对每个 episode 计算 `p_nav_distance`、`eta_distance`、`p_nav-eta`
   在 goal 方向的投影；
2. 在 margin `0 / 0.10 / 0.25 / 0.50 m` 下重新计算
   `success_margin` 与 `true_capture_margin`；
3. 报告 false-success、true success、plane miss、timeout/safety 的 paired
   变化；
4. 只有 H74-A/H74-B 支持，才预注册 capture margin 或 bias correction 的
   单因素 arm；否则继续查动作参数化/奖励。

## 6. 决策

`decision = pivot`：停止 command-demand/curvature speed governor，转入
terminal capture / p_nav estimator bias 审计。不是换 RL 算法；SAC/REDQ/DroQ
仍要求独立的 optimization/exploration 或 sample-efficiency 证据。

## 7. 证据

- `doc/goals/goal73_command_demand_speed_governor_20261110.md`
- `scripts/analyze_pc_goal73_command_demand.py`
- `tests/test_pc_goal73_command_demand.py`
- `artifacts/lowlevel/goal73_command_demand_audit_20261110/analysis/goal73.json`
- `artifacts/lowlevel/goal73_command_demand_audit_20261110/analysis/goal73.md`
- `research/paper_reading/2026-11-10-goal73-terminal-capture-estimator-bias.md`
