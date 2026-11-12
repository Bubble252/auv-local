# Blocker: 可执行链上的更早终端对准反而降低捕获

日期：2026-11-09<br>
Goal：Goal30<br>
状态：`OPEN_PIVOT_TO_TERMINAL_CONTROL_DESIGN`

## 1. 现象

Goal29 证明 C3 actor 不执行 ALOS guidance，因此 Goal30 切到真正执行的 B0 链路：

```text
ALOS command -> zero-command-residual -> PD/PI -> [rudder, stern, rpm]
```

在 dev seeds `20–59` 上，把 `terminal_alignment_min_m/max_m` 从 `20 m` 前移到
`30 m`，结果稳定变差：

- default：`success_protocol` `0.725 -> 0.600`，plane miss `0.275 -> 0.400`；
- strict125：`success_protocol` `0.500 -> 0.400`，plane miss `0.500 -> 0.600`；
- `terminal_eta_capture` default `0.125 -> 0.100`，strict125 `0.075 -> 0.025`；
- paired terminal eta offset 变差约 `0.10 m`（CI 不跨零）；
- terminal speed ratio 不变，terminal material saturation 没有升高。

控制链审计显示 40/40 seeds 的 actuator 和 eta 都改变，因此不是 Goal29 的
`INVALID_CONTROL_CHAIN`。

## 2. 证据

- `doc/goals/goal30_result_20261109.md`
- `doc/goals/goal30_screen_result_20261109.md`
- `artifacts/lowlevel/goal30_executed_terminal_capture_20261109/analysis.json`
- `artifacts/lowlevel/goal30_executed_terminal_capture_20261109/failure_audit.json`
- `artifacts/lowlevel/goal30_executed_terminal_capture_20261109/formal/{default,strict125}/{baseline,candidate}/summary.json`

关键 paired 数字：

| scenario | success_protocol Δ | plane miss Δ | terminal eta offset Δ | 3D path RMSE Δ | vertical path RMSE Δ |
|---|---:|---:|---:|---:|---:|
| default | -0.125 | +0.125 | +0.102 m | +0.338 m | +0.367 m |
| strict125 | -0.100 | +0.100 | +0.102 m | +0.334 m | +0.364 m |

## 3. 已排除项

- **不是控制链断开**：candidate 的 `actual_actuator` 在 40/40 seeds 改变，
  `eta` 也随之改变；Goal29 的逐位一致没有复现。
- **不是数值错误**：0 non-finite，0 timeout，0 safety。
- **不是速度过高**：terminal speed ratio paired delta 约 `+0.0002`，CI 跨零。
- **不是执行器饱和/权限不足**：candidate 的 terminal saturation 没有升高；
  strict125 反而下降约 `0.02`。
- **不是单纯 success 口径**：`terminal_eta_capture` 与 `success_protocol` 同方向
  变差；ever-capture `success_eta` 也没有改善。
- **不是 seed 交换**：default 6 seeds 变差 / 1 seed 变好，strict125 5 / 1。

## 4. 可证伪 hypothesis

### H1：30 m 窗口相对约 35 m 的参考路径过长，导致路径切割和终端捕获变差

机制：参考路径总长约 `35 m`。30 m 窗口让 ALOS 在轨迹的大部分时间直接指向目标，
terminal phase 占比从 baseline 的约 `50%` 升到 `82%`，放弃了路径跟踪中的垂直/横向
修正。路径被切短后，车辆在目标平面附近的几何误差更大。

若成立应看到：

- candidate 的 terminal phase fraction 显著升高；
- 3D path RMSE 和 vertical path RMSE 变差；
- terminal eta offset 与 plane miss 同方向变差；
- horizontal path RMSE 可能略好，但不能抵消垂直/终端误差。

若失败应看到：terminal phase fraction 不升高，或 path RMSE/terminal offset 不变。

最便宜证伪：现有 40-seed trace 的只读分解；已执行。

结果：**支持**。terminal phase fraction `+0.3228`（default）/ `+0.3162`
（strict125），3D RMSE `+0.338/+0.334 m`，vertical RMSE `+0.367/+0.364 m`；
horizontal RMSE 仅 `-0.067/-0.066 m` 且 CI 跨零。

### H2：固定 PD/PI 的终端横向权限/饱和是主因

机制：更早对准要求更大的 rudder/stern 修正；如果执行器饱和或 rate clip，车辆无法
在目标平面附近收敛。

若成立应看到：candidate 的 terminal material saturation、rudder/stern 限幅或
rate clip 显著高于 baseline，且与 terminal offset 同方向。

若失败应看到：saturation 不升高，或 terminal offset 变差但 saturation 更低。

最便宜证伪：同一 trace 的 terminal saturation 与 rudder/stern 统计；已执行。

结果：**证伪**。default saturation paired median `0.0000`，CI 跨零；strict125
`-0.0198`，CI 全负。rudder/stern 均值只小幅升高，未出现饱和解释。

### H3：terminal yaw error 是横向 miss 的主要解释

机制：更早对准使航向更早指向目标，但如果终端航向误差仍然大，横向 offset 会由
`distance * sin(yaw_error)` 主导。

若成立应看到：terminal yaw error 与 terminal eta offset 强相关，并且 candidate 的
yaw error 明显改善或至少与 offset 同方向。

若失败应看到：yaw-offset 相关性弱，或 yaw 改善但 offset 不变/变差。

最便宜证伪：现有 trace 的 yaw-offset 相关系数；已执行。

结果：**不支持**。相关系数 default `0.033/0.020`，strict125 `0.124/0.156`；
candidate 的 terminal yaw error 反而略差（`+0.016/+0.018 rad`）。

### H4：需要保留路径跟踪的终端横向/垂直收敛控制器，而不是更早 goal seeking

机制：如果 30 m 直接对准破坏路径跟踪，那么正确方向不是继续前移窗口，而是在最后
一段路径上同时保留路径切线/横向误差反馈和终端收敛目标，并显式限制纵向速度/过平面
几何。

若成立应看到：新控制器在不增加 3D path RMSE 的前提下降低 terminal horizontal/
vertical offset 和 plane miss；terminal saturation 不升高。

若失败应看到：terminal offset 不改善，或 path RMSE/plane miss 变差。

最便宜证伪：先做离线/回放控制器设计审计，不训练网络；再预注册一个新的单因子
terminal controller Gate。

结果：待下一轮预注册，不在本 blocker 中直接实现。

## 5. 当前裁决

- 停止 `terminal_alignment_min_m/max_m` 参数搜索；
- 不重开 command residual PPO、reward/gamma/seed 搜索；
- 不打开 held-out `100–149`；
- 进入 `pivot`：从“更早对准窗口”转向“保留路径跟踪的终端横向/垂直收敛控制”。

## 6. 下一步

1. 用 `paper-search` 检索 AUV terminal homing、cross-track control、path following
   vs goal seeking、underactuated terminal capture；
2. 写 `research/paper_reading/2026-11-09-goal30-terminal-capture-control.md`；
3. 基于文献和本 blocker 设计 Goal31 的单因子 terminal controller；
4. 在 `research/LEDGER.jsonl` 写 `decision: pivot` 和 `next`；
5. 在 Goal31 预注册前不跑新的长实验。
