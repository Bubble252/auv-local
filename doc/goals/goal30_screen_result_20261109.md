# Goal30 screen 结果：可执行链有效，但 ever-capture 不等于 terminal capture

日期：2026-11-09<br>
裁决：`SCREEN_PASS_CHAIN_VALID_METRIC_CORRECTED`（不是效果 PASS，也不是策略 FAIL）

## 1. 运行范围

- baseline：B0，`ALOS -> PD/PI -> actuator`，`zero-command-residual`。
- candidate：同一 B0，仅 `terminal_alignment_min_m/max_m = 30.0`。
- checkpoint：Goal27 C1 `final.pt`，SHA256
  `0851496b15ffebbc0a19bafa5bf40c0493c0d9a3f485267b75fde5f811dde5d9`。
- seeds：`41`；default 与 strict125 各一次 paired baseline/candidate。
- 协议：`endpoint_stop`、`realistic_calibrated`、CPU/float64、`path_dim=8`。
- 未训练、未打开 held-out、未跑 40-seed formal。

## 2. 控制链审计

| scenario | chain valid | actuator max diff | eta max diff | nu max diff | command max diff | actuator change steps |
|---|---:|---:|---:|---:|---:|---:|
| default | true | 5.7734 rpm | 1.2910 m | 0.0581 | 0.8621 rad | 1/1 seed |
| strict125 | true | 5.7734 rpm | 1.2910 m | 0.0581 | 0.8621 rad | 1/1 seed |

结论：Goal29 的 `INVALID_CONTROL_CHAIN` 不再出现。candidate 的 guidance 变化确实
进入 `actual_actuator`，并改变了 `eta`。因此本 screen 可以用于讨论 H1，但不能用
单 seed 宣称统计收益。

## 3. 终端指标

default（goal radius `2.0 m`）：

| arm | success_protocol | success_eta (ever) | terminal_eta_capture | plane miss | min eta dist (m) | final eta dist (m) | horizontal-normal offset (m) | terminal yaw (rad) | speed ratio | saturation |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 0.0 | 0.0 | 0.0 | 1.0 | 2.0033 | 2.2235 | 1.9811 | 0.3590 | 1.0000 | 0.1980 |
| candidate | 0.0 | 1.0 | 0.0 | 1.0 | 1.8093 | 2.0626 | 1.9544 | 0.3942 | 0.9998 | 0.1386 |

strict125（goal radius `1.25 m`）：

| arm | success_protocol | success_eta (ever) | terminal_eta_capture | plane miss | min eta dist (m) | final eta dist (m) | horizontal-normal offset (m) | terminal yaw (rad) | speed ratio | saturation |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 0.0 | 0.0 | 0.0 | 1.0 | 2.0033 | 2.2235 | 1.9811 | 0.3590 | 1.0000 | 0.1980 |
| candidate | 0.0 | 0.0 | 0.0 | 1.0 | 1.8093 | 2.0626 | 1.9544 | 0.3942 | 0.9998 | 0.1386 |

paired delta（candidate - baseline，n=1，不能给 CI）：

| scenario | success_protocol | success_eta | terminal_eta_capture | plane miss | horizontal offset | terminal yaw | speed ratio | saturation |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| default | 0.0 | +1.0 | 0.0 | 0.0 | -0.0267 m | +0.0352 rad | -0.0002 | -0.0594 |
| strict125 | 0.0 | 0.0 | 0.0 | 0.0 | -0.0267 m | +0.0352 rad | -0.0002 | -0.0594 |

## 4. 关键修正：ever-capture 不是 terminal capture

default candidate 的 `goal_capture_eta=True` 来自它在 step `1146` 曾进入 `1.81 m`
半径，但 termination 时距离为 `2.06 m`，仍是 plane miss。baseline 的最小距离是
`2.0033 m`，刚好在 `2.0 m` 半径外。因此：

- `success_eta` 的 `0 -> 1` 是“曾经进入半径”，不是“终端捕获成功”；
- 若直接用 `success_eta` 作为 formal 主指标，会把 plane miss episode 判成收益；
- 本 Goal 的 formal 主指标改为 `terminal_eta_capture`，即 termination 时
  `eta_distance <= goal_radius`；`success_eta` 保留为诊断量。

该修正在任何 formal 数据产生之前写入预注册文档。修正方向是保守的：它移除了
screen 表面上唯一一个 `+1` 的主指标收益。

## 5. 其他观察

- candidate 的 terminal horizontal-normal offset 只下降 `0.0267 m`（约 `1.3%`），
  远低于 formal Gate 的 `0.25 m / 10%`。
- candidate 的 terminal yaw error 反而上升 `0.0352 rad`。
- terminal speed ratio 基本不变，saturation 下降 `0.0594`，不支持“收益来自更大
  速度/权限”的解释。
- default candidate 的 full path 3D RMSE 从 `1.9089 m` 升到 `2.3931 m`；这与
  “更早对准目标会牺牲路径跟踪”一致。path RMSE 不用于 terminal Gate，但必须报告。
- 两个 scenario 的几何轨迹完全相同；strict125 只改变 goal radius，因此只影响
  capture 口径，不影响 controller 行为。

## 6. 裁决与下一步

| 观察 | 判读 | 下一步 |
|---|---|---|
| actuator/eta 随 guidance 改变 | 控制链有效 | 允许 40-seed formal |
| `terminal_eta_capture` 在 screen 无改善 | 单 seed 无收益信号 | formal 仍按预注册执行，用来估计 40-seed 方向和噪声 |
| `success_eta` 仅 ever-capture 上升 | 指标歧义 | formal 不使用它单独判 PASS |
| horizontal offset 仅 -0.027 m，yaw 变差 | 效果很弱且混合 | 不因 screen 宣称 H1 成立 |

允许：跑 default/strict125 × baseline/candidate 的 40-seed dev formal。
禁止：因为 screen 的 `success_eta +1` 直接判 PASS；禁止扫 alignment window、seed、
gain 或 reward；禁止打开 held-out。

## 7. 复现

```bash
cd /home/bubble/水下机器人/settled_局部规划器_反应式
OUT=artifacts/lowlevel/goal30_executed_terminal_capture_20261109

# baseline / candidate；default 与 strict125 同理替换环境配置
.venv/bin/python scripts/evaluate_pc_ppo.py \
  --training-config configs/lowlevel/pc_goal27_c1_ppo_default_27001.json \
  --checkpoint /tmp/goal30_evidence/c1_final.pt \
  --environment-config configs/lowlevel/pc_goal27_c1_command_residual_default_20261109.json \
  --action-mode zero-command-residual \
  --seed-start 41 --seed-count 1 \
  --device cpu --dtype float64 \
  --output "$OUT/screen/seed41/default/baseline" \
  --trace-output "$OUT/screen/seed41/default/baseline/trace.npz"

# candidate 额外加：
# --guidance-config configs/lowlevel/pc_goal29_terminal30_guidance_20261109.json

.venv/bin/python scripts/analyze_pc_goal30_traces.py \
  --trace "default:baseline=$OUT/screen/seed41/default/baseline" \
  --trace "default:candidate=$OUT/screen/seed41/default/candidate" \
  --trace "strict125:baseline=$OUT/screen/seed41/strict125/baseline" \
  --trace "strict125:candidate=$OUT/screen/seed41/strict125/candidate" \
  --output "$OUT/screen/seed41"
```

证据（不提交）：

```text
artifacts/lowlevel/goal30_executed_terminal_capture_20261109/screen/seed41/analysis.json
artifacts/lowlevel/goal30_executed_terminal_capture_20261109/screen/seed41/analysis.md
artifacts/lowlevel/goal30_executed_terminal_capture_20261109/screen/seed41/{default,strict125}/{baseline,candidate}/summary.json
artifacts/lowlevel/goal30_executed_terminal_capture_20261109/screen/seed41/{default,strict125}/{baseline,candidate}/trace.npz
```
