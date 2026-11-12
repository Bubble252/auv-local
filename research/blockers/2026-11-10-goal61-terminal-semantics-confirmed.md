# Blocker: Goal61 确认 terminal success/capture 语义错位

日期：2026-11-10<br>
Goal：Goal61<br>
状态：`CONFIRMED_PIVOT_TO_TERMINAL_CALIBRATION`

## 1. 触发条件

Goal59 `FAIL_BOUNDED_RESIDUAL` 后，Goal60 发现 protocol success 与 true-state
terminal capture 系统性不一致。Goal61 在相同 Goal60 trace 上做只读复算，确认
这不是 residual 策略特有现象，而是 `p_nav` success 与 `eta` capture 的 terminal
语义错位。

本项只读取 trace，不训练、不新增 rollout、不打开 held-out。

## 2. 现象

Policy pooled 的 success_no_capture：

| scenario | n | p_nav closer | median gap m | gap 95% CI | direction>0 | inside/outside |
|---|---:|---:|---:|---|---:|---:|
| default | 58 | `1.000` | `-0.200` | `[-0.329, -0.163]` | `1.000` | `1.000` |
| strict125 | 36 | `1.000` | `-0.242` | `[-0.321, -0.175]` | `1.000` | `1.000` |

Forced zero-residual 也复现：

| scenario | n | p_nav closer | median gap m | gap 95% CI | direction>0 | inside/outside |
|---|---:|---:|---:|---|---:|---:|
| default | 13 | `1.000` | `-0.205` | `[-0.484, -0.109]` | `1.000` | `1.000` |
| strict125 | 13 | `1.000` | `-0.134` | `[-0.503, -0.057]` | `1.000` | `1.000` |

## 3. 已排除项

- 不是 trace 数据错误：40 seeds、40 terminal 行、finite、checkpoint SHA 均通过；
- 不是 held-out 泄漏：`100–149` 未打开；
- 不是 checkpoint 选择：只用预注册 `final.pt`；
- 不是 NoisyNet：Goal59 的 8/8 deterministic mean 已饱和，`noise/det` 中位数
  只有 `0.0193`；
- 不是样本效率：Goal60 训练曲线最后 10% residual ratio 变化均值
  `-0.00022`，已平台化；
- 不是 residual 特有：zero-residual 在两场景仍有 13 个 success_no_capture。

## 4. 可证伪 hypothesis

### H-T1：tighten p_nav terminal radius 能减少 false success

在 frozen traces 上离线重放更严格的 p_nav 半径；若 false success 降低但
true capture 的 recall 和 extra-step 代价不可接受，则 H-T1 不成立。

### H-T2：dwell 条件比单步半径更稳

要求 p_nav 在收紧半径内保持一段 dwell time；若 dwell 不能改善
success/capture 对齐，或显著增加 timeout/extra steps，则 H-T2 不成立。

### H-T3：uncertainty-aware margin 能解释并修正偏差

用在线可得的 nav age/uncertainty 构造保守 margin；若 margin 与 eta 误差方向
不一致，或修正后仍然大量 false success，则 H-T3 不成立。

## 5. 排查顺序

按 `数据 → 标签 → 评测 → baseline → 架构 → 文献`：

1. 数据和 terminal row 已通过；
2. 标签：success 使用 p_nav，capture 使用 eta，已对齐到 episode；
3. 评测：zero-residual 对照确认不是策略特有；
4. baseline：保留现有 controller 和 checkpoint，不改策略；
5. 架构：下一 Goal 只做 termination/calibration 的只读反事实；
6. 文献：记录见 `research/paper_reading/2026-11-10-goal61-terminal-alignment.md`。

## 6. 决策

`decision = pivot`。停止把 residual 参数化、NoisyNet 或换算法作为当前修复；
下一 Goal 预注册 terminal calibration/termination 反事实。SAC/REDQ/DroQ 不启动，
held-out 保持关闭。

## 7. 证据

- `doc/goals/goal61_result_20261110.md`
- `artifacts/lowlevel/goal61_terminal_alignment_20261110/goal61_terminal_alignment.json`
- `artifacts/lowlevel/goal61_terminal_alignment_20261110/goal61_terminal_alignment.md`
- 服务器
  `/data/yhx/underwater-robotics/src/plan_p1/goal60_action_reward_terminal_readonly_audit_20261110/artifacts/traces/analysis/`
