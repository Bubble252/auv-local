# Blocker：limiter 改善 tracking，但 true retention 没有转化

日期：2026-11-10<br>
关联 Goal：Goal71<br>
状态：`TRACKING_IMPROVES_OUTCOME_UNCHANGED`<br>
上游：Goal70 `COMMAND_RATE_INFEASIBLE`

## 1. 现象

Goal71 用 Goal59 frozen checkpoints、dev `20–59`、CPU/float64、dwell=50，
只启用既有 guidance command rate limiter，与 Goal67 limiter-off 逐 seed 配对。

| scenario | off -> on true | original lost | recovered orig. | newly lost | true delta | heading err off -> on | pitch err off -> on |
|---|---:|---:|---:|---:|---:|---:|---:|
| default | 99 -> 97 | 9 | 5 | 8 | -2 | `0.930 -> 0.315` | `0.618 -> 0.601` |
| strict125 | 43 -> 44 | 7 | 0 | 3 | +1 | `0.364 -> 0.385` | `0.596 -> 0.410` |

limiter effective 在两组都成立：heading command-rate P90 都降到 `0.524 rad/s`，
exceed fraction 为 0。timeout/safety delta 为 0；pre-goal RMSE 改善
`1.43%/1.47%`；执行器 fin rate limit 与 reachability infeasible fraction 都为 0。

## 2. 已排除项

| 解释 | 状态 | 理由 |
|---|---|---|
| limiter 没生效 | 排除 | config enabled=true；P90 恰好落在 `0.5236 rad/s`，exceed=0 |
| NoisyNet/residual 主导 | 排除 | zero-residual default `19 -> 17`、strict125 `11 -> 11`；没有 residual 也没有收益 |
| 优化/探索失败 | 未证明 | 没有训练；frozen deterministic checkpoint 的闭环被单因素改变 |
| 样本效率失败 | 未证明 | 本 Goal 无训练 |
| 执行器率饱和 | 排除 | `fin_rate_at_limit_fraction=0` |
| command reachability 不可行 | 排除 | infeasible fraction=0 |
| timeout/safety 恶化 | 排除 | delta 均为 0 |
| limiter 完全无作用 | 排除 | H71-B 支持，tracking 明显改善 |

## 3. 关键解释

limiter 降低了 command demand 和 command-side tracking error，但没有让 true
capture 增加。default 的 heading 改善很强，但 speed error 从 `0.493` 恶化到
`0.639 m/s`，false-success 从 `29` 增至 `31`，并且新增丢失 8 个；strict125 的
pitch 改善只在 crossing 前最后 1 s 明显，2 s 窗口仅改善 `0.029 rad`。

因此当前最合理的机制不是“residual 太大”或“算法优化失败”，而是：

1. limiter 改变了接近轨迹和 speed/pitch 耦合；
2. tracking error 相对更平滑的 command 下降，但实际 terminal authority 没有同步
   增强；
3. 最终 true capture 仍由 terminal geometry/estimator/capture 口径决定。

这与 reading note 中 command rate limiter 会改变 control effort/system response
的外部讨论一致，也延续 Goal65/68 的 terminal capture 证据链。

## 4. 两个可证伪 hypothesis

### H72-A：limiter 的 tracking 改善被终端几何变化抵消

支持：逐 seed 对齐 Goal67/Goal71 trace 后，original-lost 的 `eta`/`p_nav`
terminal distance、signed offset 或 plane-miss timing 在 limiter-on 变差，且
变化集中在新增丢失的 seed。

证伪：limiter-on 的终端几何与 limiter-off 基本相同，无法解释新增丢失；则矛盾
转向 controller bandwidth 或 reward/action parameterization。

### H72-B：tracking error 不是 terminal retention 的充分统计量

支持：limiter-on 的 heading/pitch tracking error 改善，但 closing speed、terminal
speed error、或 p_nav/eta 偏差没有改善，且 true capture 不增加。

证伪：terminal 几何/速度指标与 tracking error 同向改善，却仍未 capture；那时应
检查 capture label 或未记录状态，而不是继续调 limiter。

## 5. 最便宜 falsification

只读复用 Goal71 rich trace，按 scenario/checkpoint/action-mode/seed 对齐
Goal67/Goal71：

- crossing 时刻与 crossing 前后 1–2 s 的 `eta`/`p_nav` 距离；
- terminal signed/vertical/horizontal offset；
- speed error、实际 yaw/pitch rate 与 terminal phase；
- plane miss、false-success 与 timeout/safety 的逐 seed 转移。

不训练、不扫 reward/gamma/seed、不打开 held-out；不启动 SAC/REDQ/DroQ。

## 6. 决策

`decision = continue`：继续 terminal capture/闭环几何主线，不换 RL 算法。

下一步先预注册 Goal72 的只读终端几何审计；只有该审计也失败并满足触发条件，才
升级到 controller bandwidth/terminal-aware action parameterization。SAC 与
REDQ/DroQ 仍分别需要独立的 optimization/exploration 与 sample-efficiency 证据。

## 7. 证据

- `doc/goals/goal71_result_20261110.md`
- `research/paper_reading/2026-11-10-goal71-command-rate-tracking-outcome.md`
- `artifacts/lowlevel/goal71_command_rate_limit_20261110/analysis/goal71.json`
- `artifacts/lowlevel/goal71_command_rate_limit_20261110/analysis/goal71.md`
- 服务器原始 run：
  `/data/yhx/underwater-robotics/src/plan_p1/goal60_action_reward_terminal_readonly_audit_20261110/artifacts/goal71_command_rate_limit/`
