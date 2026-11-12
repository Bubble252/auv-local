# Blocker：limiter 改善 terminal tracking，但恶化 horizontal/signed 终端几何

日期：2026-11-10<br>
关联 Goal：Goal72<br>
状态：`LIMITER_GEOMETRY_TRADEOFF`<br>
上游：Goal71 `TRACKING_IMPROVES_OUTCOME_UNCHANGED`

## 1. 现象

Goal72 只读复用 Goal67/Goal71 rich trace，逐 seed 对齐 limiter-off/on。
总体仍没有 true retention 净收益：

| scenario | off true | on true | original lost | recovered | newly lost | true delta |
|---|---:|---:|---:|---:|---:|---:|
| default | 99 | 97 | 9 | 5 | 8 | -2 |
| strict125 | 43 | 44 | 7 | 0 | 3 | +1 |

但 `newly_lost` 的 terminal geometry 与 tracking 出现明确分离：

- default 8 个 newly lost 中，terminal `eta` 中位恶化 `+0.668 m`，
  `p_nav` 恶化 `+0.770 m`，`abs(signed offset)` 恶化 `+1.736 m`，
  horizontal offset 恶化 `+1.483 m`；任一几何量恶化 `>=0.10 m` 的比例
  为 `8/8`。
- strict125 3 个 newly lost 同向：terminal `eta +0.284 m`、
  `abs(signed offset) +0.182 m`、horizontal offset `+0.282 m`，
  worse fraction `3/3`。
- default 的 heading tracking 改善 `0.577 rad`、pitch 改善 `0.160 rad`
  （crossing 前 1 s），但 closing speed 反而变差 `0.295 m/s`，
  terminal speed error 变差 `0.063 m/s`。
- 改善集中在最后 1 s：default heading improvement 从 `0.577`（1 s）降到
  `0.216`（2 s）、`0.030`（3 s）。

新增丢失不是单一平面 miss：

- default：5 个 `GOAL_PLANE_MISS`，3 个 goal termination 但
  `terminal_eta_capture=false` 的 false-success；
- strict125：1 个 plane miss，2 个 false-success。

这说明 limiter 既可能把 original-lost 拉回 goal shell，也可能把原本 true
轨迹推成 plane miss 或 `p_nav` 已进、`eta` 未进的 false-success。

## 2. 已排除项

| 解释 | 状态 | 理由 |
|---|---|---|
| NoisyNet 采样主导 | 排除 | 正式评估 `evaluate=True`、`sample_noise=False`；s240/s752/s880 noise 远小于 deterministic action；s1008 noise 较大但 default 只 `17->16` |
| residual 幅度主导 | 排除 | default newly-lost residual gate/abs 低于 both-true；strict gate 高但 zero-residual delta `0` |
| 执行器 rate saturation | 排除 | 已有 `fin_rate_at_limit_fraction=0` |
| command reachability 不可行 | 排除 | reachability infeasible fraction `0` |
| timeout/safety 恶化 | 排除 | 两个 scenario 的 delta 都为 `0` |
| 只是整体尺度变大 | 排除 | default vertical offset 反而改善 `-0.381 m`，恶化集中在 horizontal/signed/plane |
| 优化/探索失败 | 无证据 | 本 Goal 无训练，frozen deterministic checkpoint |
| 样本效率失败 | 无证据 | 本 Goal 无训练 |

## 3. 机制判断

当前最合理的机制是：command rate limiter 降低了 terminal-local 的
heading/pitch tracking error，但没有提供足够的横向/平面方向 authority，
导致 horizontal/signed offset 变大；当 offset 落在 capture 边界附近时，
就会出现 `p_nav` 已进目标但 `eta` 未进目标的 false-success。

这不是“residual 太大”的问题，也不是“NoisyNet 把动作推乱”的问题。历史上
Goal7/Goal58/59 的 NoisyNet 问题仍要保留为训练期风险，尤其是 s1008；
但当前 frozen-policy 闭环失败不能归因于它。

## 4. 可证伪下一步

### H73-A：terminal-aware action parameterization 能把 horizontal/signed offset 压回

预注册单因素 arm，保持 reward/gamma/seed/checkpoint/eval split 不变，只改变
terminal 阶段的 action 参数化或横向 authority。

支持：default newly-lost 的 terminal `abs(signed offset)` 和 horizontal offset
中位改善 `>=0.10 m`，且 false-success 不增加，true delta `>=+3`。

证伪：offset 没有改善，或改善只发生在 recovered 子集而 newly-lost 不变。

### H73-B：false-success 是 capture 口径问题，不是控制问题

如果 terminal-aware action 让 `p_nav`/`eta` 更接近，但 goal-terminated 且
`terminal_eta_capture=false` 的比例不降，则先审计 capture 口径和未记录状态，
不继续加控制 authority。

## 5. 决策

`decision = continue`：继续 terminal capture/动作参数化主线，不换 RL 算法。

下一步预注册 Goal73，优先比较：

1. terminal-aware action parameterization（horizontal/signed authority）；
2. controller bandwidth 的单因素 arm；
3. 若 false-success 仍主导，先做 capture label/state 审计。

SAC 仍要求独立的 optimization/exploration 失败证据；REDQ/DroQ 仍要求独立的
sample-efficiency 失败证据。当前两者都没有。

## 6. 证据

- `doc/goals/goal72_terminal_geometry_limiter_20261110.md`
- `doc/goals/goal72_result_20261110.md`
- `scripts/analyze_pc_goal72_terminal_geometry.py`
- `tests/test_pc_goal72_terminal_geometry.py`
- `artifacts/lowlevel/goal72_terminal_geometry_20261110/analysis/goal72.json`
- `artifacts/lowlevel/goal72_terminal_geometry_20261110/analysis/goal72.md`
