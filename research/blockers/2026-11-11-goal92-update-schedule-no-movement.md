# Blocker：Goal92 更新调度 screen 全部零移动，PPO 增量疑似被动作包线截断

日期：2026-11-11<br>
Goal：Goal92 `FAIL_NO_UPDATE_SCHEDULE_EFFECT`<br>
状态：`ACTIVE / ACTION_AUTHORITY_CLIPPING_SUSPECT / NEXT_GOAL93_EVAL_ONLY`

## 1. 现象

在同一 frozen P-A BC、corrected gate、scale=1.25、1M transitions、dev40 下，
6 个 PPO 更新调度 arm 的闭环结果几乎完全重合：

| arm | default | strict125 | PPO residual abs mean | magnitude clip rate | rate clip rate |
|---|---:|---:|---:|---:|---:|
| `PG10_3` | 31/40 | 20/40 | 0.0195 | 0.693 | 0.492 |
| `E4` | 31/40 | 20/40 | 0.0133 | 0.694 | 0.493 |
| `C10` | 31/40 | 20/40 | 0.0263 | 0.689 | 0.490 |
| `C20` | 31/40 | 20/40 | 0.0251 | 0.689 | 0.491 |
| `AN` | 31/40 | 20/40 | 0.0201 | 0.691 | 0.491 |
| `W0` | 31/40 | 19/40 | 0.0167 | 0.691 | 0.491 |

`C10/C20` 把 PPO 确定性残差放大约 `35%/29%`，但：

- success 逐 episode gain/loss 为 `0/0`；
- path 变化小于 `0.002 m`；
- `actuator_residual_magnitude_clip_rate` 和 `actuator_residual_rate_clip_rate`
  与 baseline 完全相同（约 `0.69/0.49`）。

## 2. 新证据：动作包线可能把 PPO 增量截掉

`BatchedPCActuatorResidualEnv._apply_residual` 的执行链是：

```text
requested = ALOS + scale * (r_BC + gate * r_PPO)
limited   = clamp(requested, actuator_envelope)
executed  = rate_limit(limited)
effective = executed - ALOS
```

Goal92 的 summary 显示约 `69%` 的 step 触发 magnitude clip、`49%` 触发 rate clip。
在 `scale=1.25` 下，`r_BC` 已经不是“小残差”，而是把 D3 动作放大 `1.25` 倍；
`r_PPO` 只占 `r_BC` 的约 `6–9%`，它的增量很可能在包线/速率限制处被丢弃。

这给出比“更新调度不对”更便宜、更可证伪的解释：

```text
PPO residual requested +35%
but effective actuator change ≈ 0
because envelope/rate clipping absorbs the increment
```

## 3. 已排除项

1. **不是 NoisyNet 主导**：这些配置没有 NoisyNet；训练噪声是 PPO Gaussian
   `init_noise_std≈0.10`，且 Goal91 已证伪“降低噪声能改善结果”。
2. **不是 gate scope**：Goal90 corrected gate 下 frozen-base 与 policy 的
   path 不退化，gate 只作用于 PPO 增量。
3. **不是网络接线**：Goal89 的 22 个 `frozen_bc_*` entry 与 Goal86 BC
   checkpoint 逐张量 `max abs diff=0.0`；主干与 Goal26/Goal80 同构。
4. **不是更新次数/clip/advantage/warmup**：Goal92 已逐一 screen，全部零移动。
5. **不是单纯的探索幅度**：Goal91 的 `N03/N05` 没有放大有效残差，也没有提升
   strict125。

## 4. 可证伪 hypothesis

### H92-X1：PPO 增量被执行器包线/速率限制截断，闭环看到的是近似零增量

- 机制：`scale=1.25` 下 `r_BC` 已接近或超过执行器包线，`r_PPO` 的额外请求
  大部分落在 `clamp` 或 rate limit 之外。
- 成立应看到：`effective/applied` 的 L2 生存率低，且 `C10` 的生存率不高于
  `PG10_3`；`C10 - PG10_3` 的 effective residual 差远小于 requested 差。
- 失败应看到：`effective/applied` 接近 1，且 C10 的 effective 增量明显大于
  PG10_3。
- 最便宜证伪：Goal93 对 `PG10_3/C10` 各 40 seeds 记录 trace，直接比较
  `actuator_residual_applied` 与 `actuator_residual_effective`。

### H92-X2：总残差 scale=1.25 本身过冲，scale=1.0 反而减少 clip 且不丢 success

- 机制：`r_BC` 的目标是重建 D3 动作；`scale=1.25` 把它放大到 D3 之外，
  造成大量包线饱和。
- 成立应看到：scale=1.0 的 magnitude clip rate 显著低于 1.25，success 不低
  于 1.25（允许逐 episode 1 个损失）。
- 失败应看到：scale=1.0 clip 仍高，或 success 明显低于 1.25。
- 最便宜证伪：Goal93 eval-only scale sweep `1.0/1.25/1.5/1.75/2.0`。

### H92-X3：PPO 方向本身不对，放大只会放大被截断的噪声

- 机制：`r_PPO` 与 `r_BC` 或误差下降方向不正相关，C10 的额外幅度没有终端意义。
- 成立应看到：gate-active 区域 `cos(r_PPO, r_BC)` 中位数低或为负，success/failure
  episode 的方向统计不可分。
- 失败应看到：cos 明显为正，success episode 的方向一致性高于 failure。
- 最便宜证伪：Goal93 trace 方向统计 + episode 聚类 bootstrap。

## 5. 结论与 next

`UPDATE_SCHEDULE = not_sufficient` 成立，但不能据此直接判“样本效率不足”并开 SAC。
先做 Goal93 eval-only 审计，量化：

1. PPO requested residual 有多少变成 effective actuator change；
2. 总残差 scale 与 clip/success 的关系；
3. PPO 方向是否与 BC/误差下降方向一致；
4. terminal capture 是否仍是毫米级壳层敏感。

只有明确看到“方向对、effective 增量存在，但闭环不改善”时，才开 SAC 同协议对照。

## 6. 证据

- `doc/goals/goal92_ppo_update_credit_assignment_20261111.md`
- `doc/goals/goal92_result_20261111.md`
- `artifacts/lowlevel/goal92_update_schedule_20261111/eval_dev40/*/summary.json`
- `doc/goals/goal87_residual_credit_assignment_20261110.md`（H87-O4：2× eval-only 曾提升 strict125）
- `doc/goals/goal87_terminal_capture_audit_result_20261110.md`
