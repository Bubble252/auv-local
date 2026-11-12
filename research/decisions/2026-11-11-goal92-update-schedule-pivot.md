# Decision：从 PPO 更新调度 pivot 到动作权限/残差方向审计

日期：2026-11-11<br>
关联：Goal92<br>
决定：`pivot`

## 判据

Goal92 的 6 个更新调度 arm 在 seed86201、1M、dev40 上没有产生任何闭环移动：

- default 全部 `31/40`；
- strict125 除 `W0` 外全部 `20/40`；
- `C10/C20` 把 PPO 确定性残差放大约 `35%/29%`，但逐 episode gain/loss
  仍为 `0/0`，path 变化小于 `0.002 m`；
- 所有 arm 的 magnitude clip rate 约 `0.69`、rate clip rate 约 `0.49`，
  与 baseline 几乎完全相同。

因此 `num_learning_epochs`、`clip_param`、advantage normalization 和 critic
warmup 不是当前主瓶颈。继续在同一方向上换超参数没有信息增益。

## 为什么不直接开 SAC

Goal92 只证明“更新调度不够”，还没有证明“方向对但样本效率不足”。新出现的
更强解释是：

```text
scale=1.25 下 r_BC 已接近或超过执行器包线
PPO 的 6–9% 增量大部分被 magnitude/rate limit 截掉
effective actuator change ≈ 0
```

如果这个解释成立，SAC 也会学到同样的被截断增量，换算法不会解决问题。必须先做
eval-only 的动作权限与方向审计。

## 新路线

Goal93：`residual_direction_and_authority_audit`，只读/回放既有 checkpoint。

1. 对 `PG10_3` 与 `C10` 做 eval-only total residual scale sweep
   `1.0/1.25/1.5/1.75/2.0/2.5`，dev40，default+strict125；
2. 记录 policy 与 frozen-base trace，直接计算
   `effective/applied`、逐通道 clip loss、gate-active `cos(r_PPO, r_BC)`；
3. 比较 success/failure episode 的方向统计与 terminal capture；
4. 根据预注册判据决定：
   - `ACTION_PARAMETERIZATION_CLIPPING` → 先修 scale/gain/clip-aware residual，不开 SAC；
   - `DIRECTION_POSITIVE_BUT_SAMPLE_INEFFICIENT` → 才开 SAC 同协议对照；
   - `DIRECTION_WRONG_OR_TERMINAL_SHELL` → 修 reward/terminal capture，不开 SAC。

## 不做什么

- 不打开 held-out `100–149`；
- 不改 reward、terminal、success、goal radius；
- 不用 Goal86 `T2-SAC` 冒充同 BC 基座对照；
- 不在 Goal93 中途换算法或改预注册阈值。

## 证据

- `doc/goals/goal92_ppo_update_credit_assignment_20261111.md`
- `doc/goals/goal92_result_20261111.md`
- `research/blockers/2026-11-11-goal92-update-schedule-no-movement.md`
- `artifacts/lowlevel/goal92_update_schedule_20261111/eval_dev40/*/summary.json`
