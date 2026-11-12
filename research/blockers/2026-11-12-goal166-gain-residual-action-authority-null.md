# Goal166 blocker：terminal 覆盖修复后 gain residual 仍无 action authority

**日期**：2026-11-12<br>
**状态**：`GATE166_A_PASS / GATE166_B_FAIL_ACTION_AUTHORITY_NULL`

## 现象

Goal166 把 `num_envs` 从 1024 改为 256，3 个 1M seed 都真正采到了 terminal：
completed episodes `591/582/595`，success episodes `525/516/523`。

但 gain residual 仍然没有闭环效果：default/strict125 ×
nominal/current07/current10/current15 的 8 个格子中，3 个 policy seed
与同一个 zero block 逐 seed 完全同分，Δ 全为 `0/120`。速度包线 PASS，
`eta` 峰值 `1.966–2.053 m/s`，超速为 0。

## 证据

- `doc/goals/goal166_result_20261112.md`
- `artifacts/lowlevel/goal166_terminal_horizon_20261112/analysis/training_coverage.{json,md}`
- `artifacts/lowlevel/goal166_terminal_horizon_20261112/analysis/goal165_summary.{json,md}`
- `artifacts/lowlevel/goal166_terminal_horizon_20261112/analysis/action_authority/action_authority_audit.{json,md}`
- `artifacts/lowlevel/goal166_terminal_horizon_20261112/analysis/speed_audit/speed_audit.{json,md}`

## 已排除

- 不是 terminal 覆盖：每环境 3907 步，约 590 个 completed episodes/seed；
- 不是训练崩溃：无 NaN/Inf，3 seeds 完成预算；
- 不是 residual 太小：near-goal residual norm `1.28`，policy action
  saturation `0.85–0.98`；
- 不是速度契约：所有格 `eta<=2.5`，超速为 0。

## 可证伪假设

### H166-C1：controller-gain 参数化在 near-goal 没有物理 authority

- 成立：requested action 平均差仍只有 `[0.012, 0.029, 0.173]`，
  eta 平均差 `4.5e-3 m`，尽管 residual 已经饱和。
- 失败：直接动作残差也出现同样的毫米级改变。

### H166-C2：直接 actuator residual 能提供足够 authority

- 成立：actuator residual 的 rudder/stern/rpm 改变应达到可解释的物理量级，
  并且成功集发生变化。
- 失败：actuator residual 仍无法改变 success，或带来 safety/timeout 退化。

### H166-C3：ALOS 3D command residual 在 corrected contract 下能提供终端修正

- 成立：command residual 出现 signed-error 相关，success 改变且不破坏 guards。
- 失败：与 Goal161 一样出现 command-residual harm，或仍无可测收益。

## 最便宜证伪实验

Goal167：同一 terminal-covering horizon（3 seeds × 1M，`num_envs=256`）、
同一 SAC/z_t/H3/C3/ALOS、同 reward/terminal/success/radius，比较
`actuator` 与 `command` 两种直接动作残差；zero 复用 Goal165 corrected
baseline。先看 requested action/eta 改变是否达到厘米/度级，再看 success。

## 当前裁决

不换算法、不扩 seed、不放宽 2.5、不打开 held-out。进入动作参数化对照；
如果两种直接动作残差仍无 authority，再查 reward/terminal credit。
