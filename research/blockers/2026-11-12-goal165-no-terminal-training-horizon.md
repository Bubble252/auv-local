# Goal165 blocker：1M/1024 envs 没有采到 terminal

**日期**：2026-11-12<br>
**状态**：`GATE165_C_FAIL / NO_TERMINAL_TRAINING_HORIZON / ACTION_AUTHORITY_NULL`

## 现象

Goal165 的 3 个 SAC seed 都完成 `1,000,448` transitions，但训练 manifest 的
`completed_episodes=0`、`success_episodes=0`。同一 corrected 评估协议下，
zero 闭环 episode 最短 `1291` 步、median `1749` 步、p90 `1996` 步，
而训练每环境只有 `1,000,448 / 1024 = 976` 步。训练从未到达 goal 或
goal-plane-miss terminal。

同时，policy 与 zero 在 default/strict125 × nominal/current07/current10/current15
的 8 个格子中逐 seed 完全同分：Δ 全部为 `0/120`。近 goal 的 residual norm
约 `1.15`，但 requested action 平均只改变
`[0.011, 0.022, 0.168]`（rudder rad / stern rad / rpm），闭环位置平均只改变
`4–7 mm`。

## 证据

- `doc/goals/goal165_result_20261112.md`
- `artifacts/lowlevel/goal165_controller_feedback_20261112/analysis/goal165_summary.{json,md}`
- `artifacts/lowlevel/goal165_controller_feedback_20261112/analysis/speed_audit/speed_audit.{json,md}`
- `artifacts/lowlevel/goal165_controller_feedback_20261112/analysis/action_authority/action_authority_audit.{json,md}`
- `artifacts/lowlevel/goal165_controller_feedback_20261112/train/G1_gain_near10_2/seed*/training_manifest.json`
- `artifacts/lowlevel/goal165_controller_feedback_20261112/eval/G1_gain_near10_2/*/*/zero_seed86201/episodes.json`

## 已排除

- 不是速度契约：`eta` 峰值 `1.965–2.053 m/s`，所有格 `eta<=2.5`，
  non-terminal/terminal 超速率都是 0。
- 不是训练崩溃：没有 NaN/Inf，3 个 seed 都完成预算。
- 不是 SAC 优化器发散：policy 没有学到坏动作，而是 residual 的物理效果太小。
- 不是单一 seed 噪声：3 个 seed 的成功数与 zero 完全相同。

## 可证伪假设

### H165-A：训练 horizon 是主因

如果把 1M 预算改成每环境至少 `2500` 步（例如 `num_envs=256`，保持 1M），
训练 manifest 应出现 `completed_episodes>=256`，且 terminal/success 非零；
在同一评估协议下 residual 应至少能改变轨迹，而不只是毫米级。

- 失败条件：completed episodes 仍为 0，或 residual 仍无可测闭环效果。

### H165-B：gain residual 参数化没有足够 authority

即使补齐 terminal 样本，controller-gain residual 仍可能只改变毫米级轨迹，
因为 near-goal 的 rudder/stern/rpm 改变只有 `0.01–0.17` 个单位。

- 成立条件：horizon 修复后，residual norm 仍然大，但 requested action 和
  eta 改变仍接近 0。
- 失败条件：horizon 修复后，residual 出现 signed-error 相关并改变成功数。

### H165-C：算法/样本效率不足

当前证据不支持。policy 与 zero 逐 seed 同分，且训练没有 terminal 样本；
在 horizon 和 action authority 未修复前，SAC/PPO/REDQ/DroQ 对照不可解释。

## 最便宜的证伪实验

预注册 Goal166：只改训练 horizon，不改算法、reward、terminal、success、radius。
3 seeds × 1M，`num_envs=256`，使每环境约 `3906` 步，覆盖 2500 步上限；
其余配置与 Goal165 相同。先检查 manifest 的 `completed_episodes` 和
`success_episodes`，再用同一 paired dev40 zero/policy 评估。

## 当前裁决

不换算法，不扩 seed，不放宽 2.5，不打开 held-out。先修训练 horizon；
如果修完后 gain residual 仍无 authority，再新开动作参数化 goal。
