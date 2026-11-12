# Blocker：Goal150 command residual 在 current07 未增益，SAC 大残差有害

**日期**：2026-11-11<br>
**Goal**：Goal150<br>
**状态**：`SCREEN_FAIL / PPO_NO_GAIN / SAC_RESIDUAL_HARMFUL / VERTICAL_REWARD_NOT_TRIGGERED / HELDOUT_CLOSED`<br>
**结果**：`doc/goals/goal150_result_20261111.md`

## 现象

current07、dev seeds `20–59`、K4 controller 下：

| arm | policy | zero | Δsuccess | plane miss P/Z | eta_done P/Z |
|---|---:|---:|---:|---:|---:|
| PPO ent1e4 | 35/40 | 37/40 | -2 | 5/3 | 1.2518/1.2525 |
| PPO ent1e3 | 35/40 | 37/40 | -2 | 5/3 | 1.2536/1.2525 |
| PPO ent1e2 | 35/40 | 37/40 | -2 | 5/3 | 1.2536/1.2525 |
| SAC | 9/40 | 37/40 | -28 | 31/3 | 1.9789/1.2525 |

PPO 三档 entropy 都固定丢失 seed `36/49`，没有新增成功 seed。SAC 在 current05
policy/zero 为 `10/40 vs 39/40`、current10 为 `11/40 vs 29/40`，转移同样崩溃；
timeout/safety 全 0。

## 已排除

| 候选原因 | 判读 | 证据 |
|---|---|---|
| NoisyNet 主导 | 排除 | checkpoint 无 `rho/sigma/noise` 参数；deterministic eval |
| PPO 探索不足 | 排除 | entropy 提高 100 倍后 success、失败 seed、residual 几乎不变 |
| SAC 残差塌缩 | 排除 | learner action RMS `0.79–0.90`，action abs max 约 `0.9997` |
| 纵向奖励缺失 | 排除 | PPO vertical RMSE 比 zero 低 `0.0028 m`，success 仍下降 |
| 失败由 timeout/safety 主导 | 排除 | 两者均为 0；失败全部是 plane miss |
| 评估框架/动作 scale 重复 | 排除 | policy/zero 共用 evaluator；zero 仍 `37/40`；trace residual 可解释 |
| 单 seed 随机性 | 不足以解释 | SAC 主场景与两个 transfer 同时严重退化 |

## 可证伪假设

### H151-A：SAC 伤害随 residual scale 单调增加，低 scale 应部分恢复

- 成立：在现有 SAC checkpoint 上 eval-only scale `0/0.25/0.5/1.0`，success 随
  scale 单调下降；`0.25` 接近 zero，`1.0` 约 `9/40`。
- 失败：低 scale 仍严重掉 success，或某中间 scale 反而好于 zero；此时不能把伤害
  简单归因于“权限过大”，要继续查动作方向与 reward credit。
- 最便宜证伪：不训练，只改 eval residual multiplier；current07 dev40，复用同一
  checkpoint 和 seeds。

### H151-B：SAC 的 31 个 plane miss 主要由 heading/pitch 大残差造成

- 成立：channel-restricted eval 中仅保留 speed 残差时，success 接近 zero；仅保留
  heading/pitch 时复现大部分 plane miss。
- 失败：speed-only 也大量 plane miss，或 heading/pitch mask 后仍无恢复；此时问题
  在 base command/reward/terminal capture，而不是通道权限。
- 最便宜证伪：不做新训练，按 heading/pitch/speed 通道分别 mask 现有 SAC policy，
  对 current07 dev40 做 4–6 个 eval run。

### H151-C：PPO 的 `+0.015 m/s` 速度偏置是 `-2/40` 的主要原因

- 成立：对 PPO checkpoint 做 speed-channel zero 或 scale sweep 后，seed `36/49`
  恢复成功，success 回到 `37/40`，且 heading/pitch 不变时无新增失败。
- 失败：抑制 speed 残差后 seed `36/49` 仍失败，或 heading 残差单独即可造成损失；
  此时是 terminal approach 几何对微小多通道扰动敏感，不是单通道偏置。
- 最便宜证伪：复用 PPO `ent1e4` checkpoint，在 current07 做 speed-only/全部 zero
  的 eval-only 对照。

### H151-D：reward 优化方向与 terminal success 不一致（credit 假设）

- 成立：失败 episode 的 `reward_path_progress`、`reward_path_heading` 或总 reward
  不差于成功 episode，但 `reward_goal_plane_miss`/`reward_terminal` 没有及时惩罚
  造成 plane miss 的 residual；残差在 terminal 前 32–64 步仍增大。
- 失败：失败 episode 的所有 reward 分量和 terminal penalty 都明显更差，说明 credit
  已被正确表达，问题主要在动作参数化/控制权限。
- 最便宜证伪：复用 `trace.npz` 的逐 step reward breakdown，对成功/失败 episode 做
  terminal 前窗口对齐；不训练、不改 reward。

## 当前最可能机制

SAC 把接近满幅的 learner action 映射成大 heading/pitch command residual；terminal
approach 对 heading/pitch 非常敏感，残差改变了 goal-plane entry 几何，导致 31 个
plane miss。PPO 的残差很小，但仍带固定速度偏置并损失 2 个 seed，说明 current07
本身靠近成功边界，微小系统偏置也能改变结果。

目前没有足够证据说 reward credit 是根因；必须先做 dose/channel/trace 审计，把
“权限伤害”和“reward 误导”分开。

## 最便宜 falsification 顺序

1. `SAC scale 0/0.25/0.5/1.0`，current07 dev40，eval-only；
2. `SAC channel mask`：heading/pitch/speed 单通道与组合；
3. `PPO ent1e4 speed-only zero`，current07 dev40，确认 seed `36/49`；
4. 对现有 trace 做成功/失败窗口 reward breakdown 与 terminal 前 residual 曲线；
5. 只有前三项中至少一项恢复 parity，且 trace 支持权限机制，才预注册新的低权限
   残差训练 arm。

## 不做什么

- 不扩 seed、不扩预算、不换 REDQ/DroQ；
- 不启动 Goal140 纵向 reward；
- 不在 Goal150 Gate 内改 reward、success、terminal 或 goal radius；
- 不打开 held-out；
- 不把自动分析器的 `SCREEN_INCONCLUSIVE` 写成通过。

## 下一步

写 Goal151 预注册：`residual authority / channel restriction / reward-credit /
terminal-capture read-only audit`。先只读，不训练；结果出来后再决定是否进入
低权限残差训练或 reward 修缮。
