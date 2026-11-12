# Goal150 决策：pivot 到残差权限 / reward credit / terminal capture 只读审计

**日期**：2026-11-11<br>
**Decision**：`pivot`<br>
**Next**：Goal151 eval-only residual authority / channel restriction / reward credit /
terminal capture audit

## 1. 证据

Goal150 是预注册的 1-seed screen。current07 dev40、K4 controller 下：

| arm | policy | zero | Δsuccess | plane miss P/Z |
|---|---:|---:|---:|---:|
| PPO ent1e4 | 35 | 37 | -2 | 5/3 |
| PPO ent1e3 | 35 | 37 | -2 | 5/3 |
| PPO ent1e2 | 35 | 37 | -2 | 5/3 |
| SAC | 9 | 37 | -28 | 31/3 |

PPO 三档 entropy 没有分化，固定丢失 seed `36/49`；SAC transfer 到 current05
policy/zero `10/40 vs 39/40`、current10 `11/40 vs 29/40`。timeout/safety 全 0。

SAC 的 learner action RMS 为 `0.79–0.90`，applied residual signed mean 为
heading `-0.092 rad`、pitch `+0.044 rad`、speed `+0.013 m/s`。这不是残差塌缩，
而是大权限残差改写 terminal approach。

## 2. 归因

| 候选 | 判定 |
|---|---|
| NoisyNet 主导 | 排除 |
| PPO 探索不足 | 排除 |
| SAC 样本效率不足 | 不足以解释；已学到非零大残差 |
| 纵向 reward 缺失 | 排除；PPO vertical RMSE 略好仍 loss success |
| 动作参数化/残差权限 | 高度可疑，进入只读审计 |
| reward credit/terminal 惩罚不一致 | 待审计，不能先改 reward |

## 3. 为什么不换算法

本轮没有“PPO/SAC 都接近 zero 但学不动”的证据。SAC 主动输出了接近满幅的有害动作，
PPO 则在微小速度偏置下稳定损失 2 个 seed。这更像 action authority 与 terminal
capture 的接口问题，而不是算法族样本效率问题。因此不启动 REDQ/DroQ，也不把 SAC
替换成 PPO formal。

## 4. 为什么不启动 Goal140

Goal140 的触发条件是失败主要来自纵向误差，且残差方向正确但纵向误差不降。Goal150
不满足：PPO 的 vertical RMSE 反而下降 `0.0028 m`，但 success 更差；SAC 是 31 个
plane miss，横向 RMSE 也恶化约 `0.16 m`。因此保持 Goal140 `STANDBY`，不跑
`F_z/F_zdot/F_predz` reward pilot。

用户提出的奖励修缮思想保留在 Goal140 §0/§4.2；未来若出现新的纵向主导证据，必须
另开预注册 Goal，不能救当前 Gate。

## 5. 下一步

Goal151 只读审计，不训练：

1. SAC residual scale `0/0.25/0.5/1.0` 剂量曲线；
2. SAC heading/pitch/speed channel mask；
3. PPO `ent1e4` speed-only zero，检查 seed `36/49`；
4. 现有 trace 的 terminal 前 32/64 步 reward breakdown、residual、pitch/heading
   error 和 plane-miss 几何；
5. 根据结果二选一：
   - 若权限限制可恢复 parity，预注册低权限/分通道残差训练 arm；
   - 若 reward credit 明确错位，预注册 reward/terminal correction pilot。

仍固定 PPO 与 SAC 两种算法，不扫 seed，不打开 held-out，不因失败放宽 Gate。

## 6. 约束

- 不把 `SCREEN_INCONCLUSIVE` 自动标签写成通过；
- 不在 Goal150 已记录结果上改阈值或 checkpoint；
- 不提交 `artifacts/`、checkpoint、日志或缓存；
- 不自动 commit/push。
