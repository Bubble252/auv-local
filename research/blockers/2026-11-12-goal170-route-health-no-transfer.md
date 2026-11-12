# Blocker：50 m guidance 下旧 residual checkpoint 没有 route-health 净收益

**日期**：2026-11-12<br>
**关联**：Goal170 `GATE170_B_FAIL`<br>
**状态**：`OPEN_FOR_GOAL171_ROUTE_HEALTH_TRAINING`

## 1. 现象

Goal170 在 `4 arms × 4 levels × 2 radii × 3 checkpoint seeds` 的全部 32 个 paired
cell 上都没有通过 route-health Gate。主指标是整条路线、五段和 10% keypoint，不是
terminal success：

- A actuator：route delta `+0.0040…+0.0084 m`；
- B command：route delta `+0.0016…+0.0055 m`；
- T2-SAC：route delta `+0.2310…+0.3806 m`，五段全部退化；
- T2-PPO：route delta `+0.0000…+0.0002 m`，与自己的 zero 几乎不可区分。

所有 arm 的 `route_absolute` 和 `route_relative` 检查均失败；A/B 只有 1/5 段改善，
T2-SAC 0/5 段改善，T2-PPO 的改善幅度低于 `0.0003 m`。

## 2. 关键证据

- A/B 的变化集中在最后 20% 或终点前，没有中段路线收益；
- T2-SAC 同时提高 mean speed `+0.11…+0.13 m/s`，但 cross/vertical RMSE 都变差，
  说明它不是在修正路线，而是在更大的动作/速度下放大偏差；
- T2-PPO 的 normalized residual 平均只有 `4.53e-4`；换算到物理执行器后，转向通道
  约 rudder `1.66e-5 rad`、stern `5.43e-5 rad`，主要是 rpm 通道；
- PPO 与 `zero_ppo50` 的 actual actuator 差也只有 rudder `5.33e-5 rad`、stern
  `7.91e-5 rad`、rpm `0.151`，因此 route delta 只有 `1e-4 m` 量级；
- PPO 速度 guard 失败：policy 和 zero 的 episode 峰值达到 `3.267 m/s`，违反
  `2.5 m/s` 硬上限；这是 base contract 问题，不是 residual 导致；
- T2-SAC 有 4 个 timeout episode；所有非 PPO arms 没有 safety，且地速峰值均在
  `2.5 m/s` 以下。

## 3. 已排除

- 不是缺 checkpoint seed：PPO status `COMPATIBLE`，policy `24/24`、zero `8/8`；
- 不是 `episodes.json`/trace 格式错误：PPO sibling trace 可完整分析；
- 不是 terminal-only：success/plane-miss 只作二级诊断，路线主指标没有移动；
- 不是 residual 完全为零：PPO 有数值残差，但转向物理权限过小；T2-SAC 残差很大，
  但方向/幅度伤害路线；
- 不是 goal radius、terminal 或 reward 在 Gate 内被改动：本 Goal 全部 eval-only；
- 不是同一个 base 的简单比较：PPO 已与独立 `zero_ppo50` 配对，未与 Goal169
  `zero50` 混算。

## 4. 可证伪假设

### H170-D：训练分布/动作接口不匹配导致旧 checkpoint 无路线收益

- 机制：旧 checkpoint 训练时的 guidance、reward 和终端分布与 50 m route-health
  目标不同，策略学到的是 terminal/速度修正，而不是中段 cross/vertical 修正。
- 成立：重新训练时，加入 route/keypoint reward 后，在至少 3/5 段和 20%–90%
  keypoint 上出现同向改善。
- 失败：即使 reward 直接覆盖路线误差，仍无分段改善；此时转向动作/观测/plant。
- 最便宜证伪：1 seed、200k transitions 的 route-reward pilot，先看 reward finite、
  residual 物理权限和五段 RMSE 方向，不直接开 5M。

### H170-E：PPO 的转向通道发生 residual collapse，而不是算法无效

- 机制：PPO actor 输出集中在 rpm，rudder/stern 的归一化输出接近零；需要检查
  action scaling、初始化和 credit，而不是更换算法。
- 成立：在相同 checkpoint/观测下，增大 rudder/stern 通道的 residual scale 或加
  通道级初始化/约束，能让 physical steering authority 达到可测水平，同时 PPO
  训练仍稳定。
- 失败：增大转向通道后仍无权限，或出现饱和/路线退化；此时才考虑 SAC/DroQ 等。
- 最便宜证伪：纯 eval 的 channel scaling 诊断，不改训练、不改 reward；记录
  rudder/stern physical delta、饱和率和五段路线 delta。

### H170-F：中段 route reward 比 terminal success 更能产生可迁移收益

- 机制：当前成功信号稀疏且集中在终点，pooled 3D RMSE 又混合横向/垂向；按路线
  弧长/keypoint 归一化的 dense route reward 能把 credit 推到中段。
- 成立：route-reward arm 在五段中的至少 3 段改善，且 completion/progress
  efficiency 不退，success 至少不系统性恶化。
- 失败：中段 reward 只提高进度/速度、不降低 cross/vertical，或造成 endpoint
  capture 退化；此时回查 reward 权重和动作接口。
- 最便宜证伪：与同预算 control 配对，只跑 current10/default + strict125 的小
  矩阵，先看 route Gate，不先扩所有环境和 5M。

## 5. 下一步

进入 `Goal171` 的预注册 route-health-centered training，而不是继续 replay 旧
checkpoint：

1. 保留 `z_t`/BC、50 m guidance 和 `2.5 m/s` 硬约束；
2. 奖励以五段/keypoint route error、progress、completion 为主，success 只作 guard；
3. PPO 与 SAC 都预注册，但不混算；先做 1 seed/200k 的 falsification；
4. 同时修复 PPO base 的 speed guard，否则任何 PPO 结果先记 `INVALID_SPEED`；
5. 任何新 arm 都必须报告整条路线、五段、10% keypoint、分段最坏值和速度包线。

阅读与机制参考：

- `research/paper_reading/2026-11-12-goal170-route-health-metrics.md`
- `research/paper_reading/2026-11-12-goal167-terminal-capture-authority.md`
- `doc/goals/goal140_vertical_tracking_reward_plan_20261111.md`
