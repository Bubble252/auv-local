# Goal162 blocker：配置层速度契约有效，但闭环地速越过 2.5 m/s

**日期**：2026-11-12<br>
**状态**：`INVALID_SPEED_CONTRACT / PIVOT_TO_SPEED_ENVELOPE_AUDIT / HELDOUT_CLOSED`<br>
**上游**：Goal162 `INVALID_SPEED_CONTRACT`；Goal149 已记录 strong-current 下的
ground-speed saturation

## 1. 现象

Goal162 的 3 个 SAC seed 都完成 `1,000,448` transitions，manifest 均满足
`action_interface=controller_gain`、`action_dim=2`、`direct_commands_active=false`、
`reference_speed_mps=1.2`、`speed_contract.ok=true`。但正式 dev40 evaluation 中，
H3 zero 与 gain-residual policy 都出现闭环地速超过 REMUS100 `2.5 m/s` 上限。

关键数字：

| radius | level | zero 峰值 | policy 峰值 | zero 超限比例 | policy 超限比例 |
|---|---|---:|---:|---:|---:|
| default | nominal | 2.563 | 2.562 | 0.96% | 0.96% |
| default | current07 | 2.700 | 2.700 | 3.44% | 3.43% |
| default | current10 | 2.879 | 2.879 | 7.20% | 7.20% |
| default | current15 | 3.308 | 3.308 | 14.68% | 14.70% |
| strict125 | current10 | 2.879 | 2.879 | 7.53% | 7.51% |
| strict125 | current15 | 3.308 | 3.308 | 15.08% | 15.06% |

最典型的 current10 seed 40、step 1093：goal distance `18.7 m`，策略尚未进入近端
gain gate，`command_physical` speed 为 `1.08 m/s`，实际 ground body velocity 为
`[2.87, -0.20, -0.11] m/s`，`nu` 约 `[2.27, -0.20, -0.05]`。这不是 residual
开启后的局部尖峰，而是基座/海流/转向瞬态共同造成的持续包线越界。

## 2. 证据

- current15 有 40 个以上 episode seed 出现 sustained overspeed，排除单步 NaN/飞点；
- zero 与 policy 的最大值和超限比例几乎相同，排除 gain residual 是唯一原因；
- 训练 manifest 的 `worst_case_ground_speed_mps=1.85` 只做
  `reference + current_upper + variation` 的配置层代数上界，不是闭环地速；
- strict125 只改变 success radius，不改变速度，超限比例仍在 `7.5%/15.1%` 量级，
  排除 success radius 或 terminal gate 解释；
- 课程 v2 的单测已确认 `C6=(0.90, 2.50)`，并要求每个 episode 满足
  `speed + current + variation <= 2.5`；因此越界发生在“课程采样正确”之后，
  是控制器/plant 闭环问题，不是采样器漏掉了 2.5 上限。

## 3. 已排除项

| 检查 | 结论 |
|---|---|
| held-out | `100–149` 未打开 |
| split/seed | 训练 `86201–86203`；dev `20–59`；未换 seed 救 Gate |
| 训练预算 | 每 seed `1,000,448` transitions，差异只来自 vector step 取整 |
| direct command | 三个 manifest 均为 `false` |
| residual gate | `>10 m` 的 applied gain delta 接近 0；典型超速点距离 18.7 m |
| residual 符号 | zero 与 policy 同幅超限，不能由 policy 方向解释 |
| 课程采样 | `2.5 m/s` 硬上限与 `speed+current+variation` 联合约束已有单测覆盖 |
| 评估计算 | 使用 `ground_velocity_body_mps` 的范数；峰值与 `nu + current` 一致 |

尚未排除：速度环积分饱和/缺少反向制动、转向瞬态带来的附加阻力变化、螺旋桨一阶
动态、REMUS100 plant mismatch，以及 speed command 进入闭环后的 overshoot。

## 4. 可证伪假设

### H162-S1：速度环在减速段存在积分/执行器饱和，导致指令已降但 RPM 不能及时回收

- 机制：速度误差积分或螺旋桨一阶动态在强流/转向时保持较高 RPM；当 ALOS 已降速，
  实际 `nu` 仍高于指令，形成 sustained overspeed。现有 rich trace 没有记录
  `speed_integral_rpm`，因此只能在固定 seed 的 deterministic replay 中补只读诊断。
- 成立应看到：超速区间内 `actual_actuator[rpm]` 持续高于同 speed-error 的非超速区间；
  降速后的 integral/RPM 回退滞后；关闭积分或加 anti-windup 后同一 replay 的
  `>2.5 m/s` 步数显著下降。
- 失败应看到：RPM 已明显回收而地速仍超限；此时更可能是 current/plant/转向瞬态，
  不是积分 windup。

### H162-S2：配置层包线低估了闭环 ground-speed，应由闭环安全包线而非开环代数上界约束

- 机制：`reference + current_max + variation` 只描述环境输入上界，没有描述闭环
  overshoot、转向耦合和 plant mismatch；因此即使 `ok=true`，真实 ground speed 仍可
  超过 2.5。该假设不要求 residual 有错，zero controller 本身即可触发。
- 成立应看到：固定 zero policy，按 turn rate/current/distance 分箱后超限主要集中在
  高 current + 转向瞬态；同 seed 关闭 current 或降低 speed command 后超限消失；
  超限峰值可由 `command_physical`、RPM、`nu` 和 current 的保守组合解释。
- 失败应看到：低 current、直线、无转向的 replay 也稳定超限；此时应优先检查
  plant backend、速度观测或 evaluation 的速度定义。

## 5. 最便宜证伪实验

不训练、不换算法、不改 reward/terminal/success/radius、不打开 held-out：

1. 固定 current10/current15 的 3 个 seed 和 zero policy，做 deterministic replay；
2. 补只读 trace：speed error、speed integral、commanded/actual RPM、`nu`、ground
   speed、current、turn rate、acceleration 和 reachability 标志；
3. 对 `>2.5 m/s` 与 `<=2.5 m/s` 的 step 做配对分箱，先判断 H162-S1；
4. 对同一 seed 做 zero-current / low-speed / straight-turn 子重放，判断 H162-S2；
5. 只有审计确认根因后，才预注册最小控制器修正；本 blocker 不授权新 RL 训练。

## 6. 结论与下一步

Goal162 的 success 结果只能作为方向性诊断：current10 default `+2/120`、strict125
`+7/120`、current15 default `+8/120`，但速度契约先失效，不能写 formal，不能扩
seed，也不能据此开 PPO/REDQ/DroQ 对照。

下一步：`pivot` 到 Goal163，只做速度包线审计和最小控制器修正预注册。Goal163
必须继续使用 REMUS100 `2.5 m/s` 地速硬上限；课程 v2 继续保留联合采样约束，
不得把 `2.5` 端点与强流或大速度扰动组合。
