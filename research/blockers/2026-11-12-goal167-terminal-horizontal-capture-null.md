# Blocker：Goal167 直接动作残差有 authority，但 terminal horizontal capture 不移动

**日期**：2026-11-12<br>
**关联**：Goal167 `GATE167_B_FAIL`<br>
**状态**：`OPEN_FOR_GOAL168_READ_ONLY_AUDIT`

## 1. 现象

Goal167 比较 actuator residual（A）与 ALOS 3D command residual（B）。两臂
在 primary `current10` 上逐 seed 同分：

- default `102/120 -> 102/120`；
- strict125 `99/120 -> 99/120`；
- success 集合 only-policy/only-zero 均为 `0/0`。

训练覆盖 PASS（A completed `596/587/589`，B `583/585/576`），速度包线 PASS
（policy 峰值 A `2.336`、B `2.125 m/s`，无 `eta>2.5`）。因此不是 horizon、
terminal 覆盖或速度契约问题。

## 2. 关键证据

- A near-goal requested action 改变达到 `[0.0593, 0.0805, 287.80]`，
  policy saturation `0.853`，eta 平均改变 `0.584 m`；
- B command residual 全近端有权限（outer gate 关闭），requested action 改变
  `[0.0398, 0.0562, 31.52]`，policy saturation `0.980`；
- 两臂 final-distance 绝对变化平均只有 A `0.0136 m`、B `0.0111 m`；
- primary 18 个失败全部是 `GOAL_PLANE_MISS`，横向 `2.2–11.15 m`，
  纵向 `<0.31 m`；
- A 中 11/18 失败在 gate 完全打开前越平面；另外 7 个在 gate 接近 1 时仍以
  `2.2–2.7 m` 横向误差越平面；
- B 所有失败都拥有完整 command 权限，仍无 success 翻转。

## 3. 已排除

- 不是 NoiseNet 主导：失败在确定性 dev eval 上稳定复现，且横向误差是主项；
- 不是训练 horizon/terminal 覆盖：两臂 completed episodes 均 >500；
- 不是速度契约：所有峰值 `<=2.5 m/s`；
- 不是单纯 gate 太晚：B 全程有权限仍失败，A 也有 7 个 full-gate 失败；
- 不是简单“动作幅度为 0”：A 的 actuator 改变已达到数百 rpm/0.08 rad 量级。

## 4. 可证伪假设

### H168-A：terminal homing 的横向 authority 足够，失败来自 guidance 没有真正收敛

- 成立：在 `goal_distance<=20 m` 内，所需 turn-rate/lateral acceleration
  低于实测 rudder/turn-rate 包线；且 trace 显示 robust-terminal direct-homing
  未在失败 episode 中持续生效，或 projection 在终点附近跳段。
- 失败：所需 turn-rate 已超过物理包线，或 direct-homing 一直生效但仍无法收敛。

### H168-B：终端横向 authority 本身不足，当前 success 契约在 P2 终点不可达

- 成立：失败 episode 在最后 20 m 所需的横向修正超过最大 turn-rate/crab/
  rudder 能力，且 zero/A/B 都无法在越平面前压到 2 m。
- 失败：存在至少一个 deterministic terminal homing 变体，在不改
  success/radius/reward 的情况下把失败 episode 压到 2 m 内。

### H168-C：reward/terminal credit 仍是主因

- 成立：动作 authority 与物理可达性都足够，但 residual policy 学到的
  终端动作方向与 signed error 不相关，且增大 terminal advantage 能改变行为。
- 失败：B 在 full authority 下仍只改变 final distance 约 1 cm，说明不是
  credit 不足，而是 authority/guidance 层。

## 5. 下一步

Goal168 只读审计，不训练、不换算法：

1. 抽取 18 个 `GOAL_PLANE_MISS` episode；
2. 计算最后 20 m 的 required heading/turn-rate/lateral acceleration；
3. 对比实测 rudder、turn rate、crab、current、roll；
4. 检查 ALOS projection / endpoint curvature / robust-terminal 分支；
5. 只有在 H168-A 成立时才预注册 deterministic terminal homing 对照。

