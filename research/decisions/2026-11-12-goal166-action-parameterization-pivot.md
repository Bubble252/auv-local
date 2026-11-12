# Goal166 decision：从 gain residual 转向直接动作残差对照

**日期**：2026-11-12<br>
**decision**：`pivot`<br>
**next goal**：Goal167 `actuator vs command terminal residual`

## 裁决

Goal166 证明了 terminal 覆盖修复有效，但也证明 controller-gain residual
在 near-goal 没有足够物理 authority：

- terminal coverage PASS：591/582/595 completed episodes；
- success 全部 `Δ=0/120`；
- residual norm `1.28`、policy saturation `0.85–0.98`；
- requested action 平均差 `[0.012, 0.029, 0.173]`；
- eta 平均差 `4.5e-3 m`。

因此不再继续调 gain residual 的 scale 或 gate；这只会得到更大的归一化动作，
不会得到更大的物理动作。

## Goal167 预注册方向

两个 arm，3 seeds × 1M，`num_envs=256`，terminal-covering horizon：

### A_actuator_terminal

- `action_interface=actuator`；
- `actuator_residual_scale=0.25`；
- terminal gate `10→2 m`；
- 直接改变 rudder/stern/rpm。

### B_command_terminal

- `action_interface=command`；
- `scale_heading_rad=0.0875`、`scale_pitch_rad=0.0425`、
  `scale_speed_mps=0.15`；
- terminal gate `10→2 m`；
- 直接改变 ALOS 3D 指令。

## Gate

- 所有 policy/zero `eta<=2.5 m/s`；
- current10 default `>=+4/120`，strict125 `>=+2/120`；
- guards 不显著退化；
- requested action/eta 改变达到可解释量级；
- signed-error 至少一个通道跨 seed 稳定；
- 不换算法、不扩 seed、不改 reward/terminal/success/radius、不打开 held-out。

## 失败后的分叉

- 若 actuator 有 authority 但 success 不升：查 reward/terminal credit；
- 若 command 有 authority 但带来 harm：查 command authority cap 和 terminal gate；
- 若两者都无 authority：说明 residual 被 base controller/actuator 饱和吞掉，
  需要更直接的动作空间或控制频率/authority 审计，而不是换 SAC。
