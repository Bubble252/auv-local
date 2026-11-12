# Goal148 决策：K4 通过 nominal/plant，current 压力下执行器先饱和

**日期**：2026-11-11<br>
**Decision**：`continue`<br>
**Next**：Goal149 current stress ladder + actuator authority boundary

## 1. 证据

Goal148 canonical CUDA/float64，dev40，goal radius 1.25，policy/zero 配对：

| level | K4 success | Z success | K4 plane miss | K4 eta_done |
|---|---:|---:|---:|---:|
| nominal | 40/40 | 26/40 | 0 | 1.2349 m |
| plant20 | 39/40 | 26/40 | 1 | 1.2364 m |
| current15 | 15/40 | 8/40 | 25 | 1.7531 m |
| current20 | 7/40 | 6/40 | 33 | 2.9954 m |
| combined20 | 7/40 | 5/40 | 33 | 2.9306 m |

policy 与 zero 在所有 level 上 success 相同，residual 平均绝对值 `1.5e-4–1.9e-4`。
K4 400k guard 在 nominal 是 `40/40`，combined20 是 `7/40`，与 300k 相同。

## 2. 机制

current15/20 下：

```text
rudder saturation: 0.456 / 0.514  (nominal 0.265)
rpm saturation:    0.125 / 0.145  (nominal 0.021)
ground speed p95:  2.78 / 3.20    (nominal 1.96)
ground speed >=2.5 m/s: 13% / 36%
heading error abs mean: 0.283 / 0.396 rad (nominal 0.124)
crab abs mean: 15.9 / 26.5 deg (nominal 7.6)
heading rate clip: 0.549 / 0.520 (nominal 0.504)
```

heading rate clip 变化不大，rudder/rpm/ground-speed 饱和显著上升；current20 的 crab
已超过 18 deg 上限，说明 full-compensation 路径在工作但仍失败。因此不是 rate limiter
或 crab 上限单独造成，而是执行器/速度包络先饱和。继续加 heading gain 或 residual
只会增加饱和。

## 3. 假设判定

- H148-A（K4 在所有 stress 下 `>=38/40`）：`FALSIFIED`。
- H148-B（K4 不用饱和换 success）：`FALSIFIED` under current。
- H148-C（K4 基座上 policy/zero 同向）：`SUPPORTED`。

## 4. 一次 INVALID 记录

第一次 CPU stress 用 environment config 改 `current`，但 C1 curriculum sampler 会覆盖
该字段，导致 current15/current20 与 nominal 完全相同。该批结果保存为
`goal148_k4_terminal_pitch_stress_20261111_cpu_invalid`，不参与裁决。

新增 eval-only 参数后重跑：

```text
--curriculum-current-speed-range-mps MIN MAX
--curriculum-shear-scale SCALE
```

只改 C1 的 current/shear，不改 depth/path/noise/length/speed/reward/termination/success/
goal radius。canonical nominal K4 `40/40` 与 Goal147 完全一致。

## 5. 下一步

Goal149 先做 current stress ladder，不训练、不扫 controller gain：

```text
levels: nominal, current05, current10, current15
K4 vs Z, policy/zero, dev40
```

目标是找 K4 从 `40/40` 到 `15/40` 的转折点，并同时记录 rudder/rpm/ground-speed
饱和。只有在传统 K4 仍有约 70–90% success、且存在稳定横向/纵向误差的区间，才预注册
residual 训练。current20/combined20 属于执行器饱和区，不作为训练场景。

仍不接 Goal140 reward，不换 PPO/SAC，不扫 seed，不打开 held-out。
