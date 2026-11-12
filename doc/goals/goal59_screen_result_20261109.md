# Goal59 Screen 结果：G12 工程通过，NoisyNet 不是主因

日期：2026-11-09<br>
状态：`SCREEN_ENGINEERING_PASS / FORMAL_G12_FAILED`<br>
预注册：`doc/goals/goal59_bounded_terminal_residual_20261109.md`<br>
隔离目录：`/data/yhx/underwater-robotics/src/plan_p1/goal59_bounded_terminal_residual_20261109/`

## 1. 结论

五个 1M-transition screen 全部完成，每个 128 envs、5 个并行进程、约
`29–35 min`。工程有效性检查通过：无 NaN/Inf、无 OOM、无 scale clip、checkpoint
和 manifest 均生成。G12 的 gate-active residual 非零，允许进入 formal。

Screen 只作机制与工程诊断，不按效果选 best。`N12`（关闭 NoisyNet）和 `T12`
（tanh + scale 1.0）不进入 formal。

## 2. Screen 汇总

| arm | 参数化 | gate | path_dim | transitions/s | grad/s | residual L2 ratio | gate-active residual |
|---|---|---|---:|---:|---:|---:|---:|
| G8 | softsign 0.5 | 20→5 m | 8 | 477.5 | 58.5 | 4.24% | 13.0% |
| G12 | softsign 0.5 | 20→5 m | 12 | 480.8 | 58.9 | 4.44% | 14.4% |
| U12 | softsign 0.5 | off | 12 | 469.2 | 57.5 | 13.90% | 13.9% |
| N12 | softsign 0.5, no noise | 20→5 m | 12 | 568.2 | 69.6 | 5.73% | 17.8% |
| T12 | tanh 1.0 | 20→5 m | 12 | 482.2 | 59.1 | 10.00% | 32.6% |

所有 arm 的 `residual_scale_clip_rate` 均为 `0`；G12 的 rate clip 约 `40.2%`，
U12 约 `97.8%`，说明无门控时残差长期顶着速率限制。

## 3. 确定性/NoisyNet 审计

用训练 seeds `240–247`、直到首次终止（约 `806–897` control steps）、每个状态
`8` 个噪声样本：

| arm | deterministic saturation any-dim | noise/det L2 | noise>det fraction | cosine | applied L2 ratio | gate-active residual |
|---|---:|---:|---:|---:|---:|---:|
| G12 | 4.7% | 0.581 | 0.00014 | 0.842 | 3.47% | 14.0% |
| G8 | 99.9% | 0.020 | 0.0 | 0.999 | 4.76% | 17.2% |
| U12 | 65.1% | 0.085 | 0.0 | 0.991 | 19.98% | 19.98% |
| N12 | 99.9% | 0.0 | 0.0 | 1.000 | 4.35% | 16.5% |
| T12 | 100% | 0.426 | 0.0 | 0.847 | 9.64% | 35.0% |

解读：

- **NoisyNet 不是主因**：G12 的 deterministic action L2 约 `0.609`，噪声只增加
  探索；`noise>det` 的状态比例只有 `0.014%`。关闭噪声的 N12 仍然在 `99.9%`
  状态上饱和，说明饱和来自确定性策略本身，不是噪声头。
- **G8 的 8D path 不够**：即使有 softsign、scale 0.5 和 terminal gate，G8 仍在
  三个维度上 `99.9%` 饱和；12D endpoint context 是避免“全程最大修正”的关键。
- **terminal gate 必要**：U12 无门控时 applied residual L2 ratio 约 `20%`，
  训练中 `97.8%` 的步骤触发 rate clip；G12 把 gate-active residual 控制在
  `14%` 左右。
- **softsign/scale 有效**：T12 的 tanh + scale 1.0 仍 `100%` 饱和，gate-active
  residual 约 `35%`；G12 的 heading/pitch 饱和为 `0%`，只有 speed 约 `4.7%`。

### 3.1 和“以前遇到过”的 NoisyNet 现象怎么区分

训练日志里的 `residual_abs_mean` 是 stochastic rollout 统计，天然包含噪声，
不能直接拿来判断 deterministic 评估策略是否被噪声主导。真正的判据是：

- 评估打开 `evaluate=True` 时，deterministic mean 是否已经贴近边界；
- NoisyNet 的 `softplus(rho)` 是否从初始化 `0.3/0.1` 明显膨胀；
- `noise_std > deterministic_action_norm` 的状态比例是否显著大于零；
- 关闭噪声后，残差是否从“很大”回落到近零。

Goal58 和 Goal59 都没有出现这组组合：Goal58 的 deterministic mean 三维
`100%` 饱和，final sigma 仍在初始化附近；Goal59 的 G12 deterministic
any-dim 饱和只有 `4.7%`，`noise>det` 只有 `0.014%`。反而关闭噪声的 N12
仍然 `99.9%` 饱和，说明“没有噪声时会锁在边界”也不能反推“有噪声时是噪声
把动作推出去”。

因此本项目里可以把 NoisyNet 主导限定为一个可检验条件：deterministic 动作
接近零或远离边界，但噪声样本持续把动作推到边界，并且 sigma 膨胀或
`noise>det` 很高。当前证据不满足这个条件，不能用关闭 NoisyNet 来救 Gate。

## 4. 工程判读

G12 满足预注册 screen 的工程条件：

- `finite=true`，0 scale clip，无 OOM；
- `gate_active_rate ≈ 26%`（训练全程），`residual_l2_ratio_gate_active ≈ 14.4%`；
- 128 envs 吞吐约 `481 transitions/s`、`58.9 grad/s`；
- 1M screen 约 `34.7 min`，5M formal 单 seed 预计约 `2.9 h`，8 seed 并行可接受。

## 5. 决策

进入 formal：`G12`、8 个预注册 seed、每个 5M transitions、128 envs、batch 256、
UTD 0.125、final checkpoint。`N12`/`T12` 不进入 formal；不根据 screen 效果改
reward、gamma、seed、scale 或算法。

证据：

- `artifacts/screen_g8/`、`artifacts/screen_g12/`、`artifacts/screen_u12/`、
  `artifacts/screen_n12/`、`artifacts/screen_t12/`；
- 每个目录的 `manifest.json`、`final.pt`、`noisynet_probe.json`。
