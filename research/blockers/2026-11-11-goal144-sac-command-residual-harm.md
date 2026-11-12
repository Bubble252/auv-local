# Goal144 blocker：SAC command residual 在 C1 主动破坏，actuator 只改善 C0

**日期**：2026-11-11<br>
**Goal**：Goal144<br>
**状态**：`SAC_600K_EARLY_STOPPED / COMMAND_RESIDUAL_HARM / ACTUATOR_C0_ONLY / C1_STILL_BOTTLENECK`

## 现象

gatefix2 同协议 SAC 在 600k 早停（C0 hard-cap、C1 回滚预算用尽，继续跑不会产生
新的课程裁决）。

| arm | C0 100k/200k | C1 300k | C1 400k | 回滚后 C0 500k/600k |
|---|---:|---:|---:|---:|
| SAC command | `0.925` -> `PROMOTE` | `0.15` | `0.225` -> `ROLLBACK` | `0.65 / 0.525` -> hard-cap |
| SAC actuator | `0.925` -> `PROMOTE` | `0.625` | `0.625` -> `ROLLBACK` | `0.875 / 1.000` -> hard-cap |

PPO 同协议为 C1 `0.65`，command 残差约 `1e-4`；SAC command 残差 L2 比约
`0.6%`，但 C1 反而更差。SAC actuator 残差 L2 比约 `1–5%`，C0 可到 `1.000`，
C1 仍只有 `0.625`。

## 判读

- 不是“SAC 探索不足”：SAC 已经实际改变动作，但方向错误。
- 不是“PPO 样本效率不足”：PPO 与 SAC 都没有把 C1 推到 Gate。
- 不是 reward/terminal 的首要问题：timeout/safety 为 0，失败仍是 plane miss。
- 当前首要嫌疑是 **动作参数化/权限**：command residual 直接改 ALOS 的
  heading/pitch/speed，在 C1 低曲率正弦路径上把基座带坏；actuator residual
  只在 C0 有局部收益，无法解决 C1 末段捕获。

## 可证伪假设

**H144-N：command residual 的 heading/pitch/speed 联合尺度或 rate limit 过宽。**
若把 command residual 限制到单通道、降低 C1 段尺度或只在接近终端时放开，C1
应不再从 `0.65` 掉到 `0.15–0.225`。最便宜证伪：同 checkpoint 的 eval-only
channel mask / scale sweep，不训练。

**H144-O：C1 失败是横向/纵向末段几何而非残差幅度。** 若 command residual 被
限制后 C1 仍 `~0.65`，且失败 episode 的 entry 垂直/横向误差稳定偏大，则转
terminal capture/geometry 审计；若纵向主导，再启用 Goal140 reward arm。

**H144-P：actuator residual 的 C0 收益不能迁移到 C1。** 若 actuator 在 C0 到
`1.000` 但 C1 仍 `0.625`，则说明 C1 需要路径级/终端级权限，而不是更小的 actuator
扰动。

## 决策

`continue` 到动作参数化/权限审计，不换算法：

1. 只读/eval-only 审计 command residual 三通道尺度、rate limit、terminal gate
   在 C1 的介入距离；
2. 检查失败 episode 的横向/纵向/终端几何，确认是否纵向主导；
3. 只有确认纵向误差主导，才按 Goal140 §4.2 potential reward 开新 reward arm；
4. 不启动 REDQ/DroQ，不扫 seed，不放宽 Gate，不打开 held-out。
