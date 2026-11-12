# Blocker：SAC P-A 修正评估后仍跨 seed 退化

**日期**：2026-11-11<br>
**Goal**：Goal126 / Goal127<br>
**状态**：`SAC_PA_HARM / SAC_PA_SCALE_INSUFFICIENT / NETWORK_PARITY_OK / NOISENET_ABSENT`

## 现象

修正评估侧重复 scale 后，Goal126 8 个 1M SAC checkpoint：

- strict mean `19.75/40`，default mean `28.25/40`；
- strict 只有 86201 达到 30，其余 14–21；
- 86204 有 2 个 strict timeout、1 个 default timeout；
- 评估动作 `abs_max` 已回到 `0.985–1.000`。

Goal127 剂量曲线：

| multiplier | strict mean | default mean | 结论 |
|---:|---:|---:|---|
| 0.0 | 24.00 | 33.00 | 最好 |
| 0.125 | 22.75 | 32.62 | 接近但仍差 |
| 0.25 | 21.88 | 32.38 | 更差 |
| 0.5 | 21.38 | 31.00 | 更差 |
| 1.0 | 19.75 | 28.25 | 最差 |

只有 86201 随倍率增大获益（24→30），其余 seed 多数持续下降。

## 已排除

| 候选原因 | 判读 | 证据 |
|---|---|---|
| 网络结构接错 | 排除 | base action max diff `2.4e-7`；encoder diff `0.0` |
| NoisyNet 主导 | 排除 | 无 `rho/sigma/noise` 参数；deterministic mean eval |
| path 初始化错误 | 排除 | `bc_path_dim=8`、endpoint4、BC encoder loaded |
| terminal gate 漏配 | 已修复 | 修复后 zero-residual 精确 `24/33` |
| 评估重复 scale | 已修复并归档 | `eval_pre_scale_fix/`；新动作 `abs_max<1.001` |
| 动作饱和是低倍率主因 | 排除 | `<=0.5` saturation 为 0，仍持续掉 success |
| 单一执行器通道主导 | 不支持 | 三通道 learner action RMS 接近 |
| SAC learner 被 frozen BC residual 完全压住 | 低倍率成立，高倍率不成立 | `0.125` 比例 `0.06–0.37`；`1.0` RPM 比例 `2.95` |

## 可证伪假设与当前判读

### H128-A：reward 的主项奖励与 success/terminal capture 不一致

- 成立：SAC 在低倍率下提高某些 path/energy reward 分量，但 terminal success 和
  timeout 变差；逐 step reward breakdown 能解释 86201 与其他 seed 的分化。
- 失败：reward 分量的变化与 success 同向，或没有稳定分量能解释退化。
- 最便宜证伪：对 `0.0/0.125/1.0` 记录逐 episode reward breakdown 和 terminal 事件；
  不训练。

### H128-B：terminal capture/credit 让策略学到“更平滑但更慢”的轨迹

- 成立：失败 episode 主要在 terminal 前仍快速/横向错过，或剩余时间不足；reward
  在 pre-terminal 区域给平滑/路径项更高回报，但不区分 near-goal capture。
- 失败：失败 episode 在 terminal 几何上与 control 无差异，timeout 来自随机扰动。
- 最便宜证伪：复用已有 episodes，比较 terminal eta/pnav 距离、terminal 前 32/64 步
  的路径误差、速度、reward 分量；必要时新增 trace-only evaluator。

## 最便宜 falsification

Goal128 预注册为 eval-only reward/terminal audit，先不训练。比较：

```text
zero residual (24/33)
0.125 learner residual (22.75/32.62)
1.0 learner residual (19.75/28.25)
```

输出 reward breakdown、terminal kind、terminal distance、pre-terminal 轨迹统计和
逐 seed 变化。只有找到稳定解释后，才允许改 reward 或 terminal credit 并重新训练。

## 下一步

1. 不扩 SAC formal，不换 REDQ/DroQ；
2. 预注册 Goal128 reward/terminal credit audit；
3. 若 H128-A/B 成立，再写新的训练型 Goal；
4. 保持 held-out 关闭。
