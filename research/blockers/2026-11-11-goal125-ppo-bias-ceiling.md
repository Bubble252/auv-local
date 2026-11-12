# Blocker：Goal125 固定偏置之上 PPO 仍无增量

**日期**：2026-11-11<br>
**Goal**：Goal125<br>
**状态**：`PPO_BIAS_CEILING / NETWORK_PARITY_OK / NOISENET_ABSENT / HELDOUT_CLOSED`

## 现象

8 个 10M PPO checkpoint 在 strict/default dev40 上：

- B15/B20 都是 strict 24/40、default 33/40，与 fixed-bias control 逐 seed 相同；
- BP3（advantage norm + reward scale 0.1）23.5/40；
- Z15 20/40，丢掉 fixed bias 恢复的 `29/31/33/56`；
- timeout/safety 全 0；KL 约 `9e-5–1.7e-4`。

## 已排除

| 候选原因 | 判读 | 证据 |
|---|---|---|
| 网络结构接错 | 排除 | Goal119/Goal108 initial 63/63 tensor parity；无 NoisyNet；actor 输出层零初始化 |
| NoisyNet 主导 | 排除 | 当前 PPO 没有 `rho/sigma/noise` 参数，eval 为 deterministic mean |
| fixed bias 不生效 | 排除 | policy+bias strict 24/40，success seed 与 Goal124 完全相同 |
| control 不复现 | 排除 | 86201/86202 都 strict 24、default 33 |
| 训练/评价 scale 不齐 | 排除 | B20 与 B15 均 24/24，scale 1.5 vs 2.0 无差异 |
| reward 绝对尺度 | 排除为主因 | BP3 reward×0.1 + advantage norm 仍 23.5，不升 |
| 动作通道无权限 | 排除 | Goal121/122 已显示 stern/rpm 注入有权限 |
| terminal capture/reward 完全错位 | 已审计 | Goal96/117 没有单一可修复的 terminal 缺陷；仍需保留为后续审计项 |

## 可证伪假设

### H125X-A：PPO 的优化/探索是剩余主因

- 成立时应看到：同 fixed bias 下，off-policy SAC 在 1M 内提高 strict success，
  或出现可解释的策略增量，而不是像 PPO 一样停在零附近。
- 失败时应看到：SAC 也停在 24/40，或继续把动作推到饱和。此时优化器不是主因。
- 最便宜证伪：Goal86 P-A 架构的 SAC 1M screen，2 seeds，固定同一 bias。

### H125X-B：固定偏置已经是当前观测/动作参数化的上限

- 成立时应看到：SAC 和 PPO 都不能超过 fixed-bias control，success seed 集合也不扩。
- 失败时应看到：至少一个 SAC seed 稳定得到 `>=26/40`，且不丢 control seed。
- 最便宜证伪：同上的 SAC 1M screen；若失败，不进入 5M formal。

### H125X-C：SAC 的退化来自未对齐的 path 初始化

- 成立时应看到：Goal86 P-A（`bc_path_dim=8`、endpoint4、加载 BC encoder）的
  deterministic action 不饱和，且 initial zero-residual 精确复现 fixed-bias control。
- 失败时应看到：即使对齐 P-A，SAC initial 或训练后仍出现大动作饱和。
- 最便宜证伪：SAC init-only checkpoint + dev40 zero-residual parity；在正式训练前完成。

## 下一步

预注册 Goal126 SAC P-A 同协议对照。先做 init-only parity，再跑 1M screen。
在 Gate 记录前不改 success、goal radius、seed 或 held-out。
