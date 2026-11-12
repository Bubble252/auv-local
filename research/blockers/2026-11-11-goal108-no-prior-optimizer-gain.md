# Goal108 blocker：B20 先验上 PPO/SAC 都没有增量收益

**日期**：2026-11-11<br>
**Goal**：Goal108<br>
**状态**：`NO_PRIOR_OPTIMIZER_GAIN / SAC_RESIDUAL_EXPLOSION`

## 1. 现象

Goal108 的 6 个 1M training run 全部完成并做了 dev40 CPU/float64 评价：

```text
B20       default 33/40, strict125 20/40
B20-PPO   default 33/40, strict125 20/40（3/3 seeds）
B20-SAC   default  4/5/9, strict125  2/2/5（3 seeds）
```

PPO 的 learned increment L2 ratio 只有 `0.0060–0.0095`，逐 seed 与 B20
`gained=0, lost=0`。SAC 的 applied residual/base L2 达到 `0.74–1.38`，
normalized action abs max 约 `0.996`；86201 有 4+4 个 timeout，86203 有
17+17 个 timeout 和 2+2 个 safety。

## 2. 已排除项

| 解释 | 状态 | 证据 |
|---|---|---|
| base checkpoint 错配 | 排除 | SHA256 匹配；PPO iteration 0 / SAC initial 均复现 B20 33/20 |
| 网络接线/维度错位 | 排除 | actor input 131；SAC critic 134；`z_t64 + zero-DADE64 + command3`；与 Goal89/104/107 历史 parity 一致 |
| NoisyNet 主导 | 排除 | 当前 PPO/SAC 无 `rho`/sigma；PPO 只有 `std=(3,)`，SAC 只有 policy `log_std`；evaluation deterministic |
| PPO 基座被破坏 | 排除 | PPO final 的 zero-residual 评价仍为 33/20 |
| 只是 seed 噪声 | 排除 | PPO 3/3 完全不动；SAC 3/3 都大幅低于 B20，且伴随 timeout/safety |

## 3. 可证伪 hypothesis

### H108X-A：SAC replay 能从 B20 附近学到受控的小增量

- 机制：zero-init head 加 replay 应该先学习较小的修正，再逐步扩展。
- 预测：increment L2 ratio 保持在可解释的小范围，success 不应低于 B20。
- 结果：`FALSIFIED`。SAC 的 applied residual/base 到 `0.74–1.38`，action
  接近全饱和，success 降到 `6.1/40`（default 均值）和 `3.0/40`（strict 均值）。

### H108X-B：PPO 的零初始化小增量足以修正 strict 失败种子

- 机制：失败只需要最后阶段的小方向修正，不需要重新学习控制。
- 预测：至少 2/3 seeds strict >=23，default 保住 33。
- 结果：`FALSIFIED`。3/3 PPO strict 仍是 20，default 仍是 33；increment 太小且
  对闭环 outcome 无可测影响。

### H108X-C：网络结构/NoisyNet 解释大残差

- 机制：参数噪声或编码维度错位导致动作被随机放大。
- 预测：checkpoint 中存在 noise/sigma 参数，或 deterministic mean 与 stochastic
  sample 差异大。
- 结果：`FALSIFIED`。state_dict 无 NoisyNet，评价 deterministic；大残差来自
  SAC 的 mu 本身。

### H108X-D：优化器/信用轴是当前主要瓶颈

- 机制：同 B20 下，换 SAC replay 或 PPO 小增量即可改善 strict。
- 预测：至少一个 arm 达到 Goal108 PASS。
- 结果：`FALSIFIED`。两个 arm 都没有 PASS；SAC 还出现了退化。

## 4. 判读

本 blocker 不是“训练预算不够”或“网络没接好”，而是：

```text
B20 prior + PPO/SAC optimizer axis 没有产生可用的闭环增量；
SAC 在没有 residual 约束时会把 deterministic mu 推到饱和；
PPO 的 zero-init 小增量几乎不移动闭环。
```

因此停止 PPO/SAC 优化器轴，不扩 5M，不换 REDQ/DroQ/DADE。下一步只做只读的
terminal capture / reward / residual direction 审计。

## 5. 证据

- `doc/goals/goal108_result_20261111.md`
- `doc/goals/goal108_network_and_noisenet_audit_20261111.md`
- `artifacts/lowlevel/goal108_b20_prior_optimizer_20261111/B20_PPO_s86201..03/`
- `artifacts/lowlevel/goal108_b20_prior_optimizer_20261111/B20_SAC_s86201..03/`
- `artifacts/lowlevel/goal108_b20_prior_optimizer_20261111/B20_SAC_s86201..03_{default,strict125}_cpu/`

