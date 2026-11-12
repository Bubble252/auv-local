# Decision：停止 authority 正则，转入 Goal136 diversity

**日期**：2026-11-11<br>
**输入**：Goal135 `AUTHORITY_SAC_HARM`<br>
**决策**：`pivot`

## 决策

不再扫 `rho/beta`，不扩 5M formal，不换 seed 救 Goal135。保留已有 24 个 1M
checkpoint 和 96 个评估作为失败证据。

下一执行块为 Goal136 Phase A1：

1. SAC：`D1/D2 × 86201–86203 × 1M`，与 Goal134 N0/zero 对照；
2. PPO：`N0/D1/D2 × 86201–86203 × 1M`，使用 Goal125 B15 契约，单独裁决；
3. 评估：5 levels × 2 scenarios × dev40；
4. 只有 `DIVERSITY_SIGNAL` 才扩 8 seeds；`DIVERSITY_TRADEOFF` 才进入 A2 显式 nominal anchor。

## 不做什么

- 不把 Goal135 的失败解释为“需要更多正则”；
- 不在 Goal136 中同时改 diversity 和 authority；
- 不打开 held-out；
- 不修改 reward、terminal、success、goal radius；
- 不把 SAC 和 PPO 结果合并成一个 Gate。

## 理由

Goal135 已证明“压低 residual magnitude”本身不够。Goal136 检验一个不同的可证伪
假设：当前单个 nominal 训练分布是否让策略学到对 seed 不稳定、对压力不迁移的修正。
这比继续在同一 authority 轴上扫参数更能区分“方向/覆盖不足”和“优化器不足”。
