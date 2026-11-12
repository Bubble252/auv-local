# Goal162 决策：速度契约无效，pivot 到 Goal163 审计

**日期**：2026-11-12<br>
**决策**：`pivot`<br>
**上游**：Goal162 `INVALID_SPEED_CONTRACT`

## 为什么不是继续扩 seed 或换算法

1. H3 zero 与 gain-residual policy 在相同 current level 上的峰值和超限比例几乎
   相同，residual 不是必要条件；
2. 超限出现在 `>10 m` 的远段，而 Goal162 的 gain gate 是 `10→2 m`；
3. 训练 manifest 的 `speed_contract.ok=true` 只验证配置层上界；
4. 没有证据表明这是探索不足或样本效率不足，因此不能启动 SAC/PPO/REDQ/DroQ
   对照；
5. 速度是 formal comparison 的共同安全前提，必须先修复或明确契约。

## 保留的证据

- current10 default `+2/120`、strict125 `+7/120`；
- current15 default `+8/120`；
- near-goal 固定增益/距离驱动现象和 signed-error 相关性不足；
- 这些只作为诊断信号，不作为 formal 结论。

## 新路线

Goal163：只读/最便宜的 speed-envelope audit，验证 speed integral/anti-windup、
ground-speed cap、转向瞬态和 plant mismatch；不训练、不换算法、不改 reward/
terminal/success/radius、不打开 held-out。

## 失败后的下一步

- 若 H162-S1 支持：预注册 anti-windup/RPM 回收的最小控制器修正，先做 zero-only
  deterministic replay，再重跑同一 paired Gate；
- 若 H162-S2 支持：预注册闭环 ground-speed safety cap 或更保守的 current/speed
  envelope，但保留课程 v2 的 `2.5 m/s` 联合约束；
- 若两者都不支持：检查 plant backend、速度观测定义和 evaluation 速度计算；
  仍不换 RL 算法。
