# Goal138 blocker：近终端课程出现 SAC nominal tradeoff 与 T10 safety stop

**日期**：2026-11-11<br>
**状态**：`TERMINAL_CURRICULUM_TRADEOFF / PPO_NO_TERMINAL_DENSITY_EFFECT`

## 现象

- SAC T15：2/3 seeds 在至少两个 level 上有 strict 信号，但 nominal strict 比 N0 低 10，default 高 12。
- SAC T10：combined20 出现 4 个 timeout，触发 `TERMINAL_SAFETY_STOP`。
- PPO T15/T10：4/5 个 level 与 N0 完全相同；residual L2 约 1%，没有可迁移增益。

## 已排除

- 起点实现错误：默认 0 与旧行为 parity 通过；评估强制 0，旧结果未被覆盖。
- terminal density 没变：SAC T10 episodes 约 1.4–1.5×，PPO T10 resets 约 1.7×。
- reward/terminal/success/radius 改变：均未改变。
- 单纯 seed 噪声：86203 五个 level 全正，86201 五个 level 多数负，说明是真实 seed 异质性。
- PPO 残差方向正确但幅度不足：既往 5×/20×/50× 放大无收益；本轮 residual 仍约 1%。

## 可证伪 hypothesis

1. **H138b-A：full-path anchor 能恢复 nominal strict**。若 SAC mixed-horizon（例如 50% full + 50% T15）在 nominal strict 回到 N0 附近，同时保留至少两个 stress level 的 strict 增益，则支持；若 nominal 仍低或 stress 增益消失，则否定。
2. **H138b-B：T15 的 stress 增益来自 86203 式 seed 异质性，不能通过分布混合稳定**。若 mixed-horizon 的 3 seeds 仍只有单 seed 正，则支持；若至少 2/3 seeds 在 stress 上转正且 nominal 不退，则否定。
3. **H138b-C：T10 的 safety stop 是时间预算不足，不是密度本身**。若 T10 加长最大 episode steps 后 safety 消失且 strict 改善，则支持；若仍 safety，则否定。该假设不阻塞当前 mixed-horizon 实验。

## 最便宜证伪

- SAC mixed-horizon：`M50`（50% full + 50% T15）与 `M75`（25% full + 75% T15），各 3 seeds × 1M，评估同五档 dev40；
- 只比较 nominal strict/default 与 current15/current20 strict；
- 不训练 PPO mixed-horizon；PPO 转向 gamma/GAE 信用时域。

## 决策

`pivot` 到 SAC mixed-horizon；不继续扫起点，不扩 formal，不打开 held-out。
