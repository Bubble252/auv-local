# Decision：Goal138 后转 SAC mixed-horizon 与 PPO 信用时域

**日期**：2026-11-11<br>
**输入**：Goal138 `TERMINAL_CURRICULUM_TRADEOFF / PPO_NO_TERMINAL_DENSITY_EFFECT`<br>
**决策**：`pivot`

## 决策

1. SAC 停止 100% near-terminal；预注册 mixed-horizon `M50`（50% full + 50% T15）与 `M75`（25% full + 75% T15），各 3 seeds × 1M。
2. PPO 停止 near-terminal 密度线；预注册 gamma/GAE terminal credit horizon 对照（例如 gamma `0.995/0.98/0.95`），保持 reward、action、terminal 和评估不变。
3. 不扩 8 seeds / 5M；不继续扫 `start_progress_m`；不为 86203 单 seed 开 formal。
4. 不换 REDQ/DroQ；不打开 held-out；不改 success/goal radius。

## 理由

Goal138 首次给出 SAC 的跨 level 正信号（86203），但 100% near-terminal 造成 nominal strict -10；T10 更短起点触发 safety。最便宜的下一步是恢复 full-path anchor，而不是换算法。PPO 的 null 与 residual 近零一致，需改变信用时域而不是再增加 terminal 样本。
