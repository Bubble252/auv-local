# Decision：停止 D-line，转入终端样本密度课程

**日期**：2026-11-11<br>
**输入**：Goal136 `DIVERSITY_SAFETY_STOP / PPO_RESIDUAL_COLLAPSE`<br>
**决策**：`pivot`

## 决策

停止 Phase A1 D-line 的扩 seed 和 formal，不启动 A2 nominal anchor。下一轮不再扩大
path/current/plant 宽度，而是把训练样本集中到 terminal 段：

- 保持 reward、terminal、success、goal radius、sensor、dev seeds 不变；
- 训练短时域或近终端起始 episode，使每 1M transitions 覆盖更多 goal/capture 事件；
- PPO 与 SAC 分开训练、分开裁决，各先 3 seeds × 1M；
- 只有出现跨 seed、跨 level 的 strict 增益且无 safety，才扩 8 seeds / 5M。

## 不做什么

- 不用 D1/D2 的 86203 单 seed 结果包装成 diversity signal；
- 不继续扫 D-line 的 path bank/current 幅度；
- 不通过加 residual scale 救 PPO；
- 不在没有 terminal 样本证据前开 REDQ/DroQ；
- 不打开 held-out。

## 理由

本轮同时排除了“环境覆盖更宽”和“PPO 方向只是幅度小”。剩余最便宜的区分是
terminal sample density：如果 terminal-focused 课程仍无收益，下一步应审计
reward/capture 或动作通道，而不是继续扩大环境宽度。
