# Decision：Goal86 T2 3-seed 不进入 formal，转入 C-line/terminal-path 审计

**日期**：2026-11-11<br>
**决策**：`pivot`（不换算法族）<br>
**上游**：`doc/goals/goal86_t2_3seed_result_20261111.md`<br>
**Blocker**：`research/blockers/2026-11-11-goal86-t2-3seed-path-tradeoff.md`

## 裁决

- `T2_3SEED_POSITIVE_BUT_PATH_TRADEOFF`
- success：default +1.67/40、strict +1.33/40，但 pooled CI 跨 0；
- path：pooled +0.277 m，CI `[+0.190,+0.366]`，两个 seed 单独显著；
- PPO 3 seeds 全部 0/0/40，说明弱信号来自 SAC 残差而非接口；
- 不进入 5M formal。

## 为什么不是“再扩 seed”

path 退化在两个训练 seed 上同向且 CI 不含 0，success CI 又跨 0。扩 seed 只会增加
一个已知负结果的成本，不能把“path 显著退化、success 不显著”改写成可部署收益。

## 为什么不是“换 REDQ/DroQ”

残差确实在动作（L2 0.20–0.28），且 PPO 同接口是零翻转；当前证据是
terminal capture 与 path tracking 的冲突，不是样本效率不足的直接证据。

## 下一步

1. 不重训 T2；复用 Goal128/131/132 的 reward/terminal 几何审计，定位
   success 翻转是否发生在 near-goal 边界。
2. 继续 Goal136 C-line 的 S0–S4；C-line 必须保持 Goal85 `unbiased_noise_v1`、
   `position_source=eta`、零延迟和 Goal86 的 z_t/BC 接口。
3. Goal140 纵向 tracking reward 排在 C-line 之后。
4. 只有明确证明“探索/样本效率不足”时，才预注册 REDQ/DroQ 对照。
