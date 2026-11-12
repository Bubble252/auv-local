# Blocker：Goal86 T2-SAC 3-seed 出现 success/path 交易

**日期**：2026-11-11<br>
**Goal**：Goal86 T2-SAC 3-seed<br>
**状态**：`T2_3SEED_POSITIVE_BUT_PATH_TRADEOFF / NO_FORMAL`<br>
**证据**：`doc/goals/goal86_t2_3seed_result_20261111.md`、`artifacts/lowlevel/goal86/screen/t2_sac_*_dev40/episodes.json`

## 现象

- default：28.67/40 vs zero 27/40，3/3 seeds 为正；
- strict125：15.33/40 vs zero 14/40，2/3 seeds 为正、1 个平；
- pooled success CI 跨 0；
- pooled path RMSE `+0.277 m`，CI `[+0.190,+0.366]`，86202/86203 单独显著退化；
- PPO 同协议全部 0/0/40 平局，说明弱信号来自 SAC 残差，不是接口本身。

## 已排除

| 排除项 | 证据 |
|---|---|
| T2 checkpoint 缺失/训练未完成 | 3/3 都有 1,000,192 transitions 和 final.zip |
| dev seed 不一致 | 所有 run 都是 seeds 20–59 |
| 评估起点/噪声契约变化 | 复用 Goal85 unbiased_noise_v1 + eta termination；未改环境 |
| 残差没动 | SAC residual L2 0.198–0.283，action abs max ≈1.0 |
| PPO/SAC 接口混淆 | PPO 3 seeds 全部 0 翻转；SAC 才有弱 success 信号 |

## 可证伪假设

1. **H86-3S-1：T2 success 的提升在 3 seeds 上显著且可进入 formal。**
   - 预测：default/strict pooled success CI 不跨 0，且 path 不显著退化。
   - 证伪：现有 episodes 逐 seed 配对 + bootstrap。
   - 结果：**FALSIFIED**。success CI 跨 0，path CI 显著为正。

2. **H86-3S-2：path 退化只是 seed 噪声，扩 seed 可救。**
   - 预测：path 退化应只集中在单个 seed，方向不一致。
   - 证伪：比较 3 个训练 seed 的配对 path Δ。
   - 结果：**FALSIFIED**。86202/86203 两个 seed 同向且 CI 不含 0。

3. **H86-3S-3：T2 的 success/path 冲突来自 residual 幅度过大；缩小 scale 即可保留 success。**
   - 预测：residual L2 应与 path Δ 单调正相关，且 86201 低 path Δ 对应更低 residual。
   - 证伪：读现有 residual L2 与 path Δ。
   - 结果：**NOT SUPPORTED**。86201 residual 0.256、path +0.014；86202 residual 0.198、
     path +0.420；86203 residual 0.283、path +0.396。不是简单幅度关系。

4. **H86-3S-4：冲突来自 terminal capture 与 path tracking 的 reward/惩罚不对齐。**
   - 预测：success 翻转集中在 near-goal 边界，失败 episode 的 path 误差在 pre-terminal
     已经分出；需要 Goal128/131/132 的 trace 审计确认。
   - 最便宜证伪：复用已有 Goal128/131/132 的 reward/terminal 几何结果，不重训。
   - 结果：**SUPPORTED AS NEXT HYPOTHESIS**，但不能在本 Gate 内改 reward/terminal。

## 不做什么

- 不进入 5M formal；
- 不扫 T2 scale、seed、reward scale；
- 不把弱 success 信号包装成 PASS；
- 不打开 held-out；
- 不换 REDQ/DroQ。
