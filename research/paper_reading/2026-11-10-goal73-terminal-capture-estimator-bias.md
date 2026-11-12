# Reading note：terminal capture、goal-reaching 与状态估计偏差

日期：2026-11-10<br>
关联：Goal73 / Goal74 candidate<br>
目的：判断是否需要引入外部方法，还是先用本地只读重分类解决

## 1. 问题

Goal73 Step 1 发现 false-success 不是 command-demand/曲率问题，而是
terminal `p_nav` 距离刚好在 capture 半径内、true `eta` 距离在半径外。
需要回答：

1. 这是不是常见的“goal-reaching 在噪声/偏差观测下的终端判定”问题？
2. 有没有现成方法能直接用于当前的 `p_nav`/`eta` 双状态评测？
3. 下一步应该改控制器、改 capture 口径，还是改状态估计？

## 2. 检索

使用 `paper-search` skill，查询：

```text
goal-conditioned reinforcement learning terminal capture state estimation bias noisy observations
terminal state estimation bias goal reaching reinforcement learning
observation bias state estimation error reinforcement learning control
```

来源：arXiv（主要）、Semantic Scholar/OpenAlex/OpenReview/Crossref
（部分请求遇到 429，未等待完成）。筛选年份 2019–2026。

候选论文：

| 论文 | 与本问题的关系 | 局限 |
|---|---|---|
| Revisiting Sparse Rewards for Goal-Reaching RL (arXiv 2407.00324) | 讨论 goal-reaching 的稀疏奖励和终端成功判定 | 不处理 `p_nav`/`eta` 双状态偏差 |
| Optimal Goal-Reaching RL via Quasimetric Learning (arXiv 2304.01203) | 学习 goal-conditioned 距离/表示 | 目标是表示学习，不是 capture 口径 |
| Metric Residual Networks for Sample Efficient Goal-Conditioned RL (arXiv 2208.08133) | 改进 goal-conditioned 样本效率 | 与当前 frozen policy 评测无直接关系 |
| GOPlan: Goal-conditioned Offline RL by Planning with Learned Models (arXiv 2310.20025) | 用模型做 goal-conditioned planning | 需要训练世界模型，不是当前瓶颈 |
| Data-Efficient DRL for Attitude Control of Fixed-Wing UAVs (arXiv 2111.04153) | 真实飞行中的状态估计与控制 | 平台/任务不同，不能直接迁移 |

## 3. 结论

现有文献没有直接处理“部署用 `p_nav` 估计与 true `eta` 在 capture 边界
不一致”的评测问题。当前最便宜、最可证伪的下一步不是引入新算法，而是：

1. 对现有 Goal71 episodes 做 `margin = 0 / 0.10 / 0.25 / 0.50 m` 的
   只读重分类；
2. 计算 `p_nav - eta` 在 goal 方向的投影，判断是系统偏差还是零均值噪声；
3. 如果 margin 重分类能解释 false-success，再预注册 capture margin 或
   bias correction 的单因素 arm；
4. 在此之前不换 SAC/REDQ/DroQ，也不继续加控制 authority。

## 4. 对 Step 顺序的影响

不影响 Goal73 的 Step 0/1，但把 Step 2 从“实现 speed governor”改为
“terminal capture / estimator bias 只读审计”。这是 `pivot`，不是
`stop`：主线仍是 terminal capture/动作参数化，只是先处理标签/估计。

## 5. 证据

- `research/blockers/2026-11-10-goal73-command-demand-not-discriminative.md`
- `artifacts/lowlevel/goal73_command_demand_audit_20261110/analysis/goal73.json`
- `artifacts/lowlevel/goal73_command_demand_audit_20261110/analysis/goal73.md`
