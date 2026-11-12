# Reading note：residual RL 的 cold start、尺度错配与样本效率

**日期**：2026-11-12<br>
**用途**：Goal171/172 的残差训练失败机制与 Goal173 设计依据<br>
**检索**：arXiv API，查询 `residual reinforcement learning`、`REDQ`、
`zero initialized residual`。以下结论只使用能直接对应到当前证据的条目。

## 1. 直接相关的 residual RL 工作

| 工作 | 关键结论 | 对当前项目的含义 |
|---|---|---|
| Alakuijala et al., *Residual Reinforcement Learning from Demonstrations*, arXiv:2106.08050 | residual RL 适合把传统控制器/policy 作为 base，再学习修正；但需要 reward 能区分 base 与修正后的行为 | Goal85 的 base 与 Goal171 的 reward 契约本身合理，不能只用 success 判断 residual |
| Dodeja et al., *Accelerating Residual Reinforcement Learning with Uncertainty Estimation*, arXiv:2506.17564 | residual RL 的样本效率仍受稀疏 reward 和 base policy 不确定性影响；应把探索集中在 base 不自信的区域 | 当前 residual 在零附近，可能是探索没有覆盖需要修正的状态，而不是动作接口完全无效 |
| Ma et al., *What Makes Value Learning Efficient in Residual Reinforcement Learning?*, arXiv:2602.10539 | 指出两个关键瓶颈：**cold start pathology**（critic 不知道 base 周围的价值地形）和 **structural scale mismatch**（residual 相对 base action 太小）；提出 base-policy transitions 的 value anchor、critic normalization | 与本项目 200k 下 PPO residual 近零、SAC residual 有动作但无路线净收益高度一致；Goal173 应优先做 value warmup/critic normalization，而不是直接换算法 |

## 2. 样本效率方法

| 工作 | 关键结论 | 何时才用 |
|---|---|---|
| Chen et al., *Randomized Ensembled Double Q-Learning*, arXiv:2101.05982 | REDQ 用高 UTD、Q ensemble 和 in-target minimization 提高连续动作样本效率 | 只有在相同 reward/action 下证明“增加更新次数能改善”后才值得引入 |
| Hiraoka et al., *Dropout Q-Functions for Doubly Efficient RL*, arXiv:2110.02034 | DroQ 以 dropout + layer norm 的小 ensemble 降低 REDQ 计算成本 | 作为 REDQ 的计算效率替代，不应在接线/动作方向未经验证时先上 |

当前 Goal172 没有证明“更新次数不足”是唯一原因：放大 residual 反而变差，说明
不是简单把同一方向放大即可。因此 REDQ/DroQ 仍不是下一步首选。

## 3. 对本项目的设计含义

1. **保留 base + residual 结构**：Goal85/86 的 actuator base、z_t、BC 先验不应被
   residual RL 的失败否定。
2. **先修 value learning 的 cold start**：在 Goal173 中加入 base-policy
   transitions 作为 value anchor，或做 critic normalization；这比直接增加 5M
   预算更便宜。
3. **把 residual 尺度与 base 分离**：当前 `actuator_residual_scale=0.25` 下
   PPO 确定性残差物理量过小；但单纯放大到 2×/5× 会伤路线。需要的是
   “可学习尺度 + 方向约束”，不是固定大幅放大。
4. **评价仍以整条路线为主**：不能因为 terminal success 或单个 scale 的符号变化
   就改 reward/terminal；Goal172 的 scale 表必须和五段/速度包线一起读。

## 4. 下一步的文献约束

Goal173 的预注册应明确：

- 仍然只比较 PPO 与 SAC，不新增 DDPG/TD3；
- residual cold-start 处理只改 value warmup/normalization，不同时改 reward、
  terminal、success 和 goal radius；
- 先在 1 seed/1M 做 falsification，再决定 3 seeds/1M；
- 只有出现“同 reward、同动作接口下更新效率仍不足”的直接证据，才开 REDQ/DroQ。
