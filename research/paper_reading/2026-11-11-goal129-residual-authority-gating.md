# Reading note：residual authority、action constraints 与 shielding

**日期**：2026-11-11<br>
**Goal**：Goal129<br>
**检索**：`paper-search`，query 包含 `residual control reinforcement learning base controller robotics`、`action shielding safe reinforcement learning`、`residual policy learning robot manipulation`；来源 arXiv / OpenReview；年份 2018–2026。

## 1. 最相关的四篇

### Bayesian controller fusion: Leveraging control priors in deep reinforcement learning for robotics

- 作者：Krishan Rana, Vibhavari Dasagi, Jesse Haviland 等
- 年份：2023； venue：Int. J. Robotics Res.
- 链接：https://openreview.net/forum?id=RFVrNpT1pW
- 可迁移点：不要只做“硬 gate 开/关”，而是让控制先验与学习策略按置信度/状态
  连续仲裁。Goal129 的 shell gate 是硬仲裁，G1.0 比 M1.0 更差，正好说明
  “在 approach 阶段完全撤掉 learner”会损失修正能力。
- 最小迁移：保留 frozen BC 作为先验，将 learner 的 authority 做成随距离和
  方向连续变化的权重，而不是 3 m 内直接置零。

### Residual Skill Policies: Learning an Adaptable Skill-based Action Space for Reinforcement Learning for Robotics

- 作者：Krishan Rana, Ming Xu, Brendan Tidd 等
- 年份：2022； arXiv:2211.02231
- 链接：http://arxiv.org/abs/2211.02231
- 可迁移点：在已有 skill/base controller 上学习有界 residual，比从零学习
  动作更稳；residual 的“幅值边界”是核心设计变量。
- 与 Goal129 的关系：C1.0 的 global cap 把 M1.0 从 19.75 拉回 22.75，
  支持“bounded residual”方向；但所有通道共用一个 cap，仍低于 zero。

### Benchmarking Actor-Critic Deep Reinforcement Learning Algorithms for Robotics Control with Action Constraints

- 作者：Kazumi Kasaura, Shuwa Miura, Tadashi Kozuno 等
- 年份：2023； arXiv:2304.08743；DOI:10.1109/LRA.2023.3284378
- 链接：http://arxiv.org/abs/2304.08743
- 可迁移点：动作约束/饱和会显著改变 actor-critic 的排序和稳定性；评价
  residual 方法时必须把 saturation、动作边界和约束处理单独报告。
- 与 Goal129 的关系：raw policy 近区饱和 28%–71%，cap 后 raw saturation
  仍在 0.77 左右；因此只看 success 会把“参数化饱和”与“策略方向错误”
  混在一起。Goal130 应继续记录 per-channel saturation。

### Verification-Guided Shielding for Deep Reinforcement Learning

- 作者：Davide Corsi, Guy Amir, Andoni Rodriguez 等
- 年份：2024； arXiv:2406.06507
- 链接：http://arxiv.org/abs/2406.06507
- 可迁移点：runtime shielding 能保证安全，但过于保守的 shield 会削弱任务
  性能；shield 的触发条件和覆盖范围需要和任务阶段匹配。
- 与 Goal129 的关系：3 m 内完全关闭 learner 的 shell gate 就是一个过于
  保守的 shield；它没有造成 safety 回归，但损失了 approach 修正，导致
  strict 从 19.75 进一步跌到 18.625。

## 2. 对 Goal129 的解释

四篇文献给出的共同结论不是“换一个更强算法”，而是：

```text
base controller prior 有价值；
residual 必须有界；
约束/shield 的粒度（何时、哪个通道、多大）比“有没有约束”更重要。
```

这与 Goal129 的数据一致：

- 硬 gate（G1.0）过度保守，损失 approach；
- global cap（C1.0）方向正确，但粒度过粗；
- 真正需要检验的是 channel-aware authority，而不是继续调一个全局标量。

## 3. Goal130 的最小可迁移组件

只迁移“有界 residual + 状态/通道级仲裁”，不复制任何论文的完整训练管线：

1. 对 `[rudder, stern_plane, propeller_rpm]` 分别计算
   `||L_c|| <= α_c ||B_c||`，先做 rpm-only、fin-only、per-channel 三组；
2. 增加方向一致性 `cos(L, B)` 阈值，检验“方向对但幅值错”；
3. 记录每通道的 learner/BC、raw saturation、applied saturation；
4. 仍用同一批 Goal126 checkpoint、dev20–59、strict/default、both frame；
5. 若全矩阵仍 `NO_EFFECT`，停止参数化家族，转 terminal capture /
   approach geometry。

## 4. 不采纳的部分

- 不引入新的 world model、learned controller 或离线数据管线；
- 不改变 PPO/SAC 主算法族；
- 不因文献里有“更好的 residual 结构”就跳过预注册 Gate；
- 不把论文的绝对性能数字迁移到本项目，只迁移机制和评价维度。
