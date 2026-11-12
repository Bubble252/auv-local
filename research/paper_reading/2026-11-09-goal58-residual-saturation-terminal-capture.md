# Goal58 阅读笔记：残差饱和与终端捕获

日期：2026-11-09<br>
问题：为什么 off-policy 命令残差能变大，却把 success 和 plane-miss 弄坏？下一轮
应该先换算法，还是先查动作参数化和 terminal capture？

## 1. 检索

工具：`paper-search`（arXiv 返回 18 条；Semantic Scholar 返回 429，OpenReview
未安装）；再用 arXiv API 核验关键论文编号。关键词：

- `saturated tanh action residual reinforcement learning continuous control`
- `noisy network exploration saturation actor critic`
- `residual policy action scaling bounded correction`

本地已有 terminal capture 文献见
`2026-11-09-goal31-terminal-path-vertical-gap.md`；本 note 不重复水下 docking
文献，只补充 off-policy、NoisyNet、SAC/REDQ/DroQ 和 residual 的适用条件。

## 2. 已核验论文

### DDPG，arXiv `1509.02971`

**Continuous control with deep reinforcement learning**。

- 连续动作的 deterministic actor-critic；actor 通常用 `tanh` 做有界输出。
- 对本项目的含义：DDPG 的形式本身允许 residual，但 `tanh` 边界处的梯度会变小；
  当 critic 把动作推到边界时，deterministic mean 可以长期停在 `±1`。

### TD3，arXiv `1802.09477`

**Addressing Function Approximation Error in Actor-Critic Methods**。

- twin critics、target policy smoothing、delayed policy update。
- 对本项目的含义：Goal58 已经使用 TD3-style 机制，但仍出现饱和；因此“再加一个
  稳定化技巧”不等于解决动作参数化。

### SAC，arXiv `1801.01290`

**Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor**。

- off-policy maximum-entropy，随机 actor，通常用 squashed Gaussian 和 log-prob
  Jacobian correction。
- 对本项目的含义：SAC 是“优化/探索”分支的合理候选，但它仍需要 bounded action
  参数化和 terminal-capture 目标；当前证据不足以把 Goal58 失败直接归因于 DDPG
  样本效率。

### NoisyNet，arXiv `1706.10295`

**Noisy Networks for Exploration**。

- 用可学习参数噪声做 exploration；评估时使用 mean weights。
- 对本项目的含义：Goal58 的 deterministic probe 正是 mean-weight policy；mean
  已经饱和，所以不能把问题归因于“评估时还在加噪”。NoisyNet 可能影响训练分布，
  但不是最终闭环失败的唯一解释。

### REDQ，arXiv `2101.05982`

**Randomized Ensembled Double Q-Learning: Learning Fast Without a Model**。

- ensemble + high update-to-data ratio，目标是样本效率。
- 对本项目的含义：只有未来证据表明“数据利用不足/更新不够”才是瓶颈时，才值得做
  REDQ 对照；它不能替代动作参数化/terminal capture 修复。

### DroQ，arXiv `2110.02034`

**Dropout Q-Functions for Doubly Efficient Reinforcement Learning**。

- dropout Q + high UTD，强调计算和样本效率。
- 对本项目的含义：与 REDQ 相同，属于“样本效率”分支，不是当前首要诊断。

### Residual Off-Policy RL，arXiv `2509.19301`

**Residual Off-Policy RL for Finetuning Behavior Cloning Policies**。

- frozen BC base + 轻量 per-step residual，off-policy、sparse reward。
- 对本项目的含义：residual off-policy 是可行方向，但该工作也强调 residual 的
  作用是对闭环误差做局部修正，不是无约束地重写 base command；Goal58 的 `±1`
  饱和不符合“局部修正”的尺度。

### RLPP，arXiv `2501.17311`

**RLPP: A Residual Method for Zero-Shot Real-World Autonomous Racing on Scaled Platforms**。

- 在 Pure Pursuit 上学习 residual，直接优化任务表现。
- 对本项目的含义：控制器 + residual 接口可行，但任务奖励必须对应闭环目标；
  如果 residual 只被 path proxy 奖励，terminal capture 仍可能失败。

## 3. 综合判断

1. **NoisyNet 不是唯一原因**：评估使用 mean weights 时 deterministic policy
   仍饱和；final checkpoint 的 sigma 也没有膨胀（hidden≈0.300，output≈0.100），
   所以 NoisyNet 只影响探索，不是确定性动作饱和的原因。
2. **DDPG/TD3 的边界饱和是参数化问题**：`tanh` 在 `±1` 附近梯度小，若 critic
   把动作推到边界，策略会锁死在最大命令修正。
3. **SAC/REDQ/DroQ 不是自动修复**：SAC 解决的是随机策略/熵探索，REDQ/DroQ
   解决的是样本效率；它们都不能保证 terminal capture。
4. **path RMSE 与 terminal capture 解耦**：Goal30/31 已显示 path 改善可以伴随
   terminal offset 变差；Goal58 的 scale `0.50` 也复现了同一方向。
5. **下一步应先做 bounded residual + terminal-aware objective**；如果新诊断明确
   是优化/探索，再做 SAC；如果是样本效率，再做 REDQ/DroQ。

## 4. 对下一 Goal 的约束

- 不继续 Goal58 的 reward/gamma/seed/scale 扫描；
- 不把 eval-time scale 诊断写成 PASS；
- 不打开 held-out `100–149`；
- 新 goal 必须有独立的 Research Contract、预注册 Gate 和最小 falsification；
- 先证明 residual 不在 `±1` 饱和，再讨论算法替换。

## 5. 引用核验

| 文献 | 核验方式 | 状态 |
|---|---|---|
| DDPG, arXiv 1509.02971 | arXiv API 标题/日期 | verified |
| TD3, arXiv 1802.09477 | arXiv API 标题/日期 | verified |
| SAC, arXiv 1801.01290 | arXiv API 标题/日期 | verified |
| NoisyNet, arXiv 1706.10295 | arXiv API 标题/日期 | verified |
| REDQ, arXiv 2101.05982 | arXiv API 标题/日期 | verified |
| DroQ, arXiv 2110.02034 | arXiv API 标题/日期 | verified |
| Residual Off-Policy RL, arXiv 2509.19301 | arXiv API 标题/日期 | verified |
| RLPP, arXiv 2501.17311 | arXiv API 标题/日期 | verified |
