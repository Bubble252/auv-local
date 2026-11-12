# Goal59 阅读笔记：softsign 仍饱和，先查动作、奖励和终端捕获

日期：2026-11-09<br>
问题：Goal59 已经把 `tanh` 换成 softsign、加了 terminal gate、并接入 12D
endpoint context，为什么 8 个 formal seed 的 deterministic/raw saturation 仍是
`100%`？这是不是 NoisyNet 主导？下一步应该换 SAC/REDQ/DroQ，还是先查动作参数化、
reward 和 terminal capture？

## 1. 检索与核验

工具：`paper-search`，先用多查询检索，再用 arXiv API 按 ID 核验标题、日期和摘要。

- 检索词包括：`action saturation tanh policy continuous control`、
  `bounded residual policy learning`、`residual policy learning robot control`、
  `goal conditioned reward terminal success`、`bang-bang action saturation`。
- arXiv 返回 18 条和 15 条两组结果；Semantic Scholar、OpenAlex、Crossref 在本次
  网络环境下超时或返回 429，因此本 note 的引用以 arXiv 核验结果为准，不把未核验
  的搜索结果写成结论。
- 关键新证据是 2026 年的负结果论文 `arXiv:2609.09478`：它直接检验了“恢复 tanh
  边界处 Jacobian 信号”能否改善 bang-bang 控制，结论是否定的。这与 Goal59
  的 softsign 结果相互印证：换边界形状不等于让策略学会终端任务。

## 2. 已核验论文

### DDPG，arXiv `1509.02971`

**Continuous control with deep reinforcement learning**。

- 连续动作的 deterministic actor-critic，常用 `tanh` 产生有界动作。
- 对本项目的含义：DDPG 允许 residual，但边界处梯度会变小；当 critic 把动作推到
  边界时，deterministic mean 可以长期停在边界。Goal59 的 raw saturation `1.0`
  说明 softsign 只是换了一个高幅值边界，没有改变“策略主动选择边界”这件事。

### TD3，arXiv `1802.09477`

**Addressing Function Approximation Error in Actor-Critic Methods**。

- twin critics、target policy smoothing、delayed actor update。
- 对本项目的含义：Goal58/59 已经使用 TD3-style 稳定化，仍然饱和。因此“再加一个
  稳定化技巧”不能替代动作接口和任务目标的对齐。

### SAC，arXiv `1801.01290`

**Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor**。

- off-policy maximum-entropy，随机 actor，通常用 squashed Gaussian 和 log-prob
  Jacobian correction。
- 对本项目的含义：SAC 主要改变探索和优化目标，不自动保证 residual 有界或
  terminal capture 对齐。只有诊断明确显示“优化/探索不足”时，才值得做同协议对照。

### NoisyNet，arXiv `1706.10295`

**Noisy Networks for Exploration**。

- 用可学习参数噪声做 exploration；评估时使用 mean weights。
- 对本项目的含义：Goal59 的 deterministic probe 正是 mean-weight policy。
  8/8 seed 的 mean 已经饱和，sigma 仍在初始化附近；因此 NoisyNet 可以解释训练期
  探索分布，但不能解释确定性策略为什么贴着边界。

### REDQ，arXiv `2101.05982`

**Randomized Ensembled Double Q-Learning: Learning Fast Without a Model**。

- Q ensemble + 高 update-to-data ratio，目标是样本效率。
- 对本项目的含义：只有证据表明“数据利用不足/更新不够”才是瓶颈时，才值得做
  REDQ 对照。Goal59 的 residual 非零且饱和，不是“学不动”的形态。

### DroQ，arXiv `2110.02034`

**Dropout Q-Functions for Doubly Efficient Reinforcement Learning**。

- dropout Q + 高 UTD，强调计算效率和样本效率。
- 对本项目的含义：与 REDQ 相同，属于样本效率分支；不能替代动作/奖励/终端修复。

### Residual Policy Learning，arXiv `1812.06298`

**Residual Policy Learning**。

- 在已有的、可微或不可微的控制器上学习 residual，用 RL 改善次优控制器。
- 对本项目的含义：residual 的合理语义是“对 base controller 的局部修正”。当
  residual L2 达到 `10.5%–22.5%` 且动作饱和时，它更像在重写 base command，
  而不是做局部补偿。

### Residual Off-Policy RL，arXiv `2509.19301`

**Residual Off-Policy RL for Finetuning Behavior Cloning Policies**。

- frozen BC base + 轻量 per-step residual，off-policy、sparse reward。
- 对本项目的含义：residual off-policy 可行，但前提是 residual 保持轻量并直接
  改善闭环任务指标；无约束的大 residual 不符合该工作的使用条件。

### RLPP，arXiv `2501.17311`

**RLPP: A Residual Method for Zero-Shot Real-World Autonomous Racing on Scaled Platforms**。

- 在 Pure Pursuit 上学习 residual，直接优化任务表现。
- 对本项目的含义：控制器 + residual 接口可行，但奖励必须对应闭环目标；如果
  residual 只被 path proxy 奖励，terminal capture 仍可能失败。

### Unthrottling the Tanh Jacobian in SAC，arXiv `2609.09478`

**Unthrottling the Tanh Jacobian in SAC: A Negative Result on Bang-Bang Control and MetaDrive**。

- 直接检验 `tanh` 的 `da/du = 1-a^2` 在边界处消失是否是一个需要修复的 bug。
- 在最优解为 bang-bang 的双积分器上，原始 SAC 已接近最优；额外恢复 Jacobian
  信号的无门控 bypass 让策略 `99%` 的评估步饱和到 `|a|>=0.9`，return 从约
  `-31.6` 崩到 `-195.5`；只在平坦肩部触发的 gated bypass 也失败。
- 作者结论：Jacobian 效应是真实的，但“把饱和当 bug 修掉”并不免费，也不一定
  有帮助；**到达边界不等于解决了最优解在边界上的问题**。
- 对本项目的含义：softsign 没有解决饱和并不意外。需要先证明“终端任务的最优
  修正确实在边界上”，或者证明 reward/terminal 信号在把策略往错误方向推；
  仅继续换 squash 函数或加 Jacobian 补偿，不是有证据的下一步。

## 3. 对 Goal59 现象的解释

### 3.1 为什么 softsign 仍然饱和

softsign 只改变边界附近的形状和梯度，不改变 actor 可以把输出推到高幅值这一事实。
Goal59 的 output scale 是 `0.5`，但 gate-active residual L2 仍有 `10.5%–22.5%`，
8/8 raw saturation 为 `1.0`。这说明策略不是“没有输出”，而是持续选择接近最大
修正；问题在“为什么 critic/奖励认为最大修正是好的”，不在“边界函数叫什么”。

### 3.2 为什么不是 NoisyNet 主导

正式评估使用 mean weights。8/8 seed 的 deterministic saturation 和 raw
saturation 都是 `1.0`；`noise/det` 中位数只有 `0.0193`，只有 seed `1008`
达到 `1.03`，但它的 mean 仍 100% 饱和，闭环 success/plane-miss 也没有改善。
因此 NoisyNet 不是全局主因。它可能让训练期 residual 更抖，但不能解释确定性
策略的边界饱和。

### 3.3 残差非零但闭环不动，说明方向不对

Goal59 的 residual 不是太小，而是持续接近边界；同时 terminal capture 没有改善，
success/plane-miss 在多数 seed 上恶化。path-zero 对 752/880 的 capture、plane
和 RMSE 几乎没有影响，按协议只能记为 `INVALID_ATTRIBUTION`，不能声称 12D
endpoint context 起了作用。

这与“residual 应该做局部闭环修正”的文献假设相反：当前 residual 更像一个大的
开环偏置，既没有被终端目标约束，也没有被 path 输入区分。下一步应检查 reward
的 dense path 项、goal-plane miss、terminal progress 是否与真实 terminal
capture 同向，以及终止平面/真实状态捕获/协议 success 的口径是否一致。

## 4. 为什么暂不换算法

把失败分支拆开看：

| 失败形态 | 更匹配的算法分支 | Goal59 证据 |
|---|---|---|
| residual 很小、训练不前进、探索不足 | SAC / 熵探索 | 不支持：residual 非零且饱和 |
| 数据利用不足、同样样本下学得慢 | REDQ / DroQ / 高 UTD | 不支持：5M × 8 seeds 仍方向错误 |
| residual 饱和、方向与 terminal capture 不一致 | 动作参数化 / reward / terminal capture | 当前证据指向这里 |
| 闭环完全不动 | 先查动作参数化、reward、terminal capture | path-zero 无归因，支持先查接口 |

因此当前不应启动 SAC/REDQ/DroQ。换算法不能替代“动作接口是否允许局部修正”、
“奖励是否奖励了错误行为”和“终端捕获口径是否对齐”的检查。

## 5. 下一 Goal 的可证伪诊断

建议先做只读、预注册的诊断，不训练新策略：

1. **动作参数化 H-A**：如果失败由 residual 幅值/边界导致，那么在冻结 checkpoint
   上做只读的 scale/rate 分解，应看到饱和维度与 terminal capture 误差同向；
   如果饱和维度与误差无关，则参数化不是主因。
2. **Reward H-B**：如果 reward-v2 的 dense path 项与 terminal capture 方向冲突，
   那么 trace 中应出现“path progress 为正、terminal capture 为负”的系统性片段；
   若两者同向，则不能把失败归因于 reward。
3. **Terminal capture H-C**：如果终止平面、真实状态 capture 和协议 success 口径
   不一致，那么同一终止点的 true-state capture 应与 protocol success 系统性分叉；
   若完全一致，则终端口径不是主因。
4. **优化/探索 H-D**：如果 SAC 分支成立，应看到 residual 不饱和、训练曲线离开
   零附近但 capture 不动；Goal59 的饱和形态不支持这个预测。
5. **样本效率 H-E**：如果 REDQ/DroQ 分支成立，应看到增加更新/提高 UTD 后指标
   持续移动；Goal59 的失败是方向错误，不是更新不足。

## 6. 决策规则

- 只有 H-D 被独立诊断支持，才另开预注册 Goal 做 SAC 同协议对照；
- 只有 H-E 被独立诊断支持，才考虑 REDQ/DroQ；
- 在 H-A/H-B/H-C 未分清前，不启动新算法、不扫 reward/gamma/seed/scale；
- 不打开 held-out `100–149`；
- 不把 `INVALID_ATTRIBUTION` 写成 PASS，也不把一次诊断写成正式 Gate。

## 7. 引用核验

| 文献 | 核验方式 | 状态 |
|---|---|---|
| DDPG, arXiv 1509.02971 | arXiv API 标题/日期/摘要 | verified |
| TD3, arXiv 1802.09477 | arXiv API 标题/日期/摘要 | verified |
| SAC, arXiv 1801.01290 | arXiv API 标题/日期/摘要 | verified |
| NoisyNet, arXiv 1706.10295 | arXiv API 标题/日期/摘要 | verified |
| REDQ, arXiv 2101.05982 | arXiv API 标题/日期/摘要 | verified |
| DroQ, arXiv 2110.02034 | arXiv API 标题/日期/摘要 | verified |
| Residual Policy Learning, arXiv 1812.06298 | arXiv API 标题/日期/摘要 | verified |
| Residual Off-Policy RL, arXiv 2509.19301 | arXiv API 标题/日期/摘要 | verified |
| RLPP, arXiv 2501.17311 | arXiv API 标题/日期/摘要 | verified |
| Unthrottling the Tanh Jacobian, arXiv 2609.09478 | arXiv API 标题/日期/摘要 | verified |
