# Goal117 reading note：terminal credit、goal-conditioned 样本效率与 AUV docking

**日期**：2026-11-11<br>
**用途**：Goal117 `MIXED_TERMINAL_MECHANISM` 的 autoresearch 文献对照；只用于解释
“2 m entry 已 geometry-bad + terminal reward share 为负”这一现象，不改变已冻结的
success/reward/terminal 定义。

## 1. Arumugam, Henderson & Bacon (2021)：credit assignment 的难点是信息稀疏

- 标题：*An Information-Theoretic Perspective on Credit Assignment in Reinforcement Learning*
- arXiv：https://arxiv.org/abs/2103.06224
- 本地读取：`/tmp/goal117_papers/2103.06224.pdf`（不提交）

核心观点：

- 经典直觉把 credit assignment 难归因于 reward sparse；作者认为真正的问题是
  **information sparsity**。
- 对任意稀疏 reward 加一个常数 `c` 可以让每步都有非零 reward，但最优策略和
  credit assignment 难度都不变。
- 因此“dense reward 更多/更大”不等于“terminal credit 更好”；关键是 dense 项
  是否携带 terminal 成功的信息。

对 Goal117 的含义：

- failure 的 `dense_terminal_ratio_last2s = 3.88`、success 只有 `0.68`，
  且 failure 的 `terminal_share_last2s = -0.20`。这不是“reward 总量太小”，
  而是末段 dense 项在数值上盖过了 terminal 信息。
- 不能只通过放大 `goal_bonus` 或再加一个 dense 项来救；必须检查 dense 项是否
  与 terminal 成功同向，或改成 potential-based / 退火形式。

## 2. Park et al. (2025)：sparse-to-dense reward transition

- 标题：*From Sparse to Dense: Toddler-inspired Reward Transition in Goal-Oriented Reinforcement Learning*
- arXiv：https://arxiv.org/abs/2501.17842
- 本地读取：`/tmp/goal117_papers/2501.17842.pdf`（不提交）

核心观点：

- 从 sparse 到 potential-based dense（S2D）的过渡可以保留最优策略；
- 关键是 dense shaping 必须是 potential-based，或随训练阶段退火，否则会引入
  有偏的最优策略；
- 该文在机器人臂和 egocentric 3D navigation 上验证了 transition 的收益。

对 Goal117 的含义：

- 当前 reward 的 dense path/command 项不是 potential-based，failure 末段
  dense 项占主导，和“dense 项把策略拉向路径好看而不是 terminal capture”
  的观察一致。
- 如果后续要改 reward，优先预注册：
  1. 只做 potential-based terminal-progress shaping；
  2. 或按距离/阶段退火 dense 项；
  3. 不直接放大 terminal bonus。

## 3. Liu et al. (2022)：goal-conditioned 样本效率的网络结构

- 标题：*Metric Residual Networks for Sample Efficient Goal-Conditioned Reinforcement Learning*
- arXiv：https://arxiv.org/abs/2208.08133
- 代码：https://github.com/Cranial-XIX/metric-residual-network
- 本地读取：`/tmp/goal117_papers/2208.08133.pdf`（不提交）

核心观点：

- GCRL 的 `Q*(s,a,g)` 应满足特定三角不等式；
- Metric Residual Network（MRN）把 Q 分解为“metric 项 + 非对称 residual 项”，
  在 12 个 GCRL benchmark 上比 monolithic 网络更 sample-efficient；
- 这说明 goal-conditioned 任务的样本效率问题可以通过**结构化网络**改善，
  而不是只靠换 on-policy/off-policy 算法。

对 Goal117 的含义：

- 当前 PPO 的 actor 没有显式的 goal-distance metric 结构；frozen BC residual
  也没有把 `goal_distance` 作为对称 metric 约束。
- 但 Goal117 的证据还没有到“必须改网络”的程度：2 m entry 已经 geometry-bad，
  先查 approach/terminal geometry 和 reward credit 更便宜。
- 如果后续 Goal118 证明“2 m 内可达、方向正确、但学不动”，再预注册 MRN 风格的
  goal-conditioned critic/actor 对照，而不是 NoisyNet 或盲目换 SAC。

## 4. Patil, Wehbe & Valdenegro-Toro (2021)：AUV docking 的 DRL benchmark

- 标题：*Deep Reinforcement Learning for Continuous Docking Control of Autonomous Underwater Vehicles: A Benchmarking Study*
- arXiv：https://arxiv.org/abs/2108.02665
- 本地读取：`/tmp/goal117_papers/2108.02665.pdf`（不提交）

核心观点：

- AUV docking 的 reward 需要同时包含距离、姿态、速度和 terminal 项；
- 该文比较 PPO/TD3/SAC，TD3 在其 reward 下最稳定，说明“算法族”不是唯一变量，
  reward 设计和 terminal 定义同样关键；
- docking 的成功高度依赖末段姿态和接触前几何，不能只看最终距离。

对 Goal117 的含义：

- 我们的 failure 在 2 m entry 时 `vertical=1.34 m`、`pitch=0.52 rad`，
  success 只有 `0.41 m`、`0.20 rad`；这与 docking 文献强调的“末段姿态先决”
  一致。
- 当前最值得做的不是换 SAC，而是把 approach-phase 的 vertical/pitch 收敛
  和 terminal reward credit 分开预注册。

## 5. 综合判读

四条文献线索一致指向：

```text
terminal credit 的关键不是 reward 总量，而是 terminal 信息是否被 dense 项淹没；
terminal capture 的关键不是最后 0.2 m 的 residual 幅度，而是 2 m 前的 geometry；
goal-conditioned 样本效率可以通过结构化网络改善，但应在 geometry/reward 排除后。
```

因此 Goal117 之后的下一步应是：

1. 先做 2 m entry 的 reachability / approach-phase 归因（最便宜）；
2. 若 2 m 内可达但 terminal reward 仍被 dense 项压住，预注册 potential-based
   terminal shaping 或 dense 退火；
3. 只有以上都排除后，才考虑 MRN 风格网络或 SAC 同协议对照。
