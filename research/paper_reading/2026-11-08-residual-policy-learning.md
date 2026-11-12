# Residual Policy Learning reading note

日期：2026-11-08<br>
对应 blocker：`research/blockers/2026-11-08-goal26-residual-path-reward.md`

## 1. Residual Policy Learning

- 论文：Tom Silver, Kelsey Allen, Josh Tenenbaum, Leslie Kaelbling,
  *Residual Policy Learning*, 2018。
- arXiv：<https://arxiv.org/abs/1812.06298>
- 项目页/代码：<https://k-r-allen.github.io/residual-policy-learning/>

方法要点：

- 不直接从零学习控制，而是学习一个叠加在已有 controller 上的 residual action；
- 适合“base controller 已经不错，但存在模型误差、标定误差或局部不完美”的场景；
- 论文报告 residual 能稳定改善 hand-designed policy 和 MPC，但前提是 base policy
  本身已能完成大部分任务，且 reward 能区分改进方向。

与 Goal26 的对应：

- 我们的 frozen D3 BC 已经满足“good but imperfect”前提，所以 residual 形式本身没有
  用错；
- 但 P1/P2 只把 pre-goal RMSE 改善约 `0.014–0.017 m`，success 没有提升；
- 这说明“有 residual 接口”不等于“residual 能形成有用的长时路径修正”。问题更可能在
  residual 的表达方式、信用分配或奖励冲突，而不是简单地“再加一点路径奖励”。

可复现实验：

- 保持 frozen D3 不变，比较 `action residual` 与 `z_t residual`；
- 两者都用同一 40-seed dev 矩阵、同一 success-first 门禁；
- 先做 1-seed screen，超过 `0.10 m` pre-goal 改善才扩到 40 seeds。

不能直接迁移的部分：

- 论文主要是 MuJoCo manipulation，动作频率、动力学和终止条件与 50 Hz REMUS100 不同；
- 论文主要看任务 return，本项目还必须保留 success、plane miss、timeout、safety 和
  actuator 约束；
- 不能因为 RPL 在原论文有效，就跳过本项目的 zero-residual 和 success 护栏。

## 2. Beyond Action Residuals / ZPRL

- 论文：Dongjie Yu et al., *Beyond Action Residuals: Real-World Robot Policy Steering
  via Bottleneck Latent Reinforcement Learning*, 2026。
- arXiv：<https://arxiv.org/abs/2605.19919>

方法要点：

- 论文明确指出，直接对 action 做 residual 往往会产生 noisy、poorly structured
  exploration；
- 它把预训练策略的 observation embedding 压到紧凑的 task-relevant latent，再在这个
  latent 上学 residual perturbation，由 decoder 条件化生成动作；
- 在真实机器人任务上，latent residual 比 action residual 更平滑，success 也更高。

与 Goal26 的对应：

- P1/P2 的 residual/base L2 ratio 只有 `1.8%–2.7%`，且 P2 增大路径奖励后 residual
  反而更小；
- 这与“action residual 的探索方向不够结构化，策略选择收缩 residual 来保住 success”
  的现象一致；
- 我们的 PC 架构已经有 64D `z_t`，因此下一步比“改动作头”更自然的是：
  冻结 actor 的其余部分，只学习 `z_t` 上的 residual，再让原 decoder 生成动作。

可复现实验：

- 不改 observation 和 base BC，只加一个零初始化的 `delta_z` 分支；
- 仍使用 action-level 的 `residual_scale` 作为安全上限；
- 先做 1-seed screen，再决定是否扩 40-seed dev。

不能直接迁移的部分：

- 原文是 manipulation + flow-matching policy，我们这里是 50 Hz 欠驱动水下航行器；
- `z_t` residual 会改变 actor 内部表征，必须重新验证 zero-residual、动作平滑性和
  actuator saturation；
- 不能把 latent residual 当成免检方案；如果 base `z_t` 本身没有路径相关信息，latent
  residual 仍可能学不到有效方向。

## 3. 对本项目的结论

文献不支持“继续把同一个 action residual 的路径奖励从 0.30 再加到更高”作为首选。
更合理的下一步是：

1. 先做一个便宜的 `z_t` residual 方案比较；
2. 同时检查 128-step / 2.56 s 的 credit horizon 是否太短；
3. 若两者都没有改善，再考虑 action chunking 或降低控制频率。

这三条都比继续扫 `w_path_3d` 更有信息量。
