# Reading note：课程起点分布、采样优先权与 PPO 信用时域

**日期**：2026-11-11<br>
**触发**：Goal138 的 100% near-terminal 课程确实提高样本密度，却在 SAC 上造成
nominal strict 交易并让 PPO 继续近乎零残差。下一轮需要判断：应当恢复 full-path
anchor（混合起点），还是先改 PPO 的 advantage 信用时域（gamma/GAE）。

## 检索

使用 `paper-search`（arXiv；Semantic Scholar 本次 HTTP 429，未采用其结果）检索：

```text
curriculum learning reset distribution reinforcement learning long horizon credit assignment
reverse curriculum generation reinforcement learning
prioritized level replay
generalized advantage estimation
```

本笔记只依据 arXiv 标题与摘要建立机制假设，不把未读全文的细节当成实验证据。

## 1. Reverse Curriculum：从目标附近逐步退远

Florensa et al., *Reverse Curriculum Generation for Reinforcement Learning*,
arXiv:1707.05300, 2017. <https://arxiv.org/abs/1707.05300>

摘要把 sparse-reward goal task 的困难归因于两点：从初始状态很难到达目标，且
成功信号很稀疏。方法不是把训练永久限制在目标附近，而是从目标附近的一批起点开始，
随策略变好逐步生成更远的起点，形成随性能调整的 start-state curriculum。

与 Goal138 的对应关系：

- 近终端起点确实能让 replay/rollout 更频繁看到 goal transition；
- 但 reverse curriculum 的关键是“逐渐向完整任务退远”，不是把训练分布永久替换成
  目标附近；
- 因此 Goal138 的 T15/T10 是密度诊断，不足以直接证明“最终训练分布应该只用
  near-terminal”。Goal139 的 mixed-horizon 是对这一点的最小修正。

不能直接迁移的部分：论文是 sparse reward 的 goal-reaching；我们的 reward、terminal
和成功半径已冻结，不能借课程之名改 reward。

## 2. Prioritized Level Replay：采样应依赖当前策略的学习潜力

Jiang et al., *Prioritized Level Replay*, arXiv:2010.03934, 2020.
<https://arxiv.org/abs/2010.03934>

摘要指出，固定 level 上的均匀采样忽略了“当前策略还能从该 level 学到什么”。
PLR 用 TD-error 估计 level 被再次访问时的学习潜力，并据此优先采样，形成动态、
渐难的课程；在 Procgen 上提高样本效率和 OOD 泛化。

与 Goal138 的对应关系：

- 近终端样本密度是静态分布操纵；Goal138 已证明它改变训练行为，但不是稳定收益；
- 更一般的下一步是让 full-path / near-terminal 的采样比例依赖策略表现，而不是
  只选 50/50 或 75/25 两个固定比例；
- 但当前样本量只有 3 seeds × 1M，先验证固定混合是否有信号，再考虑 PLR 式动态
  采样，避免把自适应采样器和课程比例同时引入而无法归因。

不能直接迁移的部分：当前 env 没有 per-level TD-error 回传接口；PLR 需要额外
工程和预注册，不能在本轮临时加入。

## 3. GAE：gamma/lambda 直接控制偏差—方差与有效信用长度

Schulman et al., *High-Dimensional Continuous Control Using Generalized Advantage
Estimation*, arXiv:1506.02438, 2015. <https://arxiv.org/abs/1506.02438>

摘要把 GAE 描述为用指数加权 advantage estimator 在 policy-gradient 方差与偏差之间
折中，并用 trust-region 优化稳定更新。gamma/lambda 越小，有效信用时域越短；越大，
越强调远端回报但方差更高。

与 Goal138/PPO 的对应关系：

- Goal138 让 PPO 在近终端获得更多 sample，但 PPO 的输出仍接近零残差；这说明
  “看不见 terminal transition”不是唯一解释；
- 若 PPO 的 gate-active residual 仍 `<1%`，下一项最小单因素实验应是固定 BC、
  固定 action、固定评估，只比较 gamma/GAE 信用时域；
- gamma 改变会影响 per-decision 转换，必须沿用仓库已有的
  `gamma_per_decision` 语义并写入 manifest，不能只改 config 数字而忽略 decision
  stride。

不能直接迁移的部分：GAE 论文没有 AUV terminal capture 或 frozen-BC residual 的
设定；它只支持“信用时域是独立可检验变量”，不保证某个 gamma 一定有效。

## 4. 对 Goal139 的约束

1. SAC 用 `M50`（50% full + 50% T15）和 `M75`（25% full + 75% T15）检验
   full-path anchor 是否恢复 nominal strict，不继续扫起点。
2. PPO 保持 near-terminal 训练分布与输入动作不变，只比较 gamma/GAE
   `0.995 / 0.98 / 0.95`；与 SAC 分开训练、分开裁决。
3. 先看 nominal/default 与 current15/current20 的 paired 结果；只有跨 seed 信号
   才扩 seeds 或 5M formal。
4. 不打开 held-out，不改 reward/terminal/success/goal radius。

## 5. Goal139 实测回填（2026-11-11）

Goal139 已完成上述最小对照，结论是：

- 固定 full-path/T15 混合没有形成跨 seed 收益：M50 只有 86203 一个 seed 在 3 个
  level 正向，nominal strict 整体 `-15`；M75 更高 near-terminal 比例反而更差，
  combined20 出现 6 个 timeout。
- PPO gamma/GAE `0.995/0.98/0.95` 三档 success 几乎完全相同，gate-active
  residual 仍约 `1%`；缩短信用时域没有单独救出 PPO。
- 256 reset census 证明 mixed-start 实现正确（M50 `50.39%`、M75 `27.34%`
  full-path），因此 null 不是 fraction 接线错误。
- 这与 Reverse Curriculum 的机制一致：从目标附近开始只解决“成功样本稀疏”，
  并不能替代“逐步退远”的 stage schedule；固定 50/50 或 25/75 不是完整课程。
- PLR 的启发仍然成立：下一步应让 stage 晋级取决于当前策略表现，而不是继续扫
  静态混合比例；但在实现 stage sampler 前，先补齐深度/路径/噪声/洋流的
  C-line 阶梯和 promotion/rollback，避免一次引入过多变量。

因此 Goal139 的 null 支持转入 Goal136 C-line，而不是换 REDQ/DroQ。
