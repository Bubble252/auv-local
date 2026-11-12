# Reading note：residual RL 的信用分配、零初始化与 terminal capture

日期：2026-11-10<br>
关联：Goal87 `PPO_DIAG / NOT_AGATE`、`TERMINAL_CAPTURE_MILLIMETER_ONLY`<br>
检索：`paper-search`，2020–2026，三组 query：

```text
residual policy learning frozen base controller reinforcement learning
residual reinforcement learning zero initialization credit assignment PPO
AUV terminal capture reward shaping goal proximity success criteria
```

Semantic Scholar 在三次检索中均返回 HTTP 429；以下结论主要来自 arXiv、
OpenReview、Crossref 和既有 Goal80 reading note。检索结果只用于提出可证伪的
下一步实验，不改变当前冻结 Gate。

## 1. 检索到的相关工作

### 1.1 Residual policy / frozen base controller

- **Dodeja, Schmeckpeper, Vats et al., 2025** — *Accelerating Residual
  Reinforcement Learning with Uncertainty Estimation*，OOD Workshop @ RSS 2025。
  URL: https://openreview.net/forum?id=HUSLmVdg5K
  - 关注 residual RL 的样本效率与不确定性估计；说明 residual 分支的更新信号
    可能很弱，需要额外机制才能加速。
  - 与当前问题的对应：冻结 BC 基座后，残差头的确定性均值只有 base L2 的
    `0.14%–0.57%`，不是“残差太大”，而是“残差几乎没被更新”。

- **Abbas, Chasparis, Kelleher** — *Specialized Deep Residual Policy Safe
  Reinforcement Learning-Based Controller for Complex and Continuous
  State-Action Spaces*。
  DOI: https://doi.org/10.21203/rs.3.rs-3918353/v1
  - 将 residual policy 作为安全基座控制器之上的修正层；强调 residual 的
    作用域和基座保真度。
  - 与当前问题的对应：当前 `frozen_pc_bc_residual` 的基座保真度已经验证，
    问题在 residual 分支没有产生有效修正，而不是基座被破坏。

### 1.2 Credit assignment / temporal credit assignment

- **Arumugam, Henderson, Bacon, 2021** — *An Information-Theoretic Perspective
  on Credit Assignment in Reinforcement Learning*，arXiv:2103.06224。
  URL: http://arxiv.org/abs/2103.06224v1
  - 把信用分配表述为“动作对回报的信息贡献”；当动作对回报的影响很小、
    延迟很长时，梯度信号会退化。
  - 与当前问题的对应：terminal 只在最后一步激活，residual 的毫米级动作
    改变不足以改变 terminal 类别；这会让 advantage 接近噪声。

- **de Vries, Moerland, Plaat, 2022** — *On Credit Assignment in Hierarchical
  Reinforcement Learning*，OpenReview。
  URL: https://openreview.net/forum?id=Mv5EA6Ki9w
  - 分层/多时间尺度信用分配的失败模式；高层动作与低层执行之间的时间错位
    会让学习信号变弱。
  - 与当前问题的对应：当前 residual 是每个控制步的动作，但 terminal 回报
    只在 episode 末尾出现；如果 critic 不能把 terminal 信号传回 pre-terminal
    窗口，actor 的末层梯度会一直很小。

- **Arumugam, Griffiths, 2025** — *On Temporal Credit Assignment and
  Data-Efficient Reinforcement Learning*，RLC 2025 Finding the Frame。
  URL: https://openreview.net/forum?id=ek0ZniXLac
  - 讨论数据效率与时间信用分配的关系；不能把“样本不够”与“信用分配失败”
    混为一谈。
  - 与当前问题的对应：当前 PPO 在同一 1M 预算下增加 iteration 仍不移动，
    更像信用分配/更新幅度问题，而不是单纯样本效率不足；因此暂不直接开
    REDQ/DroQ。

### 1.3 Sparse reward / goal reaching / terminal capture

- **Vasan, Wang, Shahriar et al., 2024** — *Revisiting Sparse Rewards for
  Goal-Reaching Reinforcement Learning*，arXiv:2407.00324。
  URL: http://arxiv.org/abs/2407.00324v2
  - 重新审视 goal-reaching 中的稀疏奖励；稀疏 terminal 奖励需要合适的
    episode 结构或辅助信号，否则学习会停在基座附近。
  - 与当前问题的对应：当前 success 是“刚好越过 2.0/1.25 m 球壳”，
    30 个成功里 14 个余量 <1 cm；terminal 信号非常边界化，毫米级残差
    很难稳定翻转类别。

- **Park, Yang, Lee et al., 2025** — *From Sparse to Dense: Toddler-inspired
  Reward Transition in Goal-Oriented Reinforcement Learning*，arXiv:2501.17842。
  URL: http://arxiv.org/abs/2501.17842v1
  - 从 dense 到 sparse 的奖励过渡，而不是一开始只用稀疏 terminal。
  - 与当前问题的对应：可作为未来 reward curriculum 的候选，但当前审计
    还没有证明 reward 是主因；不先改 reward。

- **Zu, Yang, Liu et al., 2021** — *A Multi-Dimensional Goal Aircraft Guidance
  Approach Based on Reinforcement Learning with a Reward Shaping Algorithm*，
  Sensors 21(16):5643。
  DOI: https://doi.org/10.3390/s21165643
  - 多维度目标制导 + reward shaping；说明 terminal/goal 项需要与制导误差
    联合设计，而不是单独加一个稀疏项。
  - 与当前问题的对应：当前 reward 的 terminal 项只在 done 行激活，且
    command tracking 仍主导；但审计显示 reward 与 terminal 同步不变，
    所以不能把“加 reward 项”当作第一反应。

### 1.4 既有 AUV terminal capture 结论

- 见 `research/paper_reading/2026-11-10-goal80-terminal-reference-plane-miss.md`：
  AUV terminal homing 通常需要独立的 terminal sensor / capture 逻辑；仅沿
  tangent 前移路径参考点不等于 terminal docking guidance。
- 本轮补充：Goal85 已经把 termination 切到 `eta`，p_nav/eta 的终止帧差异
  在 trace 中为 0；因此当前不是旧的 capture-frame 问题，而是 residual 没有
  产生足够的 terminal 位移。

## 2. 对当前 blocker 的映射

| 观察 | 文献侧解释 | 当前判断 |
|---|---|---|
| 确定性 residual 只有 base L2 的 0.14%–0.57% | residual RL 需要足够强的 residual 信号；冻结基座会让 residual 分支梯度变小 | actor 更新幅度/信用分配是主嫌疑 |
| 40 iter 后末层范数仍 0.018–0.027，KL 1e-4 | 时间信用分配退化时，末端梯度接近噪声 | 不是单纯 iteration 少 |
| BC head 非零先验导致 KL 0.09、clip 95%，40 iter 后 success 掉到 0.475 | 非零初始化 + 多 mini-batch 累计步长会破坏先验 | 需要检查 old-log-prob/更新预算，不是网络容量 |
| success 余量 0.8–31 mm，scale1 只改毫米 | 稀疏 terminal 对边界极敏感 | terminal 契约没有坏，但成功计数对微小残差极敏感 |
| reward 只改善千分之几，terminal 计数不变 | terminal 与 reward 同步不变 | 不支持“reward shape 是主因” |

## 3. 最便宜的下一步 falsification

1. **H87-O1**：frozen BC + 1M，单 seed，`num_mini_batches=10/25` 或
   actor lr `1e-4`；看末层范数、KL、确定性 residual/base L2 和 terminal
   桶计数是否同时移动。
2. **H87-O3**：只读审计 BC-head 的 warmup 前后 old/new log prob；确认
   KL 0.09 是否来自 stale old-log-prob，而不是 optimizer 累计步长。
3. 只有 H87-O1/O2 都否定、且 H87-O3 证明不是实现问题，才写 Goal88
   预注册 SAC 同协议对照；不直接换 REDQ/DroQ。

## 4. 不允许的直接迁移

- 不因为文献使用 residual RL 就自动加 uncertainty ensemble、reward
  curriculum 或 termination 改动；
- 不在当前 Gate 内改 success、goal radius、reward、terminal、action
  参数化、seed 或 held-out；
- 不把“文献说稀疏奖励难”当作“必须改 reward”的证据；
- 不在没有 H87-O1/O2/O3 结果前启动 SAC/REDQ/DroQ。

## 5. 检索记录

- residual policy / frozen base：arXiv=6、OpenReview=6、Crossref=6，
  16 unique（2 duplicates merged）；
- zero initialization / credit assignment：arXiv=6、OpenReview=6，
  12 unique；Semantic Scholar HTTP 429；
- AUV terminal capture / reward shaping：arXiv=6、Crossref=6，
  12 unique；Semantic Scholar HTTP 429。
