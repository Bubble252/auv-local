# Reading note：从 command residual 转向低维 controller-parameter residual

**日期**：2026-11-12<br>
**触发**：Goal161 Phase B Gate FAIL；command residual 在 current15 有小幅局部增益，
但 current10 反向，且 trace 显示它主要学成 near-goal 的距离 schedule。<br>
**决策用途**：判断是否值得预注册 Goal162 的 heading/pitch gain residual，而不是
继续扫 command scale 或立刻换 SAC/REDQ/DroQ。

## 1. 检索范围

使用 `paper-search`，查询主题：

- reinforcement learning residual control classical controller gain scheduling；
- reinforcement learning PID gain tuning autonomous underwater vehicle；
- AUV path following terminal guidance reinforcement learning；
- residual reinforcement learning controller parameter adaptation。

覆盖 2019–2026；arXiv/OpenAlex/OpenReview/Crossref 有结果，DBLP 和 Semantic Scholar
本次分别出现解析错误/HTTP 429，未作为唯一证据源。下面只记录能由检索元数据或摘要
直接支持的内容。

## 2. 最相关的五篇工作

### 2.1 Song et al. 2022：SAC-Auto 调 PID 参数，而不是直接出舵角

**Paper**：*Research on PID Parameter Tuning and Optimization Based on SAC-Auto for
USV Path Following*, JMSE, DOI `10.3390/jmse10121847`。

要点：LOS 负责 guidance，PID 负责 heading；SAC-auto 根据时变需求自动调整 PID
参数，尤其是在波浪干扰下。状态、动作和 reward 都围绕“参数调整”设计，而不是让
网络直接替代底层控制器。

对本项目的含义：当前 command residual 的动作是 heading/pitch/speed 的加性偏置；
SAC-Auto 提示一个更匹配的证据链——让 RL 输出低维 controller gains，把 H3/ALOS
的稳定结构保留为基座。我们的失败不是“网络不够大”，而是动作空间可能不在正确的
参数流形上。

### 2.2 Chaffre et al. 2021：在极点空间重参数化 PID，再用 SAC 探索

**Paper**：*Direct Adaptive Pole-Placement Controller using Deep Reinforcement
Learning: Application to AUV Control*, IFAC-PapersOnLine, DOI
`10.1016/j.ifacol.2021.10.113`。

要点：AUV 在未知扰动下不存在一套对所有工况都最优的 PD/PID 参数；作者把 PID
增益重参数化到保证稳定的极点空间，再用 SAC 在该空间中适应工况。重点不是无约束
地学增益，而是**把稳定性/可行域编码进动作参数化**。

对本项目的含义：如果 Goal162 做 gain residual，不能直接让 actor 输出无界
heading/pitch gain。应使用 `tanh` 映射到围绕 H3 的硬边界，或采用极点/阻尼比
参数化；至少保证 nominal 不因为增益突跳而失稳。

### 2.3 Trumpp et al. 2023：residual policy 保留 classical controller 的稳定行为

**Paper**：*Residual Policy Learning for Vehicle Control of Autonomous Racing
Cars*, IEEE IV 2023, DOI `10.1109/iv55152.2023.10186744`。

要点：residual controller 与 classical controller 并联，学习修正 classical
path-following 的不足；在 12 条复现赛道上平均降低 4.55% lap time，并能迁移到未知
赛道。论文强调 residual 的高适应性必须与 classical controller 的稳定行为并行。

对本项目的含义：residual 本身不是错，错在我们把 residual 放在“指令偏置”空间，
让它在 near-goal 以接近 cap 的固定偏置改变基控制器输出。更合理的 residual 应是
低维、有限幅、可解释的 controller-parameter 修正。

### 2.4 Ma et al. 2023：速度环交给 PI，RL 控制舵/升降舵

**Paper**：*Neural Network Model-Based Reinforcement Learning Control for AUV 3-D
Path Following*, IEEE T-IV, DOI `10.1109/tiv.2023.3282681`。

要点：ModelPPO 用传统 PI 维持所需航速，RL 控制 rudder 和 elevator fins 完成三维
路径跟踪；论文的分解避免了让 RL 同时重学速度环和姿态环。

对本项目的含义：Goal161 的 C 臂仍允许 heading/pitch/speed 三通道 command
residual。下一步应优先固定速度环，只学习 heading/pitch 的 controller parameter，
不要继续把 speed channel 放进高方差策略。

### 2.5 Song et al. 2022：低维控制器参数自适应也可以是 RL 动作

**Paper**：*Reinforcement Learning based Parameter Optimization of Active
Disturbance Rejection Control for Autonomous Underwater Vehicle*, DOI
`10.23919/jsee.2022.000017`。

要点：用简化 Q-learning 调整 LADRC 参数，实现六自由度 AUV 运动解耦和参数自适应；
对比固定参数 LADRC，验证了参数自适应的有效性。它不是深度 SAC，但说明“RL 输出
控制器参数”在 AUV 上是可解释、低维的动作设计。

对本项目的含义：我们不需要一上来让网络输出完整的 3 通道指令；3–4 维的
heading/pitch gain residual 更符合文献中的可迁移组件，也更适合 1M screen。

## 3. 与现有证据的连接

Goal157 已经给出传统控制器的静态增益证据：

| controller | nominal | current07 | current10 | current15 |
|---|---:|---:|---:|---:|
| H2 | 40/40 | 36/40 | 27/40 | 14/40 |
| H3 | 40/40 | 33/40 | 31/40 | 17/40 |
| H4 | 39/40 | 37/40 | 31/40 | 15/40 |

单看 current10/current15，H3/H4 都优于 H2；但 H3 在 current07 掉 3 个，
H4 在 nominal 掉 1 个。也就是说，**固定静态增益无法同时覆盖 nominal/current07
与 current10/current15**，而这不等于“RL 没样本效率”；它更像一个
state-dependent gain scheduling 问题。

Goal161 的 C 臂在 current15 default `+8/120`，但 current10 `-4/120`；trace
中 near-goal command residual 与 goal distance 的相关性很高，与 signed error
的相关性弱。因此下一个可证伪假设不是“更大的 command residual”，而是：

> H162-A：以 H3 为 base，只学习 bounded heading/pitch gain residual，可以根据
> current/估计误差在 nominal/current07 与 current10/current15 之间切换增益，
> 从而避免静态 H3 的 current07 损失，并在 current10/current15 保留增益。

## 4. 最小可迁移组件

1. **固定 H3 base**：`heading_gain=3.0`、`pitch_gain=4.0`、rate gains 不变；
   速度环和 ALOS guidance 不改。
2. **低维动作**：先只输出 `Δheading_gain`、`Δpitch_gain`；第二轮才考虑
   `Δheading_rate_gain`/`Δpitch_rate_gain`。
3. **硬边界/稳定参数化**：围绕 H3 使用 `tanh` 映射，例如 heading gain
   `3.0 ± 1.0`、pitch gain `4.0 ± 1.5`；任何越界都在环境层 clamp，不能只靠 reward。
4. **近端 gate**：先复用 `10→2 m` 的 terminal gate，远段保持 H3；如果近端 gate
   仍无收益，再单独预注册全程 gain residual，不能在同一 Gate 内静默切换。
5. **观测复用 `z_t`**：actor 输入保持现有 state/path/current/endpoint 编码，
   不引入新的真值观测，不改 success/termination/radius。
6. **对照**：H3 zero、固定增益 H2/H3/H4、SAC gain residual；同一 dev40、
   current10/current15 primary，nominal/current07 guard。

## 5. 明确不迁移的部分

- 不直接把完整 PID 参数交给网络；维度太高，且现有 1M screen 无法可靠区分。
- 不用 command residual 继续扫 scale；Goal160/161 已覆盖。
- 不用 REDQ/DroQ；目前没有“样本效率不足”的证据。
- 不把 reward 改动和 gain residual 混在同一个 Gate；先证明 action parameterization
  有方向，再单独预注册 reward/terminal 方案。
- 不打开 held-out，不修改 Goal161 已记录的 Gate。

## 6. 下一步

等 Goal161 Phase C mask 完成：

- 若 channel mask 在 current10/current15 稳定超过 zero，则优先在该通道上做一个小型
  gain residual screen；
- 若 mask 仍无正信号，则写 Goal162 预注册：H3 + bounded heading/pitch gain
  residual，SAC 1M screen（3 seeds），H3 zero 为唯一有效 baseline；PPO 只在
  SAC 机制成立后按同协议复制；
- 首个最便宜证伪是 1 seed × 1M，只比较 H3 zero 与 gain residual；方向不稳定就停止，
  不扩到 5M，不换算法。
