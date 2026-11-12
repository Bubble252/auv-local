# Reading note：AUV docking 的 pre-entry 几何、terminal capture 与 conservative residual

日期：2026-11-11<br>
关联：Goal110 `INSUFFICIENT_EVENTS` / pre-entry censoring<br>
检索方式：arXiv API 按已知条目核验标题与摘要；本笔记只用于机制对照，不改变当前 Gate。

## 1. 检索结论

Goal110 的失败不是“进入 2 m 后 rpm 或几何谁先坏”的可排序事件，而是大量失败在
进入前已经失去可评估性。AUV docking 文献给出的共同提示是：

1. 远场 homing 和近场 terminal capture 往往不是同一套几何/测量条件；
2. 终端成功通常有独立的 capture/docking criterion，而不是只用远场路径跟踪误差；
3. 残差动作空间需要保守约束，否则容易把局部修正推到饱和。

这些论文支持“先把 pre-entry 和 entry 几何分开审计”，但都不直接证明本项目的
失败原因，也不作为修改 success/reward/动作参数化的依据。

## 2. 关键论文

### 2.1 Acoustic-based Guidance for Automatic Docking of Holonomic AUVs

- 作者：Ravi Regalo, David Cabecinhas, António Pascoal
- arXiv：`2609.11821`
- 摘要要点：使用双 USBL、SO(3) 互补滤波和 EKF 提供相对位姿，再由 SE(3)
  geometric controller 执行 docking；摘要报告 simulation 与 water trials。
- 与 Goal110 的对应：terminal docking 的几何条件与相对位姿质量是独立问题。
  当前 trace 若在 2 m 前已经几何越界，就不能只用进入后的 rpm 事件解释。
- 不直接迁移：需要声学相对定位硬件；当前实验是 synthetic navigation，且本轮
  不做传感器或控制器替换。

### 2.2 Control System Design and Experiments for Autonomous Underwater Helicopter Docking

- 作者：Haoda Li, Xinyu An, Rendong Feng, Zhenwei Rong, Zhuoyu Zhang,
  Zhipeng Li, Liming Zhao, Ying Chen
- arXiv：`2410.06953`
- 摘要要点：把 docking 分成 Homing 和 Landing；Landing 使用分段对准策略和
  特定 docking success criterion，并通过 pool experiments / sea trials 验证。
- 与 Goal110 的对应：pre-entry/homing 与 terminal capture 应该分段统计。
  Goal110 的“进入后 stall”定义可能错过了 homing 阶段已经形成的误差。
- 不直接迁移：AUH 的机动性与 REMUS100 不同，且其固定姿态/高度 docking
  约束不能直接套用到当前 success 契约。

### 2.3 CANDO: Conservative Action-Noise Directional Optimization for Residual RL

- 作者：Zeyu Gao, Chengkai Hou, Chaoyue Li, Zhuangzhe Wu, Jiaming Liu,
  Yao Mu, Peng Jia, Li Zhang, Shanghang Zhang
- arXiv：`2611.04543`
- 摘要要点：在 frozen generative policy 上组合 noise-space directional
  residual 与 action-space residual，并用 conservative value estimation 和
  gated execution 限制 adaptation。
- 与 Goal110 的对应：与 Goal108 SAC 的 deterministic residual 饱和风险一致；
  但本 Goal 只是只读 terminal 审计，不引入 CANDO，也不改算法。
- 不直接迁移：该方法面向 generative policy 的噪声空间，当前 P_D3 是确定性
  actor + actuator residual，接口不同。

### 2.4 Locally-Guided Actor-Critic: Training a Goal-conditioned Actor with a Subgoal-aware Critic

- 作者：Olivier Serris, Stéphane Doncieux, Olivier Sigaud
- arXiv：`2608.30406`
- 摘要要点：研究 sparse goal RL 中 imagined-subgoal 正则和 potential-based
  reward shaping，指出 shaping 在 terminal states 可能产生 deceptive rewards；
  提出 LG-AC 用 subgoal-conditioned value 做 dense hindsight relabeling。
- 与 Goal110 的对应：terminal reward/geometry 的信用不能只用远场 shaping 解释；
  需要区分 pre-entry 与 post-entry 的 outcome。
- 不直接迁移：当前已有 BC prior、ALOS 和冻结 success；不因该论文改 reward 或
  引入 subgoal critic。

## 3. 对 Goal111 的最小启示

只做一个便宜的只读审计：

```text
closest approach
首次进入 3.0 / 2.5 / 2.0 m
entry 时 vertical / pitch / heading / closure
entry 后是否继续闭合
```

分类为 `APPROACH_DEFICIT`、`PREENTRY_GEOMETRY_BAD`、
`ENTRY_GEOMETRY_OK_LATE_FAILURE`，再决定是否需要新的 terminal capture 或
reward 预注册。

## 4. 不允许的直接迁移

- 不因论文使用 USBL/视觉/电磁 beacon 就新增传感器；
- 不把 SE(3) geometric controller 当作当前 residual 的替代算法；
- 不引入 CANDO/LG-AC 训练；
- 不打开 held-out，不修改 reward/terminal/success/goal radius；
- 不把 pre-entry 相关性写成因果。

## 5. 证据链接

- https://arxiv.org/abs/2609.11821
- https://arxiv.org/abs/2410.06953
- https://arxiv.org/abs/2611.04543
- https://arxiv.org/abs/2608.30406
