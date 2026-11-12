# Reading note：pre-entry 几何、homing→capture handoff 与 terminal 判据

日期：2026-11-11<br>
关联：Goal111 `PREENTRY_GEOMETRY_BAD_MAJORITY`<br>
检索方式：arXiv API 核验标题与摘要；只作机制对照，不改变当前控制/奖励契约。

## 1. 结论

Goal111 显示 strict C0 的失败更多在进入 2 m 前或 entry 时已经形成：

```text
APPROACH_DEFICIT                7/20
PREENTRY_GEOMETRY_BAD          11/20
ENTRY_GEOMETRY_OK_LATE_FAILURE  2/20
```

这与 AUV docking 文献中的分段结构一致：远场 homing、近场对准和 terminal
capture 通常有不同参考与判据。下一步应先看参考/制导几何和实际 tracking 在
`3.0/2.5/2.0 m` 如何交接，而不是继续单独调终端 rpm。

## 2. 关键论文

### Acoustic-based Guidance for Automatic Docking of Holonomic AUVs

- arXiv：`2609.11821`
- 方法：双 USBL、SO(3) 互补滤波、EKF 相对位置、SE(3) geometric controller。
- 对本项目的含义：terminal docking 依赖相对几何和专用控制阶段；远场路径
  tracking 指标不能替代 pre-entry 几何检查。
- 不能直接迁移：需要声学相对定位；当前 trace 是 synthetic navigation，且本
  Goal 不新增传感器。

### Control System Design and Experiments for Autonomous Underwater Helicopter Docking

- arXiv：`2410.06953`
- 方法：Homing 与 Landing 分阶段；Landing 使用分段对准、线性速度决策和特定
  docking success criterion。
- 对本项目的含义：应把 entry geometry 与 post-entry terminal capture 分开审计；
  “进入 2 m 后 stall”可能只是 pre-entry 误差的后果。
- 不能直接迁移：AUH 的机动性和固定姿态/高度约束与 REMUS100 不同。

### CANDO: Conservative Action-Noise Directional Optimization for Residual RL

- arXiv：`2611.04543`
- 方法：frozen generative policy 上组合 noise-space directional residual 与
  action-space residual，并用 conservative value 和 gated execution 限制 adaptation。
- 对本项目的含义：如果以后进入 residual 训练，不能只放大确定性残差；需要保守
  约束或门控。但这与本轮 pre-entry 审计无关，本 Goal 不引入。
- 不能直接迁移：当前是确定性 actor + actuator residual，不是 generative policy
  的噪声空间。

### Locally-Guided Actor-Critic

- arXiv：`2608.30406`
- 方法：研究 sparse goal RL 中的 subgoal 正则与 potential-based reward shaping，
  指出 terminal state 的 deceptive reward 风险。
- 对本项目的含义：terminal reward 与 pre-entry 几何应分开，不能仅凭 dense
  shaping 的改善推断 terminal capture 成功。
- 不能直接迁移：当前已有 BC prior、ALOS 与冻结 success，不因该论文改 reward。

## 3. 对 Goal112 的最小实验

只读 Goal107 trace，按 `3.0/2.5/2.0 m` 分段，逐段比较：

```text
command / guidance target
eta 与 nu 的实际姿态/速度
heading_error / pitch_error / speed_error
actuator base / residual / total
guidance_terminal_phase / route speed / filter 状态
```

目标是回答：

1. pre-entry 几何误差在 3.0 m 时是否已经存在；
2. 是参考命令本身偏，还是实际状态跟不上命令；
3. 2 个 late failure 是否与其余 18 个失败机制不同。

## 4. 不允许的直接迁移

- 不新增 USBL、视觉或电磁传感器；
- 不把 SE(3) 控制器直接替换当前 ALOS + residual；
- 不启动 CANDO/LG-AC 训练；
- 不改 reward、terminal、success、goal radius；
- 不打开 held-out；
- 不把 pre-entry 分类写成因果。
