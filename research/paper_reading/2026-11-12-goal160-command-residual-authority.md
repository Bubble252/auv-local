# Goal160 阅读笔记：residual RL、AUV path following 与 command authority

**日期**：2026-11-12<br>
**触发**：Goal160 command-space residual 在 H3 zero 上全面负增益；需要判断这是
residual 权限/credit 问题，还是 AUV 路径跟踪本身的终端几何问题。

## 检索范围

- `paper-search`：`AUV path following reinforcement learning residual control terminal capture`（2020–2026）
- `paper-search`：`residual reinforcement learning command space control base controller`（2019–2026）
- 下载并阅读：
  - `arXiv:1812.03201` Residual Reinforcement Learning for Robot Control
  - `arXiv:2106.08050` Residual Reinforcement Learning from Demonstrations
  - `arXiv:2111.04153` Data-Efficient Deep RL for Attitude Control of Fixed-Wing UAVs
- 通过 OpenAlex 核对摘要：
  - Ma et al. 2023, IEEE T-IV, DOI `10.1109/TIV.2023.3282681`
  - Zhang et al. 2020, IEEE Access, DOI `10.1109/ACCESS.2020.2970433`
  - Pan et al. 2026, Sensors, DOI `10.3390/s26144548`

## 论文要点

### 1. Johannink et al. 2019：residual 是“补模型未覆盖部分”，不是替代基控制器

论文把控制分解为 conventional feedback controller 的 base control 和 RL 学习的
residual，最终控制是两者叠加。它的强项是接触、摩擦等一阶模型难以刻画的残差，
而不是在已知良好的路径跟踪区域重新学习整套控制。

对本项目的含义：H3/ALOS 已经是强基控制器；如果 command residual 在 2–10 m
持续改写 yaw/pitch/speed，并把 base controller 的相位/阻尼结构破坏掉，就会得到
“residual 活跃但闭环更差”的现象。应先验证降低 residual authority 是否单调恢复，
而不是直接归因于 SAC 样本效率。

### 2. Alakuijala et al. 2021：residual 的样本效率来自 base prior

该文把 residual 叠在传统控制器或 BC 策略上，强调 residual 形式能显著降低样本
复杂度，但前提是 base policy 提供了可用的先验；residual 负责修正分布偏移和
未建模部分。

对本项目的含义：B/C 的 1M 训练不是“没有学到东西”；更可能是 base prior 很强，
而 residual 的 credit 没有把“不要破坏 base”编码进去。后续应优先做 authority/
direction 诊断，而不是扩到 5M 或 REDQ/DroQ。

### 3. Bøhn et al. 2021：数据效率、执行延迟和动力学多样性决定 sim-to-real

固定翼 UAV 的 SAC 控制可以在少量飞行数据上达到与 ArduPlane PID 可比的性能；
论文明确指出 actuation delay 和多样化动力学对迁移至关重要。

对本项目的含义：如果我们的 command residual 没有显式建模执行延迟、rate limit
和基控制器内部状态，学习到的 residual 很容易在闭环中相位错位。当前 Goal160
的 command residual 直接改指令，比 actuator residual 更容易触发这种相位问题；
这与 B 的 strict125 大幅下降一致。

### 4. Ma et al. 2023：速度环与路径/姿态环可以分解

ModelPPO 用 PI 控制维持所需航速，RL 主要控制舵/升降舵完成三维路径跟踪。也就是
说，AUV 文献中常见的有效分解是“低速/速度环交给控制器，RL 处理路径/姿态”。

对本项目的含义：command residual 同时改 heading、pitch、speed 三个通道，
可能把已经稳定的速度环也纳入了高方差策略；后续若继续 command 分支，应优先
测试 channel-restricted residual（例如只允许 pitch/stern 或只允许 heading），
而不是全通道缩放。

### 5. Zhang et al. 2020 / Pan et al. 2026：AUV 路径跟踪的稀疏奖励与 LOS 先验

Zhang 等强调 AUV 路径跟踪的稀疏奖励和样本效率问题；Pan 等的 ILLT 把 LOS
guidance 与 TD3 结合，并把 look-ahead distance 作为可学习变量，说明 LOS/ALOS
先验本身是有效结构，不应被 command residual 随意覆盖。

对本项目的含义：我们的 H3/ALOS base 已经是强先验；如果 residual 在终端附近
覆盖 ALOS 的 look-ahead/terminal alignment，就会破坏终端几何。Goal160 的
C（terminal-only）仍然更差，说明“只在近区开 residual”并不自动等于“保留 ALOS
终端结构”；需要进一步限制 residual 的方向或通道。

## 最小可迁移组件

1. **authority cap**：residual 的范数/通道幅度应相对于 base command 或 BC
   residual 限幅，而不是只用一个全局 scale。
2. **direction gate**：只在 residual 与“减小 signed terminal error”的方向一致时
   放行；否则置零或衰减。
3. **channel restriction**：优先只允许 pitch/stern 或 heading，不要同时改
   heading/pitch/speed；速度环继续由 base controller 保持。
4. **delay-aware base**：把 base controller 的执行延迟/rate limit 纳入 residual
   的观测或训练环境，避免学出相位超前的指令。

## 与 Goal160 结果的连接

- B（command，全程训练、部署 gate）在 current10 default `-7/120`；
- C（command，训练时 terminal-only）在 current10 default `-6/120`，strict
  `-14/120`；
- A（actuator）在 current10 default `+1/120`，strict `-4/120`；
- 三者 residual 都活跃，且没有 timeout/safety；说明问题不是“没开 residual”，
  而是 command residual 的 authority/credit 与 ALOS 终端结构冲突。

因此下一步先做 eval-only scale `{0.25, 0.5}` 和只读 residual-error 相关分析。
只有在 scale 扫描显示单调恢复时，才值得新开预注册 Goal 做 channel-restricted /
direction-gated command residual；否则转 terminal reward/credit 或 base
controller capture 审计。REDQ/DroQ 仍不在当前证据支持范围内。
