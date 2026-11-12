# Reading note：AUV command/actuator feedback 与 residual 机制

**日期**：2026-11-09<br>
**触发**：Goal49 的 controller/executor feedback feature coverage 不足，进入 autoresearch
pivot；需要为 Goal50 的 command-residual/guidance-timing 特征选择找外部依据。

## 1. 检索

检索工具：`paper-search` skill。查询：

```text
command residual underwater vehicle path following feedback
AUV actuator saturation tracking error residual control
partial rank correlation small sample covariates
```

时间范围 2018-2026，来源 arXiv、Crossref、Semantic Scholar。Semantic Scholar 连续
HTTP 429，最终没有贡献结果；可用结果为 arXiv 18 条、Crossref 18 条，共 36 个 unique
records。下面只记录与 Goal49 blocker 直接相关的 5 篇；不把泛化的 rank-correlation
结果当作 controller 机制证据。

## 2. 直接相关文献

### Zhu, Yang & Biglarbegian 2022, “A Fuzzy Logic-based Cascade Control without Actuator Saturation for the Unmanned Underwater Vehicle Trajectory Tracking”

- arXiv `2210.01706`；DOI `10.1007/s10846-022-01742-w`。
- 核心：UUV trajectory tracking 的 cascade control，显式处理 actuator saturation；
  用 fuzzy logic 调整控制以避免饱和。
- 对 Goal49/50 的含义：actuator saturation 是 AUV tracking 的独立机制，不应只用一个
  seed-level saturation fraction 代表；Goal50 应至少保留 per-channel saturation
  或 command-residual activity，避免再次出现特征常数化。

### Remmas, Meurer, Hamamatsu et al. 2025, “Efficient Control Allocation and 3D Trajectory Tracking of a Highly Manoeuvrable Under-actuated Bio-inspired AUV”

- arXiv `2504.19049`。
- 核心：under-actuated AUV 的 control allocation 与 3D trajectory tracking；推进器/
  fin 分配会直接影响可达控制与跟踪误差。
- 对 Goal49/50 的含义：command residual 不是抽象标量，而是经过 control allocation
  后进入 actuator；Goal50 应同时记录/使用 `command_residual_applied` 与
  `command_corrected_physical`，而不是 action-space `residual_action`。

### Vadapalli & Mahapatra 2023, “3D Path Following Control of an Autonomous Underwater Robotic Vehicle Using Backstepping Approach Based Robust State Feedback Optimal Control Law”

- DOI `10.3390/jmse11020277`；Journal of Marine Science and Engineering。
- 核心：backstepping + robust state feedback 的 3D path following；跟踪误差和
  state feedback 是控制律的核心输入。
- 对 Goal49/50 的含义：heading/pitch/speed tracking error 是合理的候选 feedback，
  但 Goal49 已显示单独的 late mean absolute error 不能解释 strict125 association；
  Goal50 不应重复同一组标量，而应转向 command-residual 和 timing。

### Rober, Hammond, Cichella et al. 2022, “3D Path Following and L1 Adaptive Control for Underwater Vehicles”

- arXiv `2204.00935`；DOI `10.1016/j.oceaneng.2022.110971`。
- 核心：L1 adaptive control 处理模型不确定性和路径跟踪；自适应项依赖跟踪误差和
  控制输入历史。
- 对 Goal49/50 的含义：controller feedback 可能是**历史/时序**的，不只是 late-window
  mean；Goal50 应使用 episode-level guidance timing、phase-entry time 或 command
  residual 的 mean/std，而不是只看末窗 fraction。

### Che 2022, “Single critic network based fault-tolerant tracking control for underactuated AUV with actuator fault”

- DOI `10.1016/j.oceaneng.2022.111380`；Ocean Engineering。
- 核心：underactuated AUV 的 actuator fault-tolerant tracking；actor/critic 结构与
  执行器故障/有效性变化相关。
- 对 Goal49/50 的含义：执行器有效性、饱和和故障状态是独立于 tracking error 的
  feedback 通道；Goal50 应报告 per-channel actuator activity/saturation，避免把
  actuator 机制压缩成一个常数标量。

## 3. 综合判断

1. Goal49 的三个常数特征不是“controller feedback 不存在”，而是 feature 选择与
   trace 语义不匹配：C1 的 residual 在 command 空间，terminal phase/route floor 在
   late 窗口没有 seed-level 变化。
2. AUV 文献把 actuator saturation、control allocation、tracking error 和 adaptive
   control history 视为不同 feedback 通道；Goal50 应优先使用已记录的
   `command_residual_applied` / `command_corrected_physical` 和 episode-level timing。
3. 小样本 rank inference 仍要求 permutation/negative control；Goal50 应沿用 Goal49
   的 same-feature two-block Gate 和 sigma permutation max-statistic，不因 Goal49
   coverage 失败而放宽阈值。
4. 如果 Goal50 的 informative recorded features 仍不能解释，才需要新的
   instrumentation replay；不能直接恢复命令残差 PPO 或 supervised residual。

## 4. 对 Goal50 的设计约束

- 复用 Goal47 traces，不新增 replay、不训练、不打开 held-out；
- 特征族先冻结，至少 10 个在两个 strict125 block 可评估；
- 必选：`command_residual_applied` / `command_corrected_physical` 的 late
  mean/std/activity；
- 必选：episode-level terminal phase duration / time since entry、route-floor
  active duration 或 early/late activation；
- 可选但需预注册：per-channel actuator saturation/activity；
- 同时报告 non-discordant negative control 和 within-block sigma permutation
  max-statistic；
- 若 coverage 仍 `<10`，判 `INSUFFICIENT_FEATURE_VARIATION`，不得继续加变量。

## 5. 引用

```text
Zhu, D., Yang, S. X., Biglarbegian, M. (2022). A Fuzzy Logic-based Cascade
Control without Actuator Saturation for the Unmanned Underwater Vehicle
Trajectory Tracking. arXiv:2210.01706. DOI:10.1007/s10846-022-01742-w.

Remmas, W., Meurer, C., Hamamatsu, Y., et al. (2025). Efficient Control
Allocation and 3D Trajectory Tracking of a Highly Manoeuvrable Under-actuated
Bio-inspired AUV. arXiv:2504.19049.

Vadapalli, S., Mahapatra, S. (2023). 3D Path Following Control of an Autonomous
Underwater Robotic Vehicle Using Backstepping Approach Based Robust State
Feedback Optimal Control Law. Journal of Marine Science and Engineering.
DOI:10.3390/jmse11020277.

Rober, N., Hammond, M., Cichella, V., et al. (2022). 3D Path Following and L1
Adaptive Control for Underwater Vehicles. arXiv:2204.00935.
DOI:10.1016/j.oceaneng.2022.110971.

Che, G. (2022). Single critic network based fault-tolerant tracking control for
underactuated AUV with actuator fault. Ocean Engineering.
DOI:10.1016/j.oceaneng.2022.111380.
```
