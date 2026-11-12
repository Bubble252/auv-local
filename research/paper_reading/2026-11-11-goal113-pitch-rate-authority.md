# Reading note（Goal113 补充）：pitch rate demand、fin authority 与 handoff

**日期**：2026-11-11<br>
**关联 Goal**：Goal113 `PITCH_TRACKING_AMBIGUOUS`；Goal114 2 m-anchored mixed-mechanism 审计<br>
**关系**：这是 `2026-11-11-goal112-pitch-tracking-authority.md` 的补充 note，不重复完整文献卡片，只聚焦 rate demand / authority / handoff。

## 1. Goal113 给出的新问题

Goal113 在 3.0 m 主窗口没有单一机制过半：

```text
AUTHORITY_LIMITED        = 0/11
COMMAND_RATE_UNREACHABLE = 2/11
SPEED_CURRENT_COUPLED    = 3/11
PREENTRY_PITCH_LAG       = 0/11
NO_PITCH_LAG             = 8/11
```

但描述层显示 pitch error >=0.45 的样本数从 3 m 的 3/11 增加到 2 m 的 7/11。
因此问题不是“3 m 前已经坏”，而是：

```text
pitch command rate / actual pitch rate / stern near-limit / speed-current coupling
在 3 m 到 2 m 之间如何排序和叠加？
```

## 2. 与文献的对应

### 2.1 Command filter 与 anti-windup

- Xu et al. 2019, “Underactuated AUV Nonlinear Finite-Time Tracking Control Based on
  Command Filter and Disturbance Observer”, `10.3390/s19224987`。
- 该文用 command filtered backstepping 避免微分膨胀，并设计 filter compensation loop
  与 anti-integral-saturation。
- 对本项目：如果 `pitch_cmd_rate` 在 3→2 m 明显上升，而 `nu[:,4]` 长期落后，应该先查
  command filter / rate limit / 参考生成，而不是把问题归因 RL 探索。

### 2.2 低速下的 fin effectiveness

- Eng et al. 2015, “Minimum Speed Seeking Control for Nonhovering Autonomous Underwater
  Vehicles”, `10.1002/rob.21625`。
- 该文指出鱼雷形、单推进器、正浮力 AUV 在低于最低速度时 fin 失效，无法稳定 depth
  control。
- 对本项目：Goal113 的 failure 中位 surge 约 1.6 m/s，和 success 接近；因此当前不能
  简单说“低速导致 fin 失效”。但如果 Goal114 的 onset 排序显示 surge 先掉、pitch error
  后升，才支持这条机制。

### 2.3 Handoff 后的终端控制是独立难点

- Page et al. 2021, “Underwater Docking Approach and Homing to Enable Persistent
  Operation”, `10.3389/frobt.2021.621755`。
- 该文用 ILOS 到 2 m handoff，再用 light tracking 完成 terminal homing；approach 到达
  handoff 的成功率很高，但 terminal homing 仍有明显失败率。
- 对本项目：3 m 到 2 m 的 pitch lag 形成，符合“handoff 后终端控制独立变难”的模式；
  Goal114 应把 2 m 作为主锚点，而不是继续把 3 m 当 entry。

### 2.4 欠驱动与海流耦合

- Paliotta et al. 2019, “Trajectory Tracking and Path Following for Underactuated Marine
  Vehicles”, `10.1109/TCST.2018.2834518`。
- 该文明确处理欠驱动 marine vehicle 与 constant/irrotational ocean currents。
- 对本项目：Goal113 的 `coupling_strong` 在 3 m 原始相关很高（10/11），但其中 8 个
  pitch error 还没过阈值；这说明相关可能来自共同趋势，不能直接写成海流因果。Goal114
  必须做逐帧 onset 排序，而不是只看窗口 Spearman。

## 3. 对 Goal114 的约束

Goal114 应该预注册：

- 主窗口固定 2.0 m；
- 3.0→2.5→2.0 m 分段；
- 对每个 failure 记录各机制的首次持续越界时刻：
  `pitch_cmd_rate`、`pitch_rate`、`stern_norm`、`speed_error`、`current_mag`、
  `pitch_err`；
- 用“谁先持续越界”而不是“窗口内相关性”做分类；
- 若 onset 时间差小于一个控制窗口，判 `MIXED_PITCH_HANDOFF`，不强行单因归因。

## 4. 边界

这些文献只提供机制候选。Goal113 的结果没有证明：

```text
failure = command filter bug
failure = low-speed fin stall
failure = ocean current
failure = SAC would fix it
```

Goal114 仍然只读，不训练、不改 reward/terminal/success/goal radius、不打开 held-out。
