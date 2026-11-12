# Reading note：AUV 终端 homing、pitch/depth tracking 与 fin authority

**日期**：2026-11-11<br>
**关联 Goal**：Goal112 `TRACKING_LIMITED`；Goal113 pitch tracking/authority 审计<br>
**用途**：只用于解释 Goal112 的机制候选和设计 Goal113 的最便宜 falsification，不作为本项目实验结论。

## 1. 为什么读这组文献

Goal112 的结果是：

```text
strict C0 pre-entry bad failure = 11
reference involved = 3/11
tracking involved  = 8/11
tracking pitch bad = 7/11
material saturation = 0/11
```

这把问题从“terminal reward/credit”推进到“实际 pitch 响应是否跟得上 command”。
需要回答的不是“要不要换 SAC”，而是：

1. 鱼雷形/欠驱动 AUV 的 pitch 权限是否天然受速度、浮力、fin effectiveness 限制；
2. 终端 homing/handoff 阶段是否本来就需要比巡航更高的控制权限；
3. ocean current / model uncertainty 是否会让 command filter 或跟踪误差累积；
4. 这些机制能否用现有 trace 的 pitch rate、vertical speed、speed error、current
   和 actuator 通道区分。

## 2. 文献卡片

### 2.1 Eng et al. 2015：非悬停 AUV 的最低速度与 fin effectiveness

- **Title**：Minimum Speed Seeking Control for Nonhovering Autonomous Underwater Vehicles
- **Authors**：You Hong Eng; Mandar Chitre; Kien Ming Ng; Kwong Meng Teo
- **Venue**：Journal of Field Robotics, 2015
- **DOI**：`10.1002/rob.21625`
- **与本 Goal 的相关点**：大多数 AUV 是单推进器、鱼雷形、用 elevator/rudder 作为
  control surface，并且为了回收方便常带正浮力。该类非悬停 AUV 存在一个最低速度；
  低于该速度时 fin 失效，剩余浮力会把载具带回水面，无法稳定 depth control。
- **对 Goal112 的含义**：如果 failure 的 pitch tracking 与低速、speed error 或
  vertical speed 同步恶化，那么“pitch command 不可达”可能不是控制器增益问题，而是
  fin authority 随速度下降的物理限制。Goal113 必须把 `nu[:,0]`（surge）、
  `speed_error_mps`、`nu[:,2]`（heave）和 pitch rate 放在同一窗口分析。
- **边界**：该文关注 minimum-speed seeking，不是 terminal homing；不能直接推出本项目
  的 2 m failure 就是低速造成，只能作为 authority 机制的候选解释。

### 2.2 Paliotta et al. 2019：欠驱动 marine vehicle 的路径跟踪与海流

- **Title**：Trajectory Tracking and Path Following for Underactuated Marine Vehicles
- **Authors**：Claudio Paliotta; Erjen Lefeber; Kristin Ytterstad Pettersen; Jose Pinto; Maria Costa; Joao Tasso de Figueiredo Borges de Sousa
- **Venue**：IEEE Transactions on Control Systems Technology, 2019
- **DOI**：`10.1109/TCST.2018.2834518`
- **与本 Goal 的相关点**：论文把 hand position point 作为输出，用 input-output
  feedback linearization 处理欠驱动水面/水下载具，并明确考虑 constant and
  irrotational ocean currents；Lyapunov 分析给出跟踪/路径跟随稳定性。
- **对 Goal112 的含义**：欠驱动载具的“位置误差”不能直接等同于“姿态 command 错”。
  在存在海流时，controller 可能通过姿态/速度耦合来补偿位置偏差，导致 command 与
  direct LOS 不完全一致。Goal113 应把 current 和 speed error 作为协变量，而不是把
  command-LOS 偏差直接判成 reference bug。
- **边界**：该文是模型基控制与稳定性证明，不是 RL residual；不能据此判断当前 PPO
  残差方向是否正确。

### 2.3 Xu et al. 2019：command filter、disturbance observer 与海流

- **Title**：Underactuated AUV Nonlinear Finite-Time Tracking Control Based on Command Filter and Disturbance Observer
- **Authors**：Hao Xu; Guo-cheng Zhang; Jian Cao; Shuo Pang; Yu-shan Sun
- **Venue**：Sensors, 2019
- **DOI**：`10.3390/s19224987`
- **与本 Goal 的相关点**：论文建立含海流扰动的 3D 跟踪误差模型，用 super-twisting
  observer 估计海流，用 command filtered backstepping 避免传统 backstepping 的
  微分膨胀，并设计 filter compensation loop 与 anti-integral-saturation。
- **对 Goal112 的含义**：如果 command 变化快、pitch rate 跟不上，可能是 command
  filter/补偿不足，而不是 RL 探索不足。Goal113 应显式比较
  `d(command_pitch)/dt` 与 `nu[:,4]`（pitch rate），并检查误差是否在 command
  快速变化后累积。
- **边界**：该文是 simulation study，且控制器结构与本项目不同；只能提供机制假设，
  不能直接套参数。

### 2.4 Page et al. 2021：ILOS 到 terminal homing 的 handoff

- **Title**：Underwater Docking Approach and Homing to Enable Persistent Operation
- **Authors**：Brian R. Page; Reeve Lambert; Jalil Chavez-Galaviz; Nina Mahmoudian
- **Venue**：Frontiers in Robotics and AI, 2021
- **DOI**：`10.3389/frobt.2021.621755`
- **与本 Goal 的相关点**：论文用 Dubins path + integral line-of-sight 跟踪，直到
  handoff 给 terminal homing；实验报告中 approach phase 在 48/48 次测试中到达
  2 m handoff，terminal homing 能处理最高 5 m offset，约 70% 成功率（12/17）。
- **对 Goal112 的含义**：2 m 是一个合理的 handoff 尺度，但 handoff 后的 terminal
  homing 本身就是独立难点。Goal111/112 观察到 6/11 在 3.0 m 已坏，说明不能只盯
  2 m entry；Goal113 应把 onset 前移到 3.0 m 甚至更早。
- **边界**：该文是 docking 场景，handoff 后的 “light tracking” 与本项目的 ALOS +
  residual 不同；70% 成功率不能拿来当本项目的 Gate。

### 2.5 Bermudez 2016：单推进器 AUV 的 terminal homing 可控性

- **Title**：Terminal Homing for Autonomous Underwater Vehicle Docking
- **Author**：Eric B. Bermudez
- **Type**：Thesis / technical report, 2016（OpenAlex 无 DOI）
- **与本 Goal 的相关点**：摘要明确指出，传统单螺旋桨 AUV 为了保持可控性，必须以
  较高的相对 approach velocity 进入 docking；否则 terminal homing 的控制权限不足。
  另一条路线是加装前后水平/垂直 cross-tunnel thrusters 获得悬停能力，从而允许更慢、
  更谨慎的接近。
- **对 Goal112 的含义**：这与 Eng et al. 的 fin effectiveness 结论一致：低速/低权限
  的 terminal phase 可能天然难以同时满足位置和姿态。Goal113 应检查 failure 是否在
  低速 + 大 pitch demand 的联合区域出现，而不是只看 pitch error 的绝对值。
- **边界**：这是 thesis，不是同行评审期刊；只作为工程直觉和假设来源，不作为定量
  阈值依据。

## 3. 综合出的 Goal113 机制假设

### H113-A：command rate / 可达性限制

pitch command 在接近 goal 时变化过快，实际 pitch rate 长期落后；tracking error
在 command 快速变化后累积，而不是单纯 controller 增益不足。

可证伪指标：

```text
|d(command_pitch)/dt| 与 |nu[:,4]| 的窗口差
command/actual pitch rate 符号一致性
pitch error 与 command rate 的滞后相关
```

### H113-B：fin / actuator authority 限制

在低速、较大 speed error 或海流下，stern/fin 的 pitch authority 不足；表现为
pitch rate 无法跟随、actuator 接近物理限幅，或 `material_saturated` 增加。

可证伪指标：

```text
nu[:,0]（surge）
speed_error_mps
nu[:,2]（heave）
actuator_base_physical[:,1] / actuator_total_physical[:,1]
material_saturated
```

### H113-C：深度/速度/海流耦合

pitch tracking error 与 vertical speed、speed error、current magnitude 的相关性
高于与 command rate 的相关性；失败不是纯姿态环，而是垂直/速度耦合。

可证伪指标：

```text
pitch_error 与 nu[:,2]、speed_error_mps、||current_ned_mps|| 的窗口 Spearman rho
```

### H113-D：pre-entry pitch lag 已经形成

3.0 m 已坏的 6 个样本在 3.0 m 之前就有稳定 pitch lag；2 m 只是记录点，不是起点。

可证伪指标：

```text
3.0 m entry 及前 4 帧的 |pitch_error|
3.0 m → 2.0 m 的 pitch error 斜率
```

## 4. 对 Goal112 判读的约束

这些文献支持“pitch tracking 受速度/权限/海流耦合影响”这一机制族，但不能证明本项目
的 11 个 failure 具体属于哪一种。Goal112 只能写成：

```text
tracking involved majority
tracking pitch is the strongest channel
not pure reference error
not material saturation
```

不能写成：

```text
failure = fin authority insufficient
failure = ocean current caused pitch lag
failure = SAC would fix it
```

## 5. 与本地历史/代码的对照

- Goal89/104 已确认 frozen BC 的 22/22 tensor 与 Goal86 P-A BC 完全一致，
  当前问题不是 BC 接线或隐藏层宽度错位；
- Goal108 已确认 PPO/SAC state encoder 3/3 层 `torch.equal`，无 NoisyNet；
- Goal108 SAC 使用 `bc_path_dim=12` + 随机 path encoder + `bc_checkpoint=null`，
  与 Goal86 P-A SAC 的 `path8 + endpoint4 + BC encoder` 不同构；因此若未来做 SAC
  对照，必须先固定 path 参数化和初始化；
- Goal109 已报告 strict C0 的 material saturation 在 success/failure 两组都为 0，
  与 Goal112 的 0/11 一致；因此当前不是“执行器打满”而是“响应/权限/耦合”问题。

## 6. 参考链接

- Eng et al. 2015: https://doi.org/10.1002/rob.21625
- Paliotta et al. 2019: https://doi.org/10.1109/TCST.2018.2834518
- Xu et al. 2019: https://doi.org/10.3390/s19224987
- Page et al. 2021: https://doi.org/10.3389/frobt.2021.621755
- Bermudez 2016: OpenAlex `W` record, thesis; 无 DOI

## 7. 备注

本 note 只读取标题、摘要和公开元数据；没有把论文中的控制器参数、成功率或仿真结论
直接迁移到本项目。Goal113 的阈值仍必须由本项目预注册 Gate 决定。
