# Reading note: terminal capture, LOS/CTE, and docking guidance

日期：2026-11-09<br>
用途：Goal30 blocker 后的文献校准；不是完整综述，不引入未经核验的结论。

## 1. 问题

Goal30 在真正执行的 `ALOS -> PD/PI -> actuator` 链路上把 terminal alignment window
从 `20 m` 前移到 `30 m`，结果 protocol success 下降、plane miss 上升、terminal
horizontal/vertical offset 变差。现有 trace 审计显示 terminal phase 占轨迹约
`82%`，3D/vertical path RMSE 变差，而 terminal saturation 没有升高。

需要回答：为什么“更早直接指向目标”会破坏终端捕获？下一步应该设计什么样的终端
控制器，而不是继续扫窗口？

## 2. 已核验文献

### 2.1 LOS/CTE 路径跟踪

**Ghosh & Mandal (2024), "AUV way-point tracking at constant velocity: cross-track
error (CTE) and line-of-sight (LOS) guidance", Engineering Research Express,
DOI `10.1088/2631-8695/ad36b0`.**

- 研究欠驱动 AUV 在 2D 平面中的路径跟踪与 way-point tracking；
- 用 CTE 和 LOS 共同确定期望航向，并在直线、尖角、曲线、圆形路径上测试；
- 强调路径跟踪误差是控制目标，而不是只看是否朝目标点前进。

对本项目的含义：Goal30 的 30 m 窗口把大部分轨迹变成“朝目标点前进”，丢掉了 CTE
反馈；这会解释为什么 horizontal path RMSE 没有显著改善，而 vertical/3D RMSE 变差。

**Huang et al. (2017), "Horizontal path following for underactuated AUV based on
dynamic circle guidance", Robotica, DOI `10.1017/s0263574715000867`.**

- 用动态辅助圆和 cross-track error `e` 构造 heading modification；
- 通过 `e -> heading modification` 的凸映射改善欠驱动 AUV 的路径跟踪；
- 关键不是固定一个很大的 lookahead，而是让期望航向随横向误差变化。

对本项目的含义：如果要在终端降低横向 offset，应该引入“随 cross-track error 调整
的终端 heading/lookahead”，而不是把固定 alignment window 继续放大。

**Yang, Chavez-Galaviz & Mahmoudian (2026), "An Adaptive Fixed-Time Line-of-Sight
Guidance Scheme for 3D Path Following of Underwater Vehicles: Theory and
Experiment", arXiv `2609.14756`.**

- 3D LOS + fixed-time stability，收敛时间不依赖初始条件；
- 用 fixed-time adaptive estimator 补偿时变 sideslip / 海流；
- 引入 time-varying lookahead，改善曲线路径上的跟踪；
- 论文摘要明确报告：time-varying lookahead 变体降低了 cross-track 和 vertical
  tracking error。

对本项目的含义：Goal30 的失败与“固定、过长 lookahead”一致；更合理的方向是
time-varying lookahead 或 cross-track/vertical error feedback，而不是全局直接
goal seeking。

### 2.2 输入约束与扰动下的路径跟踪

**Miao et al. (2023), "PECLOS path-following control of underactuated AUV with
multiple disturbances and input constraints", Ocean Engineering,
DOI `10.1016/j.oceaneng.2023.115236`.**

- 题目和摘要明确针对 unteractuated AUV 的多扰动与输入约束；
- 说明终端/路径控制器不能假设无限权限，必须把执行器约束纳入设计。

对本项目的含义：Goal30 的 saturation 没有升高，说明当前失败不是“权限不够”；
但未来控制器仍应显式报告 saturation/rate clip，不能靠更大窗口换权限。

**Du et al. (2023), "Improved Indirect Adaptive Line-of-Sight Guidance Law for
Path Following of Under-Actuated AUV Subject to Big Ocean Currents", Ocean
Engineering, DOI `10.2139/ssrn.4379644`.**

- 针对大海流下的欠驱动 AUV LOS 路径跟踪；
- 强调海流补偿/自适应项，而不是只改目标点。

对本项目的含义：本项目已有 current estimate 和 crab compensation；Goal30 的
candidate 改变了 crab/heading 轨迹，但没有改善 terminal offset。后续应把海流补偿
与 cross-track/terminal error 分开做消融，不能把“更早对准”当成海流补偿。

### 2.3 终端 docking / homing

**Li, An, Feng et al. (2024), "Control System Design and Experiments for
Autonomous Underwater Helicopter Docking Procedure Based on
Acoustic-inertial-optical Guidance", arXiv `2410.06953`.**

- docking 分为 Homing 和 Landing 两阶段，每阶段用不同 guidance；
- Landing 采用分段 aligning strategy 和 linear velocity decision；
- 强调固定朝向、姿态、深度和成功 docking criterion；
- 有 pool experiments 和 sea trials。

对本项目的含义：终端捕获不是“尽早指向目标”一个动作，而是分阶段、带速度决策和
姿态/深度对齐的 controller。Goal30 把 30 m 窗口当成单一开关，机制上过于粗糙。

**Verma, Kumar & Arya (2026), "Docking of Autonomous Vehicles with a Stationary
Docking Station in 3D Space", arXiv `2607.02478`.**

- 用 range 和 line-of-sight kinematics 构造 finite-time sliding mode docking；
- 要求车辆在接近时对齐特定朝向并降低速度，近零速度安全 docking；
- 在 MATLAB 中验证多种初始位置/朝向。

对本项目的含义：终端收敛需要显式的速度调度和朝向对齐；只改变 goal-seeking
window 不能保证在过平面时同时满足位置、朝向和速度。

**Li, Zhang & Yang (2016), "Terminal Underwater Docking of an Autonomous
Underwater Vehicle Using One Camera and One Light", Marine Technology Society
Journal, DOI `10.4031/mtsj.50.6.6`.**

- 终端 docking 依赖近距传感/视觉导引；
- 说明终端阶段的观测与控制目标与巡航路径跟踪不同。

对本项目的含义：当前仿真使用 `p_nav` 终止，真实终端捕获应同时报告 true-state
`eta` 和估计状态；Goal30 的 `terminal_eta_capture` 修正是必要的。

**Kim, Whang & Kim (2016), "Finite Horizon Integrated Guidance and Control for
Terminal Homing in Vertical Plane", JGCD, DOI `10.2514/1.g001699`.**

- 垂直平面 terminal homing 的有限时域制导/控制一体化；
- 关注终端命中而不是全程路径跟踪。

**Li, Tao, Wang et al. (2023), "Three-Dimensional Optimal Homing Guidance Without
Terminal Maneuverability Advantage", JGCD, DOI `10.2514/1.g007483`.**

- 导弹/飞行器 3D optimal homing，明确考虑终端机动能力不占优，并在 bounded
  acceleration 下推导两阶段制导律；
- 说明 terminal homing 需要在权限约束下设计，而不是假设可以任意转向。

对本项目的含义：这些 homing 文献支持“终端阶段是独立控制问题”；但它们多用于
飞行器/导弹，不能直接照搬到欠驱动 AUV，需要与 LOS/CTE 路径跟踪结合。

## 3. 综合判断

1. **路径跟踪与终端 homing 是两个目标，不能用一个固定窗口硬切换。**
   Goal30 的 30 m 窗口相对约 35 m 的路径过长，terminal phase 占 82%，等价于把
   路径跟踪替换成 goal seeking；trace 上表现为 3D/vertical RMSE 变差、terminal
   offset 变差。

2. **更合理的终端控制器应保留 cross-track / vertical error feedback。**
   LOS/CTE 文献支持用横向误差和 time-varying lookahead 调节期望航向，而不是简单
   放大 lookahead。Goal31 应把“终端横向/垂直收敛项”作为单因子，而不是再扫窗口。

3. **终端阶段需要显式速度/朝向/过平面几何。**
   Docking 文献使用 Homing + Landing、分段 aligning、linear velocity decision、
   finite-time sliding mode 等机制；Goal30 只改 alignment window，缺少这些约束。

4. **输入约束/海流必须单独报告。**
   Goal30 的 saturation 没有升高，说明本次失败不是权限问题；但后续控制器仍要
   报告 saturation、rate clip、current estimate 和 crab compensation，避免把海流
   收益误写成终端几何收益。

## 4. 对 Goal31 的约束

- 不继续扫 `terminal_alignment_min_m/max_m`；
- 不重开 command residual PPO、reward/gamma/seed 搜索；
- 新候选必须是单因子：例如保留 20 m 路径跟踪窗口，只增加 cross-track/vertical
  terminal correction，或使用随距离收缩的 time-varying lookahead；
- 主指标继续用 `terminal_eta_capture` + terminal horizontal/vertical offset；
  同时报告 `success_protocol`、plane miss、timeout、safety、speed、saturation；
- 先离线/回放审计，再 1-seed screen，再 40-seed dev formal；
- held-out `100–149` 继续冻结。

## 5. 引用核验状态

| 文献 | 核验方式 | 状态 |
|---|---|---|
| Ghosh & Mandal 2024 | Crossref 摘要 | verified |
| Huang et al. 2017 | Crossref 摘要 | verified |
| Yang et al. 2026 | arXiv 摘要 | verified |
| Miao et al. 2023 | Crossref 元数据（摘要为空） | title/DOI verified; abstract unconfirmed |
| Du et al. 2023 | Crossref 元数据 | title/DOI verified; abstract unconfirmed |
| Li et al. 2024 | arXiv 摘要 | verified |
| Verma et al. 2026 | arXiv 摘要 | verified |
| Li et al. 2016 | Crossref 摘要 | verified |
| Kim et al. 2016 | Crossref 元数据 | title/DOI verified; abstract unconfirmed |
| Li et al. 2023 | Crossref 摘要 | verified |

未核验摘要的条目只用于方向性背景，不作为具体数值或机制结论的依据。
