# Goal31 reading note：路径前视与终端平面收敛的差距

日期：2026-11-09<br>
触发：Goal31 formal FAIL，`terminal_eta_capture` 不足，terminal vertical offset
反而变差。<br>
检索：`paper-search`，arXiv 2018–2026；下列摘要通过 arXiv API 核验。

## 1. 检索问题

- 为什么“保留路径几何”的 ALOS/纯追踪能降低 path RMSE，却没有降低 terminal
  true-state capture？
- AUV docking / terminal homing 是否把终点阶段当成单独的估计与控制问题？
- 下一轮应优先做显式 terminal error feedback，还是先做 estimator/lag 诊断？

检索 query：

```text
AUV cross-track error line-of-sight path following
underwater vehicle pure pursuit lookahead path following
AUV terminal homing vertical plane guidance
AUV terminal docking path following guidance
```

## 2. 已核验论文

### Yang, Chavez-Galaviz & Mahmoudian (2026)

**An Adaptive Fixed-Time Line-of-Sight Guidance Scheme for 3D Path Following of
Underwater Vehicles: Theory and Experiment**，arXiv `2609.14756`。

- 传统 LOS 是渐近收敛；该文用 fixed-time stability + adaptive sideslip
  estimator + time-varying lookahead。
- 仿真中 time-varying lookahead 对 curved path 的 cross-track/vertical-track
  RMSE 分别降低 `69.37%/67.46%`；Iver 3 实航也显示路径误差下降。

对本项目的含义：time-varying lookahead 是**路径跟踪**层面的有效工具，但论文的
指标是 cross-track/vertical-track RMSE，不是 goal-plane terminal capture。
Goal31 的 path RMSE 改善与 terminal vertical offset 变差并不矛盾。

### Zieglmeier et al. (2025)

**Data-Enabled Predictive Control with Predictive Adaptive Line-of-Sight Guidance
for 3-D Path Following of Autonomous Underwater Vehicles**，arXiv `2510.25309`。

- 将 ALOS 扩成 predictive multistep（PALOS），在 REMUS 100 6-DoF 仿真中
  比 ALOS-PI/PID 降低约 `28%` cross-track error。
- 重点是路径跟踪和数据驱动预测控制，不是 terminal docking capture。

对本项目的含义：更强的前视/预测路径控制可以改善路径误差，但仍不能自动解决
terminal 阶段的相对位姿与平面穿越误差；Goal31 不能把 path 指标当成终点指标。

### Rober et al. (2022)

**3D Path Following and L1 Adaptive Control for Underwater Vehicles**，arXiv
`2204.00935`。

- 外层 LOS 产生 path-following command，内层用 L1 adaptive control 产生
  rudder，且给出外/内环性能边界。
- 结果强调路径跟踪是“guidance + inner-loop actuation”分层问题。

对本项目的含义：Goal31 只改外层 terminal reference，PD/PI 内环和状态估计未变；
若 terminal true-state 垂直偏差由内环滞后/估计误差主导，单改参考点会失败。

### Gutnik & Groper (2023)

**Terminal Phase Navigation for AUV Docking: An Innovative Electromagnetic
Approach**，arXiv `2311.13078`。

- 针对 AUV terminal guidance phase，使用磁力计 + 电磁 beacon + 信号处理，
  在不依赖 direct line-of-sight 的情况下做 3D localization。
- 仿真、实验室原型和全尺寸水池实验的平均定位精度低于 `3 cm`。

对本项目的含义：真实 terminal docking 往往需要专门的近距定位；用同一个
`p_nav` 路径状态同时承担 terminal capture 是不充分的。H2 的 estimator/nav
诊断有文献依据。

### Regalo, Cabecinhas & Pascoal (2026)

**Acoustic-based Guidance for Automatic Docking of Holonomic AUVs**，arXiv
`2609.11821`。

- 使用双 USBL、`SO(3)` 非线性互补滤波、EKF 相对位置估计和 `SE(3)` 几何控制；
- 在未知海流、低能见度条件下完成 simulation 和 water trials。

对本项目的含义：terminal docking 被明确拆成 relative pose estimation +
geometric control；Goal31 的“路径前视点”没有改变相对位姿估计，也没有给终点平面
提供显式误差反馈，因此不能期待它自动修正 true-state vertical offset。

## 3. 综合判断

1. **路径跟踪 ≠ 终端捕获。** LOS/ALOS、pure pursuit、predictive ALOS 的论文
   主要报告 cross-track/vertical-track RMSE；terminal capture 还取决于终点平面
   法向、相对位姿估计、内环带宽和最终速度/姿态。
2. **Goal31 的结果与文献一致。** candidate 的 path RMSE 改善，但 terminal
   vertical offset 变差；说明只优化路径参考点不足以修正终点误差。
3. **下一轮应先拆 H2/H3。** 用现有 trace 检查 `eta` vs `p_nav`、terminal
   stern/rudder、saturation 和 vertical offset 的关系；如果估计/滞后解释成立，
   不应继续增加路径前视或窗口。
4. **若进入显式 feedback，目标应是 goal-plane 误差。** 使用终点切向的法向/垂直
   轴构造 terminal LOS/vertical correction；仍保持 20 m 窗口和 B0 控制器，避免
   重开 Goal30 的窗口搜索。

## 4. 对下一 Goal 的约束

- 不继续扫 `terminal_path_preserving_enabled` 的 lookahead；
- 不重开 terminal window、reward、gamma/lambda、seed 或 residual 主线；
- 先做只读 H2/H3 诊断，再决定是否预注册显式 terminal error-feedback；
- 主指标仍是 `terminal_eta_capture` + horizontal/vertical offset；
- `success_eta` 只作 ever-capture 诊断，path RMSE 只作护栏/机制量；
- held-out `100–149` 继续冻结。

## 5. 引用核验

| 文献 | 核验方式 | 状态 |
|---|---|---|
| Yang et al. 2026, arXiv 2609.14756 | arXiv API 摘要 | verified |
| Zieglmeier et al. 2025, arXiv 2510.25309 | arXiv API 摘要 | verified |
| Rober et al. 2022, arXiv 2204.00935 | arXiv API 摘要 | verified |
| Gutnik & Groper 2023, arXiv 2311.13078 | arXiv API 摘要 | verified |
| Regalo et al. 2026, arXiv 2609.11821 | arXiv API 摘要 | verified |

本 note 只使用上述摘要和 Goal30 已核验 reading note 的方向性结论；没有把未核验
摘要当作具体数值依据。
