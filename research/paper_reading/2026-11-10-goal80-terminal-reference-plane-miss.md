# Reading note：AUV terminal homing、capture region 与 goal-plane termination

日期：2026-11-10<br>
关联：Goal80 `TERMINAL_REFERENCE_HURTS`<br>
检索：`paper-search`，2018–2026，query
`AUV terminal homing line-of-sight guidance goal plane crossing capture region`

## 1. 检索结论

检索到的主要工作把 AUV 的 terminal homing 分成两类：

1. **分阶段 homing/docking**：远处用 LOS/路径跟踪，近处切换到专用视觉、电磁
   或声学终端传感器；terminal 阶段的目标不是“把路径点推过终点”，而是获得
   更准的相对位置并切换到 capture/docking 控制。
2. **LOS/path-following**：CTE 或 LOS 主要解决路径跟踪和横流扰动，不直接定义
   tight capture 或 goal-plane 终止语义。

这与 Goal80 的结果一致：terminal reference 沿 tangent 前移改变了 heading/tracking，
但没有让载具进入 tight radius，反而更早触发 `GOAL_PLANE_MISS`。因此下一步应查
terminal capture/termination，而不是继续扫 reference 前移量。

## 2. 关键论文

### Zuo et al., 2021 — A Unified Approach for Underwater Homing and Docking of over-Actuated AUV

- Venue：*Journal of Marine Science and Engineering* 9(8):884；
- DOI：`10.3390/jmse9080884`；
- 方法：统一 task planning、guidance/control 和 thrust allocation；homing 早期用
  LOS 跟直线，之后跟多个预设 target points，进入 `10 m` 内再用 LED 视觉阵列
  做 terminal docking。
- 与 Goal80 的对应：真正 terminal 阶段用了不同于远场路径跟踪的测量和控制模式。
  仅把一个远场/路径参考点沿 tangent 推过 goal，并不等价于 terminal docking
  guidance，也没有改变 capture 或 termination 定义。
- 不能直接迁移：对方是 over-actuated AUV，且有近距离视觉和 QP thrust allocation；
  当前环境是 underactuated REMUS100、没有 terminal vision。

### Gutnik & Groper, 2023 — Terminal Phase Navigation for AUV Docking

- arXiv：`2311.13078`，https://arxiv.org/abs/2311.13078；
- 方法：用 magnetometer + docking station 内的电磁 beacon，实时估计 docking
  component 的三维相对位置；摘要报告平均定位精度低于 `3 cm`。
- 与 Goal80 的对应：terminal capture 需要足够准确的相对位置；当前 terminal
  `p_nav/eta` 偏差约米级，靠前移目标无法创造不存在的相对定位精度。
- 不能直接迁移：需要电磁 beacon、磁力计和专门 docking 硬件；当前只有 synthetic
  navigation 和既有 checkpoint。

### Ghosh & Mandal, 2024 — AUV way-point tracking at constant velocity: CTE and LOS guidance

- Venue：*Engineering Research Express*；DOI：`10.1088/2631-8695/ad36b0`；
- 方法：用 CTE 和 LOS 计算期望航向，结合 AUV 运动学/动力学做 2D 路径跟踪；
  结果主要评估直线、转弯、曲线和圆轨迹的跟踪误差与扰动鲁棒性。
- 与 Goal80 的对应：LOS/CTE 是 path-following 指标，不是 terminal true-capture
  指标；Goal80 中 heading tracking 改善而 true 变差，正说明二者不能混为一谈。
- 不能直接迁移：2D、underactuated 路径跟踪，没有三维 goal plane 和 dwell/guard。

### Lee, Ann, Cho et al., 2018 — Capturability of Guidance Laws for Interception of Nonmaneuvering Target with Field-of-View Limit

- Venue：*Journal of Guidance, Control, and Dynamics*；
  DOI：`10.2514/1.g003860`；
- 关注点：带视场限制的拦截制导可捕获性，强调 termination/capturability 条件，
  而不是只看瞬时跟踪误差。
- 与 Goal80 的对应：即使 heading error 下降，也不代表 capture 条件成立；应检查
  进入捕获域的几何条件和终止判据。
- 不能直接迁移：拦截问题与低速 AUV terminal docking 的动力学和目标语义不同。

## 3. 与当前失败模式的映射

| 观察 | 论文侧的解释 | 当前推断 |
|---|---|---|
| extension 改变 heading/tracking，但 true 下降 | LOS/CTE 成功不等于 capture 成功 | 追踪指标不是充分任务指标 |
| 17/18、8/10 新增丢失是 plane miss | terminal docking 通常有独立 capture/switch 逻辑 | 先查 plane-miss guard 与 tight capture 语义 |
| zero-residual 也 plane miss 增加 | terminal phase 不依赖 residual 也可失败 | 不是 NoisyNet/residual 主导 |
| 目标前移后更早穿面 | 推过终点的 reference 不等于 terminal sensor 对齐 | 停止扫 extension 值，转 termination/capture |

## 4. 最便宜的验证实验

使用 Goal80 的已有 `trace.npz` 做只读复盘，不需要新 rollout：

- 对每个 baseline-true / candidate-lost episode，找首次 `signed_offset >= 0`；
- 记录该时刻 `p_nav` 和 `eta` 到 goal 的距离、horizontal/vertical offset、
  `goal_plane_miss`、`guidance_terminal_phase`；
- 离线比较“当前 plane-miss guard”与“只在 tight radius 内终止/延迟 guard”的
  true、false-success、timeout 和 safety；
- 如果离线规则能恢复 true 且不增加 false-success/safety，再预注册单因素
  termination arm；否则查动作参数化或 controller bandwidth。

## 5. 不允许的直接迁移

- 不把 Zuo 的视觉 LED 或 Gutnik 的电磁 beacon 直接当作当前环境的可观测输入；
- 不因论文使用 terminal docking 就换算法或加传感器；
- 不扫 extension `0.5/2/4 m`、terminal window 或 reward 权重；
- 不重开 Goal27 的 residual/reward/gamma/seed；
- 不在没有 optimization/exploration 或 sample-efficiency 证据时启动 SAC/
  REDQ/DroQ。

## 6. 证据与链接

- Zuo et al. (2021), DOI `10.3390/jmse9080884`；
- Gutnik & Groper (2023), arXiv `2311.13078`；
- Ghosh & Mandal (2024), DOI `10.1088/2631-8695/ad36b0`；
- Lee et al. (2018), DOI `10.2514/1.g003860`；
- 本轮 `paper-search` 结果：arXiv=8、OpenAlex=8、Crossref=8，
  Semantic Scholar 因 HTTP 429 在重试后无结果；共 23 个 unique records。
