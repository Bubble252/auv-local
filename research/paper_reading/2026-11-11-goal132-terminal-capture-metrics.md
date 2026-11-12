# Reading note：terminal capture、waypoint 几何与评价分布

**日期**：2026-11-11<br>
**Goal**：Goal132<br>
**检索**：`paper-search`；query 覆盖 AUV path following、terminal waypoint
capture、capture radius、robot policy evaluation；年份 2015–2026；来源
arXiv / OpenAlex / OpenReview / Crossref。Semantic Scholar 返回 429，未纳入结论。

## 1. 最相关工作

### Deep Reinforcement Learning Controller for 3D Path-following and Collision Avoidance by Autonomous Underwater Vehicles

- 作者：Simen Theie Havenstrøm, Adil Rasheed, Omer San
- 年份：2020；arXiv:2006.09792
- 链接：http://arxiv.org/abs/2006.09792
- 可迁移点：AUV 的 DRL 工作通常以路径跟踪误差、碰撞与约束满足为主，
  很少把“固定 terminal bias”当作独立评价对象。
- 对本项目：Goal132 只能说明固定 bias 不足，不能外推为 DRL 在 AUV
  path following 上必然无效。

### Deep Interactive Reinforcement Learning for Path Following of Autonomous Underwater Vehicle

- 作者：Qilei Zhang, Jinying Lin, Qixin Sha 等
- 年份：2020；arXiv:2001.03359
- 链接：http://arxiv.org/abs/2001.03359
- 可迁移点：学习器通常与底层 controller/teacher 交互，学习的是控制修正，
  不是直接替代整个制导链。
- 对本项目：支持继续保留 ALOS/BC 作为 prior，但要求 learner 有足够状态和
  训练信号，而不是只加固定 bias。

### Successive Waypoints Tracking of an Underactuated Surface Vehicle

- 作者：Ning Wang, Hamid Reza Karimi
- 年份：2019；DOI:10.1109/TII.2019.2922823
- 链接：https://doi.org/10.1109/tii.2019.2922823
- 可迁移点：waypoint switching/capture 是独立几何问题；接近速度、进入方向
  和切换条件会共同决定是否“捕获”，不是单看路径误差。
- 对本项目：解释 Goal132 为什么单独加 rpm closure 不够；terminal capture
  需要进入顺序、姿态和速度的联合条件。

### Capturability of a Sliding-Mode Guidance Law With Finite-Time Convergence

- 作者：Kebo Li, Hyo-Sang Shin, Antonios Tsourdos
- 年份：2019；DOI:10.1109/TAES.2019.2948519
- 链接：https://doi.org/10.1109/TAES.2019.2948519
- 可迁移点：capturability 是制导律与执行约束共同决定的区域/条件，
  不能由单步 actuator bias 保证。
- 对本项目：支持把“进入 radius 后的条件成功率”和连续距离曲线作为诊断，
  而不是只报告一个固定 radius 的二元 success。

### Robot Learning as an Empirical Science: Best Practices for Policy Evaluation

- 作者：Hadas Kress-Gazit, Kunimatsu Hashimoto, Naveen Suresh Kuppuswamy 等
- 年份：2024；arXiv:2409.09491
- 链接：http://arxiv.org/abs/2409.09491
- 可迁移点：policy 评价应报告分布、方差和边界行为，而不是只给单一均值/成功率。
- 对本项目：Goal132 的 1–6 cm 边界翻转正好说明需要 near-radius 分布作为
  二级诊断；但预注册 success 仍保持不变。

## 2. 综合判断

检索结果没有支持“给 ALOS/BC 加固定 terminal bias 就能提高 success”的直接
证据。更一致的结论是：

```text
capture 是几何、速度、姿态和执行约束的联合条件；
固定 bias 只能改变其中一个自由度；
评价应同时报告二元 success 与 near-radius 连续分布。
```

## 3. 对 Goal133 的最小迁移

只做一件事：固定物理正确符号，测 vertical dose-response。若 vertical error
单调改善但 success 仍不动，就不再继续 eval-only bias 扫描，转入训练型或压力
测评问题。

不引入新 world model、不换 PPO/SAC、不改 success/radius、不复制论文的完整
训练管线。
