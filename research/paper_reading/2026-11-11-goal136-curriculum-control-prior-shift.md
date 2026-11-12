# Reading note：residual policy 的课程不能先改命令分布

日期：2026-11-11<br>
触发：Goal136 C0/C1 direct command 使冻结 BC 先验从 `0.90` 掉到 `0.00`。<br>
用途：为 Goal143 的 C0/C1 ALOS 控制修复提供外部依据，不作为已精读论文的定论。

## 1. 检索

使用 `paper-search` 检索了 2022–2026 年的 curriculum learning、behavior
cloning covariate shift、frozen-policy residual adaptation 相关论文。检索词包括：

- `curriculum learning reinforcement learning distribution mismatch`
- `action space mismatch residual policy fine-tuning`
- `teacher student behavior cloning distribution shift`

返回结果较杂，以下三篇与当前问题最相关；目前只读标题/摘要级信息，未下载全文，
因此只把它们当作 hypothesis 来源。

## 2. 相关论文

### Stable-BC: Controlling Covariate Shift With Stable Behavior Cloning

- 作者：Shaunak A. Mehta, Yusuf Umut Ciftci, Balamurugan Ramachandran 等；
- 年份：2025；
- 链接：OpenReview / arXiv 检索结果；
- 要点：BC 策略在部署分布偏离训练分布时会出现 covariate shift；稳定 BC 通过
  约束/数据覆盖降低这种偏移。
- 对当前的启示：C0/C1 把 ALOS command 换成随机 direct command，等价于在
  BC 先验的输入分布上制造 covariate shift；仅靠 residual PPO 不应该被假设为
  能自动吸收这个偏移。

### R-VLA: Residual Off-Policy Adaptation of a Frozen Vision-Language-Action Policy

- 作者：Srihith Mudakala, Arnav Patil；
- 年份：2026；
- 链接：OpenReview `jAz4zSLbmr`（RL4VLA）；
- 要点：在冻结基础策略上学习 residual，而不是重新学习整个策略；残差适配的
  前提是基础策略的观测/动作语义保持可比。
- 对当前的启示：如果 C0/C1 先改变 command 语义，residual 需要先补偿“基础策略
  已经不在分布内”，这会消耗本应用于跟踪/终端修正的样本预算。保留 ALOS 更符合
  residual adaptation 的假设。

### Combined Constraint on Behavior Cloning and Discriminator in Offline Reinforcement Learning

- 作者：Shunya Kidera, Kosuke Shintani, Toi Tsuneda 等；
- 年份：2024； venue：IEEE Access；
- 要点：BC 约束可以稳定 offline RL，但不能修复动作空间/命令空间本身的不匹配。
- 对当前的启示：继续加 BC 正则或 residual 约束，不如先把 C0/C1 的命令语义改回
  BC 先验覆盖的 ALOS。

## 3. 对 Goal136/Goal143 的直接结论

1. 当前 C0/C1 的 direct command 不是“更简单的课程”，而是对冻结 BC 先验的
   OOD 命令分布；
2. residual PPO 在 100k 内没有、也不应该被要求同时补偿命令分布偏移和跟踪误差；
3. 最小修复是保留 C0–C6 的难度轴，只把 C0/C1 的控制语义改回 ALOS；
4. 如果 Goal143 的 ALOS 臂仍不能通过 C0 Gate，才考虑重采 direct-command BC
   数据，或把 C0/C1 从课程中移除；在此之前不换 PPO/SAC 算法。

## 4. 下一步

- 预注册 Goal143：`C-full-alos`、`C-no-current-alos` 与 `C-hard` 同协议对照；
- 先跑 1M、dev40 的 C0 Gate，不直接开 5M；
- 若 C0 Gate 通过，再进入 screen；若失败，写新的 blocker，不扫 seed、不放宽阈值。

