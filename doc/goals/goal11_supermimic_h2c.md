# Goal11：SuperMimic 启发的水下低层鲁棒控制

记录日期：2026-11-04

状态：**提案，未开始实现。** 本文只记录评估、研究判断和执行计划，不修改现有训练与评测代码。

关联材料：

- 论文：`/home/bubble/水下机器人/research/supermimic.pdf`
- 参考实现：`https://github.com/RaraAlu/Supermimic`
- 现有低层复现：`doc/goals/goal9_odarrl_control_reproduction.md`
- 批量 Torch 后端：`doc/goals/goal10_batch_torch_remus100_report.md`
- 当前服务器验证：`doc/goals/goal9_pc_server_validation_plan.md`

> 本文同时承担项目背景、技术栈和执行计划三部分。真正进入代码实现前，如果团队要求严格拆成三份文档，再把第 1、2、3 节分别拆为独立的 background / tech-stack / execution-plan 文档。

## 1. 结论先行

**可以借鉴 SuperMimic，但不建议照搬“学习一个完整世界模型”这一层。**

原因是我们的低层 REMUS100 动力学本身已经是可微的 Torch 模型，而且已经验证可以批量前向和反向传播。再用同一个仿真器采数据、学习一个 surrogate world model，再在这个 surrogate 上训练策略，存在明显的循环自证：模型误差、数据覆盖和策略优化会互相放大，最后很难判断收益来自真实动力学还是来自 surrogate 的偏差。

更合适的主线是：

> **H2C-AUV：History-to-Control with Privilege-Separated Differentiable Rollouts**
>
> 用已知可微 REMUS100 做短时域 BPTT；actor 只接收真实可部署的传感器历史；真实海流、传感器故障和仿真上下文只进入训练动力学与 critic；rollout 起点按“传感器退化/执行器风险”采样。

这不是“再做一个 DADE”，也不是“再做一个 PPO 残差”。它把 SuperMimic 真正有价值的三件事迁移过来：

1. 短时域可微 rollout，而不是只依赖无模型的 PPO 采样；
2. critic 对 horizon 之外的回报做自举；
3. 噪声和仿真上下文放在动力学模型之外，避免模型学习不可预测的随机性。

预期原创性应定位为 **中等偏上的系统与机制贡献**，而不是宣称“首个水下世界模型”：

- 与 SuperMimic 的区别：不学习完整世界模型，使用已知可微动力学；从 humanoid 的单速率全身控制转到 AUV 的多速率传感器退化控制；
- 与 AquaWAM 的区别：AquaWAM 面向水下 embodied planning 的 world-action model，本文面向低层轨迹跟踪和可部署观测契约；
- 与 ODARRL 的区别：ODARRL 是 DADE + 冻结 BC 残差 PPO，本文不改 DADE 的地位，而是用可微 rollout 重新解决信用分配和鲁棒性问题；
- 与“可微 AUV 系统辨识”的区别：本文不把系统辨识本身当终点，而是把已知动力学、残差模型和策略学习放在同一个训练闭环里。

初步 prior-art 检索只能说明“目前没有看到完全相同的组合”，**不能据此宣称完全新颖**。正式写作前还需要做一次系统检索。

### 1.1 初步 prior-art 与仓库状态

截至 2026-11-04 的聚焦检索，最近的相关工作至少包括：

| 工作 | 主要问题 | 与本文的关系 |
|---|---|---|
| SuperMimic，arXiv `2610.28480` | 用可微世界模型的短时域想象训练 humanoid 全身控制 | 借鉴其 BPTT、critic 自举、上下文条件和噪声外置；不复用其完整世界模型设定 |
| AquaWAM，arXiv `2609.33299` | 水下 embodied agent 的 dynamics-aware world/action model | 更偏规划和动作生成；本文聚焦低层轨迹跟踪、可部署观测契约和可微动力学控制 |
| Differentiable Composite Approximation，arXiv `2606.19711` | 从海试数据做可微 AUV 操纵性建模与系统辨识 | 偏建模与辨识；本文把模型放进策略训练闭环，且只有在失配可测时才学习残差 |
| ODARRL，DOI `10.3390/jmse14161501` | DADE 双时域历史 + 冻结 BC 残差 PPO | 本文的直接对照；DADE 是候选历史编码器，不是必须保留的主方法 |

参考仓库 `RaraAlu/Supermimic` 是**独立复现**，不是作者原码，README 也说明部分实现来自第三方 MIT 代码。它适合用来核对训练循环、BPTT 和日志口径，不能把其中的超参数或工程假设直接当成论文结论。我们的方案应以论文方法为准，并对仓库实现做独立验证。

## 2. SuperMimic 到底做了什么

### 2.1 论文的核心机制

SuperMimic 的训练循环可以概括为：

1. 当前策略在仿真器中采集 transition，加入 replay buffer；
2. 用 replay buffer 中长度为 `k=8` 的短窗口拟合一个可微世界模型；
3. 冻结世界模型，从 replay buffer 的状态出发，想象 `H=32` 步 rollout；
4. 通过 BPTT 把想象回报的梯度反传到策略；
5. critic 估计 horizon 之外的回报，给短时域 rollout 做自举；
6. 世界模型条件于每个 episode 的仿真上下文 `ξ`，例如摩擦、标定偏差、质心；
7. 观测噪声和执行噪声在模型之外注入，世界模型学习无噪声的确定性动力学。

论文中最值得注意的不是“世界模型”四个字，而是这套分工：

- 世界模型负责可微的动力学、奖励、观测和终止预测；
- 策略只通过短时域想象获得解析梯度；
- critic 负责长时域信用分配；
- 噪声不被强行塞进确定性模型；
- 仿真上下文 `ξ` 让同一个模型覆盖不同域随机化条件。

### 2.2 可以直接迁移的部分

| SuperMimic 机制 | 在本项目中的对应做法 | 判断 |
|---|---|---|
| 短时域可微 rollout | 直接用 `BatchedRemus100Dynamics.step` 做 `H=8/16/32` 的 BPTT | 强烈推荐 |
| critic 对 horizon 外自举 | 保留 actor-critic，critic 估计 `H` 步之后的价值 | 推荐 |
| 仿真上下文 `ξ` | 海流、传感器延迟/失锁、执行器偏差、模型参数 | 推荐，但只能进训练动力学/critic |
| 噪声外置 | DVL/USBL/IMU 噪声和 dropout 在观测生成处注入，不进可微核心 | 推荐 |
| replay buffer 起始状态 | 从真实闭环 replay 中采样 BPTT 起点 | 推荐 |
| return-based reset sampling | 改成 risk-based reset sampling，优先采样高退化/高饱和状态 | 可作为原创点 |
| 学习完整世界模型 | 用已知可微 REMUS100 替代 | 不推荐先做 |
| 多技能共享世界模型 | 我们的任务不是 motion library，暂不需要 | 暂缓 |

### 2.3 不能直接照搬的部分

- **Humanoid 的刚体接触动力学和 AUV 的水动力差别很大。** SuperMimic 的世界模型需要处理接触、摩擦和关节约束；我们的 Fossen 模型、推进器、舵面和海流已经有较强的解析结构。
- **SuperMimic 的 50 Hz 单速率设定不适合我们的多速率传感器。** 我们的 IMU、DVL、USBL、海流估计可能分别是 50/10/1/0.2 Hz，直接把所有东西按 50 Hz 堆历史会制造大量零阶保持重复值。
- **SuperMimic 只在仿真中训练，没有真实机器人数据。** 我们的 sim-to-real 问题更突出，尤其是 DVL 失锁、USBL 延迟、海流估计滞后和推进器死区。
- **SuperMimic 的 `H=32` 在 50 Hz 下只有 0.64 s。** 海流时间尺度是 5–30 s，短 horizon 不可能直接学会慢变海流；必须依赖历史编码和 critic 自举，不能把“BPTT 32 步”误认为“学会了长时海流”。

## 3. 当前项目证据与真正的缺口

### 3.1 已经得到的负结果

当前最需要尊重的证据不是某个训练曲线，而是固定 seed 的闭环评测：

- C3 五个训练 seed、固定评估集：默认 `2.0 m` 成功率约 `0.6875`（no-DADE）对 `0.7000`（DADE）；严格 `1.25 m` 约 `0.3906` 对 `0.4000`；
- 逐 seed 仍有退化，不能写成 DADE 稳定全面胜出；
- C4 压力迁移：no-DADE 与 DADE 都约 `0.3156`，timeout 约 `0.6844`，DADE 没有提供鲁棒性增益；
- 普通 DADE 与 DADE-head 的历史消融只改变动作约 `0.0036%–0.0453%`，低于 `0.1%` 门禁，说明历史通路当前基本没有被策略有效使用。

这些结果说明：**问题不是“再多训练一点”就能自动解决，而是信用分配、观测捷径和鲁棒性目标没有对齐。**

### 3.2 环境和执行器边界

| 基线 | success | timeout | 3D 路径 RMSE | saturation |
|---|---:|---:|---:|---:|
| C2 mild teacher | `1.0000` | `0` | `1.8852 m` | `0.1812` |
| C2 mild BC-only | `1.0000` | `0` | `1.9591 m` | `0.1696` |
| C2 clean teacher | `1.0000` | `0` | `1.6231 m` | `0.0465` |
| C3 realistic teacher | `0.7344` | `0.2656` | `4.6164 m` | `0.0718` |
| C3 realistic BC-only | `0.6875` | `0.3125` | `5.3919 m` | `0.1311` |
| H3 teacher | `1.0000` | `0` | `1.7727 m` | `0.4919` |
| H3 BC-only | `0.8438` | `0.1562` | `9.9935 m` | `0.3892` |
| C4 teacher | `0.1875` | `0.8125` | `8.6205 m` | `0.1731` |

几个关键解释：

- H3 teacher 的 success 是 `1.0`，但 saturation 已经约 `0.49`，说明执行器长期接近边界；只看成功率会掩盖控制余量不足。
- C4 teacher 只有 `0.1875`，说明强扰动下经典教师本身也接近系统边界。若新策略没有超过这个参照，不能简单归因于“DADE 无效”。
- C3 teacher 的 saturation 只有 `0.0718`，但成功率只有 `0.7344`，更像是海流/路径/控制能力问题，而不是单纯饱和。

### 3.3 批量 Torch 后端已经具备可行性

Goal10 已验证：

- `4096×256`、26D、float64、compile default 的长训练端到端约 `267k env-steps/s`；
- 单 iteration 中位数约 `3.74 s`，GPU 峰值显存约 `424 MiB`；
- 动力学单步与 PVS 的数值一致性已经过验证；
- 4096 环境的主要瓶颈是 Python/kernel launch，不是 GPU 算力。

这意味着：**从吞吐上看，做短时域 BPTT 是可行的。** 真正要验证的是 BPTT 的显存、梯度稳定性和闭环收益，而不是“能不能跑 4096 环境”。

## 4. 研究问题与假设

### 4.1 研究问题

在真实可部署观测存在噪声、延迟、失锁和估计滞后的条件下，如何让 AUV 低层策略：

1. 不依赖真实海流等特权信息；
2. 真正使用历史信息，而不是只做当前帧残差映射；
3. 在强时变海流、DVL 失锁、USBL 延迟和高速度下保持成功率与安全；
4. 用比无模型 PPO 更少的交互和更稳定的信用分配完成训练。

### 4.2 核心假设

**H1：** 用已知可微动力学做短时域 BPTT，能给 actor 提供比 PPO 更直接的信用分配，减少残差策略在训练早期的系统性退化。

**H2：** actor 只使用可部署的多速率历史，而真实海流/故障只进入训练动力学与 critic，可以在不泄漏特权信息的前提下提高 realistic/C4 的鲁棒性。

**H3：** 按风险状态采样 BPTT 起点，比均匀采样 replay 状态更能改善失败尾部和 timeout，而不会牺牲 clean 条件性能。

**H4（可选）：** 只有在已知动力学与高保真 PVS/真实数据存在可测失配时，加入学习残差动力学才有价值；否则完整世界模型是冗余的。

### 4.3 可辩护的 delta

初步 delta 可以写成：

> Unlike SuperMimic, which learns a single-rate deterministic world model from simulator transitions and conditions it on full sim context, the proposed work uses known differentiable REMUS100 dynamics, exposes only deployable multi-rate sensor history to the actor, keeps true current and sensor faults inside training dynamics and the critic, and samples short BPTT windows around sensor-degradation and actuator-risk events for robust low-level AUV tracking.

这句话还没有经过系统 novelty proof，正式论文中必须补充更完整的 prior-art 对比。

## 5. H2C-AUV 方案

### 5.1 总体结构

```text
真实可部署观测（带噪声/延迟/失锁）
        │
        ├──> 多速率历史缓冲 ──> 历史编码器 ──> actor ──> 动作 a_t
        │                                              │
        │                                              ▼
        │                                  已知可微 REMUS100 动力学
        │                                   （训练时可用真实海流 ξ）
        │                                              │
        │                                              ▼
        │                                       H 步想象 rollout
        │                                              │
        │                                              ▼
        └──────────────> critic（可看特权上下文）<── 奖励/终止/价值自举
```

关键约束：

- actor 推理路径绝不接收真实海流、真实故障标签或仿真内部状态；
- critic 可以使用真实海流和故障上下文，但只在训练期存在；
- BPTT 只通过已知动力学和策略，不通过传感器噪声；
- 训练用 replay 起点必须来自真实闭环，而不是从固定初始状态反复想象。

### 5.2 三个核心机制

#### 机制 A：特权分离的可微 rollout

训练动力学可以使用：

- 真实海流 `current_ned`；
- 传感器延迟/失锁/野值状态；
- 执行器偏差、推进器死区、模型参数扰动；
- episode 级仿真上下文 `ξ`。

actor 只能使用：

- 当前可部署观测；
- 可部署历史；
- 历史中的 valid/age 元数据（是否进入主实验由消融决定）。

critic 可以使用：

- actor 观测；
- 真实海流；
- 故障上下文；
- 必要时使用真实状态。

这不是新发明“非对称 actor-critic”，但在水下低层控制和可微 rollout 的组合里是必须明确写出的契约。任何 actor 侧泄漏都必须让实验作废。

#### 机制 B：风险感知的 replay 起点采样

SuperMimic 使用 return-based reset sampling。我们可以把它改成更适合鲁棒控制的 risk-based sampling：

```text
score_i = w1 * 近期失败/终止
        + w2 * DVL 失锁或 USBL 延迟
        + w3 * 执行器接近饱和
        + w4 * 海流剪切/时变强度
        + w5 * 当前价值误差或高 TD 误差
```

从高 score 状态多采 BPTT 起点，但不完全放弃普通状态。这样做的目的不是“只训练难例”，而是让短 horizon 的梯度更多落在真正决定失败尾部的状态上。

这部分可以作为原创点，但必须做消融：

- 均匀采样；
- return-based 采样；
- risk-based 采样；
- risk + 均匀混合。

#### 机制 C：多速率历史编码

第一阶段先做 50 Hz 单速率，避免一开始把系统做复杂。第二阶段再引入：

- IMU/姿态：50 Hz；
- DVL：10 Hz 或按配置；
- USBL：1 Hz 或更低；
- 海流估计：0.2–1 Hz；
- 每个 token 带原生时间戳/age，而不是把重复值当新测量。

编码器可以是 GRU/TCN/Transformer；DADE 只作为其中一个候选，不作为不可替代的主方法。判断标准不是“模块是否先进”，而是：

- 历史消融是否改变动作超过门禁；
- 在 DVL 失锁/USBL 延迟下是否真的改善；
- 是否比单帧策略更好；
- 是否能在 PVS 闭环中复现。

### 5.3 BPTT 目标

对一个从 replay 采样的起点 `s_0`，用当前策略和已知动力学展开 `H` 步：

```text
a_h = π_θ(o_h, history_h)
s_{h+1} = f_known(s_h, a_h, ξ_h)
r_h = r(s_h, a_h, ξ_h)
c_{h+1} = c_h * (1 - d_{h+1})
```

优化目标可以写成：

```text
L_policy = -E[ Σ_{h=0}^{H-1} γ^h c_h r_h
               + γ^H c_H V_ψ(s_H, ξ_H) ]
```

其中：

- `V_ψ` 是 critic，负责 horizon 外自举；
- `d` 是终止/失败指示；
- `c` 是 survival weight；
- `ξ` 是训练期可见、actor 不可见的上下文；
- 噪声在观测生成处注入，不通过可微核心反传。

实际实现时先用短 horizon `H=8/16` 验证梯度，再尝试 `H=32`；不要一上来做 `H=128`。

### 5.4 为什么不先学完整世界模型

如果训练动力学和评估动力学是同一个 `BatchedRemus100Dynamics`，学习世界模型只是在拟合一个已知函数：

- 数据效率看似提高，但收益可能来自 surrogate 的平滑偏差；
- 评估时仍然回到原动力学，训练/评估不一致；
- 很难区分“策略更好”和“世界模型更适合当前策略”；
- 海流、传感器故障等随机性本来就不该由确定性模型学习。

只有满足下面任一条件，才进入“已知动力学 + 学习残差”：

1. 已知动力学与 PVS 高保真模型存在稳定、可测的轨迹误差；
2. 有真实海试数据，且能定位系统性模型偏差；
3. 残差模型在独立验证集上能降低多步预测误差，而不是只降低单步误差；
4. 残差模型的不确定性可以用来缩短 BPTT horizon 或拒绝高风险想象。

这时才做：

```text
f_hybrid(s, a, ξ) = f_known(s, a, ξ) + f_residual(s, a, ξ)
```

并用不确定性门控 horizon，避免在模型不可信的区域做长时域 BPTT。

### 5.5 原创性边界

建议在论文中明确区分：

- **不是原创：** 非对称 actor-critic、BPTT、Dyna、replay、海流随机化、传感器噪声注入、DADE 本身；
- **可以是原创：** 面向 AUV 低层控制的“已知可微动力学 + 特权分离 + 风险感知起点 + 多速率可部署历史”的组合；
- **需要实验证明：** 风险采样、多速率历史、可选残差模型是否真的带来闭环收益；
- **不能宣称：** “首个水下世界模型”“首个可微 AUV 控制”“DADE 被证明无效”。

## 6. 实验设计

### 6.1 对照臂

至少保留以下五组：

| 臂 | 说明 | 作用 |
|---|---|---|
| BC-only | 冻结 BC，不接 PPO | 学生基线 |
| PPO no-DADE | 当前无模型 PPO | 现有训练基线 |
| PPO DADE | 当前 DADE | 历史编码基线 |
| supervised residual + PPO | 教师残差监督热启动后 PPO | 当前正在探索的路线 |
| H2C-AUV | 本文主线 | 待验证 |

如果 DADE 仍然没有历史敏感性，不应为了“模块完整性”强行保留；可以把 DADE 降级为消融项。

### 6.2 环境与压力集

- C2 clean / mild：检查基本跟踪和低扰动；
- C3 realistic：主战场，检查海流和传感器退化；
- C4：强时变流、强剪切、DVL 失锁、USBL 野值/延迟；
- H3/H4：高速和执行器饱和边界；
- 未见海流剖面：训练分布外的时变/剪切组合；
- 传感器退化单项：只关 DVL、只加 USBL 延迟、只加海流估计滞后。

### 6.3 指标

正式结果至少报告：

- 默认 `2.0 m` 成功率；
- 严格 `1.25 m` 成功率；
- timeout 率；
- safety termination；
- 三维路径 RMSE；
- material saturation；
- 失败尾部（最差 10% episode）；
- DVL 失锁后的恢复时间；
- 海流估计误差；
- 历史消融的动作敏感度；
- 训练/推理时延。

成功率定义保持现有口径：一个评估 seed 对应一个 episode，`success_rate = mean(goal_terminations > 0)`；成功条件是真实位置到 goal 的三维距离不超过 `goal_radius_m`，默认 `2.0 m`，严格评测 `1.25 m`；最大 `2500` 步，50 Hz 下约 `50 s`。

### 6.4 seed 与预算

- 小规模筛查：1 个训练 seed，20 个固定评估 seed；
- 正式比较：至少 3 个训练 seed，64 个固定评估 seed；
- 训练 seed 与评估 seed 严格分离；
- 不使用训练 rollout success 代替闭环评估；
- 64 seed 时 1 个 episode 变化是 `1.5625%`，不要把 `1–2` 个 episode 的差异写成稳定收益。

### 6.5 计算预算建议

第一阶段不要直接上 4096 环境做 BPTT。建议：

| 阶段 | 环境数 | BPTT batch | horizon | 目的 |
|---|---:|---:|---:|---|
| P2a | 64–256 | 64 | 8 | 梯度和显存 |
| P2b | 256 | 128–256 | 16 | 稳定性 |
| P3 | 1024 | 256–512 | 32 | 小规模训练 |
| P4 | 4096 采集 + 512 BPTT | 512 | 32 | 正式训练 |

4096 环境用于采集，不等于 BPTT batch 必须是 4096。短时域 BPTT 的显存主要取决于 `batch × horizon × 状态维度`，应单独测。

## 7. 执行计划

### P0：立项与契约审计

- [ ] 确认本文的开放问题（见第 9 节）；
- [ ] 审计 actor 观测中是否仍存在真实海流、真实故障或仿真内部状态泄漏；
- [ ] 确认 PVS 是正式评估 ground truth，批量 Torch 只用于训练和预筛；
- [ ] 记录当前 C2/C3/H3/C4 基线与 checkpoint 哈希；
- [ ] 冻结本轮不修改 DADE 主路径，先做独立 H2C 模块。

通过条件：观测契约、评估口径、基线哈希三项没有歧义。

Git：

```bash
git status --short
git add doc/goals/goal11_supermimic_h2c.md
git diff --cached --stat
git commit -m "docs(goal11): add SuperMimic-inspired H2C research plan"
git push origin "$(git branch --show-current)"
```

### P1：离线 replay 与可微 rollout 适配

- [ ] 设计 replay 记录字段：`s_t, o_t, a_t, r_t, done, ξ_t, sensor_valid, sensor_age`；
- [ ] 复用现有 `BatchedRemus100Dynamics.step`，不复制一套动力学；
- [ ] 写独立 `pc_bptt.py`，只做短时域 rollout 和损失，不改现有 PPO 入口；
- [ ] 用 float64 做 `H=2` 的 autograd vs 有限差分检查；
- [ ] 检查 actor 历史输入和 critic 特权输入没有混线。

通过条件：短 horizon 梯度相对误差低于 `1e-4`；无 NaN/Inf；单元测试覆盖 batch、mask、reset 和终止。

Git：

```bash
git add src/auv_lowlevel/batch/pc_bptt.py tests/test_pc_bptt.py
git diff --cached --stat
git commit -m "feat(goal11): add differentiable short-horizon rollout scaffold"
git push origin "$(git branch --show-current)"
```

### P2：BPTT 可行性与性能门禁

- [ ] 在 CPU 和 GPU 上测 `B=64/128/256`、`H=8/16/32` 的显存、前向、反向和梯度范数；
- [ ] 记录 `θ` 的梯度是否有限，是否出现 horizon 增长导致的爆炸；
- [ ] 比较 `H=8/16/32` 的训练损失和闭环短评；
- [ ] 若 `H=32` 不稳定，先退到 `H=16`，不要盲目加 grad clip 掩盖问题；
- [ ] 记录 BPTT 更新耗时，要求不超过同规模 PPO 更新的可接受倍数。

通过条件：`B=256,H=32` 在目标 GPU 上无 OOM；梯度有限；单次更新耗时可接受；短 horizon 闭环不出现安全退化。

Git：

```bash
git add src/auv_lowlevel/batch/pc_bptt.py scripts/train_pc_bptt.py tests/test_pc_bptt.py
git diff --cached --stat
git commit -m "feat(goal11): add BPTT feasibility probe"
git push origin "$(git branch --show-current)"
```

### P3：H2C 小规模对照

- [ ] 在 C2 mild 和 C3 realistic 上跑 BC-only、PPO no-DADE、H2C 三个臂；
- [ ] 所有臂使用同一 actor 观测、同一动作包线、同一训练 seed；
- [ ] 先做 1 个训练 seed、20 个固定评估 seed；
- [ ] 检查成功率、timeout、RMSE、saturation、safety；
- [ ] 做 history-zero / current-zero / history-shuffle 消融，确认 actor 真的用了历史；
- [ ] 若 H2C 只在训练 reward 上更好、闭环不更好，判定为负结果。

通过条件：至少一个 realistic 条件相对 BC-only 或 PPO no-DADE 有可解释的提升，且没有安全/饱和退化；否则进入失败分析，不扩大训练。

Git：

```bash
git add configs/lowlevel/<新增配置>.json scripts/train_pc_bptt.py scripts/evaluate_pc_ppo.py
git diff --cached --stat
git commit -m "exp(goal11): add H2C small-scale comparison"
git push origin "$(git branch --show-current)"
```

### P4：三 seed 正式比较

- [ ] 固定 3 个训练 seed，BC/PPO/H2C 严格配对；
- [ ] 固定 64 个评估 seed，默认 `2.0 m` 与严格 `1.25 m` 都报告；
- [ ] 报告逐 seed 结果、均值、标准差和最差 seed；
- [ ] 加入 C4 压力迁移和至少一个未见海流剖面；
- [ ] 加入 DVL 失锁/USBL 延迟单项压力；
- [ ] 只在训练 seed 上选 checkpoint，不用评估 seed 反向挑选。

通过条件：至少 2/3 训练 seed 在 C3 realistic 上不差于基线，且至少 1 个压力条件有稳定改善；不能只看均值掩盖退化 seed。

Git：

```bash
git add configs/lowlevel/<新增配置>.json scripts/train_pc_bptt.py scripts/summarize_<实验>.py doc/goals/goal11_results.md
git diff --cached --stat
git commit -m "exp(goal11): add three-seed H2C evaluation"
git push origin "$(git branch --show-current)"
```

### P5：多速率历史与可选残差模型

- [ ] 只在 P4 有正收益后启动；
- [ ] 实现原生时间戳/age，避免把零阶保持的重复值当新测量；
- [ ] 比较 50 Hz 单速率、10 Hz DVL、1 Hz USBL、0.2–1 Hz 海流估计的多速率编码；
- [ ] 若已知动力学与 PVS 的多步误差稳定超阈值，再训练残差模型；
- [ ] 残差模型必须做独立验证集和不确定性门控；
- [ ] 如果多速率没有超过单速率，保留单速率主线，记录负结果。

通过条件：多速率或残差模型在至少一个传感器退化条件上带来闭环收益，且推理时延可接受。

Git：

```bash
git add src/auv_lowlevel/batch/<新增模块>.py tests/<新增测试>.py configs/lowlevel/<新增配置>.json
git diff --cached --stat
git commit -m "feat(goal11): add multi-rate history or residual dynamics"
git push origin "$(git branch --show-current)"
```

### P6：PVS 复核与报告

- [ ] 用 CPU PVS 做最终 64-seed 复核；
- [ ] 批量 Torch 预筛结果不得直接替代 PVS；
- [ ] 报告正结果、负结果、未完成项和系统边界；
- [ ] 给出“能否写论文”的明确判断；
- [ ] 若没有稳定收益，转为负结果报告，不继续堆训练。

通过条件：PVS 上至少一个主指标稳定为正，安全不退化；否则判定 H2C 当前配方不成立。

Git：

```bash
git add doc/goals/goal11_results.md artifacts/lowlevel/<结果目录>/summary.json
git diff --cached --stat
git commit -m "docs(goal11): add PVS validation and result report"
git push origin "$(git branch --show-current)"
```

## 8. 风险与停止条件

| 风险 | 早期信号 | 处理 |
|---|---|---|
| BPTT 梯度爆炸/消失 | grad norm 指数增长或 NaN | 降 horizon、检查归一化、必要时用 gradient clipping；不要先加长 horizon |
| 模型自证 | 训练 reward 好但 PVS 闭环差 | 停止扩大训练，回到真实 replay 和 PVS 评估 |
| 特权泄漏 | actor 对真实海流异常敏感 | 立即作废实验，修观测契约 |
| 历史无效 | history ablation 动作变化 < `0.1%` | 不保留 DADE 主方法地位，改编码器或任务目标 |
| 风险采样过拟合 | C4 好但 C2/C3 退化 | 混合均匀采样，报告分布内/外两侧 |
| 多速率实现过重 | 推理时延或训练吞吐明显下降 | 先保留单速率，把多速率作为后续工作 |
| 执行器饱和 | saturation 接近 0.5 且成功率不降 | 不能只报 success，必须报饱和和约束违反 |
| 评估噪声 | 64 seed 只差 1–2 个 episode | 不宣称稳定收益，扩大 seed 或改配对统计 |

**总停止条件：** 如果 P3 的小规模 H2C 在 C3 realistic 上连续两个 seed 都不如 BC-only/PPO no-DADE，且历史消融仍低于门禁，则停止 H2C 主线，转为负结果分析，不再扩大训练预算。

## 9. 需要确认的开放问题

1. 目标是偏“方法论文”还是“系统/实验论文”？如果偏方法，必须把风险采样或多速率历史做成可独立验证的机制；如果偏系统，可以更重视完整闭环和工程贡献。
2. 是否有真实海试数据或高保真 PVS 日志可用于模型失配分析？如果没有，暂不做学习残差世界模型。
3. 当前传感器配置是否已经能提供原生时间戳、valid/age 和多速率数据？如果不能，多速率阶段需要先补数据契约。
4. DADE 是否还值得保留为主方法？当前证据更支持“历史编码器可以保留，但 DADE 不是主线”。
5. 服务器训练窗口和 GPU 预算是否允许 3 个训练 seed × 多个臂？如果不够，先做 1 seed 的 P3，再决定是否扩。

在这些问题确认前，只执行 P0/P1，不启动长训练。

## 10. 最终建议

**建议立项，但按“小步验证”的方式立项。**

最值得做的不是“复刻 SuperMimic”，而是把它最重要的训练思想拆出来，和我们已经有的可微 REMUS100、真实可部署观测契约、PVS 评估体系结合：

1. 先用已知可微动力学做短时域 BPTT，验证它能否解决 PPO 残差的信用分配问题；
2. 再用风险感知起点采样，把训练重点放到 DVL 失锁、USBL 延迟、强剪切和高饱和状态；
3. 最后才考虑多速率历史和残差世界模型，而且必须用 PVS/真实数据证明模型失配存在。

如果 P3 不能给出正收益，应该果断停止，而不是继续加网络、加 horizon 或加训练步数。
