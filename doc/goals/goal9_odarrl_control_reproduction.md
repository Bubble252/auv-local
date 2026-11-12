# 任务：ODARRL 控制本体复现与我们的残差控制路线

记录日期：2026-10-30

更新日期：2026-11-05（新增最高优先级的 ABC 前置 P0/P1/P2/P3 与 A/B/C 时间感受野、决策频率和 Transformer actor 实验组；P0/P1/P2/P3 只做前置筛查，不替代正式 ABC。它只新增实验门禁，不覆盖历史结果。2026-11-02 的 PC 臂优先规则继续有效：PC v1 使用合成传感器，Isaac Sim/OceanSim 原生传感器后置且不阻塞主实验；新 PC 实验使用 `±20°` 舵/艉水平舵包线，旧 I/O/R 保持 `±15°`；保留 2026-11-01 的 hidden-current 契约修复、seed-matched 配对、时变/空间剪切海流结果、独立特权 critic 和 DADE history ablation）

状态：**G0 已完成；I 臂为优先主链；G1 服务器 B0 50-seed、I 臂 255k 教师采集和 27D I0 50-seed 门禁已完成；r1/r2 未通过 I1，r3 的 RPM-only 残差在两组 50-seed、2.0 m/1.25 m 上通过跟踪与执行器门禁，I1-26D 可进入 I2；r4 在 model_20 主动停止，默认不再续训，保留为 27D 历史结果和可选迁移 teacher。goal10 已完成多环境基础设施、26D 批量 pilot、P3 直接 26D 门禁、P4 长训练、4096/1024/256 等预算规模对照和最终报告。P4 是 O 臂 residual-only 预检，`model_25` 通过 50-seed 控制门禁，但相对 B0 没有精度收益；它不是 I1/I2。此后新实验默认使用 26D；27D 只保留给旧 checkpoint、历史数据、兼容回归和可选迁移 teacher。BC/DAgger/动作蒸馏不是 26D 的前置条件，只在需要复用旧 27D 能力时使用。I1-26D r1 学到近似与海流无关的常值残差；r2 的全三通道监督热启动/PPO 有跟踪收益但执行器退化；r3 通过通道消融确认应只保留 RPM 残差，canonical 配置已固定。**

海流契约与配对复跑状态（2026-11-01）：最终 actor 的 hidden 契约已经落地为启动门禁。`speed_feedback='ground'` 使索引 11/12 不再包含真实海流，索引 17–19 恒零，ALOS 与教师也不得消费真实海流；`water_surge`/`speed_error` 的单帧反推路径被回归测试和 `_validate_current_visibility()` 同时封堵。独立特权 critic 已改为 `[hidden 26D, true current 3D]` 的分离 MLP，actor 布局不变；正式训练为负结果，不能写成收益。DADE 的 seed-matched 2×2 已完成：legacy 在 seed `62000` 为 `+0.089 m`、seed `72000` 变成 `-0.441 m` 且成功率 `0.78→0.70`；hidden 在 seed `62000` 为 `+0.676 m`、默认成功率 `0.84→0.82`，seed `72000` 只有 `+0.051 m` 且共同成功 `-0.307 m`。因此当前首要结论不是“移除泄漏后 DADE 变好”，而是 **PPO/DADE 跨 seed 严重不稳定；契约修复是观测合法性的必要修复，但不是性能增益来源**。快速时变流下 DADE 四格总 RMSE 和成功率均不降，是目前最强的性能证据，但共同成功子集仍约 `-0.01 m`，优势主要是失败尾部恢复。空间剪切流零样本迁移仍以失败尾部收益为主：严格半径成功率 `+0.04`，但默认 `0–49` 损失 `0.02`。随后在同一 8 s 快速时变+剪切场上继续训练 `4096×256×40`，`model_39` 把四格总 RMSE 改善提高到 `+0.401/+1.087/+0.445/+0.976 m`，但默认 `0–49` 成功率仍为 `0.82→0.80`；因此“训练分布外”只能解释一部分退化，不能解释默认成功率缺口。**推理期 history ablation 进一步表明：将短/长历史全部替换为当前帧几乎不改变结果，剪切场训练后仍然如此，因此当前 DADE 性能不能解释为“从历史推断海流”；它更像当前观测上的残差映射。**

26D I0 状态（2026-10-31）：**已通过正式门禁，可作为 I1-26D 的冻结基础策略。**直采 26D 教师数据和单次 BC 已分别完成，但单次 BC 闭环发散；加入一轮 40-episode DAgger 后，50-seed teacher-student 三维 RMSE 差 `0.03751 m`、成功率下降 `0.0`、student material saturation `0.00572`，B0 对照三维 RMSE 改善 `0.12218 m`。1.25 m 严格捕获率为 teacher `0.92`、student `0.94`。checkpoint、数据/训练 lineage、80 个 rollout 文件的有限性审计全部通过。

观测 schema 决策（2026-10-31）：**长期统一为 26D。goal10 的 `BatchTorchVecEnv`、配置驱动的 `num_obs`、GPU rollout 和 rsl_rl 适配等多环境基础设施已经完成；新实验默认直接使用 26D。现有 27D v1 只保留给已训练 checkpoint、历史数据、迁移 teacher 和前后端回归，不再作为新实验默认，也不再启动新的 27D 长训练。若需要复用旧 27D 策略的能力，使用 BC/DAgger/动作蒸馏迁移到 26D；26D 不要求先构造 27D teacher，也可以从在线 PD/PI 基座或从零训练。**

26D 执行状态（2026-10-31）：批量 Torch 的 `4096×256` 10 iteration pilot、canonical 配置、CPU PVS 50-seed 复核、P4 长训练和等预算规模对照均已完成。P3 pilot 的 26D residual 与 B0 同 seed 对比：成功率均 `50/50 = 1.00`；三维 RMSE 为 B0 `1.514748 m`、residual `1.515458 m`，劣化 `0.000710 m`；material saturation 均为 `0`；控制环 p95 延迟均值约 `9.42 ms`。原先 `325.48` 次/episode 的 constraint violations 已拆解为 `15,559` 次幅值微裁剪和 `715` 次速率微裁剪，最大舵角请求偏差 `0.0502 deg`、RPM 偏差 `0`，是基座靠近 15° 上限时的亚材料级裁剪，不是部署级饱和。P4 O 臂预检的 `model_25` 50-seed 三维 RMSE 差 `+0.00804 m`、成功率 `1.00`、material saturation `0`；4096/1024/256 等预算 `model_100` 的 20-seed 三维 RMSE 差分别为 `+0.01804/+0.02265/+0.03229 m`。4096 吞吐最高、质量最接近 B0，但所有正式候选都没有超过 B0。P3/P4 门禁均通过；新实验直接使用 26D，但 O1/O2 要先做奖励/超参优化，I2 要先有正收益的 I1。若要追求控制增益，应另开奖励、PPO 超参、探索和 early-stop 实验。

26D canonical 配置（2026-10-31）：环境侧 `configs/lowlevel/odarrl_control_v2_26d.json`（canonical hash `c19a5867…0f72e7`，与 base 仅 `observation.dim` 不同）；批量训练侧 `configs/lowlevel/odarrl_control_batch4096_v2_26d.json`，显式指向该环境配置。历史 `odarrl_control_batch4096_v2.json` 和已训练 checkpoint 保留原哈希，不回写；后续新实验不得再用“v2 schema + 27D base 文件”的隐含组合。

I1-26D 首轮状态（2026-10-31）：`i1_residual_ppo_26d_r1` 从严格零残差开始，但 50-seed 结果没有正收益；`model_10` 三维 RMSE 差约 `+0.00338 m`，残余动作约 `0.00492`，material saturation 从 `0.00440` 升到 `0.00440`（any-saturation 从 `0.00449` 升到 `0.12505`），约束违反从 `10.56` 升到 `218.72`。无海流审计中残差幅值几乎不下降，说明它没有学到“海流条件化”的动作映射，而是近似常值偏置。详见 14.4.4。

I1-26D 第二轮状态（2026-10-31）：`i1_residual_ppo_26d_r2_warmstart` 先用 B0 current-aware 教师数据监督热启动、再做保守 PPO 微调。50-seed 有海流三维 RMSE `3.344359 → 3.069486 m`（改善 `0.274873 m`），成功率 `0.78 → 0.84`；但 material saturation `0.004400 → 0.022956`、any-saturation `0.004488 → 0.128099`、约束违反 `10.56 → 203.24`。跟踪收益是真实的，执行器代价也是真实的；按门禁，r2 **未通过**。详见 14.4.5。

I1-26D 第三轮状态（2026-10-31）：r3 不再让三个执行器同时接收残差，而是做通道消融。rudder/stern 残差是饱和和约束违反的主要来源，RPM 残差没有 material 裁剪。canonical `configs/lowlevel/odarrl_control_i_residual_26d_rpm_only.json` + `model_10.pt` + `residual_gains=[0,0,1]` 在 `0–49` 与独立 `50–99` 两组 seeds、2.0 m 和 1.25 m 下都有正跟踪收益，成功率和执行器指标保持近中性。I1-26D 门禁通过，I2 从该 RPM-only 配置启动；全三通道 r2 checkpoint 不得用于 DADE。详见 14.4.6。

I2-26D 历史栈基线状态（2026-10-31）：显式短/长历史栈 `BatchTorchHistoryVecEnv` 已完成 4096×256×40 正式训练和两组 checkpoint 评估。布局为 `[当前 26D, 短窗 10×26, 长窗 50×26]`，策略输入 `1586D`。canonical `model_20` 在 `0–49`/`50–99`、2.0 m/1.25 m 四个有海流条件上相对冻结 I0 的三维 RMSE 改善分别为 `0.147981/0.305760/0.067704/0.169353 m`，均高于 I1 的 `0.080261/0.109863/0.030546/0.049493 m`；成功率不降，material saturation 最大增量 `+0.000148`，无海流残差幅值 `0.004443` 对 `0.015280`，近中性和条件化均成立。这是 **I2-a 显式历史栈基线，不是 DADE**，不能把增益写成 DADE 收益。详见 14.4.7；该节是旧 `water_relative` 泄漏契约，hidden 结论见 14.4.9。

I2-26D DADE 状态（2026-10-31）：完整 Transformer 双时间尺度编码器在修复热启动死梯度后完成 `4096×256×40` 正式训练。两个 checkpoint 在四个有海流条件上均略优于 I1，但显著弱于 I2-a 显式历史栈，未通过“相对 I2-a 不退化”的强基线门禁。10Hz 抽样时间窗消融在四个有海流条件上进一步低于 50Hz DADE。I2-a 仍是当前最佳 I2 版本；DADE 保留为负结果，不能写成通过，也不能把 I2-a 的收益算到 DADE 头上。详见 14.4.8；该节是旧 `water_relative` 泄漏契约，hidden 结论见 14.4.9。

观测契约泄漏修复状态（2026-11-01）：审计发现旧 I 臂的 `water_surge` 与 `speed_error` 仍使用真实海流；因为同一帧还有 `linear_x`，可以精确恢复 `current_body_x`。因此 **旧 I0/I1/I2/DADE 结果必须重标注为 water-relative-feedback、单帧可恢复海流的实验，不能作为“隐藏海流推断”的证据**。现已加入 `observation.speed_feedback`（`water_relative`/`ground`）和 `observation.current_contract`（`legacy`/`hidden`/`oracle`），hidden 契约下索引 11/12 只用地速，真实海流只进 reward、诊断、教师（若显式允许）或辅助目标。旧 26D `pi0` 的训练数据在 `current.enabled=false` 下采集，地速与对水速度数值相同，因此生成只补契约元数据、不改权重的 `pi0_hidden_ground.pt` 用于 hidden 重跑。hidden 重跑已完成：I1 基本没有可重复正收益；I2-a 仍有小幅但方向一致的四格正收益；DADE 的 8-seed 屏幕看起来很强，但 50-seed 留出集默认条件退化和成功率下降，未通过强基线门禁；时变海流的 I2-a 四格均通过；时变海流 DADE 的 RMSE、留出集和无海流方向翻转均通过，但严格 `0–49` 成功率仍下降 `0.04`，scale `0.8/0.9` 消融未修复。terminal capture 奖励复跑把严格 `0–49` 成功率缺口缩小到 `0.02`、RMSE 改善扩大到约 `0.795 m`，但仍未通过；终端包络仍无效。独立特权 critic 通路已实现，4096 正式训练已完成；但原始版和保守版在默认时变海流条件下都显著退化，不能写成性能收益。详见 14.4.9。

### Goal10 结论对 Goal9 的硬约束（2026-10-31）

Goal10 已完成，但它的结果是“后端和 26D 可行性完成”，不是“残差 PPO 已经优于传统控制器”。Goal9 后续必须按以下约束执行：

1. **训练后端已经切换**：新训练默认使用 `BatchTorchVecEnv` 的 `4096×256`、float64 物理和 26D schema；CPU PVS 只负责正式 50-seed 评估和保真复核，不再用 4096 个 PVS 进程训练。
2. **P4 不是 I 臂结果**：P4 配置中 `experiment.arm="I"` 是历史标签，但 `base_policy.mode="online_pd_pi"`，语义上是 **O 臂的 residual-only 预检**，不是冻结 `pi0` 的 I1。不得把 P4 的 `model_25` 写成 Controller-IL 或 I 臂成果。历史配置不改，以免破坏哈希；后续配置必须让 `arm` 与 `base_policy.mode` 一致。
3. **O 臂先验是负结果**：P4 的 26D online-PD/PI 残差策略成功率通过，但 50-seed 三维 RMSE 差 `+0.00804 m`，没有正收益。因此在没有修复奖励、PPO 超参、探索和 early-stop 之前，不应直接启动 O1/O2 的编码器阶段。
4. **I 臂已经进入 26D**：2026-10-31 已用同一在线 PD/PI teacher 在 26D 观测下完成教师采集、单次 BC 负结果、一轮 DAgger 和 50-seed 门禁。正式基础策略是 `artifacts/lowlevel/i0_26d_bc_dagger_r1_lineage_20261031/pi0.pt`，后续 I1-26D 必须冻结它。旧 27D I0/r1–r4 仍保留为 v1 历史；本次没有使用 27D 删列热启动，也没有把 P4 的 O 臂 residual 网络当作初始化。
5. **r4 不再续训为默认动作**：r4 是 27D 兼容产物，保留为历史结果和可选迁移 teacher。除非明确要补全 27D 对照，否则不继续 r4；新工作优先把 26D I0 做出来。
6. **正式门禁不变**：所有新策略仍以 CPU PVS、同训练 seed、50 个固定评估 seed 为准；成功率下降不超过 `0.02`、三维 RMSE 劣化不超过 `0.05 m`、material saturation 不上升，才允许进入下一阶段。提高训练速度不能替代闭环评估。

论文来源：`research/ODARRL.pdf`

论文题目：*ODARRL: Obstacle- and Disturbance-Aware End-to-End Residual Reinforcement Learning for Underwater Robot Trajectory Tracking with Obstacle Avoidance*

本地工作目录：`/home/bubble/水下机器人`

本文件只覆盖 **控制本体**：三维轨迹跟踪、海流扰动、执行器级控制、MPC/PD 基线和残差 PPO。**不覆盖论文第三阶段的障碍物避障、声呐感知和避障奖励。**

编号说明：本文件的 G0–G7 是 ODARRL 控制复现路线自己的阶段门，不与 `goal8_ppo_isaaclab.md` 的 G0–G8 混用。

## 0. 当前状态与优先级

> **2026-11-02 最高优先级路线调整（只新增，不覆盖历史）：** 后续先做 **PC 臂：ALOS 外置 + 纯控制器策略 + actor/critic 分离状态编码器 + 分离 DADE**。ALOS 不再蒸馏进网络，PC actor 也不再是冻结 BC/PD/PI 上的残差策略。旧 I/O/R 臂、26D schema、27D 兼容结果和 14.4 的全部 DADE 正负结果继续按原文保留，不改写、不删除、不重跑。

用户已确认：**先做 I 臂；GPU/CPU 富余时再并行 O 臂；R 臂作为论文 MPC 复现对照，不阻塞 I 臂。**

```text
已完成：G0 文档冻结
已完成：G1 JSON schema、路径、海流、27D v1 兼容观测（长期统一 26D）、动作限幅、奖励、ShadowSpeedIntegrator
已完成：服务器 B0 正式 50-seed 基线（终端捕获 + 2.0 m 到达半径）
已完成：服务器 I 臂 180 episodes / 255,467 样本无海流教师采集
已完成：I 臂 PD/PI 教师采集脚本、BC 训练脚本、I0 评估门禁脚本
已完成：单次 BC 负结果、两轮 DAgger、27D I0 50-seed teacher-student 门禁通过
已完成：26D 教师数据、单次 BC 负结果、一轮 DAgger、26D I0 50-seed 门禁、B0 对照、1.25 m 严格捕获和完整性审计
已完成：goal10 多环境基础、26D canonical 配置、26D checkpoint metadata/schema 校验、批量 pilot 和 PVS 50-seed 门禁
已完成：26D 微裁剪诊断；新实验默认 schema 固定为 26D
已完成：goal10 P4 长训练、候选择优、4096/1024/256 等预算规模对照和最终报告；结论是后端可用但 PPO 没有正控制收益
已决定：r4 默认不续训，保留为 27D 历史结果/可选迁移 teacher
待执行：I1-26D → I2-26D；O1/O2 暂缓，先做奖励/超参/early-stop；若复用旧 27D 能力，再做可选 BC/DAgger/蒸馏 student
最高优先级（2026-11-05 起）：先做 14.4.18 的 ABC 前置 P0/P1/P2/P3，再做 A/B/C 时间感受野、决策频率与 Transformer actor 组，最后做 14.4.19 的 ALOS-free V0/V1/V1b/V2/V3；顺序为 P0→P1→P2→P3→A→B→C→V，旧 I/O/R 和新的 DADE/DAgger 轮次让路
口径例外（2026-11-02 起）：PC 臂使用独立的 encoder-native schema，不套用旧 26D observation metadata；旧 26D 仍适用于 I/O/R 历史链路
```

### 0.1 最高优先级新增实验：ALOS 外置的纯控制器策略（PC 臂，2026-11-02）

> **PC 最新设计规则（2026-11-02，仅适用于本节的新 PC 臂）**：海流估计 `v̂_c` 只以**历史序列**形式进入 actor DADE；不进入 actor 的编码器输入 `o_t^a`，因此也不进入 `z_t^a = f_enc(o_t^a)`，也不进入 `c_t`。海流真值 `v_c` 只以**历史序列**形式进入 critic DADE；不作为 critic 的直接拼接项。actor 的 `o_t^a` 和 `c_t` 除移除海流估计外保持原样。旧 I/O/R 实验的提示词、配置和结果不受本条影响。

> **实验顺序约束（2026-11-02）**：PC 主实验先使用**不带 `valid/age`** 的 DADE 输入；`valid/age` 只记录用于诊断，不进入策略。等不带 `valid/age` 的 PC 主实验基本完成后再做 `valid/age` 消融，且该消融不得阻塞、抢占或改写主实验。触发条件建议为主实验 C0–C4、BC 热启动对照、`d_z`/学习率 screen 和 5-seed 正式评估均已完成或只剩最终汇总。

#### 0.1.1 实验定位

PC（Pure Controller）臂回答的问题是：**当 ALOS 继续负责制导、actor 直接负责执行器控制时，带特权 critic 和 DADE 的 PPO 能否在真实传感器误差下稳定跟踪？**

与旧 I/O/R 臂的区别：

- ALOS 始终是外置制导层，不把 ALOS 蒸馏进 actor，也不让 actor 复现 ALOS 的中间量；
- ALOS 固定关闭海流补偿：`guidance.compensate_current=false`，`u_d` 始终是地速命令，不使用真实或估计海流；
- actor 是纯控制器，部署时直接输出 `[rudder, stern, rpm]`，不是残差，也不冻结 BC 基座；允许用 PD/PI 做一次控制器 BC 热启动，但热启动后 actor 仍直接输出执行器；
- actor 与 critic 使用**不同输入、不同编码器、不同 DADE**；
- actor 的编码器输入 `o_t^a` 和 `c_t` 不接收海流估计；海流估计 `v̂_c^b` 只进入 actor DADE 的历史输入；
- critic 的编码器输入 `o_t^c` 使用绝对理想 6-DOF 位姿、体速度和参考误差，`c_t` 与 actor 命令同源；critic 的 `o_t^c` 和 `c_t` 不直接接收真实海流，真实海流 `v_c^n` 只进入 critic DADE 的历史输入；
- 顶层接口固定为 `[z_t, c_t, ĉ_t]`。以后课程学习增加低维状态时复用编码器预留槽位，增加高维状态时走独立编码分支；actor/critic 顶层结构不变。

本轮已确认、后续实现不得擅改的 PC 口径：

- 全链路始终使用 DVL 底跟踪**地速**，`u_d` 是地速命令；不允许把 actor 改成水相对速度控制；
- PC v1 的全部 actor/critic 传感器观测使用**合成传感器链**（真值 → 有状态误差/延迟/失锁/采样模型）；Isaac Sim/OceanSim 原生 DVL、IMU、气压计和声呐统一后置，不作为主训练依赖、实验门禁或当前实现项；未来只作为独立的“原生 vs 合成”后置对照；
- C2–C4 的 ALOS 路径投影、`P_ref`、前视参考、终端捕获和 `c_t` 生成全部使用与 actor 同源的 `p_nav`；不得读取真实位置 `p_true`/精确 `eta`。critic 和评估仍可使用真值；
- ALOS 继续保持 `guidance.compensate_current=false`，既不读取真实海流，也不读取海流估计；
- 课程从**无 ALOS 的直接命令跟踪**开始，学会 `ψ_d, θ_d, u_d` 跟踪后，再切换到 ALOS 路径跟踪环境适应；
- 允许 PD/PI 行为克隆（BC）做一次热启动，但 BC 只改变初始化；actor 不冻结、不做残差、不蒸馏 ALOS，热启动后仍直接输出 `[δ_r, δ_s, n]`。
- 自 2026-11-02 起，所有新 PC 实验的舵角和艉水平舵角上限统一为 `±20°`；这是 PC 专属执行器包线，不回溯修改旧 I/O/R 的 `±15°` 契约、checkpoint 和结果。
- PC 只使用稀疏安全终止：普通命令误差、动作饱和、DVL 失锁和 USBL 野值不终止；`30°` 滚转是强惩罚/质量阈值，不是立即失败，只有持续 `45°/1.0 s` 或 `90°/s/0.5 s` 才进入安全终止候选，具体按 0.1.9.1 标定。
- actor/critic 编码器各预留少量恒零槽位，并保存 `active_mask`：首版默认 8 个输入保留槽、4 个 `z_t` 输出保留槽；这只保证顶层宽度稳定，不表示新增观测可零样本生效。
- NP3O 只作为 PC 主实验完成后的可选安全 RL 消融，参考 TITA 的 cost-critic 实现；它不提高违禁事件触达率，也不阻塞主实验，具体见 0.1.12.1。

这条路线不推翻旧 26D 结论，而是新增一个实验臂。旧 26D I/O/R checkpoint、旧 DADE 结果和 27D 兼容链路继续有效；PC 的观测 schema 与它们分开命名，不能把 PC 配置误标成 26D。

推荐链路：

```text
Trajectory
  → 外置 3D ALOS
  → c_t = [ψ_d, θ_d, u_d]

真实传感器 o_t^a → actor encoder → z_t^a ┐
                    c_t                    ├→ actor → a_t → 执行器 → PVS
actor DADE(o_t^a 历史, v̂_c^b 历史) → ĉ_t^a ┘

理想状态 o_t^c → critic encoder → z_t^c ┐
                c_t                       ├→ critic → V(s_t^c)
critic DADE(o_t^c 历史, v_c^n 历史) → ĉ_t^c ┘
```

#### 0.1.2 统一输入契约

actor 和 critic 都使用同一种顶层形式：

```text
ŝ_t = [z_t, c_t, ĉ_t]
```

其中：

- `z_t = f_enc(o_t)` 是编码后的状态特征，维度记为 `d_z`；actor 与 critic 各自独立编码，不共享 `f_enc`；
- `c_t` 是 ALOS 给出的命令，不是当前状态；
- `ĉ_t` 是各自 DADE 输出的 64 维扰动特征，actor 与 critic 不共享 DADE 权重和历史缓冲。

命令先归一化再进入网络：

```text
c_t = [
  wrap(ψ_d) / π,
  θ_d / (π/2),
  u_d / 2.5
]
```

最终输入宽度固定为：

```text
actor:  d_z + 3 + 64
critic: d_z + 3 + 64
```

海流不进入 actor 的 `o_t^a` 或 `c_t`，也不进入 critic 的 `o_t^c` 或 `c_t`；因此 `z_t^a` 和 `z_t^c` 也不含海流。actor 的海流估计只进入 actor DADE 的历史，critic 的海流真值只进入 critic DADE 的历史。以后新增低维课程状态时，优先复用 0.1.10.1 的编码器预留槽；高维状态走独立编码分支并投影到保留 `z_t` 槽，actor/critic 顶层宽度始终不变。`d_z` 和学习率都不是先验结论，按 0.1.10 搜索。

所有进入编码器、DADE、actor 和 critic 的量必须先做固定尺度归一化。归一化参数写入 checkpoint metadata，部署时按同一参数执行；不能做以 episode 或 batch 统计量为基础的在线 min-max 归一化，否则部署和训练会不一致。

#### 0.1.3 Actor 编码前观测 `o_t^a`

首版 actor 编码前状态为 12 维，不包含海流估计：

```text
o_t^a = [
  x, y, z,                         # USBL/INS/DVL 融合后的位置，NED，m
  φ, θ, ψ,                         # IMU/罗经姿态，rad
  p, q, r,                         # IMU 角速度，rad/s
  u, v, w,                         # 艇体系地速 v_g^b，m/s
]
```

这里的 `u,v,w` 首版定义为艇体系**地速** `v_g^b = [u,v,w]`，直接来自 DVL 底跟踪，不表示仿真真值，也不使用海流估计重建。底跟踪失锁时保持最后有效地速并记录失锁状态，不能为了补齐 `o_t^a` 而用 `v_r^b + v̂_c^b` 反推，否则海流估计会间接泄漏进 encoder。海流估计只作为 DADE 历史侧输入，不进入 `o_t^a`；水相对速度和水跟踪值暂不默认加入 DADE 契约，若后续要使用，必须单独扩展 `H_t^a` 并重新做泄漏审计。因此 actor 的编码器输入在正常情况下仍能拿到地速，但不能默认总能拿到理想水相对速度。

PC 主实验（`v1-main`）的 actor DADE 每个历史时刻使用：

```text
[o_t^a, v̂_cx^b, v̂_cy^b, v̂_cz^b]
```

主实验不输入 `valid/age`：失锁和延迟仍由传感器 wrapper 仿真，零阶保持后的海流值直接进入 DADE；`valid/age` 只写入 rollout 日志用于诊断。这个版本不能保证区分“海流真的没变”和“估计失锁后保持旧值”，因此它是主基线，不是最终结论。

后续消融版本（`v1-valid-age`）才使用：

```text
[o_t^a, v̂_cx^b, v̂_cy^b, v̂_cz^b, current_valid, current_age_s / 1.0]
```

`v1-valid-age` 必须在主实验基本完成后，用相同 seed、相同传感器 profile、相同 DADE 结构配对运行；不得把它的结果混入主实验，也不得用它阻塞主实验。

这些海流侧变量只进入 `DADE_a`，不进入 `z_t^a`、`c_t` 或 `s_t^a` 的直接拼接部分。若传感器 wrapper 还需要 `bottom_lock_valid`、`water_track_valid`、`position_age_s` 等元数据，首版也应优先放在 DADE 侧输入中；确实需要进入 encoder 时，必须单独审计它是否含海流信息。

建议的首版归一化尺度如下。`position_scale_m` 要按地图范围固定并在配置中记录，不能按每个 episode 动态变化；这里给的是工程起点，不是最终结论：

```text
x,y,z                 / position_scale_m（首版可先用 100 m）
φ,θ,ψ                 / π
p,q,r                 / 1.0 rad/s
u,v,w                 / 2.5 m/s（横向/垂向可分别用 1.0 m/s）
```

actor DADE 侧的海流估计归一化：

```text
v̂_cx^b,v̂_cy^b,v̂_cz^b / 0.5 m/s
```

actor 的编码器输入 `o_t^a` 是 12 维传感器状态，不含海流估计；编码器输出 `z_t^a = f_enc(o_t^a)` 的维度是 `d_z`，不是 12。**actor 可以使用海流估计值 `v̂_c^b`，但只能以历史序列进入 `DADE_a`，不能使用仿真真值 `v_c`。** 真实海流、绝对无噪声速度、理想参考误差和 PVS 内部状态不得进入 `o_t^a`、`z_t^a` 或 `c_t`；actor DADE 的历史也只能使用带噪声、延迟和失锁的估计值历史，不能使用真实海流历史。

这里仍是 **12 个 active 物理量**；按 0.1.10.1 送入 MLP 前会追加 8 个恒为零的保留输入槽，所以 encoder 的实际输入宽度是 `12+8=20`，但新增槽不产生传感器语义，也不属于 `o_t^a` 的 active schema。`z_t^a` 的总 `d_z` 中包含 4 个恒为零的保留输出槽，actor 顶层宽度仍为 `d_z+3+64`。

#### 0.1.4 Critic 编码前观测 `o_t^c`

critic 可以使用特权信息。首版编码前状态为 15 维：

```text
o_t^c = [
  x, y, z, φ, θ, ψ,               # 绝对理想 6-DOF 位姿，NED
  u, v, w, p, q, r,               # 绝对理想体速度/角速度
  e_ψ, e_θ, e_u                   # 参考减当前，理想值
]
```

三个误差定义为：

```text
e_ψ = wrap(ψ_d - ψ)
e_θ = θ_d - θ
e_u = u_d - u
```

其中 `ψ_d, θ_d, u_d` 与 actor 的 `c_t` 来自同一个 ALOS 输出。真实海流不直接拼接到 critic 的 `z_t^c` 或 `c_t`，而是只作为 critic DADE 的历史侧输入：

```text
H_t^c = {[o_{t-k}^c, v_{c,t-k}^n], ... , [o_t^c, v_{c,t}^n]}
ĉ_t^c = DADE_c(H_t^c) ∈ R^64
```

因此：

```text
s_t^c = [z_t^c, c_t, ĉ_t^c]
```

真实海流 `v_c^n = [v_cx^n, v_cy^n, v_cz^n]` 是 NED 坐标系下的三维速度，只在 critic DADE 的历史序列中编码。现有海流实现中 `current_ned_mps` 的 z 分量通常为 0，但 PC 契约保留完整 3D；艇体系的 `current_true_body` 只作为变换后的诊断或辅助目标，不能混淆两种坐标。

critic DADE 的历史输入是“特权状态序列 + 真实海流序列”；actor DADE 的历史输入是“真实传感器状态序列 + 海流估计序列”。critic 的 `ĉ_t^c` 可以包含对真实海流的时域编码，但不得把 critic 的真实海流回灌给 actor 的 `z_t^a`、`c_t` 或 `DADE_a`。

critic 的 15 个 active 物理量同样追加 8 个恒为零的保留输入槽，encoder 实际输入宽度为 `15+8=23`；`z_t^c` 也包含 4 个恒为零的保留输出槽。actor 与 critic 的 active mask、槽位启用状态和归一化参数各自独立，不能因为当前都未启用就共用一份 metadata。

#### 0.1.5 Actor 与 critic 的输出

actor 输出三维修正后的执行器控制量：

```text
a_t = [a_r, a_s, a_n]

δ_r = 20° * clip(a_r, -1, 1)
δ_s = 20° * clip(a_s, -1, 1)
n   = 1525 * 0.5 * (clip(a_n, -1, 1) + 1)
```

即：舵角、艉水平舵角和螺旋桨 RPM。`a_t` 是 actor 的均值输出；PPO 训练时再从该均值构造高斯分布采样。环境侧仍执行幅值和速率限制，网络不能绕过安全检查。

首版必须记录：

```text
a_mean, a_sample, a_actual, requested_delta_rudder, requested_stern, requested_rpm,
rate_clipped, magnitude_clipped, material_saturation
```

critic 只输出状态价值：

```text
V(s_t^c) ∈ R
```

critic 不输出动作，也不参与执行器映射。DADE 的输出 `ĉ_t` 是特征，不是控制命令。

#### 0.1.5.1 PC 专属 `±20°` 执行器包线

自 2026-11-02 起，新 PC 实验统一使用：

| 项目 | PC 新契约 | 旧 I/O/R 契约 |
|---|---:|---:|
| 艉舵角 `δ_r` | `±20°` | `±15°` |
| 艉水平舵角 `δ_s` | `±20°` | `±15°` |
| 舵角速率 | `150°/s`，暂不改变 | `150°/s` |
| 螺旋桨 RPM | `0–1525`，暂不允许负 RPM | `0–1525` |
| 速度包线 | `1.0–2.5 m/s`，不因 20° 而扩大 | `1.0–2.5 m/s` |

这不是把旧实验的常数值全局改掉，而是新增一个 PC 专属 profile。配置必须显式写 `actuator_envelope="pc20"`，并把 `max_rudder_angle_rad`、`max_stern_plane_angle_rad` 都设为 `20°`；旧 `remus100_vehicle()`、旧 26D/27D 配置和旧结果继续使用 `15°`。

动作归一化和逆映射相应改为：

```text
a_r = δ_r / (20°)
a_s = δ_s / (20°)
a_n = 2 * n / 1525 - 1

δ_r = 20° * clip(a_r, -1, 1)
δ_s = 20° * clip(a_s, -1, 1)
n   = 1525 * 0.5 * (clip(a_n, -1, 1) + 1)

action_full_scale = [20°, 20°, 1525 rpm]
```

需要同步处理的量：

- `o_t^a` 的 12 维定义不含舵角/RPM，因此 actor encoder 输入宽度不变；姿态仍按 `/π` 归一化，速度仍按约 `2.5 m/s`，海流估计仍按 `0.5 m/s`，这些不因 20° 改变。
- 如果后续把上一时刻执行器加入 `o_t` 或 DADE token，舵角和艉水平舵必须分别除以 `20°`，RPM 除以 `1525`；不能继续使用旧代码里硬编码的 `15°` full scale。
- 物理舵速率仍为 `150°/s`，但按新满量程折算后的归一化速率从 `10 s⁻¹` 变为 `7.5 s⁻¹`；从 `-20°` 到 `+20°` 至少需要 `0.267 s`。PPO 的探索标准差、动作速率惩罚和 BC 逆映射必须重新 screen，不能直接沿用 15° 调好的参数。
- `material saturation` 仍按“超过 1% 物理满量程”定义；PC20 下舵角和艉舵的材料阈值变成 `0.2°`，不再按旧 `15°` 的 `0.15°` 计算。
- 动作惩罚建议直接使用物理角度或按 `20°` 满量程归一化后的量，避免同一个归一化动作在 20° 包线下实际偏转变大而惩罚权重含义改变。

后端约束是硬性的：**现有 PVS REMUS100 本体内部把舵角饱和在 `±15°`，因此不能把 actor 输出改成 `±20°` 后继续拿 PVS 当 20° 真值后端。** PC20 主训练使用我们可参数化的 batched Torch 模型并把本体限幅设为 `20°`，正式对比或抽查使用 MSS 的 `20°` 模型；PVS 只保留作 15° legacy 对照。启动时必须断言：

```text
policy_max_rudder <= plant_max_rudder
policy_max_stern  <= plant_max_stern
```

若后端仍是 PVS 的 `15°`，PC20 配置必须拒绝启动，不能静默依赖后端二次裁剪。

教师和 BC 也必须使用 PC20 包线。现有 `Remus100TrackingController` 从 `VehicleSpec` 读取 `15°` 上限；直接拿旧 teacher 采集的 BC 数据会让 20° 区间完全没有标签。PC20 应重新采集闭环 BC/DAgger 数据，teacher 的舵角限幅改为 `20°`；旧的 15° BC/PPO 权重不能直接加载后当作 PC20 结果，如要迁移只能作为新实验臂并重新标定动作语义。

`±20°` 只扩大操纵权限，不会自动让 `2.5 m/s` 在顶流下可达；高速度阶段仍受 `1525 rpm` 和速度包线限制。正式对照必须增加同包线的 `ALOS + PD/PI (20°)`，旧 `ALOS + PD/PI (15°)` B0 只作为历史基线。

#### 0.1.6 ALOS、`P_ref` 与“策略和 ALOS 的关系”

PC 不蒸馏 ALOS，也不让网络重新学习制导。ALOS 仍然根据全局/局部路径、当前状态和海流估计计算：

```text
P_ref（前视参考点）
ψ_d（期望航向）
θ_d（期望俯仰）
u_d（期望速度）
```

`P_ref` 由路径投影加前视距离得到；终端捕获阶段用终点替换。它主要用于 ALOS 的参考构造、奖励和诊断，不是 actor 的输出，也不是 actor 的必需输入。actor 看到的只是 `c_t` 和当前传感器状态，因此 actor 与 ALOS 的职责是分离的，但不是“完全无关”：actor 以 ALOS 的命令为条件完成底层跟踪。

PC v1 固定采用：

```text
guidance.compensate_current = false
u_d = 地速参考
```

ALOS 不使用真实海流，也不使用海流估计；海流估计只进入 actor DADE，真实海流只进入 critic DADE、奖励、诊断和训练期辅助目标。这样 `c_t` 不携带任何海流信息，actor 的地速控制语义也最清楚。

**ALOS 导航输入口径（2026-11-02 冻结）**

C2–C4 中 actor 和 ALOS 必须共享同一个导航解，禁止“actor 看 `p_nav`、ALOS 看真值”的系统级特权：

- ALOS 的路径投影、最近点、横向/垂向误差、`P_ref`、前视距离、终端捕获判断和最终 `c_t` 全部由 `p_nav` 计算；不得在制导内部回退到 `p_true` 或精确 `eta`；
- ALOS 使用与 actor 同一时刻、同一 profile、同一延迟后的 `p_nav`；如果未来要让制导使用额外的滤波或预测值，必须作为新实验臂单列，不能默认混入 PC v1；
- clean profile 下 `p_nav == p_true`，因此该口径与 clean 基准数值等价；realistic/strong profile 下两者不同，必须以 `p_nav` 版本为准；
- critic 仍使用理想 `eta`、理想速度和真实海流历史，奖励/终止/评估可使用真值；真值只用于训练监督和评估，不得通过 ALOS 命令泄漏给 actor；
- 启动门禁必须覆盖：C2 起 ALOS 无法读取 `p_true`，路径投影使用 `p_nav`，且 clean 下与旧真值路径结果一致，realistic 下出现可解释的导航误差影响。

如果后续要研究“ALOS 使用海流估计补偿”，必须作为新的实验臂并重新定义契约，因为它会让估计值通过 `c_t` 间接进入 actor，违反当前“海流估计只进 DADE”的规则。该变体不能混入 PC v1，也不能与 PC v1 共用 checkpoint 结论。

#### 0.1.7 真实传感器信号与“相对水速度”

需要区分三类速度：

- **DVL 底跟踪**：声波打到海底，测量艇相对海底的地速 `v_g^b`；海底锁定丢失时无效。
- **DVL 水跟踪**：声波打到水中悬浮颗粒，测量艇相对局部水体的速度 `v_r^b`；需要足够散射体、合适的距离单元和水体稳定假设，可能受剪切流、气泡、近底/近表面和失锁影响。
- **海流估计**：若同时有有效的地速和水速，可估计 `v̂_c^b = v_g^b - v_r^b`；也可使用 ADCP 或模型/滤波估计。只有底跟踪时，无法从单帧直接恢复真实水相对速度，必须依赖海流估计或历史推断。

因此 actor 的编码器输入 `o_t^a` 只包含直接测到的 12 维状态（其中 `u,v,w` 是 DVL 底跟踪地速），不包含 `v_r^b`、`v̂_c^b` 或其反推组合，编码后得到 `d_z` 维 `z_t^a`；带有效性、延迟和误差的海流估计只以历史序列进入 actor DADE。仿真不要求实现完整的商业 DVL/ADCP 估计算法，但必须生成与真值配对、具有相关误差和失效时刻的估计序列，便于 critic DADE、奖励和辅助损失使用真值，而 actor DADE 只能使用估计历史。

**PC v1 传感器实现口径（2026-11-02 最终确认）**

PC v1 的**主训练、课程切换、smoke、超参筛选和正式评估全部使用合成传感器**，不接入 Isaac Sim/OceanSim 原生传感器作为依赖或门禁：

1. 合成链从仿真真值 `η, ν, v_c^n` 出发，生成 actor 可见的 `p_nav`、IMU 姿态/角速度、DVL 底跟踪地速、DVL 水跟踪（仅用于生成估计链）和海流估计；critic 仍使用理想状态和真实海流历史。所有合成量必须带时间戳、采样率、延迟、失锁、有效标志、零阶保持和 epoch 内相关的零偏/尺度误差。
2. PC v1 的 `sensor_source` 一律记录为 `synthetic`；schema 可以预留 `native_isaac_oceansim` 枚举，但本阶段不实现、不测试、不作为实验门禁。训练和评估日志不得把合成观测写成原生 OceanSim 结果。
3. 当前不要求安装、启动或验收 OceanSim 的 DVL/IMU/气压计；此前关于 OceanSim 原生能力的调研、接口记录和 smoke 结果只保留为未来参考，不改变 PC v1 的观测、延迟、失锁和噪声模型。
4. 未来若要做原生传感器验证，必须作为独立的“原生 vs 合成”后置对照：相同观测 schema、相同坐标变换、相同固定输入和相同评估 seeds，先证明语义一致，再讨论是否进入训练。不得在 PC 主实验中途切换传感器来源。
5. 合成模型的真实性由可复现的误差过程、清晰的坐标/单位、保守的 profile 范围和单元测试保证；“先合成”不是降低标准，而是先把控制与 DADE 主问题做完，把原生传感器工程后置。

真机可部署性对照：

| 观测 | 真机来源 | actor 是否使用 | critic 是否使用 |
|---|---|---|---|
| `x,y,z` | USBL/INS/DVL 融合后的导航位置 `p_nav` | 是，接收 `p_nav`；误差按 0.1.7.1 合成，不直接接收原始 USBL | 是，直接使用理想 NED 位置 |
| `φ,θ,ψ` | IMU、磁罗经/航向融合 | 是，带零偏、漂移、磁偏角和滤波延迟 | 是，直接使用理想姿态 |
| `p,q,r` | IMU 陀螺 | 是，带零偏、噪声和尺度误差 | 是，直接使用理想角速度 |
| `u,v,w` | DVL 底跟踪地速 | 是，进入 `z_t^a`；失锁时保持最后值，不用海流估计反推 | 是，直接使用理想体速度 |
| 海流估计 `v̂_c^b` | ADCP 或“地速−水速”估计 | 仅进入 actor DADE 历史，带误差、延迟和失锁 | 可不使用；critic 使用真值 DADE |
| 真实海流 `v_c^n` | 仿真真值，真机没有直接量 | **禁止** | 仅进入 critic DADE 历史；奖励/辅助损失可用 |
| `e_ψ,e_θ,e_u` | 由 ALOS 命令和状态计算 | 由 `c_t` 与带噪状态自行隐含 | 是，使用理想参考误差 |

六类信号的首版建模重点：

1. **DVL 底跟踪**：白噪声、慢变零偏、尺度误差、海底锁定丢失、1–10 Hz 更新率、0–200 ms 延迟；失锁时保持最后有效值并给出 `bottom_lock_valid=0`。
2. **IMU 姿态/航向**：姿态白噪声、零偏、随机游走漂移、安装角误差、磁偏角、航向滤波延迟、时间同步误差；航向误差可以比横滚/俯仰更大。0–50 ms 延迟。
3. **DVL 水跟踪**：有效标志、白噪声、零偏、失锁、距离单元偏差、剪切流偏差、1–5 Hz 更新率、0–300 ms 延迟；水跟踪无效时不能让网络误以为相对水速为真值。
4. **USBL/INS 位置**：原始 USBL 具有低频、延迟、野值和短时失锁；0.1–1 Hz 更新、0–1 s 延迟。actor 的 `x,y,z` 不直接接收原始 USBL，而接收按 0.1.7.1 生成的融合导航位置 `p_nav`；原始 USBL、真实位置误差、有效标志和龄期只用于诊断与评估。
5. **执行器**：舵机死区、滞回、舵速限制、RPM 一阶滞后、推力/舵效随速度变化、安装偏差；需要与实际控制周期一致。
6. **海流估计器**：以“真值 + 相关误差”的合成估计代替真正估算器，至少包含零偏、低通滞后、延迟、随机游走、失锁保持和有效标志。估计值只进入 actor DADE 历史；真实海流只进入 critic DADE 历史、奖励、诊断和训练期辅助损失。

首版建议范围如下，所有范围都是训练分布配置，不是“真机一定如此”的保证。先保证策略可学习，再逐级扩大：

| 信号 | 首版建议范围 |
|---|---|
| DVL 底跟踪 | 噪声 `σ=0.005–0.03 m/s`；零偏 `0.005–0.02 m/s`；尺度 `0.97–1.03`；失锁 `0–10%`；`1–10 Hz`；延迟 `0–200 ms` |
| IMU 姿态 | 横滚/俯仰噪声 `0.1–1°`；航向噪声 `0.5–3°`；漂移 `0.1–1°/min`；磁偏角 `±5°`；延迟 `0–50 ms` |
| IMU 角速度 | 噪声 `σ=0.001–0.01 rad/s`；零偏 `0.001–0.005 rad/s`；尺度 `0.95–1.05` |
| DVL 水跟踪 | 噪声 `σ=0.01–0.05 m/s`；零偏 `0.01–0.05 m/s`；失锁 `0–20%`；`1–5 Hz`；延迟 `0–300 ms` |
| USBL/INS | 位置噪声 `σ=0.5–5 m`；野值 `0–5%`；`0.1–1 Hz`；延迟 `0–1 s` |
| 执行器 | 死区 `0–1°`；滞回 `0–0.5°`；RPM 时间常数 `0.1–0.5 s`；推力/舵效缩放 `0.8–1.2` |
| 海流估计 | 噪声 `σ=0.01–0.05 m/s`；零偏 `0.01–0.03 m/s`；延迟 `0–2 s`；失锁 `0–20%` |

传感器 wrapper 必须统一处理：

- 每个信号带时间戳、采样率、有效标志和龄期；
- 延迟通过队列模拟，不能只在当前帧加随机数；
- 失锁/丢包使用零阶保持，并同时输出 `valid=0/age>0`；
- 野值用稀疏脉冲或拉普拉斯分布，不要用连续高斯把 USBL 跳点平均掉；
- 安装角、磁偏角、尺度误差在一个 episode 内保持相关，不能每步独立重采样；
- 所有随机化都必须记录 profile 和 seed，评估时可复现。

#### 0.1.7.1 PC v1 的 `p_nav` 经验误差模型（2026-11-02 冻结）

**结论：PC v1 不实现完整 EKF，也不把原始 USBL 直接送入 actor。** actor 的 `x,y,z` 固定解释为导航解 `p_nav`；仿真中允许用真值加“有状态导航误差过程”生成 `p_nav`。网络只看到合成后的导航解，critic 仍看到理想真值。该选择不是降低真实性要求，而是把实现重点放在可观测的误差统计上，避免为 PC 实验额外引入一个未被验证的 EKF。

`p_nav` 的最小生成模型为：

```text
p_nav(t) = p_true(t) + b_ep + e_ou(t) + e_event(t)
```

其中：

- `b_ep` 是 episode 固定位置零偏，覆盖安装、初始对准和 USBL 标定误差；
- `e_ou(t)` 是时间相关的慢漂移，用一阶 Gauss–Markov/Ornstein–Uhlenbeck 过程生成，不能每步独立采样；
- `e_event(t)` 由 USBL 校正、USBL 野值、DVL 失锁漂移和短时导航失效组成；
- 最终 `p_nav` 经过延迟队列后才进入 actor；`valid/age` 主实验仍只记录，不进入网络。

首版参考实现如下，允许在单元测试中逐项关闭：

```text
e_ou[t+1] = rho * e_ou[t] + sigma_ou * sqrt(1-rho^2) * w_t
rho = exp(-dt / tau_ou)

if DVL bottom lock invalid:
    e_event[t+1] += v_drift_outage * dt

if USBL valid and inlier:
    e_event[t] += K_usbl * (p_usbl - (p_true[t] + b_ep + e_ou[t] + e_event[t]))

if USBL outlier not rejected:
    e_event[t] += jump_usbl

p_nav_observed[t] = p_nav[t - delay_nav]
```

模型要求：

- USBL 校正形成可见的锯齿误差，而不是把位置误差每步拉回零；
- DVL 失锁期间位置误差继续增长，不能冻结 `p_nav`；
- USBL 野值分两条路径验证：默认经过门限/鲁棒滤波后拒绝；C4 单独注入未被拒绝的跳点作为压力测试；
- 航向/安装误差可与位置漂移耦合，但首版不要求实现完整误差状态方程；
- 参数范围先用工程先验，后续必须用真实海试日志重新标定；
- 真实位置 `p_true` 只用于生成观测和 critic，actor 的任何输入路径都不能读取它。

建议的 PC v1 profile：

| profile | `p_nav` 固定零偏 | 慢漂移 | USBL 更新/延迟 | DVL 失锁 | USBL 野值 | 用途 |
|---|---:|---:|---:|---:|---:|---|
| clean | `0` | `0` | 理想连续 | `0` | `0` | C0 基本控制 |
| mild | `0.1–0.5 m` | `σ=0.05–0.2 m`，`τ=5–20 s` | `0.5–1 Hz`，`0–0.2 s` | `0–2%`，持续 `1–3 s` | `0–1%` | C1/C2 |
| realistic | `0.2–2 m` | `σ=0.1–0.5 m`，`τ=10–60 s` | `0.2–1 Hz`，`0.1–0.5 s` | `2–10%`，持续 `1–10 s` | `1–3%` | C3 主训练 |
| strong | `2–5 m` | `σ=0.5–1.5 m`，`τ=20–120 s` | `0.1–0.5 Hz`，`0.5–1 s` | `10–20%`，持续 `5–20 s` | `3–5%` | C4/留出压力测试 |

原始 USBL 的测量噪声仍按 0.1.7 的 `σ=0.5–5 m` 生成，USBL 更新率仍按 `0.1–1 Hz`；这些原始量用于生成校正事件和诊断，**不直接作为 actor 的 50 Hz 位置输入**。`p_nav` 的误差幅度应明显小于每一步直接叠加的原始 USBL 噪声，否则融合没有意义。

调查到的典型现实情况和对应建模点：

| 现实情况 | 对导航的影响 | 本模型中的对应项 |
|---|---|---|
| USBL 多径、反射、近水面/近结构物、通信丢失 | 位置野值、短时失锁、偶尔大跳点 | `jump_usbl`、`usbl_valid=0` |
| 声速剖面时变、分层和声线弯曲 | 随距离/方向变化的系统偏差，不是零均值白噪声 | `b_ep` + 随 episode 改变的 USBL 偏差 |
| USBL 与 IMU/姿态时间不同步、延迟到达 | 位置修正滞后，快速转向时误差增大 | `delay_nav`、时间戳偏移 |
| DVL 底跟踪失锁、离底高度不合适、底质差 | 速度中断，纯惯导航位置误差持续增长 | `v_drift_outage`、失锁时长 |
| DVL 安装角/尺度因子/波束角度误差 | 地速方向或幅值有偏，误差随航程累积 | DVL 零偏、尺度、安装角随机化 |
| IMU 航向零偏、磁罗经漂移和磁干扰 | 航位推算方向逐渐偏转，位置呈弧形漂移 | `e_ou` 与航向误差相关项 |
| 直线航行时部分误差状态不可观 | 某些零偏在长时间直航中无法被区分，误差可能积累 | 长直航 episode + episode 固定零偏 |

主要资料依据如下。文献用于支持失效机理和建模方向，不直接把论文中的实验数值当作本项目的参数标定：

| 资料 | 对本计划的支撑 |
|---|---|
| Rigby et al., *Towards Geo-Referenced AUV Navigation Through Fusion of USBL and DVL Measurements*, OCEANS 2006, doi:10.1109/OCEANS.2006.306898 | USBL 与 DVL 融合用于地理参考导航的基本结构 |
| Ridao et al., *USBL/DVL navigation through delayed position fixes*, ICRA 2011, doi:10.1109/ICRA.2011.5980110 | USBL 位置存在延迟和乱序修正，必须按时间戳处理 |
| *Voronoi Diagram-based USBL Outlier Rejection for AUV Localization*, 2024, doi:10.26748/ksoe.2024.049 | USBL 野值会显著破坏定位可靠性，需要显式拒绝 |
| *Two-Step Correction Based on In-Situ Sound Speed Measurements for USBL Precise Real-Time Positioning*, 2023, doi:10.3390/rs15205046 | 声速剖面时变、声线弯曲会造成系统性 USBL 偏差 |
| *Robust AUV Navigation Method Based on DVL Outlier Detection*, 2025, doi:10.1109/CAC67268.2025.11487024 | DVL 异常值检测是组合导航的必要环节 |
| *Context-Aware Reconstruction for DVL Outlier Detection in Robust AUV Navigation*, 2026, doi:10.1002/rob.70369 | 浅水环境干扰会产生 DVL 异常，真实海试需要时间上下文 |
| *Continuous INS/DVL Fusion in Situations of DVL Outages*, 2020, doi:10.1109/AUV50043.2020.9267945 | DVL 中断期间必须保持惯导连续传播，不能停止更新 |
| *Navigating Error Analysis for Acoustic Doppler Velocity Log*, 2013, doi:10.4028/www.scientific.net/AMM.303-306.453 | 姿态、速度、安装偏差和阵列偏心都会进入 DVL 导航误差 |
| *Observability Analysis of DVL/PS Aided INS for a Maneuvering AUV*, 2015, doi:10.3390/s151026818 | 不同机动下可观测误差状态不同，直线航行不能保证所有零偏可估 |
| *Design and Experimental Validation of a USBL Underwater Acoustic Positioning System*, 2016, doi:10.3390/s16091491 | USBL 与 INS 组合定位的工程实现和实验验证 |

`SyntheticNav` 的验收条件：

- clean profile 下 `p_nav == p_true`，其余 profile 下 actor 无法读取 `p_true`；
- USBL 校正后误差出现回拉，但不会在每个控制步被强制归零；
- DVL 失锁期间位置误差方差随失锁时长增加，且 `p_nav` 不冻结；
- USBL 缺失时继续传播，恢复后不会产生无穷大或 NaN；
- 野值被拒绝和未被拒绝两种路径都能复现，且日志中保留原始 USBL、`p_nav`、误差、`valid` 和 `age`；
- 延迟队列、时间同步、episode 随机种子和 profile 全部可复现；
- 指标至少报告 `p_nav` 三维误差的均值、RMSE、95 分位、自相关时间，以及失锁前后误差增长量。

#### 0.1.8 域随机化

PC 首版至少随机化以下维度：

| 维度 | 建议范围/方式 |
|---|---|
| 期望纵向速度 | `u_d=1.0–1.8 m/s`；低速段和终端捕获单独覆盖 |
| 路径/命令 | 路径几何、初始位姿、期望航向和俯仰由 ALOS 产生；命令不能固定成单一常数 |
| 海流大小 | `0.2–0.6 m/s`；包含零流和接近能力边界的强流 |
| 海流方向 | `[-π,π]`；可加入短时方向摆动和阵风 |
| 海流时间尺度 | 周期 `5–30 s`；另做快速 `1–5 s` 压力测试 |
| 空间剪切 | 使用现有 `G` 缩放，先小剪切后增大；z 方向剪切不能和传感器误差混为一谈 |
| DVL/IMU/USBL 原始量 | 按 0.1.7 的范围采样噪声、零偏、延迟、失锁、野值 |
| 导航位置 `p_nav` | 按 0.1.7.1 的 clean/mild/realistic/strong profile 生成；原始 USBL 不直接进入 actor |
| 执行器 | 死区、滞回、RPM 滞后、舵效和推力缩放 |
| 海流估计 | 误差、延迟、失锁、有效标志和龄期 |
| 初始状态 | 位置、姿态、速度、初始航向误差、初始海流相位 |

训练时必须同时保留 clean profile 和 realistic profile。clean profile 用于确认策略是否学会基本控制，realistic profile 用于确认鲁棒性；不能只报告其中一个。strong profile 默认只做小比例采样或最终评估，避免早期把策略压死。

#### 0.1.9 课程学习

建议先做“无 ALOS 的直接命令跟踪”，再切换到“ALOS 路径跟踪”。两个环境共用同一个 actor 接口 `[z_t,c_t,ĉ_t]`，只改变命令来源、奖励和终止条件。

| 阶段 | 环境/命令 | 海流 | 传感器/执行器 | DADE | 目的 |
|---|---|---|---|---|---|
| C0 | 无 ALOS：固定 `ψ_d,θ_d,u_d`，逐 episode 更换 | 零流 | 无噪声、无延迟、无失锁 | `ĉ=0` 或关闭 | 学会基本执行器控制和稳态跟踪 |
| C1 | 无 ALOS：随机阶跃/斜坡/平滑命令，`u_d=1.0–1.8` | 零流 | 小噪声、无失锁 | `ĉ=0` 或关闭 | 学会命令条件和基本抗扰 |
| C2 | 接入 ALOS：简单路径、`u_d=1.0–1.5` | 零流或小常值流 | 小噪声、无失锁 | 开启 DADE，低学习率预热 | 从命令跟踪迁移到路径跟踪 |
| C3 | ALOS：随机路径、`u_d=1.0–1.8` | 时变流、小剪切 | 真实噪声、少量失锁/延迟 | 完整 DADE | 学会估计器误差和短时失效 |
| C4 | ALOS：全范围 | 强时变、强剪切、阵风 | 较大延迟、失锁、野值 | 完整 DADE | 鲁棒性压力测试和留出评估 |

C0/C1 的命令跟踪环境不计算路径投影、`P_ref` 或 ALOS 参考，奖励只由命令误差组成：

```text
eψ_n = wrap(ψ_d - ψ) / π
eθ_n = (θ_d - θ) / (π/2)
eu_n = (u_d - u_g) / 2.5

r_cmd = -[1.5|eψ_n| + 0.5|eθ_n| + 1.0|eu_n|]
        + w_hold * Δh_t
        - w_phi * roll_pen
        - w_p * roll_rate_pen
        - 0.5 * ||Δa_norm||²
        - 0.1 * s_sat
```

其中：

```text
h_t = min(1, 连续满足命令容差的时间 / T_hold)
Δh_t = h_t - h_{t-1}
a_norm = [δ_r / 20°, δ_s / 20°, n / 1525]
Δa_norm = a_norm[t] - a_norm[t-1]
s_sat = mean(relu(|a_req| - 1)) + mean(relu(|Δa_req| / rate_norm - 1))
```

`h_t` 是保持势能，离开容差时清零；首版允许 `Δh_t` 为负，以惩罚提前离开容差。若负项导致策略过度保守，再改成 `max(Δh_t, 0)`。`s_sat` 是连续饱和度，不是 boolean；`a_norm` 使用实际执行器动作，请求动作只用于饱和诊断。`w_hold` 首版可设 `5.0`，动作平滑和饱和项的权重分别从 `0.5` 和 `0.1` 起 screen。

C0/C1 的成功不按路径进度判断，而按命令进入容差后的保持时间、超调和稳态误差判断。C2 之后**不恢复路径进度奖励**；路径进度、路径误差和曲率只用于 ALOS 制导、诊断和评估，不进入 PC 奖励。不能把 C0/C1 的命令跟踪收益直接算成 ALOS 路径跟踪收益。

**C2–C4 奖励：ALOS 命令跟踪（2026-11-02 冻结）**

PC 的职责固定为“跟踪 ALOS 命令”，不是替代 ALOS 做路径规划或路径误差补偿。因此 C2–C4 与 C0/C1 使用同一套命令跟踪误差，只是命令来源从手动变成 ALOS，并关闭 C0/C1 的保持奖励：

```text
eψ_n = wrap(ψ_d - ψ) / π
eθ_n = (θ_d - θ) / (π/2)
eu_n = (u_d - u_g) / 2.5

r_alos = -[1.5|eψ_n| + 0.5|eθ_n| + 1.0|eu_n|]
         - w_phi * roll_pen
         - w_p * roll_rate_pen
         - 0.5 * ||Δa_norm||²
         - 0.1 * s_sat
         + safety_penalty
```

统一奖励模式：

```text
reward_mode = "pc_command_track"
hold_bonus = true only for C0/C1
hold_bonus = false for C2/C3/C4
```

C2–C4 明确不包含：

- 路径进度 `Δs` 或 `w_prog * Δs`；
- 最近路径投影误差 `e_h/e_v`；
- 路径切线航向误差；
- 曲率奖励或曲率惩罚；
- `goal_bonus`、`timeout_penalty`、`terminal_progress_weight`；
- ALOS 的 `P_ref`、前视距离和参考点误差。

只保留 `safety_penalty`：出界、NaN/Inf、极端滚转或持续滚转超限等安全事件。到达终点只终止 episode，不额外给路径到达奖励；超时按 truncated 处理，不附加任务奖励。这样 PC 的训练目标与“ALOS 负责路径、PC 负责跟踪”完全一致，路径成功率、三维 RMSE 和捕获率只作为外部评估指标。

**滚转约束（适用于 C0–C4）**

actor 的 `o_t^a` 已经包含滚转角 `φ` 和滚转角速度 `p`，因此不需要新增观测维度；滚转约束只进入奖励、终止条件和评估指标。PC v1 没有滚转命令通道，`c_t` 仍只有 `ψ_d, θ_d, u_d`，所以默认 `φ_ref=0`。但这不是要求物理上绝对零滚转：REMUS 这类欠驱动 AUV 没有独立副翼，转弯时由舵产生的耦合滚转无法完全消除，必须使用软约束。

建议的滚转项：

```text
e_phi = wrap(φ - φ_ref)
roll_pen = (relu(|e_phi| - φ_dead) / (φ_soft - φ_dead))²
roll_rate_pen = (relu(|p| - p_dead) / (p_soft - p_dead))²
```

首版 provisional 参数：

| 参数 | 首版值 | 含义 |
|---|---:|---|
| `φ_dead` | `5°` | 小于 5° 不惩罚，避免正常转弯耦合被误罚 |
| `φ_soft` | `10°` | 超过 5° 后平滑加罚，10° 左右达到单位惩罚 |
| `φ_strong` | `30°` | 明显强惩罚和感知质量不合格阈值，不是立即终止阈值 |
| `φ_term` | `45°` | 安全终止候选；候选范围 `45–60°`，先用 teacher 分布标定 |
| `T_phi_term` | `1.0 s` | `|φ|` 连续超过 `φ_term` 的持续时间 |
| `p_dead` | `10°/s` | 小滚转角速度不惩罚 |
| `p_soft` | `30°/s` | 滚转振荡达到 30°/s 左右时单位惩罚 |
| `p_term` | `90°/s` | 安全终止候选；候选范围 `90–120°/s` |
| `T_p_term` | `0.5 s` | `|p|` 连续超过 `p_term` 的持续时间 |
| `w_phi` | `0.05` | 滚转角权重，先按 reward budget 校准 |
| `w_p` | `0.02` | 滚转角速度权重，主要抑制振荡 |

规则：

- C2–C4 的 `r_alos` 和 C0/C1 的 `r_cmd` 同样附加 `-w_phi * roll_pen - w_p * roll_rate_pen`，不能只在 C0/C1 约束滚转；
- 奖励使用仿真真值 `φ,p` 计算，actor 仍只看到带噪 IMU 观测；不得用 actor 的估计值反过来计算训练奖励；
- 滚转角在 `5°` 内不罚；`5–10°` 平滑加罚；`10–30°` 明显加罚；`30°` 是强惩罚/质量门禁，不是立即终止；只有 `|φ| >= φ_term` 持续 `T_phi_term`，或 `|p| >= p_term` 持续 `T_p_term`，才触发安全终止；
- 滚转速率和滚转角分开惩罚，避免“角度不大但高频摇摆”通过奖励；
- 感知任务的实际允许滚转若小于上述值，应整体缩放 `φ_dead/φ_soft/φ_strong/φ_term`，不要只改 `w_phi`；
- 滚转项在总回报中的目标贡献控制在约 `5%` 以内，不能压过航向、俯仰和地速命令误差；
- 评估必须报告 `|φ|` 均值、p95、最大值，`|p|` 的 p95，超出 `φ_soft` 的时间比例，以及滚转符号翻转次数；
- 若 PD/PI teacher 在正常转弯中天然超过 `φ_soft`，先记录 teacher 的滚转分布，再决定是放宽阈值还是调整 teacher 增益，不能把正常耦合当成策略失败。

**PD/PI BC 热启动（允许，但不是最终控制器结构）**

这里的 BC 是 PC 专用热启动，不改变旧 26D 关于“BC 不是前置条件”的结论。BC 只用于避开 PPO 从随机初始化开始在 C0/C1 早期大量撞限幅或闭环发散的问题。首版流程：

1. 在 C0/C1 的无 ALOS 命令跟踪环境运行现有 PD/PI 控制器，teacher 与部署 actor 使用同一动作语义：输出 `[δ_r, δ_s, n]`，其中速度控制始终是地速；不要让 teacher 读取真实海流或使用水相对速度；
2. 采集闭环 episode，而非单点随机状态；覆盖固定命令、阶跃、斜坡和平滑命令，并记录 `o_t^a`、`c_t`、teacher 请求动作和实际执行动作；进入 C2 后可增采少量 ALOS 闭环 PD/PI 数据帮助命令语义迁移，但仍只监督执行器；
3. 用 actor encoder + actor head 做监督训练，目标是归一化后的 `[a_r,a_s,a_n]`，即 teacher **请求动作**经动作 adapter 逆变换后的值，不是饱和后的实际执行量；不训练 critic，不把 BC 做成残差，也不把 ALOS 的中间量加入标签；
4. 先按 BC-only 闭环评估。若离线 loss 正常但闭环发散，最多补一轮 PC 专用 DAgger，把 actor 自己访问到的状态交给 PD/PI 重新标注；不能只看离线 MSE 判定成功；
5. 从 BC 权重初始化 PPO；PPO 阶段不冻结 actor、不加 residual adapter、不改变动作输出语义。BC 后的探索标准差要单独 screen，避免一开始把已学好的闭环打散；
6. BC-only（不接 PPO、不接 DADE）必须作为正式对照保留。若 BC-only 已足够，而 PPO/DADE 没有稳定增益，应如实记录为负结果。

课程控制原则：

- 先固定路径和命令，再随机命令；
- 先无 ALOS 的命令跟踪，再接 ALOS 路径跟踪；
- C1→C2 切换时降低 actor 学习率，并保留一部分命令跟踪 episode 防止遗忘；
- 先加传感器白噪声，再加相关零偏/漂移；
- 先加短延迟，再加失锁和野值；
- 先常值流，再时变流，最后剪切流；
- C3 是真实传感器域的主训练阶段；C4 只占少量训练采样，主要作为压力测试和留出评估集；
- 若 C0 都不能稳定闭环，不允许用“传感器太真实”解释失败，必须先检查动作映射、归一化和 PPO 基本设置。

**高速课程 H1–H4（与 C 阶段正交，叠加在 C2/C3 上）**

`0.1.8` 的 `u_d=1.0–1.8 m/s` 是 PC 主训练包线；`PC20` 的 `1.0–2.5 m/s` 是执行器允许包线。两者不是同一件事。高速阶段必须作为独立课程，在 H0 基线和对应 C 阶段通过后再开启，不能把 C3 的普通速度结果直接当作 2.5 m/s 结论。

| 阶段 | 纵向地速命令 | 训练采样比例 | 前置条件 | 目的 |
|---|---|---:|---|---|
| H0 | `1.0–1.8 m/s` | C1/C2/C3 主训练 | 已通过对应 C 阶段门禁 | 基本控制、传感器真实化和 DADE 基线 |
| H1 | `1.8–2.0 m/s` | 通过后占 `10–20%` | H0 成功率稳定，20° 本体已完成速度/舵效标定 | 从主包线平滑进入高速 |
| H2 | `2.0–2.25 m/s` | 通过后占 `10–20%` | H1 门禁通过，顶流可达性检查通过 | 中高速下的航向/俯仰/速度耦合 |
| H3 | `2.25–2.4 m/s` | 通过后占 `5–10%` | H2 门禁通过，DVL/导航延迟按高速重新审计 | 接近包线上沿的稳定控制 |
| H4 | `2.5 m/s` | 默认只评估；若训练不超过 `5%` | H3 门禁通过且有足够算力 | 包线极限和压力测试，不作为主训练结论 |

高速阶段必须满足：

- 训练前用同一 `pc20` 本体做速度标定：静水最大地速、顶流最大地速、20° 舵阶跃响应、回转半径和 RPM 稳态关系；
- 每个 episode 采样命令时检查可达性：`||u_d * e_ground - v_c||` 不得超过当前 RPM/速度包线下本体的可达水相对速度；不可达命令要重采样，不能把物理不可达当成策略失败；
- 高速度下 1 s USBL 延迟对应 2.5 m 位置误差，必须复用 `SyntheticNav` 的 realistic/strong profile，并单独报告高速导航误差；
- 高速阶段仍使用 `±20°` 舵角、`150°/s` 舵速率和 `0–1525 RPM`，不得回退到旧 15° 动作语义；
- 每个 H 阶段先做短预算 smoke，再决定是否进入下一阶段；若 H1 已出现不可恢复的发散或持续饱和，不得用 H2/H3 的更大速度掩盖问题。

H1–H3 的 provisional 门禁：相对 H0 同 profile 的成功率下降不超过 `0.05`，三维 RMSE 不超过 H0 的 `1.25×`，命令跟踪稳态误差和超调不超过 H0 的 `1.2×`，且无持续 material saturation；H4 只报告成功率、RMSE、饱和率、约束违反和最坏轨迹，不作为训练通过/失败的唯一依据。

#### 0.1.9.1 PC 终止条件与 episode 边界（2026-11-02 冻结）

有必要设计终止条件，但只做**稀疏安全终止**，不做“跟踪误差一差就结束”的密集早停。终止的作用是减少无意义 episode、避免极端状态污染 value，并让安全失败和正常超时在 PPO 中具有不同语义；它不能替代奖励塑形，也不能让策略把“避免终止”当成唯一目标。

| 条件 | 类型 | 处理 |
|---|---|---|
| C2–C4 到达路径终点 | `terminated` | 只结束 episode，不给 PC 任务奖励；不 bootstrap |
| 达到最大 episode 步数 | `truncated` | 不加 timeout 任务惩罚；按 time-limit 规则 bootstrap |
| 出界、NaN/Inf、非法数值/姿态状态 | `terminated` | 一次性安全惩罚；不 bootstrap |
| 进入明确的海床/水面危险深度区 | `terminated` | 仅在场景边界明确定义时启用；一次性安全惩罚 |
| 持续极端滚转/滚转速率 | `terminated` | 满足持续阈值后结束；一次性安全惩罚；不得因瞬时尖峰终止 |
| 普通滚转或短时滚转振荡 | 不终止 | 使用 `roll_pen`/`roll_rate_pen` 软约束 |
| 动作饱和、DVL 失锁、水跟踪无效、USBL 野值、海流估计失锁 | 不终止 | 这些是训练分布和鲁棒性的一部分，不是失败 |
| 命令误差暂时很大、正在做必要机动 | 不终止 | 由命令跟踪奖励和固定 horizon 处理 |

滚转终止的首版 provisional 口径：

```text
phi_term   = 45°       # 候选范围 45–60°，按 teacher 的合法机动分布标定
T_phi_term = 1.0 s     # |φ| 连续超过 phi_term
p_term     = 90°/s     # 候选范围 90–120°/s
T_p_term   = 0.5 s     # |p| 连续超过 p_term
```

计时必须使用仿真时间，不能用固定步数隐式假设永远是 50 Hz；超过阈值时累计，回到阈值以下时清零。首版先用 `45°/1.0 s` 和 `90°/s/0.5 s` 作为可执行默认值，再根据 PD/PI teacher 的正常转弯分布与感知任务容忍度校准。若 teacher 的正常机动本身超过这些值，应先检查 teacher 增益、速度包线和耦合滚转是否合理，不能为了让所有 episode 活下来而无限放宽；反过来，也不能把正常转弯的瞬时滚转当成安全失败。

终止与 PPO 的边界必须严格区分：

- `terminated` 表示安全失败或任务目标已经到达，不 bootstrap 下一状态价值；其中到达终点只表示 episode 正常完成，不附加 `goal_bonus`；
- `truncated` 只表示时间上限或外部截断，必须保留 time-limit bootstrap；不能把超时当作安全失败；
- 同时触发安全和超时时，以安全终止优先，并记录 `termination_reason="safety_*"`；
- 每条 transition 记录 `termination_reason`，至少包含 `goal`、`time_limit`、`out_of_bounds`、`nan_or_inf`、`invalid_state`、`depth_limit`、`safety_roll_angle`、`safety_roll_rate`；
- 安全终止惩罚是稀疏一次性项 `r_safety_terminal`，首版建议从 `-50` 起 screen，必须检查它是否过小以致策略把提前终止当作逃避负奖励的手段，也不能大到让策略完全不敢机动；
- 时间戳不单调、动作形状错误、后端非有限值等接口契约违反应作为训练基础设施错误中止或隔离，不能静默伪装成普通的策略安全终止；
- C0/C1 不因命令误差或保持失败提前终止，只按固定 horizon 截断；C2–C4 也不因普通路径偏差提前终止。

终止条件必须与奖励和评估分开审计：训练日志报告各终止原因的比例、平均 episode 长度和“安全终止前后 2 s 的最小/最大滚转”；正式评估报告 goal/time_limit/safety 三类比例。若安全终止率异常升高而任务误差没有同步恶化，应优先检查阈值、计时代码和终止惩罚是否造成训练逃逸，而不是把它解释成策略鲁棒性提升。

#### 0.1.10 `d_z` 与学习率搜索

编码器首版建议：

```text
Linear(input_dim, 128) → ELU → Linear(128, 128) → ELU → Linear(128, d_z)
```

actor 与 critic 各一份编码器。`d_z` 暂时没有理论最优值，必须先做小规模 screen：

```text
d_z ∈ {16, 32, 64, 128}
actor + encoder lr ∈ {1e-4, 3e-4, 1e-3}
critic lr ∈ {1e-4, 3e-4}
```

工程起点可先用 `d_z=32`、actor/encoder lr `1e-4`、critic lr `3e-4`，但这只是为了让第一轮 smoke 有可执行配置，不是最终结论。

推荐搜索顺序：

1. 固定 `d_z=32`，先扫 actor/encoder lr，再扫 critic lr；
2. 固定学习率，扫 `d_z`；
3. 前两名组合各跑 3 个训练 seed 的短预算，比较 clean 和 realistic；
4. 最终配置跑 5 个训练 seed，再用固定评估 seed `0–49`、`50–99` 做正式报告。

筛选时同时看成功率、三维 RMSE、动作饱和、约束违反和 realistic/clean 退化差；不能只看训练回报。每次改动 `d_z` 都要重新记录 checkpoint metadata 和归一化参数，避免旧 checkpoint 误加载。

#### 0.1.10.1 编码器预留槽位与 `active_mask`（2026-11-02 冻结）

PC v1 的编码器同时预留**少量输入槽位**和**少量 `z_t` 输出槽位**，目的是让以后增加低维观测时不必修改 actor/critic 的顶层 `[z_t,c_t,ĉ_t]` 宽度。预留槽位不是“以后加数据就能零样本生效”的插件机制；新增有效观测后仍必须更新 schema、重新训练或微调，并重新做泄漏与归一化审计。

首版默认：

```text
n_in_reserved = 8
n_z_reserved  = 4

o_t^enc = [normalize(o_t^active), 0, ..., 0]        # 8 个零输入槽
m_t^o   = [1, ..., 1, 0, ..., 0]                   # 输入 active mask

d_z = d_z_active + n_z_reserved
z_t = [f_enc_active(o_t^enc), 0, ..., 0]           # 4 个零输出槽
m_t^z = [1, ..., 1, 0, ..., 0]                     # 输出 active mask
```

`d_z` 的搜索值 `{16,32,64,128}` 均按**包含保留槽的总维度**解释，例如总 `d_z=32` 且 `n_z_reserved=4` 时，实际有效维度是 `d_z_active=28`。actor 和 critic 各自保存自己的 `active_mask`，但两者的总 `d_z`、`c_t` 宽度和 `ĉ_t=64` 保持固定。

实现约束：

- 零是合法观测值，因此必须同时保存并传递 `active_mask`；mask 写入 checkpoint metadata、schema hash 和 rollout 日志，不能只靠“这一列现在恒为零”猜测槽位是否有效；
- 归一化只对 active 特征做固定尺度的逐特征缩放，然后再补零；不得对包含 padding 的整段输入做 BatchNorm 或全局 LayerNorm，否则零槽会污染统计量并让训练/部署不一致；
- 当前未被激活的输入槽不参与梯度；输出保留槽在 v1 固定为零。以后启用某个槽位时必须提升 schema 版本、更新 mask、调整最终投影层并重新训练/微调，不能声称旧策略无需训练即可直接使用新观测；
- 每个保留槽必须有稳定 ID 和 registry，记录名称、单位、归一化尺度、来源、有效条件和启用版本；同一槽 ID 不得在不同 checkpoint 中改变语义；
- 低维新观测可以复用输入零槽；高维新观测必须走独立的小编码器并投影到 `n_z_reserved=4` 个保留输出槽，不能把几十维新信号硬塞进 8 个标量输入槽；若 4 维表达不足，必须提升 schema、重新做 `d_z` screen，不能动态加宽 actor/critic；
- 预留槽位只属于 encoder 输入和 `z_t`；DADE 历史 token 仍只包含实际观测/海流序列，不能填充零 token，也不能把无效槽当作真实测量；
- actor 和 critic 的编码器、mask、归一化参数和槽位启用状态完全独立，不能共享，也不能因为两者当前都是零就误用同一份 metadata；
- 预留槽会增加死通道和少量无用输入宽度；若后续对照发现它显著损害容量或收敛，应把“无预留槽”作为后置消融，而不是静默删掉 schema 记录。

#### 0.1.11 两套 DADE 与让 DADE 效果更明显

actor DADE：

```text
H_t^a = {[o_{t-k}^a, v̂_{c,t-k}^b], ... , [o_t^a, v̂_{c,t}^b]}       # v1-main
ĉ_t^a = DADE_a(H_t^a) ∈ R^64
```

主实验不使用 `valid/age`；`v1-valid-age` 消融才把 `valid/age` 追加到每个历史 token，并在主实验基本完成后单独运行。

critic DADE：

```text
H_t^c = {[o_{t-k}^c, v_{c,t-k}^n], ... , [o_t^c, v_{c,t}^n]}
ĉ_t^c = DADE_c(H_t^c) ∈ R^64
```

也就是说，两套 DADE 都编码“观测状态序列 + 海流序列”：actor 编码真实传感器状态与海流估计历史，critic 编码特权状态与海流真值历史。`c_t` 不进入 DADE。

上面的 `H_t^a`/`H_t^c` 是 **v1 的控制步对齐口径**，不表示原始传感器同频。真实信号的原生采样率明显不同：IMU 通常 50–200 Hz，DVL 底跟踪 1–10 Hz，DVL 水跟踪 1–5 Hz，USBL/INS 0.1–1 Hz，海流估计通常 0.1–5 Hz，而控制器是 50 Hz。v1-main 把这些信号通过零阶保持和延迟队列映射到 50 Hz 控制步；`valid/age` 只记录用于诊断，不进入主实验网络。因此 token 等长不等于物理采样率相同。

PC v1 **复用现有 DADE Transformer**，不在其前面再叠一层单独的 self-attention，也不为 IMU/DVL/海流分别建 attention 分支。当前输入本质上是多源数值时间序列，不是必须做模态对齐的异构图文输入；额外 attention 会增加参数量和推理延迟，但尚未证明能改善本任务。`valid/age` 是可靠性元数据，不是 self-attention 的先决条件，主实验先不输入。只有在 v1 已证明瓶颈确实来自原生采样率错配时，才进入 v2 的多采样率编码与 cross-attention；若论文需要证明 attention 结构本身的价值，再增加同预算的 GRU/TCN/no-attention 基线，这些均不得阻塞主实验。

`v1-main` 每个 token 使用 `[o_t, current_t]`，不加 `valid/age`；这是主实验默认基线。`v1-valid-age` 才使用 `[o_t, current_t, valid, age]`，属于后置消融。无论哪个版本，都要在仿真和 rollout 日志中记录真实失效/龄期，避免把零阶保持后的重复值误当成同频真实测量；但主实验不把这些信息输入网络。

如果 v1 在 realistic 条件下仍对 history 不敏感，再考虑 **v2 多采样率 DADE**：保留原生时间戳，把状态/IMU/DVL 快流与海流慢流分别编码，按时间戳对齐后用 late fusion 或 cross-attention 融合。cross-attention 在这里才真正有结构上的理由；但它不是首版默认，必须先证明 v1 的瓶颈不是窗口、延迟、估计器质量或当前帧泄漏。

两者不共享权重、不共享历史缓冲、不共享归一化统计。首版沿用短/长历史窗口 `10/50` 步；50 Hz 下约 `0.2 s / 1.0 s`。但海流周期可能是 `5–30 s`，因此 `1.0 s` 的长窗可能根本不足以观察慢变流；必须增加长窗消融：

```text
短窗：10 步 / 50 步（约 0.2 s / 1.0 s）
长窗：50 步 / 250 步（约 1.0 s / 5.0 s）
```

**折扣视野与 DADE 慢流学习（PC 必做 screen）**

50 Hz 下 `gamma=0.99` 的名义折扣视野约为 `1/(1-gamma)=100` 步，即约 `2 s`；`gamma=0.995` 约为 `4 s`，`gamma=0.998` 约为 `10 s`，`gamma=0.999` 约为 `20 s`。海流时间尺度是 `5–30 s`，因此只把历史窗口从 50 步加到 250 步，不一定能让 DADE 学会慢变流，信用分配视野也必须一起检查。GAE 的 `lambda` 会进一步缩短有效信用分配，不能只报告 `gamma` 而不记录 `lambda`。

PC 的 DADE 诊断分支至少做以下配对 screen：

```text
gamma ∈ {0.99, 0.995, 0.998, 0.999}
lambda = 0.95 固定做第一轮；必要时再 screen {0.97, 0.98, 0.99}
```

规则：

- DADE on/off、短窗/长窗、`gamma`/`lambda` 的对比必须使用相同 seed、相同传感器 profile 和相同训练预算，不能只给 DADE 组更长的折扣视野；
- 先比较 realistic C3 的控制收益，再看 `current-only`/`current-zero` 的历史敏感性；长视野如果只改善 value loss 而不改善控制，不算 DADE 收益；
- 若 `gamma>=0.995` 让基础 PPO 变得不稳定，优先改用 0.1.11 的辅助损失，而不是继续增大 `gamma`；
- 至少保留一个 `gamma>=0.995` 的 DADE 组和一个辅助损失组，才能下“DADE 在此任务上无效”的结论；只跑 `gamma=0.99` + 短窗不能作为否定 DADE 的充分证据；
- 所有训练配置必须记录 `gamma`、`lambda`、DADE 窗口、辅助损失权重和是否使用 BC 热启动。

若 PC 训练后 DADE 仍然对 history 不敏感，优先检查以下因素，而不是继续加大网络：

1. **估计器太理想**：如果 `v̂_c^b` 几乎等于真值，DADE 的当前帧已经足够，DADE 没有学习历史的动力。必须保留延迟、低通滞后、零偏和失锁。
2. **海流太慢或太常数**：常值流只需当前帧误差即可识别。训练要包含时变流、方向变化、阵风和剪切流，并让变化落在历史窗口内。
3. **历史窗口太短**：50 Hz 下 50 步只有 1 s，不能覆盖 5–30 s 的海流变化。优先做 `50/250` 步的长窗消融。
4. **当前帧泄漏**：不能让 actor 通过当前帧的某个变量反推出真值海流。必须做 `current-frame dropout`、`current-zero` 和输入置换测试。
5. **DADE 输出没有被使用**：要测 actor 对 `ĉ_t` 的敏感性，而不是只看网络里有没有这个模块。可以固定其他输入、只改变历史，比较 actor 动作变化。
6. **任务本身不需要历史**：如果 clean 条件下没有 DADE 也能完成，DADE 的收益只能在 realistic、延迟、失锁和强时变条件下体现。必须在这些条件上单独报告。

让 `ĉ_t^a` 更可解释、更容易被训练信号约束的候选方案是训练期辅助损失：

```text
L = L_PPO + λ * || g(ĉ_t^a) - v_c^b ||²
```

其中 `g` 是只在训练期使用的线性/MLP 辅助头，`v_c^b` 是真实海流的艇体系表示。λ 建议从 `0.01–0.1` 起搜；辅助头不得进入 actor 推理路径，真实海流也不得进入 `ĉ_t^a` 的输入。这个损失只用于让 DADE 特征保留海流信息，不能把“辅助头预测得好”直接写成控制收益。

DADE 正式有效性门禁（与 PPO 控制门禁分开报告）：

- full history 相对 current-only 在 realistic C2/C3 上有一致改善，且不是只靠失败尾部拉低总 RMSE；
- `current-frame dropout=0.5` 或 `current-zero` 下，完整历史明显优于无历史；
- actor 动作对历史变化有可测敏感性；
- 辅助头能在留出集上预测海流，且该预测与控制改善相关；
- 若以上任一不成立，只能写“DADE 在当前数据上没有形成历史依赖”，不能写“DADE 有效”。

#### 0.1.12 PC 评估、基线与门禁

PC 正式对比至少包含五组，且所有 PC 组统一使用 `±20°` 包线：

1. `ALOS + PD/PI`：必须增加同包线的 `ALOS + PD/PI (20°)` matched baseline；旧 `ALOS + PD/PI (15°)` B0 只作为历史基线，不能与 PC20 直接混算；
2. `PC BC-only`：PD/PI 监督热启动，不接 PPO、不接 DADE，验证纯控制器和 BC 本身的闭环能力；
3. `PC BC + PPO without DADE`：从 BC 初始化，`ĉ_t` 置零或移除 DADE，验证 PPO 和编码器控制本体；
4. `PC BC + PPO with DADE`：actor/critic 各一套 DADE，作为完整 PC 方法组；
5. 旧 `I2 DADE (15°)`：只保留为历史对照，不与 PC20 混成同一实验臂。

若预算允许，再补一组 `PC from-scratch PPO without DADE`，用于判断 BC 是加速收敛还是实际收益来源；它不是主门禁的必需项，不能阻塞前五组。

必须同时报告：

- clean 与 realistic；
- 默认到达半径 `2.0 m` 和严格捕获半径 `1.25 m`；
- 评估 seed `0–49` 与 `50–99`；
- 成功率、三维/水平/垂向 RMSE、平均到达时间；
- ALOS 命令跟踪指标：`ψ、θ、u_g` 相对 `ψ_d、θ_d、u_d` 的 RMSE、p95、稳态误差、超调和建立时间；路径指标与命令指标必须分开报告；
- material saturation、约束违反、舵/RPM 速率裁剪；报告时写明 `rudder/stern full scale = 20°`，旧 15° 结果的饱和率不能与 PC20 直接横向比较；
- 滚转指标：`|φ|` 均值/p95/最大值、`|p|` 的 p95、超出 `φ_soft` 的时间比例、滚转符号翻转次数；这些指标必须与感知任务的实际滚转容忍度一起报告；
- 终止统计：`goal/time_limit/safety` 三类比例、各 `termination_reason` 计数、平均 episode 长度、安全终止前 2 s 的滚转统计；确认没有“靠提前终止逃避负奖励”；
- actor 推理延迟 p50/p95、传感器 wrapper 开销；
- 海流估计误差、有效率和龄期；
- actor 对真实海流的泄漏测试结果；
- 海流拓扑审计：actor 的 `z_t^a`/`c_t` 不含海流估计，actor 的 `ĉ_t^a` 只来自估计历史；critic 的 `z_t^c`/`c_t` 不含真实海流，critic 的 `ĉ_t^c` 只来自真值历史。
- BC 的离线动作误差、闭环命令跟踪误差、超调、稳态误差和饱和；必须区分“离线拟合好”和“闭环能飞”。

建议门禁：

- **PC-G0 契约与启动门禁**：输入宽度、归一化、坐标变换、动作限幅、海流拓扑、预留槽位和 checkpoint metadata 全部通过；启动时必须硬校验 `guidance.compensate_current=false`、`speed_feedback='ground'` 且 `u_d` 为地速命令，同时硬校验 `actuator_envelope="pc20"`、`action_full_scale=[20°,20°,1525 rpm]`、policy 舵/艉舵满量程均为 `20°`、后端 plant 限幅不低于 `20°`；配置不满足就拒绝启动，不能只打印 warning；actor 的真实海流泄漏测试为 0；海流估计只从 actor DADE 历史进入，真实海流只从 critic DADE 历史进入；encoder 的 active mask、输入/输出保留槽数和 schema hash 必须可复现；BC/PPO checkpoint 必须记录是否经过热启动、动作包线及其 lineage。
- **PC-G1a 命令跟踪门禁（C0/C1）**：无 ALOS、零海流条件下闭环不发散，命令保持成功率 `≥0.95`，稳态误差和超调不超过 PD/PI teacher 的 `1.2×`，material saturation 接近 0；`SyntheticNav` 的 clean profile 必须满足 `p_nav == p_true`，且全部单元测试通过；未通过先查动作映射、归一化、导航 wrapper 和 BC/PPO 基本设置，不进入 ALOS 课程。
- **PC-G1b ALOS clean 门禁（C2/C3 的 clean profile）**：与同包线 `ALOS + PD/PI (20°)` 比较，clean 成功率 `≥0.95`，三维 RMSE 不超过其 `1.2×`；同时报告 `ψ、θ、u_g` 命令跟踪误差并与 teacher 对照。未通过先查 C1→C2 命令语义迁移和基本 PPO，不进入传感器强随机化。
- **PC-G1c 感知滚转门禁**：`|φ|` 的 p95 不超过 `10°`、最大值不超过 `30°`，`|p|` 的 p95 不超过 `30°/s`，超出 `φ_soft` 的时间比例低于 `5%`；`30°` 是强惩罚/质量门禁，不是安全终止阈值；安全终止只在 `|φ|>=φ_term` 持续 `T_phi_term` 或 `|p|>=p_term` 持续 `T_p_term` 时触发。若 PD/PI teacher 的正常转弯分布本身更宽，必须同时报告 teacher 分布并重新标定阈值，不能只放宽权重。严重超限或持续振荡按失败记录。
- **PC-G1d 终止语义门禁**：`terminated/truncated`、bootstrap 规则和 `termination_reason` 均通过单元测试；goal 不产生任务奖励，time limit 保留 bootstrap，安全终止只由明确安全阈值触发，普通 command error/饱和/DVL 失锁/USBL 野值不终止；安全终止惩罚不会让策略通过提前结束 episode 逃避负奖励。
- **PC-G2 realistic 门禁**：realistic 相对 clean 的成功率下降 `≤0.05`，三维 RMSE 退化 `≤0.20 m`，命令跟踪误差退化单独报告；同时报告 `p_nav` 误差均值/RMSE/95 分位、DVL 失锁期间误差增长量和 USBL 野值拒绝率；该阈值是 PC 首版 provisional gate，不替代旧 26D 门禁。
- **PC-G3 搜索门禁**：用 3 个训练 seed 在短预算上完成 `d_z`/学习率/探索标准差筛选，并先确认 BC 热启动没有把 PPO 锁死在 teacher 的局部行为；20° 包线下归一化动作速率和物理偏转都变了，不能直接复用 15° 的 action std、动作惩罚或饱和阈值。
- **PC-G4 正式门禁**：5 个训练 seed，评估 `0–49`、`50–99`，`2.0 m`/`1.25 m` 两档，并完成 clean/realistic 与 DADE on/off 的配对报告。
- **PC-G4b 视野与辅助损失门禁**：DADE 诊断分支必须至少完成一个 `gamma>=0.995` 组和一个辅助损失组，并与 `gamma=0.99` 主组同 seed、同预算配对；若长视野导致基础 PPO 不稳定，记录为负结果并回到辅助损失，不把 DADE 的短视野失败直接写成结论。
- **PC-G5 高速门禁**：H1–H3 按上面的 provisional 阈值逐级放行；H4 只作为包线压力测试。任何高速组启动前必须通过 20° 本体速度/舵效标定和命令可达性检查。

PC 与旧 I2 的差异是动作权限、动作包线和输入权限都不同，不能把 PC 的收益直接算成旧 DADE 的收益；反过来，旧 I2 的负结果也不能直接否决 PC，因为 PC 已经移除了冻结 BC/残差权限问题，并启用了更大的 20° 操纵包线。

#### 0.1.12.1 可选后置安全 RL：NP3O 消融（2026-11-02）

**定位：可选、后置、不阻塞主实验。** NP3O 只在 PC 主实验基本完成后启动，前置条件是 C0–C4、BC-only、`d_z`/学习率 screen、五组正式对照和 5-seed 评估均已完成或只剩最终汇总。它不改变 PC 主实验配置、checkpoint、DADE 结论和评估 seed，也不作为 PC-G0–G5 的通过条件。若时间和算力不足，直接记录为“未执行”，不能用一个未完成的安全 RL 分支拖住主结论。

本消融回答的问题是：**在相同任务奖励、动作包线、传感器 profile 和 BC 初始化下，加入 cost critic 与约束惩罚，能否降低安全约束违反，同时保持命令跟踪和路径成功率？** 它不以“提高违禁事件触达率”为目标。若目标是提高近边界或违规状态的覆盖，应使用独立的场景生成/近边界课程，而不是把 NP3O 的约束优化目标反过来优化违规率。

如果主 PC 实验已经通过 PC-G1c/PC-G2，且 safety termination、material saturation、滚转超限都接近 0，则可以直接跳过 NP3O 并记录“当前基线已满足安全需求”；跳过不等于 NP3O 失败，也不等于它有效。

**参考实现**

只读参考 TITA 的本地实现，不直接复制其 IsaacGym、足式机器人、深度教师或数据依赖：

- `/home/bubble/桌面/tita/tita_rl/algorithm/np3o.py`
- `/home/bubble/桌面/tita/tita_rl/runner/rollout_storage.py`
- `/home/bubble/桌面/tita/tita_rl/configs/tita_constraint_config.py`
- `/home/bubble/桌面/tita/tita_rl/NP3O_ALGORITHM_EXPLAINED.md`

参考代码可复用的核心是：奖励 critic + cost critic、多类 cost、cost GAE、clipped cost surrogate、归一化 violation term、逐 cost 的 `k_value` 权重和 `k_value` 渐增。不能照搬的是：足式机器人六类关节/接触 cost、`ActorCriticRMA`/深度教师/student 模块、IsaacGym rollout 假设，以及参考实现中 time-out 分支用当前 cost 而不是 cost value 做 bootstrap 的写法；PC 移植必须使用 `V_c(s_T)`，并用单元测试覆盖 `terminated/truncated` 两种路径。

需要特别说明：参考实现实际是“带 cost head 和归一化 cost violation 的 penalized PPO”，不是带显式拉格朗日乘子或 KKT 求解的 CPO/严格约束算法。因此实验结论只能写“降低 cost/违规率”，不能写“提供硬安全保证”。

**PC cost 设计**

首版建议 6 个 cost 通道，和当前奖励中的安全/平滑项一一对应，避免同一项既在 reward 又在 cost 中重复惩罚：

```text
c_action_mag   = mean(relu(|a_req| - 1))
c_action_rate  = mean(relu(|Δa_req| / rate_norm - 1))
c_action_delta = mean(||Δa_norm||²)
c_roll         = (relu(|φ| - φ_dead) / (φ_strong - φ_dead))²
c_roll_rate    = (relu(|p| - p_dead) / (p_soft - p_dead))²
c_safety_term  = 1 if terminated by safety event else 0
```

cost 使用仿真真值 `φ,p` 和请求动作 `a_req` 计算，actor 仍只能看到带噪 IMU；DVL 失锁、水跟踪无效、USBL 野值和海流估计误差是外部扰动，不得作为“策略违规 cost”。所有 cost 必须有限、按固定尺度归一化并按通道裁剪，不能按 episode 或 batch 在线重标定；`cost_d_values` 首版全设为 0 做硬约束 screen，若策略明显过度保守，只允许对 `c_roll`、`c_roll_rate` 和 `c_action_delta` 这类软约束单独给非零预算，硬件幅值/速率和 terminal safety 仍保持 0。

终止和 cost return 必须严格区分：`truncated` 用 `V_c(s_T)` 做 cost bootstrap；`terminated` 不 bootstrap，安全终止额外加 `c_safety_term`，正常到达终点不加安全 cost。cost GAE 不能复用 reward GAE 的 done/value 逻辑；参考实现中的 time-out 分支必须修正后再移植。

**Cost critic 与损失**

首版 actor 完全不变，actor/critic 顶层 `[z_t,c_t,ĉ_t]` 宽度不变，不新增 actor 观测，也不改变 actor DADE。新增 `K=6` 维 cost head：

```text
V_c(s_t^c) = CostHead(stopgrad([z_t^c, c_t, ĉ_t^c])) ∈ R^6
```

优先使用 stop-gradient，避免 cost 损失反向改变 critic encoder/DADE 表征，从而把 NP3O 与 DADE 的效应混在一起；若 cost head 明显欠拟合，再增加独立 cost encoder，而不是让它改写 actor 或 actor DADE。cost head 输出使用 softplus 保证非负，训练使用 cost value loss；策略损失沿用参考实现的 clipped cost surrogate + 归一化 violation term：

```text
L = L_PPO
    + λ_V * L_V
    + λ_Vc * L_Vc
    + λ_viol * Σ_k k_k * relu(mean(A_c,k * ratio_clipped) + v_k)
    + entropy
```

其中 `v_k` 使用 `(1-γ)(R_c,k - d_k)` 与 cost advantage 的归一化形式，`k_k` 从较小值按参考实现的 ramp screen 到 1；首版 `cost_value_loss_coef`、`cost_viol_loss_coef` 从参考的 `0.1` 附近起 screen，不直接固定为最终值。`k_value`、`λ`、cost 权重和 `d_values` 必须写入配置与日志，不能只存在于代码默认值。

为了避免“NP3O 有效”其实只是“多加了惩罚”的混淆，NP3O arm 的 reward 移除与 cost 重复的滚转、动作平滑、饱和和稀疏 `safety_penalty` 项，只保留命令跟踪（以及 C0/C1 的 hold）；同时必须增加一个 `PC PPO + scalarized costs` 控制组，把这 6 个 cost 以同一固定权重的标量惩罚直接加进 reward。只有 NP3O 相对这个控制组仍然降低 cost，才能把收益归因于约束优化，而不是额外惩罚。

**最小实验矩阵**

所有组使用同一个 BC 初始化、同一 seed 集合、同一传感器 profile、同一 `pc20` 动作包线和同一训练预算：

1. `PC PPO without DADE`：主实验现有基线；
2. `PC PPO + scalarized costs without DADE`：相同 cost 直接折进 reward 的控制组；
3. `PC NP3O without DADE`：第一优先的 NP3O 消融，用来隔离安全优化器；
4. 只有第 3 组在初步 screen 中不劣于第 2 组，才增加 `PC NP3O with DADE`，不得一开始就把 NP3O 和 DADE 两个因素混在一起。

主 screen 先跑 3 个训练 seed 的短预算；通过后才进入 5-seed 正式评估。若 NP3O 没有降低 cost、或必须牺牲明显跟踪性能才能降低 cost，按负结果记录，不继续增加 cost 通道或权重来“调出”正结果。

**PC-G6（可选 NP3O 门禁，provisional）**

- 任务性能：相对同 seed、同预算的匹配 PPO 控制组，成功率下降不超过 `0.03`，三维 RMSE 不超过 `1.10×`，命令跟踪稳态误差/超调不超过 `1.10×`，平均到达时间不超过 `1.10×`；
- 安全性能：若基线 `safety_termination_rate` 或滚转超限时间非零，目标至少相对降低 `30%`；material saturation 非零时目标相对降低 `50%`；基线已经接近 0 时改用绝对阈值和 cost episode 均值/p95，不制造虚假的相对改善；
- 不得用过度保守换取低违规：速度跟踪、路径完成率、time-to-goal 和 false safety termination 都不能明显恶化；
- 数值稳定：cost value loss、violation loss、`k_value`、cost advantage 和 cost return 全程有限，无 NaN/Inf，cost bootstrap 在 `terminated/truncated` 下均通过单元测试；
- 归因：NP3O 至少要在某个安全指标上优于 `PC PPO + scalarized costs`，否则只能写“额外惩罚有效/无效”，不能写“NP3O 有效”。

NP3O 的风险主要不是学习率，而是 cost 定义、cost/done 语义和双目标权重失衡。若 cost 把正常机动误判为违规，策略会变保守；若 terminal cost 太弱，策略会靠提前终止逃避负奖励；若 cost head 与 reward critic 共用梯度而未审计，DADE 结论会被污染。以上任一出现都应停止该分支，而不是把它混入 PC 主结果。

#### 0.1.13 实施顺序与 Git

执行顺序：

- [x] 新增 PC 配置与 schema：`configs/lowlevel/odarrl_control_pc_*.json`，显式标记 `arm="PC"`、`policy_mode="pure_controller"`、`actuator_envelope="pc20"`，舵/艉舵满量程均为 `20°`；新 PC 使用独立 vehicle profile，不复用 26D 的 observation metadata，也不修改旧 `remus100_vehicle()` 的 15° 契约；
- [x] 新增 PC-G0 启动硬校验：`guidance.compensate_current=false`、`speed_feedback='ground'`、`u_d` 为地速命令、`actuator_envelope="pc20"`、`policy_max_rudder/stern=20°` 且 `plant_max_rudder/stern>=20°`；任一不满足直接拒绝启动，不能静默依赖后端裁剪；
- [x] 新增 actor/critic 分离的状态编码器：输入宽度可配置，输出总 `d_z`；首版为 `o_t` 追加 8 个恒零输入槽，`z_t` 末尾保留 4 个恒零输出槽，并保存各自 `active_mask`、`d_z_active`、`n_in_reserved`、`n_z_reserved` 和 schema hash；
- [x] 新增编码器槽位测试：零输入槽不参与归一化统计和梯度、零输出槽严格为零、`active_mask` 可复现、checkpoint 拒绝 mask/槽数不匹配、启用一个保留槽后 shape 不变但仍能完成训练前向、高维新观测不能硬塞进标量槽；
- [x] 新增 actor/critic 分离 DADE：独立权重、独立 buffer、各自 64 维输出；主实验 actor DADE 输入 `[o_t^a, v̂_c^b]` 历史，critic DADE 输入 `[o_t^c, v_c^n]` 历史；
- [ ] 完成 v1-main 控制步对齐 DADE：所有原生信号经零阶保持和延迟队列映射到 50 Hz token；`valid/age` 只记录、不进入主实验网络；
- [ ] 记录原生采样率与 token 采样率的映射表，禁止把零阶保持后的重复值当成同频真实测量；
- [ ] 仅在 v1 失败后做 v2 多采样率消融：快状态流与慢海流流分别编码，比较 late fusion 和 cross-attention；
- [ ] 完成 PC 折扣/GAE 与辅助损失 screen：至少 `gamma ∈ {0.99,0.995,0.998,0.999}`，固定 `lambda=0.95` 做首轮，并增加一个辅助损失组；所有组记录 `gamma`、`lambda`、窗口和 seed；
- [x] 新增纯控制器动作 adapter：`[a_r,a_s,a_n] → [δ_r,δ_s,n]`，使用 `δ_r,δ_s=20°·a`、`n=1525·0.5·(a_n+1)`，记录请求量、实际量和裁剪原因；
- [x] 新增 PC20 后端契约测试：batched plant 的本体限幅必须显式参数化为 `20°`；MSS 抽查使用 20°；PVS 因内部 15° 上限不得作为 PC20 主训练/正式评估后端，只保留 15° legacy 对照；
- [x] 新增 C0/C1 无 ALOS 命令跟踪环境：复用同一 actor 接口，奖励改用命令误差，零海流，成功按保持时间、超调和稳态误差判定；
- [x] 新增 PC 奖励 v2 配置：统一 `reward_mode="pc_command_track"`；C0/C1 打开 hold bonus，C2–C4 关闭 hold bonus；C2–C4 只含命令误差、滚转、动作平滑、饱和和安全惩罚，禁止进度、路径误差、曲率、`P_ref`、goal bonus、timeout penalty 进入奖励；
- [x] 新增 PC 奖励单元测试：C2–C4 路径项恒为 0、地速语义、动作使用实际执行器、滚转软约束、出界/NaN 安全惩罚、goal/timeout 不产生任务奖励；
- [x] 新增 PD/PI BC 数据采集：teacher 与 actor 使用相同地速命令和 `[δ_r,δ_s,n]` 动作语义；teacher 必须实例化为 20° 包线并重新采集，不能复用 15° teacher 的 BC 数据；以 C0/C1 闭环 episode 为主，可在 C2 增采少量 ALOS 闭环数据，不只采集随机单帧；C0–C3 闭环示教已按阶段重采并合并，见 2026-11-03 记录；
- [ ] 新增 BC 训练与 BC-only 闭环评估：离线动作误差和闭环跟踪分开报告；若 BC-only 闭环发散，最多补一轮 PC 专用 DAgger；
- [ ] 新增 PPO 从 BC 权重热启动：不冻结 actor、不做 residual、不改变动作语义；记录初始化 checkpoint 和探索噪声；
- [x] 新增全合成传感器 wrapper：DVL 底/水跟踪、IMU、原始 USBL/INS、执行器、海流估计、时间戳/延迟/失锁/野值；全部由仿真真值驱动，PC v1 不依赖原生 Isaac Sim/OceanSim 传感器；原始 USBL 只用于生成导航校正和诊断，不直接进入 actor；海流估计只送入 actor DADE；真实海流只送入 critic DADE，奖励和辅助损失按各自契约使用真值；
- [x] 新增传感器来源矩阵：PC v1 每个 actor/critic 输入统一标记 `sensor_source="synthetic"`，记录 profile、seed、坐标变换、延迟/失锁参数和验证证据；schema 预留 `native_isaac_oceansim` 但本阶段不实现，不将任何原生传感器接入作为主训练前置条件；
- [ ] **后置可选，不阻塞 PC v1**：未来再做 OceanSim 原生 DVL/IMU/气压计“原生 vs 合成”对照；必须使用相同观测 schema、坐标变换、固定测试输入和评估 seeds，先做单环境语义验收，再决定是否纳入训练；
- [x] 新增 `SyntheticNav` 导航误差 wrapper：按 0.1.7.1 生成 `p_nav`，包含 episode 固定零偏、O-U 慢漂移、USBL 低频校正、DVL 失锁增长、野值拒绝/注入和最终延迟队列；actor 只读取 `p_nav`，不得读取 `p_true`；
- [x] 新增 `SyntheticNav` 单元测试：clean 恒等、USBL 锯齿校正、DVL 失锁误差增长、USBL 缺失继续传播、野值拒绝与未拒绝双路径、延迟队列、profile/seed 可复现、无 NaN/Inf；
- [x] 新增固定归一化与 checkpoint metadata：保存 `position_scale_m`、各速度/角度尺度、`action_full_scale=[20°,20°,1525 rpm]`、plant 限幅、传感器 profile、`active_mask`、保留槽数和 schema hash；旧 15° checkpoint 或槽位 mask 不一致的 checkpoint 不能静默加载为 PC20；
- [x] 新增 PC20 单元测试：动作逆映射、±20° 幅值裁剪、`150°/s` 速率裁剪、`0.2°` material tolerance、旧 15° checkpoint 拒绝、PVS plant 拒绝，全部覆盖；
- [x] 新增滚转约束奖励与安全测试：`φ_dead/φ_soft/φ_strong` 平滑分段、`p` 速率惩罚、`φ_term/T_phi_term` 与 `p_term/T_p_term` 持续超限终止、瞬时超限不终止、clean 与 realistic 下滚转指标统计；先用 PD/PI teacher 日志标定阈值，再固定权重；
- [x] 新增 PC episode termination manager：区分 goal/time-limit/safety，输出 `terminated`、`truncated`、`termination_reason` 和连续超限计时；覆盖 goal 无任务奖励、timeout bootstrap、NaN/Inf/出界/危险深度/持续滚转、普通误差/饱和/失锁不终止、不同 `dt` 下计时与复位，以及安全终止惩罚的逃逸检查；
- [x] 新增泄漏测试：actor 的 `o_t^a`/`c_t` 无法读取 `current_true_ned` 或 `v̂_c^b`，因此 `z_t^a` 也不含海流；actor 的 `ĉ_t^a` 只能来自估计历史；critic 的 `o_t^c`/`c_t` 无法读取 `current_true_ned`，因此 `z_t^c` 也不含真实海流，critic 的 `ĉ_t^c` 只能来自真值历史；
- [x] 新增 ALOS 导航泄漏测试与 C2 启动校验：路径投影、`P_ref`、终端捕获和 `c_t` 都只能读取 actor 同源的 `p_nav`，无法读取 `p_true`/精确 `eta`；clean profile 下与真值基准等价，realistic profile 下误差影响可解释且可复现；
- [x] 新增 history/dropout 测试：`current-only`、`current-zero`、`current-frame dropout`、delay/lock-loss 均能复现；新增 `tests/test_lowlevel_pc_history_ablation.py` 与 `scripts/diagnose_pc_dade_history.py`，环境层 dropout 和传感器延迟/失锁测试同时保留；
- [x] 完成 PC-G1a 命令跟踪 smoke：C0 clean、C1 加小噪声和随机命令，两阶段均为零海流且 DADE 关闭；50-seed BC-only 的 `command_hold_success_rate` 为 C0 `1.000`、C1 `0.960`，material saturation 为 `0.0156/0.0334`，安全终止率均为 `0`，闭环未发散；
- [x] 完成 PC-G1b ALOS 切换 smoke：C2 简单路径、小常值流、DADE 关闭的混合 C2 BC 在 clean `0–49/50–99` 成功率均为 `1.0`，三维 RMSE 为 teacher 的 `1.048×/1.044×`；确认 C1 命令语义可迁移到 ALOS 路径跟踪；
- [x] 完成 C3 传感器真实化 smoke：至少覆盖 DVL 失锁、IMU 漂移、USBL 跳点、执行器滞后和海流估计延迟；C4 只建立固定压力测试集，不混入 C3 主训练统计；C3 v3 已通过 64-seed teacher/BC/PPO 闭环运行，realistic_calibrated 的 DVL bottom outage `2.01%`、water outage `4.18%`、current estimate valid `96.53%`、USBL outlier `1.87%`，且无安全终止；
- [x] 完成 `d_z`/学习率/探索标准差 screen：采用 successive halving，`d_z=64`、actor lr `1e-5`、critic lr `3e-4`、`init_noise_std=0.1` 在三 seed 短预算上稳定；16/128 及非最终 lr/std 候选在首 seed 淘汰，详细结果见 2026-11-03 PC-G3 小节；
- [x] 完成 PC20 高速包线标定：同 `REMUS100`、`±20°`、`1525 rpm`、`dt=0.02 s` 下测量 rpm–速度曲线、8 方向海流可达性、20° 舵/艉舵阶跃和收油滑行；H0–H4 的命令可达性过滤全部通过；
- [ ] 完成 H1–H4 高速课程：在已完成的标定和 H0–H4 teacher/BC 阶梯之上，按 H1→H2→H3 逐级 smoke 并训练残差；H4 只做固定压力评估，且单独报告 2.5 m/s 的导航延迟、饱和和最坏轨迹；
- [ ] 完成速度×海流解耦验证：训练域与留出域分离，至少留出“高速+强流”组合；C4 只作压力测试，并用 held-out 成功率、RMSE、strict125 与 saturation 联合判定是否真正缓解过拟合；
- [ ] 完成五组正式对照：`ALOS+PD/PI (20°)`、`PC BC-only (20°)`、`PC BC+PPO no DADE (20°)`、`PC BC+PPO with DADE (20°)` 和旧 `I2 DADE (15°)` 历史对照；预算允许时再加 `PC from-scratch PPO no DADE (20°)`；
- [ ] 完成 5-seed 正式评估：clean/realistic、DADE on/off、2.0/1.25 m、`0–49`/`50–99`；
- [ ] **后置消融，不阻塞主实验**：在上述主实验基本完成后，增加 `v1-valid-age`，把 `[valid, age]` 追加到 actor DADE token；用相同 seed、传感器 profile 和网络结构，与 `v1-main` 配对比较；
- [ ] **后置可选安全 RL，不阻塞主实验**：上述 PC 主实验和 5-seed 评估基本完成后，按 0.1.12.1 执行 NP3O 最小矩阵（`PPO`、`PPO + scalarized costs`、`NP3O without DADE`；通过 short screen 后再考虑 `NP3O with DADE`）。NP3O 不属于 PC-G0–G5，也不参与 DADE 主结论归因；
- [ ] 写结果时保留负结果：若 PC 不如 ALOS+PD/PI，或 DADE 没有历史依赖，按失败记录，不把“用了 RL”写成收益。

**2026-11-02 PC 基础设施进度（代码级，不等于性能门禁）**

- C0–C4 环境配置与 PPO 配置已建立；C0–C4 均完成 1-iteration、8 env、8 step 的 CPU smoke，realistic/strong/ALOS 运行均无非有限观测、动作或奖励；
- PC 专用测试 `tests/test_lowlevel_pc_*.py` 当前为 `45 passed`，Ruff 对 PC 核心文件通过；
- C2–C4 checkpoint 已核验 `observation_schema=pc_v1`、actor/critic 宽度 `915/1098`、`pc_stage`、`sensor_profile`、`sensor_source=synthetic`、`actuator_envelope=pc20` 和 `action_full_scale=[20°,20°,1525 rpm]`；
- 以上只证明环境、网络、checkpoint 和契约可运行。PC-G1a/G1b/G2 的闭环成功率、误差、滚转和 realistic 退化仍需真实训练后判定，不能把 64-step smoke 记成门禁通过。

**2026-11-03 C2 评估口径修正与稳定性 screen（进行中）**

- `scripts/evaluate_pc_bc.py` 和 `scripts/evaluate_pc_ppo.py` 的 `path_*` 指标已改为把真实 `eta[:, :3]` 投影到参考路径；ALOS 基于 `p_nav` 的投影改名为 `alos_projection_*`，仅作诊断。此前把 ALOS 投影直接当作真值路径误差，C2 `mild` profile 下会轻微美化结果；修正后 C2 teacher/BC/PPO 的三维路径 RMSE 分别变化约 `-0.065/-0.033/-0.037 m`，不是主要结论来源。
- C2 ALOS teacher 已把 `terminal_capture_distance_m` 从 `16` 调到 `20 m`、`horizontal_gain` 从 `1.0` 调到 `2.5`；原参数在 16 m 终端捕获时仍有约 `1.2 m/s` 速度，容易越过目标。新 teacher 在 64 个固定评估 seed 上成功率为 `1.0`，三维真实路径 RMSE `1.871 m`，material saturation `0.185`，无安全终止。teacher 的高饱和率必须保留报告，不能隐藏。
- BC 热启动曾把 DADE 的 64 个输入列固定为零，导致这些列从未训练、DADE 一启用 actor 均值爆炸。现改为 `PC_DADE_WARMSTART_INPUT_SCALE=0.01`，保留梯度路径；metadata 记录 `actor_dade_input_columns_scaled=true` 和实际 scale。该修复不改变 C0/C1 BC 的 actor 输出语义。
- C2 固定 64 个评估 seed 的闭环结果（成功率按真实 `eta` 的 2.0 m goal，路径误差按真实 `eta` 投影）：

| policy | success | 3D path RMSE | horizontal RMSE | material saturation |
|---|---:|---:|---:|---:|
| ALOS + PD/PI teacher | `1.000000` | `1.871269 m` | `1.781949 m` | `0.185376` |
| C0/C1 BC-only | `0.875000` | `3.136380 m` | `1.992991 m` | `0.126464` |
| DADE PPO `model_10` | `0.890625` | `3.233730 m` | `2.271038 m` | `0.120138` |
| DADE PPO `final` | `0.671875` | `5.028059 m` | `3.566131 m` | `0.110399` |
| no-DADE PPO fixed `model_0` | `0.875000` | `3.105712 m` | `1.998588 m` | `0.125207` |
| no-DADE PPO fixed `model_10` | `0.859375` | `2.702574 m` | `1.940495 m` | `0.128221` |
| no-DADE PPO stable `model_0` | `0.890625` | `2.995679 m` | `2.005192 m` | `0.126852` |

- DADE PPO 的训练日志显示后段明显失稳：`approx_kl` 多次达到 `2–8`，`ratio_max` 最高超过 `1,700`；成功率的闭环轨迹为 `ck0/10/20/30/31 = 0.875/0.891/0.891/0.781/0.672`。这不是单个最终点的偶然，因此**不能把 `final.pt` 直接作为 C2 部署点**，必须按固定、与训练 seed 不相交的评估集选择 checkpoint。
- 仅关闭 DADE 并未解决问题：固定更新配置的 no-DADE 仍有 `approx_kl≈2–4` 和 `ratio_max` 达数十到上百；保守 no-DADE 组把 KL 压到约 `0.003–0.028`，但成功率仍只有 `0.84–0.89`。当前证据表明主要风险是 actor/历史编码器一次更新跨度过大以及 C2 分布迁移，而不是 DADE 单独导致。
- 已采集 C0+C1+C2 的 teacher 闭环示教 `196,608` 样本、`147` episodes（`artifacts/lowlevel/pc_bc_demos_c0_c1_c2_v3_20261103/`）；混合 BC 的 best epoch 为 `191`，checkpoint SHA-256 `51cd5dec…66efc`。它将 C2 的真实 ALOS 命令分布补进 BC，解决了旧 C0/C1 BC 的 C2 迁移缺口：

| policy / seed pool | success | 3D path RMSE | horizontal RMSE | material saturation |
|---|---:|---:|---:|---:|
| 混合 C2 BC，mild，`0–63` | `1.000000` | `1.923502 m` | `1.836946 m` | `0.178240` |
| 混合 C2 BC，mild，`50–99` | `0.980000` | `2.485654 m` | `1.877005 m` | `0.174020` |
| teacher，mild，`50–99` | `0.980000` | `2.174922 m` | `1.986543 m` | `0.182323` |
| 混合 C2 BC，clean，`0–49` | `1.000000` | `1.670529 m` | `1.601762 m` | `0.044538` |
| teacher，clean，`0–49` | `1.000000` | `1.594587 m` | `1.526755 m` | `0.045899` |
| 混合 C2 BC，clean，`50–99` | `1.000000` | `1.617544 m` | `1.546728 m` | `0.050029` |
| teacher，clean，`50–99` | `1.000000` | `1.550017 m` | `1.479294 m` | `0.051635` |

- 混合 C2 BC 在 clean 的 `0–49` 和 `50–99` 上成功率均为 `1.0`，三维 RMSE 分别为 teacher 的 `1.048×` 和 `1.044×`，满足 PC-G1b smoke 的“成功率 ≥0.95、三维 RMSE ≤1.2× teacher”。这证明 C1 命令语义可以迁移到 C2 ALOS；但正式 PC-G4 仍需 PPO/DADE、1.25 m 严格捕获和更多训练 seed，因此只把 PC-G1b smoke 记为通过，不把整个 C2 写成完成。
- 从混合 C2 BC 出发，用相同 stable PPO 配置（2 epochs、`lr=5e-5`、`clip=0.05`、adaptive KL、32 iterations、64 env）分别训练 DADE on/off，并以 `ck20` 做同 seed `91205` 配对评估：

| eval pool | DADE 3D RMSE | no-DADE 3D RMSE | DADE − no-DADE | success | material saturation |
|---|---:|---:|---:|---:|---:|
| mild，`0–63` | `1.922655 m` | `1.975303 m` | `-0.052648 m`（`-2.7%`） | `1.00 / 1.00` | `0.171590 / 0.169043` |
| mild，`50–99` | `2.359889 m` | `2.455385 m` | `-0.095496 m`（`-3.9%`） | `0.98 / 0.98` | `0.167634 / 0.166032` |
| clean，`0–49` | `1.672972 m` | `1.746365 m` | `-0.073393 m`（`-4.2%`） | `1.00 / 1.00` | `0.041186 / 0.042736` |
| clean，`50–99` | `1.632777 m` | `1.693716 m` | `-0.060939 m`（`-3.6%`） | `1.00 / 1.00` | `0.046638 / 0.046613` |

- 同 seed 配对下 DADE 在四个池上方向一致，成功率不降；但效应只有约 `2.7%–4.2%`，远小于“明显改变行为”的量级。`ck0/10/20/30/31` 的 screened RMSE 仍存在 checkpoint 波动（DADE `1.911/1.929/1.923/1.966/1.953 m`，no-DADE `1.926/1.943/1.975/1.922/1.917 m`），所以不能只报告单个最优点。
- actor 第一层中 DADE 64 列权重范数从 BC warm-start 的 `0.076576` 增至 `ck20=0.082720`、`ck31=0.102146`；同期 `z_t` 输入权重范数约 `7.96`。DADE 路径确实被训练、且配对收益可见，但相对主导输入仍很小。若 C3 realistic 下效应仍只有几个百分点，下一步应优先做 DADE 辅助损失/显式特征缩放或 current-prediction 目标，而不是单纯延长训练；NP3O 仍保持后置可选。

**2026-11-03 C3 v3：导航校准、USBL 门限与真实传感器链路修正**

- 先固定 C3 PPO 配置的引用：`odarrl_control_ppo_pc_c3_realistic_{dade,nodade}_stable.json` 不能再指向旧 v1 环境配置；v2 结果保留为历史，新主训练使用 v3。
- 64-seed teacher/BC 对照表明 C3 的瓶颈不是 BC：在 v2 上 student 与 teacher 成功率完全同为 clean `0.75`、realistic `0.484`，三维 RMSE 仅差约 `0.25/0.61 m`。因此不能靠继续训练 BC 解决 realistic 失败。
- 轨迹级诊断定位到两个传感器模型问题：
  - 旧 USBL 野值门限为 `5σ+3 m`；当 `σ=5 m` 时门限可达 `28 m`，`10–25 m` 野值会被当成有效测量。现改为 `max(σ+3 m, 5 m)`，并新增高噪声野值拒绝测试。
  - 旧 `realistic` 导航在任务开始时直接带着最多 `2 m` 固定偏置，再以固定 `0.35` 增益追随高噪声 USBL；轨迹中 `p_nav` 垂直误差可达约 `4 m`，ALOS 因而在应下潜时抬头。旧 `realistic` profile 原样保留，不覆盖历史结果。
- 新增独立 profile `realistic_calibrated`，作为 C3 v3 当前候选传感器域：
  - 预任务导航校准消除 `80%` 固定偏置，保留 `20%` 残余偏置；真实任务在入水前做 USBL/INS 对准，不能默认把全量安装偏置带进闭环。
  - USBL 融合增益从 `0.35` 降为 `0.03`；在其他噪声/失锁/延迟字段不变时，64-seed 导航 RMSE 从旧约 `2.49 m` 降到 `0.64–0.67 m`。
  - clean profile 语义、旧 `realistic`、旧 C2/C3 结果和旧 checkpoint 均不变；新实验必须显式使用 `realistic_calibrated` 或 v3 配置。
- **2026-11-07 传感器 profile 修订**：`realistic_calibrated` 已重定义为 nominal
  传感器域；上述 C3 v3 历史语义保留为 `realistic_calibrated_legacy`，旧 `realistic`
  极端范围新增显式别名 `realistic_stress`。新实验优先使用
  `configs/lowlevel/odarrl_control_pc_c3_random_realistic_v1_nominal_20261107.json`，
  压力测试使用 `..._v1_stress_20261107.json`。详见
  `doc/goals/goal21_sensor_profiles_nominal_stress_20261107.md`。
- 新配置 `configs/lowlevel/odarrl_control_pc_c3_realistic_v3.json` 指向 `realistic_calibrated`；C3 DADE/no-DADE stable PPO 配置已改为引用 v3。v2 只保留历史诊断，不再作为新训练入口。
- guidance 小范围 screen 显示末端捕获不能太小：`5 m` 会把无流成功率降到约 `0.06–0.13`，`15 m` 在强流下也不稳；`20–25 m` 与 `horizontal_gain=2.5` 是当前折中点。不能为了提高有流成功率把捕获半径降到 `1.25 m`。
- v3 64-seed teacher 结果（ground-speed feedback，`20°` 包线）：

| profile | current | success | 3D RMSE | goal dist p95 | nav RMSE |
|---|---|---:|---:|---:|---:|
| clean | none | `1.0000` | `1.449 m` | `1.999 m` | `0.000 m` |
| clean | config | `0.7812` | `4.326 m` | `21.172 m` | `0.000 m` |
| realistic_calibrated | none | `1.0000` | `1.461 m` | `1.998 m` | `0.637 m` |
| realistic_calibrated | config | `0.6875` | `5.091 m` | `24.876 m` | `0.664 m` |
| realistic_calibrated | uniform | `0.7344` | `4.419 m` | `21.358 m` | `0.657 m` |
| realistic_calibrated | timevarying | `0.7812` | `4.027 m` | `20.326 m` | `0.658 m` |
| realistic_calibrated | shear | `0.6094` | `5.839 m` | `26.302 m` | `0.674 m` |

- 旧 C3 BC v2 在新 v3 上仍可用但不等价：teacher `0.6875`，student `0.65625`，三维 RMSE `5.091 → 5.692 m`。这说明传感器分布修正没有破坏 BC 接口，但正式 C3 BC 仍应重采 v3 演示，不能用旧 v2 数据代替最终结果。
- 当前仍未通过 C3 完整门禁：有流成功率 `0.61–0.78`，且随机路径尚未实现；v3 只是修复了不可信的导航误差和末端捕获参数屏幕。下一步优先训练 DADE/no-DADE 对比，观察历史编码是否能学习海流与导航残差补偿，而不是继续放宽成功半径。
- 新增/修改测试：`tests/test_lowlevel_pc_sensors.py` 覆盖高噪声 USBL 野值拒绝、预任务偏置校准；相关 PC 传感器/teacher/env 测试当前 `21 passed`，Ruff 通过。

**2026-11-03 C3 v3 BC/PPO 短程证据与随机路径门禁**

- C3 clean/realistic v3 teacher 示教已重采：`artifacts/lowlevel/pc_bc_demos_c3_clean_realistic_v3_20261103/demos.npz`，共 `131,072` 样本、`97` episodes。realistic_calibrated 统计为 nav RMSE `0.67676 m`、DVL bottom outage `2.01%`、DVL water outage `4.18%`、current estimate valid `96.53%`、USBL outlier `1.87%`，与 v3 环境分布一致。
- 与 C0–C2 v3 合并后得到 `artifacts/lowlevel/pc_bc_demos_c0_c1_c2_c3_v3_20261103/demos.npz`，`327,680` 样本、`244` episodes，dataset SHA-256 `fbba18e3b959cc00fbea8fa86355746c95f876a04de082a67ecd2c22e0a1e87d`。混合 BC checkpoint 为 `artifacts/lowlevel/pc_bc_c0_c1_c2_c3_v3_20261103/pc_bc_actor.pt`，SHA-256 `08d7ec1269214da9ef540f598c85cfd9917f66523a783427c24f5c6400775a08`；best epoch `180`，validation normalized RMSE `[0.10404, 0.06540, 0.02365]`，physical RMSE `[0.036318, 0.022828, 18.0355]`。
- C3 v3 64-seed teacher/BC 闭环门禁（固定路径）：

| profile | policy | success | 3D RMSE | material saturation | speed RMSE | yaw RMSE |
|---|---|---:|---:|---:|---:|---:|
| clean | teacher | `0.78125` | `4.325739 m` | `0.037414` | `0.414473` | `0.727106` |
| clean | BC | `0.78125` | `4.229354 m` | `0.036519` | `0.424706` | `0.727995` |
| realistic_calibrated | teacher | `0.687500` | `5.090602 m` | `0.077178` | `0.414448` | `0.808636` |
| realistic_calibrated | BC | `0.625000` | `5.654386 m` | `0.135579` | `0.420691` | `0.887132` |

clean 的 BC 成功率与 teacher 持平且 RMSE 略低；realistic 的 BC 成功率低 `6.25` 个百分点、饱和率明显更高，说明 BC 可用但必须继续 PPO 微调，不能把 BC-only 当成 C3 完成。
- 固定路径 C3 v3 的 no-DADE 短训练（64 env × 256 step × 32 iter = `524,288` steps，seed `91302`）已经证明 PPO 能改善 BC：`ck0 → ck20/ck30` 的成功率 `0.640625 → 0.71875/0.734375`，三维 RMSE `5.784851 → 5.069714/4.924537 m`。但 material saturation 仍约 `0.14`，不能视为安全门禁通过。
- 固定路径 DADE 短训练（同预算，seed `91305`）的 scheduled-checkpoint 结果为：

| arm | ck0 success | ck10 success | ck20 success | ck30 success | ck31 success | ck20 3D RMSE | ck20 saturation |
|---|---:|---:|---:|---:|---:|---:|---:|
| no-DADE | `0.640625` | `0.718750` | `0.718750` | `0.734375` | `0.718750` | `5.069714 m` | `0.141114` |
| DADE | `0.625000` | `0.609375` | `0.609375` | `0.593750` | `0.578125` | `5.570727 m` | `0.131528` |

DADE 在这组短程固定路径上降低了饱和率约 `1` 个百分点，但成功率下降约 `11` 个百分点，三维 RMSE 也未改善。因为两组训练 seed 不同（`91302` vs `91305`），这不能当作严格的同 seed 配对结论；它只能登记为“短程 DADE 未显示控制收益”，正式归因必须等同 seed 随机路径配对结果。
- 随机路径基础设施已经实现并保持默认兼容：`ReferenceConfig` 新增 `path_bank_size/path_bank_seed`；`generate_sinusoidal_3d_path_bank()` 生成同采样数、同长度量级、首条严格等于旧固定路径的 deterministic bank；`BatchedReferencePathBank/Set` 支持逐环境 `project`、`point_at_s`、`curvature_at_s` 和 start/goal/tangent 查询；`BatchedPCLowLevelEnv` 在 reset/auto-reset 时按 `episode_seed % path_bank_size` 选路。默认 `path_bank_size=1` 时所有旧配置和旧结果不变。
- 新增测试覆盖：默认固定路径逐点一致、bank 可复现、等长与采样间距、单路径/路径集投影等价、windowed projection、逐 seed 选路、subset reset 和 auto-reset 路径一致性。当前 PC/路径/guidance/config 回归为 `119 passed`，Ruff 通过。
- 新配置 `configs/lowlevel/odarrl_control_pc_c3_random_clean_v1.json` 与 `configs/lowlevel/odarrl_control_pc_c3_random_realistic_v1.json` 使用 `path_bank_size=32`、`path_bank_seed=41`，不覆盖固定路径 v3 结果。64-seed teacher 门禁：

| profile | success | 3D RMSE | material saturation | nav RMSE |
|---|---:|---:|---:|---:|
| random clean | `0.78125` | `4.373625 m` | `0.034284` | `0.000 m` |
| random realistic_calibrated | `0.734375` | `4.616441 m` | `0.071773` | `0.667609 m` |

随机路径没有造成 teacher 崩溃；realistic 成功率反而略高于固定路径 `0.6875`，因此可以作为下一阶段环境，而不是靠放宽成功标准换取结果。
- 随机路径 C3 示教与 BC：`artifacts/lowlevel/pc_bc_demos_c3_random_clean_realistic_v1_20261103/demos.npz`，`131,072` 样本、`97` episodes，dataset SHA-256 `59052de5a1808eaf4b08c084ebc5cdb53be867fe25b1bbd8c685e856c2fc48df`；与 C0–C2 v3 合并为 `artifacts/lowlevel/pc_bc_demos_c0_c1_c2_c3random_v1_20261103/demos.npz`，dataset SHA-256 `eeac3e63266a1a96451d913f23c5b8df5ee53f5db68a6efbf94367ddedda3ded`。随机路径 BC checkpoint `artifacts/lowlevel/pc_bc_c0_c1_c2_c3random_v1_20261103/pc_bc_actor.pt`，SHA-256 `3fd41bae51dd3666d99b93a6d331a5337f4a477fc980ee008e4c8641cb053fda`，best epoch `168`，validation normalized RMSE `[0.09775, 0.07015, 0.02550]`。
- 随机路径 64-seed teacher/BC 结果：

| profile | policy | success | 3D RMSE | material saturation | speed RMSE | yaw RMSE |
|---|---|---:|---:|---:|---:|---:|
| random clean | teacher | `0.78125` | `4.373625 m` | `0.034284` | `0.417942` | `0.762834` |
| random clean | BC | `0.81250` | `4.191449 m` | `0.033446` | `0.415348` | `0.742341` |
| random realistic_calibrated | teacher | `0.734375` | `4.616441 m` | `0.071773` | `0.420012` | `0.810000` |
| random realistic_calibrated | BC | `0.718750` | `5.216060 m` | `0.126575` | `0.414618` | `0.851855` |

随机路径 BC 在 clean 上超过 teacher；realistic 成功率仅低 `1.56` 个百分点，但三维 RMSE 为 teacher 的 `1.13×`、饱和率为 `1.76×`，因此进入 PPO 微调而不是直接验收。
- 随机路径 DADE/no-DADE 的同 seed（`91400`）短程训练已完成，但 `stable` 配置的 adaptive 调度会把学习率抬高，训练中后期出现退化；该组只保留为“调度失败/非主结果”，不作为 DADE 归因。随后改用同 seed（`91430`）、同 BC、同 `128 × 256 × 32` 预算、固定 `lr=1e-5` 的严格配对。64-seed realistic_calibrated 结果：

| checkpoint | no-DADE success | no-DADE 3D RMSE | no-DADE saturation | DADE success | DADE 3D RMSE | DADE saturation |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | `0.718750` | `5.209724 m` | `0.126467` | `0.718750` | `5.217443 m` | `0.126778` |
| 10 | `0.734375` | `5.028345 m` | `0.127018` | `0.734375` | `5.060674 m` | `0.127264` |
| 20 | `0.734375` | `5.037607 m` | `0.126972` | `0.718750` | `5.242828 m` | `0.126093` |
| 30 | `0.703125` | `5.337974 m` | `0.124956` | `0.734375` | `5.053959 m` | `0.126793` |
| 31 | `0.703125` | `5.340176 m` | `0.124988` | `0.734375` | `5.046439 m` | `0.126808` |

固定 lr 配对没有证明 DADE 控制收益：DADE 末段优于同 seed no-DADE 末段，但 no-DADE `ck10/20` 仍是全组最低三维 RMSE（`5.028/5.038 m`），且 DADE 没有提高成功率或降低饱和。训练轨迹差异只能说明两条参数轨迹落在不同局部解，不能据此声称历史编码有效。
- 新增 `scripts/diagnose_pc_dade_history.py` 和 `src/auv_lowlevel/batch/pc_history_ablation.py`，对同一个 checkpoint 做 `current-only/current-zero/history-zero/history-reverse` 干预，并分别测量 `z_t`、`c_hat_t` 和归一化动作变化。测试 `tests/test_lowlevel_pc_history_ablation.py` 覆盖输入不被原地修改、当前帧保留/删除、历史清零和时间反序。
- 在随机路径 DADE `model_30` 上以 8 seeds、512 steps（`4096` 个样本）诊断：`current-only` 的动作 L2 变化均值仅 `6.68e-5`，`history-zero` 为 `5.95e-4`，`history-reverse` 为 `6.69e-7`；基线动作 L2 均值为 `0.928`。虽然 `current-only` 会让 DADE 特征改变 `0.0197`（基线特征范数 `0.0959`），但它几乎没有传到 actor 输出。`current-zero` 的 `0.835` 动作变化主要来自 `z_t` 被清零（`z` 变化均值 `2.95`），不是历史通路。无 DADE 对照在所有历史干预下变化严格为 0。
- 同一 checkpoint 的 actor 第一层权重也支持上述判断：`z_t` 输入列范数均值约 `1.489`，DADE 输入列约 `0.0103`；DADE 特征虽被训练，但在控制输出中仍近似旁路。结论是 **当前 DADE 收益/波动不能归因于历史依赖，继续延长 PPO 训练没有优先级**。
- 为此新增训练期 `aux_current` 辅助目标：在 actor DADE 输入中遮蔽最新历史行的 3 维海流估计列，仅用更早历史预测当前归一化海流侧信道；目标只训练 actor DADE 和辅助头，不改变 actor 推理输入、`z_t`、`c_t` 或 critic 契约。配置入口为 `configs/lowlevel/odarrl_control_ppo_pc_c3_random_realistic_dade_aux1e-1_s4_fixed1e5.json`（`coef=0.1`、每 iter 4 步）。单测和 1-iteration smoke 已通过；与 fixed DADE/no-DADE 同 seed 的短程 A/B 正在运行，完成前不写有效性结论。
- `aux_current` 在 `ck10` 上确实把 DADE 特征范数从约 `0.096` 提高到 `0.73`，`history-zero` 的特征变化也从 `0.116` 增大到 `0.354`；但动作变化仍只有 `4.3e-4`（基线动作 L2 `0.95`）。原因是 BC 热启动把 actor 首层 DADE 输入列缩放到 `0.01`，辅助损失只训练 DADE/辅助头，不能直接提高 actor 对 `c_hat` 的使用权重。因此必须把热启动尺度作为独立因素 screen，不能只继续加辅助损失。
- 新增 `pc.dade_warmstart_input_scale`（默认 `0.01`，保持旧结果兼容），并让 `load_pc_bc_actor_prior` 记录实际尺度。`scale=0.1` 组训练稳定（首轮 `approx_kl≈0.0011`）；`scale=1.0` 组首轮即出现 `approx_kl=12.04`、`ratio_max=1190.9`，属于信任域失稳，已停止并保留为负结果，不进入 5-seed。当前继续比较 `scale=0.01/0.1` 与 fixed DADE/no-DADE，先看历史敏感性和闭环指标是否同时改善。
- `aux_current` 的 `scale=0.01` 与 `scale=0.1` 两组 32-iteration 训练均已完成，`losses_finite=true`，但 64-seed realistic 闭环 A/B 均为负结果（同 BC、同 seed `91430`、同 `128×256×32` 预算、同 fixed `lr=1e-5`）：

| arm | ck0 success / 3D RMSE | ck10 | ck20 | ck30 | ck31 |
|---|---|---|---|---|---|
| no-DADE | `0.718750 / 5.209724 m` | `0.734375 / 5.028345 m` | `0.734375 / 5.037607 m` | `0.703125 / 5.337974 m` | `0.703125 / 5.340176 m` |
| fixed DADE | `0.718750 / 5.217443 m` | `0.734375 / 5.060674 m` | `0.718750 / 5.242828 m` | `0.734375 / 5.053959 m` | `0.734375 / 5.046439 m` |
| aux, actor DADE scale `0.01` | `0.718750 / 5.213396 m` | `0.687500 / 5.432455 m` | `0.671875 / 5.423358 m` | `0.671875 / 5.417443 m` | `0.671875 / 5.416688 m` |
| aux, actor DADE scale `0.10` | `0.703125 / 5.377046 m` | `0.687500 / 5.470641 m` | `0.687500 / 5.453181 m` | `0.671875 / 5.529478 m` | `0.687500 / 5.448545 m` |

  `scale=0.01` 后段相对自身起点下降 `4.69` 个百分点、RMSE 增大约 `0.21 m`；`scale=0.10` 的 `ck0` 已低于 BC 起点，后段也没有恢复。两组都未超过 fixed DADE 的 `ck30/31`，不能把辅助损失或放大输入尺度写成有效改进。`scale=1.0` 的信任域失稳继续保留为负结果，不重跑。
- 对应的 8-seed、512-step 历史消融诊断为：`scale=0.01` 的 `ck20` 在 `history-zero` 下动作 L2 变化均值 `5.12e-4`、`current-only` 为 `4.49e-5`、`history-reverse` 为 `1.17e-6`；`scale=0.10` 的 `ck20` 分别为 `2.40e-3`、`2.08e-4`、`2.91e-6`，而基线动作 L2 为 `0.94`。即使提高到 `0.10`，历史通路对动作的改变仍只有约 `0.3%`，说明辅助目标主要把 DADE 特征本身训练大，并没有让 actor 真正依赖历史。
- 根因修复实验：新增 `pc.batch.pc_bc` 的零尺度语义和测试，`dade_warmstart_input_scale=0.0` 时 BC 行为严格不变，actor 首层 DADE 输入列为零；必须同时启用 `aux_current`，由辅助目标先突破 DADE 输出零初始化，随后 PPO 才能通过正常的 actor 输入列梯度学习历史通路。新配置 `configs/lowlevel/odarrl_control_ppo_pc_c3_random_realistic_dade_aux1e-1_s4_scale0p0_fixed1e5.json` 已启动同 seed、同预算训练；在 64-seed 闭环结果出来前只登记为“symmetry-breaking screen”，不写成 DADE 有效。
- 零初始化 `scale=0.0 + aux=0.1` 已完成 32 iterations，训练全程有限、无信任域失稳，且 `ck0` 的 64-seed 结果为 `0.718750 / 5.213812 m`，与 BC 起点一致。完整 scheduled-checkpoint 结果全部维持成功率 `0.718750`，三维 RMSE 为 `5.213812 → 5.215398 → 5.257887 → 5.266725 → 5.265929 m`（ck0/10/20/30/31）；没有退化，但也没有超过 no-DADE 的 `ck10/20`（`0.734375 / 5.028345`、`0.734375 / 5.037607 m`）。历史消融仍只改变动作约 `5e-4–7e-4`，说明“保住 BC + 打开梯度”是必要条件，但 PPO 在 32 iter 内没有把这条弱路径学成有效控制器。
- 为区分“DADE 特征无用”和“actor 输入尺度过小”，对零初始化 `ck20/31` 做只放大 actor 首层 DADE 输入列的离线探针（不重训、不覆盖原 checkpoint，产物在 `artifacts/lowlevel/pc_ppo_c3_random_realistic_dade_scale0p0_amp_probe_20261103/`）。`ck31` 的 64-seed 结果为：

| amplification | DADE column norm | success | 3D RMSE | speed RMSE |
|---|---:|---:|---:|---:|
| `1×` | `0.00843` | `0.718750` | `5.265929 m` | `0.415904` |
| `2×` | `0.01685` | `0.718750` | `5.261522 m` | `0.4154` |
| `5×` | `0.04213` | `0.718750` | `5.248745 m` | `0.4138` |
| `10×` | `0.08426` | `0.718750` | `5.227180 m` | `0.4111` |
| `20×` | `0.16853` | `0.718750` | `5.166507 m` | `0.4068` |
| `30×` | `0.25279` | `0.718750` | `5.117797 m` | `0.4016` |
| `50×` | `0.42132` | `0.687500` | `5.230656 m` | `0.3949` |
| `100×` | `0.84264` | `0.640625` | `5.218739 m` | `0.3762` |

  这说明 DADE 特征确实含有一部分可用信息，且 `1×` 远低于有用区间；但放大到 `50×` 后成功率开始下降，不能用无限放大制造表面收益。`20×–30×` 是当前可控区间，对应 DADE 输入列范数约 `0.17–0.25`，约等于从随机初始化直接使用 `dade_warmstart_input_scale≈0.025–0.035`。该探针只用于确定下一组训练尺度，最高 `amp30` 仍为 `5.118 m`，没有超过 no-DADE `ck10` 的 `5.028 m`，不能作为最终策略或 DADE 正结果。
- 下一步采用有证据支撑的中间尺度 `scale=0.03 + aux_current` 进行同 seed、同预算正式训练；不再继续试 `0.1` 或无限放大。新增零尺度校验单测，当前 PC BC/policy/config 回归 `29 passed`，Ruff 通过。

**2026-11-03 C3 随机路径 DADE `scale=0.03` 正式短程对照**

- 训练配置：`configs/lowlevel/odarrl_control_ppo_pc_c3_random_realistic_dade_aux1e-1_s4_scale3e-2_fixed1e5.json`；输出：`artifacts/lowlevel/pc_ppo_c3_random_realistic_dade_aux_current_scale3e-2_fixed1e5_short32_seed91430_20261103/`。与 no-DADE、fixed DADE、零初始化组使用同一随机路径 realistic 环境、同一 BC checkpoint、同一训练 seed `91430`、同一 `128 × 256 × 32` 预算和固定 `lr=1e-5`。32 iterations 正常完成，无量测、非有限值或信任域爆炸。
- 64-seed realistic_calibrated scheduled-checkpoint 结果：

| checkpoint | success | 3D RMSE | material saturation | yaw RMSE | speed RMSE |
|---:|---:|---:|---:|---:|---:|
| 0 | `0.734375` | `5.053100 m` | `0.127360` | `0.829707 rad` | `0.414403 m/s` |
| 10 | `0.687500` | `5.430898 m` | `0.125636` | `0.889204 rad` | `0.418189 m/s` |
| 20 | `0.687500` | `5.457291 m` | `0.125523` | `0.890447 rad` | `0.421051 m/s` |
| 30 | `0.703125` | `5.307880 m` | `0.126805` | `0.869094 rad` | `0.418036 m/s` |
| 31 | `0.718750` | `5.235351 m` | `0.127824` | `0.854197 rad` | `0.415903 m/s` |

  `ck0` 相对随机路径 BC-only 的 `0.718750 / 5.216060 m` 有一次起点改善（`+1.56` 个百分点、`-0.163 m`），也略优于 no-DADE `ck0`；但 PPO 后续把成功率降到 `0.6875–0.71875`，没有超过 no-DADE 最优 `ck10/ck20` 的 `0.734375 / 5.028345 m`、`0.734375 / 5.037607 m`。因此只能登记为“DADE 热启动改善了 BC 起点”，不能写成“DADE 训练出持续控制收益”。
- `ck0` 的 8-seed、512-step 历史通路消融：baseline action L2 mean 为 `0.9552`；`current-only` 的 action delta mean/max 为 `2.81e-5 / 2.61e-4`，`history-zero` 为 `4.77e-4 / 2.24e-3`，`history-reverse` 为 `6.68e-7 / 4.10e-6`；对应 DADE feature (`chat`) delta mean/max 分别为 `8.22e-3 / 5.48e-2`、`1.20e-1 / 2.58e-1`、`7.34e-5 / 2.88e-4`。历史特征本身有响应，但对动作的最大影响仍不足基线动作的 `0.24%`，说明当前控制依赖几乎仍来自 `[z_t,c_t]`。
- `scale=0.03` 的训练 rollout 在第 5–7 iterations 成功率一度为 `1.0`，第 10 iteration 跌到 `0.18`；严格配对的 no-DADE 在同一位置同样从 `1.0` 跌到 `0.127659`，且前四轮 KL、clip fraction、value loss 轨迹同量级。因此这次中期成功率波动不是 DADE 特有，而是当前 PPO/rollout 评估窗口和探索噪声下的批次波动；不得把它归罪于 DADE，也不得把 rollout 的瞬时 `1.0` 当作闭环泛化。
- 强传感器压力集 `configs/lowlevel/odarrl_control_pc_c3_random_stress_v1.json`（32 条随机路径、`strong` 传感器、`0.35–0.6 m/s` 强剪切时变流）的 64-seed 结果为：no-DADE 与 DADE `ck10–31` 成功率全部约 `0.234375`，3D RMSE `9.82–9.89 m`，timeout rate 约 `0.7656`，current-estimate age 约 `2.573 s`、valid rate 约 `0.887`；`scale=0.03 amp30` 更差，为 `0.218750 / 9.876795 m`。当前没有证据显示 DADE 提升强扰动鲁棒性。
- 同一压力集的 teacher/BC 归因补齐：20° PD/PI teacher 为 `0.203125 / 9.502048 m`、material saturation `0.170864`，随机路径混合 BC 为 `0.234375 / 9.879410 m`、saturation `0.228205`；两者 timeout rate 分别为 `0.796875 / 0.765625`，均无安全终止。PPO 与其 BC 起点成功率和三维 RMSE 基本持平，说明强流下的主要失效发生在“无海流补偿的地速 ALOS + 执行器/传感器链”，不能把压力集失败归因于 DADE 或 PPO 学习退化。该集合仍作为固定压力测试保留，不用放宽阈值或移除强流来制造通过。

**2026-11-03 长视野 `gamma` screen：先暴露 value 欠拟合，再修正优化器**

- 新建严格配对配置：在 `gamma=0.995/0.998/0.999` 下分别保留 DADE `scale=0.03 + aux=0.1` 与 no-DADE，其余仍为同 BC、seed `91430`、`128 × 256 × 32`、fixed `lr=1e-5`。no-DADE 配置由 DADE 配置 clone 后只关闭 DADE/aux，避免旧 no-DADE 文件后来改 seed 导致的非配对问题。
- 结果不是“长视野略差”，而是第一步 PPO 更新就破坏 BC：no-DADE `gamma=0.995/0.998/0.999` 的首个 scheduled checkpoint 全部为成功率 `0.046875`、3D RMSE `16.1892/16.1974/16.1999 m`、material saturation `0`；`gamma=0.999 ck10` 仍只有 `0.0625 / 16.2989 m`。DADE `gamma=0.995` 在第 10 iteration 的 rollout 成功率也只有 `0.008696`。这些训练已主动停止，剩余 checkpoint 不评估，保留为“长 gamma 在当前 critic 配方下不稳定”的负结果。
- 失败与 value 尺度一致：`gamma=0.99` 的 value loss 约 `20–150`，而长 gamma 首轮约 `300–500`；critic 初始随机输出与近乎全 episode 的负回报目标严重不匹配，产生的 advantage 直接把 actor 推离 BC。当前不能用“长 gamma + 原单 lr、无 warmup”直接测试 DADE 的长时信用分配。
- 新增向后兼容的可选双学习率与 critic warmup：`algorithm.learning_rate_actor`、`algorithm.learning_rate_critic`、`algorithm.critic_warmup_iterations`；缺省分别回退到 `learning_rate` 和 `0`，旧配置行为不变。分离 lr 或 warmup 仅允许 `schedule="fixed"`。实现会把 PC policy 参数严格划分为 actor/encoder/DADE/aux/std 与 critic/encoder/DADE 两组；warmup 前 actor 组 lr=0，只更新 critic，之后恢复 actor 目标 lr。`rsl_train_cfg()` 会剥离这些扩展字段，避免传入上游 PPO。
- 验证：配置/policy/BC 回归 `31 passed`，Ruff 通过；3-iteration smoke 记录 actor lr 序列 `0, 0, 1e-5`、critic lr 恒为 `3e-4`，`losses_finite=true`。下一组实验固定 `gamma=0.995`、`actor_lr=1e-5`、`critic_lr=3e-4`、warmup `5`，分别跑 DADE/no-DADE；只有 warmup 后 BC 起点不被破坏且 critic value loss 明显下降，才继续评估长 gamma 的控制收益。
- C3 完整门禁仍未通过：有流成功率 `0.72–0.78`，realistic BC 饱和率偏高，随机路径 PPO 和 5-seed 正式评估尚未完成。下一步必须按固定 seed 集评估随机路径 DADE/no-DADE 的 scheduled checkpoints，并继续做传感器强随机化、`d_z`/学习率 screen 和高速课程；不能把当前短程收益写成最终结论。

**2026-11-03 奖励尺度审计与独立 reward-scale 实验臂**

- 审计发现旧 `BatchTorchVecEnv` 一直把 `reward.ppo_scaled = 0.01 × reward.total` 送入 PPO，而 PC 的 `BatchTorchPCVecEnv` 实际把未缩放的 `reward.total` 送入 PPO；PC 的 `ppo_scaled` 字段此前只用于诊断，没有参与训练。这是接口不一致，不是 DADE 或 actor 结构本身的结果。
- 该不一致与长 gamma 的 value 失稳方向一致：PC 每个 50 s episode 的原始回报约为 `-1,500–-2,600`，长 gamma 下随机初始 critic 的 value loss 可达数百；legacy 路径则先缩放 `0.01`。在没有直接 A/B 前，不能断言它是唯一根因，但必须把它从混杂因素中隔离。
- 已新增显式、opt-in 的 `reward.pc_ppo_scale`。未设置时 PC 保持历史未缩放语义（`resolved_pc_ppo_scale=1.0`），legacy 仍保持硬编码 `0.01`；只有显式设置的新配置使用缩放。缩放只改变 PPO 返回，日志继续记录原始 reward，旧配置、旧 checkpoint、旧结果不被改写。
- 代码验证：PC reward 缩放、PC vec-env 返回 scaled reward 但记录 raw return、旧默认行为、配置解析和 PC 契约均覆盖；`tests/test_lowlevel_pc_*.py` 为 `74 passed`，Ruff 通过。
- 新建严格配对的 reward-scale 实验臂，保持 seed `91430`、`128×256×32`、`gamma=0.995`、actor lr `1e-5`、critic lr `3e-4`、critic warmup `5` 和 BC 起点不变，只把 `reward.pc_ppo_scale` 设为 `0.01`：
  - `configs/lowlevel/odarrl_control_ppo_pc_c3_random_realistic_dade_aux_current_scale3e-2_gamma995_warmup5_critic3e-4_fixed_rewardscale001.json`
  - `configs/lowlevel/odarrl_control_ppo_pc_c3_random_realistic_nodade_gamma995_warmup5_critic3e-4_fixed_rewardscale001.json`
  - 对应环境配置：`configs/lowlevel/odarrl_control_pc_c3_random_realistic_v1_rewardscale001.json`
- 该臂的用途是判断“value 尺度/优势尺度”是否为主因：先比较 warmup 后 ck0 是否保持 BC、value loss 是否从数百降到可解释量级，再做 64-seed scheduled-checkpoint 评估。若缩放后仍无法保住 ck0，则停止延长 gamma，转向 critic 预训练、return normalization 或先把短 gamma/短 episode 学稳。

**2026-11-03 首次 warmup 启动无效：缺少 BC 热启动**

- 首次 `gamma=0.995 + critic warmup=5` 的 DADE/no-DADE 并行启动命令遗漏了 `--warmstart-actor`。两个输出目录的 manifest 都是 `warmstart_actor=null`，因此 warmup 期间保护的是随机初始化 actor，不是 BC 先验。
- no-DADE 的固定 64-seed `model_0.pt` 评估为 `success=0.046875`、3D RMSE `16.210883 m`，与随机策略一致；这直接推翻了“本次 warmup ck0 保住 BC”的假设。DADE 组在 23/32 iteration 主动停止，剩余 checkpoint 不评估。
- 两个无效目录已各自写入 `INVALID_NO_BC_WARMSTART.md`，原 manifest、records 和 checkpoint 均保留，不删除、不覆盖；它们只作为“启动命令缺少 BC”的审计记录，不作为长 gamma/warmup/DADE/reward-scale 的正负结果。
- 同时修复了 no-DADE 的 BC 热启动校验：DADE 关闭时允许 `dade_warmstart_input_scale=0`，不再错误要求辅助海流头；DADE 开启时零尺度仍必须配合辅助目标。新增回归测试后，PC 测试为 `75 passed`。
- 第二次启动已改为显式 `--warmstart-actor artifacts/lowlevel/pc_bc_c0_c1_c2_c3random_v1_20261103/pc_bc_actor.pt`，并在 manifest/metadata 中记录 BC SHA-256 与 `pc_ppo_scale`。四个严格配对组为 `{DADE,no-DADE} × {pc_ppo_scale=1.0,0.01}`，输出目录均带 `_bcws_` 后缀；在 ck0 评估完成前不写性能结论。

**2026-11-03 正确 BC warmup 的 no-DADE 结果**

`ck0` 在原始 reward 与 `pc_ppo_scale=0.01` 两组均为 `success=0.718750`、3D RMSE `5.216059 m`、material saturation `0.126575`，证明 warmup 的 actor lr=0 确实保住了同一个 BC 先验。两个 arm 的 64-seed 固定 seed `0–63` 结果为：

| no-DADE arm | ck0 | ck10 | ck20 | ck30 | ck31 |
|---|---|---|---|---|---|
| raw reward（历史语义） | `0.718750 / 5.216059 m` | `0.703125 / 5.276708 m` | `0.703125 / 5.277459 m` | `0.703125 / 5.282235 m` | `0.703125 / 5.283021 m` |
| `pc_ppo_scale=0.01` | `0.718750 / 5.216059 m` | `0.718750 / 5.198587 m` | `0.718750 / 5.209225 m` | `0.734375 / 5.058933 m` | `0.734375 / 5.065430 m` |

- 缩放组的 value loss 约 `0.003–0.01`，raw 组约 `79–185`；缩放把 critic 目标压到可训练量级，但优势在 rollout 内还会整体归一化，因此两组的 actor 轨迹和 KL 量级相近。当前不能把 reward scale 说成“改变了策略探索”，它主要改善 value 拟合和训练数值条件。
- 缩放组后段达到成功率 `0.734375`、3D RMSE `5.0589–5.0654 m`，优于同 warmup 的 raw 组 `0.703125 / 5.28 m`，但仍未超过历史 no-DADE 最佳的 `0.734375 / 5.028345 m`。这是“稳定性改善”，不是新的性能上限。
- material saturation 在两组都约 `0.125–0.127`，与 BC 起点持平；没有用降低机动或冻结策略换取稳定。DADE 两臂仍在训练，不能提前归因。

**2026-11-03 正确 BC warmup 的 DADE reward-scale 结果与历史通路诊断**

DADE 两臂使用与 no-DADE 完全相同的 BC checkpoint、seed、预算和评估集。`ck0` 的四臂均为 `success=0.718750`、3D RMSE `5.216059 m`、material saturation `0.126575`，确认 warmup 保护的是同一个 BC 先验。DADE 两臂的 64-seed 固定 seed `0–63` 结果为：

| DADE arm | ck0 | ck10 | ck20 | ck30 | ck31 |
|---|---|---|---|---|---|
| raw reward（历史语义） | `0.718750 / 5.216059 m` | `0.703125 / 5.278237 m` | `0.671875 / 5.569150 m` | `0.687500 / 5.490174 m` | `0.687500 / 5.479949 m` |
| `pc_ppo_scale=0.01` | `0.718750 / 5.216059 m` | `0.703125 / 5.278779 m` | `0.703125 / 5.283104 m` | `0.734375 / 5.061310 m` | `0.734375 / 5.058462 m` |

- reward scale 同样消除了 DADE 后段的明显退化：raw 组在 `ck20` 降到 `0.671875 / 5.569150 m`，缩放组稳定在约 `0.734375 / 5.06 m`。但该最佳值与同配方的 no-DADE 几乎重合，且仍未超过历史 no-DADE 最佳 `0.734375 / 5.028345 m`；因此这里只能记录为“训练稳定性改善”，不能记录为“DADE 带来任务性能收益”。
- DADE 缩放组的 material saturation 仍约 `0.125–0.128`，没有通过冻结动作或显著保守化换取稳定。四臂训练目录分别为 `artifacts/lowlevel/pc_ppo_c3_random_realistic_nodade_gamma995_warmup5_critic3e-4_fixed{,_rewardscale001}_bcws_short32_seed91430_20261103` 与 `artifacts/lowlevel/pc_ppo_c3_random_realistic_dade_aux_current_scale3e-2_gamma995_warmup5_critic3e-4_fixed{,_rewardscale001}_bcws_short32_seed91430_20261103`。
- 对 DADE 缩放组 `ck30/ck31` 做了 8-seed、512-step 推理期历史消融，产物为 `artifacts/lowlevel/diag_pc_dade_history_bcws_rewardscale001_seed91430_ck30_8seed_512step_20261103/` 和 `..._ck31_8seed_512step_20261103/`。`ck31` 的 base action L2 mean 为 `0.955697`，base DADE feature L2 mean 为 `0.573002`；把两段历史全部置零后，`ĉ_t` 的 L2 变化均值达到 `0.319531`，但动作 L2 变化均值只有 `0.000734`，约为基线动作的 `0.077%`。`current-only` 的动作变化为 `7.38e-5`，`history-reverse` 为 `1.31e-6`。
- 这组诊断说明：DADE 特征确实对历史输入有响应，但 actor 几乎没有把这条历史通路用于动作决策。继续延长训练或只增大 DADE 权重不能证明 DADE 有效；后续应把“显式历史使用正则/特征缩放/门控”做成独立实验臂，并与当前主 DADE 结论分开归因。
- 因此 `rewardscale001 + gamma995 + warmup5` 暂时作为后续短预算 screen 的稳定配方：它改善 value 数值条件和后段稳定性，但不改变 DADE 无显著收益这一结论。PC-G3 的 `d_z`/学习率筛选应在这个配方上做，不能把 raw reward 或未 warmup 的旧失败组混入比较。

**2026-11-03 PC-G3：`d_z`、actor/critic lr 与探索标准差 screen**

BC 先验统一使用 `artifacts/lowlevel/pc_bc_demos_c0_c1_c2_c3random_v1_20261103/demos.npz`（SHA-256 `eeac3e63266a1a96451d913f23c5b8df5ee53f5db68a6efbf94367ddedda3ded`）、同一归一化和 seed `0`。`d_z=32` 沿用已有 BC checkpoint；`d_z=16/64/128` 重新训练 BC，checkpoint 分别为 `artifacts/lowlevel/pc_bc_c0_c1_c2_c3random_v1_dz{16,64,128}_20261103/pc_bc_actor.pt`。四者的 validation normalized RMSE 接近，说明后续差异不是 BC 拟合失败造成的：16 `[0.095187, 0.072470, 0.025181]`、32 `[0.097749, 0.070145, 0.025504]`、64 `[0.101999, 0.071453, 0.024508]`、128 `[0.099444, 0.074733, 0.024712]`。

screen 使用 successive halving：先让每个候选跑 1 个训练 seed，再只把前两名提升到 3 个训练 seed（`91430/91431/91432`）；所有评估均为固定 seed `0–63`，比较 `ck0/10/20/30/31`，不使用训练 rollout 成功率代替闭环评估。所有候选均为 no-DADE、`pc_ppo_scale=0.01`、`gamma=0.995`、critic warmup `5`、`128×256×32`，除被 screen 的变量外完全相同。

`d_z` 在 `actor_lr=1e-5`、`critic_lr=3e-4`、`init_noise_std=0.1` 下：

| `d_z` | 训练 seed 数 | `ck30` success | `ck30` 3D path RMSE | `ck31` success / RMSE |
|---|---:|---:|---:|---:|
| 16 | 1 | `0.671875` | `5.423470 m` | `0.671875 / 5.423678 m` |
| 32 | 3 | `0.697917` | `5.364172 m` | `0.692708 / 5.390192 m` |
| 64 | 3 | `0.718750` | `5.158613 m` | `0.718750 / 5.159405 m` |
| 128 | 1 | `0.640625` | `5.580135 m` | `0.640625 / 5.570981 m` |

- `d_z=16/128` 在首 seed 的成功率和 RMSE 都明显落后，按 successive halving 淘汰；`d_z=32/64` 进入三 seed 复核。
- `d_z=64` 的三个 seed 在 `ck30` 上分别为 `0.718750 / 5.191765 m`、`0.718750 / 5.139649 m`、`0.718750 / 5.144426 m`，没有 seed 掉队；`d_z=32` 从 `ck0` 的 `0.718750 / 5.216059 m` 退化到 `ck30` 的均值 `0.697917 / 5.364172 m`。因此选 `d_z=64`，不是只看单 seed 最优。
- BC 热启动没有被锁死：`d_z=64` 的 `ck0` 为 `0.687500 / 5.391934 m`，训练到 `ck30` 后变为三 seed 均值 `0.718750 / 5.158613 m`；策略行为、成功率和路径误差都发生变化，不是只做 teacher 的静态复制。

固定 `d_z=64` 后，learning-rate screen：

| 变量 | 候选 | seed 数 | `ck30` 结果 | 结论 |
|---|---|---:|---|---|
| actor lr | `1e-5` | 3 | `0.718750 / 5.158613 m` | 保留，三 seed 全部 `0.718750` |
| actor lr | `3e-5` | 1 | `0.703125 / 5.362867 m` | 首 seed 即落后，淘汰 |
| actor lr | `1e-4` | 3 | `0.687500 / 6.056662 m` | `ck30/31` 出现 `0.5625–0.7656` 大幅不稳定，淘汰 |
| critic lr | `3e-4` | 3 | `0.718750 / 5.158613 m` | 保留 |
| critic lr | `1e-4` | 1 | `0.703125 / 5.335311 m` | 首 seed 落后，淘汰 |
| critic lr | `1e-3` | 1 | `0.703125 / 5.318252 m` | 首 seed 落后，淘汰 |

actor lr `1e-4` 的 aggregate 会被单 seed 最优点误导：其 `ck20` 三 seed 均值为 `0.718750 / 5.245661 m`，但 `ck30/31` 最差 seed 降到 `0.578125 / 8.2450 m` 和 `0.562500 / 8.8150 m`；因此最终仍选 `1e-5`。critic lr 的两个方向都未超过 `3e-4`，不把 critic 学习率继续加大。

探索标准差在相同 `d_z=64`、lr 和 BC 下 screen：

| `init_noise_std` | seed 数 | 最佳 scheduled checkpoint | 结论 |
|---|---:|---|---|
| `0.05` | 1 | `ck31=0.718750 / 5.179569 m` | 与基线持平但 `ck30` 更差，不证明优势 |
| `0.10` | 3 | `ck30=0.718750 / 5.158613 m` | 保留，seed 间最稳定 |
| `0.20` | 1 | `ck30=0.718750 / 5.199010 m` | RMSE 和速度误差均未改善，淘汰 |

最终 PC-G3 选择：`d_z=64`、actor lr `1e-5`、critic lr `3e-4`、critic warmup `5`、`init_noise_std=0.1`、`gamma=0.995`、`pc_ppo_scale=0.01`、no-DADE 配置 `configs/lowlevel/odarrl_control_ppo_pc_c3_random_realistic_nodade_dz64_gamma995_warmup5_critic3e-4_fixed_rewardscale001.json`。该配置的三个训练 seed 的 scheduled checkpoint 评估目录均以 `eval_pc_ppo_c3_random_realistic_nodade_dz64_gamma995_warmup5_rewardscale001_bcws_seed9143{0,1,2}_ck*_64seed_20261103` 保存。下一步是在同一 `d_z=64`、同一 BC/seed/预算下补 DADE 臂，然后进入高速课程和正式五组对照；不能把本节的 no-DADE 结果写成 DADE 结论。

**2026-11-03 `d_z=64` DADE 配对与历史通路诊断（负结果）**

- 新建同 `d_z=64`、同 BC（`pc_bc_c0_c1_c2_c3random_v1_dz64_20261103`）、同 seed `91430`、同 `128×256×32`、同 reward scale/warmup/lr 的 DADE 配置 `configs/lowlevel/odarrl_control_ppo_pc_c3_random_realistic_dade_aux_current_scale3e-2_dz64_gamma995_warmup5_critic3e-4_fixed_rewardscale001.json`。DADE 使用 `aux_current=0.1/4 steps`、`dade_warmstart_input_scale=0.03`，训练 32 iterations 正常完成且数值有限。
- 64-seed realistic_calibrated 配对：

| checkpoint | no-DADE success / RMSE | DADE success / RMSE |
|---:|---:|---:|
| 0 | `0.687500 / 5.391934 m` | `0.687500 / 5.391934 m` |
| 10 | `0.687500 / 5.395578 m` | `0.703125 / 5.302920 m` |
| 20 | `0.687500 / 5.432476 m` | `0.703125 / 5.290152 m` |
| 30 | `0.718750 / 5.191765 m` | `0.703125 / 5.276808 m` |
| 31 | `0.718750 / 5.193245 m` | `0.703125 / 5.283282 m` |

- DADE 自身从 `ck0` 的 `0.687500 / 5.391934 m` 改善到 `ck30/31` 的 `0.703125 / 5.2768–5.2833 m`，但没有超过同 seed no-DADE 的 `0.718750 / 5.1918–5.1932 m`；成功率低 `1.5625` 个百分点、RMSE 高约 `0.085 m`。因此不能把 `d_z=64` 写成 DADE 的修复。
- 对 DADE `ck31` 做 8-seed、512-step 历史消融，产物为 `artifacts/lowlevel/diag_pc_dade_history_dz64_scale3e-2_rewardscale001_seed91430_ck31_8seed_512step_20261103/`。baseline action L2 mean `0.962408`；`history-zero` 的 `ĉ_t` 变化均值 `0.257193`，但动作变化仅 `0.000316`，约为基线动作的 `0.033%`；`current-only=4.74e-5`、`history-reverse=8.51e-7`。与 `d_z=32` 一样，DADE 特征对历史有响应，但 actor 几乎不使用该通路。
- 该负结果说明：仅把 `d_z` 从 32 提到 64 不能解决 DADE 的历史利用问题。下一步不应继续盲目增加维度、训练时长或 DADE 权重；若要评估“让 DADE 效果更明显”，必须做成独立实验臂，优先方向是历史关键场景（更长的海流估计延迟/失锁、快速时变流）和显式历史使用正则/门控，并与 no-DADE、普通 DADE 分开归因。

**后置可选：NP3O 安全 RL（只有在 0.1.12.1 的前置条件满足后才执行）**

- [ ] 新增独立 `pc_np3o` 配置分支：`algorithm="np3o"`、`num_costs=6`、固定 cost 尺度和 `cost_d_values`，不修改主 PC 配置、DADE checkpoint 和默认训练入口；参考实现只读 `/home/bubble/桌面/tita/tita_rl/algorithm/np3o.py`，不直接复制其足式机器人依赖；
- [ ] 新增 PC cost wrapper：动作幅值、动作速率、动作变化、滚转、滚转率和 terminal safety 六通道；使用真值 `φ,p` 和请求动作，固定归一化、有限性检查与 per-channel clip；DVL 失锁、USBL 野值和海流估计误差不计作策略 cost；
- [ ] 扩展 rollout 与 GAE：记录 reward/cost 两套 value、return、advantage、violation；`truncated` 用 `V_c(s_T)` bootstrap，`terminated` 不 bootstrap，安全终止追加 terminal cost；修正参考实现 time-out 分支使用当前 cost 的问题，并加单测；
- [ ] 新增 `K=6` cost head：默认 `CostHead(stopgrad([z_t^c,c_t,ĉ_t^c]))`，softplus 输出；actor、actor encoder、actor DADE 和顶层输入宽度不变；cost head 欠拟合时才另做独立 cost encoder，不能改写 actor；
- [ ] 实现 NP3O loss 与日志：clipped cost surrogate、归一化 violation、`k_value` ramp、`cost_value_loss_coef`、`cost_viol_loss_coef`，记录每个通道的 cost、value、advantage、violation 和权重；
- [ ] 新增 `PC PPO + scalarized costs without DADE` 控制组，与 `PC NP3O without DADE` 使用完全相同的 6 个 cost 和固定权重；NP3O arm 的 reward 移除重复的滚转、动作平滑、饱和和稀疏 `safety_penalty` 项；
- [ ] 完成 3-seed 短预算 screen：先比较 PPO、PPO+scalarized costs、NP3O；若 NP3O 未优于 scalarized control 或任务性能退化超过 PC-G6，停止，不进入 5-seed；
- [ ] 仅在无 DADE 的 NP3O screen 通过后，增加 `PC NP3O with DADE`，并保持与主实验同 seed、profile、预算和评估集；不得把 NP3O 和 DADE 两个因素混成一个结论。

每一步单独提交，禁止 `git add -A` 把当前仓库已有的其他修改一并带入。文档阶段提交命令：

```bash
git add doc/goals/goal9_odarrl_control_reproduction.md
git commit -m "docs(goal9): add ALOS-external pure-controller PC arm"
git push origin "$(git branch --show-current)"
```

代码阶段按实际新增文件提交，例如：

```bash
git add configs/lowlevel/odarrl_control_pc_*.json \
        src/auv_lowlevel/batch/ \
        tests/test_lowlevel_pc_*.py
git commit -m "feat(lowlevel): add PC pure-controller arm"
git push origin "$(git branch --show-current)"
```

提交前检查：

```bash
git status --short
git diff --check
pytest -q tests/test_lowlevel_pc_*.py
```

当前代码位于 `settled_局部规划器_反应式/src/auv_lowlevel/`，配置位于 `configs/lowlevel/`。没有修改 PVS、IsaacLab 或 rsl_rl 源码，没有向 `oceansim` 或 IsaacLab venv 安装依赖。

服务器正式证据：

- **B0 最终 50-seed**：成功率 1.00，三维 RMSE `1.515 m`，水平 RMSE `1.469 m`，垂向 RMSE `0.324 m`，平均到达 `27.94 s`，饱和率 0，约束违反 0；控制环 p95 的 p95 为 `19.71 ms`，PVS 单步 p95 的 p95 为 `0.44 ms`；
- **I 臂教师采集**：180 episodes、255,467 样本、成功率 1.00、约束违反 0、shadow-teacher 积分差 0；
- **单次 BC 负结果**：50-seed 三维 RMSE 差 `26.08 m`、成功率下降 `0.96`、student 大幅发散；根因是闭环分布漂移，不是验证 MSE 或标签错误；
- **DAgger 修复**：第 1 轮 40 episodes / 98,938 样本把三维 RMSE 差降到 `0.119 m`；第 2 轮 60 episodes / 86,109 样本进一步降到 `0.0203 m`；
- **I0 最终门禁**：teacher 三维 RMSE `1.355 m`、student `1.375 m`、成功率均 1.00；旧的 any-clip rate 为 `0.0073`，按“超过 1% 满量程”重算的 material saturation 为 `0.00665`，门禁通过；
- **终止条件修复**：初版 `2.5 m`、无终端捕获时 50-seed 成功率为 0.70，15 个 episode 在 `progress=35 m` 后继续沿末端切线飞出 20 m 以上；加入 16 m 终端捕获后，`1.0 m` 半径成功率 0.96，`1.25 m` 半径 1.00，最终统一用 `2.0 m` 作为到达半径并另行报告严格捕获率；
- **BC smoke**：4 episodes × 40 步 = 160 样本、`32,16` 网络仅用于接口验证，不作为正式证据。
- **26D I0 正式门禁**：180 episodes / 255,467 样本；单次 BC 失败（RMSE 差 `22.28 m`、成功率 `0.38`）；一轮 40 episodes / 80,239 样本 DAgger 后，teacher/student 三维 RMSE `1.35505/1.39257 m`，成功率均 1.00，B0 对照改善 `0.12218 m`；1.25 m 严格捕获率 teacher/student `0.92/0.94`。

## 1. 目标与边界

### 1.1 本轮目标

回答四个问题：

1. 在 REMUS100 + PVS 六自由度执行器级仿真中，论文的“MPC 专家 → 行为克隆 → 冻结基础策略 → 残差 PPO → 双时间尺度扰动编码器”是否可复现？
2. 在我们现有的 `ALOS + PD/PI + PVS` 链路上，残差 PPO 是否真的能改善海流下的三维轨迹跟踪，而不是只改善二维水平误差？
3. 与传统在线 MPC 相比，残差 PPO 的优势是否仍然存在；如果不存在，是否应该保留传统控制器作为主方案？
4. 先把现有 `ALOS + PD/PI` 控制器模仿成 `pi0`，再冻结 `pi0` 学习残差 PPO + DADE，是否优于直接冻结在线 PD/PI？

最终要交付：

- 一套可重复的三维轨迹跟踪控制环境；
- 一个在线 PD/PI 基线；
- 一个在线 MPC 基线；
- 一个 MPC 专家数据采集与行为克隆链；
- 一个现有 PD/PI 专家数据采集与行为克隆链；
- 一个冻结基础策略的残差 PPO 链；
- 一个执行器感知的双时间尺度编码器；
- 与论文可比的误差、饱和、平滑度和推理延迟报告；
- 失败时保留负结果，不把“用了 RL”写成“优于传统控制”。

### 1.2 明确不做

- 不做论文第三阶段的障碍物、避障模式机和避障奖励。
- 不使用声呐、点云、OceanSim 渲染或视觉输入训练控制本体。
- 不把 BlueROV2 Heavy 的八推进器模型直接搬到 REMUS100；REMUS100 只使用 `[rudder, stern_plane, rpm]` 三个执行器。
- 不修改 PVS、IsaacLab 或 rsl_rl 的源码来“跑通”；需要适配时只增加 wrapper、配置和测试。
- 不使用 4096 个 PVS worker 训练。PVS 是 CPU 串行模型，会在服务器上严重超卖；新训练使用已通过 Goal10 门禁的 `BatchTorchVecEnv` 4096 环境，PVS 只用有限 worker 做正式评估和抽检。
- 不把 `MPC-imitation policy` 当成“在线 MPC”；论文比较的是模仿网络，不是在线优化控制器。
- 不把 `controller-imitation policy` 当成“在线 PD/PI”；它是固定网络，必须与在线控制器分开报告。
- 不把当前的 P1 二维 PPO checkpoint 直接接到 PVS 上。它的动作是 `[u_dot, r]`，不是执行器指令。
- 不因为一次评估成功就宣称 50Hz 稳定；必须记录 p50/p95 推理延迟和固定种子结果。

### 1.3 冻结假设

| 项目 | 第一版冻结值 | 说明 |
|---|---|---|
| 被控对象 | REMUS100 + PVS | 复用 `PvsRemus100Backend` |
| 控制频率 | 50 Hz，`dt=0.02 s` | 与现有控制链一致 |
| 参考维度 | `x, y, z, psi` | 论文的控制变量；roll/pitch 由恢复力和控制器稳定 |
| 动作 | `[rudder, stern_plane, rpm]` | 三个真实执行器 |
| 舵角上限 | `±15 deg` | PVS 和 `VehicleSpec` 一致 |
| 舵速上限 | `15 deg / 0.1 s` | 复用现有 `Remus100TrackingController` 限制 |
| RPM 上限 | `1525 rpm` | REMUS100 模型上限 |
| 参考速度 | `1.2 m/s` | REMUS 最小速度 `1.0 m/s`，不能照搬论文的 `0.85 m/s` |
| 参考路径长度 | `35 m` | 与论文对齐 |
| 参考路径生成 | 三维正弦 + 弧长重采样 | 第一版实现假设，替代论文未给出的 QPMI 细节 |
| 终端制导 | 距终点 16 m 切入直接指向终点 | 修复 ALOS 在路径末端只跟踪切向、越过后不回头的问题，见 1.6 |
| 到达半径 | `2.0 m` | 第一版实现假设；PD 航向环存在约 1.3–1.9 m 终端稳态偏差，严格捕获率另行报告，见 1.6 |
| episode 上限 | `2500` 控制步 = `50 s` | 35 m / 1.2 m/s 约 29 s，留出扰动和跟踪余量 |
| 海流 | 水平流，`0.2–0.4 m/s`，方向随机；可叠加时间调制和线性空间剪切 | 批量后端与 PVS 标量路径都支持；垂向流仍不支持 |
| 并行环境 | 训练 `4096×256`；PVS 评估 64–128 workers | 训练用 `BatchTorchVecEnv`，评估用 CPU PVS；4096 个 PVS 进程会超卖 CPU |
| 训练位置 | 服务器 | 本地只做编辑、单元测试和小规模 smoke |
| 评价种子 | 固定 50 组 | 与论文的 50 次独立评估对齐 |

### 1.4 论文中不明确、需要在 G1 前确认的内容

论文没有公开代码，以下细节没有完整给出：

- MPC 的具体预测模型、时域、权重、约束和求解器；
- 策略观测向量的精确维数和归一化；
- 八推进器的具体分配矩阵；
- 残差系数 `alpha_p=0.1`、`alpha_v=0.05` 的精确作用位置；
- 控制频率和 episode 的物理时长；
- 奖励中动作平滑项的具体形式；
- 第三阶段事件奖励的精确数值。

本文件对这些缺口给出默认实现，但必须在配置和最终报告中标注为“我们的实现假设”，不能写成论文原文。

### 1.5 非破坏式实现与环境隔离

本路线沿用 goal8 的“零侵入优先”原则，但保护对象换成 PVS、rsl_rl 和现有 Python 环境。

受保护路径：

```text
python/PythonVehicleSimulator-master/**
simulator/IsaacLab/**
任何 venv 或 conda 环境的 site-packages/**
现有 rsl_rl 安装目录
```

实现顺序必须是：

1. 优先通过 JSON 配置、wrapper、adapter、subclass 或独立新包实现；
2. 其次通过向量环境接口、runner 注入或配置扩展实现；
3. 只有在确认是上游 bug、且前两步都无法完成时，才允许最小补丁；
4. 不允许直接把实验逻辑写进 PVS、IsaacLab 或 rsl_rl 源码。

允许的改动：

- 新增 `src/auv_lowlevel/`、`scripts/`、`configs/lowlevel/` 和测试；
- 在 `auv_planner` 中新增独立的 MPC 文件，不改变现有 PD/PI 行为；
- 通过 wrapper 调用 `PvsRemus100Backend`、`ThreeDAlosGuidance` 和 `Remus100TrackingController`；
- 通过 rsl_rl 的公开接口接入自定义向量环境；
- 新增 JSON schema、manifest 和小型证据文件。

禁止的改动：

- 直接修改 `python/PythonVehicleSimulator-master` 中的 REMUS100 模型；
- 直接修改 IsaacLab 或 rsl_rl 的 `site-packages`；
- 在 `oceansim` conda 环境或 IsaacLab venv 中执行 `pip install`、`pip uninstall`、`pip install -U`；
- 使用 `--force-reinstall`、`--upgrade` 或升级 PyTorch、NumPy、Isaac Sim；
- 把 PVS 模型复制一份后静默改参数，导致论文结果和实际 plant 不一致；
- 用 `git reset --hard`、`git checkout --`、`git clean -fd` 清理当前工作树。

如果确实需要上游补丁：

- 补丁必须单独放在 `patches/`，不能在 site-packages 中手改；
- 必须写明原因、影响文件、上游版本和替代方案为何不可行；
- 补丁前后各跑一次 `N=1` 对齐测试和相关 smoke；
- 在 manifest 中标记 `upstream_patch=true`；
- 提供回滚命令和未打补丁时的失败证据；
- 没有完成以上步骤，不得进入训练阶段。

环境与源码指纹：

- 安装任何新依赖前，记录目标 Python 环境的 `pip freeze` SHA-256；
- 记录 PVS 源文件 SHA-256、PVS 目录的 Git commit 或归档哈希；
- 记录 IsaacLab 和 rsl_rl 的版本、源码路径和 `git status --short`；
- 新依赖只安装到隔离 venv，不安装到 `oceansim` 或 IsaacLab venv；
- 安装后再次记录 `pip freeze`，任何已有包版本变化都视为失败并停止；
- 所有实验 manifest 同时记录 `config_sha256` 和实现指纹。

最低测试要求：

- `N=1` 单环境与批量环境的观测、奖励和终止条件逐项对齐；
- PVS 相同输入下可重复；
- 动作幅值、舵速、RPM 和时间戳检查不被绕过；
- 多 worker 启动、异常退出和回收无泄漏；
- 训练前后基础控制器行为不变；
- 现有测试和导入边界不被破坏。

Git 约束：

- 不执行 `git add -A`、`git add .` 或 `git commit -a`；
- 只添加本阶段审阅过的文件；
- 不删除或覆盖 goal6/goal7/goal8 的配置、门禁和历史结果；
- 每阶段结束先执行 `git diff --check` 和 `git status --short`。

### 1.6 第一版实现假设与服务器证据

以下内容不是论文原文，而是当前代码已经采用的实现选择。正式报告必须原样列出，不能写成论文参数。

| 项目 | 当前实现 | 原因与后续 |
|---|---|---|
| 路径生成 | 三维正弦曲线 + 等弧长重采样 | 论文未给出 QPMI 细节；先用可复现的解析路径，后续可替换生成器并保持同一路径长度 |
| 终端捕获 | 距终点 16 m 内直接朝终点制导 | 初版无终端捕获时，ALOS 只测横向/垂向误差；越过终点后沿末端切线飞出 20 m 以上，50-seed 成功率仅 0.70。捕获距离 8 m 仍太晚，16 m 可消除飞越 |
| 到达半径 | `2.0 m` | 即使无海流，PD 航向环仍有约 1.3–1.9 m 的终端稳态偏差；`1.0 m` 半径成功率 0.96，`1.25 m` 为 1.00，最终取 `2.0 m` 留出执行器/海流余量。严格 `≤1.25 m` 捕获率必须单独报告 |
| 参考速度可行域 | 裁剪到 REMUS 的 `[min_speed, max_speed]` | 海流补偿可能给出不可行水相对速度；裁剪事件记录在 `reference_speed_clipped` |
| 海流观测 | `observation.current_contract`: `legacy` / `hidden` / `oracle` | `hidden` 要求索引 17–19 置零且速度反馈为地速；`oracle` 显式保留真实海流，只能作为上界对照；旧配置缺字段时按 `legacy` 加载 |
| 速度反馈 | `observation.speed_feedback`: `water_relative` / `ground` | `hidden` 必须用 `ground`；索引 11 用地速、索引 12 用地速跟踪误差，禁止 `linear_x - water_surge` 反推真实海流 |
| 制导与教师海流 | hidden 臂 `guidance.compensate_current=false`、`base_controller.use_current_field=false` | 否则参考航向/速度和教师输出会间接编码真实海流，索引 17–19 置零只是形式上的隐藏 |
| B0 基线 | current-aware 教师 | B0 同时使用真实海流做 ALOS 补偿和速度误差计算，是 privileged 参考基线，不是可部署的无海流先验 |

配置加载器现在执行两条硬校验。第一，当 `current.enabled=true` 且 `observation.expose_current=false` 时，`guidance.compensate_current` 和 `base_controller.use_current_field` 必须都为 `false`，否则直接报 `hidden current leak`。第二，当 `observation.current_contract="hidden"` 时，`expose_current` 必须为 `false` 且 `speed_feedback="ground"`；`oracle` 必须 `expose_current=true`。第一条防止调参时把 I 臂变成 privileged 策略，第二条防止 `water_surge`/`speed_error` 把真实海流重新泄漏给最终 actor。

`goal_radius=2.0 m` 是当前最影响结果解释的实现假设。正式报告必须同时给出最近终点距离分布和严格阈值（例如 `1.0/1.25 m`）下的捕获率；不能只报 2.0 m 成功率，也不能把它与论文的成功率直接横向比较。注意：2.0 m 终止会产生删失，轨迹在约 `1.99 m` 就结束，不能从这类 NPZ 的 minimum distance 推出严格捕获率；严格捕获率必须用 `--goal-radius 1.0` 或 `1.25` 重新评估。

`artifacts/lowlevel/i0_gate_dagger2_50seed/strict_metrics.json` 的早期版本曾把删失轨迹直接算成 1.0/1.25 m 捕获率为 0。该文件已改为明确写 `null` 和 `censored=true`，原始错误版本保留为 `strict_metrics_censored_original.json`；这些旧 NPZ 只能报告 2.0 m 捕获率与 2.0 m 截断后的终点距离，不能据此声称严格捕获失败。

服务器 B0 50-seed 证据（`artifacts/lowlevel/server_b0_50seed_r2`）：

| 指标 | 数值 |
|---|---:|
| 成功率（2.0 m） | 1.00 |
| 三维 RMSE | 1.515 m |
| 水平 RMSE | 1.469 m |
| 垂向 RMSE | 0.324 m |
| 平均到达时间 | 27.94 s |
| 饱和率 / 约束违反 | 0 / 0 |
| 控制环 p95 的 p95 | 19.71 ms |

负证据保留在 `artifacts/lowlevel/server_b0_50seed`（无终端捕获、2.5 m，成功率 0.70）和 `server_b0_50seed_cap16_r1`（16 m 捕获、1.0 m，成功率 0.96）。

## 2. 论文方法拆解

### 2.1 总体结构

ODARRL 的三阶段课程是：

```text
Stage I：无海流、无障碍
  MPC 专家 → 行为克隆 → 基础策略 pi0

Stage II：随机海流、无障碍
  冻结 pi0 → PPO 学习残差 Δa → 扰动感知策略 pi1

Stage III：随机海流 + 障碍
  pi1 → 障碍感知奖励与观测 → 最终策略 pi2
```

本文只复现 Stage I 和 Stage II。Stage III 的避障模式机、障碍观测和 clearance 奖励不在本轮范围。

### 2.2 Stage I：MPC 专家与行为克隆

论文在无海流、无障碍环境用 MPC 采集：

```text
D_IL = {(x_t, a_t^MPC)}
```

其中 `x_t` 包含机器人状态和参考信息，`a_t^MPC` 是 MPC 产生的推进器级动作。然后用 MLP 最小化均方误差：

```text
L_IL = (1/N) * sum(||pi0(x_t) - a_t^MPC||^2)
```

论文规模：

| 项目 | 数值 |
|---|---:|
| rollout 数 | 50 |
| 样本数 | 250,000 |
| 训练样本 | 200,000 |
| 验证样本 | 50,000 |
| 最佳验证损失 | 0.00394 |

论文特别说明：`pi0` 不是最终控制器，只是后续 PPO 的稳定初始策略。`MPC` 和 `MPC-imitation policy` 是两种不同东西：前者在线求解优化，后者只是一个前馈神经网络。

#### 2.2.1 我们的偏差：单次 BC 不够，I 臂改用 DAgger

论文只描述了一次行为克隆。我们按同样方式先做了单次 BC：255,467 个无海流教师样本、`256,128` MLP、验证 MSE 很小（舵角 RMS 约 0.08°、RPM RMS 约 2.4 rpm）。但 50-seed 闭环门禁出现严重复合误差：student 三维 RMSE 比 teacher 差 `26.08 m`，成功率下降 `0.96`，RPM 在几百步内从 729 掉到 540 并持续振荡。

诊断结果是闭环分布漂移，不是标签错误：在教师轨迹的 state 上，policy 输出与教师动作几乎一致；一旦学生自己的上一舵角偏离教师分布，网络会把“上一舵角接近饱和”误当成 RPM 指令的强相关特征，随后速度环和积分项连锁发散。低通滤波不能修复，因为偏差是系统性的。

因此 I 臂的基础策略改为 **DAgger**，仍然是“冻结基础策略”的思路，但训练数据不再只来自教师轨迹：

```text
D_0 = 教师无海流轨迹
for k = 1, 2:
  用当前 pi_k 在无海流环境 rollout，并加入小幅动作噪声
  在每个学生到达的 state 上调用教师，得到 a_teacher
  把 (state, a_teacher) 加入 D_k
  在 D_0 ∪ ... ∪ D_k 上重新训练 pi_{k+1}
```

这不是把教师藏回评估回路：评估时只运行冻结的 `pi0`，教师只在离线数据采集阶段使用。正式报告必须把 I0 标为 “DAgger-imitated PD/PI”，不能写成一次 BC。

### 2.3 Stage II：冻结基础策略，学习残差

最终动作：

```text
a_t = a_t^base + Δa_t
a_t^base = pi0(x_t)
Δa_t ~ pi_theta(Δa_t | s_hat_t)
```

其中：

```text
s_hat_t = [z_t, c_t, c_hat_t]
```

- `z_t`：当前状态编码；
- `c_t`：参考轨迹相关指令；
- `c_hat_t`：DADE 从历史观测提取的扰动特征。

`pi0` 的参数冻结，只训练残差 actor、critic 和扰动编码器。这样做的目的是保留基础轨迹执行行为，把 PPO 的搜索空间限制在“海流引起的修正量”上。

### 2.4 DADE：双时间尺度扰动编码器

DADE 把历史观测分成两个窗口：

```text
短窗口：最近 10 步
长窗口：最近 50 步
```

每个窗口经过一个 Transformer encoder，然后均值池化：

```text
z_short = Pool(Transformer_short(Embed(history_short)))
z_long  = Pool(Transformer_long(Embed(history_long)))
c_hat   = MLP_fuse([z_short; z_long])
```

论文配置：

| 参数 | 数值 |
|---|---:|
| 短窗口 | 10 步 |
| 长窗口 | 50 步 |
| 输入嵌入维度 | 64 |
| 注意力头 | 4 |
| 前馈维度 | 128 |
| Dropout | 0.1 |
| 池化 | Mean pooling |
| 融合 MLP | 128-64-64 |
| 扰动特征维度 | 64 |

`c_hat` 不是显式海流估计，而是“海流对控制影响的隐表示”。

### 2.5 奖励

Stage II 奖励：

```text
r_t = w_prog * Δl_t
      - w_xy * e_t^H
      - w_yaw * e_t^psi
      - w_z * e_t^V
```

权重：

```text
w_prog = 15.0
w_xy   = 2.0
w_yaw  = 0.8
w_z    = 0.1
```

`Δl_t` 是沿参考路径的进度增量，`e_t^H` 是水平误差，`e_t^V` 是垂向误差，`e_t^psi` 是航向误差。

论文声称奖励还考虑控制平滑，但公式中没有显式的动作变化惩罚。我们的实现必须补充 `Δa` 惩罚，否则容易出现执行器抖动。

### 2.6 论文训练参数

| 参数 | 数值 |
|---|---:|
| 并行环境 | 64 |
| rollout 长度 | 1024 |
| PPO epochs | 4 |
| minibatches | 64 |
| actor 学习率 | `1.5e-4` |
| critic 学习率 | `1.5e-4` |
| gamma | `0.99` |
| GAE lambda | `0.95` |
| PPO clip | `0.15` |
| entropy 系数 | `5e-4` |
| 参考速度 | `0.85 m/s` |
| 路径长度 | `35 m` |
| 海流速度 | `0.2–0.4 m/s` |
| 海流方向 | 随机 |

### 2.7 论文结果与关键限制

论文报告的三维跟踪误差：

| 方法 | 三维平均误差 | 三维 RMSE |
|---|---:|---:|
| DARRL | 0.175 m | 0.312 m |
| PPO | 0.257 m | 0.436 m |
| MPC-imitation policy | 0.570 m | 0.894 m |
| A2C | 0.649 m | 0.903 m |
| SAC | 0.323 m | 0.582 m |
| VNRS-SAC | 0.236 m | 0.391 m |

消融：

| 方法 | 三维平均误差 | 三维 RMSE |
|---|---:|---:|
| PPO | 0.257 m | 0.436 m |
| Residual PPO without DADE | 0.224 m | 0.388 m |
| DARRL with DADE | 0.175 m | 0.312 m |

必须注意：论文比较的是 `MPC-imitation policy`，不是在线 MPC。`MPC-imitation` 是一个固定网络，海流变化后不能重新优化，所以它在海流下变差是合理的。**不能据此写成“RL 优于 MPC”。** 我们的复现必须加入在线 MPC 基线，否则结论不完整。

另一个注意点：论文的深度误差权重只有 `0.1`，水平误差权重是 `2.0`。因此论文的主要提升在水平方向，垂向提升较小。我们的任务是三维控制，不能照搬这个权重；奖励主项改为各向同性的三维位置误差，见 6.5。

## 3. 与我们系统的映射

| 论文 | 我们的实现 |
|---|---|
| BlueROV2 Heavy | REMUS100 |
| 8 个推进器 | 舵角 + 艉水平舵 + RPM |
| MarineGym | PVS REMUS100 + 我们的向量化 worker |
| 推进器级动作 | 执行器级动作 |
| 4 自由度参考 `x,y,z,psi` | 相同 |
| 6 自由度仿真 | PVS 六自由度 |
| 随机水平海流 | 均匀流、平滑时变流、线性水平空间剪切流 |
| DADE | 执行器感知双时间尺度编码器 |
| 障碍第三阶段 | 本轮不做 |

现有可复用代码：

- 制导：`settled_全局规划器/src/auv_planner/guidance.py:81`
- 跟踪控制器：`settled_全局规划器/src/auv_planner/control/remus100.py:88`
- PVS 适配器：`settled_全局规划器/src/auv_planner/dynamics/remus100.py:65`
- 现有 PVS 闭环测试：`settled_全局规划器/tests/test_pvs_remus100.py`
- 现有 BC 工具：`settled_局部规划器_反应式/src/auv_rl/bc/`
- 现有 PPO 训练配置风格：`settled_局部规划器_反应式/configs/ppo/isaaclab/`

## 4. 我们的方法定位

### 4.1 三条实验臂

为了避免“只换平台、没有自己的问题”，本文件同时定义三条实验臂。三条臂共用同一套 PVS 环境、观测契约、奖励、训练器和评估种子，只改变基础策略或专家来源。

**I 臂：优先主链（现有控制器模仿）**

```text
参考轨迹
  → ALOS 制导
  → 现有 PD/PI 控制器（教师）
  → 采集 teacher 数据
  → 单次 BC 失败 → 两轮 DAgger
  → 冻结 pi0
  → 残差 PPO
  → DADE / 执行器感知双时间尺度编码器
```

用途：第一优先级主链。复用现有控制器产生教师数据，用 DAgger 训练 `pi0` 固化基础行为，再只学习残差。I 臂不依赖 MPC 的 G2/G3A，可以在 G1 数据契约完成后直接进入 G3B。27D v1 I0 的 50-seed 门禁已通过，正式结果见 14.3；Goal10 后还需先得到 26D I0，不能直接把 27D I0 当作新主链。

**O 臂：并行消融臂（直接冻结 PD/PI）**

```text
参考轨迹
  → ALOS 制导
  → 现有 PD/PI 基础控制器
  → 冻结基础控制器
  → 残差 PPO
  → 执行器感知双时间尺度编码器
```

用途：不做 BC，直接冻结在线 PD/PI 后学习残差。它作为 I 臂的并行消融，用来回答“先做控制器模仿是否真的有必要”。空闲 GPU/CPU 足够时与 I 臂同时跑；资源不足时排在 I 臂之后。

**R 臂：论文复现臂**

```text
MPC 专家
  → 行为克隆 pi0
  → 冻结 pi0
  → 残差 PPO
  → DADE
```

用途：验证论文控制本体是否能在 REMUS100/PVS 上复现。R 臂需要先完成 G2/G3A；它是论文对照，不作为 I 臂的前置阻塞项。

三条臂的关键区别：

| 臂 | 教师/基础策略 | 是否 BC | 在线基础控制器 | 主要用途 |
|---|---|---|---|---|
| I | 现有 PD/PI | 是，得到 Controller-IL `pi0` | 否 | 优先主链 |
| O | 无教师，直接冻结 PD/PI | 否 | 是 | 并行消融：是否值得 BC |
| R | MPC 专家 | 是，得到 MPC-IL `pi0` | 否 | 论文复现 |

### 4.2 推荐主链

第一版的历史主链是 `G1 → G3B → I0 → I1 → I2`，但它跑在 27D v1 上；Goal10 之后的新主链改为：

```text
G3B-26D（27D I0 → 26D student，或重新采 26D BC/DAgger 数据）
  → I0-26D 50-seed 门禁
  → I1-26D（冻结 26D pi0 + 残差 PPO，无 DADE）
  → I2-26D（只有 I1 有正收益才加 DADE）
```

O 臂的 26D residual-only 预检已经由 Goal10 P4 覆盖，结论是“成功率不降但没有精度收益”。因此 O1/O2 不自动开工；先做奖励、PPO 超参、探索和 early-stop 的独立优化。R 臂仍在 G2/G3A 准备好后跑，作为论文复现对照，不阻塞 I 臂。三条臂共用 26D 环境契约和 G4/G5 的残差训练器；O 臂用在线 PD/PI，R/I 臂用各自冻结的 26D `pi0`，只改变基础策略来源。

最终报告不能把 O 臂或 I 臂写成“ODARRL 复现”。O 臂应明确命名为：

```text
Guidance-Aware Actuator-Level Residual PPO
GAR-PPO
```

`GAR-PPO` 只指 O 臂这条并行消融，不自动成为 I 主链的方法名。I 臂暂时使用中性名称 `Controller-IL` 或 `PD/PI-IL`；验证有效后若要取正式方法名，再单独命名，不能把 I 臂结果写进 GAR-PPO，也不能命名为 DARRL。

### 4.3 我们相对论文的差异

1. 同时比较三种基础策略：在线 PD/PI（O）、现有控制器模仿网络（I）和 MPC 模仿网络（R）。
2. 动作是 REMUS100 的真实执行器量，不是抽象推进器指令。
3. 残差输入包含 ALOS 的横向/垂向误差、路径曲率、参考速度和执行器状态。
4. 编码器不仅看海流，还看舵机滞后、RPM 响应和上一时刻执行器。
5. 必须加入在线 MPC 和在线 PD/PI 对照，避免把模仿网络误当成在线控制器。
6. 奖励使用各向同性的三维位置误差，不把深度误差权重压到 `0.1`；水平和垂向误差只作为审计指标。
7. 保留现有控制器的 safety envelope：网络输出永远经过幅值、速率和有效性检查。

## 5. 技术栈与目录

### 5.1 技术栈

| 层 | 选择 | 说明 |
|---|---|---|
| 被控对象 | PVS REMUS100 | 六自由度、执行器动态、现有适配器 |
| 制导 | `ThreeDAlosGuidance` | 复用现有三维 ALOS |
| 基础控制 | `Remus100TrackingController` | 复用现有 PD/PI |
| 专家控制 | 新增 LTV-MPC | 不修改 PVS 源码 |
| RL | rsl_rl PPO | 复用现有 PPO 后端 |
| 向量环境 | 训练 `BatchTorchVecEnv`；评估 PVS worker pool | 训练 4096 环境 GPU 批量推进，正式 50-seed 仍由 CPU PVS 复核 |
| 配置 | JSON | 参数唯一事实来源 |
| 记录 | JSON + Markdown + 小型 CSV | 大模型和 rollout 不进普通 Git |

### 5.2 建议新增目录

```text
settled_全局规划器/src/auv_planner/control/mpc_remus100.py
settled_全局规划器/src/auv_planner/experiments/odarrl_control.py
settled_全局规划器/tests/test_mpc_remus100.py

settled_局部规划器_反应式/src/auv_lowlevel/
  contracts.py          # 低层观测、动作、奖励契约
  env.py                # 单环境 PVS + ALOS + 基础控制器
  workers.py            # 多进程 PVS worker pool
  vector_env.py         # rsl_rl 兼容向量环境
  base.py               # 基础控制器适配器
  encoder.py            # DADE / 轻量双时间尺度编码器
  ppo.py                # 残差 PPO actor/critic
  evaluate.py           # 固定种子评估

settled_局部规划器_反应式/scripts/
  collect_odarrl_control_demos.py  # expert_type = mpc | pd_pi
  train_odarrl_control_bc.py       # 同一 BC 训练器，数据集不同
  train_odarrl_control_residual.py
  eval_odarrl_control.py

settled_局部规划器_反应式/configs/lowlevel/
  odarrl_control_base.json
  odarrl_control_smoke.json
  odarrl_control_r_arm.json
  odarrl_control_i_arm.json
  odarrl_control_o_arm.json
  odarrl_control_server_resources.json
  odarrl_control_eval.json
```

目录名可以在 G1 前调整，但**不能把 RL 训练代码塞进 `auv_planner` 核心包**，以免破坏现有导入边界。

## 6. 接口与契约

### 6.1 单步闭环

```text
参考轨迹 + 当前 PVS 状态 + 海流
  → ThreeDAlosGuidance
  → Reference
  → 基础控制器（PD/PI 或 pi0）
  → 基础执行器指令 a_base
  → 残差 actor
  → a_total = a_base + scaled_residual
  → 幅值/速率/时间戳检查
  → PvsRemus100Backend.step
  → 新状态、奖励、终止标志
```

### 6.2 观测向量：v1 兼容与 v2 26D 统一目标

第一版使用 27 维低层观测，不使用声呐。最终目标是 v2：只保留下表索引 `0–25` 的 26 维 core，索引 26 不进入新策略；v1 27D 只用于旧 checkpoint、历史数据和前后端回归。

这里统一的是低层 ODARRL/PVS 控制骨架的观测 schema，不影响 `goal8_ppo_isaaclab.md` 中 P1 二维避障的 17 维感知契约。两者任务、动作和后端不同，不能为了“统一维度”混用同一观测向量。

| 序号 | 含义 | 归一化 |
|---:|---|---|
| 0 | 航向误差 `wrap(psi_ref - psi)` | 除以 `pi` |
| 1 | 航向误差 `sin` | `[-1,1]` |
| 2 | 航向误差 `cos` | `[-1,1]` |
| 3 | 俯仰误差 `theta_ref - theta` | 除以 `pi/2` |
| 4 | 横滚角 `phi` | 除以 `pi/2` |
| 5 | 航向角速度误差 `r_ref - r` | 除以 `0.5` |
| 6 | 俯仰角速度误差 `q_ref - q` | 除以 `0.5` |
| 7 | 横滚角速度 `p` | 除以 `0.5` |
| 8 | 艇体系前向速度 `u` | 除以 `2.5` |
| 9 | 艇体系横向速度 `v` | 除以 `1.0` |
| 10 | 艇体系垂向速度 `w` | 除以 `1.0` |
| 11 | 对水前向速度 `u - u_current_body` | 除以 `2.5` |
| 12 | 速度误差 `u_ref - (u - u_current_body)` | 除以 `1.0` |
| 13 | 水平横向误差 | 除以 `2.0 m` |
| 14 | 垂向误差 | 除以 `2.0 m` |
| 15 | 路径进度 `s` | 除以路径总长 |
| 16 | 路径曲率 | 除以 `1.0 1/m` |
| 17 | 海流艇体系 x 分量 | 除以 `0.5 m/s` |
| 18 | 海流艇体系 y 分量 | 除以 `0.5 m/s` |
| 19 | 海流艇体系 z 分量 | 除以 `0.5 m/s` |
| 20 | 上一时刻舵角 | 除以 `15 deg` |
| 21 | 上一时刻艉水平舵角 | 除以 `15 deg` |
| 22 | 上一时刻 RPM | 除以 `1525` |
| 23 | 上一时刻残差舵角 | `[-1,1]` |
| 24 | 上一时刻残差艉水平舵角 | `[-1,1]` |
| 25 | 上一时刻残差 RPM | `[-1,1]` |
| 26 | 基础控制器速度积分 `speed_integral_rpm` | 除以 `400 rpm` |

索引 26 的积分项对应现有 `Remus100TrackingController` 的 `speed_integral_rpm`，用于保证 I 臂的 BC 输入包含教师控制器的积分状态。索引 0–26 共 27 维。I 臂和 O 臂使用真实值；R 臂的 MPC 专家没有相同的积分状态，固定填零。v1 中三条臂使用同一个 27 维 schema，否则不同臂的输入信息不一致；这不再代表后续所有控制器的长期 schema。

I 臂冻结 `pi0` 后不能依赖“现场仍在运行的教师控制器”提供这个积分值。环境必须实现一个 `ShadowSpeedIntegrator`：使用与 `Remus100TrackingController` 相同的增益、限幅、anti-windup 条件和 `dt`，只根据观测中的水相对前向速度、速度误差和参考速度递推，不使用教师动作，也不读取隐藏的海流向量。BC 采集时同时记录教师 `control()` 调用前的积分和 shadow 积分；两者必须在容差内一致。O 臂仍使用真实控制器的积分，R 臂固定为零。

第一版不能省略索引 26 再声称“纯 BC 失败是算法问题”，因为它会先把状态不可观测误判成学习能力不足。后续已决定：goal10 多环境基础稳定后启用 v2 的 26D schema，保留索引 0–25，删除索引 26，作为以后所有新控制器和新实验的默认输入。

当前实现边界：低频标量 `ObservationBuilder`、配置加载器、批量 vec-env、PPO 配置和 checkpoint 元数据都由 `observation_schema` 驱动，并校验 schema/dim 一致性。canonical 26D 环境配置为 `configs/lowlevel/odarrl_control_v2_26d.json`，canonical 批量训练配置为 `configs/lowlevel/odarrl_control_batch4096_v2_26d.json`；后续新实验必须使用这两者，而不是继续引用 27D base。26D checkpoint 的 PVS 评估强制要求匹配的 `observation_schema=v2`、`observation_dim=26` metadata；旧 27D checkpoint 只作为 legacy 兼容。该配置与 PVS 50-seed 复核均已完成，P3 核心门禁通过。

26D 不是无损删列，但也不是必须经过 27D 教师。goal10 的 P2.5 主 pilot 直接跑 26D；P3 再根据目标选择直接训练或可选迁移：

- **直接 26D（默认可用）**：在线 PD/PI 可以在控制器内部继续维护速度积分，但学习策略只接收 26D 观测；可以冻结 PD/PI 学残差 PPO，也可以从零训练 26D。该路线不要求先有 27D checkpoint。
- **复用旧 27D 能力（可选迁移）**：若要继续使用现有 27D `pi0`/r4 类策略，先做 naive drop-column BC 下界，再固定 27D teacher，在闭环中采集 student 实际访问的状态；teacher 使用完整 27D，student 只使用 26D core，标签为 teacher 动作，至少做一轮 DAgger 修正。
- 迁移时也可使用动作蒸馏、显式观测历史或循环网络补足时序信息；可以把 27D 网络除最后输入列外的权重复制给 26D 作为热启动，但删除列置零后仍必须做闭环训练和评估。
- 部署时不得偷偷读取在线教师积分；26D 学生的单帧输入必须只有约定的 26 维。若使用历史栈或循环网络，必须显式记录网络输入形状和时序结构。
- 不论直接训练还是迁移，都在 CPU PVS 上做同 seed 配对的 50-seed 对比；暂定通过门禁为：成功率下降不超过 `0.02`、三维 RMSE 劣化不超过 `0.05 m`、material saturation 不上升。直接 26D 路线以 B0/可用 27D 旧策略为参照。
- P2.5 主 pilot 和 P3 PVS 50-seed 门禁均已完成；所有新实验的默认输入统一为 26D，可以用 26D 策略作为 PPO 初始化，或按同一 26D 配置从零训练。27D 只保留显式 legacy 兼容入口，不再新增训练分支。
- 未通过则继续改直接训练基线或蒸馏数据和闭环损失，不把第 27 维重新塞回主 schema。现有 I0 的一次离线 BC 曾因复合误差在 50 seeds 上失败，因此选择迁移路线时必须包含 DAgger/闭环修正。

历史结果口径：G1–G4 已完成的 27D 数据、`pi0`、r1–r4 和对应报告继续作为 v1 结果保留，不回写成 26D，也不删除。r4 当前是 27D 兼容产物；除非只是为了补全旧对照，不再启动新的 27D 长训练。后续 I2/O2、批量 PPO 和新控制器只接受显式 `schema_version`，默认使用 26D。

状态编码器输入 `z_t = f_enc(obs_t)`，输出 64 维。参考指令 `c_t` 可以取观测中的参考误差和路径曲率子集，第一版不再单独增加高维 reference encoder。

“R/I 臂不喂显式海流”只表示索引 17–19 的海流艇体系分量置零；索引 11/12 的水相对前向速度和速度误差仍是允许的 DVL 类测量，不能被一起置零。否则 `ShadowSpeedIntegrator` 和教师模仿都会失去必要状态。

只置零索引 17–19 还不够。I/R 臂的配置必须同时关闭 `guidance.compensate_current` 和 `base_controller.use_current_field`，否则 ALOS 参考航向、参考速度或教师的速度误差会通过索引 0–14 间接编码真实海流。配置加载器已把这一组合作为硬约束；O 臂若保留显式海流，必须设置 `observation.expose_current=true` 并在结果中标记为 privileged。

### 6.3 动作向量

```text
a = [a_rudder, a_stern, a_rpm] in [-1, 1]^3
```

残差映射：

```text
Delta_rudder = 0.10 * 15 deg * a_rudder
Delta_stern  = 0.05 * 15 deg * a_stern
Delta_rpm    = 0.05 * 1525   * a_rpm
```

总执行器：

```text
rudder_cmd = clip(rudder_base + Delta_rudder, -15 deg, +15 deg)
stern_cmd  = clip(stern_base  + Delta_stern,  -15 deg, +15 deg)
rpm_cmd    = clip(rpm_base    + Delta_rpm,      0,       1525)
```

然后统一执行：

```text
舵速限制：|Delta rudder| <= max_rudder_rate * dt
艉水平舵速限制：|Delta stern| <= max_stern_rate * dt
时间戳检查：command.stamp_s 必须与当前 PVS 时间一致
```

网络永远不直接控制未经限制的 PVS 执行器。安全包络在环境侧，不在网络侧。

裁剪计数使用物理容差（舵角/艉舵 `1e-6 rad`、RPM `1e-3`），不把网络刚好落在 ±15° 上的浮点噪声记成饱和。正式报告的 `material saturation` 指“请求量被限制超过 1% 满量程”的步数比例；同时单独报告 teacher/student 的 at-limit 比例。当前路径下教师有 53.6% 的时间停在舵限、24.0% 停在艉舵限，不能用一个脱离教师行为的 5% “任何裁剪”阈值判断学生。

### 6.4 基础控制器适配

PD/PI 基础控制器是有内部状态的，包括积分项和上一时刻舵角。第一版规定：

- 基础控制器的积分和 anti-windup 仍按其名义指令更新；
- 残差是叠加在名义指令后的外部修正；
- 环境必须记录 `a_base`、`Delta_a`、`a_total`、`a_actual` 四个量；
- 任何幅值/速率违反都记 `constraint_violation`，不能静默裁剪后当作正常样本。

如果后续发现积分项与残差耦合导致不稳定，再改为“以总指令做 anti-windup”的适配器；这属于 G4 的诊断项，不阻塞 G1。

### 6.5 奖励 v1

> 本节是旧 I/O/R 臂的奖励 v1，保留作历史记录，不适用于 PC 臂。PC 的奖励口径以 0.1.9 的 `pc_command_track` 为准：C0/C1 使用手动命令跟踪加 hold bonus，C2–C4 使用 ALOS 命令跟踪且不包含路径进度、路径误差、曲率、`P_ref`、goal bonus 或 timeout penalty。

```text
progress = w_prog * (s_t - s_{t-1})
track_3d = -w_3d * ||p - p_ref||
track_yaw = -w_yaw * |wrap(psi - psi_ref)|
track_u  = -w_u * |u_water - u_ref|
action   = -w_action * ||a_t - a_{t-1}||^2
sat      = -w_sat * saturation_indicator
total    = progress + track_3d + track_yaw + track_u + action + sat + terminal
```

其中 `p_ref` 是参考路径上当前进度对应的最近点，`track_3d` 同时包含水平偏差和垂向偏差。水平和垂向误差不再分别进入奖励，只作为日志和评估指标，避免两项重复计权或深度项被低权重压制。

第一版权重：

| 权重 | 数值 | 说明 |
|---|---:|---|
| `w_prog` | 15.0 | 沿用论文 |
| `w_3d` | 2.0 | 各向同性三维位置误差，主跟踪奖励 |
| `w_yaw` | 0.8 | 沿用论文 |
| `w_u` | 0.5 | 速度误差 |
| `w_action` | 0.05 | 动作变化平滑 |
| `w_sat` | 0.10 | 饱和惩罚 |
| terminal goal | `+50` | 到达终点 |
| terminal timeout | `-20` | 超时 |
| terminal out-of-bounds | `-50` | 出界 |
| `ppo_reward_scale` | `0.01` | 只用于 PPO 输入，审计保留原始值 |

第一版不加入障碍物 clearance 奖励。若轨迹经过海床或边界，按 out-of-bounds 处理。

### 6.6 终止条件

下面保留旧 I/O/R v1 的通用口径；PC 臂的终止条件、滚转持续阈值、`termination_reason` 和 bootstrap 规则以 `0.1.9.1` 为准，不能用本节旧口径覆盖也不得回溯修改旧实验记录。

- 到达路径终点 `goal_radius` 内；
- 超出场景边界；
- 与海床/水面边界冲突（如果场景定义）；
- 达到最大 episode 步数；
- 状态出现 NaN/Inf；
- PVS 时间戳或执行器限制违反。

`terminated` 和 `truncated` 必须分开记录，PPO 的 time-limit bootstrap 不能混用。

## 7. 执行计划

执行顺序不等于章节编号。当前优先级是：

```text
Goal10 已完成：26D + BatchTorchVecEnv + PVS 50-seed 门禁
├─ O-residual 预检（P4）已完成       # online PD/PI，无 encoder；负结果
├─ I0-26D → I1-26D → I2-26D         # 新优先主链；先做 26D I0
├─ reward/超参/early-stop 优化       # 先于 O1/O2 encoder
└─ G2 → G3A → R0 → R1 → R2          # 论文复现对照，不阻塞 I
```

I 臂只需要 G1 的数据契约和 G3B 的教师数据，不需要等待 MPC 的 G2/G3A。O 臂的 residual-only 预检已经完成，但结果不支持直接进入编码器阶段；R 臂在 MPC 专家准备好后再进入，或只在剩余资源充足时并行。

### G0：文档冻结与接口确认

- [x] 阅读 `research/ODARRL.pdf` 的摘要、方法、实验和结论。
- [x] 新建本文件。
- [x] 明确只做控制本体，不做论文第三阶段避障。
- [x] 冻结 REMUS100、PVS、50Hz、三执行器、1.2m/s 参考速度等边界。
- [x] 冻结 R 臂（MPC-IL 复现）、O 臂（ALOS+PD/PI 残差）和 I 臂（PD/PI-IL 残差）三条实验臂。
- [x] 按用户要求加入 I 臂：现有控制器 → 数据采集 → BC `pi0` → 冻结 `pi0` → 残差 PPO + DADE。
- [x] 按用户要求将 I 臂设为优先主链；O 臂在空闲资源足够时并行，R 臂作为论文复现对照。
- [x] 用户确认 G1 的实验范围、默认参数和三条实验臂；I 臂优先，O 臂空闲资源并行，R 臂作复现对照。

**G0 验收**：文档审阅通过；未修改 PVS/IsaacLab/rsl_rl 源码；没有执行训练。

**Git 检查点**（用户明确要求提交时执行）：

```bash
cd "/home/bubble/水下机器人"
git status --short
git diff --check
git add "settled_局部规划器_反应式/doc/goals/goal9_odarrl_control_reproduction.md"
git commit -m "docs: add odarrl control reproduction plan"
git push origin main
```

不要使用 `git add -A` 或 `git commit -a`，当前工作树有大量前序未提交修改。

### G1：控制本体基线与数据契约

- [x] 新增低层 JSON schema 和 `odarrl_control_base.json`。
- [x] 实现三维平滑参考轨迹生成器（第一版为三维正弦 + 等弧长重采样）。
- [x] 实现随机海流：速度 `0.2–0.4 m/s`，方向随机，episode 内保持不变。
- [x] 实现 50 组固定评估种子，训练种子与评估种子分离。
- [x] 跑通 `ALOS + Remus100TrackingController + PvsRemus100Backend` 的 50Hz 单环境闭环（本地 seed 0–2 smoke）。
- [x] 实现 `ShadowSpeedIntegrator`，输入只允许来自观测索引 11/12（水相对前向速度、速度误差）和参考速度；验证它与在线 PD/PI 的 `speed_integral_rpm` 在固定种子下逐步一致。
- [x] 实现 27 维观测、3 维动作、奖励和终止条件。
- [x] 记录 `a_base`、`Delta_a`、`a_total`、`a_actual` 和约束违反次数。
- [x] 输出基线报告：水平/垂向/三维误差、航向误差、速度误差、饱和率、控制平滑度、到达时间；服务器 50-seed 已完成。
- [x] 增加配置级 hidden-current leak 校验，禁止非 oracle 臂通过制导或基础控制器间接读取真实海流。
- [x] 新增 G1 core tests、BC tests、既有 M2 闭环测试和 import guard 的本地验证入口。

**G1 验收**：

- 50 个固定种子全部可重复（服务器 `server_b0_50seed_r2` 已完成；同一配置可复跑）；
- 无 NaN/Inf；
- 观测、动作和奖励维度与 JSON 契约一致；
- 基线结果可画轨迹和误差曲线；
- 单步 PVS 成本、控制频率、p50/p95 延迟有记录。

**Git 检查点**：

```bash
cd "/home/bubble/水下机器人"
git status --short
git diff --check
git add "settled_局部规划器_反应式/configs/lowlevel" \
        "settled_局部规划器_反应式/src/auv_lowlevel" \
        "settled_局部规划器_反应式/tests"
git commit -m "feat(lowlevel): add remus100 control baseline contract"
git push origin main
```

### G2：在线 MPC 基线与专家控制器

G2 只服务于 B1/B2 和 R 臂，不是 I 臂或 O 臂的前置条件。

- [ ] 实现 `mpc_remus100.py`，第一版采用 LTV-MPC，不直接上完整 NMPC。
- [ ] 预测模型使用 PVS 当前状态附近的线性化模型；Jacobian 用有限差分。
- [ ] 决策量为 `[rudder, stern_plane, rpm]`，显式加入幅值和速率约束。
- [ ] 预测时域先设为 20 步、`dt=0.1 s`；求解频率 10Hz，控制保持到 50Hz。
- [ ] 如果 10Hz 解算仍过慢，降级为 4 自由度参考级 MPC，但必须记录模型差异。
- [ ] 实现在线 MPC 两个基线：`MPC-nominal`（无海流前馈）和 `MPC-current`（有海流前馈，标为 privileged upper bound）。
- [ ] 在无海流环境验证 MPC 能稳定跟踪，再进入海流评估。

**G2 验收**：

- MPC 单步求解时间有 p50/p95；
- 约束违反次数为零或全部有记录；
- `MPC-nominal` 和 `MPC-current` 分开报告；
- 不能用 MPC 专家在无流下的结果冒充在线 MPC 在海流下的结果。

**Git 检查点**：

```bash
cd "/home/bubble/水下机器人"
git status --short
git diff --check
git add "settled_全局规划器/src/auv_planner/control/mpc_remus100.py" \
        "settled_全局规划器/tests/test_mpc_remus100.py" \
        "settled_局部规划器_反应式/configs/lowlevel" \
        "settled_局部规划器_反应式/scripts"
git commit -m "feat(control): add remus100 mpc baseline and expert"
git push origin main
```

### G3A：R 臂——MPC 专家数据与行为克隆

G3A 只属于 R 臂。I 臂的教师数据在 G3B 采集，不应等待 MPC 专家。

- [ ] 用无海流、无障碍环境采集 MPC 专家数据。
- [ ] 在服务器完成资源 preflight；数据采集进程记录 GPU、CPU 绑定、worker 数和输出目录。
- [ ] 目标样本量按 250,000 条设计；按 1.2m/s、35m、50Hz 估算，单条约 1458 步，需要约 170 条 rollout，不能只采 50 条就宣称对齐论文。
- [ ] 数据字段至少包含观测、参考、MPC 动作、时间戳、episode id、seed 和 PVS 配置哈希。
- [ ] 训练 MLP `pi0`，输入为观测，输出为 3 维执行器动作。
- [ ] 做训练/验证切分，记录最佳验证 MSE。
- [ ] 在无海流环境中比较在线 MPC 与 `pi0` 的轨迹、误差和控制量。
- [ ] 如果 `pi0` 不能复现 MPC 的基本跟踪，停止进入残差 PPO，先修数据或网络。

**G3A 验收**：

- `pi0` 在无海流环境的多数轨迹段误差小于 `0.05 m`（对齐论文的定性结果）；
- 验证集 MSE 稳定，无过拟合；
- 数据 manifest 包含 SHA-256；
- 大型 demo 数组不进入普通 Git，只提交 manifest 和统计。

**Git 检查点**：

```bash
cd "/home/bubble/水下机器人"
git status --short
git diff --check
git add "settled_局部规划器_反应式/src/auv_lowlevel" \
        "settled_局部规划器_反应式/scripts/collect_odarrl_control_demos.py" \
        "settled_局部规划器_反应式/scripts/collect_odarrl_control_dagger.py" \
        "settled_局部规划器_反应式/scripts/merge_odarrl_control_demos.py" \
        "settled_局部规划器_反应式/scripts/train_odarrl_control_bc.py" \
        "settled_局部规划器_反应式/configs/lowlevel" \
        "settled_局部规划器_反应式/artifacts/lowlevel"
git commit -m "feat(bc): add mpc expert collection and imitation baseline"
git push origin main
```

只提交小型 manifest、JSON、报告和必要的配置；不提交完整 demo 数组和 checkpoint。

### G3B：I 臂——现有控制器专家数据与行为克隆

历史说明：本节前四条 27D 记录是 v1 兼容链。2026-10-31 已按 14.0.2 在 26D v2 下重新完成教师采集、BC/DAgger 和门禁，后续 I1 必须使用 26D checkpoint。

- [x] 在无海流、无障碍环境采集 `ALOS + Remus100TrackingController` 教师数据；教师参数必须与 B0 完全一致（脚本已实现并完成 4-episode smoke）。
- [x] 在服务器完成资源 preflight；I 臂数据采集使用 `taskset 160-191`、16 workers 和独立输出目录，记录在 `odarrl_control_server_resources.json`。
- [x] 复用 G3A 的 250,000 样本预算、观测 schema 和 MLP 架构，只把 `expert_type` 从 MPC 改为 `pd_pi`，避免数据量和模型容量成为混杂因素（默认 180 episodes，约 250k 样本；正式数量以服务器 manifest 为准）。
- [x] 每个样本保存 27 维观测，其中索引 26 是教师控制器的 `speed_integral_rpm`；必须在调用教师 `control()` 之前采样，不能使用该步更新后的积分值。
- [x] 同一时刻记录 shadow 积分；如果它与教师积分不一致，先修递推或 anti-windup，再训练 BC（环境内逐步断言，当前 margin 为 0）。
- [x] BC 标签使用教师控制器输出的执行器指令 `a_teacher_cmd`，同时记录 PVS 的实际执行器 `a_actual`；不能把两者混成一个字段。
- [x] 做训练/验证切分，记录动作 MSE、分执行器误差和最佳验证 checkpoint（按 episode 切分，避免同一轨迹泄漏）。
- [x] 在无海流环境闭环比较 `B0`、在线 PD/PI 和 `I0 = pi0` 的三维误差、饱和率与动作平滑度；50-seed 门禁已完成。
- [ ] 在有海流环境只做诊断性评估，确认 `pi0` 的退化程度；第一版残差训练仍以无海流 `pi0` 为起点。
- [x] 单次 BC 50-seed 门禁失败：三维 RMSE 差 `26.08 m`、成功率下降 `0.96`；保留为负结果，不删除。
- [x] 用两轮 DAgger 修复复合误差：第 1 轮 40 episodes / 98,938 样本，第 2 轮 60 episodes / 86,109 样本。
- [x] 重新跑 I0 50-seed 门禁：三维 RMSE 差 `0.0203 m`、成功率下降 `0.0`、material saturation `0.0073`，通过。
- [x] 在 26D v2 下重跑同一流程：180 episodes / 255,467 样本教师数据、单次 BC 负结果、一轮 DAgger 和 50-seed 门禁；正式结果见 14.0.2。

**G3B 验收**：

- 验证集动作 MSE 稳定，分执行器误差没有只被 RPM 或舵角一项掩盖；
- 无海流闭环中，`I0` 相对在线 PD/PI 的三维 RMSE 差距不超过 `0.05 m`，且无持续饱和或失稳（已满足：`0.0203 m`）；
- 保存 teacher-student 闭环差距、动作误差和失败样本；
- 数据 manifest 包含 SHA-256，完整 demo 数组不进入普通 Git。

**Git 检查点**：

```bash
cd "/home/bubble/水下机器人"
git status --short
git diff --check
git add "settled_局部规划器_反应式/src/auv_lowlevel" \
        "settled_局部规划器_反应式/scripts/collect_odarrl_control_demos.py" \
        "settled_局部规划器_反应式/scripts/train_odarrl_control_bc.py" \
        "settled_局部规划器_反应式/configs/lowlevel" \
        "settled_局部规划器_反应式/artifacts/lowlevel"
git commit -m "feat(bc): add pd-pi expert collection and imitation baseline"
git push origin main
```

只提交小型 manifest、JSON、报告和必要的配置；不提交完整 demo 数组和 checkpoint。

### G4：I/O 臂——冻结基础策略，训练残差 PPO

本节历史记录针对 27D v1 和 PVS 64-worker；Goal10 后的 26D I1/O1 使用 `BatchTorchVecEnv` 4096×256，具体顺序见 14.0。

- [x] 实现 `BaseControllerAdapter`，同时支持在线 `Remus100TrackingController`（O 臂）和冻结 `pi0`（I 臂）。
- [x] I 臂的 27 维观测由环境维护 shadow 积分；不能在评估时偷偷启动一个在线 PD/PI 作为隐藏输入。
- [x] 实现残差 actor/critic，输出 3 维归一化修正；I1 使用 rsl_rl 公共 `ActorCritic`，初始 actor 输出层零初始化，保证训练起点严格等于 I0。
- [x] O 臂冻结 PD/PI 参数；I 臂冻结 `pi0` 全部参数；两种情况都只训练残差网络和编码器。
- [x] 实现 64 个 PVS worker；每个 worker 独立进程，父进程通过 Pipe 汇总，超时或崩溃后回收重启。
- [ ] 每次正式运行固定一张 GPU 和一段 CPU 核；如果有多张空闲 GPU，优先并行跑独立 seed/臂/消融，而不是强上单任务多卡。
- [x] 实现 rsl_rl 兼容的向量环境接口：`get_observations/reset/step`、`time_outs`、`infos["observations"]`、`episode_length_buf` 和 `infos["log"]` 均已按 rsl_rl 2.3.3 公共契约验证。
- [x] 使用论文初始参数：64 envs、rollout 1024、4 epochs、64 minibatches、`lr=1.5e-4`、`gamma=0.99`、`lambda=0.95`、`clip=0.15`、`entropy=5e-4`；全部记录在独立 JSON `odarrl_control_ppo_i1.json`；r2 探索不足后的 r3 参数偏差单独记录在 14.4.2。
- [x] I 臂跑 smoke：8 环境、32 步，`initial_residual_abs_max=0`、无 NaN、无 worker 重启，证据 `artifacts/lowlevel/i1_smoke_8x32`。
- [x] 历史 27D 路线扩到 64 个 PVS worker 做稳态测试；64×256 窗口采集吞吐 `1,659.9 steps/s`，未达到当时的 `3,200 steps/s` 门禁，原因是共享服务器 CPU 争用而不是 PVS 单步异常：单 worker 在独立核上 p50 约 `16.5 ms`，64 worker 平均只有约 `38.6 steps/s/worker`。Goal10 后训练主路径改为 `BatchTorchVecEnv`，该门禁只作为历史 PVS 路线的限制。
- [x] I 臂复用同一训练入口，只把基础策略从在线 PD/PI 换成冻结 `pi0`；同时支持 `--resume` 从小 checkpoint 继续训练。
- [ ] I1-26D 正式训练前，先在无海流环境记录残差幅值；如果残差主要是在补偿 26D BC/蒸馏误差，先回到 26D I0 迁移步骤。
- [ ] O 臂在空闲资源允许时并行跑 smoke 和训练；资源不足时排在 I 臂之后。
- [x] 27D I/O 两臂已做过历史训练和评估，保存了 checkpoint、配置展开文件和 `config_sha256`；r3 已停止，不作为新主链。新的 26D I/O 路线按 14.0 执行。
- [x] 记录 `a_base`、`Delta_a`、`a_total`、`a_actual`、any-clip、material saturation 和 residual clip rate；训练日志按 episode 写入 TensorBoard。

**G4 验收**：

- smoke 无 NaN/Inf、无时间戳倒退、无 worker 泄漏；已满足；
- 新训练门禁为 `BatchTorchVecEnv` 4096×256 端到端至少 `200k env-steps/s`；历史 64 个 PVS worker 的 `1,659.9 steps/s` 只作为旧路线限制保留，不再阻塞 26D 批量训练；
- 每次运行都有 `odarrl_control_server_resources.json`，实际 GPU、CPU 绑定和输出目录与 manifest 一致，且没有 CPU/GPU 超卖；
- 残差动作在训练前后都经过限幅；
- O 臂与冻结 PD/PI、I 臂与 `I0` 分别在相同 50 个种子上比较，至少完成一次完整评估；
- I 臂必须额外报告无海流时的残差幅值，证明残差不是在补偿 BC 模仿误差；
- 如果任一臂的残差 PPO 不能改善或显著增加饱和/抖动，记录负结果，不进入 G5 的完整 DADE 阶段。

**Git 检查点**：

```bash
cd "/home/bubble/水下机器人"
git status --short
git diff --check
git add "settled_局部规划器_反应式/src/auv_lowlevel" \
        "settled_局部规划器_反应式/scripts/train_odarrl_control_residual.py" \
        "settled_局部规划器_反应式/configs/lowlevel" \
        "settled_局部规划器_反应式/tests"
git commit -m "feat(rl): add residual ppo for online and imitation bases"
git push origin main
```

### G5：执行器感知的双时间尺度编码器

- [x] I2-26D 只在 I1-26D 通过 50-seed 且出现可重复正收益后启动；O2 还要以 O-residual 的奖励/超参优化先产生正收益为前置条件。I2 已启动并完成；O2 仍未启动。
- [x] 先实现轻量版：短窗 10 步、长窗 50 步，两个 MLP/GRU 编码器，替代完整 Transformer。轻量 MLP 为负结果，见 14.4.8。
- [x] 再实现 DADE 版：两个 Transformer encoder，64 维嵌入、4 头、128 前馈、dropout 0.1、mean pooling、128-64-64 融合。完整 DADE 已训练和评估，但未通过相对 I2-a 的强基线门禁，见 14.4.8。
- [x] 编码器输入包含执行器状态、上一时刻残差、海流艇体系分量和路径曲率。I 臂的显式海流槽位按契约置零，海流只能由历史隐式推断；训练历史由同一 26D 观测逐控制步或按 stride 抽样构成。
- [ ] O2 和 I2 使用完全相同的编码器结构、窗口和训练预算；只允许基础策略不同。
- [x] 在 50Hz 下测单步推理 p50/p95；超过 20ms 必须降级。DADE m20/m39 的 CPU 单线程 p95 分别为 `0.646/0.638 ms`，通过。
- [x] 做三组消融：无编码器、轻量双窗、完整 DADE。无编码器为 I1，轻量 MLP 和完整 DADE 均为负结果。
- [x] I 臂做 `I0 → I1 → I2` 消融，分别回答 BC 初始化和编码器是否各自带来收益。I0 通过 teacher-student 门禁，I1 RPM-only 通过，I2-a 显式历史栈通过；I2-b DADE 未通过强基线门禁。
- [x] 做时间窗消融：10/50 步与 10/250 步（或 10Hz 抽样下的 10/50 步）。10Hz 抽样在四个有海流条件上均低于 50Hz DADE，为负结果；见 14.4.8。
- [x] 比较水平、垂向、三维 RMSE，不只报水平误差。DADE 在严格条件下主要改善垂向，水平甚至在 m20/m39 的 1.25 m `0–49` 上轻微退化，已单列。

**G5 验收**：

- 完整 DADE 的推理 p95 小于 20ms，或明确降级；
- I2 相对 I1 是主门禁；O2 相对 O1 作为并行对照，DADE 在三维 RMSE 上至少有可重复改善；
- 垂向误差不能被水平改善掩盖；
- 训练和评估使用同一个编码器配置哈希。

**Git 检查点**：

```bash
cd "/home/bubble/水下机器人"
git status --short
git diff --check
git add "settled_局部规划器_反应式/src/auv_lowlevel/encoder.py" \
        "settled_局部规划器_反应式/src/auv_lowlevel/ppo.py" \
        "settled_局部规划器_反应式/configs/lowlevel" \
        "settled_局部规划器_反应式/tests"
git commit -m "feat(rl): add actuator-aware dual-horizon disturbance encoder"
git push origin main
```

### G6：完整评估与论文对照

- [ ] 固定 50 个评估种子，所有方法共用。
- [ ] 在服务器执行完整评估；不同方法/seed 可以按空闲 GPU 并行，但必须使用同一 PVS 配置、同一参考轨迹和同一组评估种子。
- [ ] 至少评估：在线 PD/PI `B0`、MPC-nominal `B1`、MPC-current `B2`、MPC-IL `R0`、PD/PI-IL `I0-26D`、PPO from scratch `S0`、`R1/R2`、`O1/O2`、`I1-26D/I2-26D`；27D 结果只作为历史对照另表列出。
- [ ] 记录水平平均误差、垂向平均误差、三维平均误差、三维 RMSE、航向 RMSE、速度 RMSE。
- [ ] 记录成功率、超时、出界、最小边界余量、路径进度、到达时间。
- [ ] 记录舵角/艉水平舵/RPM 饱和率、舵速饱和率、动作抖动、控制能量。
- [ ] 记录策略推理 p50/p95 和 PVS 环境步吞吐。
- [ ] 单独记录 I 臂的 teacher-student 闭环差距，以及无海流/有海流时的残差幅值。
- [ ] 生成轨迹图、误差曲线、消融表和最终报告。
- [ ] 报告必须明确区分 R、O、I 三条臂，不能把 I 臂的收益写成 ODARRL 复现收益。

**G6 验收**：

- 50 个种子结果齐全，无选择性删除失败；
- 所有方法使用同一 PVS、同一参考轨迹、同一海流种子；
- 在线 MPC 与 MPC-IL 分表；
- 在线 PD/PI 与 PD/PI-IL 分表；
- I0-26D 必须先通过 teacher-student/蒸馏差距门禁，才能进入 I1-26D/I2-26D 结论；
- 至少一个消融能解释性能来源；
- 如果 I2 没有超过 PD/PI 或在线 MPC，报告主链负结果并建议保留传统控制器；O2 的负结果单独记录，不替换 I2 的主链结论。

**Git 检查点**：

```bash
cd "/home/bubble/水下机器人"
git status --short
git diff --check
git add "settled_局部规划器_反应式/scripts/eval_odarrl_control.py" \
        "settled_局部规划器_反应式/artifacts/lowlevel" \
        "settled_局部规划器_反应式/doc/goals/goal9_odarrl_control_reproduction.md"
git commit -m "docs(lowlevel): record odarrl control reproduction results"
git push origin main
```

只提交小型 JSON、Markdown、图表和 manifest；不提交 checkpoint、完整 rollout、视频和大数组。

### G7：结论与后续决策

- [ ] 汇总 G1–G6 的证据。
- [ ] 判断是否保留 R 臂、O 臂、I 臂或只保留传统基线。
- [ ] 明确哪些结论来自论文，哪些来自我们的实现假设。
- [ ] 明确是否值得进入论文第三阶段避障；如果控制本体没有优势，不进入。
- [ ] 给出下一步：继续 Controller-IL、保留 GAR-PPO 作为并行方案、转在线 MPC、或回到 PD/PI。

**G7 验收**：形成一份可独立阅读的结论报告，失败也保留。

**Git 检查点**：

```bash
cd "/home/bubble/水下机器人"
git status --short
git diff --check
git add "settled_局部规划器_反应式/doc/goals/goal9_odarrl_control_reproduction.md" \
        "settled_局部规划器_反应式/artifacts/lowlevel"
git commit -m "docs(lowlevel): conclude odarrl control reproduction"
git push origin main
```

## 8. 实验矩阵

| ID | 方法 | 基础策略 | 残差 | 编码器 | 海流 | 用途 |
|---|---|---|---|---|---|---|
| B0 | PD/PI | 无 | 无 | 无 | 是 | 现有控制器基线 |
| B1 | MPC-nominal | 无 | 无 | 无 | 是 | 无海流前馈的在线 MPC |
| B2 | MPC-current | 无 | 无 | 无 | 是 | 有海流前馈的 privileged 上界 |
| R0 | MPC-IL `pi0` | MPC 专家 | 无 | 无 | 是 | 论文初始化策略 |
| R1 | Residual PPO | MPC-IL | 是 | 无 | 是 | 论文残差消融 |
| R2 | DARRL | MPC-IL | 是 | DADE | 是 | 论文控制本体复现 |
| O0 | PD/PI 冻结 | PD/PI | 无 | 无 | 是 | I 臂的直接残差对照基础 |
| O1 | GAR-PPO-light | PD/PI | 是 | 轻量双窗 | 是 | 并行消融第一版 |
| O2 | GAR-PPO-DADE | PD/PI | 是 | 执行器感知 DADE | 是 | 并行消融完整版 |
| I0 | PD/PI-IL `pi0` | PD/PI 专家 | 无 | 无 | 是 | 优先主链的 BC 初始化 |
| I1 | Controller-IL Residual PPO | PD/PI-IL | 是 | 无 | 是 | 主链残差消融 |
| I2 | Controller-IL + DADE | PD/PI-IL | 是 | DADE | 是 | 优先主链完整版 |
| S0 | PPO from scratch | 无 | 无 | 无 | 是 | 从零学习对照 |

Goal10 P4 的位置需要单独写清：它是 **online PD/PI + 26D + residual-only + 无 encoder** 的 O 臂预检，不是上表中的 O1/O2，也不是 I1。P4 的 `experiment.arm="I"` 只是历史配置标签，实际 `base_policy.mode="online_pd_pi"`；任何结果表都必须按基础策略归臂。P4 已经给出“成功率不降、三维 RMSE 没有正收益”的负结果，因此 O1/O2 的启动条件是先解决奖励/超参/探索/early-stop，而不是直接加 encoder。

观测口径：从 2026-10-31 起，新跑的 I1/I2/O1/O2/S0 以及新的 B/R 残差实验默认使用 26D。已有 R0–R2、I0/r1–r4 等 27D 结果只作为 v1 历史结果和兼容对照，不重跑成 26D；若要复用它们的策略能力，另走可选的 BC/DAgger/蒸馏 student 路线。

这里的“是否输入海流”必须区分三个对象：

- **`B0 PD/PI`**：当前实现是 current-aware 教师，ALOS 和速度误差都使用真实海流；它是 privileged 参考基线，不是“完全不知道海流”的在线 PD/PI。无海流 `I0` 与 B0 的比较必须同时报告这一点。
- **R 臂的 MPC 专家**：只在无海流、无障碍环境离线采数据，不接收海流向量；由它蒸馏出的 `pi0` 在残差阶段也不接收索引 17–19 的显式海流，只依赖 DADE 从历史中隐式推断。
- **`B1 MPC-nominal`**：在线 MPC，但预测模型不做海流前馈；海流只作用在 PVS plant 上。
- **`B2 MPC-current`**：在线 MPC，并向预测模型提供精确海流向量作为 privileged feedforward。它只是不可部署的性能上界和诊断基线，不是 R 臂，也不是论文的 Stage-I MPC 专家。

实验顺序：

```text
Goal10 已完成
├─ O-residual 预检（P4）：已完成，负结果
├─ 26D I0 迁移/重训 → I0 50-seed 门禁
├─ I1-26D → I2-26D             # I 臂新主链
├─ 奖励/超参/early-stop → O1 → O2  # O 臂仅在优化后再继续
└─ G2 → G3A → R0 → R1 → R2     # 论文复现对照

B0 是 G1 在线基线；B1/B2 依赖 G2；S0 可选，用于确认从零学习难度。
```

历史第一版主链是 27D 的 `G3B → I0 → I1 → I2`；Goal10 后的新主链必须先得到 `I0-26D`，且通过 teacher-student 或蒸馏 student 的 50-seed 门禁，再进入 I1-26D。O 臂不能因为“训练后端已经很快”就跳过负结果诊断；R 臂需要 G2/G3A，作为论文复现对照，不阻塞 I 臂。

## 9. 评价指标与门禁

### 9.1 指标

| 类别 | 指标 |
|---|---|
| 跟踪 | 水平平均误差、垂向平均误差、三维平均误差、三维 RMSE |
| 姿态 | 航向 RMSE、俯仰 RMS、横滚 RMS |
| 速度 | 对水速度 RMSE、速度超调 |
| 任务 | 成功率、超时率、出界率、路径进度、到达时间 |
| 执行器 | 舵角饱和率、艉水平舵饱和率、RPM 饱和率、舵速饱和率 |
| 平滑 | 动作变化 RMS、动作抖动、控制能量 |
| 计算 | 策略推理 p50/p95、MPC 求解 p50/p95、环境步吞吐 |
| 稳定性 | NaN/Inf、时间戳违反、worker 泄漏、reset 串扰 |

### 9.2 第一版门禁

进入 G6 前必须满足：

- B0 在 50 个固定种子上可重复；
- I0 在 50 个固定种子上完成 teacher-student 差距评估，并达到 G3B 门禁；
- I0 的 saturation 门禁使用 material clipping（超过 1% 满量程）而不是浮点噪声触发的 any-clip；必须同时报告 teacher/student 的 at-limit 比例；
- I1 smoke 无 NaN/Inf、无时间戳倒退、无 worker 泄漏；O1 若并行运行也必须通过同一 smoke；
- 训练使用 `BatchTorchVecEnv` 时，`4096×256` 端到端吞吐至少 `200k env-steps/s`；CPU PVS 评估只需满足 50-seed 评估时限，并如实记录吞吐和 p95；
- 观测、动作、奖励维度与 JSON 完全一致；
- 所有动作经过幅值和速率检查；
- 评估和训练使用不同的随机种子池；
- 单步策略推理 p95 小于 `20ms`（50Hz 要求）。

最终结论门禁：

- I2 相对 B0 的三维 RMSE 至少有可重复改善；如果只改善水平误差而垂向变差，不能宣称三维控制更好；
- I2 不得显著增加饱和率、动作抖动或出界率；
- I2 相对 I1 的改善必须与 I1 相对 I0 的改善分开报告，不能把 BC 初始化收益算成 DADE 收益；
- O2 作为并行消融，必须与 I2 使用相同的残差训练预算和评估种子，否则不能回答“是否值得先做控制器模仿”；
- 必须报告与 B1/B2 在线 MPC 的比较；
- 如果 I2 只与 R0（MPC-IL）比较，不能得出“优于 MPC”的结论；
- 如果所有 RL 方法都不如 B0/B1/B2，保留传统控制器，记录负结果。

## 10. 训练与算力

所有正式的数据采集、BC 训练、残差 PPO 训练和完整评估都在服务器执行。本地只做代码编辑、单元测试、静态检查和小规模 smoke；不能把本地跑出的短程结果当作正式训练证据。

每次进入 G3A/G3B/G4/G5/G6 前，先做服务器资源 preflight，记录：

- `hostname`、执行日期、仓库绝对路径和当前 Git commit；
- `nvidia-smi` 的 GPU 编号、显存占用、利用率和进程列表；
- `nproc`、可用 CPU 核数、内存、磁盘余量和 `taskset` 可用范围；
- Python、torch、CUDA、PVS、rsl_rl 版本和 `config_sha256`；
- 本次允许使用的 GPU 列表、每个并行任务的 CPU 核范围和输出目录。

禁止把服务器密码、token 或私钥写进仓库、JSON、日志或本文件。

### 10.1 单次运行的并行规模

Goal10 已经验证训练和评估应使用不同后端：

| 用途 | 后端 | 默认规模 | 证据/约束 |
|---|---|---:|---|
| PPO 训练 | `BatchTorchVecEnv` | 4096×256 | 26D、float64 物理、compile default；Goal10 长训练墙钟约 `267k env-steps/s`，峰值 allocated `424 MiB` |
| 等预算训练对照 | `BatchTorchVecEnv` | 1024/256 | 资源不足可降级；4096 吞吐和等预算质量最接近 B0 |
| 正式 50-seed 评估 | CPU PVS worker pool | 64–128 workers | 以保真复核为准，不使用 batch 后端替代正式结论 |
| 故障定位/smoke | CPU PVS 或小 N batch | 小规模 | 用来定位 reset、路径和物理差异，不作为吞吐门禁 |

建议：

- 新训练直接使用 `configs/lowlevel/odarrl_control_v2_26d.json` 和 `BatchTorchVecEnv`，不再用 64 个 PVS 进程当训练主路径；
- 不允许为了“上 4096”启动 4096 个 PVS 进程；那是 CPU 进程数，不是 GPU 批量环境；
- 1024 环境是资源不足时的合法降级，但必须同时报告墙钟吞吐和 20/50-seed 质量，不能只报 collection 吞吐；
- 新编码器先在 4096 batch 后端做短程试跑，再决定是否扩大训练；
- 任何 batch 训练得到的候选都必须回到 CPU PVS 做 50-seed 评估，二者不能混用数字。

### 10.2 多 GPU 并行实验

如果 preflight 显示有多张空闲 GPU，优先并行跑独立实验组，而不是强行把单个 PVS 实验做成多 GPU：

- 两张卡空闲时，优先同时跑 I 臂和 O 臂；第三张卡只有在 G2/G3A 准备好后才用于 R 臂；
- 每个任务绑定一张 GPU：`CUDA_VISIBLE_DEVICES=<gpu_id>`；
- 每个任务使用独立输出目录、独立 `config_sha256`、独立日志和 checkpoint；
- 每个任务分配互不重叠的 CPU 核范围，例如 `taskset -c 0-15`、`taskset -c 16-31`；
- 默认不做 DDP；这套 PPO 网络较小，多卡数据并行未必比“多开独立实验”更有效；
- 只有实测单卡利用率长期偏高、单次训练成为瓶颈时，才评估 DDP 或其他分布式方案。

适合并行的是相互独立的实验，例如不同臂、不同 seed、不同编码器窗口或不同残差缩放；不要并行写同一个输出目录。启动前写入 `odarrl_control_server_resources.json`：

```json
{
  "schema_version": "auv_server_resources.v1",
  "host_role": "training_server",
  "gpu_ids": [0, 1],
  "max_parallel_runs": 2,
  "cpu_ranges": ["0-15", "16-31"],
  "allow_gpu_oversubscription": false,
  "allow_cpu_oversubscription": false
}
```

上面的 GPU 编号和 CPU 范围只是格式示例，实际值必须来自本次 preflight，不能照抄。如果只有一张空闲 GPU，就串行跑单次实验；如果多张卡空闲但 CPU 核不足，也不能盲目多开。

### 10.3 服务器运行约束

- 长任务使用 `tmux`、`nohup` 或等效方式，不能依赖易断开的交互式终端；
- 启动后在 manifest 中记录 GPU、CPU 绑定、环境数、worker 数、开始时间和父进程；
- 并行任务之间不能共享可写 checkpoint、TensorBoard 目录、demo 数组或评估输出；
- 每次扩展并行规模前先跑短 smoke，确认吞吐、显存、内存和 p95 延迟没有恶化；
- 如果资源争用导致吞吐下降、NaN、worker 超时或评估延迟抖动，立即降低并行度并保留证据。

## 11. JSON 配置草案（与当前实现对齐）

历史 27D 兼容配置以 `configs/lowlevel/odarrl_control_i_residual.json` 为准，schema 为 `auv_odarrl_control.v1`。新实验的环境 canonical 配置是 `configs/lowlevel/odarrl_control_v2_26d.json`，训练配置另用 `configs/lowlevel/odarrl_control_batch4096_v2_26d.json`。下面的草案仍保留为历史字段说明；其中 `encoder` 和 `ppo` **尚未接入 env parser**，训练参数必须留在独立训练 JSON，避免环境 schema 承担训练配置。

注意：下面草案仍是 v1 兼容配置，`observation.dim: 27` 不是新实验默认。P2.5 起的批量主 pilot 和 P3 后新增的正式配置必须写 `observation.dim: 26` 和显式的 `observation.schema_version: "v2"`；旧 v1 配置只用于回归和旧 checkpoint。

```json
{
  "schema_version": "auv_odarrl_control.v1",
  "experiment": {
    "name": "odarrl_i_residual_ppo",
    "arm": "I",
    "seed": 20000,
    "output": "artifacts/lowlevel/i_residual"
  },
  "plant": {
    "backend": "pvs_remus100",
    "vehicle": "REMUS100",
    "dt_s": 0.02,
    "control_hz": 50.0,
    "max_episode_steps": 2500,
    "goal_radius_m": 2.0,
    "bounds_m": [[-200.0, 200.0], [-200.0, 200.0], [0.0, 200.0]]
  },
  "reference": {
    "generator": "sinusoidal_3d_arclength",
    "length_m": 35.0,
    "speed_mps": 1.2,
    "x_span_m": 30.0,
    "lateral_amplitude_m": 1.5,
    "lateral_cycles": 1.0,
    "depth_start_m": 10.0,
    "depth_slope_m": 2.0,
    "depth_amplitude_m": 0.4,
    "depth_cycles": 2.0,
    "sample_spacing_m": 0.1,
    "start_jitter_xy_m": 0.5,
    "start_jitter_z_m": 0.5,
    "start_jitter_yaw_rad": 0.0872664626
  },
  "current": {
    "enabled": true,
    "type": "uniform_horizontal",
    "speed_mps": [0.2, 0.4],
    "direction_rad": [-3.1415926536, 3.1415926536],
    "constant_within_episode": true
  },
  "guidance": {
    "horizontal_lookahead_m": 3.0,
    "vertical_lookahead_m": 2.0,
    "min_lookahead_m": 0.5,
    "max_lookahead_m": 20.0,
    "horizontal_gain": 1.0,
    "vertical_gain": 1.0,
    "lookahead_error_gain": 1.0,
    "validity_period_s": 0.5,
    "compensate_current": false,
    "terminal_capture_distance_m": 16.0
  },
  "observation": {
    "dim": 27,
    "expose_current": false
  },
  "base_controller": {
    "type": "remus100_tracking_pd_pi",
    "use_current_field": false,
    "nominal_step_s": 0.02,
    "heading_gain": 2.0,
    "heading_rate_gain": 0.6,
    "pitch_gain": 2.0,
    "pitch_rate_gain": 1.0,
    "speed_feedforward_rpm_per_mps": 600.0,
    "speed_gain_rpm_per_mps": 80.0,
    "speed_integral_gain_rpm_per_mps_s": 10.0,
    "max_speed_integral_rpm": 400.0
  },
  "residual": {
    "scale_rudder": 0.10,
    "scale_stern": 0.05,
    "scale_rpm": 0.05,
    "max_action": 1.0
  },
  "encoder": {
    "type": "dual_horizon_mlp",
    "short_window": 10,
    "long_window": 50,
    "feature_dim": 64,
    "fusion_hidden": [128, 64],
    "output_dim": 64
  },
  "ppo": {
    "num_envs": 64,
    "num_steps_per_env": 1024,
    "num_learning_epochs": 4,
    "num_mini_batches": 64,
    "learning_rate_actor": 0.00015,
    "learning_rate_critic": 0.00015,
    "gamma": 0.99,
    "lam": 0.95,
    "clip_param": 0.15,
    "entropy_coef": 0.0005,
    "reward_scale": 0.01,
    "max_iterations": 5000
  },
  "reward": {
    "w_prog": 15.0,
    "w_3d": 2.0,
    "w_yaw": 0.8,
    "w_u": 0.5,
    "w_action": 0.05,
    "w_sat": 0.1,
    "goal_bonus": 50.0,
    "timeout_penalty": -20.0,
    "out_of_bounds_penalty": -50.0
  },
  "evaluation": {
    "seeds": "0-49",
    "methods": ["pd_pi", "mpc_nominal", "mpc_current", "mpc_il", "pd_pi_il", "residual_ppo", "controller_il_residual", "gar_ppo", "controller_il_dade"],
    "latency_gate_ms_p95": 20.0
  }
}
```

JSON 是参数唯一事实来源。每次迭代新建版本号，不覆盖历史配置。

## 12. Git 工作流

### 12.1 每次提交前

```bash
cd "/home/bubble/水下机器人"
git status --short
git diff --check
git diff --stat
```

当前工作树有大量前序未提交修改。禁止：

```text
git add -A
git add .
git commit -a
git reset --hard
git checkout -- <file>
git clean -fd
```

只添加本阶段明确审阅过的文件。

### 12.2 提交粒度

| 阶段 | 建议 commit |
|---|---|
| G0 | `docs: add odarrl control reproduction plan` |
| G1 | `feat(lowlevel): add remus100 control baseline contract` |
| G2 | `feat(control): add remus100 mpc baseline and expert` |
| G3A | `feat(bc): add mpc expert collection and imitation baseline` |
| G3B | `feat(bc): add pd-pi expert collection and imitation baseline` |
| G4 | `feat(rl): add residual ppo for online and imitation bases` |
| G5 | `feat(rl): add actuator-aware dual-horizon disturbance encoder` |
| G6 | `docs(lowlevel): record odarrl control reproduction results` |
| G7 | `docs(lowlevel): conclude odarrl control reproduction` |

### 12.3 推送

```bash
git push origin main
```

只有用户明确要求提交时才执行 `git add`、`git commit`、`git push`。本文件创建本身不自动提交。

### 12.4 不进普通 Git 的内容

- `checkpoint_latest.pt`、`checkpoint_best.pt`、optimizer state；
- 完整 rollout、TensorBoard event、视频；
- 大型 demo `.npz`、完整观测历史；
- 大型 CSV 和 PVS 轨迹数组；
- 任何包含绝对服务器路径且无法复现的临时文件。

只提交：

- JSON 配置、展开配置和 `config_sha256`；
- 小型 summary、manifest、SHA-256；
- 报告、图表、固定种子的摘要；
- 必要的测试和小型 fixture。

## 13. 风险与停止条件

### 13.1 主要风险

| 风险 | 影响 | 缓解 |
|---|---|---|
| 论文无代码 | MPC、观测、动作细节需猜测 | 所有假设写入配置和报告 |
| 把 MPC-IL 当在线 MPC | 结论虚假 | 必须跑 B1/B2 在线 MPC |
| CPU PVS 与 GPU batch 后端不一致 | 训练快但正式评估可能出现分布差异 | 训练用 `BatchTorchVecEnv` 4096，正式结论只用 CPU PVS 50-seed；修改物理/路径后跑跨后端回归 |
| REMUS 最小速度 1.0 | 不能照搬 0.85m/s | 固定 1.2m/s |
| 三维误差被水平误差掩盖 | 策略只优化水平面 | 奖励用三维欧氏误差，水平和垂向 RMSE 分项必报 |
| 残差太小 | 海流补偿不足 | 先按论文 0.1/0.05，G4 做缩放消融 |
| 残差太大 | 破坏基础控制器 | 限幅、速率限制、冻结基础策略 |
| Transformer 延迟 | 50Hz 不达标 | 先轻量 MLP/GRU，再完整 DADE |
| 动作抖动 | 执行器磨损、评估不真实 | 加 `Δa` 惩罚和饱和统计 |
| PVS 质量随机化 | 模型修改风险 | 第一版只随机海流/初始状态/延迟，质量扰动后置 |
| 有状态教师 BC | 27D I0 依赖速度积分；26D 删除积分后，教师动作不再是当前观测的严格函数，残差可能被迫补偿不可观测状态 | 26D I0 先用 27D teacher 做 DAgger/动作蒸馏，或重新采 26D 闭环数据；50-seed I0 门禁；无海流残差幅值审计；不把第 27 维塞回主 schema |
| 冻结后隐藏依赖教师 | 评估时暗中运行 PD/PI，I 臂不再是独立策略 | `ShadowSpeedIntegrator` 只读观测/参考；禁止在线教师注入；实现指纹审计 |
| 多实验争用资源 | CPU 超卖、显存不足、worker 超时或吞吐抖动 | preflight；每任务独立 GPU/CPU 范围/输出目录；扩并发前先 smoke |

### 13.2 停止条件

- B0 基线不能在固定种子下复现；
- 观测/动作/奖励维度与 JSON 契约不一致；
- PVS 时间戳倒退或执行器限制被静默绕过；
- `BatchTorchVecEnv` smoke 或 PVS 评估 smoke 出现 NaN/Inf、worker 泄漏或 reset 串扰；
- I1 训练完全不收敛，且确认不是配置错误；
- O1/O2 不收敛只停止 O 消融臂，不自动停止 I 主链；但如果并行资源因此超卖，必须降并发；
- 26D I0 即使经过 DAgger/蒸馏，仍不能复现 27D I0 或在线 PD/PI 的基本跟踪；
- I1 在无海流环境的残差幅值超过 G1 冻结阈值，说明它主要在补偿 BC 误差；
- 残差 PPO 只靠增大动作幅度获得收益，饱和率显著上升；
- 完整 DADE p95 超过 20ms 且轻量版也无法满足；
- 评估混用训练种子或删除失败 episode；
- 并行运行共享输出目录、CPU 核范围或 checkpoint，导致结果无法归因；
- 只有水平误差改善，却宣称三维控制更好；
- 需要修改 PVS/IsaacLab/rsl_rl 源码才能继续，但没有补丁审查和回滚方案。

## 14. 当前下一步

当前优先级已经冻结，不再等待方案确认。执行顺序如下。

### 14.0 2026-10-31 路线重排（Goal10 之后）

这一节覆盖后续执行优先级；14.1–14.4 的历史记录保留，用于追溯 27D 结果。新的默认动作不是“继续 r4”，而是先完成 26D I 臂的迁移和门禁。

- [x] 把 r1–r4、旧 I0 和旧数据标记为 27D v1 历史产物；默认不续训 r4，只有补全 27D 对照或作为迁移 teacher 时才使用；
- [x] 生成 26D I0：默认用同一在线 PD/PI teacher 在 26D 观测下做 BC/DAgger，可用现有 27D I0 做删列热启动；若直接 26D DAgger 不稳定，再固定 27D I0 teacher 做闭环动作蒸馏；
- [x] 26D I0 必须在 CPU PVS 上用同一 50 个 seed 做 teacher-student 和 B0 对照，成功率、三维 RMSE、material saturation 全部过门禁后，才进入 I1-26D；
- [x] I1-26D r1 使用冻结 26D `pi0` + residual PPO、无 encoder，训练后端用 `BatchTorchVecEnv` 4096×256；完成无海流审计和有海流 50-seed，判定为负结果，见 14.4.4；
- [x] I1-26D r2 先用监督残差热启动，再做 40 iteration 保守 PPO 微调；已完成无海流审计、有海流 50-seed 和严格 1.25 m 复查。跟踪正收益成立，但执行器门禁未通过，见 14.4.5；
- [x] I1-26D r3 完成低幅/通道探针：全三通道降低 scale 仍无法消除舵面饱和；RPM-only 在两组 seeds、2.0 m/1.25 m 上通过“跟踪正收益 + 执行器近中性”门禁，canonical 配置已固定，见 14.4.6；
- [x] I2-26D 显式历史栈基线实现完成：短窗 10、长窗 50、策略输入 `1586D`，训练与评估均从 checkpoint 元数据恢复布局；完成 4×32 smoke 和闭环评估往返，见 14.4.7；
- [x] I2-26D 历史栈基线正式 4096 环境训练与 50-seed base-vs-history 评估；两个 checkpoint 在两组 seeds、2.0/1.25 m 上均相对 I1 有增量收益，允许进入真正的 DADE；
- [x] I2-26D 从 I1 `model_10.pt` + RPM-only canonical 配置热启动；没有把 BC/DAgger 迁移收益或 I1 的 RPM 残差收益算成 DADE 收益；
- [x] 不把 Goal10 P4 的 `model_25` 当作 I 臂基础策略；它的 base 是 online PD/PI，属于 O 臂 residual-only 预检，且是负结果；
- [x] O1/O2 暂缓：已独立做 conservative/explore/terminal-reward/frequent-update 四组 26D O-residual 复查；四组均未通过正收益或海流条件化门禁，O1/O2 encoder 阶段继续暂缓，见 14.4.10；
- [x] 观测契约泄漏修复：新增 `observation.speed_feedback` 与 `observation.current_contract`；hidden 契约禁止 water-relative surge/speed-error，旧配置保持 hash 不变，见 14.4.9；
- [x] hidden-current I1/I2-a/I2-b 完整 50-seed 重跑与对比；I1 无稳健正收益，I2-a 小幅四格同向；第一轮 hidden DADE 在留出集失败，seed `62000` 复跑则表现为“总 RMSE 强正、共同成功子集和默认成功率退化”；seed-matched 的 legacy `72000` 复跑进一步证明 DADE 跨 seed 不稳定，旧 water-relative 结果已重标注；
- [x] seed-matched DADE 2×2：`hidden_seed62000` 与 `legacy_seed72000` 均完成 `4096×256×40` 训练及四格 50-seed 评估；结果显示 legacy 本身存在严重跨 seed 翻转（`+0.089→-0.441 m`），hidden 也不是稳定正收益。结论改为“seed 不稳定优先于契约效应”，见 14.4.9；I1/I2-a 的 seed-matched 复跑仍待补；
- [x] 独立特权 critic 通路：`observations["critic"] = hidden 26D + true current NED 3D`，actor 宽度不变；旧 checkpoint 走共享特征兼容路径，新配置走 `independent_critic`；8 环境接口 smoke、1-iteration 全训练脚本 smoke 和 checkpoint/eval 往返通过；
- [x] 独立特权 critic 的 4096 环境正式训练已完成；有海流默认 `0–49` 成功率 `0.84→0.70`，为负结果，不得写成性能收益；
- [x] 时变海流实验：episode 内平滑变化，actor 保持 hidden 契约；I2-a 四格通过，DADE 的 RMSE/留出集/无海流翻转通过，但严格 `0–49` 成功率下降 `0.04`；
- [x] 快速时变海流零样本留出：DADE 四格总 RMSE `+0.389/+0.787/+0.393/+1.003 m`，成功率全不降；但共同成功子集约 `-0.009~-0.021 m`，优势来自失败尾部而非成功轨迹精度。显式历史栈留出总 RMSE 两格为负、共同成功子集为正。静态 DADE 迁移失败，说明时变训练分布必要；
- [x] DADE 推理期 history ablation：`current-only` 在默认和严格半径下与完整历史几乎重合，`zero` 才明显退化；因此性能收益来自当前观测映射，当前证据不支持 encoder 已学会从历史推断海流，见 14.4.9；
- [x] 评估汇总新增 `paired_analysis`：自动输出共同成功/共同失败子集与成功翻转，防止失败轨迹主导总 RMSE；新增单元测试并通过；
- [ ] DADE 严格成功率修复：scale `0.8/0.9` 与终端包络已证伪；terminal capture 奖励把 `0–49` 缺口从 `0.04` 缩小到 `0.02`，但仍未通过，下一步转向严格半径课程或终端控制权切换；
- [x] 终端控制权 handoff：`8 / 3 m` 在严格 `0–49`/`50–99` 上成功率 `0.64→0.64`/`0.60→0.60`，RMSE 改善 `0.655/0.662 m`；这是部署侧切换，不是 DADE 学会终端捕获，且 `0–49` 有 seed `1/49` 行为互换；
- [x] 空间剪切流实验：保持 hidden 契约，真实海流只进 reward/诊断/教师/辅助目标；零样本四格以失败尾部收益为主，默认 `0–49` 仍 `0.82→0.80`；在剪切场继续训练 `4096×256×40` 后总 RMSE 仅再提升约 `0.03–0.04 m`，翻转 seed 不变，见 14.4.9；
- [ ] DADE 历史依赖修复（进行中）：`current-only` 在均匀时变、空间剪切零样本和剪切训练后都几乎重合；已实现环境层 current-frame dropout（同时隐藏当前块与历史末行，避免 PPO mask 不一致），正在以同步 seed 训练 dropout=0.5 对照，并用新增 `current-zero` 做历史-only 门禁；尚未得到支持历史依赖的结论，见 14.4.9；
- [x] 新配置必须让 `experiment.arm` 与 `base_policy.mode` 一致，并在 manifest 中记录 `observation_schema=v2`、`observation_dim=26`、训练后端和 checkpoint 选择依据；
- [x] I 臂海流契约启动门禁：开启海流时必须是显式 `hidden`，`legacy` 只能通过 `--allow-legacy-current-contract` 做历史复现，manifest 记录校验模式。

### 14.0.1 下一执行提示词：26D I0

这一步的目标不是训练 I1，而是把现有 27D I0 的能力迁移到一个独立、可部署、只接收 26D 观测的 `pi0_26d`。完成并通过门禁后，才允许启动 I1-26D。

输入：

- 历史 27D I0 checkpoint：服务器 `artifacts/lowlevel/i_bc_model_dagger2/pi0.pt`；
- 历史 DAgger 数据和 manifest：`artifacts/lowlevel/i_bc_dagger2/combined.npz`、`combined.manifest.json`；
- 26D 环境 canonical 配置：`configs/lowlevel/odarrl_control_v2_26d.json`；
- 教师仍是原来的在线 PD/PI，不使用 P4 的 O 臂 residual checkpoint。

执行要求：

- [x] 本次直接走在线 PD/PI teacher 的 26D 路线，不依赖旧 27D checkpoint；旧 I0 仅保留为 v1 历史，未用删列热启动或隐藏蒸馏；
- [x] 同一在线 PD/PI teacher 在 26D 观测下采集 180 episodes / 255,467 样本，并完成 BC/DAgger；
- [x] 直接 26D DAgger 在闭环中已稳定，不需要固定旧 27D I0 做动作蒸馏；条件分支未触发，没有混入第二种 teacher；
- [x] 最小化代码改动：扩展 BC dataset/checkpoint、采集器、训练/评估脚本和 `BaseControllerAdapter` 的 schema 校验；不修改 PVS、IsaacLab 或 rsl_rl 源码；
- [x] 26D student checkpoint 记录 `observation_schema=v2`、`observation_dim=26`、`training_config_sha256`、数据 hash、初始化方式、git 状态和 teacher 输入 hash；在线 PD/PI 没有 checkpoint，因此教师侧记录数据/config lineage，`teacher_checkpoint_sha256` 标注为不适用；
- [x] 评估脚本、适配器和 checkpoint 均硬校验 26D；80 个正式 rollout 的观测宽度为 26，所有 float 数组无 NaN/Inf；
- [x] 输出放在独立目录，保存数据/训练 manifest、50-seed summary、1.25 m strict summary、audit 和复现命令；完整 npz/checkpoint 只保留在服务器 artifacts，不提交普通 Git。

通过门禁：

- [x] 26D I0 与 27D I0、B0 使用同一 `0–49` 50 个评估 seed；
- [x] 成功率下降 `0.0 <= 0.02`、三维 RMSE 劣化 `0.03751 m <= 0.05 m`；student material saturation `0.00572`，低于旧 27D I0 的 `0.0073` 和 `0.05` 门限；
- [x] 同时报告 B0 对照和严格捕获率：相对 B0 三维 RMSE 改善 `0.12218 m`，1.25 m 捕获率 teacher/student `0.92/0.94`；
- [x] 无 NaN/Inf、无隐藏 27D 输入、无 manifest schema 不一致；`scripts/verify_odarrl_i0.py` 审计通过；
- [x] 将 `I0-26D` 固定为 I1-26D 的基础策略；后续不得在 I1 中继续修改或重训 `pi0`。

资源允许时可以并行做 O 臂的 reward/超参/early-stop 预检，但 O1/O2 的 encoder 阶段仍然暂缓；GPU 不足时 26D I0 优先。

### 14.0.2 26D I0 正式结果（2026-10-31）

正式 checkpoint：服务器 `artifacts/lowlevel/i0_26d_bc_dagger_r1_lineage_20261031/pi0.pt`。权重与首次 DAgger 训练完全一致，state-dict hash 为 `e3632d63b5810f1711de3d151efc6ebfc94089ca4b59b4035da7c9050ee53ea6`；lineage 版额外写入训练配置 hash 和来源信息。

数据与训练：

- `configs/lowlevel/odarrl_control_i_bc_26d.json`，config SHA-256 `2a5a45103f744ea60ead044925b19a4b4a5b1574c471475e00ecbfe4e6799544`；
- 教师数据 180 episodes / 255,467 样本，`observation_schema=v2`，成功率 1.00，约束违反 0，shadow-teacher 积分差 0；
- 单次 BC：三维 RMSE 差 `22.28460 m`、成功率 `0.38`，保留为负结果；
- DAgger r1：40 episodes / 80,239 样本，合并后 220 episodes / 335,706 样本；训练 checkpoint 数据 hash `33ee9f7a52e6a199f978c162885158af8f3a65a6513df8102285c5beada2f5ba`，`training_config_sha256=05e70d17bbcea67124ef232907adf4db249d9b7a5d48bd7e213ec4fb88d47a6e`；
- 初始化：`scratch`；服务器目录不是 git checkout，因此 checkpoint 中 `git_commit=null`，但不影响 artifacts/hash 追溯。

50-seed 主门禁（`artifacts/lowlevel/i0_26d_gate_dagger_r1_lineage_50seed/summary.json`）：

| 指标 | teacher | student | delta / 门限 |
|---|---:|---:|---:|
| 成功率 | 1.00 | 1.00 | 下降 0.0 ≤ 0.02 |
| 三维 RMSE | 1.35505 m | 1.39257 m | +0.03751 m ≤ 0.05 m |
| 水平 RMSE | 1.31876 m | 1.35792 m | +0.03916 m，报告 |
| 垂向 RMSE | 0.30400 m | 0.30117 m | -0.00283 m，报告 |
| material saturation | 0.0 | 0.00572 | ≤ 0.05，且低于旧 27D I0 `0.0073` |
| 约束违反 | 0.0/episode | 14.72/episode | 报告；均为亚材料级微裁剪 |
| 控制环 p95 的 p95 | 19.97 ms | 19.98 ms | 报告 |

B0 对照：B0 三维 RMSE `1.51475 m`、成功率 1.00；student 相对 B0 改善 `0.12218 m`，成功率相同。

严格捕获（`artifacts/lowlevel/i0_26d_gate_dagger_r1_lineage_50seed_strict125/summary.json`）：`goal_radius=1.25 m` 时 teacher 成功率 `0.92`、student `0.94`；student 没有比 teacher 更差。

完整性审计：`artifacts/lowlevel/i0_26d_gate_dagger_r1_lineage_50seed/audit.json`。50 个 teacher + 50 个 student rollout 共 2,000 个数组全部有限；teacher/student 观测宽度均为 26；schema、config hash、训练配置 hash、数据 hash、B0 对照和 gate 全部一致，审计 `passed=true`。

复现命令（服务器，CPU PVS 评估）：

```bash
cd /data/yhx/underwater-robotics/src/plan_p1/settled_局部规划器_反应式
export PYTHONPATH="$PWD/src:$PWD/../settled_全局规划器/src"

python scripts/collect_odarrl_control_demos.py \
  --config configs/lowlevel/odarrl_control_i_bc_26d.json \
  --output artifacts/lowlevel/i0_26d_demos_20261031 --episodes 180 --workers 16

python scripts/collect_odarrl_control_dagger.py \
  --config configs/lowlevel/odarrl_control_i_bc_26d.json \
  --checkpoint artifacts/lowlevel/i0_26d_bc_20261031/pi0.pt \
  --output artifacts/lowlevel/i0_26d_dagger_r1_20261031 \
  --episodes 40 --workers 16 --seed-start 11000 --episode-id-offset 1000 --noise-std 0.05

python scripts/merge_odarrl_control_demos.py \
  --inputs artifacts/lowlevel/i0_26d_demos_20261031/demos.npz \
           artifacts/lowlevel/i0_26d_dagger_r1_20261031/dagger.npz \
  --output artifacts/lowlevel/i0_26d_dagger_r1_20261031/combined.npz

python scripts/train_odarrl_control_bc.py \
  --demos artifacts/lowlevel/i0_26d_dagger_r1_20261031/combined.npz \
  --demos-manifest artifacts/lowlevel/i0_26d_dagger_r1_20261031/combined.manifest.json \
  --output artifacts/lowlevel/i0_26d_bc_dagger_r1_lineage_20261031 \
  --expected-observation-dim 26 --device cuda

taskset -c 160-191 python scripts/evaluate_odarrl_control_bc.py \
  --config configs/lowlevel/odarrl_control_i_bc_26d.json \
  --checkpoint artifacts/lowlevel/i0_26d_bc_dagger_r1_lineage_20261031/pi0.pt \
  --output artifacts/lowlevel/i0_26d_gate_dagger_r1_lineage_50seed \
  --seeds 0-49 --workers 16 \
  --b0-summary artifacts/lowlevel/server_b0_50seed_r2/summary.json

taskset -c 160-191 python scripts/evaluate_odarrl_control_bc.py \
  --config configs/lowlevel/odarrl_control_i_bc_26d.json \
  --checkpoint artifacts/lowlevel/i0_26d_bc_dagger_r1_lineage_20261031/pi0.pt \
  --output artifacts/lowlevel/i0_26d_gate_dagger_r1_lineage_50seed_strict125 \
  --seeds 0-49 --workers 16 --goal-radius 1.25 \
  --b0-summary artifacts/lowlevel/server_b0_50seed_r2/summary.json

python scripts/verify_odarrl_i0.py \
  --config configs/lowlevel/odarrl_control_i_bc_26d.json \
  --checkpoint artifacts/lowlevel/i0_26d_bc_dagger_r1_lineage_20261031/pi0.pt \
  --gate-summary artifacts/lowlevel/i0_26d_gate_dagger_r1_lineage_50seed/summary.json \
  --strict-summary artifacts/lowlevel/i0_26d_gate_dagger_r1_lineage_50seed_strict125/summary.json \
  --output artifacts/lowlevel/i0_26d_gate_dagger_r1_lineage_50seed/audit.json
```

### 14.0.3 I1-26D 下一执行提示词

下一步只做 I1 的第二次迭代，不启动 I2；首轮 `i1_residual_ppo_26d_r1` 已判负，保留证据：

- base：冻结 `artifacts/lowlevel/i0_26d_bc_dagger_r1_lineage_20261031/pi0.pt`，禁止梯度回传和参数更新；
- 观测：26D v2，海流索引 17–19 在无海流/隐藏海流条件下保持一致语义，不能偷读第 27 维积分；
- 监督热启动：用 B0 current-aware 教师的 50-seed rollouts 生成 26D 标签，输入仍屏蔽 `17:20` 海流与 `23:26` 残差历史；标签是相对冻结 I0 的归一化残差，输出层先缩放到 `0.15`；
- 热启动门禁：先做无海流 8×500 和 50-seed，再和同 seed I0/B0 对比；成功率、三维 RMSE、any/material saturation、约束违反和残差幅值任一明显恶化时拒绝进入 PPO；
- 训练：仅在热启动通过后使用 `BatchTorchVecEnv` 4096×256、无 encoder、保守 residual PPO 微调；必须显式加载并校验 warm-start checkpoint，不能悄悄退回零初始化；
- 先做无海流 smoke 和残差幅值审计。若残差主要补偿 BC 误差，停止并修 I0，不能把它当控制收益；
- 再做有海流 50-seed base-vs-residual，和 I0/B0 同 seed；必须同时报告成功率、三维 RMSE、material saturation、残差幅值/饱和和延迟；
- I1 相对 I0/B0 没有可重复正收益时记录负结果，不启动 I2-DADE。

### 14.1 立即可做：本地收尾

- [x] `ruff check` 通过；
- [x] G1 core tests 与 BC tests 通过；
- [x] 配置级 hidden-current leak 校验通过；
- [x] 合并跑一次 `test_lowlevel_*`、`test_m2_pvs_closed_loop.py`、`test_import_guard.py`；
- [x] 终端捕获改动后重跑合并测试：29 passed；`mypy src/auv_lowlevel` 无错误。
- [x] `git diff --check` 通过；`git status --short` 已人工审阅，未清理既有工作树。

### 14.2 服务器 B0 正式 50-seed 基线

- [x] 记录 `odarrl_control_server_resources.json`：hostname、日期、GPU/CPU/内存/磁盘、Python/torch/CUDA、PVS 路径、仓库 commit。
- [x] 只同步新增文件：`src/auv_lowlevel`、`configs/lowlevel`、四个脚本和测试；不覆盖服务器已有工作成果。
- [x] 用 `odarrl_control_base.json` 跑 seed `0-49`，固定 CPU 核范围，输出到独立目录。
- [x] 保存每 seed 的小型 summary、轨迹/误差图和 50-seed 聚合指标；不得只保留成功样本。
- [x] 记录 `config_sha256`、仓库来源 commit、服务器资源绑定；服务器不是 git checkout，`git_commit=null` 已记录。

### 14.3 服务器 I 臂数据与 BC

- [x] 用 `odarrl_control_i_bc.json` 采集 180 episodes、255,467 样本，16 workers，独立 CPU 范围和输出目录。
- [x] 检查每个 episode 的 `max_shadow_teacher_margin_rpm=0`、约束违反为 0、成功率 1.00。
- [x] 单次 BC 用默认 MLP `256,128`、batch 4096、200 epochs 训练；验证 MSE 很小，但 50-seed 闭环失败（RMSE 差 `26.08 m`），保留负结果。
- [x] DAgger 第 1 轮：40 episodes / 98,938 样本，RMSE 差降到 `0.119 m`；第 2 轮：60 episodes / 86,109 样本，合并后 440,514 样本训练，RMSE 差 `0.0203 m`。
- [x] 用 `evaluate_odarrl_control_bc.py` 跑 I0 的 50-seed 门禁：三维 RMSE 差距 `0.0203 m` ≤ `0.05`、成功率下降 `0.0` ≤ `0.10`、material saturation `0.0073` ≤ `0.05`，通过。
- [x] 门禁通过后再进入 I1；不能把 DAgger 的收益写成残差 PPO 或 DADE 的收益。

I0 最终门禁（50 seeds，`artifacts/lowlevel/i0_gate_dagger2_50seed`）：

| 指标 | teacher | student | 门禁 |
|---|---:|---:|---:|
| 成功率 | 1.00 | 1.00 | 下降 ≤ 0.10 |
| 三维 RMSE | 1.355 m | 1.375 m | 差 ≤ 0.05 m |
| 水平 RMSE | 1.319 m | 1.340 m | 报告 |
| 垂向 RMSE | 0.304 m | 0.304 m | 报告 |
| material saturation | 0 | 0.0073 | ≤ 0.05 |
| 约束违反 | 0 | 18.54/episode | 报告 |

at-limit 比例：teacher 舵角 53.6%、艉舵 24.0%；student 舵角 11.4%、艉舵 16.3%。学生没有比教师更依赖饱和。

### 14.4 服务器 I1/I2

注意：以下 r1–r4 和 I1/I2 记录是 27D v1 历史。Goal10 之后的新 I1/I2 必须先按 14.0 得到 26D I0；不得把旧 27D I1 直接续训成新主链，也不得把 P4 的 online-base residual 当作 I1。

- [x] 实现 `BaseControllerAdapter`、PVS 多进程 worker pool 和 rsl_rl `VecEnv`，不改 rsl_rl 源码。
- [x] 先跑 8 env × 32 步 smoke，再扩到 64 env；检查吞吐、时间戳、worker 回收和 reset 串扰。
- [x] 64 env 吞吐探针得到 `1,659.9 steps/s`，低于 `3,200 steps/s` 门禁；原因是共享服务器 CPU load 约 240，不是 PVS 单步异常，不能写成吞吐达标。
- [x] I1：冻结 `pi0`，先训练无 DADE 的残差 PPO；在无海流环境记录残差幅值，排除“残差补偿 BC 误差”。r1/r2 为中间负结果，r3 的 RPM-only 配置通过，见 14.4.6。
- [x] I1 训练完成后，用默认 2.0 m 半径跑 50-seed base-vs-residual 评估，并用 `--goal-radius 1.25` 重跑严格捕获；r3 还补了独立 `50–99` holdout。
- [x] 用无海流环境配置跑同一 I1 checkpoint 的 50-seed 审计；r3 在 `0–49` 和 `50–99` 上成功率均为 `1.00`，残差幅值远低于有海流。
- [x] I2：在 I1 基础上加入 DADE；短窗 10 步、长窗先按 50 步，另做 10/250 步或 10Hz 抽样消融。50Hz DADE 与 10Hz 抽样均未超过 I2-a 显式历史栈；10Hz 在四个有海流条件上还低于 50Hz DADE，见 14.4.8。
- [ ] 所有对比共用 50 个评估种子、同一 PVS 配置、同一参考路径和同一训练预算。

#### 14.4.1 I1 首轮作废与冻结 BC 输入契约修复

首轮 `i1_residual_ppo_r1` 在第 42 轮被发现存在闭环发散，产物已整体保留为 `artifacts/lowlevel/i1_residual_ppo_r1_invalid_base_obs`，不能用于任何 I1 结论。问题不是 PPO 的输出幅度过大：残差归一化均值只有约 `0.005`，但冻结的 `pi0` 在闭环中把 RPM 从约 700 推到 1525，三维 RMSE 从 `0.78 m` 恶化到 `3.4–4.2 m`。

根因是观测契约：

- 27D 观测的索引 `23:26` 是上一时刻残差；
- 教师与 DAgger 采集时这三维持恒为 0，BC 训练得到 `std=1e-5` 的零方差下限；
- I1 输出任意非零残差后，`BCPolicy` 把 `0.004` 归一化成约 `400σ`，使从未见过该输入的 `pi0` 进入分布外并输出饱和命令；
- 该错误会通过 `previous_actuator` 和 `previous_residual` 反馈，形成复合发散。

修复只改策略输入契约，不改 PVS/IsaacLab/rsl_rl：

- `BCPolicy` 自动检测索引 `23:26` 的 normalize std 是否小于等于 `1e-4`；
- 对常量残差历史维度，在归一化后固定为训练均值对应的 0；PPO 的 ActorCritic 仍接收完整 27D 观测；
- 训练 manifest 记录 `base_policy.mask_previous_residual`，r2/r3 均为 `true`；
- 新增回归测试，确认修改残差历史三个输入不会改变冻结 BC 的输出。

诊断证据（8 seeds × 500 steps，仅作故障定位，不是正式评估）：

| checkpoint | 残差历史屏蔽 | residual 三维 RMSE | base 三维 RMSE | material saturation | constraint violations |
|---|---:|---:|---:|---:|---:|
| `model_30`（作废） | 否 | `3.422 m` | `0.780 m` | `0.0675` | `1238.1` |
| `model_30`（作废） | 是 | `0.780 m` | `0.780 m` | `0.00525` | `18.0` |

修复后 8×32 训练 smoke 通过，`initial_residual_abs_max=0`、worker 重启为 0。r2 从零初始化重新训练，不复用作废 r1 的 optimizer 或 actor 权重。

#### 14.4.2 r2 探索不足与 r3 参数调整

r2 在修复后的稳定基座上训练到约第 46 轮，不再发散，但确定性残差始终接近常值小偏置：

- 8 seeds × 500 steps、有海流：残差三维 RMSE `0.780438 m`，base `0.780439 m`，残差均值 `0.00482`；
- 8 seeds × 2500 steps、有海流：residual 比 base 仅改善 `0.00184 m`，成功率同为 `0.75`；
- 8 seeds × 500 steps、无海流：残差均值 `0.00455`，与有海流的 `0.00482` 几乎相同；
- 残差动作的 `q99` 与均值接近，说明网络没有形成明显的状态/海流响应，而是学到了小常值偏置。

r2 产物保留为 `artifacts/lowlevel/i1_residual_ppo_r2_underpowered`，manifest 标记为 `stopped_underpowered_exploration`，不作为 I1 结论。参数问题的判断是：

- 原残差满幅只有舵 `1.5°`、艉舵 `0.75°`、RPM `76`，对 `0.2–0.4 m/s` 海流补偿权限偏小；
- 原 `init_noise_std=0.05` 对应的舵角探索只有约 `0.08°`，PPO 很难从单步奖励中分辨动作效果；
- `1.5e-4` 学习率下，40 多轮后确定性动作仍几乎是零附近的常值。

r3 只调整这些与探索/权限直接相关的参数，环境、奖励、观测、BC checkpoint、网络结构和评估种子不变：

| 参数 | r2 | r3 |
|---|---:|---:|
| `residual.scale_rudder` | `0.1` | `0.3` |
| `residual.scale_stern` | `0.05` | `0.2` |
| `residual.scale_rpm` | `0.05` | `0.15` |
| `init_noise_std` | `0.05` | `0.15` |
| `learning_rate` | `1.5e-4` | `3e-4` |
| `entropy_coef` | `5e-4` | `1e-3` |

r3 配置：`configs/lowlevel/odarrl_control_i_residual_r3.json` 与 `configs/lowlevel/odarrl_control_ppo_i1_r3.json`；输出目录 `artifacts/lowlevel/i1_residual_ppo_r3`。r3 的 8×32 smoke 同样满足初始残差 0、无 worker 重启。

为避免只看训练曲线调参，另做了教师-state 残差需求分析：取 B0 current-aware 教师的 8 条轨迹，把观测中的海流和残差历史置零后喂给冻结 `pi0`，统计 `a_teacher - a_BC`。这不是严格标签，但能给出动作权限量级：

| 指标 | 舵角 | 艉舵 | RPM |
|---|---:|---:|---:|
| 原始差值平均绝对值 | `1.94°` | `2.49°` | `113 rpm` |
| 原始差值 p95 绝对值 | `5.18°` | `6.54°` | `242 rpm` |
| 按 r2 满幅归一化后的平均需求 | `1.29` | `3.32` | `1.48` |
| 按 r3 满幅归一化后的平均需求 | `0.43` | `0.83` | `0.49` |

结论：r2 的动作范围会长期触顶，不能测试残差 PPO 的真实能力；r3 的平均需求落在满幅内，但艉舵 p95 仍约为 `2.18`，可能触顶。r3 第 30 轮后要用 `residual_clip_rate` 和轨迹检查决定是否扩大艉舵/舵角范围，不能只看 loss。

#### 14.4.3 r3 止损、监督残差热启动与 r4

r3 训练到第 50 轮后主动停止，manifest 标记为 `stopped_no_action_learning`。第 50 轮与第 30 轮没有形成有效动作映射，继续烧到 100 轮不会回答新问题。8 seeds × 500 steps 的诊断结果：

| 环境 | base 三维 RMSE | residual 三维 RMSE | 残差平均绝对值 | 结论 |
|---|---:|---:|---:|---|
| 有海流 | `0.780439 m` | `0.782514 m` | `0.01276` | 反而变差 |
| 无海流 | `0.518033 m` | `0.517958 m` | `0.01498` | 基本为零收益 |

因此 r3 不进入 50-seed，不作为 I1 结论。r3 的 actor 仍接近常值小偏置，说明仅放大 r2 的动作权限和探索噪声不足以从零学出残差映射。

为验证“先监督、再 PPO”的路线，新增可复现脚本 `scripts/pretrain_odarrl_control_residual.py`。标签来自两类数据：

- B0 有海流教师轨迹的 50 个 seed，共约 65k 状态；
- 两轮无海流 DAgger 学生轨迹，按 `anchor_fraction=0.25` 抽样，防止大量零残差标签淹没海流修正；
- 输入时把 `17:20` 的真实海流和 `23:25` 的残差历史置零，与部署观测契约一致；
- 标签是归一化后的 `(pi0 + teacher_mix * (teacher - pi0) - pi0) / (residual_scale * ACTION_SCALES)`，默认 `teacher_mix=1.0`；
- 用与 rsl_rl 完全一致的 `ActorCritic.actor`（ELU、256×128）训练，避免用 ReLU 原型权重热启动 ELU PPO。

监督数据得到 73,568 个训练样本、20,171 个验证样本，best validation loss `0.01294`。未缩放 actor 在闭环中输出偏大，因此对最后一层做 `output_scale=0.15`；这等价于部署侧 `residual_scale=0.15`，但把缩放写进 checkpoint，r4 PPO 仍保留 r3 的环境动作权限，可以继续学习增大修正量。24-seed 全长度对比如下：

| 方法 | 有海流成功率 | 有海流三维 RMSE | material saturation | 无海流成功率 | 无海流三维 RMSE |
|---|---:|---:|---:|---:|---:|
| `pi0` base | `0.8333` | `2.40466 m` | `0.00466` | `1.00` | `1.31221 m` |
| 监督残差，scale 0.15 | `0.8750` | `2.30540 m` | `0.11056` | `1.00` | `1.31099 m` |
| 监督残差，scale 0.25 | `0.8750` | `2.28318 m` | `0.15199` | `1.00` | `1.31031 m` |

结论：监督初始化的方向是有效的，24 seeds 上成功率和 RMSE 都改善，且无海流没有明显破坏；但 saturations/constraint violations 明显上升，不能直接当最终控制器。选择更保守的 `output_scale=0.15` 进入 PPO，而不是 0.25/1.0。

r4 配置：`configs/lowlevel/odarrl_control_i_residual_r4.json`、`configs/lowlevel/odarrl_control_ppo_i1_r4.json`；输出 `artifacts/lowlevel/i1_residual_ppo_r4_warmstart`。相对 r3 只改 PPO 微调参数：`init_noise_std=0.03`、`learning_rate=1e-4`、`entropy_coef=1e-4`、`clip_param=0.1`、60 iterations、每 5 轮保存。r4 smoke 的 `initial_residual_abs_max=0.1924`，说明 warmstart actor 已真正加载，worker 重启为 0。

运行命令（服务器仓库根目录，GPU 3）：

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
PYTHONPATH=src:../settled_全局规划器/src CUDA_VISIBLE_DEVICES=3 \
taskset -c 0-255 /data/yhx/underwater-robotics/envs/venvs/isaaclab/bin/python \
scripts/train_odarrl_control_residual.py \
  --training-config configs/lowlevel/odarrl_control_ppo_i1_r4.json \
  --warmstart-actor artifacts/lowlevel/i1_warmstart_b0a_s015/actor_warmstart.pt \
  --output artifacts/lowlevel/i1_residual_ppo_r4_warmstart --overwrite
```

早期信号：r4 `model_5` 的 8×500 有海流三维 RMSE 为 `0.780439 → 0.776307 m`，residual abs mean `0.04893`，material saturation `0.0545`。随后在 `model_20` 做 24-seed 全长度复查：有海流成功率 `0.83333 → 0.875`，三维 RMSE `2.40466 → 2.30842 m`，material saturation `0.11774`；无海流成功率维持 `1.00`，三维 RMSE `1.31221 → 1.31146 m`。收益主要来自热启动方向，PPO 继续训练没有显示出足够稳定的动作改进，且饱和度仍偏高，因此在 `model_20` 主动停止并保留 checkpoint；Goal10 之后默认不再续训 r4，只作为 27D 历史结果或可选迁移 teacher。监督残差 scale 0.15 的无海流 24-seed 结果为成功率 `1.00`、三维 RMSE `1.31099 m`、material saturation `0.00663`、残差均值 `0.01084`。

#### 14.4.4 I1-26D r1 负结果与 I2 阻断（2026-10-31）

26D I1 首轮配置为 `configs/lowlevel/odarrl_control_ppo_i1_26d.json`，环境为 `configs/lowlevel/odarrl_control_i_residual_26d.json`，基础策略冻结为 26D I0 `pi0.pt`。训练使用 `BatchTorchVecEnv`、`4096×256`、100 iterations，共 `104,857,600` env steps；实际 wall time `448.78 s`，manifest 的 `initial_residual_abs_max=0.0` 说明起点与 I0 严格一致。所有训练 checkpoint 和逐轮记录保留在 `artifacts/lowlevel/i1_residual_ppo_26d_r1`，不删除、不续训。

8 seeds × 500 steps 的有海流诊断显示“短窗看似不变、残差持续增大”：

| checkpoint | base 三维 RMSE | residual 三维 RMSE | RMSE 差 | residual abs mean | any-saturation | constraint violations |
|---|---:|---:|---:|---:|---:|---:|
| `model_20` | `0.719913` | `0.718436` | `-0.001477 m` | `0.010474` | `0.00375 → 0.08000` | `2.625 → 40.750` |
| `model_30` | `0.719913` | `0.719323` | `-0.000590 m` | `0.016867` | `0.00375 → 0.07975` | `2.625 → 40.625` |
| `model_40` | `0.719913` | `0.719764` | `-0.000149 m` | `0.021300` | `0.00375 → 0.05225` | `2.625 → 26.875` |
| `model_50` | `0.719913` | `0.721886` | `+0.001973 m` | `0.029267` | `0.00375 → 0.02600` | `2.625 → 13.750` |

50-seed 全长度有海流复查没有正收益：

| checkpoint | base/residual 成功率 | base 三维 RMSE | residual 三维 RMSE | RMSE 差 | residual abs mean | any-saturation | constraint violations |
|---|---:|---:|---:|---:|---:|---:|---:|
| `model_10` | `0.78 / 0.78` | `3.344359` | `3.347743` | `+0.003384 m` | `0.004916` | `0.00449 → 0.12505` | `10.56 → 218.72` |
| `model_20` | `0.78 / 0.78` | `3.344359` | `3.356043` | `+0.011684 m` | `0.012474` | `0.00449 → 0.10106` | `10.56 → 170.08` |
| `model_30` | `0.78 / 0.78` | `3.344359` | `3.363896` | `+0.019536 m` | `0.020307` | `0.00449 → 0.13391` | `10.56 → 222.34` |

无海流 `model_10` 审计是决定性证据：base/residual 成功率均为 `1.00`，三维 RMSE `1.392567 → 1.394596 m`（差 `+0.002029 m`）；residual abs mean 为 `0.004930`，与有海流的 `0.004916` 几乎相同；any-saturation `0.00612 → 0.20635`，constraint violations `14.72 → 304.76`。若残差真的补偿海流，隐藏海流索引后其幅值应显著变化；当前结果说明它学成了近似与海流无关的常值偏置，同时大量触发幅值/速率微裁剪。

因此本轮 I1 判定为**负结果**：不满足“有海流三维 RMSE 改善、成功率不降、material saturation 不上升”的门禁。**I2-DADE 禁止从该 checkpoint 启动**，也不得把这段训练写成残差 PPO 或 DADE 的收益。下一轮只允许在 I0 不变的前提下，先用监督残差热启动得到可解释的初始方向，再评估是否值得做保守 PPO 微调；若热启动本身在无海流下就显著增加饱和或约束违反，应直接拒绝。

资源规则补充：用户希望未来优先使用 4096 环境提高吞吐，但这适用于 **IsaacLab 原生 GPU 批量环境**；当前 `PvsVecEnv` 每个环境一个进程，4096 个 worker 会把 384 核服务器严重超卖。资源不足时保持 64–128 个 PVS worker，不为追求 4096 牺牲稳定性。
以下 r3 评估命令只保留为命令模板；r3 已作废。正式 r4 评估把 training config 换成 `odarrl_control_ppo_i1_r4.json`，checkpoint 换成 `i1_residual_ppo_r4_warmstart/final.pt`（或选中的 `model_*.pt`）。

训练后固定评估命令（服务器仓库根目录）：

```bash
# 默认 2.0 m，有海流，base 与 residual 成对评估
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
/data/yhx/underwater-robotics/envs/venvs/isaaclab/bin/python \
scripts/evaluate_odarrl_control_residual.py \
  --training-config configs/lowlevel/odarrl_control_ppo_i1_r3.json \
  --checkpoint artifacts/lowlevel/i1_residual_ppo_r3/final.pt \
  --output artifacts/lowlevel/i1_eval_current_2m_50seed \
  --workers 16

# 严格 1.25 m 捕获；1.0 m 只在 1.25 m 结果接近边界或论文对照需要时再跑
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
/data/yhx/underwater-robotics/envs/venvs/isaaclab/bin/python \
scripts/evaluate_odarrl_control_residual.py \
  --training-config configs/lowlevel/odarrl_control_ppo_i1_r3.json \
  --checkpoint artifacts/lowlevel/i1_residual_ppo_r3/final.pt \
  --output artifacts/lowlevel/i1_eval_current_strict125_50seed \
  --workers 16 --goal-radius 1.25

# 无海流残差审计；环境配置覆盖不改变 checkpoint 和策略网络
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
/data/yhx/underwater-robotics/envs/venvs/isaaclab/bin/python \
scripts/evaluate_odarrl_control_residual.py \
  --training-config configs/lowlevel/odarrl_control_ppo_i1_r3.json \
  --environment-config configs/lowlevel/odarrl_control_i_bc.json \
  --checkpoint artifacts/lowlevel/i1_residual_ppo_r3/final.pt \
  --output artifacts/lowlevel/i1_eval_nocurrent_2m_50seed \
  --workers 16
```

资源顺序：I1 训练结束后先跑默认 2.0 m 与无海流审计，再跑严格 1.25 m；全部完成后才允许启动 O 臂。当前服务器 GPU 0–7 均忙、CPU load 超过 250，O 臂不并行。

#### 14.4.5 I1-26D r2：监督热启动有跟踪收益，但执行器门禁未通过（2026-10-31）

r2 的目标不是继续从零探索，而是先给冻结 I0 一个可解释的残差方向，再判断是否值得做 PPO 微调。所有实验都保持 base 为 `artifacts/lowlevel/i0_26d_bc_dagger_r1_lineage_20261031/pi0.pt`，不更新 `pi0` 参数。

监督热启动的数据与训练：

- 教师数据：`artifacts/lowlevel/i1_residual_teacher_26d_b0_50seed_20261031/demos.npz`，50 episodes / 69,851 samples，success `1.00`，constraint violations `0`，SHA-256 `3f5cf43c0d221d718d3afc960412840de881f29c33689f25adb100b1786f1efe`；
- 标签：相对冻结 I0 的归一化残差；观测仍为 26D，海流索引 `17:20` 与残差历史 `23:26` 被屏蔽，避免教师海流和旧残差泄漏；
- 热启动：MLP `256,128`、batch `4096`、200 epochs、best epoch `197`、validation loss `0.00700308`，train/validation samples `75,509 / 18,356`；最终输出层整体缩放到 `output_scale=0.15`；
- checkpoint：`artifacts/lowlevel/i1_26d_residual_warmstart_s015_20261031/actor_warmstart.pt`，SHA-256 `9b43a42cfba3590ce3adbb2fcaf75154929773cb67d9282f30f1670bd51cabc9`。

PPO 微调只加载该 warm-start actor，训练 40 iterations、`4096×256`、共 `41,943,040` env steps，wall time `170.72 s`；manifest 记录 `initial_residual_abs_max=0.18510`，说明 warm-start 权重确实被加载，而不是退回零残差。选中的 `model_10.pt` SHA-256 为 `39678294b52a7403dae863cd6cd636d745f6e9d2106cd8d1cf184e2d4d3f43ec`。

有海流、到达半径 2.0 m、同一组 `0–49` 50 seeds：

| 策略 | 成功率 base → residual | 三维 RMSE base → residual | RMSE 改善 | residual abs mean | material saturation base → residual | any-saturation base → residual | 约束违反 base → residual |
|---|---:|---:|---:|---:|---:|---:|---:|
| 纯监督热启动 | `0.78 → 0.84` | `3.344359 → 3.072774 m` | `0.271585 m` | `0.047429` | `0.004400 → 0.022546` | `0.004488 → 0.125284` | `10.56 → 198.16` |
| PPO `model_10` | `0.78 → 0.84` | `3.344359 → 3.069486 m` | `0.274873 m` | `0.047962` | `0.004400 → 0.022956` | `0.004488 → 0.128099` | `10.56 → 203.24` |

无海流审计、同一组 50 seeds：

| 策略 | 成功率 base → residual | 三维 RMSE base → residual | RMSE 差 | residual abs mean | material saturation base → residual | any-saturation base → residual | 约束违反 base → residual |
|---|---:|---:|---:|---:|---:|---:|---:|
| 纯监督热启动 | `1.00 → 1.00` | `1.392567 → 1.391624 m` | `-0.000942 m` | `0.005963` | `0.005715 → 0.005744` | `0.006121 → 0.116456` | `14.72 → 172.30` |
| PPO `model_10` | `1.00 → 1.00` | `1.392567 → 1.391402 m` | `-0.001165 m` | `0.007367` | `0.005715 → 0.005811` | `0.006121 → 0.151710` | `14.72 → 223.58` |

严格到达半径 1.25 m、有海流、50 seeds，只对 PPO `model_10` 做了复查：

| 指标 | base | residual | 差值 |
|---|---:|---:|---:|
| 成功率 | `0.44` | `0.46` | `+0.02` |
| 三维 RMSE | `5.799747 m` | `5.716276 m` | `-0.083470 m` |
| material saturation | `0.004142` | `0.048674` | `+0.044532` |
| any-saturation | `0.004207` | `0.146961` | `+0.142754` |
| 约束违反/episode | `12.02` | `292.42` | `+280.40` |

判定：

- **跟踪收益成立**：在 50-seed、默认 2.0 m 和严格 1.25 m 下都出现了同方向的三维 RMSE 改善，且成功率不降；这不是 r1 那种常值偏置造成的偶然波动。
- **残差具备条件性**：有海流时 residual abs mean 约 `0.04743–0.04796`，无海流时约 `0.00596–0.00737`，相差约 `6.5–8.0` 倍；说明热启动确实学到了与海流相关的主方向，而不是完全固定的偏移。
- **执行器门禁未通过**：虽然默认 2.0 m 的 material saturation `0.02296` 低于文档早期的 `0.05` 参考阈值，但正式门禁要求“material saturation 不上升”。r2 相对 I0 仍上升约 `5.2` 倍，any-saturation 上升约 `28.5` 倍，约束违反上升约 `19.2` 倍；严格 1.25 m 下 material saturation 也上升约 `11.8` 倍，接近 `0.05`。因此不能把 r2 写成“通过 I1”。
- **PPO 的增量收益很小**：`model_10` 相对纯热启动只多改善约 `0.00329 m`，却略微增加 saturation 和约束违反；当前证据不能证明 PPO 微调本身有稳定增益，主要收益来自监督热启动。
- **I2 继续阻断**：不得从 `i1_residual_ppo_26d_r2_warmstart/model_10.pt` 启动 DADE，也不得把 r2 的 `0.274873 m` 写成 DADE 或残差 PPO 的独立成果。

下一步只做低幅残差探针：对 warm-start actor 使用 `--residual-scale`，把有效输出 scale 从 `0.15` 降到约 `0.10` 和 `0.12`（对应 `--residual-scale=0.6667/0.8`），先跑同 seed 的 8×500 有海流/无海流诊断，再对候选做 50-seed 复核。候选必须同时满足：保留大部分三维 RMSE 收益、成功率不降、material saturation 和约束违反接近 I0；任一不满足就保留 r2 为“跟踪正收益、执行器未过门禁”的中间结果，不启动 I2。

#### 14.4.6 I1-26D r3：RPM-only 残差通过门禁（2026-10-31）

r2 的失败不是“残差完全不能用”，而是三个执行器共享同一残差输出时，舵角和艉舵在基线已经贴近限幅的情况下继续向外请求。r3 先做诊断，再固定执行器权限：

- 在 `scripts/evaluate_odarrl_control_residual.py` 增加向后兼容的 `--residual-gains r,s,p`，只用于评估和消融；默认 `1,1,1`，旧命令和旧 summary 不受影响；
- 新增 `configs/lowlevel/odarrl_control_i_residual_26d_rpm_only.json`，把 `scale_rudder/scale_stern` 固定为 `0`，把 `scale_rpm` 固定为 `0.225`；它等价于旧配置上的 `residual_scale=1.5` 加 `residual_gains=[0,0,1]`；
- canonical 环境配置 SHA-256 为 `c000c64a92eb60b85b849dbd4db3b42869dffc7c227163a1544b0e44ee1d8188`；训练配置仍为 `configs/lowlevel/odarrl_control_ppo_i1_26d_warmstart.json`（SHA `ab273fab…dececd8`）；
- 残差 checkpoint 仍为 PPO r2 的 `artifacts/lowlevel/i1_residual_ppo_26d_r2_warmstart/model_10.pt`，SHA-256 `39678294b52a7403dae863cd6cd636d745f6e9d2106cd8d1cf184e2d4d3f43ec`；base 仍是冻结 26D I0 `pi0.pt`。

通道消融（有海流、2.0 m、`0–49` 50 seeds、PPO `model_10`，r3 先以旧配置 `scale=0.6667` 对比）：

| 残差通道 | 成功率 base → residual | 三维 RMSE 改善 | material 增量 | any 增量 | 约束违反 base → residual |
|---|---:|---:|---:|---:|---:|
| rudder only | `0.78 → 0.80` | `0.009013 m` | `+0.002663` | `+0.085795` | `10.56 → 133.98` |
| stern only | `0.78 → 0.82` | `0.155006 m` | `+0.006494` | `+0.027422` | `10.56 → 54.48` |
| RPM only | `0.78 → 0.80` | `0.038881 m` | `-0.000018` | `-0.000010` | `10.56 → 10.50` |
| stern + RPM | `0.78 → 0.82` | `0.176327 m` | `+0.006490` | `+0.027918` | `10.56 → 55.44` |
| rudder + stern | `0.78 → 0.80` | `0.029350 m` | `+0.009615` | `+0.110947` | `10.56 → 181.02` |

结论很明确：**舵角和艉舵残差贡献了几乎全部新增饱和，RPM 残差是唯一同时保持跟踪正收益和执行器近中性的通道**。因此 canonical r3 只保留 RPM 通道；这不代表 RPM 是理论上唯一有效方向，只代表在冻结 I0 的当前控制权限下，它是唯一通过门禁的方向。

canonical RPM-only 正式结果：

| 评估集 | 成功率 base → residual | 三维 RMSE base → residual | RMSE 改善 | material base → residual | any base → residual | 约束违反 base → residual |
|---|---:|---:|---:|---:|---:|---:|
| 有海流 2.0 m，`0–49` | `0.78 → 0.80` | `3.344359 → 3.264098 m` | `0.080261 m` | `0.00440002 → 0.00436571` | `0.00448827 → 0.00450932` | `10.56 → 10.42` |
| 有海流 1.25 m，`0–49` | `0.44 → 0.46` | `5.799747 → 5.689884 m` | `0.109863 m` | `0.00414187 → 0.00412236` | `0.00420678 → 0.00423920` | `12.02 → 11.90` |
| 无海流 2.0 m，`0–49` | `1.00 → 1.00` | `1.392567 → 1.392875 m` | `-0.000308 m` | `0.00571541 → 0.00574121` | `0.00612137 → 0.00609208` | `14.72 → 14.60` |
| 有海流 2.0 m，`50–99` | `0.92 → 0.92` | `2.520383 → 2.489837 m` | `0.030546 m` | `0.00473940 → 0.00480166` | `0.00490945 → 0.00495799` | `10.86 → 10.94` |
| 有海流 1.25 m，`50–99` | `0.62 → 0.62` | `4.344065 → 4.294572 m` | `0.049493 m` | `0.00439926 → 0.00447770` | `0.00454541 → 0.00461632` | `12.18 → 12.26` |
| 无海流 2.0 m，`50–99` | `1.00 → 1.00` | `1.242628 → 1.243236 m` | `-0.000608 m` | `0.00505287 → 0.00503885` | `0.00605760 → 0.00598639` | `14.20 → 13.92` |

canonical 结果的 residual abs mean 约为 `0.01035`（有海流 2.0 m）、`0.01396`（有海流 1.25 m）、`0.00166`（无海流 2.0 m），仍有明显的海流条件性，但比 r2 全通道的 `0.04796` 小很多。

门禁判定：

- **跟踪收益可重复**：两组互不重叠的 50 个 seeds 上，2.0 m 和 1.25 m 都出现同方向三维 RMSE 改善；成功率不下降，1.25 m 下两组分别 `+0.02/0.00`。
- **执行器近中性**：material saturation 的最大绝对增量为 `0.000078`，any-saturation 最大绝对增量为 `0.000071`，约束违反最大增量为 `0.08/episode`。全三通道 r2 的对应增量分别是 `+0.0185/+0.1236/+192.68`，量级相差两到三个数量级。
- **门禁口径固定**：material saturation 的“不上升”按**绝对容差 `0.001`**判定，同时仍要求 material saturation `< 0.05`；any-saturation 也报告，但不再把 `10^-5` 量级的轨迹敏感性当成部署级饱和。这个容差不改变 r2 的负结论，也不允许把舵面残差重新放回主链。
- **无海流审计通过**：两组 seeds 的成功率均为 `1.00`，残余动作幅值约 `0.0017–0.0025`，远低于有海流的 `0.010–0.014`；无海流 RMSE 变化约 `0.0003–0.0006 m`，属于近中性。
- **I2 解禁**：I2-DADE 只允许从 `model_10.pt` + `odarrl_control_i_residual_26d_rpm_only.json` 启动；不得从全三通道 r2 配置启动，也不得把 I1 的 `0.080261 m` 写成 DADE 的收益。

I1-r3 的正式结论是：**在冻结 I0 和当前 PVS 执行器包络下，残差控制权限应限制在 RPM；三通道残差虽然能取得更大的名义 RMSE 改善，但代价是不可接受的舵面饱和。** I2 的主要问题因此变成“DADE 能否从历史中推断海流并改善 RPM 残差”，而不是继续放大舵面残差。

### 14.4.7 I2-26D 显式历史栈基线（2026-10-31）

> **泄漏警告**：本节所有数字来自旧 `water_relative` 契约，`water_surge = linear_x - current_body_x` 与 `linear_x` 同帧存在，真实海流可被单帧精确恢复。它们只能作为旧基线复现，不是 hidden-current 推断证据。hidden 重跑见 14.4.9。

在实现 DADE 之前，先建立一个可解释、可复现的显式历史栈基线。它只回答“把短历史和长历史直接拼进策略输入是否有收益”，不包含学习得到的双时间尺度编码器，因此不能命名为 DADE。

实现边界：

- 新增 `src/auv_lowlevel/batch/history.py`，提供不可变 `HistoryLayout` 和 rsl_rl 兼容的 `BatchTorchHistoryVecEnv`；
- 不修改环境物理、PVS、IsaacLab 或 rsl_rl 源码；历史缓冲只在环境外层维护；
- 输入布局固定为 `[当前 26D, 短窗 10×26, 长窗 50×26]`，总维度 `26 × (1 + 10 + 50) = 1586`；
- episode reset 时两个窗口都用 reset 观测填满，避免把上一条 episode 的历史带进新 episode；
- checkpoint 元数据记录 `policy_kind=history_mlp`、`policy_input_dim=1586`、`observation_dim=26`、短窗和长窗长度；
- `scripts/evaluate_odarrl_control_residual.py` 从 checkpoint 元数据自动重建历史布局；旧 MLP checkpoint 不改变原有评估路径；
- 热启动只把旧 `model_10.pt` 的前 26 列复制到新 actor，新增 1560 个历史列用 `std=1e-3` 的小随机数初始化，不能把随机历史列误认为已训练编码器；
- `configs/lowlevel/odarrl_control_ppo_i2_26d_history.json` 使用 RPM-only canonical 环境、4096 envs、256 steps、40 iterations、seed `60000`、历史窗口 `10/50`。

训练与 checkpoint：

| 项目 | 结果 |
|---|---|
| I2 配置 SHA-256 | `be74ec96b65ae0ec7d32cea95cbbd590dcdc69f5805b4c5626b7976b7e3fabf7` |
| 训练规模 | `4096 × 256 × 40 = 41,943,040` env steps，状态 `completed` |
| 训练耗时/显存 | wall time `189.30 s`，峰值 allocated `13,293 MiB`、reserved `13,688 MiB` |
| 初始残差 | `initial_residual_abs_max=0.185617`，热启动确实加载 |
| smoke | `artifacts/lowlevel/i2_history_smoke_20261031`，`4×32` 成功；`i2_history_eval_smoke_20261031` 完成 1586D 往返 |
| canonical checkpoint | `model_20.pt`，SHA-256 `55a3a92b042b5d9003653c1969811550524b2d00d52f8dce80f4880d1a86b2de` |
| 复现 checkpoint | `model_39.pt` = `final.pt`，SHA-256 `184c4faa6b8ae7cae0511530456be4f2f0a58ab4b8a53bdfad059d1674410726` |
| 本地回归 | `107 passed`（`test_lowlevel_*` + `test_batch_*`）；ruff 与 `git diff --check` 通过 |

canonical `model_20` 的正式闭环结果如下。所有行都是同一 PVS 环境、同一 RPM-only 动作权限、base 与 residual 成对评估：

| 评估集 | 成功率 base → history | 三维 RMSE base → history | RMSE 改善 | residual abs | material base → history | any base → history | 约束违反 base → history |
|---|---:|---:|---:|---:|---:|---:|---:|
| 有海流 2.0 m，`0–49` | `0.78 → 0.78` | `3.344359 → 3.196378 m` | `0.147981 m` | `0.015280` | `0.004400 → 0.004367` | `0.004488 → 0.004499` | `10.56 → 10.48` |
| 有海流 1.25 m，`0–49` | `0.44 → 0.44` | `5.799747 → 5.493987 m` | `0.305760 m` | `0.022570` | `0.004142 → 0.004079` | `0.004207 → 0.004186` | `12.02 → 11.72` |
| 有海流 2.0 m，`50–99` | `0.92 → 0.92` | `2.520383 → 2.452679 m` | `0.067704 m` | `0.011137` | `0.004739 → 0.004887` | `0.004909 → 0.005058` | `10.86 → 11.14` |
| 有海流 1.25 m，`50–99` | `0.62 → 0.62` | `4.344065 → 4.174712 m` | `0.169353 m` | `0.016445` | `0.004399 → 0.004516` | `0.004545 → 0.004650` | `12.18 → 12.24` |
| 无海流 2.0 m，`0–49` | `1.00 → 1.00` | `1.392567 → 1.391768 m` | `0.000798 m` | `0.004443` | `0.005715 → 0.005741` | `0.006121 → 0.006221` | `14.72 → 15.10` |

第二个 checkpoint `model_39` 的复现结果同方向：有海流 2.0 m `0–49` 改善 `0.143926 m`、1.25 m `0–49` 改善 `0.290078 m`、2.0 m `50–99` 改善 `0.067147 m`；成功率均不降，material 最大增量 `+0.000205`，无海流改善 `0.001311 m`、residual abs `0.004699` 对 `0.015078`。因此增益不是单点 checkpoint 偶然。

与 I1 canonical 的增量对照：

| 条件 | I1 RMSE 改善 | I2-a `model_20` RMSE 改善 | I2-a 相对 I1 增量 |
|---|---:|---:|---:|
| 有海流 2.0 m，`0–49` | `0.080261 m` | `0.147981 m` | `+0.067720 m` |
| 有海流 1.25 m，`0–49` | `0.109863 m` | `0.305760 m` | `+0.195897 m` |
| 有海流 2.0 m，`50–99` | `0.030546 m` | `0.067704 m` | `+0.037158 m` |
| 有海流 1.25 m，`50–99` | `0.049493 m` | `0.169353 m` | `+0.119860 m` |

门禁判定：

- **增量收益可重复**：两个 checkpoint 在两组互斥 seeds、两个捕获半径上都是同方向改善，且 `model_20` 相对 I1 的增量在四个条件下均为正；
- **成功率不降**：所有有海流条件的 base/history 成功率逐项相同，无海流保持 `1.00`；
- **执行器近中性**：material 最大增量 `+0.000148`、any 最大增量 `+0.000149`，均低于固定绝对容差 `0.001`；最大约束违反增量 `+0.38/episode`；
- **残差条件化**：有海流 residual abs `0.015280`、无海流 `0.004443`，相差约 `3.44` 倍；严格 1.25 m 有海流为 `0.022570`；
- **训练曲线不能替代闭环评估**：训练日志中的 episode 指标不稳定，模型选择以固定 seeds 的闭环评估为准，不宣称单调收敛。

当前阶段判定：**I2-a 通过门禁，但不能命名为 DADE。**它的科学结论是“显式 10/50 历史栈相对 I1 有可重复增量”，下一阶段必须实现真正的学习编码器，并用 I2-a 作为强基线；DADE 只有在相对 I2-a 至少不退化、并带来额外精度或推理成本优势时才算通过。

### 14.4.8 I2-26D 双时间尺度编码器实验（静态 hidden DADE 为负结果；时变另见 14.4.9）

> **泄漏警告**：本节 DADE/10Hz 结果同样使用旧 `water_relative` 契约，不能解释为“DADE 从历史推断隐藏海流”。hidden-current DADE 重跑见 14.4.9；那里是新的负结果，结论以那一节为准。

I2-b 把显式历史栈替换成学习得到的两级编码器。短窗 `10×26` 和长窗 `50×26` 分别编码成固定维度的扰动特征，actor 和 critic 只接收 `[当前 26D, 扰动特征 64D]`；完整策略输入仍由环境按 1586D 扁平布局提供，因此不需要修改 rsl_rl 的 rollout 存储接口。实现位于 `src/auv_lowlevel/batch/encoded_policy.py`，checkpoint 会记录 `policy_kind` 和完整 `policy_config`，评估端从 checkpoint 自动恢复，不依赖文件名或隐藏默认值。

热启动规则：I1 actor 的当前观测列原样复制，新增的 64 个扰动输入列使用 `std=1e-3` 的小方差随机初始化，critic 的当前观测列也复制。不能把扰动输入列置零：actor 对编码器输出的雅可比会同时为零，编码器从第一次 PPO 更新起就收不到梯度，后续训练只会退化成 I1，而不是 DADE。随机初始化使初始闭环行为接近 I1，同时保留一条可学习的编码器通路；测试会显式检查编码器参数在反向传播后存在非零梯度。

#### I2-b-MLP：轻量编码器负结果

配置：`configs/lowlevel/odarrl_control_ppo_i2_26d_encoder_mlp.json`；训练规模 `4096×256×40 = 41,943,040` steps；输出 `artifacts/lowlevel/i2_residual_ppo_26d_encoder_mlp`。

| 项目 | 结果 |
|---|---|
| 训练配置 canonical SHA-256 | `d431478d34551c63da35cac4cb57b24064310cb64b9b148c1d6c9ca8d8171d94` |
| 训练耗时 | wall time `229.50 s` |
| 显存峰值 | allocated `13,331.56 MiB`，reserved `13,806 MiB` |
| `model_20.pt` SHA-256 | `9b5b82fe55ec9f3019a155cbd4b6e8921fcca23a71267ebe4310b17b5313a2ba` |
| `model_39.pt` SHA-256 | `4e8326c46af9ce3c6c16f0501de500142c268e0eba29d42500c99555a1cba696` |

有海流 2.0 m、seeds `0–49` 的闭环结果：

| checkpoint | 成功率 base → residual | 三维 RMSE base → residual | RMSE 改善 | residual abs | material base → residual | any base → residual | 违反 base → residual |
|---|---:|---:|---:|---:|---:|---:|---:|
| `model_20` | `0.78 → 0.80` | `3.344359 → 3.259581 m` | `0.084778 m` | `0.010567` | `0.004400 → 0.004368` | `0.004488 → 0.004521` | `10.56 → 10.44` |
| `model_39` | `0.78 → 0.80` | `3.344359 → 3.260034 m` | `0.084325 m` | `0.010777` | `0.004400 → 0.004362` | `0.004488 → 0.004515` | `10.56 → 10.44` |

判定：**轻量 MLP 编码器为负结果，不能作为 I2 的通过版本。**两个 checkpoint 的改善分别为 `0.084778 m` 和 `0.084325 m`，只接近 I1 的 `0.080261 m`，明显弱于 I2-a 的 `0.147981 m`。成功率提高和残差幅值下降不足以弥补相对显式历史栈的精度退化；因此不能写成“学习编码器优于显式历史栈”。

#### I2-b-DADE：完整 Transformer 编码器

完整 DADE 配置为 `configs/lowlevel/odarrl_control_ppo_i2_26d_dade_transformer.json`：两个窗口各使用一层 Transformer encoder，`feature_dim=64`、`num_heads=4`、`feedforward_dim=128`，融合层 `128→64` 输出 64D 扰动特征。为避免 4096 环境 rollout 存储与长窗反传激活叠加导致 OOM，正式配置保持 `4096×256×40` 等预算，使用 `num_mini_batches=256`；单轮 smoke 为 `65.65 s`，峰值 reserved `14,974 MiB`。正式训练输出目录为 `artifacts/lowlevel/i2_residual_ppo_26d_dade_transformer`。

修复后的正式训练和评估已完成，结果见下方“修复后正式 DADE 结果”。DADE 必须与 I2-a 的 `model_20` 对照；只有相对 I2-a 不退化，并同时满足成功率、执行器近中性、残差条件化和推理延迟门禁，才可以写“DADE 通过”。本轮未满足该门禁。

#### 首轮 DADE 无效：零初始化造成编码器死梯度

第一轮正式 DADE 在 `model_20` 的 50-seed、有海流 2.0 m、seeds `0–49` 评估中只改善 `0.080060 m`，成功率 `0.78 → 0.80`，residual abs `0.010412`，与 I1 的 `0.080261 m` 几乎相同，明显弱于 I2-a 的 `0.147981 m`。这不是“Transformer 没学到海流”的正常负结果，而是实现错误：

- 热启动把 actor 第一层的 64 个扰动输入列全部置零；
- 因此 actor 输出对编码器特征的一阶导数在这些列上恒为零，编码器在第一次反向传播时得到零梯度；
- 扰动列本身仍会从 PPO 获得梯度，但编码器在初始更新中只能得到零或极小的学习信号，训练因此近似退化为只学习当前 26D 观测到残差的映射。

该轮产物已重命名为 `artifacts/lowlevel/i2_residual_ppo_26d_dade_transformer_dead_encoder_20261031`，对应评估目录也带 `_dead_encoder_20261031` 后缀；**这些结果只作为 bug 证据，不得写成 DADE 负结果**。修复后扰动列改为 `std=1e-3` 随机初始化，新增反向传播测试，正式训练已从干净输出目录重新启动。修复 smoke 的 warmstart metadata 为 `disturbance_columns_init="normal"`、`disturbance_columns_std=0.001`，首轮更新 loss 有限。重训 `model_0 → model_5` 的参数变化也证明编码器确实参与优化：短编码器 L2 变化 `1.4399`（初始范数 `19.9238`）、长编码器 `2.1084`（`19.9301`）、融合层 `1.0982`（`12.2411`），不再是冻结的随机特征。

时间窗消融采用同一输入宽度的 10Hz 抽样方案，而不是把长窗直接扩成 250 个 token：`configs/lowlevel/odarrl_control_ppo_i2_26d_dade_transformer_10hz.json` 把短窗和长窗都设为每 5 个控制步采样一次，因此编码器仍接收 `10+50` 个样本，但物理覆盖范围从短窗 `0.2 s`、长窗 `1.0 s` 扩展到短窗 `1.0 s`、长窗 `5.0 s`。`BatchTorchHistoryVecEnv`、checkpoint metadata 和评估端共用同一组 `short_stride/long_stride` 约定，避免训练时按 10Hz、评估时却按 50Hz 拼接。该配置保持 `4096×256×40` 等预算，正式结果见下方“10Hz 时间窗消融”。

#### 修复后正式 DADE 结果

> **稳定性警告（2026-11-01）**：本节是旧 legacy water-relative、训练 seed `62000` 的结果。后续 seed-matched `legacy_seed72000_paired` 在同一配置和预算下变为 `-0.441 m`、成功率 `0.78→0.70`，证明这里的小幅正收益不能跨 seed 复现；DADE 的 seed 方差大于当前契约差异。阅读本节时必须结合 14.4.9 的 2×2 配对结论，不能把下表当作稳定方法性能。

训练 lineage：配置 `configs/lowlevel/odarrl_control_ppo_i2_26d_dade_transformer.json`，canonical SHA-256 `6b547e012ac4d3fa6fd35e2550f829c46e3db39a3bc4781a545f606e749cfc8a`；`4096×256×40 = 41,943,040` steps；wall time `2,350.19 s`；训练峰值 allocated `14,933.20 MiB`、reserved `15,018 MiB`；训练 manifest `status=completed`。两个 checkpoint：

| checkpoint | SHA-256 |
|---|---|
| `model_20.pt` | `3dea145a007d33eee94b4d74fd3f4992bade59bc040219973071f1c497297852` |
| `model_39.pt` / `final.pt` | `3ffd04261ed77ec7decdb88119f151754774331a907444ede5343dab4a77cf21` |

闭环评估输出为 `artifacts/lowlevel/i2_dade_gradfix_eval_*`，推理延迟为 `artifacts/lowlevel/i2_dade_latency_model20/` 和 `i2_dade_latency_model39/`，中间 checkpoint 筛查为 `artifacts/lowlevel/i2_dade_gradfix_screen8_current_model*/`。

四个有海流条件和无海流审计均使用 50 个 seed、同一 PVS 配置、同一参考路径和 `residual_gains=[0,0,1]`。三维 RMSE 改善为 base 减 residual，正值表示变好：

| 条件 | I1 `model_10` | I2-a `model_20` | DADE `model_20` | DADE `model_39` | DADE m20 相对 I2-a |
|---|---:|---:|---:|---:|---:|
| 2.0 m 默认，seeds `0–49` | `0.080261 m` | `0.147981 m` | `0.088846 m` | `0.098990 m` | `-0.059135 m` |
| 1.25 m 严格，seeds `0–49` | `0.109863 m` | `0.305760 m` | `0.123805 m` | `0.138037 m` | `-0.181955 m` |
| 2.0 m 留出，seeds `50–99` | `0.030546 m` | `0.067704 m` | `0.034548 m` | `0.040575 m` | `-0.033156 m` |
| 1.25 m 留出，seeds `50–99` | `0.049493 m` | `0.169353 m` | `0.056818 m` | `0.060373 m` | `-0.112535 m` |

DADE 两个 checkpoint 的闭环与执行器指标：

| checkpoint / 条件 | 成功率 base → residual | 三维 RMSE base → residual | 水平改善 | 垂向改善 | material base → residual | residual abs |
|---|---:|---:|---:|---:|---:|---:|
| m20 2.0 m `0–49` | `0.78 → 0.80` | `3.344359 → 3.255513 m` | `0.005824 m` | `0.014455 m` | `0.004400 → 0.004370` | `0.010419` |
| m20 1.25 m `0–49` | `0.44 → 0.46` | `5.799747 → 5.675942 m` | `-0.008958 m` | `0.006671 m` | `0.004142 → 0.004127` | `0.013874` |
| m20 2.0 m `50–99` | `0.92 → 0.92` | `2.520383 → 2.485835 m` | `-0.000830 m` | `0.004848 m` | `0.004739 → 0.004795` | `0.008761` |
| m20 1.25 m `50–99` | `0.62 → 0.62` | `4.344065 → 4.287247 m` | `-0.005974 m` | `0.003953 m` | `0.004399 → 0.004469` | `0.011364` |
| m39 2.0 m `0–49` | `0.78 → 0.80` | `3.344359 → 3.245369 m` | `0.009931 m` | `0.016160 m` | `0.004400 → 0.004335` | `0.010664` |
| m39 1.25 m `0–49` | `0.44 → 0.46` | `5.799747 → 5.661709 m` | `-0.006475 m` | `0.008731 m` | `0.004142 → 0.004092` | `0.014098` |
| m39 2.0 m `50–99` | `0.92 → 0.92` | `2.520383 → 2.479808 m` | `0.002160 m` | `0.005880 m` | `0.004739 → 0.004822` | `0.009260` |
| m39 1.25 m `50–99` | `0.62 → 0.62` | `4.344065 → 4.283691 m` | `-0.006924 m` | `0.005139 m` | `0.004399 → 0.004493` | `0.011821` |

无海流 2.0 m、seeds `0–49`：m20 三维 RMSE 改善 `-0.000017 m`，residual abs `0.001448`；m39 改善 `0.000228 m`，residual abs `0.001891`。有海流 residual abs 约为 `0.0104–0.0107`，因此“有流时残差显著大于无流时”的条件化成立。

推理延迟用 CPU 单线程、1000 次测量：m20 p50/p95 `0.629/0.646 ms`，m39 `0.621/0.638 ms`，低于 `20 ms` 门禁。

编码器使用程度：热启动时 64 个扰动输入列的 L2 范数约为 `0.128`；训练后 m20 为 `0.138076`（增加约 7.9%），m39 为 `0.165994`（增加约 29.7%）。编码器通路已经非零且确实参与优化，但 actor 仍极少使用编码器特征。8-seed 中间 checkpoint 筛查为 m20/m25/m30/m35/m39 = `0.193194/0.203986/0.206210/0.209815/0.214202 m`，增益随训练缓慢上升，但同 8 个 seed 上 I2-a `model_20` 为 `0.349633 m`，差距没有收敛迹象。

判定：**修复后的完整 DADE 仍然未通过 I2 的强基线门禁。**它在四个有海流条件上均略优于 I1，成功率在主条件上提高 `0.02`、material saturation 近中性、无海流残差幅值低，说明功能实现和条件化都成立；但相对 I2-a 显式历史栈在四个条件上全部退化，严格捕获和留出集上的差距尤其大。I2-a 继续作为当前最佳 I2 版本，DADE 作为可复现负结果保留。后续如果继续迭代 DADE，优先解决“64D 编码特征没有被 actor 有效使用”和信息瓶颈，而不是只延长训练；单纯延长当前 schedule 的收益斜率远不足以追上 I2-a。

#### 10Hz 时间窗消融

配置 `configs/lowlevel/odarrl_control_ppo_i2_26d_dade_transformer_10hz.json`，canonical SHA-256 `020a4a5c8dc46b2767d4511fdddb7dcd2ebce9b6e5e93c72047eee7e6495de5e`；`short_stride=5`、`long_stride=5`，输入仍是 `10+50` 个 token，但物理覆盖范围变为短窗 `1.0 s`、长窗 `5.0 s`。训练保持 `4096×256×40 = 41,943,040` steps、相同 I1 热启动和相同算法预算；wall time `2,357.95 s`，峰值 allocated `15,034.82 MiB`、reserved `16,018 MiB`。checkpoint：

| checkpoint | SHA-256 |
|---|---|
| `model_20.pt` | `f6b4457332c2660c035e2f3cce5202fe5e1819e70463ff8250a230f396ef44fe` |
| `model_39.pt` / `final.pt` | `bfaf0dfe4e147b311b608b9a75778a946c29c4274ba30d57aed015ca8ebf27d7` |

训练输出 `artifacts/lowlevel/i2_residual_ppo_26d_dade_transformer_10hz/`，评估输出 `artifacts/lowlevel/i2_dade_10hz_eval_*`，延迟输出 `artifacts/lowlevel/i2_dade_10hz_latency_model20/`。

三维 RMSE 改善对比（50Hz 为同一 DADE 结构、`stride=1`；10Hz 为每 5 控制步抽样）：

| 条件 | m20 50Hz | m20 10Hz | m20 变化 | m39 50Hz | m39 10Hz | m39 变化 |
|---|---:|---:|---:|---:|---:|---:|
| 2.0 m `0–49` | `0.088846 m` | `0.070717 m` | `-0.018129 m` | `0.098990 m` | `0.072612 m` | `-0.026378 m` |
| 1.25 m `0–49` | `0.123805 m` | `0.085035 m` | `-0.038770 m` | `0.138037 m` | `0.089372 m` | `-0.048666 m` |
| 2.0 m `50–99` | `0.034548 m` | `0.027078 m` | `-0.007470 m` | `0.040575 m` | `0.028041 m` | `-0.012534 m` |
| 1.25 m `50–99` | `0.056818 m` | `0.033425 m` | `-0.023393 m` | `0.060373 m` | `0.034336 m` | `-0.026037 m` |

10Hz 的成功率与 50Hz 相同（主条件 `0.78 → 0.80`，严格 `0.44 → 0.46`），material saturation 近中性；但水平改善在严格和留出条件下转为负值，垂向改善也明显缩小。无海流 residual abs 为 m20 `0.001996`、m39 `0.001911`，条件化仍成立。CPU 单线程推理 p95 为 `0.744 ms`，延迟门禁通过。

判定：**10Hz 抽样消融为负结果。**在不增加 token 数的前提下把窗口拉长到 5 秒，反而在全部四个有海流条件上低于 50Hz DADE，说明短时间尺度的执行器/速度信息比更长的稀疏历史更重要。若未来继续做慢时间尺度，应使用真正的 10/250 token 双分支或独立慢上下文编码器，而不是把 10/50 个 token 简单降采样。

### 14.4.9 观测契约泄漏修复、hidden-current 重跑与时变海流

#### 根因：不是“维度太多”，而是真实海流进了最终 actor

旧实现的 26D/27D 观测同时包含：

- 索引 8：`linear_x`，地速纵荡；
- 索引 11：`water_surge = linear_x - current_body_x`；
- 索引 12：`speed_error = reference_speed - water_surge`。

因此任何单帧都可以精确恢复 `current_body_x = linear_x - water_surge`。`observation.expose_current=false` 只把索引 17–19 置零，并没有隐藏这个派生量；`speed_error` 又是同一条海流的线性组合。旧 `_validate_current_visibility()` 只检查 `guidance.compensate_current` 和 `base_controller.use_current_field`，漏掉了这两个速度反馈项。所以旧 I 臂结果不能解释为“DADE 从历史推断隐藏海流”。

修复前的实现也没有独立的特权 critic：`BatchTorchVecEnv.num_privileged_obs = None`，且 `EncodedActorCritic` 的 actor/critic 观测同宽、critic 调用同一套 `_features()`。因此当时不能把 `water_surge` 解释成“只有训练期 critic 能看的特权信息”。**本轮后续已补上真正独立的 critic 输入通路**：actor 继续只吃 hidden 26D/history，critic 额外吃 3D 真实海流，且不顺带经过 actor encoder；旧 checkpoint 仍保留共享特征兼容路径。实现、smoke 和 4096 正式训练均见“独立特权 critic”小节；正式训练结果为负结果，不能写成特权 critic 带来性能收益。

#### 修复后的两套契约

| 契约 | `expose_current` | `speed_feedback` | 索引 11/12 | 允许的真实海流用途 |
|---|---:|---|---|---|
| `hidden` | `false` | `ground` | 地速纵荡 / 地速跟踪误差 | reward、诊断、独立特权 critic、教师、辅助目标；最终 actor 不可见 |
| `oracle` | `true` | 默认 `water_relative` | 对水速度 / 对水速度误差 | 显式上界对照；结果必须标注 oracle |
| `legacy` | 旧配置缺字段时使用 | `water_relative` | 旧行为 | 只用于复现旧结果，不得再声称 hidden-current |

实现改动：

- `ObservationConfig` 增加 `speed_feedback` 和 `current_contract`；
- `_parse_observation()` 接受可选字段，旧配置不改 JSON、不加字段，canonical hash 不变；
- `ObservationBuilder` 和 `BatchedObservationBuilder` 共用同一语义：`ground` 下索引 11 只用 `linear_x`，索引 12 只用 `reference_speed - linear_x`；
- `hidden` 强制 `expose_current=false`、`speed_feedback=ground`，并继续拒绝 `guidance.compensate_current` / `base_controller.use_current_field`；
- `oracle` 强制 `expose_current=true`；显式 `expose_current=true` 但不写契约时按 oracle 处理；
- `current.enabled=false` 时禁止残留 `current.time_variation`，避免“关掉海流但正弦项仍产生非零流”的配置漏洞；
- BC checkpoint、批量 PPO checkpoint 和训练 manifest 记录 `observation_speed_feedback` / `observation_current_contract` / `observation_expose_current`；旧 checkpoint 缺字段时按 `legacy` + `water_relative` 读取；
- 加载冻结 BC 时可传入期望契约，旧 `pi0` 不会被静默喂入 ground 观测。

#### 哈希口径：canonical hash 与文件 hash 不是同一个值

本轮排查过“checkpoint 里的 hash 和评估 summary 不一致”，结论不是配置漂移，而是两个合法口径：

- `config_sha256`：`json.dumps(raw, sort_keys=True, separators=(",", ":"))` 后取 SHA-256，用于评估 summary 和 dataclass 元数据；
- 文件 SHA-256：直接对 JSON 原文字节取 SHA-256，用于训练 manifest 和 checkpoint metadata。

两者内容相同但字节布局不同，因此数值不同。hidden 配置的两个口径如下：

| 配置 | canonical `config_sha256` | 文件 SHA-256 |
|---|---|---|
| `odarrl_control_i_residual_26d_rpm_only_hidden.json` | `0bff1983…c690f6` | `de8b8a65…c7401` |
| `odarrl_control_ppo_i1_26d_hidden.json` | `916bbf46…8d67a` | `b57e64e4…5aef` |
| `odarrl_control_ppo_i2_26d_history_hidden.json` | `188bf0fb…87cbb` | `db6ab136…c3baf` |
| `odarrl_control_ppo_i2_26d_dade_transformer_hidden.json` | `b16aa739…2f794` | `9ae2334a…c1b0` |

后续引用 checkpoint 时，必须注明用的是哪个口径；不能把 `188bf0…` 和 `db6ab1…` 当成两次不同训练。

#### 回归证据

- `tests/test_lowlevel_hidden_current.py`：同一 hidden 配置下，把真实海流从 `0.35 m/s` 改为 `-0.25 m/s`，标量与批量观测逐元素完全相同；索引 11 只等于 `linear_x / 2.5`，索引 12 只等于 `reference_speed - linear_x`；oracle 仍保留索引 17–19 的真实海流。
- `tests/test_lowlevel_config.py`：hidden + water-relative 被拒绝；hidden + expose_current 被拒绝；oracle + expose_current=false 被拒绝；旧配置默认契约保持 legacy/oracle。
- `tests/test_lowlevel_time_varying_current.py`：时变流周期/相位确定性、批量环境逐 step 更新、`current.enabled=false` + `time_variation` 被拒绝。
- `tests/test_lowlevel_hidden_runtime_contract.py`：hidden 配置下，同一状态/同一指令换任意真实海流，批量化 PD/PI 教师输出逐元素不变；oracle 配置下则随海流改变，证明测试对泄漏通道敏感。
- `tests/test_lowlevel_residual_eval.py`：终端残差 governor 必须绑定环境；距目标 `4.0/2.625/1.25 m` 时按 `1.0/0.5/0.0` 缩放；未启用 governor 时动作逐元素不变。
- `tests/test_lowlevel_*.py tests/test_batch_*.py`：本地全量 `414 passed`；ruff 通过。
- 服务器 512 环境 smoke：hidden 26D、时变流、无 NaN/Inf；checkpoint 元数据为 `observation_speed_feedback=ground`、`observation_current_contract=hidden`、`observation_expose_current=false`。

#### 旧结果重标注

| 旧结果 | 旧解释 | 正确标注 |
|---|---|---|
| I0 26D BC/DAgger | 隐藏海流 student | 训练数据 `current.enabled=false`，ground 与 water-relative 数值相同；可作为 hidden 基础策略，但不含海流推断能力 |
| I1 r3 RPM-only | 隐藏海流残差 | water-relative-feedback、单帧可恢复海流；不是 hidden-current 证据 |
| I2-a 显式历史栈 | 隐藏海流历史推断 | 同上；历史栈的增益可能部分来自单帧泄漏 |
| I2-b DADE / 10Hz | 隐藏海流编码器 | 同上；DADE 负结果仍然有效，但不能再说“它没能从历史推断海流” |
| O-residual / P4 | 隐藏海流对照 | oracle 或 legacy water-relative；不能与 hidden 结果混表 |

#### hidden-current 正式重跑

环境配置 `configs/lowlevel/odarrl_control_i_residual_26d_rpm_only_hidden.json`：

- `observation = {dim:26, expose_current:false, speed_feedback:"ground", current_contract:"hidden"}`；
- `current.enabled=true`，速度 `0.2–0.4 m/s`，方向 `[-π, π]`，episode 内恒定；
- `guidance.compensate_current=false`、`base_controller.use_current_field=false`；
- canonical SHA-256 `0bff1983…c690f6`，文件 SHA-256 `de8b8a65…c7401`。

训练配置和输出：

| 实验 | 配置 | 输出 | 状态 |
|---|---|---|---|
| I1 hidden RPM-only | `configs/lowlevel/odarrl_control_ppo_i1_26d_hidden.json` | `artifacts/lowlevel/i1_residual_ppo_26d_hidden_ground` | 完成 |
| I2-a hidden 显式历史 | `configs/lowlevel/odarrl_control_ppo_i2_26d_history_hidden.json` | `artifacts/lowlevel/i2_residual_ppo_26d_history_hidden_ground` | 完成 |
| I2-b hidden DADE | `configs/lowlevel/odarrl_control_ppo_i2_26d_dade_transformer_hidden.json` | `artifacts/lowlevel/i2_residual_ppo_26d_dade_transformer_hidden_ground` | 完成 |
| 快速时变+空间剪切 DADE | `configs/lowlevel/odarrl_control_ppo_i2_26d_dade_transformer_hidden_fast_timevarying_shear.json` | `artifacts/lowlevel/i2_residual_ppo_26d_dade_transformer_hidden_fast_timevarying_shear` | 完成，负结果 |

冻结基础策略迁移：旧 `pi0.pt` 的 180 个 BC episode 在 `current.enabled=false` 下采集，ground 与 water-relative 的索引 11/12 数值完全相同。因此服务器生成 `artifacts/lowlevel/i0_26d_bc_dagger_r1_lineage_20261031/pi0_hidden_ground.pt`，只补契约元数据、不改任何网络权重；SHA-256 为 `38ce943c…e79f9b`。

关键 checkpoint（SHA-256）：

| 实验/用途 | checkpoint | SHA-256 |
|---|---|---|
| hidden I1 | `i1_residual_ppo_26d_hidden_ground/model_10.pt` | `7d7782af…21218c` |
| hidden I2-a | `i2_residual_ppo_26d_history_hidden_ground/model_20.pt` | `d4da8f20…71dbc9` |
| hidden DADE | `i2_residual_ppo_26d_dade_transformer_hidden_ground/model_39.pt` | `2dbe5677…fe80c2` |
| 时变 I1 | `i1_residual_ppo_26d_hidden_timevarying/model_20.pt` | `715db1b9…d1f47c` |
| 时变 I2-a | `i2_residual_ppo_26d_history_hidden_timevarying/model_30.pt` | `2a6ee3df…bcccb9` |
| 时变 DADE canonical | `i2_residual_ppo_26d_dade_transformer_hidden_timevarying/model_20.pt` | `56ad4783…e797ef` |
| 时变 DADE 最终 | `i2_residual_ppo_26d_dade_transformer_hidden_timevarying/model_39.pt` | `a3fd790b…26980d` |
| 剪切训练 DADE | `i2_residual_ppo_26d_dade_transformer_hidden_fast_timevarying_shear/model_39.pt` | `3f578dcb…02781d` |
| hidden DADE seed62000 paired | `i2_residual_ppo_26d_dade_transformer_hidden_seed62000_paired/model_20.pt` | `0c20de0e…e5e15e` |
| hidden DADE seed62000 paired final | `i2_residual_ppo_26d_dade_transformer_hidden_seed62000_paired/model_39.pt` | `c3e33221…12c845` |
| legacy DADE seed72000 paired | `i2_residual_ppo_26d_dade_transformer_legacy_seed72000_paired/model_20.pt` | `03ac9efa…8aa984` |
| legacy DADE seed72000 paired final | `i2_residual_ppo_26d_dade_transformer_legacy_seed72000_paired/model_39.pt` | `f1994bf4…fd071d` |
| hidden 基础策略 | `i0_26d_bc_dagger_r1_lineage_20261031/pi0_hidden_ground.pt` | `38ce943c…e79f9b` |

#### 旧 water-relative 与 hidden：seed-matched DADE 2×2

配对性修正（2026-11-01）：旧泄漏实验和第一轮 hidden 重跑使用的训练 seed 不同——I1 是 `50000` 对 `70000`，I2-a 是 `60000` 对 `71000`，DADE 是 `62000` 对 `72000`。因此早期的跨 seed 表只能作描述性对照，不能把全部差异归因于观测契约。为此补了两个 seed-matched 配置：`hidden_seed62000_paired` 和 `legacy_seed72000_paired`，均保持 `4096×256×40`、同一冻结 base 权重、同一 reward、同一评估 seeds，只切换 actor 契约。

必须强调：这仍不是“只改 DADE actor”的纯因果消融。hidden 契约同时改变了冻结 BC 基座的速度反馈（`water_relative` → `ground`），因此结果包含“base 输入变化 + DADE 输入变化”的联合效应；不能把差异全部写成 encoder 的功劳。

`model_20` 在默认 2.0 m、seeds `0–49` 的 2×2 结果：

| actor 契约 | seed `62000` | seed `72000` |
|---|---|---|
| legacy water-relative | `+0.088846 m`，成功率 `0.78→0.80`，共同成功 `-0.003850 m`，residual abs `0.0104` | `-0.440934 m`，成功率 `0.78→0.70`，共同成功 `-0.181381 m`，residual abs `0.2188` |
| hidden ground | `+0.676014 m`，成功率 `0.84→0.82`，共同成功 `-0.026388 m`，residual abs `0.3302` | `+0.051163 m`，成功率 `0.84→0.84`，共同成功 `-0.306661 m`，residual abs `0.1281` |

同一 seed 下并没有稳定的“hidden 更好”方向：seed `62000` 的 hidden 总 RMSE 很大但默认成功率下降；seed `72000` 的 hidden 总 RMSE 只略正、共同成功严重退化；legacy 自身从 seed `62000` 的小幅正收益变成 seed `72000` 的四格全负。legacy seed `72000` 的另外三格为：严格 `0–49` `-0.324842 m`、成功率 `0.44→0.36`、共同成功 `-0.269744 m`；默认 `50–99` `-0.372254 m`、成功率 `0.92→0.84`、共同成功 `-0.114929 m`；严格 `50–99` `-0.157307 m`、成功率 `0.62→0.56`、共同成功 `-0.108014 m`。该 checkpoint 的 residual abs 为 `0.21–0.23`，不是旧 seed `62000` 的 `~0.01`。

`model_39` 完整复现该负结果：总 RMSE 改善 `-0.439594/-0.317564/-0.371588/-0.153421 m`（顺序同四格），成功率 `0.78→0.70`、`0.44→0.36`、`0.92→0.84`、`0.62→0.56`，residual abs `0.207–0.227`。两个 checkpoint 的行为几乎一致，说明这不是某个保存点的偶然。

判定：**观测泄漏修复是科学合法性的必要条件，但它不是 DADE 性能增益的来源；当前更大、更致命的问题是 PPO/DADE 的跨 seed 不稳定。**legacy 和 hidden 都能在不同 seed 下产生“总 RMSE 正、共同成功负、成功率下降”或“总 RMSE 负、执行器激进”的不同失败模式。仅用 2 个 seed 无法判断契约的平均效应；更不能再用“移除泄漏后 DADE 变好”作为结论。下一步必须先做至少 5 个训练 seed 的多重启动、固定评估 seed 集，并把共同成功子集、成功率和 residual abs 一起作为门禁，再讨论 encoder 结构。

#### hidden I1：基本没有可重复正收益

`model_10` 是 8-seed 屏幕中相对最好的 checkpoint；`model_20` 只作为敏感性对照。50-seed 结果（改善 = base RMSE − residual RMSE，正数更好）：

| 条件 | I1 `model_10` 改善 | 成功率 base → residual | residual abs |
|---|---:|---:|---:|
| 默认 2.0 m，`0–49` | `+0.002772 m` | `0.84 → 0.84` | `0.001917` |
| 严格 1.25 m，`0–49` | `-0.081568 m` | `0.58 → 0.56` | `0.001955` |
| 默认 2.0 m，`50–99` | `+0.001223 m` | `0.92 → 0.92` | `0.001891` |
| 严格 1.25 m，`50–99` | `-0.006135 m` | `0.58 → 0.58` | `0.001918` |

`model_20` 在 `0–49` 上也分别是 `-0.003377 m`（默认）和 `-0.011919 m`（严格），没有更好。判定：**hidden I1 的残差幅值只有 `~0.002`，默认条件收益在毫米级，严格条件为负；不能作为“隐藏海流残差控制已通过”的证据。**这也说明旧 I1 的 `0.080/0.110 m` 主要不是来自隐藏海流推断。

#### hidden I2-a：小幅但四格同向的正收益

canonical `model_20`：

| 条件 | I2-a 改善 | 成功率 base → residual | residual abs |
|---|---:|---:|---:|
| 默认 2.0 m，`0–49` | `+0.036340 m` | `0.84 → 0.84` | `0.018054` |
| 严格 1.25 m，`0–49` | `+0.063367 m` | `0.58 → 0.58` | `0.018933` |
| 默认 2.0 m，`50–99` | `+0.018750 m` | `0.92 → 0.92` | `0.017670` |
| 严格 1.25 m，`50–99` | `+0.155074 m` | `0.58 → 0.60` | `0.018420` |

`model_10` 敏感性：默认 `0–49` `+0.043660 m`、默认 `50–99` `+0.021997 m`、严格 `50–99` `+0.173979 m`，但严格 `0–49` 为 `-0.019258 m` 且成功率 `0.58 → 0.56`。因此 canonical 选 `model_20`，不是因为它单格最高，而是因为它四格都不退化、成功率不降。判定：**hidden I2-a 有真实但很小的正收益，远小于旧 water-relative 的 `0.148/0.306/0.068/0.169 m`。**

无海流对照（`0–49`、默认 2.0 m）：I2-a `model_20` 改善 `+0.023951 m`、residual abs `0.015560`；有海流 residual abs `0.018054`。**无海流只下降约 14%，不是旧实验的 71%。**因此 hidden I2-a 的残差主要是通用跟踪修正，海流条件化很弱；旧实验里“强条件化”很可能大部分来自 `water_surge` 泄漏。这里旧/新 I2-a 仍是跨 seed 对照，尚未做 seed-matched 复跑，不能给出精确因果比例。

#### hidden DADE：seed 差异很大，总 RMSE 改善由失败轨迹主导

第一轮 hidden DADE（seed `72000`）的短屏幕曾经给出 `model_10/20/30/39` 约 `+0.16/+0.18/+0.16/+0.18 m` 的 8-seed 改善，但 50-seed 四格否定了它。`model_39`（最终 checkpoint）结果：

| 条件 | 总 RMSE 改善 | 成功率 base → residual | 共同成功子集改善 | residual abs |
|---|---:|---:|---:|---:|
| 默认 2.0 m，`0–49` | `+0.0511 m` | `0.84 → 0.84` | `-0.3014 m` | `0.1265` |
| 严格 1.25 m，`0–49` | `+0.0836 m` | `0.58 → 0.50` | `-0.3801 m` | `0.1218` |
| 默认 2.0 m，`50–99` | `-0.2919 m` | `0.92 → 0.84` | `-0.1796 m` | `0.1290` |
| 严格 1.25 m，`50–99` | `+0.6603 m` | `0.58 → 0.56` | `-0.1994 m` | `0.1251` |

共同成功子集指 base 和 residual 都成功的 seed；它排除了“失败轨迹误差变小”对总体均值的影响。上表说明第一轮 hidden DADE 即使总 RMSE 为正，也在成功轨迹上大幅退化；默认留出集还直接损失成功率。`model_10/20` 复现同样模式。

seed-matched 的 hidden DADE（seed `62000`，同 base、同 `4096×256×40`、同 reward）给出另一种结果。`model_20`：

| 条件 | 总 RMSE 改善 | 成功率 base → residual | 共同成功子集改善 | residual abs |
|---|---:|---:|---:|---:|
| 默认 2.0 m，`0–49` | `+0.676014 m` | `0.84 → 0.82` | `-0.026388 m` | `0.3302` |
| 严格 1.25 m，`0–49` | `+1.665063 m` | `0.58 → 0.60` | `-0.012478 m` | `0.3391` |
| 默认 2.0 m，`50–99` | `+0.232468 m` | `0.92 → 0.90` | `-0.033118 m` | `0.3231` |
| 严格 1.25 m，`50–99` | `+1.442396 m` | `0.58 → 0.58` | `-0.019725 m` | `0.3269` |

`model_39` 复现同样方向：总 RMSE 改善 `+0.6730/+1.5835/+0.2289/+1.4394 m`，成功率 `0.84→0.82`、`0.58→0.58`、`0.92→0.90`、`0.58→0.58`，共同成功子集改善 `-0.0310/-0.0179/-0.0368/-0.0226 m`。两个 checkpoint 的 residual abs 都在 `0.32–0.34`，约为 I2-a 的 18 倍。

判定：**hidden DADE 的总 RMSE 正收益是真实存在的，但主要来自超时/失败轨迹的误差收敛，不是成功轨迹上的高精度跟踪；它未通过“成功率不降 + 共同成功子集不退化”的强基线门禁。**seed `62000` 与 `72000` 的差异也远大于契约本身，不能从单个 seed 推断“移除泄漏后 DADE 变好”。DADE 目前更像高增益恢复器；若要保留 I2-a 的精度收益，应做分误差门控或 hybrid（小误差用 I2-a，大误差/离轨时用 DADE），而不是直接用它替代历史栈。

#### 时变海流：I2-a 四格通过，DADE 通过失败尾部/成功率门禁

实现：`CurrentVariationConfig` 增加速度幅值、方向幅值和周期；`SmoothTemporalCurrent` 以 episode 相位做平滑正弦调制；批量环境每 step 重算当前海流，自动 reset 时同步重置相位。本小节当时仍是水平、空间均匀流；后续已在 14.4.9 增加线性水平空间剪切，但不支持垂向流。

实验配置：

| 配置 | canonical `config_sha256` | 文件 SHA-256 |
|---|---|---|
| `odarrl_control_i_residual_26d_rpm_only_hidden_timevarying.json` | `04cbc7b9…d81079` | `50fc0489…aa07f` |
| `odarrl_control_ppo_i1_26d_hidden_timevarying.json` | `69d68119…7346de` | `528b09f0…36b49a` |
| `odarrl_control_ppo_i2_26d_history_hidden_timevarying.json` | `34e050d8…fa1cd4` | `eb056a75…45fda0` |
| `odarrl_control_ppo_i2_26d_dade_transformer_hidden_timevarying.json` | `197f9e65…be39145` | `07a1483e…18d6` |

变化幅度：速度 `±0.08 m/s`、方向 `±0.5 rad`、周期 `15 s`；基准速度 `0.2–0.4 m/s`、方向 `[-π, π]`。

I1 时变（`model_20`，8-seed 屏幕最优）：

| 条件 | I1 改善 | 成功率 base → residual | residual abs |
|---|---:|---:|---:|
| 默认 `0–49` | `+0.013979 m` | `0.84 → 0.84` | `0.002855` |
| 严格 `0–49` | `+0.033178 m` | `0.64 → 0.64` | `0.003442` |
| 默认 `50–99` | `+0.006820 m` | `0.92 → 0.92` | `0.002527` |
| 严格 `50–99` | `+0.037665 m` | `0.60 → 0.60` | `0.003114` |

I2-a 时变（`model_30`，8-seed 屏幕最优）：

| 条件 | I2-a 改善 | 成功率 base → residual | residual abs |
|---|---:|---:|---:|
| 默认 `0–49` | `+0.0661 m` | `0.84 → 0.84` | `0.0191` |
| 严格 `0–49` | `+0.1215 m` | `0.64 → 0.64` | `0.0206` |
| 默认 `50–99` | `+0.0396 m` | `0.92 → 0.92` | `0.0187` |
| 严格 `50–99` | `+0.4157 m` | `0.60 → 0.64` | `0.0205` |

无海流对照：I2-a 时变改善 `+0.0219 m`、residual abs `0.0169`；有海流 `0.0191`。**时变 I2-a 的四格收益高于静态 hidden I2-a，但无海流仍有正收益，残差幅值只下降约 12%，所以不能把全部收益都写成海流补偿。**I1 时变无海流改善 `-0.0021 m`、residual abs `0.0022`，说明 I1 的微小时变收益更依赖海流。

DADE 时变训练已完成（`4096×256×40`、`cuda:6`）。`model_10` 和 `model_20` 的四格 50-seed 结果一致，且显著高于 I2-a；`model_20` 作为 canonical 候选：

| 条件 | DADE `model_20` 改善 | 成功率 base → residual | residual abs |
|---|---:|---:|---:|
| 默认 `0–49` | `+0.5338 m` | `0.84 → 0.86` | `0.0615` |
| 严格 `0–49` | `+0.6599 m` | `0.64 → 0.60` | `0.0635` |
| 默认 `50–99` | `+0.3330 m` | `0.92 → 0.94` | `0.0595` |
| 严格 `50–99` | `+0.7132 m` | `0.60 → 0.60` | `0.0654` |

`model_10` 敏感性为 `+0.5517/+0.6933/+0.3432/+0.7391 m`，`model_39` 为 `+0.5179/+0.6327/+0.1701/+0.6927 m`；三个 checkpoint 都是四格正收益，没有“只有早期 checkpoint 偶然好”的问题。`model_39` 的默认留出集低于 `model_20`，所以 canonical 选 `model_20`，不是选最终 checkpoint。

执行器指标在 `model_20` 上同步改善：material saturation 四格从 `0.004250/0.004274/0.004743/0.004360` 降到 `0.004031/0.003979/0.004093/0.003807`，约束违反从 `10.24/11.64/10.56/12.62` 降到 `9.56/10.68/9.08/10.86`。这说明增益不是靠更激进的执行器换来的；代价是 steps mean 增加约 `45–85` 步（默认约 1–2 秒、严格约 2–3 秒），因此它是“跟踪更准、执行器更平滑”，不是“完成更快”。

最关键的无海流审计：

| DADE 时变 checkpoint | 有海流默认 `0–49` | 无海流默认 `0–49` | 成功率（无流） | residual abs（有流/无流） |
|---|---:|---:|---:|---:|
| `model_10` | `+0.5517 m` | `-0.1559 m` | `1.00 → 0.98` | `0.0656 / 0.0521` |
| `model_20` | `+0.5338 m` | `-0.1567 m` | `1.00 → 0.98` | `0.0615 / 0.0517` |

**有流和无流的方向相反，说明它学到的是海流条件化的残差，而不是通用的跟踪偏置。** residual abs 只下降约 20%，但收益方向翻转，说明关键不是“残差幅值变小”，而是“残差方向随海流改变”。这与静态 hidden DADE 的失败形成鲜明对照。

判定：**时变海流 DADE 首次通过“显著 RMSE 改善 + 留出不退化 + 无海流方向翻转”的性能门禁，且 `model_10/20/39` 三个 checkpoint 一致。**唯一未满足的是严格 `0–49` 成功率 `0.64 → 0.60`（少 2/50 个 seed）；按原门禁这仍是失败项，不能写成“完全通过”。下一步应优先用成功率加权、终端捕获课程或终端残差抑制修这 `0.04`，而不是继续延长训练。**但性能通过不等于历史推断成立；后续 history ablation 已证明当前 encoder 没有可测的历史依赖，见下一小节。**

#### 严格成功率残差尺度消融（负结果）

为判断严格条件少 2/50 是否由残差幅度过大导致，对同一个 `model_20` checkpoint 做评估时 residual-scale 消融。基础策略和 seeds 完全不变，只缩放最终残差：

| residual scale | 默认 `0–49` RMSE 改善 | 成功率 | 严格 `0–49` RMSE 改善 | 成功率 | 严格 residual abs |
|---:|---:|---:|---:|---:|---:|
| `1.0` | `+0.533848 m` | `0.84 → 0.86` | `+0.659906 m` | `0.64 → 0.60` | `0.063491` |
| `0.9` | `+0.508508 m` | `0.84 → 0.86` | `+0.590014 m` | `0.64 → 0.60` | `0.056858` |
| `0.8` | `+0.481877 m` | `0.84 → 0.86` | `+0.448787 m` | `0.64 → 0.58` | `0.050086` |

结论：**降低残差幅度不修复严格成功率，`0.8` 反而更差。**严格 `1.0` 的四次成功翻转是 seed `1` timeout→goal、seed `25/42/49` goal→timeout。后三个 seed 的残差轨迹最小目标距离分别为 `1.2626/1.2894/1.4081 m`，只差 `1.25 m` 门限 `0.01–0.16 m` 就捕获，但随后继续前进并最终 timeout。因此问题更像是终端捕获行为和奖励/课程，而不是“残差太大”或 RMSE 拟合不足。下一步不应继续扫 scale，也不应单纯延长训练；应优先修 terminal capture 的奖励、残差在终端区的抑制，或做严格半径课程。

#### 终端残差包络消融（负结果）

为避免把严格成功率下降误判成“残差幅度过大”，新增部署侧 `TerminalResidualGovernor`：距目标超过 `full_scale_radius_m` 时保留完整残差，小于等于 goal radius 时残差为零，中间用 smoothstep 平滑过渡。评估脚本新增 `--terminal-residual-radius`，zero radius 固定为当前 `goal_radius_m`；这不是新的学习策略，只是评估期安全包络。

对同一个时变 DADE `model_20`、同一批 `0–49` seeds、严格 `1.25 m` 条件做 50-seed 消融：

| full-scale radius | 严格 RMSE 改善 | 成功率 base → residual | residual abs | 成功翻转 |
|---:|---:|---:|---:|---|
| 无包络（`1.0` scale） | `+0.659906 m` | `0.64 → 0.60` | `0.063491` | `1,25,42,49` |
| `3.0 m` | `+0.636497 m` | `0.64 → 0.60` | `0.061230` | `1,25,42,49` |
| `4.0 m` | `+0.617527 m` | `0.64 → 0.60` | `0.059757` | `1,25,42,49` |
| `6.0 m` | `+0.611334 m` | `0.64 → 0.62` | `0.057188` | `1,42,49` |
| `8.0 m` | `+0.582090 m` | `0.64 → 0.62` | `0.054933` | `1,42,49` |

结论：**终端包络能降低残差幅值，但不能修复严格捕获。**`3/4 m` 完全无效，`6/8 m` 只救回一个 seed，仍比 base 低 `0.02`；失败 seed 仍停在约 `1.24–1.37 m` 的最小距离，随后 timeout。根因更可能是终端捕获奖励/严格半径课程和终端行为，而不是残差大小。后续应训练带 terminal capture reward 或 strict-radius curriculum 的策略；不能把终端包络的 RMSE 改善写成 DADE 学会了终端捕获。

#### terminal capture 奖励复跑（部分改善，仍未通过）

为验证“严格成功率差 2/50 是否只是终端奖励不足”，在时变海流 DADE 上新增 `terminal_progress_weight=15`，并把训练环境的目标半径直接设为 `1.25 m`。配置为 `configs/lowlevel/odarrl_control_ppo_i2_26d_dade_transformer_hidden_timevarying_strict_capture.json`，环境配置为 `configs/lowlevel/odarrl_control_i_residual_26d_rpm_only_hidden_timevarying_strict_capture.json`；从时变 DADE `model_20.pt` 热启动，`4096×256×40`、`cuda:6`、seed `74000` 完成训练，manifest 记录 `observation_current_contract=hidden`、`observation_speed_feedback=ground`、`observation_expose_current=false`。

`seeds 0–7` 严格屏幕（`1.25 m`）：

| checkpoint | 三维 RMSE 改善 | 成功率 base → residual |
|---|---:|---:|
| `model_10` | `+1.881088 m` | `0.375 → 0.500` |
| `model_20` | `+1.808648 m` | `0.375 → 0.500` |
| `model_30` | `+1.816984 m` | `0.375 → 0.500` |
| `model_39` | `+1.837529 m` | `0.375 → 0.500` |

8-seed 结果只用于选候选，不能代替 50-seed。`model_10` 的正式 50-seed 四格如下（改善 = base RMSE − residual RMSE，正数更好）：

| 条件 | 三维 RMSE 改善 | 成功率 base → residual | residual abs |
|---|---:|---:|---:|
| 2.0 m，`0–49` | `+0.501106 m` | `0.84 → 0.84` | `0.302618` |
| 2.0 m，`50–99` | `+0.362724 m` | `0.92 → 0.94` | `0.292901` |
| 1.25 m，`0–49` | `+0.794848 m` | `0.64 → 0.62` | `0.307056` |
| 1.25 m，`50–99` | `+0.806149 m` | `0.60 → 0.60` | `0.303510` |

和上一版时变 DADE 相比，RMSE 改善从约 `0.66/0.71 m` 提高到 `0.79/0.81 m`，严格 `0–49` 成功率缺口从 `0.04` 缩小到 `0.02`；代价是 steps mean 增加约 `63–68` 步（默认口径约慢 `1.3 s`），material saturation 和约束违反均下降。`model_39` 的结果方向一致但略弱（严格 `0–49` 为 `+0.766854 m`、成功率 `0.64→0.62`），因此 canonical 仍取 `model_10`。

无海流对照：

| 条件 | 改善 | 成功率 base → residual | 解读 |
|---|---:|---:|---|
| 2.0 m，`0–49` | `-0.154896 m` | `1.00 → 0.98` | 有流训练收益不能直接迁到无流；方向翻转仍成立 |
| 1.25 m，`0–49` | `+0.073185 m` | `0.94 → 0.94` | 严格半径下无流也略有正收益，说明并非单一方向偏置 |

终端残差包络对新模型仍未修复成功率：`4/6/8 m` 的严格 `0–49` 成功率都是 `0.64→0.62`，RMSE 改善分别降到 `+0.745154/+0.707695/+0.672881 m`。失败 seed 仍是 `42/49`（最小距离 `1.296/1.449 m`，随后 timeout），同时救回 seed `1`，净少 1 个成功 seed；这不是“残差太大”能解释的。

判定：**terminal capture reward 是当前最好的严格成功率修复尝试，但仍是失败项，不能写成“严格捕获已通过”。**它把缺口减半，却没有消除；下一步应做严格半径课程或在终端区切换回 base/显式捕获控制器，而不是继续增加 terminal progress 权重或继续扫 residual scale。

#### 终端控制权 handoff：8 m → 3 m 通过聚合严格门禁

在 terminal capture 模型上评估终端残差 governor：距目标大于 `full_scale_radius` 时保留完整残差，距目标小于 `zero_radius` 时把残差严格置零，中间用 smoothstep 过渡。该机制只改变部署时的残差权限，不改变 DADE 网络，因此不能写成“DADE 学会终端捕获”。

严格 `1.25 m`、`0–49` 50-seed：

| full / zero radius | RMSE 改善 | 成功率 base → residual | steps mean | 判定 |
|---:|---:|---:|---:|---|
| `6 / 2 m` | `+0.694616 m` | `0.64 → 0.62` | `1887.2 → 1951.6` | 成功率仍降 |
| `6 / 3 m` | `+0.678710 m` | `0.64 → 0.62` | `1887.2 → 1950.7` | 成功率仍降 |
| `8 / 2 m` | `+0.660550 m` | `0.64 → 0.62` | `1887.2 → 1949.7` | 成功率仍降 |
| `8 / 3 m` | `+0.655157 m` | `0.64 → 0.64` | `1887.2 → 1938.2` | 聚合门禁通过 |

`8 / 3 m` 在独立留出集 `50–99` 上复核：RMSE `5.311116 → 4.649301 m`（改善 `+0.661814 m`），成功率 `0.60 → 0.60`，steps mean `1918.4 → 1973.9`。`50–99` 没有成功/失败翻转；`0–49` 有两个翻转：seed `1` timeout（最小距离 `1.706 m`）→ goal（`1.246 m`），seed `49` goal（`1.249 m`）→ timeout（`1.386 m`）。因此聚合成功率不降，但单个 seed 的行为仍发生互换，不能把这一格写成“所有失败 seed 都被修复”。

结论：**`8 / 3 m` 是目前唯一在 `0–49` 和 `50–99` 两段都通过“严格成功率不降 + RMSE 正收益”的终端 handoff 配置。**代价是 steps mean 增加约 `51–56` 步；约束违反从 `11.64` 降至 `10.92`。该结果应报告为部署侧控制权切换，而不是 DADE 的终端捕获能力；若要求逐 seed 行为不变，仍需严格半径课程或专用终端控制器。

#### 独立特权 critic：actor 与 critic 的职责已经分开

新增 `BatchTorchHistoryVecEnv(..., privileged_current=True)`：

- actor 输入仍是原布局 `[current hidden 26D, short history, long history]`，例如 `1586D`；
- 额外输出 `extras["observations"]["critic"] = [current hidden 26D, true current NED 3D]`，宽度 `29D`；
- `num_privileged_obs=29`，但不是 actor 观测；真实海流不进入当前 26D、历史栈或 actor encoder；
- `EncodedActorCritic(..., independent_critic=True)` 用独立 MLP 读 critic 输入，`evaluate()` 不再调用 actor 的 `_features()`；
- checkpoint metadata 新增 `critic_mode` 与 `critic_observation_dim`，评估端按 metadata 重建 actor/critic 宽度；
- 旧 checkpoint 的 `critic_mode` 缺省为 `shared_actor_features`，保持原共享特征路径，因此历史结果仍可复现，不会被静默改义。

新增配置：

| 配置 | canonical `config_sha256` | 文件 SHA-256 |
|---|---|---|
| `odarrl_control_ppo_i2_26d_dade_transformer_hidden_timevarying_privcritic.json` | `744f8129…3bc254` | `e2245ba7…a9cdd` |
| `odarrl_control_ppo_i2_26d_history_hidden_timevarying_privcritic.json` | `8243d992…52033d5` | `5128ce04…40b83f` |

接口审计：

- 8 环境真实 `BatchTorchVecEnv + BatchTorchHistoryVecEnv` smoke：actor `(8, 1586)`、critic `(8, 29)`，actor 索引 `17:20` 恒为零，critic 尾三列随真实时变海流变化，stepped observation 有限；
- 完整 `train_batch_lowlevel_ppo.py` 1-iteration smoke：DADE 路径的 manifest 记录 `privileged_current_critic=true`、`critic_observation_dim=29`；checkpoint 中 actor 首层 `(256, 90)`、critic 首层 `(256, 29)`，证明 critic 没有复用 actor 的 `26+64` 特征；
- `history_mlp` 路径同样完成 1-iteration smoke：actor 首层 `(256, 1586)`、critic 首层 `(256, 29)`，manifest 与 checkpoint metadata 均为 `privileged_current_independent`；
- 两个 checkpoint 都用评估脚本重建并跑通 1-seed/20-step rollout；
- 本地全量测试 `407 passed`；服务器 hidden/time-varying/history/encoded 定向测试 `22 passed`。

#### 独立特权 critic 正式对照（负结果）

接口链路已经完成，随后在 `cuda:2` 上正式训练 DADE privcritic：

| 项目 | 值 |
|---|---|
| 训练预算 | `4096×256×40`，`41,943,040` env steps |
| 配置 | `odarrl_control_ppo_i2_26d_dade_transformer_hidden_timevarying_privcritic.json` |
| actor | hidden 26D 历史布局，`1586D` |
| critic | `[hidden 26D, true current NED 3D]`，`29D`，独立 MLP |
| 最终 checkpoint SHA-256 | `4a702572…9301a7d` |
| 峰值显存 | allocated `7702.4 MiB`，reserved `8708.0 MiB` |
| collection 吞吐 | `935,686` env steps/s（manifest 汇总） |

8-seed `0–7` 屏幕与已有 shared-critic DADE 对照：

| 模型 | 默认 2.0 m 改善 | 成功率 base → residual | residual abs |
|---|---:|---:|---:|
| shared DADE `model_20` | `+0.551232 m` | `0.75 → 0.75` | `0.081195` |
| privcritic `model_39` | `-4.303684 m` | `0.75 → 0.375` | `0.096315` |
| conservative privcritic `model_9` | `-1.916870 m` | `0.75 → 0.50` | `0.047137` |

`privcritic model_39` 的 50-seed 默认 `0–49` 确认不是短屏幕偶然：

| 条件 | 改善 | 成功率 base → residual | residual abs |
|---|---:|---:|---:|
| 默认 2.0 m，`0–49` | `-1.972790 m` | `0.84 → 0.70` | `0.094074` |
| 严格 1.25 m，`0–7` | `-1.430706 m` | `0.375 → 0.375` | `0.096106` |
| 无海流默认 2.0 m，`0–7` | `-0.014678 m` | `1.00 → 1.00` | `0.071923` |

保守版（`lr=1e-5`、2 epochs、`clip=0.05`、`desired_kl=0.005`、10 iterations；canonical `c8ed0c4e…087944e`，file `307f2004…061d7403`）也没有救回来：默认 `-1.916870 m`、成功率 `0.75→0.50`；严格 `-0.774317 m`、成功率 `0.375→0.375`；无海流 `+0.056852 m`、成功率 `1.00→1.00`。原始 privcritic 的 `model_10/15/20/25/30/35/39` 在默认条件下全部约 `-4.3` 到 `-4.47 m`，不是单个 checkpoint 的问题。

判定：**actor/critic 隔离本身正确，但当前“critic 只看当前 26D + 真实海流”的非对称 critic 配方在这一任务上不稳定。**有海流时价值/优势估计退化，无海流时基本正常；因此不能声称特权 critic 提升了 DADE，也不能把这条负结果写成隐藏海流推断成功。下一步若要继续，应让 critic 使用与 actor 同时间尺度的历史特征（独立 critic encoder，不共享 actor `_features()`），并考虑 actor/critic 分离学习率、value normalization 或先冻结 actor 只训练 critic；在没有这些改动前，shared critic 仍是唯一可用基线。

#### History ablation 前仍未完成

- 独立特权 critic 正式训练已完成但为负结果；当前不能声称特权 critic 有性能收益，也不能把 privcritic 结果当成 hidden-current DADE 的正结果。
- `legacy_seed72000_paired` 已证明 legacy 自身也会从 seed `62000` 的 `+0.089 m` 翻到 seed `72000` 的 `-0.441 m`、成功率 `0.78→0.70`；因此 2 个 seed 不足以判断观测契约的平均效应。进入下一个方法前，至少要做 5 个训练 seed 的 DADE 多重启动，并固定评估 seeds。
- DADE 的总体 RMSE 正收益由失败/超时轨迹主导，共同成功子集反而略退化；不能再用总 RMSE 单项判定“DADE 更准”。下一步若要继续 DADE，应优先做 I2-a/DADE 分误差门控或 hybrid，并增加“共同成功子集不退化”的硬门禁，而不是继续放大 residual 或只换 seed。
- history ablation 已完成且为负结果：完整历史、仅当前帧和仅短窗的结果几乎相同；因此目前只能证明 DADE 的总误差/成功率表现，不能证明它“确实从历史推断海流”。详见“推理期 history ablation”。
- terminal capture reward 把严格 `0–49` 成功率缺口从 `0.04` 缩小到 `0.02`，但未通过；新模型上的终端包络 `4/6/8 m` 也仍为 `0.64→0.62`。下一步转向严格半径课程或终端控制权切换，而不是继续扫 residual scale/包络；
- 已实现时间变化、水平均匀流和线性水平空间剪切流；垂向流、真实 DVL/IMU/定位误差尚未加入。
- DVL 底锁/失锁、IMU 零偏/航向漂移、定位延迟、执行器死区/响应延迟属于后续“部署性审计”，不能和本轮契约修复混成一个结论。

#### 可用性审计与 DADE 再检验（2026-11-01）

这一轮把“真实海流只作为训练特权信息”落成了可执行契约，而不只是文档声明。最终 actor 的信息按部署可获得性审计如下：

| 最终 actor 输入 | 现实可获得性 | hidden 配置中的实现 | 处理结论 |
|---|---|---|---|
| 索引 8–10 地速 `u/v/w` | DVL 底跟踪；深水可能失锁，`v/w` 不一定全测 | PVS 精确状态 | 当前保留；后续加噪声、失锁和延迟 |
| 索引 11 `feedback_surge` | hidden 下只允许地速 | `linear_x`，不再减真实海流 | 已修复单帧泄漏 |
| 索引 12 `speed_error` | hidden 下只用地速参考误差 | `reference_speed - linear_x` | 已修复单帧泄漏 |
| 索引 17–19 海流槽位 | 真实策略没有精确海流 | hidden 恒为零 | 不是隐藏海流证据；oracle 才允许非零 |
| 水平/垂向误差、`s`、曲率 | 需要定位、路径投影和地图 | 精确状态与最近点 | 保留；真实部署需定位误差和投影跳变测试 |
| 姿态、角速度 | IMU/AHRS/陀螺仪 | 精确状态 | 保留；后续加零偏、噪声和航向漂移 |
| 上一步执行器、上一步残差 | 指令可测；策略自知 | 精确记录 | 保留；后续加执行器响应延迟和死区 |

真实海流在本轮 hidden 配置中的允许用途只有四类：奖励、诊断/分组日志、独立特权 critic、教师或辅助目标。`guidance.compensate_current=false` 与 `base_controller.use_current_field=false` 同时保证参考量和基底控制器不会把真实海流重新编码进 actor 输入；`water_surge`/`speed_error` 也不再进入 hidden actor。独立 critic 已单独输出 `[hidden 26D, true current NED 3D]`，actor 宽度和历史布局不变；其正式训练结果是负结果，不能因为“隔离正确”就写成性能收益。

为防止后续实验再次把旧配置误当 hidden，本轮新增 `validate_arm_current_contract()` 并在 PPO 启动入口强制调用：只要 I 臂开启海流，`observation.current_contract` 必须是 `hidden`；旧 `legacy` 配置只有显式传入 `--allow-legacy-current-contract` 才能作为历史复现运行。manifest 同时记录 `arm_current_contract_validation`。回归测试覆盖“legacy I 配置默认拒绝、显式历史例外通过、hidden I 配置通过”。

评估汇总也新增了 `paired_analysis`（`src/auv_lowlevel/scripts_support.py`、`scripts/evaluate_odarrl_control_residual.py`，单元测试 `tests/test_lowlevel_scripts_support.py`）：同时输出 base/residual 共同成功 seed 子集、共同失败子集和成功翻转列表。原因是失败/超时 episode 的 RMSE 很大，会主导总体均值；只看总 RMSE 会把“失败轨迹被拉回一点”误读成“成功轨迹更准”。以后 DADE/I2 的正结果必须同时报告总 RMSE、成功率和共同成功子集三项。

#### 快速时变海流零样本留出

已有 15 s 周期时变流下 DADE 的 RMSE 证据较强，但仍需回答“DADE 是否真的更适应变化海流”，还是只适应了训练分布。为此新增 `configs/lowlevel/odarrl_control_i_residual_26d_rpm_only_hidden_fast_timevarying.json`：周期从 `15 s` 收紧到 `8 s`，速度摆幅从 `0.08` 增至 `0.10 m/s`，方向摆幅从 `0.5` 增至 `0.9 rad`，其余路径、奖励、执行器和 hidden 契约完全不变。

这不是重新训练，而是对已有 I2-a 和 DADE checkpoint 的零样本分布移出评估。两种方法使用同一 base、同一 `0–49` seeds、同一环境配置，分别跑默认 `2.0 m` 和严格 `1.25 m`。预注册判据是：DADE 必须在默认和严格两档都不弱于 I2-a，且相对 base 的 RMSE 改善不能以成功率下降换取；若 DADE 只在名义 15 s 流上占优、在 8 s 流上退化，则不能声称“DADE 更适合变化海流”。

50-seed 结果：

| 方法/条件 | 总 RMSE 改善 | 成功率 base → residual | 共同成功子集改善 | steps mean | residual abs |
|---|---:|---:|---:|---:|---:|
| 历史栈，默认 2.0 m，`0–49` | `+0.113714 m` | `0.90 → 0.90` | `+0.014876 m` | `1560.8 → 1555.2` | `0.0393` |
| 历史栈，严格 1.25 m，`0–49` | `+0.115182 m` | `0.66 → 0.66` | `+0.024757 m` | `1854.3 → 1852.9` | `0.0411` |
| 历史栈，默认 2.0 m，`50–99` | `-0.200405 m` | `0.94 → 0.92` | `+0.017389 m` | `1490.3 → 1513.3` | `0.0388` |
| 历史栈，严格 1.25 m，`50–99` | `-0.293767 m` | `0.74 → 0.68` | `+0.028227 m` | `1749.0 → 1805.6` | `0.0405` |
| DADE，默认 2.0 m，`0–49` | `+0.389420 m` | `0.90 → 0.90` | `-0.009536 m` | `1560.8 → 1632.8` | `0.2964` |
| DADE，严格 1.25 m，`0–49` | `+0.787068 m` | `0.66 → 0.66` | `-0.009366 m` | `1854.3 → 1904.3` | `0.2990` |
| DADE，默认 2.0 m，`50–99` | `+0.393040 m` | `0.94 → 0.96` | `-0.020585 m` | `1490.3 → 1539.6` | `0.2862` |
| DADE，严格 1.25 m，`50–99` | `+1.003164 m` | `0.74 → 0.78` | `-0.009381 m` | `1749.0 → 1764.9` | `0.2899` |

**按“总 RMSE + 成功率”这一预注册判据，DADE 四格通过：总 RMSE 全部为正、成功率全部不降，且两档都显著强于显式历史栈。**在 `0–49`，DADE 的默认/严格总 RMSE 改善是历史栈的 `3.4×/6.8×`；在 `50–99`，历史栈两格都退化为负值，而 DADE 仍为 `+0.393/+1.003 m`，严格成功率还从 `0.74` 升到 `0.78`。

但共同成功子集揭示了机制：历史栈在四格的共同成功 seed 上都改善 `+0.015~+0.028 m`，DADE 则略退化 `-0.009~-0.021 m`。因此 DADE 的优势不是“成功轨迹更准”，而是**把失败/超时轨迹的误差拉回来，并在留出集把少数 timeout 翻成 goal**。准确表述应是“DADE 更快变化流下更稳健、失败尾部更小”，不能写成全面精度提升。若目标是高精度跟踪，I2-a 仍是更好的底座；DADE 更适合作为离轨/大误差恢复分支。

对照实验也说明“在时变流上训练”是必要的：未在时变流上训练的静态 hidden DADE，直接迁移到 8 s 流时默认 `0–49` 为 `-0.280 m`、成功率 `0.90→0.86`、共同成功子集 `-0.315 m`；严格 `0–49` 总 RMSE 虽为 `+0.106 m`，但成功率 `0.66→0.64`、共同成功子集 `-0.407 m`。因此不能只归功于 Transformer 结构；训练分布和时变历史既改善了失败恢复，也把成功轨迹的退化从静态模型的 `-0.3~-0.4 m` 压到约 `-0.01 m`。

代价仍然明确：DADE 的 residual abs 约 `0.29`，是历史栈的 7 倍；steps mean 增加 `49–72` 步。该方法的定位应写成“更快变化流下失败尾部更小、成功率更稳”，而不是“所有成功轨迹更准”，也不是“更省执行器或更快完成”。

#### 推理期 history ablation（负结果，2026-11-01）

为验证 DADE 是否真的使用时间历史，而不是只做当前观测到残差的映射，新增评估专用参数 `--history-ablation`。它保留真实历史缓冲区不变，只修改送进策略的短窗/长窗：

- `none`：完整历史；
- `current-only`：短窗和长窗都重复当前帧，完全移除时间信息；
- `short-only`：保留短窗，长窗替换为当前帧；
- `long-only`：保留长窗，短窗替换为当前帧；
- `zero`：两个历史块全部置零，作为分布外对照。

实现位于 `scripts/evaluate_odarrl_control_residual.py`，不修改训练、checkpoint 或环境代码；单元测试覆盖各模式和形状校验。统一使用 8 s 快速时变流、`0–49` seeds、`model_20`（checkpoint SHA-256 `56ad4783…e797ef`）、training config SHA-256 `07a1483e…d18d6` 和 fast environment config SHA-256 `486e3f55…5d8d31`。

默认 `2.0 m`：

| 历史输入 | 总 RMSE 改善 | 成功率 base → residual | 共同成功子集 | residual abs | steps mean |
|---|---:|---:|---:|---:|---:|
| 完整历史 `none` | `+0.389420 m` | `0.90 → 0.90` | `-0.009536 m` | `0.2964` | `1560.8 → 1632.8` |
| 仅当前帧 `current-only` | `+0.391158 m` | `0.90 → 0.90` | `-0.007277 m` | `0.2970` | `1560.8 → 1632.1` |
| 仅短窗 `short-only` | `+0.391095 m` | `0.90 → 0.90` | `-0.007373 m` | `0.2970` | `1560.8 → 1632.2` |
| 仅长窗 `long-only` | `+0.389489 m` | `0.90 → 0.90` | `-0.009423 m` | `0.2963` | `1560.8 → 1633.0` |
| 全置零 `zero` | `+0.323682 m` | `0.90 → 0.90` | `-0.080931 m` | `0.2600` | `1560.8 → 1665.7` |

严格 `1.25 m`：

| 历史输入 | 总 RMSE 改善 | 成功率 base → residual | 共同成功子集 | residual abs | steps mean |
|---|---:|---:|---:|---:|---:|
| 完整历史 `none` | `+0.787068 m` | `0.66 → 0.66` | `-0.009366 m` | `0.2990` | `1854.3 → 1904.3` |
| 仅当前帧 `current-only` | `+0.788898 m` | `0.66 → 0.66` | `-0.006323 m` | `0.2996` | `1854.3 → 1903.8` |
| 全置零 `zero` | `+0.727705 m` | `0.66 → 0.66` | `-0.103549 m` | `0.2630` | `1854.3 → 1937.7` |

四种非零输入的成功翻转 seed 完全相同；`current-only` 与完整历史的总 RMSE 差只有 `0.002 m` 量级。**因此当前 DADE 的收益不能归因于历史编码：把全部历史替换成当前帧后，性能和成功行为几乎不变。** `zero` 的退化更可能来自输入分布外，而不是“移除了历史信息”，因为同一网络在更自然的 `current-only` 重复帧输入下没有退化。

这是一个关键负结果：DADE 目前应被描述为“带双时间尺度接口、但实测主要依赖当前观测的残差策略”，不能写成“从历史推断海流”。要让历史真正参与，需要改训练目标或约束，例如 history-only/current-dropout 训练、海流预测辅助损失、历史扰动一致性、或显式限制当前帧信息；在完成这些对照前，不继续增加 encoder 容量，也不把速度历史当成已验证机制。

#### 空间剪切流：零样本迁移（2026-11-01）

快速时变均匀流证明了 DADE 对训练分布附近的时变扰动有失败尾部收益，但不能回答空间剪切。为此新增 `CurrentShearConfig`，环境支持线性水平剪切：

`v_current(NED) = v_base(t) + G @ (p - p_origin)`，其中 `G` 的第三行为零，因此不引入垂向流。

剪切环境配置为 `configs/lowlevel/odarrl_control_i_residual_26d_rpm_only_hidden_fast_timevarying_shear.json`（canonical SHA-256 `7129d69d…09855a`），在 8 s 快速时变流上加入：

| 项目 | 值 |
|---|---|
| `G` 的前两行 | `[[0.002, 0.001, 0.010], [-0.001, -0.002, -0.008]]` |
| `origin_ned_m` | `[15.0, 0.0, 11.0]` |
| 约束 | 只支持水平剪切；`G[2, :] = 0` |

actor 仍是 hidden 契约：`speed_feedback=ground`，索引 17–19 为零；真实海流只进入奖励、诊断和训练特权通路。评估复用仅在 15 s 均匀时变流上训练过的 `model_20`，属于**零样本 OOD 迁移**，不是剪切场训练结果。

完整历史、`0–49`：

| 条件 | 总 RMSE 改善 | 成功率 base → residual | 共同成功子集 | residual abs |
|---|---:|---:|---:|---:|
| 默认 `2.0 m` | `+0.373691 m` | `0.82 → 0.80` | `-0.001825 m` | `0.297360` |
| 严格 `1.25 m` | `+1.045570 m` | `0.62 → 0.66` | `-0.004387 m` | `0.301385` |

独立留出 `50–99`：

| 条件 | 总 RMSE 改善 | 成功率 base → residual | 共同成功子集 | residual abs |
|---|---:|---:|---:|---:|
| 默认 `2.0 m` | `+0.432323 m` | `0.90 → 0.90` | `+0.001507 m` | `0.286654` |
| 严格 `1.25 m` | `+0.948963 m` | `0.66 → 0.70` | `-0.004344 m` | `0.291035` |

`0–49` 的 `current-only` 消融与完整历史几乎重合：默认改善 `+0.375239 m`、成功率 `0.82 → 0.80`、共同成功 `+0.000413 m`；严格改善 `+1.047297 m`、成功率 `0.62 → 0.66`、共同成功 `-0.001827 m`。**因此零样本剪切结果也没有显示历史编码被使用。**

判定：**剪切场本身比均匀流明显更难**（同一 base 在默认 `0–49` 从 `0.90` 降到 `0.82`，严格从 `0.66` 降到 `0.62`）。DADE 的主要收益仍来自失败/超时尾部（严格成功率 `+0.04`，50–99 默认成功率保持、严格 `+0.04`），但默认 `0–49` 损失 2 个百分点，且共同成功子集在四格中三格为负。**不能把这组写成“DADE 已适应空间剪切流”**；它只说明剪切 OOD 下失败尾部恢复仍有效。下一组必须做剪切场训练，才能区分“分布外”与“encoder 不会建模空间流”。

#### 空间剪切流：在剪切训练分布内继续训练（2026-11-01）

为区分零样本 OOD 与“结构本身不会处理空间海流”，从上一节 `model_20` 热启动，在同一快速时变+剪切环境继续训练 `4096×256×40`（`cuda:3`，seed `73000`）。配置为：

- `configs/lowlevel/odarrl_control_ppo_i2_26d_dade_transformer_hidden_fast_timevarying_shear.json`（文件 SHA-256 `957f5e1c…c999a9`）；
- 环境仍为上一节 canonical hidden 配置（canonical SHA-256 `7129d69d…09855a`）；
- warm-start actor：时变均匀流 `model_20.pt`，SHA-256 `56ad4783…e797ef`；
- canonical checkpoint：`model_39.pt`，SHA-256 `3f578dcb…02781d`；
- 8-seed 筛选：`model_10/20/30/39` 总 RMSE 改善 `+0.720/+0.736/+0.741/+0.763 m`，成功率均为 `0.625→0.625`，共同成功子集为 `-0.072/-0.066/-0.067/-0.062 m`。选 `model_39` 后进入 50-seed 复核。

50-seed 结果（改善 = base RMSE − residual RMSE）：

| 条件 | 总 RMSE 改善 | 成功率 base → residual | 共同成功子集 | residual abs | steps mean |
|---|---:|---:|---:|---:|---:|
| 默认 `2.0 m`，`0–49` | `+0.401050 m` | `0.82 → 0.80` | `+0.002571 m` | `0.299512` | `1647.4 → 1730.3` |
| 严格 `1.25 m`，`0–49` | `+1.087284 m` | `0.62 → 0.66` | `-0.000575 m` | `0.304501` | `1895.2 → 1908.1` |
| 默认 `2.0 m`，`50–99` | `+0.444662 m` | `0.90 → 0.90` | `+0.003393 m` | `0.285732` | `1541.7 → 1606.3` |
| 严格 `1.25 m`，`50–99` | `+0.976408 m` | `0.66 → 0.70` | `-0.003602 m` | `0.291555` | `1818.7 → 1840.9` |

与零样本相比，训练只把四格总 RMSE 再提高约 `0.03–0.04 m`；成功翻转 seed 完全相同（默认 `0–49` 仍为 `{1,2,24,31,34,36,44}`，严格 `0–49` 仍为 `{2,49}`，留出集两组翻转集合也逐 seed 一致）。这说明模型主要在已有失败尾部上做更细的残差映射，没有学会处理默认半径下的新成功样本。

训练后 history ablation 同样没有历史依赖：

| 历史输入（`0–49`） | 总 RMSE 改善 | 成功率 | 共同成功子集 | residual abs |
|---|---:|---:|---:|---:|
| 默认，完整历史 | `+0.401050 m` | `0.82 → 0.80` | `+0.002571 m` | `0.299512` |
| 默认，`current-only` | `+0.404329 m` | `0.82 → 0.80` | `+0.006849 m` | `0.301085` |
| 严格，完整历史 | `+1.087284 m` | `0.62 → 0.66` | `-0.000575 m` | `0.304501` |
| 严格，`current-only` | `+1.091601 m` | `0.62 → 0.66` | `+0.004601 m` | `0.305943` |

判定：**在剪切训练分布内继续训练只带来小幅 RMSE 增益，没有修复默认 `0–49` 的 `0.82→0.80`，也没有让 DADE 开始使用历史。** 这排除了“只是零样本 OOD”的解释；当前 DADE 的主要失效是默认到达半径下少数成功 seed 被牺牲，而严格半径的成功率提升来自失败尾部恢复。真正的海流推断仍未被训练目标约束出来，下一步应做 history-only/current-dropout、海流预测辅助损失或显式 current estimator，而不是继续在同类剪切数据上延长训练。

#### current-frame dropout 与 `current-zero` 历史依赖门禁（进行中，2026-11-01）

前一轮 history ablation 只做了 `current-only`：把短/长历史都替换成当前帧、但仍保留独立的当前帧。因此它能排除“历史窗口贡献”，不能排除“策略只看当前帧”。要检验 DADE 是否真的能从历史推断隐藏海流，必须反过来隐藏当前帧、保留真实历史。

第一版实现把 dropout 放在 `EncodedActorCritic._features()` 内，审查后判定为**无效且会破坏 PPO 口径**，已经撤回：

- 当前帧在布局中出现两次：独立的 26D 当前块，以及短/长历史窗口的最后一行。只把独立块置零时，encoder 仍可从任一历史窗口的末行精确恢复当前观测；
- dropout 在每次 `_features()` 前向时重新随机采样。rollout 收集动作 log-prob 时的 mask 与 PPO update 重新计算 log-prob 时的 mask 不同，造成同一 stored observation 对应不同策略分布；
- 因而第一版即使训练出结果，也不能证明历史依赖，不能作为有效实验。

修正后的契约：

- dropout 在 `BatchTorchHistoryVecEnv` 中执行，并**写入实际返回给 PPO 的观测张量**。rollout 与 update 读取同一份已 mask 的输入，不依赖策略内部的随机状态；
- 被 drop 的样本同时隐藏独立的当前 26D 块和短/长窗口最后一行；reset bootstrap 时整个窗口都由当前观测复制而来，因此 drop 时整个窗口置零；
- 真实历史缓冲不被改写，后面的 step 仍会正常前移；训练结束或 `eval()` 时策略本身没有额外随机 mask；
- 评估新增 `--current-ablation {none,zero}`。`zero` 与训练 mask 同语义：隐藏当前块和窗口末行，reset 首帧隐藏整个窗口；它保留之前 9/49 个真实历史帧；
- `current-only` 与 `current-zero` 必须同时报告，二者是不同方向的干预。`current-only` 去掉时序、保留当前；`current-zero` 保留时序、去掉当前。若 dropout 模型在 `current-only` 下明显退化，说明推理时确实使用了历史；若它在 `current-zero` 下仍接近完整输入且优于正常组，说明历史足以替代当前。单独看 `current-zero ≈ none` 不能区分“善用历史”和“几乎不用观测”，必须结合 base 误差与正常组交叉解释。

同 seed、同环境、同 `4096×256×40` 预算的两组训练：

| 组别 | 配置 | 训练输出 | device |
|---|---|---|---|
| 正常时变 DADE 对照 | `configs/lowlevel/odarrl_control_ppo_i2_26d_dade_transformer_hidden_fast_timevarying_seed73010.json` | `artifacts/lowlevel/i2_residual_ppo_26d_dade_transformer_hidden_fast_timevarying_seed73010` | `cuda:3` |
| 50% current-frame dropout | `configs/lowlevel/odarrl_control_ppo_i2_26d_dade_transformer_hidden_fast_timevarying_dropout05_seed73010.json` | `artifacts/lowlevel/i2_residual_ppo_26d_dade_transformer_hidden_fast_timevarying_dropout05_seed73010` | `cuda:4` |

两组 seed 均为 `73010`，使用快速时变隐藏海流环境；dropout 只改变训练时观测可见性，不改变 actor 维度、特权 critic、奖励或环境海流。两组都从同一 lineage 热启动：`artifacts/lowlevel/i2_residual_ppo_26d_dade_transformer_hidden_timevarying/model_20.pt`（SHA-256 `56ad47833910f7201b7376a56b1d8a364b005d1cc4bf8154a9a1006498e797ef`），避免随机初始化导致 encoder 冷启动差异，并直接检验 dropout 能否改变一个已经训练过的 DADE。正式结果出来后依次做 8-seed checkpoint 屏幕、50-seed 四格复核，并对同一 checkpoint 跑 `none/current-only/current-zero`。

预先登记的历史依赖判定：

1. **主判据**：dropout 组的 `current-only` 相对 `none` 必须出现明显性能损失，而正常组此前在 `current-only` 下几乎不变；否则仍不能声称策略使用了历史。
2. **可替代性判据**：dropout 组在 `current-zero` 下的总 RMSE、成功率和共同成功子集应优于正常组同一消融，说明去掉当前帧后仍能靠历史维持控制；如果两组同样崩掉，只说明当前帧不可替代。
3. **不能用单一 RMSE 改善代替机制证据**。若 `none` 的总 RMSE 更好但 `current-only` 仍与 `none` 重合，则 DADE 仍只是当前观测映射；若 `current-zero` 与 `none` 重合但两者都显著差于 base，则它主要是在低质量地忽略输入。

50-seed 四格结果（改善 = base RMSE − residual RMSE；两组均使用同 seed、同环境、同预算与同热启动 checkpoint）：

| 组别 | 条件 | checkpoint | 总 RMSE 改善 | 成功率 | 共同成功子集 | residual abs |
|---|---|---:|---:|---:|---:|---:|
| 正常 DADE | 默认，`0–49` | `model_20` | `+0.386247 m` | `0.90 → 0.90` | `-0.014957 m` | `0.292558` |
| 正常 DADE | 严格，`0–49` | `model_20` | `+0.797429 m` | `0.66 → 0.66` | `-0.015825 m` | `0.295452` |
| 正常 DADE | 默认，`50–99` | `model_20` | `+0.388179 m` | `0.94 → 0.96` | `-0.026204 m` | `0.280961` |
| 正常 DADE | 严格，`50–99` | `model_20` | `+1.008356 m` | `0.74 → 0.78` | `-0.015969 m` | `0.284964` |
| 50% dropout | 默认，`0–49` | `model_39` | `+0.413867 m` | `0.90 → 0.90` | `-0.001510 m` | `0.304358` |
| 50% dropout | 严格，`0–49` | `model_39` | `+0.849095 m` | `0.66 → 0.66` | `-0.000274 m` | `0.307948` |
| 50% dropout | 默认，`50–99` | `model_39` | `+0.455067 m` | `0.94 → 0.98` | `-0.026172 m` | `0.294180` |
| 50% dropout | 严格，`50–99` | `model_39` | `+1.031090 m` | `0.74 → 0.78` | `-0.006293 m` | `0.298897` |

dropout=0.5 相对正常组在四格总 RMSE 上分别再改善约 `0.028/0.052/0.067/0.023 m`，成功率不降，且严格 `0–49` 的共同成功子集从 `-0.0158 m` 缩小到 `-0.0003 m`。这是**真实但有限的性能增益**，仍不是历史推断证据。

机制消融（默认 `2.0 m`，`0–49`）：

| 模型 | 输入 | 总 RMSE 改善 | 成功率 | 共同成功子集 | residual abs |
|---|---|---:|---:|---:|---:|
| 正常，`model_20` | 完整 | `+0.386247 m` | `0.90 → 0.90` | `-0.014957 m` | `0.292558` |
| 正常，`model_20` | `current-only` | `+0.388423 m` | `0.90 → 0.90` | `-0.012237 m` | `0.293580` |
| 正常，`model_20` | `current-zero` | `+0.360688 m` | `0.90 → 0.92` | `-0.020005 m` | `0.149572` |
| 0.5 dropout，`model_39` | 完整 | `+0.413867 m` | `0.90 → 0.90` | `-0.001510 m` | `0.304358` |
| 0.5 dropout，`model_39` | `current-only` | `+0.416375 m` | `0.90 → 0.90` | `+0.001404 m` | `0.305977` |
| 0.5 dropout，`model_39` | `current-zero` | `+0.286346 m` | `0.90 → 0.90` | `-0.009082 m` | `0.186830` |

`50–99` 也是同一方向：正常组完整/`current-only` 为 `+0.388179/+0.391760 m`，dropout 组为 `+0.455067/+0.459226 m`，均几乎重合。也就是说，**dropout=0.5 并未让模型开始使用历史**；它改善的是当前帧到残差的映射。更反直觉的是 dropout 组的 `current-zero` 从完整输入的 `+0.413867 m` 掉到 `+0.286346 m`，降幅大于正常组的 `0.386247→0.360688 m`。这与训练时一半样本隐藏当前帧的预期相反，说明仅靠随机 dropout 不足以阻止 warm-start actor 的当前帧通路继续主导推理；必须继续做 100% dropout 或改训练目标，不能把现有增益归因于 DADE 的历史编码。

当前判定：**dropout=0.5 为“性能小幅改善、机制不成立”的负结果。** 下一步先用 100% current-frame dropout 检验历史-only 是否可行；若仍不能在 `current-only` 下产生退化、或在 `current-zero` 下维持控制，则停止同类 dropout 调参，转向海流预测辅助损失/显式 estimator 或移除 actor 的直接当前帧旁路。

#### PC 隔离式 DADE-head 历史门禁（2026-11-03，负结果）

为避免“`c_hat` 已被 actor 主干使用，但 action head 学习率过低”这一混杂因素，PC-C3 新增隔离式 DADE action head：actor 主干看到的 `c_hat` 槽位严格置零，`c_hat` 只经 action head 产生动作修正，并给 head 独立 `5e-4` 学习率。短程 `ck31` 的默认半径成功率为 `0.687500`、3D RMSE `5.385307 m`，与普通 DADE-head 的 `0.687500` / `5.443713 m` 相比没有成功率增益；严格 `1.25 m` 成功率从普通 DADE-head 的 `0.406250` 降到 `0.390625`，timeout 从 `0.593750` 升到 `0.609375`，safety 两臂均为 `0`。

闭环 `current-only` 与 `history-zero` 消融仍与完整历史逐项重合。32-seed × 512-step 诊断中，基线动作 L2 为 `0.981248`；`current-only`、`history-zero`、`history-reverse` 的平均动作变化分别为 `3.51e-5`、`4.45e-4`、`3.56e-7`，即 `0.0036%`、`0.0453%`、`0.000036%`，全部低于 `0.1%` 门禁。`history-zero` 虽使 `c_hat` 平均变化 `0.666049`，但没有把变化传到动作。

判定：**隔离 action head 仍未建立可测的历史依赖，不扩到 B 级三 seed，也不进入 C 级。** 该结果把问题进一步收窄为训练目标/输入通路，而不是 head 学习率；后续若继续，应优先尝试显式海流预测辅助损失、100% current-frame dropout 或移除 actor 当前帧旁路，并保留现有负结果作为对照。

#### 冻结 PC BC + 残差策略实验组（2026-11-03，已实现，结果待产出）

为避免把“BC 热启动”误当成“冻结 BC 残差”，PC 新增独立实验路径 `base_policy.mode="frozen_pc_bc_residual"`：

- 独立复制 PC BC encoder 与 actor，加载 `pc_bc_actor.pt` 后设置 `requires_grad=False`，并在 `policy.train()` 中保持 `eval()`；
- 动作定义为 `a = a_bc + 0.25 * delta_theta(obs)`，`base_action()` 在 `no_grad` 下执行；
- 残差 actor 与 DADE action head 输出层零初始化，训练开始时校验残差最大绝对值 `<= 1e-7`，且总动作与冻结 BC 动作一致；
- 冻结模块不进入 optimizer；训练结束比较 `model_initial.pt` 与 `final.pt` 的全部 `frozen_bc_*` state entry，要求最大绝对漂移严格为 `0`，结果写入 `training_manifest.json` 的 `frozen_bc_audit`；
- 本地 `104 passed`、ruff 通过；1-iteration CPU smoke 已确认 `initial_residual_abs_max=0.0`、frozen 参数 `requires_grad=False`、optimizer 仅含 actor 13 个和 critic 12 个可训练参数。

已排队四个实验臂，均为 `4096×256×100`、训练 seed `91430/91431/91432`：

| 实验臂 | DADE | action head | 历史 dropout | 用途 |
|---|---|---|---|---|
| `frozen_bc_residual_nodade` | 否 | 否 | 无 | 冻结残差基线 |
| `frozen_bc_residual_dade` | 是 | 否 | 无 | DADE 经 actor 拼接 |
| `frozen_bc_residual_dade_head` | 是 | 是 | 无 | DADE 独立动作头 |
| `frozen_bc_residual_dade_sidechannel10` | 是 | 是 | `side_channel=1.0` | 只遮挡最新行海流侧通道 |

评测固定 `0–63` seeds、默认 `2.0 m` 与严格 `1.25 m`，checkpoint 为 `initial/0/20/40/60/80/99`；`initial` 是零残差等价 BC 的强制对照。另在 `ck99` 上补 `current-only` / `history-zero` 闭环消融和 H0–H4/C4 压力评估。

判定口径：先过冻结与初始化门禁，再看残差是否真正非零、性能是否不低于 `initial`、safety/saturation 是否退化；DADE 臂还必须让历史消融动作敏感度达到基线动作 L2 的 `0.1%`。三 seed 不同号或跨过退化阈值即登记为不稳定/负结果，不用延长训练掩盖。当前只登记为“已实现并排队”，不能提前写正结论。

2026-11-03 服务器状态：本地 227 个 low-level 测试和 ruff 通过；服务器同一 venv 下 227 个 low-level 测试通过；GPU2 的 `4096×256×1` frozen smoke 产出 `initial_residual_abs_max=0.0`、`frozen_bc_parameter_max_abs_drift=0.0`、`frozen_bc_parameters_require_grad=false`；统一扩训队列已用 `setsid` 启动。截至 16:15，`frozen_bc_residual_nodade_seed91430` 已完成 100 iterations / `104,857,600` env-steps，正式 `frozen_bc_audit` 记录 12 个冻结参数与 14 个冻结 state entry、最大绝对漂移 `0.0`、`requires_grad=false`；`frozen_bc_residual_dade_seed91430` 和 `frozen_bc_residual_dade_head_seed91430` 正在训练，`sidechannel10_seed91430` 与两个后续 seed 仍在主队列中等待。当前没有性能正结论；B 级缺失两 seed 由 post-queue gapfill 补齐。

同日发现并修复一个影响 DADE-head 结论的 optimizer 分组缺陷：旧 `_configure_optimizer_learning_rates()` 在配置没有显式 `learning_rate_dade_head` 时，把 `actor_dade_head.*` 从 actor 参数组中剔除却没有创建独立 head 组，导致 head 参数完全未参与优化。服务器 checkpoint 的 `actor_dade_head.2.weight/bias` 从 `model_0` 到 `model_80` 的最大绝对值始终为精确 `0.0`，确认这是实现缺陷，不能解释为“DADE head 学不会”。修复后无独立学习率时并回 actor 组，独立学习率时保留 `dade_head` 组；manifest 写入 optimizer 参数组计数，新增 `tests/test_lowlevel_ppo_optimizer.py` 防止可训练参数遗漏或重复。受影响的 `dade_head` / `dade_head_dropout05` / frozen head / sidechannel10 旧 run 保留但标记为作废，不能用于 DADE 负结论；no-DADE、普通 DADE（无 action head）和显式 `learning_rate_dade_head=5e-4` 的 history-required 不受影响。正确重跑必须使用新的输出根。

进一步审计发现，旧 frozen `dade_head` 并未隔离 `c_hat`，旧 frozen `sidechannel10` 也没有启用 action head；B 级 `dade_head` 三个 seed 还跨越了不同训练脚本版本。因此这些目录已分别写入 `INVALID_FOR_DADE_CONCLUSIONS.json`，保留原 checkpoint 和日志，但禁止用于任何 DADE 正负结论。正确的 action-head 实验将统一使用 `dade_action_head_isolated=true` 与显式 `learning_rate_dade_head=5e-4`。

为补足“冻结 BC + 残差策略”的有效实验组，新增 S8b 高容量 corrected 组：三臂 × 三 seed × `4096×256×100`，统一 `actor_lr=3e-4`、`residual_scale=1.0`；DADE 臂使用隔离 action head 和显式 `5e-4`。三臂分别为 no-DADE 同预算基线、隔离 DADE-head、pose-history + 最新侧通道 dropout 的 history-only。输出根为 `artifacts/lowlevel/pc_server_scale_frozen_bc_residual_v2_20261103/`，队列为 `scripts/run_pc_frozen_bc_residual_v2_20261103.sh`。训练后必须检查冻结参数漂移为零、optimizer 中存在 `dade_head` 组、`model_10` 的 head 权重已非零，再进入固定 seed 双半径与历史消融评测。该组与 S8 第一版分开统计，禁止用旧作废目录替代。

本地三臂 1-iteration CPU smoke 均通过，初始化残差为零、冻结漂移为零；两个 head 臂的 optimizer 都有非零 `dade_head` 组，更新一次后 head 最大绝对权重为 `0.12497`。部署到服务器后，独立队列使用 GPU4/5 并行训练，避免阻塞在已占满的 GPU1。截至 17:10，no-DADE `seed91430` 已完成 100 iterations / `104,857,600` env-steps，manifest 状态为 `completed`，冻结审计仍为零漂移；隔离 DADE-head 约 57/100，history-only 约 29/100。已对 no-DADE 的 `initial/0/40/60/80/99` 启动固定 64-seed 双半径早期评测，`ck20` 复用已完成的 interim 结果。当前仍没有性能正结论。完整 low-level 回归 `252 passed`，ruff 与 `git diff --check` 通过。

同时修复了汇总脚本的 arm 推断：corrected 配置名包含旧臂子串，原先会被错误归并到 `frozen_bc_residual_nodade` / `frozen_bc_residual_dade_head` / `frozen_bc_residual_dade_history_only`，导致 S8b 的跨臂配对和 checkpoint 配对失真。现在三个 corrected 名称及其 H1/H2/C4 v2 名称均在通用前缀匹配之前精确识别，并新增回归测试；该修复不改变任何旧 checkpoint 或旧 summary。

为扩大训练规模和补足高速/强扰动能力，新增 S10b hard-env v2：`{H1,H2,C4} × 3 seed × 4096×256×100`，配置分别为 `odarrl_control_ppo_pc_{h1_speed_1p8_2p0,h2_speed_2p0_2p25,c4_stress}_frozen_bc_residual_nodade_lr3e-4_scale1p0_*.json`。三臂继续使用同一冻结 BC，但改为 corrected 的 `actor_lr=3e-4`、`residual_scale=1.0`，避免旧硬环境队列 `1e-5 / 0.25` 因残差权限不足而提前失败。队列为 `scripts/run_pc_frozen_bc_residual_hardenv_v2_20261103.sh`，复用 S8b 队列锁和 GPU4/5 锁，因此会在 S8b 之后自动开始，不覆盖旧硬环境目录。训练后先审计冻结漂移、初始化残差和 optimizer 分组，再分别评测 H1/H2/C4 内分布，并在默认/严格半径上检查跨环境退化。三组本地 1-iteration smoke 均给出 `initial_residual_abs_max=0`、冻结漂移 `0`、`requires_grad=false`、`actor=94018 / critic=94141`；服务器队列已以 `setsid` 启动并在等待 S8b 锁，尚无性能结论。

为补足服务器上仍未跑的基础实验，2026-11-04 新增 C2/H3 基础队列 `scripts/run_pc_remaining_basics_20261103.sh`，服务器 PID `3745939`。C2 使用 `d_z=32` 的 C0/C1/C2 混合 BC，训练 `c2_nodade/c2_dade × 3 seed × 2048×256×32`，训练 seed 为 `91205/91206/91207`；H3 使用 corrected 冻结 BC 残差基线 `frozen_bc_residual_nodade_h3_lr3e-4_scale1p0 × 3 seed × 4096×256×100`，训练 seed 为 `91430/91431/91432`。队列并行跑了 C2/C3/C4/H1–H4 teacher 与 C2/C3/H3 BC-only 基线，再自动评测 C2 mild/clean 和 H3 分布内/默认/严格/H4 压力。C2 teacher 64-seed 已出：mild success `1.0000`、三维路径 RMSE `1.8852 m`；clean success `1.0000`、三维路径 RMSE `1.6231 m`。目前只证明基线可运行，C2 PPO 与 H3 残差尚无性能结论。

`seed91430` 的 corrected no-DADE 已完成固定 64-seed 双半径早期评测（单 seed，不能代替三 seed 结论），`initial` 与 `ck0` 完全一致，说明零残差基线没有被训练前保存污染：

| tag | checkpoint | success | timeout | 3D RMSE (m) | residual/base L2 | Δ success vs initial | Δ RMSE vs initial (m) |
|---|---:|---:|---:|---:|---:|---:|---:|
| default | initial | 0.6875 | 0.3125 | 5.3919 | 0 | 0 | 0 |
| default | 20 | 0.6875 | 0.3125 | 5.4014 | 0.002923 | 0 | +0.009497 |
| default | 40 | 0.6875 | 0.3125 | 5.3992 | 0.003523 | 0 | +0.007286 |
| default | 60 | 0.6875 | 0.3125 | 5.3966 | 0.003316 | 0 | +0.004673 |
| default | 80 | 0.6875 | 0.3125 | 5.4006 | 0.005836 | 0 | +0.008681 |
| default | 99 | 0.6875 | 0.3125 | 5.3905 | 0.006732 | 0 | -0.001393 |
| strict125 | initial | 0.40625 | 0.59375 | 7.1178 | 0 | 0 | 0 |
| strict125 | 99 | 0.40625 | 0.59375 | 7.1152 | 0.006427 | 0 | -0.002623 |

残差幅值确实随训练增长到 BC 动作 L2 的约 `0.67%`，但默认和严格半径的成功/timeout 计数逐项不变；中间 checkpoint 的 RMSE 点估计均比初始 BC 略差，只有 `ck99` 出现毫米级改善。**初步判定：单 seed 上没有证据表明该残差足以改变闭环能力，不能把“训练 rollout success 较高”写成固定 seed 性能提升。**需要等待三训练 seed 与 DADE 臂，若方向不变，应转向有监督残差/DADE 热启动，而不是继续单纯增加 PPO 步数。

对应的旧 frozen no-DADE 残差也已做 16-seed 幅值审计：`ck99` 相对 `initial` 的成功率均为 `0.625`、timeout 均为 `0.375`，3D RMSE `5.64738 → 5.63210 m`；残差动作绝对均值 `0.002323`、最大 `0.004352`、相对 BC 动作 L2 仅 `0.006211`，平均舵角修正约 `0.001337 rad`、艉舵 `0.000185 rad`、RPM `3.98`。因此旧配方 `actor_lr=1e-5`、`residual_scale=0.25` 下残差虽非零，但没有达到改变能力的量级，登记为初步负结果。最新评测器已加入残差幅值和物理修正指标，后续正结论必须先证明残差有效且性能门禁不退化。

#### history-only DADE：移除动态状态捷径（2026-11-03，已实现，结果待产出）

现有 DADE 的负结果不能继续用 dropout/head 学习率解释。`DualHorizonEncoder` 虽然不读取顶层 `state`，但历史 token 仍包含 `u/v/w/p/q/r`；最新历史行足以反推当前海流。辅助损失只遮挡最新行的侧通道，没有遮挡同一行的动态状态，因此 `c_hat` 可以变化而动作不依赖历史。

新增实验臂把 actor-DADE 输入改为 `pose + side_channel`：

- `pc.dade_state_mode="pose"` 只保留 `x,y,z,phi,theta,psi` 和海流侧通道；旧配置默认 `full` 不变；
- 训练时 `current_frame_dropout_prob=1.0`、`mode="side_channel"`，最新侧通道始终不可见；
- 辅助损失 `aux_current_coef=0.5`、`steps=8`，从更早的 pose-history 预测当前侧通道；
- 使用隔离 DADE action head（`5e-4`）进入残差，避免 64D 特征被 actor 主干淹没；
- 两个机制 pilot：`dade_history_only` 与 `frozen_bc_residual_dade_history_only`，均三 seed、`4096×256×40`。

新增 `scripts/check_pc_dade_history_gate.py`：在 `ck39` 上跑 `32 seed × 512 step` 的 `current-only` / `history-zero` 诊断，任一臂任一 seed 的动作变化低于基线动作 L2 的 `0.1%` 即门禁失败。无论门禁结果都跑默认/严格半径的 `ck0/39`；只有门禁通过才自动追加 `h1/h2/h4/c4` 压力评测。

验证状态：本地 `230 passed`，服务器同 venv `230 passed`；服务器 history-only 1-iteration smoke 完成；`scripts/run_pc_history_only_dade_20261103.sh` 已用 `setsid` 启动，PID `2032209` 正在等待主扩训队列锁。当前不能写“DADE 已学会使用历史”，必须等三 seed 消融门禁与双半径评测。

#### H1/H2/C4 硬环境真实训练（2026-11-03，已排队）

此前 H0–H4/C4 只是 C3 checkpoint 的最终压力评测，不能证明策略通过训练获得了高速或强扰动能力。新增三个冻结 BC 残差训练臂，均为三 seed、`4096×256×100`、`residual_scale=0.25`、无 DADE：

| 实验臂 | 环境 | 目的 |
|---|---|---|
| `frozen_bc_residual_nodade_h1` | H1 `1.8–2.0 m/s` | 高速适配，检查 saturation 与跟踪 |
| `frozen_bc_residual_nodade_h2` | H2 `2.0–2.25 m/s` | 更高速度上限 |
| `frozen_bc_residual_nodade_c4` | C4 强时变流、强剪切、强传感器 | 强扰动鲁棒性 |

评测先跑训练分布的 `ck0/50/99`，再跑默认 `2.0 m` 与严格 `1.25 m` 的 `ck99`；固定 32 seeds，必须同时报告 success、timeout、safety、material saturation、三维 RMSE、speed/yaw RMSE。若 H1/H2 主要受执行器饱和限制，登记系统边界；C4 若 safety 退化则淘汰。

本地 ruff、JSON 解析和 H1 1-iteration CPU smoke 已通过；服务器配置解析、bash 语法和 H1 smoke 已通过。硬环境队列曾以 PID `2042069` 等待，但为避免旧版汇总把 H1 误作 H2/C4 基线，已停止该未执行进程并重启修正版 PID `2102665`；新评测加入 `initial`，并将 `ck0/50/99` 与各自 `initial` 配对。尚无性能结论。

#### history-required DADE：阻断 actor 当前动态状态旁路（2026-11-03，已完成，负结果）

此前的 history-only DADE 仅移除历史 token 中的动态列，但 actor 顶层 `state` 仍直接包含 `p/q/r/u/v/w`，策略可以绕过 DADE 从当前状态获得同等信息。PC 新增 `actor_state_mode="pose"`：冻结 BC 仍读取完整状态以保持先验，而可训练残差 actor 与 DADE 顶层状态只保留姿态/位置；同时新增 `current_frame_dropout_mode="history"`，训练时 100% 抹掉最新历史 token，迫使 DADE 从更早的完整历史推断当前动态状态。DADE action head 继续隔离、零初始化，并用 `current_estimate` 辅助损失训练表示。

新增配置 `odarrl_control_ppo_pc_c3_random_realistic_frozen_bc_residual_dade_history_required_dz64_gamma995_warmup5_critic3e-4_fixed_rewardscale001.json` 和队列 `scripts/run_pc_history_required_dade_20261103.sh`。该 pilot 为三 seed、`4096×256×40`，`ck39` 做 `32 seed × 512 step` 的 `current-only`、`history-zero`、`history-reverse` 诊断；默认门禁通过才追加 H1/H2/H4/C4，且始终跑默认/严格半径与闭环历史消融。

本地完整 low-level 回归 `239 passed`、ruff 通过；服务器同 venv `241 passed`；本地与服务器 1-iteration CPU smoke 均通过，冻结审计仍为 `drift=0`、`requires_grad=false`。三 seed 训练和评测均已完成，但门禁为 **fail**。

三个 seed 的 `ck39` 历史机制诊断结果一致为负：

| Seed | base action L2 | `current-only` 动作变化 | `history-zero` 动作变化 | `current-only` 比例 | `history-zero` 比例 |
|---|---:|---:|---:|---:|---:|
| `91430` | `0.97784` | `4.02e-5` | `6.93e-4` | `0.0041%` | `0.0709%` |
| `91431` | `0.97780` | `4.12e-5` | `7.40e-4` | `0.0042%` | `0.0757%` |
| `91432` | `0.97826` | `4.31e-5` | `5.96e-4` | `0.0044%` | `0.0609%` |

两种模式都低于预注册的 `0.1%` 门禁。`history-zero` 使 `c_hat` 产生明显变化，但传不到动作；其残差动作 L2 只有约 `0.003`，约为 base action 的 `0.3%`。闭环结果也与 `ck0` 几乎重合：default 两模式均为成功率 `0.6875`、RMSE `4.922–4.924 m`，strict125 为 `0.4688 / 6.508 m`，没有可测的历史收益。结论：**阻断顶层动态状态旁路后，DADE 仍未学会依赖较早历史；该臂不再扩到更多 seed，也不再继续 dropout/head 调参。** 后续若继续 DADE，应改为显式状态估计目标或可证明的信息瓶颈，而不是只增加训练预算。

产物：`artifacts/lowlevel/pc_server_scale_history_required_dade_{gate,diagnostics,summary}_20261103/`；服务器门禁 `gate.json` 与本地汇总一致，三 seed 均为 `passed=false`。

### 14.4.10 O 臂 residual-only 超参/奖励/更新频率复查（负结果）

为回答“O 臂是否只是奖励/超参不对”，在 26D RPM-only O 臂上补了四组 residual-only 复查，均使用 online PD/PI 基础策略、4096 环境、相同 50-seed 评估口径。结果如下（三维 RMSE 改善为正表示 residual 更好）：

| 变体 | checkpoint | 默认 2.0 m `0–49` | 严格 1.25 m `0–49` | 默认 `50–99` | 严格 `50–99` | 无海流 | 判定 |
|---|---|---:|---:|---:|---:|---:|---|
| conservative（lr `5e-5`、noise `0.05`、clip `0.1`） | `model_10` | `+0.001128 ± 0.000268 m` | `+0.001073 m` | `+0.000858 m` | `+0.000830 m` | `+0.000251 m` | 统计可见但仅 `~0.07%`，操作上可忽略 |
| explore（lr `3e-4`、noise `0.15`、clip `0.15`） | `model_39` | `-0.005037 m` | `-0.004601 m` | `-0.006380 m` | `-0.006088 m` | 报告 | 明确退化 |
| terminal reward（`w_3d=3`、`goal=200`、`timeout=-100`） | `model_5` | `-0.000983 m` | `-0.000906 m` | 报告 | 报告 | 报告 | 退化 |
| frequent update（80 iter × 128 steps） | `model_79` | `-0.004425 m` | `-0.004254 m` | 报告 | 报告 | `-0.001506 m` | 退化 |

conservative 变体的 residual abs 为有海流 `0.000746`、无海流 `0.000674`，只下降约 10%，没有形成明显的海流条件化；explore 变体为 `0.00974` 对 `0.00864`，同样没有条件化证据。四组都没有通过“有可重复正收益且 residual 随海流显著变化”的门禁。

判定：**O 臂 residual-only 的奖励/超参/更新频率复查仍为负结果；不启动 O1/O2 encoder 阶段。**如果后续还要做 O 臂，应先解决基础控制器与残差权限的匹配问题（例如 residual 只作用于 RPM、重新设计终端奖励、加入 current-conditioned 课程），而不是直接加 DADE。

### 14.4.11 五 seed 扩训、冻结残差负结果与自动评测修复（2026-11-04）

S10c 的 C2/H3 三 seed 队列仍在运行，并已确认 C2 no-DADE `seed91205` 与 H3 frozen-BC residual `seed91430` 完成。为避免等待串行，另行启动独立扩训队列 PID `3779144`，在空闲 GPU2/4/5 增加 C2 `91208/91209`、H3 `91433/91434`、C3 frozen residual `91433/91434` 以及 H1/H2/C4 frozen residual `91433/91434`。DADE 不再新增实验臂，C2 的 no-DADE/DADE 只做计划内同 seed 配对。

扩训完成后自动执行冻结审计、固定 seed 闭环评测、两 seed 扩展汇总，并等待原队列释放锁后生成五 seed 联合汇总。远端由 `setsid nohup` 托管，本机待机不会停止。脚本为 `scripts/run_pc_basics_scale_extension_20261103.sh`，日志为 `logs/pc_basics_scale_extension_20261103/queue.log`。

本次修复两个自动汇总缺陷：C2 直采 PPO 没有 `model_initial.pt` 时，评测跳过可选 `initial`，汇总回退到首个已保存 checkpoint `0` 并保留 requested/effective 字段；`history-*` 消融目录不再被误解析为训练 seed `0`，且普通性能汇总默认排除消融结果。本地完整回归 `585 passed`，服务器汇总回归 `15 passed`。

已有三 seed corrected 冻结残差结果已足以否定“同一 PPO 配方继续加预算”的路线：H1 `ck99` 成功率 `0.875→0.875`、RMSE `+0.0089 m`；H2 `0.9063→0.9063`、RMSE `+0.0409 m`；C4 `0.250→0.250`、RMSE `+0.0470 m`，均为不改善或轻微退化。五 seed 扩展用于稳健性确认；若结论不变，应转向教师残差监督/DAgger 或重新设计残差权限。

早鸟评测（前两个训练 seed，仅作方向判断）：C2 no-DADE `seed91205` mild `ck0→final` 成功率保持 `1.0000`、RMSE `1.9509→1.9321 m`，clean 为 `1.7248→1.7052 m`；`seed91208` 对应为 `1.9738→1.9443 m` 与 `1.7538→1.7198 m`。H3 frozen residual `seed91430` 从 `ck0` 的 success `0.8438`、RMSE `9.9935 m` 退化到 `ck99` 的 `0.8125`、`11.1107 m`，`seed91431` 同样为 `0.8438→0.8125`、`9.9935→11.1117 m`。当前证据表明 C2 基础复现稳定，而 H3 的 PPO 残差配方跨 seed 一致无效。五 seed 结果由最终收尾器统一写入 `pc_server_*_summary_all5_20261103/`。

### 14.4.12 冻结 BC + 教师残差监督实验组（2026-11-04）

为绕开“PPO 从零残差探索不到教师修正”的问题，新增教师残差监督实验组。标签直接使用

```text
(teacher_action - frozen_BC_action) / residual_scale
```

动作均为 PC 归一化 `[rudder, stern, rpm]`。BC encoder/actor 全程冻结，残差网络输出层零初始化；保存的 `model_initial.pt` 严格复现 BC，`final.pt` 为监督残差策略。该 checkpoint 同时可被现有 PC 闭环评测器直接加载，并可通过 `train_batch_lowlevel_ppo.py --warmstart-residual` 作为后续 PPO 初值。

覆盖 C3、H1、H2、H3、C4 五个环境，每个环境采集 `16×4096` 条教师轨迹、训练 `91430/91431/91432` 三个 seed。每个 run 自动完成冻结审计、分布内 `initial/final` 固定 seed 评测；H1/H2/H3/C4 额外在 C3 默认/严格速度上做最终 checkpoint 的跨环境检查，并按 arm 生成配对汇总。所有产物写入 `artifacts/lowlevel/pc_teacher_residual_demos_20261104/`、`pc_supervised_residual_20261104/`、`pc_supervised_residual_eval_20261104/` 与 `pc_supervised_residual_summary_20261104/`。

本地验证：真实 C3 demo 单 epoch smoke 的 validation loss 为 `0.007150 → 0.006935`，冻结漂移 `0`；1-seed 闭环加载成功；PPO residual warm-start 的 1-iteration smoke 完成，初始残差幅值 `0.023729`。

服务器执行结果（队列 PID `3823406`，2026-11-04 11:48 结束）：

- 教师数据 `5/5`、监督训练 `15/15` 完成，`frozen_bc_parameter_max_abs_drift=0`，审计失败 `0`；自动评测 `84` 个 run-level summary，队列退出码 `0`；
- 分布内 `initial→final`：H2 success `0.9062→0.9583`、RMSE `6.6946→4.0056 m`；H3 `0.8438→0.9792`、`9.9935→3.3118 m`；H1 `0.8750→0.8750`、`5.6609→5.3578 m`；C4 `0.2500→0.2396`、`8.4932→8.6479 m`；
- 跨环境检查显示强过拟合：H3 在默认环境 success `0.6875→0.4479`、RMSE `4.9269→12.5591 m`，严格 `1.25 m` 为 `0.4688→0.2812`、`6.5153→13.7405 m`；H1/H2 同样退化；
- 判定：**监督教师残差是分布内有效、跨环境失真的局部修复器。** 不再追加 DADE；若继续推进，应优先多环境混合教师标签或 DAgger，并在保存策略时加入跨环境门禁。
- 汇总入口：`artifacts/lowlevel/pc_supervised_residual_summary_20261104/index.json`。

### 14.4.13 六环境混合教师残差与单次 DADE 结构对照（2026-11-04）

14.4.12 已证明单环境教师监督在分布内有效、跨环境明显过拟合，因此新增独立的六环境混合残差队列。教师数据覆盖 C2 mild、C3 realistic、H1、H2、H3、C4 strong，每个环境采集 `16×4096` 条轨迹，并保留部署所需的 `current_estimate_body` 流估计历史。六份数据合并后分别训练：

- `frozen_bc_residual_nodade_multienv_lr3e-4_scale1p0`
- `frozen_bc_residual_dade_multienv_pose_lr3e-4_scale1p0`

两臂使用相同数据、冻结 BC checkpoint、训练 seed `94030/94031/94032` 和教师残差标签；BC encoder/actor 保持冻结，残差输出层零初始化。DADE 臂只做一次结构化验证：`dade_state_mode=pose`、隔离动作头、幅值 `0.03` 的 DADE 热启动、当前估计辅助损失 `0.5`，用于阻断直接速度旁路并检查历史依赖，不继续扩展 DADE 超参搜索。

队列启动后自动执行：冻结漂移审计、`initial/final` 闭环评测（默认/严格 `1.25 m` 各 64 seed，H1/H2/H3/C4 各 32 seed）、no-DADE 对 DADE 的配对汇总、C3 默认/严格速度 DADE 历史消融及固定门禁。远端由 `setsid nohup` 托管，主机待机不影响执行。全量队列 PID `3873284`，脚本 `scripts/run_pc_multienv_residual_20261104.sh`，日志 `logs/pc_multienv_residual_full_20261104/queue.log`，产物根目录 `artifacts/lowlevel/pc_multienv_supervised_residual_*_full_20261104/`。

smoke 已完成混合采样、合并、两臂训练、严格 checkpoint 加载、闭环评测和历史诊断；冻结漂移 `0`，DADE/动作头/当前估计头状态完整。smoke 的历史门禁为负结果：`current-only` 动作敏感性约 `8.6e-5`，低于 `1e-3`；`history-zero` 约 `1.4e-3`。因此 DADE 只保留这一组全量结构实验，失败时作为负证据记录，不再追加调参臂。

现有冻结 BC + PPO 残差训练组由 S10c/S10d 覆盖：C2、C3、H1、H2、H3、C4 的 corrected frozen-BC residual 训练，五 seed `91430–91434`（C2 为 `91205–91209`），训练后自动做冻结审计、分布内/跨环境评测和五 seed 联合汇总；收尾器 PID `3792998` 会等待两组队列释放后统一生成 all-5 报告。

### 14.4.14 PC20 高速标定、H0 基线与 Teacher/BC 阶梯（2026-11-04）

此前高速实验只有 H3 BC-only 与零散 teacher 摘要，缺少同 seed、同包线的 H0 参照，也没有把海流方向性和制动距离纳入可达性判断。本节补齐了这部分，不再把“空场最大速度”误写成“顶流可控”。

**标定协议。** 使用同一 `REMUS100` 本体、PC20 `±20°` 舵/艉舵、`150°/s` 舵速率、`0–1525 rpm` 和 `dt=0.02 s`，对 H0–H4 分别执行：

- 9 点 rpm–稳态速度曲线；
- 8 方向均匀流扫描（`0°` 为顺流、`180°` 为顶流），流速覆盖 `0/0.1/0.25/0.4/0.5/0.55 m/s`；
- 20° 正负舵阶跃和艉舵阶跃；
- 从各阶段高速命令收油，到地速降至 `0.2 m/s` 的滑行时间与距离。

各阶段真实海流上界并非统一 `0.5 m/s`：H0/H1/H2/H3/H4 分别为 `0.55/0.50/0.25/0.10/0.00 m/s`。H4 是不可达顶流的空场压力测试，不能写成“H4 顶流可控”。

| 阶段 | 高速命令 | 静态最大地速 | 海流上界 | 顶流余量 | 最小方向余量 | 收油至 0.2 m/s 距离 | 时间 |
|---|---:|---:|---:|---:|---:|---:|---:|
| H0 | 1.8 | 2.5536 | 0.55 | 0.2036 | 0.2036 | 9.80 m | 18.20 s |
| H1 | 2.0 | 2.5536 | 0.50 | 0.0536 | 0.0536 | 10.34 m | 18.46 s |
| H2 | 2.25 | 2.5536 | 0.25 | 0.0536 | 0.0536 | 10.96 m | 18.74 s |
| H3 | 2.4 | 2.5536 | 0.10 | 0.0536 | 0.0536 | 11.30 m | 18.88 s |
| H4 | 2.5 | 2.5536 | 0.00 | 0.0536 | 0.0536 | 11.53 m | 18.98 s |

判定：**H0–H4 的静态和配置海流上界均可达，但 H1–H4 只剩约 `0.054 m/s` 顶流余量；高速失速保护必须依赖前馈制动，不能依靠收油滑行。** 从 2.5 m/s 收油到 `0.2 m/s` 需要约 `11.53 m / 18.98 s`，因此高速终端的提前制动距离和传感器延迟必须一起验收。

**同 seed 闭环阶梯。** 固定评估 seeds `0–31`，同一共享 BC checkpoint，分别评估 H0–H4 的 teacher 和 BC-only。teacher 是同一控制器参考，不是性能上界；H0 上 BC 甚至略高于 teacher，说明不能只用 teacher 排名判断策略好坏。

| 阶段 | Teacher success | BC success | Teacher 3D RMSE | BC 3D RMSE | BC/Teacher RMSE | BC 饱和率 | BC 安全终止 |
|---|---:|---:|---:|---:|---:|---:|---:|
| H0 | 0.7812 | 0.8125 | 5.08 | 4.72 | 0.93× | 0.2320 | 0 |
| H1 | 0.9688 | 0.8750 | 2.57 | 5.66 | 2.20× | 0.4065 | 0 |
| H2 | 1.0000 | 0.9062 | 1.75 | 6.69 | 3.83× | 0.4236 | 0 |
| H3 | 1.0000 | 0.8438 | 1.77 | 9.99 | 5.64× | 0.3892 | 0 |
| H4 | 1.0000 | 0.4688 | 2.94 | 18.88 | 6.42× | 0.3091 | 0 |

以 H0 BC 为参照的 provisional 诊断门禁（缺 overshoot 指标，因此不冒充正式 PC-G5 通过）：

- H1：success 实际提高 `0.0625`，三维 RMSE 为 H0 的 `1.20×`，但 tracking 误差最大为 `2.11×`，且 saturation 增加超过 `0.05`；`diagnostic_fail`；
- H2：三维 RMSE `1.42×`、tracking 最大 `2.89×`、saturation 增加超过 `0.05`；`diagnostic_fail`；
- H3：三维 RMSE `2.12×`、tracking 最大 `3.39×`、saturation 增加超过 `0.05`；`diagnostic_fail`；
- H4：teacher success `1.0`、BC success `0.4688`，只作为包线压力证据，不参与训练通过/失败判定。

结论：**高速阶梯已从“能跑”推进到可审计的 teacher/BC 差距证据，但 BC-only 不具备 H1–H3 课程通过条件，H4 明确是压力测试。** H1 的 teacher success 低于 H2/H3，说明阶梯本身也不单调；后续应优先用多环境教师残差/DAgger 修复高速跟踪，而不是继续给 BC-only 或 DADE 加预算。

复现实验入口：

- 标定：`scripts/calibrate_pc_high_speed.py`，产物 `artifacts/lowlevel/pc_high_speed_calibration_v2_20261104/calibration.json`；
- H0–H4 BC/teacher 同 seed 评估：`scripts/run_pc_high_speed_bc_baselines_20261104.sh`，产物 `artifacts/lowlevel/pc_server_remaining_baselines_20261103/bc_h{0,1,2,3,4}_seed0_31/summary.json`；
- 汇总：`scripts/summarize_pc_high_speed_20261104.py`，产物 `artifacts/lowlevel/pc_high_speed_report_20261104/high_speed_report.{json,md}`；
- 本地/服务器测试：`tests/test_pc_high_speed_calibration.py`、`tests/test_pc_high_speed_report.py` 均通过；标定代码包含方向扫描和收油滑行，报告代码保留“缺 H0 时不判定”的显式状态。

### 14.4.15 高速残差扩训、mixed-domain、DAgger 与速度×海流网格（2026-11-04，已完成）

六环境等权混合解决了单环境残差的跨环境崩溃，但分布内 H1/H2/H3 的提升不能证明 held-out 泛化：H1/H2/H3/C4 都参与训练，且 H1/H2/H3 的 material saturation 已接近 `0.48–0.50`；strict125 和 C4 仍退化。五个训练 seed `94030–94034` 的 no-DADE 汇总已经完成，结果如下（final 相对 frozen-BC initial；success/RMSE 为 `mean +/- std`）：

| Tag | Success | 3D RMSE | Saturation | Δsuccess | ΔRMSE | Δsaturation |
|---|---:|---:|---:|---:|---:|---:|
| default | `0.7219 +/- 0.0301` | `5.1374 +/- 0.1877 m` | `0.1355` | `+0.0344` | `-0.2545 m` | `+0.0043` |
| strict125 | `0.3719 +/- 0.0373` | `7.5631 +/- 0.1532 m` | `0.1238` | `-0.0344` | `+0.4453 m` | `+0.0063` |
| H1 | `0.9000 +/- 0.0713` | `4.4950 +/- 2.0265 m` | `0.4709` | `+0.0250` | `-1.1659 m` | `+0.0644` |
| H2 | `0.9437 +/- 0.0640` | `4.2273 +/- 2.4086 m` | `0.4856` | `+0.0375` | `-2.4673 m` | `+0.0620` |
| H3 | `0.9812 +/- 0.0280` | `2.4551 +/- 1.0930 m` | `0.4973` | `+0.1375` | `-7.5384 m` | `+0.1081` |
| C4 | `0.2125 +/- 0.0261` | `8.5388 +/- 0.0710 m` | `0.2573` | `-0.0375` | `+0.0456 m` | `+0.0160` |

因此五 seed 证据仍只支持“混合分布内拟合改善”，不支持 held-out 泛化通过：strict125 成功率均值下降 `0.0344`、RMSE 劣化 `0.4453 m`；C4 成功率下降 `0.0375`；H1/H2/H3 均提高残差饱和，其中 H3 增加 `0.1081`。H3 从 `0.8438 / 9.9935 m` 改善到 `0.9812 / 2.4551 m` 是稳定的分布内收益，但不能替代留出速度×海流组合。

- 五 seed 汇总：`artifacts/lowlevel/pc_multienv_nodade_5seed_summary_20261104/summary.{json,md}`。

因此并行准备两组后续实验，均不重启、不覆盖现有全量队列：

**A. H1–H3 高速残差扩训（已启动）。** 使用 no-DADE 多环境教师残差的 5 个 seed `94030–94034` 作为 `--warmstart-residual`，分别在 H1/H2/H3 上训练 `4096×256×100`，共 15 个 run；每个 run 先做冻结 BC 漂移、非零残差加载和 warm-start SHA 审计，再跑 `initial/final × 32 seeds` 的 H1/H2/H3 闭环，以及 H3 final 在 H4 的固定压力测试。门禁仍按 H0 BC 的 PC-G5 provisional 阈值执行，并额外保留 final-vs-initial 配对差值。

- 队列脚本：`scripts/run_pc_high_speed_residual_20261104.sh`；服务器 PID `3935752`；
- 1-iteration smoke 已通过：三阶段 `initial_residual_abs_max=0.29736048`、frozen BC drift `0`、严格 warm-start 加载成功；
- 汇总脚本：`scripts/summarize_pc_high_speed_20261104.py --residual-summary ... --require-pass`；
- 产物：`artifacts/lowlevel/pc_high_speed_residual_{ppo,eval,summary,report}_20261104/`。

高速残差队列已完成 15 个训练 run（H1/H2/H3 × 5 seeds）和全部自动评测，训练审计无失败、冻结 BC 漂移全为 `0.0`。最终报告门禁为 **fail**：

| Stage | Success | 3D RMSE | Saturation | Final-vs-initial Δsuccess | Final-vs-initial ΔRMSE | H0 BC relative gate |
|---|---:|---:|---:|---:|---:|---|
| H1 | `0.9500` | `2.92 m` | `0.4705` | `+0.0500` | `-1.58 m` | **diagnostic_fail** |
| H2 | `0.9437` | `4.70 m` | `0.4695` | `0.0000` | `+0.47 m` | **diagnostic_fail** |
| H3 | `0.9625` | `3.03 m` | `0.4849` | `-0.0188` | `+0.58 m` | **diagnostic_fail** |
| H4 | `0.9437` | `3.81 m` | `0.4907` | n/a | n/a | pressure-only |

H1 的残差训练有稳定正收益，H2/H3 没有稳定的 final-vs-initial 改善；三者仍然主要靠接近上限的执行器权限换取成功率。相对 H0 BC，H1/H2/H3 的 saturation 分别高约 `0.239/0.238/0.253`，pitch tracking 最大约 `2.19/2.79/3.39×`，因此都不能通过 PC-G5 provisional 门禁。H4 压力测试成功率达到 `0.9437`，但 saturation `0.4907` 已接近 `0.5`；overshoot 仍未测量。报告见 `artifacts/lowlevel/pc_high_speed_residual_report_20261104/high_speed_report.{json,md}`。

**B. 速度×海流解耦 mixed-domain（已启动）。** 训练域固定为 C2（低速弱流）、C3（低速强流）、H2（高速弱/中流）、H3（高速弱流）；**H0/H1 的中高速强流和 C4 的强剪流/强传感器全部留出**。C4 不再等权混入训练，只作压力测试。这样 H1 是主要的“高速+强流”零样本组合测试，H0 提供另一个强流留出点，C4/H4 只报告压力证据。

- 队列脚本：`scripts/run_pc_mixed_domain_residual_20261104.sh`；服务器 PID `3943998`；
- 训练 seed `94030–94034`，每 seed 独立训练监督残差；合并来源仅为 C2/C3/H2/H3；
- 自动审计必须看到四个允许的 config，任何 H0/H1/C4 训练数据泄漏即失败；
- 首次自动评测暴露了 arm→training-config 映射缺失（训练本身五 seed 全部完成）；已补齐 mixed-domain 映射，并在评测启动前显式拒绝未知 arm，随后仅重跑评测、不重训；
- 闭环评测：`default/strict125/H0/H1/H2/H3/H4/C4 × initial/final × 32 seeds`；
- 门禁：in-domain 相对 multienv 的成功率下降 ≤`0.05`、RMSE ≤`1.10×`、saturation 增量 ≤`0.05`；H1 同时过 H0 BC provisional 门禁和 multienv 保留门禁；strict125 单独要求不退化；C4/H4 只作压力报告；
- 汇总/门禁：`scripts/summarize_pc_mixed_domain_20261104.py`，结果写入 `artifacts/lowlevel/pc_mixed_domain_generalization_{summary,report}_20261104/`；
- 本地测试：`tests/test_pc_mixed_domain_report.py`、`tests/test_pc_server_eval_summary.py`、`tests/test_pc_high_speed_report.py` 共 `26 passed`，Ruff 通过。

mixed-domain 五训练 seed × 32 评估 seed 的 80 个闭环结果已经完成，门禁为 **fail**。相对六环境等权 multienv 基线：

| Tag | Role | Success | 3D RMSE | Saturation | Δsuccess | RMSE ratio | Gate |
|---|---|---:|---:|---:|---:|---:|---|
| default | in-domain | `0.6813` | `5.0831 m` | `0.1460` | `-0.0125` | `1.04×` | pass |
| strict125 | in-domain | `0.4688` | `6.7742 m` | `0.1362` | `+0.0563` | `0.99×` | pass |
| H2 | in-domain | `0.9563` | `3.7789 m` | `0.4852` | `+0.0125` | `0.89×` | pass |
| H3 | in-domain | `0.9625` | `3.7878 m` | `0.4908` | `-0.0187` | `1.54×` | **fail**（RMSE） |
| H0 | held-out | `0.8000` | `5.1733 m` | `0.2377` | n/a | `1.10× H0 BC` | pass |
| H1 | held-out | `0.9125` | `5.3358 m` | `0.4452` | `+0.0125` | `1.19× multienv` | **fail**（saturation/tracking） |
| C4 | held-out pressure | `0.2375` | `8.9685 m` | `0.2523` | `+0.0250` | `1.05×` | pressure-only |
| H4 | pressure-only | `0.9563` | `4.2809 m` | `0.4950` | n/a | n/a | pressure-only |

等种子比较最初漏掉了训练 seed `94030`，只覆盖 `94031–94034`。已经补跑 `94030` 的 default/strict125 32-seed 基线，并把最终报告写入 `pc_mixed_domain_generalization_{summary,report}_5seed32_20261104/`；上表为完整的五 seed 对五 seed 结果，原四 seed 报告保留不覆盖。补齐后门禁结论不变，仍为 `fail`：H3 的 RMSE ratio `1.54×`，H1 同时失败于 H0 BC 的 saturation/tracking 和相对 multienv 的 RMSE 保留。

这组结果支持三点：第一，把 C4 移出训练后，strict125 从 `0.3719 / 7.5631 m` 改善到 `0.4688 / 6.7742 m`，C4 也从 `0.2125 / 8.5388 m` 变为 `0.2375 / 8.9685 m`（成功率 `+0.0250`，RMSE ratio `1.05×`），说明 C4 等权污染的判断有方向性支持；第二，H0 的 held-out 组合能过 H0 BC provisional 门禁；第三，H1 失败仍主要是残差权限与跟踪误差，saturation 比 H0 BC 高 `0.2133`、tracking 最大约 `2.16×`，而 H3 为了高速成功牺牲了相对 multienv 的 RMSE。因此当前 mixed-domain 配方仍不能称为“过拟合已解决”，也不满足正式 PC-G4/PC-G5 的 held-out 通过条件。

自动化上还修复了两个队列缺陷：评测脚本缺失 mixed-domain arm 的 training-config 映射；监督残差汇总会自动给 arm 加 `supervised_` 前缀，而门禁脚本原先只按无前缀名查找，导致完整结果被误报为 missing。两者均已加显式失败检查/别名归一化与回归测试，未重训或覆盖任何 checkpoint。

DAgger 排在 mixed-domain 之后：只有当留出域仍失败时才启动一轮，并复用扩大后的域；不把 DAgger 当作补出缺失速度-海流格子的替代品。

由于 mixed-domain 的正常留出格仍未通过，且现有训练格来自历史配置拼接，新增 S8c 显式 `2×2` 速度×海流网格：训练低弱、低强、高弱三格，严格留出高强一格，C4/H4 只作压力测试。计划、参数、门禁和产物路径见 `doc/goals/goal12_speed_current_grid_overfit.md`；队列脚本为 `run_pc_speed_current_grid_20261104.sh`，服务器队列 PID `4002421`。

S8c 五训练 seed × 32 评估 seed 已完成，门禁为 **fail**。训练审计确认只使用三格、高强格未泄漏、冻结 BC 漂移全为 `0.0`。结果如下：

| Tag | Role | Success | 3D RMSE | Saturation | Gate |
|---|---|---:|---:|---:|---|
| `grid_lw` | in-domain | `1.0000 → 0.9750` | `1.87 → 2.28 m` | `0.1833` | **fail** |
| `grid_ls` | in-domain | `0.6250 → 0.6500` | `5.01 → 4.88 m` | `0.1437` | pass |
| `grid_hw` | in-domain | `0.9688 → 0.9812` | `4.34 → 2.95 m` | `0.4779` | pass |
| `grid_hs` | held-out | `0.8438 → 0.8313` | `6.05 → 6.54 m` | `0.4190` | **fail** |
| `default` | retention | `0.6875 → 0.6062` | `4.93 → 5.54 m` | `0.1339` | **fail** |
| `strict125` | retention | `0.4688 → 0.4250` | `6.52 → 6.56 m` | `0.1253` | pass |
| `c4` | pressure-only | `0.2500 → 0.2062` | `8.49 → 8.84 m` | `0.2335` | pressure-only |
| `h4` | pressure-only | `0.4688 → 0.9375` | `18.88 → 5.86 m` | `0.4749` | pressure-only |

高强留出格相对混合训练没有建立正迁移：成功率下降 `0.0125`、RMSE ratio `1.081×`；相对 multienv final 成功率低 `0.04375`、RMSE ratio `1.271×`，相对 mixed-domain final 成功率低 `0.05625`。低弱训练格和 default 保留也失败，高弱格的 saturation 已到 `0.4779`，说明即使显式补齐三格，残差仍主要靠执行器权限换取分布内收益。严格 1.25 m 和 C4 没有改善。报告见 `artifacts/lowlevel/pc_speed_current_grid_report_20261104/report.{json,md}`，审计见 `artifacts/lowlevel/pc_speed_current_grid_supervised_residual_20261104/audit.json`。

由于高强留出格失败，已按预注册完成一轮复用同一网格域的 DAgger 闭环筛选：五个学生 seed 各采集 `98,304` 个学生轨迹样本（`103–108` episode，无 safety termination），与原三格数据合并后从头重训；审计确认训练配置仍只含低弱、低强、高弱三格，高强格未泄漏，冻结 BC 漂移全为 `0.0`。DAgger 队列 `PID 4035912` 已完成全部 80 个 `initial/final × 32 seeds` 评测，门禁仍为 **fail**：

| Tag | Initial success | No-DAGGER final | DAgger final | Initial RMSE | No-DAGGER RMSE | DAgger RMSE | DAgger saturation | Gate |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `grid_lw` | `1.0000` | `0.9750` | `0.9750` | `1.87 m` | `2.28 m` | `2.33 m` | `0.1888` | **fail** |
| `grid_ls` | `0.6250` | `0.6500` | `0.6500` | `5.01 m` | `4.88 m` | `4.88 m` | `0.1481` | pass |
| `grid_hw` | `0.9688` | `0.9812` | `0.9875` | `4.34 m` | `2.95 m` | `2.56 m` | `0.4879` | **fail** |
| `grid_hs` | `0.8438` | `0.8313` | `0.8500` | `6.05 m` | `6.54 m` | `5.86 m` | `0.4470` | **fail** |
| `default` | `0.6875` | `0.6062` | `0.6062` | `4.93 m` | `5.54 m` | `5.47 m` | `0.1384` | **fail** |
| `strict125` | `0.4688` | `0.4250` | `0.4313` | `6.52 m` | `6.56 m` | `6.60 m` | `0.1298` | pass |
| `c4` | `0.2500` | `0.2062` | `0.2062` | `8.49 m` | `8.84 m` | `8.82 m` | `0.2336` | pressure-only |
| `h4` | `0.4688` | `0.9375` | `0.9688` | `18.88 m` | `5.86 m` | `4.17 m` | `0.5015` | pressure-only |

DAgger 相对 no-DAGGER 在高强留出格上有弱正信号：成功率 `+0.01875`、三维 RMSE `-0.679 m`（`-10.4%`），但相对 pretrain initial 只提高 `0.00625` 成功率和 `3.1%` RMSE，没有达到预注册的 `+0.02` 或 `0.95×` 阈值。与此同时所有格的 saturation 都上升，`grid_hw` 的 saturation 增量 `+0.052` 超门禁，`grid_lw` 和 default 仍失败，`h4` 压力 saturation 达到 `0.5015`。因此单轮 DAgger 只能保留为“教师标签覆盖带来弱正信号”，不能写成 held-out 泛化或过拟合问题已解决。报告见 `artifacts/lowlevel/pc_grid_dagger_report_20261104/report.{json,md}`，审计见 `artifacts/lowlevel/pc_grid_dagger_supervised_residual_20261104/audit.json`。按预注册停止条件，不再续做同配方 DAgger，转去诊断域覆盖、教师质量、执行器权限和观测/残差通路。

复现与提交命令（本地验证后执行，不自动 commit/push）：

```bash
cd /home/bubble/水下机器人/settled_局部规划器_反应式
../.venv/bin/pytest -q tests/test_pc_mixed_domain_report.py tests/test_pc_server_eval_summary.py tests/test_pc_high_speed_report.py
../.venv/bin/ruff check scripts/summarize_pc_mixed_domain_20261104.py scripts/summarize_pc_high_speed_20261104.py
git add configs/lowlevel/odarrl_control_ppo_pc_mixed_domain_*.json \
  configs/lowlevel/odarrl_control_pc_grid_*.json \
  configs/lowlevel/odarrl_control_ppo_pc_grid_*.json \
  scripts/run_pc_high_speed_residual_20261104.sh \
  scripts/run_pc_mixed_domain_residual_20261104.sh \
  scripts/run_pc_speed_current_grid_20261104.sh \
  scripts/run_pc_grid_dagger_20261104.sh scripts/collect_pc_dagger_demos.py \
  scripts/summarize_pc_high_speed_20261104.py \
  scripts/summarize_pc_mixed_domain_20261104.py \
  scripts/summarize_pc_speed_current_grid_20261104.py \
  scripts/summarize_pc_server_evals_basics_20261103.py \
  src/auv_lowlevel/batch/pc_vec_env.py \
  tests/test_pc_high_speed_report.py tests/test_pc_mixed_domain_report.py \
  tests/test_pc_speed_current_grid_report.py tests/test_pc_server_eval_summary.py \
  tests/test_m4_bc.py \
  doc/goals/goal9_odarrl_control_reproduction.md \
  doc/goals/goal12_speed_current_grid_overfit.md
git commit -m "experiments: add high-speed and mixed-domain residual queues"
git push
```

**C. DADE-pose GPU 扩 seed 与自动评测（2026-11-04，闭环评测进行中）。** 全量 multienv 的 DADE 监督预训练此前只排了 CPU，`seed94030` 在 CPU 上运行约四小时后仍无产物；同数据集的 GPU 训练约 7 分钟完成。因此停止原 CPU 子进程 `3879051` 和其父 shell `3873284`，在 GPU1 上重训 `seed94030`，并保留 `seed94031–94034` 的既有 GPU 产物。`seed94030` 已完成并通过结构审计：`status=completed`、`best_epoch=34`、`epochs_completed=65`、`dade_enabled=true`、`training_device=cuda`、冻结 BC 最大漂移 `0.0`，checkpoint 含 `residual_actor_*`、`residual_current_head_*`、`residual_dade_head_*` 和 `actor_dade.*`。审计另存为 `artifacts/lowlevel/pc_multienv_supervised_residual_full_20261104/audit_dade_seed94030.json`，未覆盖四 seed 审计备份。

`scripts/supervise_pc_multienv_dade_20261104.sh`（最新服务器 PID `4141284`；首轮 PID `4132687` 已退出）负责依赖顺序：

- [x] 首批 GPU 完成后，在 GPU1/5 追加 `seed94033/94034`，使 DADE 与 no-DADE 对齐为五个训练 seed；四个 GPU seed 均为 `completed`、DADE 开启、冻结 BC 漂移 `0.0`；
- [x] 校验 GPU 重训 `seed94030` 的 `final.pt`、manifest、冻结 BC 漂移和 DADE 状态；不再恢复会等待 CPU 子进程的主 shell；
- [x] 五 seed `default/strict125` 历史机制诊断完成：10 个 run 全部通过，`current-only` 动作敏感度 `5.3–7.9%`，`history-zero` `15.9–28.8%`，阈值 `0.1%`；门禁文件为 `artifacts/lowlevel/pc_dade_history_multienv_5seed_gate_20261104.json`；
- [x] 五 seed × 2 arm × 2 checkpoint 的 default/strict125（64 seeds）与 H1/H2/H3/C4（32 seeds）闭环评测和汇总完成；120 个 summary 全部 `rc=0`，汇总见 `artifacts/lowlevel/pc_multienv_supervised_residual_summary_full_20261104/`；

扩展产物仍写入 `artifacts/lowlevel/pc_multienv_supervised_residual_{full,eval_full,summary_full}_20261104/`；3-seed summary 会另存为 `pc_multienv_supervised_residual_summary_full_3seed_20261104/`。监督日志为 `logs/pc_multienv_dade_extension_20261104/queue.log`。该扩展只增加 seed 数和自动化，不改变 DADE-pose 配方；若 5-seed 结果仍显示历史机制门禁失败，则按既有负结果停止条件处理，不再追加同配方调参。

当前状态：mixed-domain 五 seed 筛选已完成且门禁 `fail`，保留为负结果；显式速度×海流网格五 seed 也已完成且门禁 `fail`，高强零样本格没有改善；高速队列 15 个残差训练和全部闭环评测已完成，门禁 `fail`；单轮网格 DAgger 的 80 个评测也已完成，门禁 `fail`，仅相对 no-DAGGER 有弱正信号，按预注册不再续做同配方；DADE-pose 五个 seed 已完成并通过结构审计，五 seed 历史机制门禁全部通过，且五 seed 闭环配对评测也已完成。DADE 的结论是“历史机制确实被使用，但闭环性能不足以通过验收”，因此按既定停止条件不继续同一 DADE 配方调参。

#### 五 seed 闭环结果（DADE vs no-DADE，final checkpoint）

| tag | no-DADE success / RMSE / sat | DADE success / RMSE / sat | paired Δsuccess | paired ΔRMSE |
|---|---:|---:|---:|---:|
| default | `0.7219 / 5.1374 / 0.1355` | `0.6875 / 5.3401 / 0.0982` | `-0.0344` | `+0.2026 m` |
| strict125 | `0.3719 / 7.5631 / 0.1238` | `0.3844 / 7.2025 / 0.0889` | `+0.0125` | `-0.3606 m` |
| H1 | `0.9000 / 4.4950 / 0.4709` | `0.9500 / 3.5638 / 0.4436` | `+0.0500` | `-0.9312 m` |
| H2 | `0.9437 / 4.2273 / 0.4856` | `0.9688 / 3.7004 / 0.4752` | `+0.0250` | `-0.5269 m` |
| H3 | `0.9812 / 2.4551 / 0.4973` | `0.9875 / 2.6830 / 0.4882` | `+0.0063` | `+0.2279 m` |
| C4 | `0.2125 / 8.5388 / 0.2573` | `0.2250 / 8.6416 / 0.2252` | `+0.0125` | `+0.1028 m` |

判定：**DADE 历史机制 PASS，闭环验收仍 FAIL。** 五 seed 的 default/strict125 历史诊断全部通过，说明 DADE 特征不是“挂件”；闭环上 H1/H2 的 success 和 RMSE 有明确正收益，C4/H3 基本持平，但 default 退化，strict125 仍只有 `0.3844 < 0.50`，H1/H2/H3 的 saturation 仍高于 `0.40/0.45` 门禁。故保留这组正/负结果，不再对同一 DADE-pose 配方追加调参。

自动化入口：

- `scripts/run_pc_multienv_eval_5seed_20261104.sh`：primary（default/strict125，64 seeds）→ hard（H1/H2/H3/C4，32 seeds）→ 自动汇总；已有 `summary.json` 会跳过，适合断点续跑；
- `scripts/run_pc_multienv_dade_history_5seed_20261104.sh`：五 seed × default/strict125 的 DADE 历史消融和固定门禁；
- 闭环产物：`artifacts/lowlevel/pc_multienv_supervised_residual_eval_full_20261104/`；汇总：`artifacts/lowlevel/pc_multienv_supervised_residual_summary_full_20261104/summary.{json,md}`；
- 历史诊断：`artifacts/lowlevel/pc_dade_history_multienv_5seed_20261104/`；门禁：`artifacts/lowlevel/pc_dade_history_multienv_5seed_gate_20261104.json`。

**D. 显式速度×海流网格上的 DADE 配对验证（已完成，`overall_status=fail`）。** 为避免把历史拼接域上的 DADE 失败误判为结构无效，新增严格配对的 `grid_lw/grid_ls/grid_hw` 训练域，并把高速强流 `grid_hs` 完全留出。两臂共享冻结 BC、同一批三格 teacher demos、同一训练 seed `94030–94034` 和同一评估 seed `0–31`，唯一差异是 DADE-pose；`220` 个闭环评测全部 `rc=0`。正式报告见 `artifacts/lowlevel/pc_dade_grid_report_20261104/report.{json,md}`，完整审计与实验协议见 `doc/goals/goal13_dade_speed_current_grid.md`。

| Tag | DADE success / RMSE / timeout / sat | no-DADE success / RMSE / timeout / sat | Gate |
|---|---|---|---|
| `grid_lw` | `1.0000 / 1.88 m / 0.0000 / 0.1574` | `0.9812 / 2.20 m / 0.0188 / 0.1837` | pass |
| `grid_ls` | `0.6813 / 4.79 m / 0.3187 / 0.1274` | `0.6625 / 4.82 m / 0.3375 / 0.1431` | pass |
| `grid_hw` | `0.9750 / 3.48 m / 0.0250 / 0.4607` | `0.9875 / 2.52 m / 0.0125 / 0.4797` | **fail** |
| `grid_hs` | `0.9000 / 4.90 m / 0.1000 / 0.4382` | `0.8438 / 6.11 m / 0.1563 / 0.4278` | pass |
| default | `0.6813 / 4.96 m / 0.3187 / 0.1171` | `0.6375 / 5.20 m / 0.3625 / 0.1359` | **fail** |
| strict125 | `0.4562 / 6.36 m / 0.5437 / 0.1074` | `0.4062 / 6.63 m / 0.5938 / 0.1249` | **fail** |
| H1 | `0.9437 / 3.76 m / 0.0563 / 0.4455` | `0.9250 / 4.36 m / 0.0750 / 0.4481` | **fail** |
| H3 | `1.0000 / 1.84 m / 0.0000 / 0.5098` | `0.9750 / 3.65 m / 0.0250 / 0.5029` | **fail** |

`grid_hs` 的同 seed 配对为 success `+0.0563`、RMSE 比 `0.8220`、saturation `+0.0105`，是该实验最明确的 held-out 正信号；但它仍有 `0.10` timeout 和 `4.90 m` RMSE。`grid_hw` 的 RMSE 比 `1.523`，说明 DADE 在高速弱流训练格上引入轨迹误差；default/strict125 虽然相对 no-DADE 有小幅改善，但绝对 success `0.6813/0.4562` 未过线；H1/H3 的 saturation 仍为 `0.4455/0.5098`。因此结论只能是“DADE 对留出速度×海流组合有迁移信号”，不能写成跨环境泛化通过，也不能再用同配方调参或挑选 seed 改写结果。后续优先转向 actor 的实际地速跟踪、闭环状态覆盖、执行器权限/饱和奖励；DADE 暂停作为主调参对象。

### 14.4.16 ALOS-R 制导改进与零样本闭环评测（2026-11-04，R1K 五 seed 与 held-out 均未过，R2B/R3B 负结果）

ALOS-R 是一个独立的制导实验组，不新增学习残差、不改变 actor 输入维度、不重新训练 actor。目标是在冻结 no-DADE 策略的前提下，只改 ALOS 制导层：终端航向捕获、部署传感器海流蟹行前馈、高速 lookahead 与饱和调速。代码默认关闭所有 robust 功能，旧配置行为保持不变；R1 先在服务器独立目录运行，不修改或重启正在执行的 multienv/DADE 主队列。

最新约束口径：**整条航线的命令速度不得低于本 episode 设定巡航速度的 75%；途中尽量保持巡航，终端允许降到 75%，但不得降到 0 或重新引入“最后爬行超时”。** `minimum_speed_ratio=0.75` 在配置层校验终端目标，并在 PCEnv 中对终端逻辑、饱和调速和命令速率限制之后的最终速度再次钳位，避免组合模块绕过下限。该 75% 是控制器的硬下限，不是终端距离的函数。起步螺旋桨允许非零，初始速度也允许非零：`initial_propeller_rpm_ratio` 控制起步转速前馈，`initial_speed_ratio` 控制 episode 初始纵荡速度；两者默认分别为 `0.0` 和 `1.0`，旧配置行为不变。另新增实际航迹推进速度比 `ground_speed_ratio_min/mean` 和 `ground_route_speed_ratio_min/mean` 作为诊断指标：前者是真实 NED 地速模长比，后者把真实 NED 地速投影到参考路径切向后除以本 episode 巡航速度。它们不进入硬门禁，因为未知海流、转向和起步瞬态无法由指令下限消除；对应证据是 R1K default 的指令最低比 `1.0000`，实际地速最低比仅 `0.3781`（出现在约 10.7 s），不能把“命令速度达标”写成“实际路线最低速度达标”。

#### 实现

- `src/auv_lowlevel/config.py`：新增 `robust_terminal_enabled`、`terminal_speed_ratio`、`terminal_braking_enabled`、`minimum_speed_ratio`、`terminal_alignment_*`、`current_crab_*`、`current_estimate_*`、`speed_adaptive_lookahead_enabled`、`saturation_governor_*`、`route_speed_floor_*`、`initial_propeller_rpm_ratio`、`initial_speed_ratio` 和命令速率限制字段；旧配置不写这些字段时 robust 功能保持关闭，速度下限默认 `0.75`。
- `src/auv_lowlevel/batch/geometry.py`：新增 `body_to_world()`，用于把部署可用的 `current_estimate_body` 转到 NED。
- `src/auv_lowlevel/batch/guidance.py`：新增 robust 终端航向捕获、速度自适应 lookahead 和可选海流蟹行；默认不做距离制动，只有显式开启 `terminal_braking_enabled` 时才在终端段线性减速，且下限由 `minimum_speed_ratio` 保证，永远不会降到 0。
- `src/auv_lowlevel/batch/pc_env.py`：估计海流仅在 `current_estimate_valid` 且 age `<= 0.6 s` 时进入 1.5 s 低通；记录 `current_valid_rate`、age、crab、speed scale、terminal phase、terminal speed ratio 和 sat EMA；高速饱和调速只对 `u_cmd > 1.8 m/s` 生效，`goal_distance` 不进入调速公式；最终命令速度在速率限制后再次应用 75% 硬下限。`_reset_episode_command_speeds()` 在 episode reset 后按实际采样巡航速度设置初速和起步 RPM，而不是使用固定参考速度；初始纵荡速度和预置 RPM 同时受车辆最大速度与 `plant_max_propeller_rpm` 钳位，避免高巡航速度下 1.25× 初速越过 REMUS100 包线。
- `src/auv_lowlevel/batch/pc_vec_env.py`：把 ALOS-R 诊断字段从底层 env 透传到评估 loop。
- `scripts/evaluate_pc_ppo.py`：新增 `--guidance-config`，仅替换环境配置中的 `guidance` 块，保持 stage、传感器、动作包线和归一化不变；自动输出 ALOS-R 指标、全航线命令速度比 `guidance_route_speed_ratio_min`，以及按真实 NED 地速在路径切向投影计算的实际航迹速度比 `ground_route_speed_ratio_min/mean`。
- `scripts/run_pc_alos_r_20261104.sh`：独立 ALOS-R 评测队列，支持 `R0/R1/R1B/R1C/R1D/R1E/R1F/R1G/R1H/R1I/R1J/R1K/R4A/R4B/R4C/R2/R2B/R3/R3B`、多训练 seed、多 tag、并行 CPU worker 和自动汇总；不触碰主 multienv 队列。
- `scripts/summarize_pc_alos_r.py`：按 tag 聚合 success、timeout、3D RMSE、saturation、全航线最低速度比、terminal speed ratio、crab、current valid rate、speed scale 和 sat emergency，并生成门禁报告。
- 新增配置：`configs/lowlevel/odarrl_control_alos_r1_terminal.json`（只开 robust terminal）、`odarrl_control_alos_r1b_legacy20_floor75.json`（保留 20 m 捕获，终端线性降至 75%）、`odarrl_control_alos_r1c_ground_floor75.json`（地速反馈 + 100% 起步 RPM）、`odarrl_control_alos_r1e_terminal20_prespin75.json`（无地速反馈 + 100% 起步 RPM）、`odarrl_control_alos_r1f_terminal20_noprespin75.json`（无地速反馈 + 0% 起步 RPM）、`odarrl_control_alos_r1g/h/i/j_*.json`（地速反馈开/关 × 起步 RPM 0/75% 的 2×2 因子）、`odarrl_control_alos_r1k_ground_floor_init100_nospin.json`（地速反馈 + 初速 1.0× + 0% 起步 RPM）、`odarrl_control_alos_r4a_launch_init125_nospin.json`（R1K + 初速 1.25× + 0% RPM）、`odarrl_control_alos_r4b_launch_init125_prespin75.json`（R1K + 初速 1.25× + 75% RPM）、`odarrl_control_alos_r4c_launch_init125_prespin100.json`（R1K + 初速 1.25× + 100% RPM）、`odarrl_control_alos_r2b_crab_floor_init100.json`（R1K + 蟹行）、`odarrl_control_alos_r3b_governor_floor_init100.json`（R2B + 高速 lookahead/饱和调速）。旧配置未修改。
- 新增单元测试：`tests/test_batch_alos_r.py`，覆盖终端 1 m 不减速回归、75% 路线下限、低于下限的配置拒绝、速率限制后仍守住下限、终端航向、invalid current、age gate、饱和调速高速度限定、旧配置默认关闭、profile 开关组合，以及 `initial_speed_ratio`/`initial_propeller_rpm_ratio` 按 episode 速度缩放和预置 RPM 钳位。

#### 参数

| 参数 | 值 | 说明 |
|---|---:|---|
| `terminal_speed_ratio` | `1.0` | R1 无终端减速；旧配置未设置时也默认 `1.0` |
| `terminal_braking_enabled` | `false` | 新增可选开关；关闭时完全不进入距离制动曲线 |
| `minimum_speed_ratio` | `0.75` | 整条航线命令速度相对 episode 巡航速度的硬下限 |
| `initial_speed_ratio` | `1.0` | episode 初始纵荡速度 = 实际 episode 巡航速度 × 该比例；允许非零，默认保持旧行为 |
| `initial_propeller_rpm_ratio` | `0.0` | 起步螺旋桨前馈 RPM = 比例 × 速度前馈 × 初始速度；默认 0，旧配置不变 |
| `route_speed_floor_launch_speed_ratio` / `duration` | `1.0` / `2.0 s` | 起飞窗口速度倍率/时长；默认 1.0 即不额外加速 |
| `terminal_alignment_horizon_s` | `6.0` | `d_align = clip(6*u_cmd, 6, 16)` |
| `terminal_alignment_min_m` / `max_m` | `6` / `16` | 对准提前量上下限 |
| `current_estimate_max_age_s` | `0.6` | 超时立即关闭蟹行并回退基线 |
| `current_estimate_tau_s` | `1.5` | 海流估计低通时间常数 |
| `current_crab_relative_speed_limit` | `1.12` | 可全补偿的相对水速上限 |
| `current_crab_max_angle_rad` | `18°` | 蟹行角上限 |
| `high_speed_threshold_mps` | `1.8` | 高速 lookahead/调速只在高于该速度时启用 |
| `saturation_ema_tau_s` | `1.0` | 材料饱和 EMA 窗口约 1 s |
| `heading_rate_limit_rad_s` | `30°/s` | R3 命令速率限制 |
| `pitch_rate_limit_rad_s` | `15°/s` | R3 命令速率限制 |
| `speed_rate_limit_mps2` | `0.25` | R3 命令速率限制 |

#### R1 正式结果

服务器独立目录：`/data/yhx/underwater-robotics/src/plan_p1/settled_局部规划器_反应式_alos_r`。使用冻结 no-DADE multienv `seed94030` 的 `final.pt`，固定评估 seeds `0–31`。R0 是同一 checkpoint、同一代码、无 ALOS-R overlay 的当前基线；R1 是只开启 robust terminal 的实验组。

| tag | R0 success | R1 success | R0 timeout | R1 timeout | R0 3D RMSE | R1 3D RMSE | R1 saturation | R1 terminal ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| default | 0.7188 | 0.3750 | 0.2812 | 0.6250 | 4.6411 | 8.2618 | 0.1575 | 1.0000 |
| strict125 | 0.4688 | 0.2188 | 0.5312 | 0.7812 | 6.5512 | 8.7730 | 0.1576 | 1.0000 |
| h1 | 1.0000 | 0.3750 | 0.0000 | 0.6250 | 1.7466 | 13.1718 | 0.4109 | 1.0000 |
| h3 | 1.0000 | 0.5625 | 0.0000 | 0.4375 | 1.7735 | 8.6590 | 0.4461 | 1.0000 |
| c4 | 0.2188 | 0.1562 | 0.7812 | 0.8438 | 8.4783 | 10.2396 | 0.2342 | 1.0000 |

判定：**R1 fail，不进入 R2。** `terminal_speed_ratio=1.0` 的目标本身通过，但 strict125 的 success、timeout、RMSE 全部未达标；default/H1/H3 也显著退化。尤其 H1/H3 从 `1.0` 掉到 `0.375/0.5625`，说明终端制导改动不是单纯“去掉爬行”，而是破坏了原有终端捕获。

#### R1B：20 m 捕获 + 75% 终端下限

按最新速度约束新增 `R1B` 对照：保留旧终端捕获距离 `20 m`，仅在终端段把速度线性降到 `0.75 u_cmd` 后保持；`minimum_speed_ratio=0.75` 同时约束终端、饱和调速和速率限制后的最终命令。新队列使用独立产物目录 `artifacts/lowlevel/pc_alos_r_floor75_20261104`，不覆盖第一轮证据。

| tag | success | timeout | 3D RMSE | saturation | route speed min | terminal ratio | 门禁 |
|---|---:|---:|---:|---:|---:|---:|---|
| default | 0.5625 | 0.4375 | 5.0593 | 0.1418 | 0.8333 | 0.8854 | fail（success、RMSE） |
| strict125 | 0.3750 | 0.6250 | 6.2917 | 0.1377 | 0.8333 | 0.8838 | fail（success、timeout） |
| h1 | 0.8750 | 0.1250 | 4.6098 | 0.4550 | 0.7659 | 0.8905 | fail（success、saturation） |
| h3 | 1.0000 | 0.0000 | 1.7654 | 0.5016 | 0.7693 | 0.8873 | fail（saturation） |
| c4 | 0.1562 | 0.8438 | 8.6987 | 0.2325 | 0.8333 | 0.8850 | pass（无硬门禁） |

判定：**R1B 仍 fail。** 20 m 捕获恢复了 H3 的成功率和 RMSE，并把 strict125 RMSE 从 `6.5512` 降到 `6.2917 m`；但 strict125 success 只有 `0.3750`、timeout `0.6250`，default success 也只有 `0.5625`。所有 tag 的全航线最低速度比均 `>= 0.75`，说明新增硬约束执行正确；但终端减速本身没有解决 strict125/default 的按时到达问题，且 H1/H3 的饱和没有下降。

#### 终端诊断

为解释 R1 失败，固定同 checkpoint、同 seeds 做了不入正式队列的诊断：

| 变体 | default success / RMSE | strict125 success / RMSE | h3 success / RMSE | 说明 |
|---|---:|---:|---:|---|
| R0 基线 | `0.7188 / 4.6411` | `0.4688 / 6.5512` | `1.0000 / 1.7735` | 旧 terminal capture，`terminal_capture_distance=20 m` |
| R1 align16，无减速 | `0.6562 / 4.7089` | `0.3750 / 6.9957` | `0.6250 / 6.8829` | 只把对准提前量拉到上限 16 m |
| R1 align16，终端下限 65% | `0.5312 / 4.9430` | `0.4062 / 5.8466` | `0.6250 / 7.7301` | `terminal_braking_enabled=true` |
| R1 align16，终端下限 30% | `0.5000 / 5.1931` | `0.3750 / 6.1495` | `0.6250 / 8.0696` | 仍未降到 0 |
| R1 align7.2，终端下限 65% | `0.3438 / 8.0562` | `0.1875 / 8.5216` | `0.5938 / 8.4176` | 对准距离过短 |
| R1B 旧 20 m，终端下限 75% | `0.5625 / 5.0593` | `0.3750 / 6.2917` | `1.0000 / 1.7654` | 路线最低速度比最低 `0.7659` |

关键结论：

- R0 的 `terminal_speed_mps=0` 在 PCEnv 中被车辆最小速度 `min_speed_mps=1.0` 钳位，因此**实际并没有线性降到 0，也没有真正爬行到 0**；原假设需要修正。
- R1 的主要退化来自两点叠加：对准提前量从旧终端的 `20 m` 缩到 `7.2 m`（即使拉到规格上限也只到 `16 m`），以及取消了终端小减速。16 m 仍比旧 20 m 更晚，H1/H3 高速下尤其不够。
- R1B 证明“恢复 20 m 捕获 + 有界终端减速”能把 H3 救回基线，但不能改善 default/strict125；strict125 的主要问题不是终端速度，而是当前冻结策略在窄时限/当前扰动下无法稳定进入终点球。
- 因此 R1/R1B 都没有通过前置门禁，R2/R3 不启动，也不应把 H3 的分布内成功率当成泛化通过。

#### R1C–R1F：地速反馈与起步 RPM 预置

在 R1B 之后，控制层增加了“地速不低于命令下限 75%”的反馈，并测试起步螺旋桨是否可以非零。以下均为 no-DADE `seed94030`、评估 seeds `0–31`；`RPM` 指 `initial_propeller_rpm_ratio`，`初速` 指 `initial_speed_ratio`。

| 变体 | 地速反馈 | 起步 RPM | 初速 | default success / RMSE | strict125 success / RMSE | h1 RMSE / sat | h3 RMSE / sat | h4 RMSE / sat |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| R1C | 开 | 100% | 1.0 | `0.6250 / 6.2933` | `0.4062 / 8.0178` | `2.5467 / 0.4853` | `3.2077 / 0.4940` | `3.7897 / 0.4848` |
| R1E | 关 | 100% | 1.0 | `0.5938 / 5.9671` | `0.4375 / 7.1192` | `1.7446 / 0.4891` | `3.2803 / 0.4938` | `3.7897 / 0.4848` |
| R1F | 关 | 0% | 1.0 | `0.7188 / 4.6411` | `0.4688 / 6.5512` | `1.7466 / 0.4759` | `1.7735 / 0.4976` | `3.4628 / 0.4964` |

- R1C 的 strict125 timeout 为 `0.5938`、R1E 为 `0.5625`、R1F 为 `0.5312`；100% 起步 RPM 在 default 上没有收益，反而把 default success 从 R1F 的 `0.7188` 降到 `0.5938–0.6250`。
- 三个变体的 `guidance_route_speed_ratio_min` 都为 `1.0000`，说明命令速度从未低于 75%；但真实地速最低比仍只有 `0.5317`（R1C）、`0.2962`（R1E）和 `0.2458`（R1F），说明问题不能由命令下限单独解决。
- R1D（地速反馈 + 蟹行 + 100% 起步 RPM）只保留配置，没有扩大正式队列，避免把蟹行和起步 RPM 混在一起。

#### R1G–R1K：初速与起步 RPM 的 2×2 因子

为避免 8-seed smoke 的噪声，进一步跑了 32-seed 的完整因子：地速反馈开/关 × 起步 RPM `0/75%`，统一 `initial_speed_ratio=1.10`；随后补 `R1K` 作为“反馈开、RPM 0、初速 1.0×”的干净对照。所有变体均为 20 m 终端捕获、`terminal_speed_ratio=1.0`、`minimum_speed_ratio=0.75`。

| 变体 | 反馈 | RPM | 初速 | default success / RMSE | strict success / RMSE | h1 success / RMSE / sat | h3 success / RMSE / sat | c4 success / RMSE |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| R1G | 开 | 0% | 1.10 | `0.7812 / 4.7543` | `0.5000 / 7.3160` | `0.9688 / 2.4879 / 0.4699` | `1.0000 / 1.7893 / 0.5040` | `0.2188 / 9.9063` |
| R1H | 开 | 75% | 1.10 | `0.6250 / 6.3082` | `0.4375 / 7.8910` | `1.0000 / 1.7711 / 0.4944` | `0.9688 / 3.2228 / 0.4983` | `0.1875 / 10.3646` |
| R1I | 关 | 0% | 1.10 | `0.6875 / 4.8698` | `0.4062 / 7.0924` | `1.0000 / 1.7361 / 0.4808` | `1.0000 / 1.7699 / 0.4998` | `0.2188 / 8.4369` |
| R1J | 关 | 75% | 1.10 | `0.5938 / 5.9776` | `0.4062 / 7.1809` | `1.0000 / 1.7490 / 0.4952` | `0.9688 / 3.2880 / 0.4977` | `0.2188 / 9.1010` |
| R1K | 开 | 0% | 1.00 | **`0.8125 / 4.5738`** | **`0.5000 / 7.4043`** | `0.9688 / 2.5064 / 0.4681` | `1.0000 / 1.8135 / 0.5004` | `0.2188 / 9.8906` |

- `initial_speed_ratio=1.10` 不是收益来源：在无反馈、无预置 RPM 的同条件下，R1I 的 default `0.6875 / 4.8698`、strict `0.4062 / 7.0924` 都差于 R1F 的 `0.7188 / 4.6411`、`0.4688 / 6.5512`。
- 75% 预置 RPM 同样不是收益来源：反馈组 R1H 的 strict success 从 R1G 的 `0.5000` 降到 `0.4375`；无反馈组 R1J 的 default success 从 R1I 的 `0.6875` 降到 `0.5938`。因此“起步可以非零”作为能力保留，但当前最优配方不应默认开启。
- 地速反馈仍然只解决 default 的一部分：R1K 把 default 提升到 `0.8125 / 4.5738 m`，达到 default 的成功率和 RMSE 门禁；strict125 success 恰好 `0.5000`、timeout `0.5000` 也达标，但 strict RMSE `7.4043 m > 6.5 m`，H1/H3 的 saturation 仍分别 `0.4681 > 0.40`、`0.5004 > 0.45`。
- 所有变体的命令速度最低比仍为 `1.0000`；R1K 的真实地速最低比只有 `0.3781`（default/strict），出现在约 `10.7 s`。**因此实际地速 75% 下限尚未实现；当前只能证明命令层硬下限有效。**

判定：**R1K 的单 seed 结果暂时通过 default/strict 主成功门禁，但跨 5 个 no-DADE 训练 seed 后 strict success 均值只有 `0.4938 < 0.50`；严格 RMSE、H1/H3 saturation 和真实地速最低比仍未通过。** 不应把 R1K 写成完整验收通过；下一步若继续，需要改执行器速度跟踪或闭环覆盖，而不是继续加起步 RPM。

#### R1K 五 seed 稳定性

把 R1K 扩到 no-DADE `94030–94034` 五个训练 seed、每个 32 个评估 seed、`default/strict125/H1/H3/C4` 五个环境；H4 单独跑同样五个训练 seed × 32 个评估 seed。结果按训练 seed 取均值：

| tag | success（mean ± std） | timeout | 3D RMSE | saturation | command speed min | 真实地速 min |
|---|---:|---:|---:|---:|---:|---:|
| default | `0.8000 ± 0.0319` | `0.2000` | `4.8163 ± 0.2856` | `0.2550 ± 0.0053` | `1.0000` | `0.3769` |
| strict125 | `0.4938 ± 0.0125` | `0.5063` | `7.5770 ± 0.1791` | `0.2392 ± 0.0052` | `1.0000` | `0.3769` |
| H1 | `0.8875 ± 0.0580` | `0.1125` | `5.3728 ± 1.5336` | `0.4705 ± 0.0088` | `1.0000` | `0.4692` |
| H3 | `0.9625 ± 0.0459` | `0.0375` | `3.2171 ± 1.9004` | `0.4982 ± 0.0093` | `1.0000` | `0.6071` |
| C4 | `0.2000 ± 0.0319` | `0.8000` | `10.2437 ± 0.2814` | `0.3128 ± 0.0080` | `1.0000` | `0.1327` |
| H4（独立队列） | `0.9438 ± 0.0538` | `0.0563` | `4.3161 ± 1.7241` | `0.4962 ± 0.0094` | `1.0000` | `0.6605` |

- default 跨 seed 稳定过 success/RMSE；strict success 均值只差 `0.0062`，但方差很小，不能把单 seed 的 `0.5000` 当作稳定通过。
- H1 的 success 只有 `0.8875 < 0.95`，H3 虽然 success `0.9625`，但 saturation `0.4982 > 0.45`；H4 RMSE `4.3161 > 3.8 m`、saturation `0.4962 > 0.45`。
- 所有 tag 的命令速度最低比仍为 `1.0000`；真实地速最低比在 default/strict 仅 `0.3769`，说明起速与地速反馈仍不能保证“实际路线最低速度 ≥ 75%”。

#### R2B/R3B smoke：蟹行与饱和调速

在 R1K 基线上补充 `R2B`（加部署传感器海流蟹行）和 `R3B`（R2B + 高速 lookahead + 饱和调速），先跑 8 个评估 seed 的 smoke。结果如下：

| 变体 | tag | success | timeout | 3D RMSE | saturation | command speed min |
|---|---|---:|---:|---:|---:|---:|
| R2B | default | `0.8750` | `0.1250` | `4.4119` | `0.2904` | `0.9668` |
| R2B | strict125 | `0.3750` | `0.6250` | `8.9109` | `0.2728` | `0.9668` |
| R2B | h1 | `0.8750` | `0.1250` | `4.5984` | `0.4796` | `0.9823` |
| R2B | h3 | `0.8750` | `0.1250` | `3.7770` | `0.4860` | `0.9947` |
| R3B | default | `0.7500` | `0.2500` | `6.3520` | `0.2890` | `0.9748` |
| R3B | strict125 | `0.6250` | `0.3750` | `7.0025` | `0.2937` | `0.9748` |
| R3B | h1 | `0.7500` | `0.2500` | `4.8164` | `0.4657` | `0.8800` |
| R3B | h3 | `0.7500` | `0.2500` | `5.9306` | `0.4818` | `0.8797` |

- R2B 的蟹行角均值很小，但 strict125 从 R1K 的 `0.5000` 掉到 `0.3750`，H1/H3 也从 `0.9688/1.0000` 掉到 `0.8750/0.8750`；说明当前海流估计/蟹行分支会破坏窄时限和高速闭环，不进入 32-seed。
- R3B 的调速把 h1/h3 的成功率压到 `0.7500`，saturation 仍为 `0.4657/0.4818`，没有达到 `0.40/0.45` 门禁；并且 H3 下降超过 `0.02`，按预设应关闭调速模块。
- R2B/R3B 都没有扩大训练 seed 或进入五 seed；它们的负结果保留，不覆盖 R1K。

#### 独立 held-out 评估：评估 seeds `32–63`（2026-11-04，已完成，未复现收益）

之前的 ALOS-R 结果都在评估 seeds `0–31` 上筛选，存在把同一批评估 seed 用于选配方的风险。为确认 R1K 和“非零起步速度”是否真的稳定，新增一组完全不参与此前筛选的 held-out 闭环评估：

- 训练 checkpoint 固定为 no-DADE `94030–94034` 的 `final.pt`，不重新训练 actor；
- 评估 seeds 固定为 `32–63`，每 tag 32 个 episode；
- tags 为 `default strict125 h1 h2 h3 h4`；其中 H2 只作诊断，不新增硬门禁；
- 三条队列分别跑无 ALOS-R 的 `R0`、当前最优 `R1K`（初速 1.0×、预旋 0）和 `R1G`（初速 1.1×、预旋 0），用于同 seed 配对比较“初速非零但预旋关闭”是否可泛化；
- 共 `3 stage × 5 train seed × 6 tag = 90` 个 `summary.json`，任一评测失败都会由队列返回非零，不会静默丢 tag。

服务器进程与目录：

| stage | 队列 PID | CPU 核段 | 产物根 |
|---|---:|---|---|
| R0 | `359152` | `64–127` | `artifacts/lowlevel/pc_alos_r_holdout32_r0_20261104` |
| R1K | `359366` | `144–207` | `artifacts/lowlevel/pc_alos_r_holdout32_r1k_20261104` |
| R1G | `359542` | `224–287` | `artifacts/lowlevel/pc_alos_r_holdout32_r1g_20261104` |

三条队列已全部完成，`90/90` 个 summary 均为 `rc=0`。结果如下（5 个训练 seed 的均值）：

| stage | tag | success | timeout | 3D RMSE (m) | saturation | ground speed min |
|---|---|---:|---:|---:|---:|---:|
| R0 | default | `0.6562` | `0.3438` | `5.545` | `0.1494` | `0.1073` |
| R0 | strict125 | `0.3937` | `0.6062` | `7.750` | `0.1419` | `0.1073` |
| R0 | H1 | `0.9437` | `0.0563` | `3.377` | `0.4709` | `0.2958` |
| R0 | H2 | `0.9812` | `0.0188` | `3.001` | `0.4905` | `0.5017` |
| R0 | H3 | `0.9750` | `0.0250` | `3.258` | `0.5030` | `0.6095` |
| R0 | H4 | `0.9187` | `0.0813` | `6.252` | `0.4944` | `0.6613` |
| R1K | default | `0.6937` | `0.3063` | `5.994` | `0.2374` | `0.2114` |
| R1K | strict125 | `0.4000` | `0.6000` | `8.831` | `0.2260` | `0.2114` |
| R1K | H1 | `0.9375` | `0.0625` | `4.280` | `0.4727` | `0.3348` |
| R1K | H2 | `0.9750` | `0.0250` | `2.830` | `0.4932` | `0.5147` |
| R1K | H3 | `0.9812` | `0.0188` | `3.174` | `0.5047` | `0.6120` |
| R1K | H4 | `0.9187` | `0.0813` | `6.252` | `0.4944` | `0.6613` |
| R1G | default | `0.7063` | `0.2938` | `5.869` | `0.2388` | `0.2308` |
| R1G | strict125 | `0.3937` | `0.6062` | `8.926` | `0.2244` | `0.2308` |
| R1G | H1 | `0.9313` | `0.0688` | `4.745` | `0.4765` | `0.3528` |
| R1G | H2 | `0.9812` | `0.0188` | `2.788` | `0.5017` | `0.5345` |
| R1G | H3 | `0.9750` | `0.0250` | `3.217` | `0.5054` | `0.6341` |
| R1G | H4 | `0.9313` | `0.0688` | `5.962` | `0.4960` | `0.6836` |

相对同一训练 seed 的 R0 配对差值：

- R1K default：success `+0.0375`，但 RMSE `+0.449 m`、saturation `+0.0880`；strict125 success 仅 `+0.0063`，RMSE `+1.081 m`、saturation `+0.0841`。
- R1K H1：success `-0.0063`、RMSE `+0.904 m`；H3：success `+0.0063`、RMSE `-0.083 m`、saturation `+0.0016`；H4 与 R0 完全相同。
- R1G default success `+0.0500`，但 RMSE `+0.324 m`、saturation `+0.0894`；strict125 success `0.0000`、RMSE `+1.176 m`；H1 success `-0.0125`、RMSE `+1.368 m`。`initial_speed_ratio=1.10` 没有带来稳定收益。

判定：**held-out 评估没有复现 seeds `0–31` 上 R1K 的收益。** 主要问题不是 success 略升，而是 RMSE 和 saturation 同时退化；R0 自身在 held-out 上也只有 default `0.6562`、strict125 `0.3937`，说明先前种子集偏乐观。因此不能继续用 `0–31` 的单 seed 或五 seed 结果宣称 ALOS-R 通过，也不应继续细调终端/RPM 系数。当前控制层结论仍只支持：终端不降到 0 有效，但仅靠 ALOS 命令速度下限无法保证实际地速，且不能修复执行器饱和。

配对汇总：`artifacts/lowlevel/pc_alos_r_holdout32_compare_20261104/summary.{json,md}`；各 stage 报告分别在 `artifacts/lowlevel/pc_alos_r_holdout32_{r0,r1k,r1g}_20261104/report/{R0,R1K,R1G}/`。队列日志在 `logs/pc_alos_r_holdout32_{r0,r1k,r1g}_20261104/`。本轮只新增 `h2` 路由，没有改 ALOS-R 控制公式。

#### R4：起步预旋与初始速度校准（2026-11-04，已完成，fail）

新增 R4 是因为逐步 trace 把最低地速分成了不同阶段：H1/H3 的部分 run 在起步后 1–3 s 最低，default/strict125 的若干 run 在 7–16 s 的转向段最低；H1 的另一些 run 在错过终点后 `goal_distance` 再次增大的回追段最低。因此不能把一次 `initial_speed_ratio=1.10` 的负结果解释成“预旋无效”，也不能用命令速度下限替代实际地速。

R4 只隔离起步：不改变 actor 输入、动作包线、终端捕获、蟹行或饱和调速。相对 R1K 只改 `initial_speed_ratio=1.25` 和 `initial_propeller_rpm_ratio`，形成三条臂：

| 阶段 | 初速比例 | 起步 RPM | 目的 |
|---|---:|---:|---|
| `R1K` | `1.00×` | `0%` | 现有对照 |
| `R4A` | `1.25×` | `0%` | 只看初速 |
| `R4B` | `1.25×` | `75%` | 初速 + 温和预旋 |
| `R4C` | `1.25×` | `100%` | 初速 + 前馈预旋 |

初始速度使用实际 episode 巡航速度缩放，并在 `BatchedPCLowLevelEnv` 中钳位到 REMUS100 最大速度 `2.5 m/s`；因此 H3 的 `2.25–2.4 m/s` 巡航只能保留约 `1.04×` 的起步余量。R4 不通过拉长 terminal braking 或让 `goal_distance` 进入速度公式来伪造最低速度。

预注册门禁：

- smoke 使用 1 个 no-DADE 训练 seed × 8 个评估 seed，tags 为 `default strict125 h1 h3`；先看 `ground_speed_ratio_min`、`ground_speed_ratio_min_time_s`、success、timeout 和 saturation。
- 只有当 R4A/B/C 中至少一条相对 R1K 把起步段（`ground_speed_ratio_min_time_s <= 5 s`）的最低实际地速提高，且 default/strict 的 success 不降超过 `0.02`、H1/H3 的 saturation 不增超过 `0.02`，才扩到 5 个训练 seed × 32 个评估 seed。
- 正式验收同时报告：success、timeout、3D path RMSE、material saturation、yaw/speed tracking error、current estimate valid rate、crab angle、terminal speed ratio、全航线命令速度最低比、实际地速最低比及其发生时间。实际地速 `>= 0.75` 是目标硬约束；未达到时保留负结果，不能只报命令速度。
- 5-seed 标签至少为 `default strict125 h1 h3 h4 c4`，并继续使用 held-out 评估 seeds `32–63`，不能只用 `0–31` 筛选。

服务器 smoke 命令（在独立 ALOS-R 目录执行）：

```bash
cd /data/yhx/underwater-robotics/src/plan_p1/settled_局部规划器_反应式_alos_r
PY=/data/yhx/underwater-robotics/envs/venvs/isaaclab/bin/python
ARTIFACT_ROOT="$PWD/artifacts/lowlevel/pc_alos_r_r4_launch_smoke_20261104" \
STAGES="R4A R4B R4C" SEEDS=94030 \
TAGS="default strict125 h1 h3" SEED_COUNT=8 EVAL_WORKERS=4 \
  scripts/run_pc_alos_r_20261104.sh
```

smoke 完成后按 stage 汇总：

```bash
for stage in R4A R4B R4C; do
  "$PY" scripts/summarize_pc_alos_r.py \
    --eval-root artifacts/lowlevel/pc_alos_r_r4_launch_smoke_20261104/eval \
    --stage "$stage" \
    --output "artifacts/lowlevel/pc_alos_r_r4_launch_smoke_20261104/report/$stage"
done
```

若 smoke 通过，再跑 5-seed × 32 评估 seed；smoke 或正式结果未产生前，不把 R4 写成正结果。

**Smoke 结果（seed94030，评估 seeds 0–7，已与 R1K 严格配对）：**

| 阶段 | tag | success | 3D RMSE | saturation | ground min | ground min time | 判定 |
|---|---|---:|---:|---:|---:|---:|---|
| R1K | default | `0.8750` | `4.4143 m` | `0.2904` | `0.3737` | `3.94 s` | 对照 |
| R4A | default | `0.8750` | `4.0336 m` | `0.2923` | `0.3925` | `5.04 s` | 改善 RMSE，success 不降 |
| R4B | default | `0.7500` | `5.4508 m` | `0.2401` | `0.5079` | `12.89 s` | 预旋退化 |
| R4C | default | `0.6250` | `6.7776 m` | `0.2230` | `0.5180` | `12.86 s` | 预旋退化 |
| R1K | strict125 | `0.3750` | `8.9114 m` | `0.2727` | `0.3737` | `16.47 s` | 对照 |
| R4A | strict125 | `0.3750` | `8.8869 m` | `0.2725` | `0.3925` | `14.88 s` | 不变，仍 fail |
| R1K | h1 | `0.8750` | `4.5992 m` | `0.4813` | `0.4692` | `2.67 s` | 对照 |
| R4A | h1 | `1.0000` | `1.6961 m` | `0.4975` | `0.4989` | `2.95 s` | 明显改善，饱和仍超线 |
| R1K | h3 | `1.0000` | `2.0659 m` | `0.4953` | `0.6212` | `1.12 s` | 对照 |
| R4A | h3 | `1.0000` | `2.0588 m` | `0.5010` | `0.6429` | `1.21 s` | RMSE 持平，饱和仍超线 |

判定：**R4A 通过 smoke 的扩 seed 条件**（起步最低地速上升、success 不回退、saturation 增量 `<0.02`），但全航线实际地速仍只有 `0.39–0.64×`，strict125 和 H1/H3 saturation 均未过正式门禁。R4B/R4C 的预旋在 8-seed smoke 上导致 default/strict 退化，停止，不进入 5-seed。R4A 已在 held-out seeds `32–63` 上启动 5 个训练 seed × 6 个 tag 的独立队列，产物根为 `artifacts/lowlevel/pc_alos_r_r4a_holdout32_5seed_20261104/`。

**Held-out 5-seed × 32-eval 结果（评估 seeds 32–63，与 R1K 同训练 seed 配对）：**

| tag | R1K success / RMSE | R4A success / RMSE | R1K timeout / sat | R4A timeout / sat | R1K ground min | R4A ground min | 判定 |
|---|---:|---:|---:|---:|---:|---:|---|
| default | `0.6937 / 5.994 m` | `0.7188 / 5.663 m` | `0.3063 / 0.2374` | `0.2812 / 0.2415` | `0.2114` | `0.2517` | success/RMSE 小幅改善，实际地速仍 fail |
| strict125 | `0.4000 / 8.831 m` | `0.3937 / 8.945 m` | `0.6000 / 0.2260` | `0.6062 / 0.2246` | `0.2114` | `0.2517` | success/RMSE/timeout 全部退化 |
| h1 | `0.9375 / 4.280 m` | `0.9437 / 3.905 m` | `0.0625 / 0.4727` | `0.0563 / 0.4859` | `0.3348` | `0.3756` | success/RMSE 小幅改善，saturation 仍超线 |
| h3 | `0.9812 / 3.174 m` | `0.9750 / 3.343 m` | `0.0188 / 0.5047` | `0.0250 / 0.5057` | `0.6120` | `0.6252` | success/RMSE 退化，saturation 仍超线 |
| h4 | `0.9187 / 6.252 m` | `0.9187 / 6.252 m` | `0.0813 / 0.4944` | `0.0813 / 0.4944` | `0.6613` | `0.6613` | 与 R1K 相同，压力门禁 fail |
| c4 | — | `0.1688 / 12.114 m` | — | `0.8313 / 0.2634` | — | `0.1709` | 压力测试仍 fail |

判定：**R4A 没有跨 held-out 复现 smoke 的正收益。** default 的 success `+0.025`、RMSE `-0.331 m` 是唯一方向一致的小改善；strict125 success `-0.0063`、RMSE `+0.114 m`、timeout `+0.0062`，H3 success `-0.0062`、RMSE `+0.169 m`，都不通过。H1/H3/H4 的 saturation 和全航线实际地速仍远离验收线。R4 结论为 **fail**：起步预旋/初速只能改善起步段，不能解决中段转向、终端捕获和饱和，停止继续调初速/RPM 系数；不启动 R4B/R4C 的 5-seed。

#### RPM-only 残差权限隔离（2026-11-04，已完成，门禁失败）

为回答“full-action 残差的舵面误差是否掩盖了 RPM 学习信号”，新增 `mask=[0,0,1]` 的监督残差臂：冻结 BC 保持舵角/艉舵，只有 RPM 通道接收残差。五训练 seed `94030–94034`、32 评估 seed、`grid_lw/grid_ls/grid_hw/grid_hs/default/strict125/H1/H2/H3/H4/C4` 全部完成；训练审计 `failures=[]`，冻结 BC 最大漂移 `0.0`，25 个 run 的 rudder/stern 残差严格为 0，`grid_hs` 未泄漏。参考臂使用同一新版 evaluator 重评 55 个 run，success/timeout/RMSE/saturation 与旧缓存最大差值 `0.0`。

| Tag | RPM success / RMSE / timeout / sat | Full success / RMSE / timeout / sat | Route min ep-mean (RPM / Full) | Gate |
|---|---:|---:|---:|---|
| `grid_lw` | `1.0000 / 1.87 m / 0.0000 / 0.1940` | `0.9812 / 2.20 m / 0.0187 / 0.1837` | `0.7195 / 0.7041` | pass |
| `grid_ls` | `0.6312 / 4.93 m / 0.3688 / 0.1528` | `0.6625 / 4.82 m / 0.3375 / 0.1431` | `0.4595 / 0.4717` | fail |
| `grid_hw` | `0.9938 / 2.26 m / 0.0063 / 0.4595` | `0.9875 / 2.52 m / 0.0125 / 0.4797` | `0.6800 / 0.6835` | pass |
| `grid_hs` | `0.7875 / 7.62 m / 0.2125 / 0.3986` | `0.8438 / 6.11 m / 0.1562 / 0.4278` | `0.5135 / 0.5588` | fail |
| default | `0.6188 / 5.36 m / 0.3812 / 0.1409` | `0.6375 / 5.20 m / 0.3625 / 0.1359` | `0.4910 / 0.4989` | fail |
| strict125 | `0.4313 / 6.45 m / 0.5687 / 0.1312` | `0.4062 / 6.63 m / 0.5938 / 0.1249` | `0.4329 / 0.4373` | fail |
| H1 | `0.8750 / 6.18 m / 0.1250 / 0.4225` | `0.9250 / 4.36 m / 0.0750 / 0.4481` | `0.5685 / 0.6100` | fail |
| H2 | `0.8562 / 9.69 m / 0.1437 / 0.4108` | `0.9875 / 2.69 m / 0.0125 / 0.4992` | `0.5844 / 0.6643` | diagnostic |
| H3 | `0.5312 / 18.17 m / 0.4688 / 0.3224` | `0.9750 / 3.65 m / 0.0250 / 0.5029` | `0.4635 / 0.6484` | fail |
| H4 | `0.2375 / 22.09 m / 0.7188 / 0.3014` | `0.9625 / 4.32 m / 0.0375 / 0.4906` | `0.3018 / 0.6340` | pressure-only |
| C4 | `0.2188 / 8.89 m / 0.7812 / 0.2390` | `0.2062 / 8.83 m / 0.7937 / 0.2339` | `0.4466 / 0.4323` | pressure-only |

配对结论：`grid_lw` success `+0.0187`、RMSE 比 `0.8646`，`grid_hw` success `+0.0063`、RMSE 比 `0.9478`、saturation `-0.0202`，说明 RPM-only 在部分训练格确实有局部收益；但 `grid_ls` success `-0.0312` 超过训练格下降上限，held-out `grid_hs` success `-0.0563`、RMSE 比 `1.2738`，没有跨组合迁移。retention/robustness 更差：default success `-0.0187` 且实际航迹最低速度 `0.4910`；strict success 虽 `+0.0250`，但绝对 `0.4313`、timeout `0.5687`、实际航迹最低速度 `0.4329` 均未过线；H1 success `-0.0500`、RMSE 比 `1.5479`；H3 success `-0.4437`、RMSE 比 `6.2748`；H4 success `-0.7250`、RMSE 比 `8.4655`。H3/H4 的 saturation 下降是失速/超时的副产物，不能算执行器收益。

**判定：`overall_status=fail`。** 只修 RPM 不足以兑现实际地速，也不足以解决留出速度×海流和高速环境的闭环鲁棒性；不追加同配方调参。产物：`artifacts/lowlevel/pc_grid_rpm_only_{supervised_residual,eval,reference_eval,summary,report}_20261104/`，报告入口为 `artifacts/lowlevel/pc_grid_rpm_only_report_20261104/report.md`。

#### 运行方式

服务器独立目录中运行（不要覆盖主仓库）：

```bash
cd /data/yhx/underwater-robotics/src/plan_p1/settled_局部规划器_反应式_alos_r
export PYTHONPATH="$PWD/src:$PWD/../settled_全局规划器/src"

# R0 当前基线；R1 无终端减速；R1B 旧 20 m 捕获 + 75% 下限；固定 32 个评估 seed
STAGES="R0 R1 R1B" SEEDS=94030 SEED_COUNT=32 EVAL_WORKERS=4 \
  ARTIFACT_ROOT="$PWD/artifacts/lowlevel/pc_alos_r_floor75_20261104" \
  LOG_ROOT="$PWD/logs/pc_alos_r_floor75_20261104" \
  TRAIN_ROOT="/data/yhx/underwater-robotics/src/plan_p1/settled_局部规划器_反应式/artifacts/lowlevel/pc_multienv_supervised_residual_full_20261104" \
  bash scripts/run_pc_alos_r_20261104.sh

# 当前最优单 seed 配方：R1K = 20 m 捕获 + 地速反馈 + 75% 命令下限 + 初速 1.0× + RPM 0
STAGES="R1K" SEEDS=94030 TAGS="default strict125 h1 h3 c4" SEED_COUNT=32 EVAL_WORKERS=4 \
  ARTIFACT_ROOT="$PWD/artifacts/lowlevel/pc_alos_r_r1k_init100_32seed_20261104" \
  LOG_ROOT="$PWD/logs/pc_alos_r_r1k_init100_32seed_20261104" \
  TRAIN_ROOT="/data/yhx/underwater-robotics/src/plan_p1/settled_局部规划器_反应式/artifacts/lowlevel/pc_multienv_supervised_residual_full_20261104" \
  bash scripts/run_pc_alos_r_20261104.sh

# R1K 跨训练 seed 扩训：5 个 no-DADE seed × 32 个评估 seed × 6 个 tag
STAGES="R1K" SEEDS="94030 94031 94032 94033 94034" TAGS="default strict125 h1 h3 c4 h4" \
  SEED_COUNT=32 EVAL_WORKERS=4 \
  ARTIFACT_ROOT="$PWD/artifacts/lowlevel/pc_alos_r_r1k_5seed_32eval_20261104" \
  LOG_ROOT="$PWD/logs/pc_alos_r_r1k_5seed_32eval_20261104" \
  TRAIN_ROOT="/data/yhx/underwater-robotics/src/plan_p1/settled_局部规划器_反应式/artifacts/lowlevel/pc_multienv_supervised_residual_full_20261104" \
  bash scripts/run_pc_alos_r_20261104.sh

# R2B 蟹行 / R3B 调速 smoke；已跑过并失败，保留命令供复核
STAGES="R2B R3B" SEEDS=94030 TAGS="default strict125 h1 h3 c4" SEED_COUNT=8 EVAL_WORKERS=4 \
  ARTIFACT_ROOT="$PWD/artifacts/lowlevel/pc_alos_r_r2b_r3b_smoke_20261104" \
  LOG_ROOT="$PWD/logs/pc_alos_r_r2b_r3b_smoke_20261104" \
  TRAIN_ROOT="/data/yhx/underwater-robotics/src/plan_p1/settled_局部规划器_反应式/artifacts/lowlevel/pc_multienv_supervised_residual_full_20261104" \
  bash scripts/run_pc_alos_r_20261104.sh

# H4 压力测试单独入队（本轮五 seed 结果已产出）
STAGES="R1K" SEEDS="94030 94031 94032 94033 94034" TAGS="h4" SEED_COUNT=32 EVAL_WORKERS=4 \
  ARTIFACT_ROOT="$PWD/artifacts/lowlevel/pc_alos_r_r1k_h4_5seed_32eval_20261104" \
  LOG_ROOT="$PWD/logs/pc_alos_r_r1k_h4_5seed_32eval_20261104" \
  TRAIN_ROOT="/data/yhx/underwater-robotics/src/plan_p1/settled_局部规划器_反应式/artifacts/lowlevel/pc_multienv_supervised_residual_full_20261104" \
  bash scripts/run_pc_alos_r_20261104.sh

# held-out 评估：seeds 32-63；三个 stage 并行，各自使用独立 CPU 核段
for spec in "R0 64" "R1K 144" "R1G 224"; do
  read -r stage cpu_base <<< "$spec"
  lower="${stage,,}"
  STAGES="$stage" SEEDS="94030 94031 94032 94033 94034" \
    TAGS="default strict125 h1 h2 h3 h4" SEED_START=32 SEED_COUNT=32 \
    EVAL_WORKERS=4 CPU_BASE="$cpu_base" CPU_STRIDE=16 \
    ARTIFACT_ROOT="$PWD/artifacts/lowlevel/pc_alos_r_holdout32_${lower}_20261104" \
    LOG_ROOT="$PWD/logs/pc_alos_r_holdout32_${lower}_20261104" \
    TRAIN_ROOT="/data/yhx/underwater-robotics/src/plan_p1/settled_局部规划器_反应式/artifacts/lowlevel/pc_multienv_supervised_residual_full_20261104" \
    bash scripts/run_pc_alos_r_20261104.sh &
done
wait

# 三条 stage 完成后做按训练 seed 配对的 R1K/R1G vs R0 汇总
"$PY" scripts/summarize_pc_alos_r_holdout_20261104.py \
  --artifact-root "$PWD/artifacts/lowlevel" \
  --output "$PWD/artifacts/lowlevel/pc_alos_r_holdout32_compare_20261104"
```

产物：

- 第一轮服务器评测/报告：`artifacts/lowlevel/pc_alos_r_20261104/{eval,report}/{R0,R1}`；
- 75% 约束服务器评测/报告：`artifacts/lowlevel/pc_alos_r_floor75_20261104/{eval,report}/{R0,R1,R1B}`；
- 实际航迹速度比复核：`artifacts/lowlevel/pc_alos_r_floor75_actual_20261104/{eval,report}/{R0,R1B}`（新指标不覆盖旧报告）；
- 起速与地速反馈 32-seed 因子：`artifacts/lowlevel/pc_alos_r_initfloor_2x2_32seed_20261104/report/{R1G,R1H,R1I,R1J}`；
- R1K 单 seed 32-seed：`artifacts/lowlevel/pc_alos_r_r1k_init100_32seed_20261104/report/R1K`；
- R1K 五 seed 32-seed：`artifacts/lowlevel/pc_alos_r_r1k_5seed_32eval_20261104/report/R1K`；
- R1K 五 seed H4 压力测试：`artifacts/lowlevel/pc_alos_r_r1k_h4_5seed_32eval_20261104/report/R1K`；
- R2B/R3B 8-seed smoke：`artifacts/lowlevel/pc_alos_r_r2b_r3b_smoke_20261104/report/{R2B,R3B}`；
- held-out `32–63`：`artifacts/lowlevel/pc_alos_r_holdout32_{r0,r1k,r1g}_20261104/report/{R0,R1K,R1G}`；配对汇总 `artifacts/lowlevel/pc_alos_r_holdout32_compare_20261104/summary.{json,md}`；
- 队列日志：`logs/pc_alos_r_floor75_20261104/queue.log`；
- 本地只同步 `report/*/summary.{json,md}`，不同步 checkpoint 和完整 rollout。

#### 验收线

- [x] default：R1K 五 seed 均值 success `0.8000 >= 0.70`、RMSE `4.8163 <= 5.0 m`；
- [ ] held-out default：R1K 在评估 seeds `32–63` 上 success `0.6937 < 0.70`、RMSE `5.994 > 5.0 m`，未通过；
- [ ] strict125：R1K 五 seed 均值 success `0.4938 < 0.50`，未通过；单 seed 的 `0.5000` 不作为稳定结论；
- [ ] strict125：五 seed 均值 RMSE `7.5770 > 6.5 m`，未通过；
- [ ] held-out strict125：R1K success `0.4000 < 0.50`、RMSE `8.831 > 6.5 m`、timeout `0.6000 > 0.531`，未通过；
- [ ] H1：R1K 五 seed 均值 success `0.8875 < 0.95`、RMSE `5.3728 m`、saturation `0.4705 > 0.40`，未通过；
- [ ] H3：R1K 五 seed 均值 success `0.9625`、RMSE `3.2171 m`、saturation `0.4982 > 0.45`，未通过；
- [ ] H4：R1K 五 seed 均值 RMSE `4.3161 > 3.8 m`、saturation `0.4962 > 0.45`，未通过；
- [ ] held-out H1/H3/H4：R1K 在评估 seeds `32–63` 上分别为 H1 `0.9375 / 4.280 m / sat 0.4727`、H3 `0.9812 / 3.174 m / sat 0.5047`、H4 `6.252 m / sat 0.4944`，均未通过硬门禁；
- [x] 所有 ALOS-R 结果同时报告 success、timeout、三维 RMSE、material saturation、yaw/speed tracking error、current valid rate、crab angle、terminal speed ratio 和全航线最低命令速度比；
- [x] 所有已跑 tag 的 `guidance_route_speed_ratio_min >= 0.75`（R1B 最低 `0.7659`）；
- [x] 新增实际地速模长/航迹投影速度诊断并定位起步与转向瞬态；不把该诊断直接当作命令层硬门禁；
- [ ] 真实地速最低比 `>= 0.75`：R1K 五 seed 的 default/strict 仅 `0.3769`，未通过；不能把命令速度下限写成实际速度下限；
- [x] R1/R1B 单 seed × 32 评估 seeds 已完成；R2/R3 未启动（前置门禁失败）；
- [x] R1C/R1E/R1F 与 R1G–R1K 单 seed × 32 评估 seeds 已完成；
- [x] R2B/R3B 8-seed smoke 已完成并失败，不进入 32-seed 或五 seed；
- [x] R1K 五 seed × 32 评估 seed 稳定性扩训已完成；H4 独立五 seed 压力测试也已完成。

#### Git 提交与推送步骤

本代理没有自动 commit/push。建议按文件粒度提交，先验证再执行：

```bash
cd /home/bubble/水下机器人

git add \
  'settled_局部规划器_反应式/src/auv_lowlevel/config.py' \
  'settled_局部规划器_反应式/src/auv_lowlevel/batch/geometry.py' \
  'settled_局部规划器_反应式/src/auv_lowlevel/batch/guidance.py' \
  'settled_局部规划器_反应式/src/auv_lowlevel/batch/pc_env.py' \
  'settled_局部规划器_反应式/src/auv_lowlevel/batch/pc_vec_env.py' \
  'settled_局部规划器_反应式/scripts/evaluate_pc_ppo.py' \
  'settled_局部规划器_反应式/scripts/run_pc_alos_r_20261104.sh' \
  'settled_局部规划器_反应式/scripts/run_pc_multienv_eval_5seed_20261104.sh' \
  'settled_局部规划器_反应式/scripts/run_pc_multienv_dade_history_5seed_20261104.sh' \
  'settled_局部规划器_反应式/scripts/summarize_pc_alos_r.py' \
  'settled_局部规划器_反应式/scripts/summarize_pc_alos_r_holdout_20261104.py' \
  'settled_局部规划器_反应式/tests/test_batch_alos_r.py' \
  'settled_局部规划器_反应式/configs/lowlevel/odarrl_control_alos_r1_terminal.json' \
  'settled_局部规划器_反应式/configs/lowlevel/odarrl_control_alos_r1b_legacy20_floor75.json' \
  'settled_局部规划器_反应式/configs/lowlevel/odarrl_control_alos_r1c_ground_floor75.json' \
  'settled_局部规划器_反应式/configs/lowlevel/odarrl_control_alos_r1d_ground_floor_crab75.json' \
  'settled_局部规划器_反应式/configs/lowlevel/odarrl_control_alos_r1e_terminal20_prespin75.json' \
  'settled_局部规划器_反应式/configs/lowlevel/odarrl_control_alos_r1f_terminal20_noprespin75.json' \
  'settled_局部规划器_反应式/configs/lowlevel/odarrl_control_alos_r1g_ground_floor_init110_nospin.json' \
  'settled_局部规划器_反应式/configs/lowlevel/odarrl_control_alos_r1h_ground_floor_init110_prespin75.json' \
  'settled_局部规划器_反应式/configs/lowlevel/odarrl_control_alos_r1i_noground_init110_nospin.json' \
  'settled_局部规划器_反应式/configs/lowlevel/odarrl_control_alos_r1j_noground_init110_prespin75.json' \
  'settled_局部规划器_反应式/configs/lowlevel/odarrl_control_alos_r1k_ground_floor_init100_nospin.json' \
  'settled_局部规划器_反应式/configs/lowlevel/odarrl_control_alos_r4a_launch_init125_nospin.json' \
  'settled_局部规划器_反应式/configs/lowlevel/odarrl_control_alos_r4b_launch_init125_prespin75.json' \
  'settled_局部规划器_反应式/configs/lowlevel/odarrl_control_alos_r4c_launch_init125_prespin100.json' \
  'settled_局部规划器_反应式/configs/lowlevel/odarrl_control_alos_r2_crab.json' \
  'settled_局部规划器_反应式/configs/lowlevel/odarrl_control_alos_r2b_crab_floor_init100.json' \
  'settled_局部规划器_反应式/configs/lowlevel/odarrl_control_alos_r3_governor.json' \
  'settled_局部规划器_反应式/configs/lowlevel/odarrl_control_alos_r3b_governor_floor_init100.json' \
  'settled_局部规划器_反应式/doc/goals/goal9_odarrl_control_reproduction.md' \
  'settled_局部规划器_反应式/doc/goals/goal9_pc_server_validation_plan.md'
git diff --cached --check
git commit -m "feat(pc): add ALOS-R terminal and launch-speed calibration"
git push origin main
```

不要 `git add -A`；不要提交 `artifacts/`、`logs/`、checkpoint 或完整 rollout。

### 14.4.17 DADE 闭环 DAgger：留出收益没有复现（2026-11-04，已完成，fail）

goal13 的 DADE-pose 曾在严格留出的 `grid_hs` 上给出 success `0.9000` 对
no-DADE `0.8438`、RMSE `4.90` 对 `6.11 m` 的正信号。S13 因此用 DADE-pose
学生 `final.pt` 在 `grid_lw/grid_ls/grid_hw` 上执行闭环 DAgger：student 加
`0.05` 归一化动作噪声访问闭环状态，采集 teacher 标签，与原 demos 合并后重新
监督同一个 DADE 残差结构；`grid_hs` 继续严格留出。没有新增第二个残差网络，
没有改 actor 输入维度，也没有把真实海流送入 actor。

五训练 seed `94030–94034` × 32 评估 seed 的 DADE 基线和 DAgger 各 `110/110`
评测完成。审计 `failures=[]`，冻结 BC 参数最大漂移 `0.0`，每个 run 只有
`lw/ls/hw` 三个唯一训练格，每格都有原始 demos 和 DAgger demos 两路，`grid_hs`
没有进入采集或训练。汇总 `220` 行，无非有限值。

| Tag | DAgger success / RMSE / timeout / sat | DADE baseline success / RMSE / timeout / sat | Gate |
|---|---:|---:|---|
| `grid_lw` | `0.9750 / 2.16 m / 0.0250 / 0.1658` | `1.0000 / 1.88 m / 0.0000 / 0.1574` | fail |
| `grid_ls` | `0.6438 / 4.94 m / 0.3563 / 0.1344` | `0.6813 / 4.79 m / 0.3187 / 0.1274` | fail |
| `grid_hw` | `0.9938 / 2.19 m / 0.0063 / 0.4736` | `0.9750 / 3.48 m / 0.0250 / 0.4607` | pass |
| `grid_hs` | `0.8875 / 5.38 m / 0.1125 / 0.4428` | `0.9000 / 4.90 m / 0.1000 / 0.4382` | fail |
| `default` | `0.6625 / 5.10 m / 0.3375 / 0.1244` | `0.6813 / 4.96 m / 0.3187 / 0.1171` | fail |
| `strict125` | `0.4375 / 6.45 m / 0.5625 / 0.1150` | `0.4562 / 6.36 m / 0.5437 / 0.1074` | fail |
| `h1` | `0.9250 / 4.60 m / 0.0750 / 0.4483` | `0.9437 / 3.76 m / 0.0563 / 0.4455` | fail |
| `h3` | `1.0000 / 1.83 m / 0.0000 / 0.5196` | `1.0000 / 1.84 m / 0.0000 / 0.5098` | fail |

`grid_hs` 的配对 success delta 为 `-0.0125`（逐 seed `-0.0625~+0.03125`），
RMSE ratio `1.1018`，saturation `+0.0046`；既没有达到 `+0.02`/`0.95×` 的迁移
阈值，绝对 success `0.8875 < 0.90` 和 RMSE `5.38 > 5.0 m` 也未过线。训练格
`grid_lw`/`grid_ls` 的 success 分别下降 `0.0250`/`0.0375`，只有 `grid_hw`
改善。default/strict125 的 success、RMSE、timeout 均退化；H1 RMSE ratio
`1.2451` 且 saturation `0.4483 > 0.40`；H3 的 success/RMSE 持平但 saturation
`0.5196 > 0.45`。H4 压力测试 RMSE `2.61 m`、saturation `0.5126`，仍未改善。

DAgger 相对同 run 的零残差 `initial` 确实学到东西：`grid_hs` success `+0.0438`、
RMSE ratio `0.8885`，H3 success `+0.15625`、RMSE ratio `0.1828`。但这些收益
已经被现有的 DADE-pose 基线包含，DAgger 没有提供增量，反而在多个保留环境退化。
因此判定 **fail**，停止同配方轮次，不追加第二轮 DAgger、不调动作噪声或
DADE dropout/head 来挑 seed。下一步应处理实际地速跟踪、执行器权限/饱和奖励或
新的控制层设计，而不是继续重复“DADE + 监督 DAgger”。

产物：`artifacts/lowlevel/pc_dade_grid_dagger_{demos,supervised_residual,eval,summary,report}_20261104/`；
训练审计：`artifacts/lowlevel/pc_dade_grid_dagger_supervised_residual_20261104/audit.json`；
完整说明：`doc/goals/goal14_dade_closed_loop_dagger.md`。未自动 commit/push。

### 14.4.18 ABC 前置 P0/P1/P2/P3 与 A/B/C 时间感受野、决策频率、Transformer actor 组（2026-11-05，已完成：S15 冻结 A0_stride1，C=not_triggered）

状态：**已完成。** 它只新增实验，不改写 14.4.1–14.4.17 的历史结论。
本组必须先完成 **P0/P1/P2/P3 前置门禁**，再执行 A/B/C；P0/P1/P2/P3 只做低预算筛查，
不能替代正式 3×32 或 CPU PVS 64-seed。P0/P1/P2/P3 未完成前，暂停新增 DADE/DAgger、
重复 S14 配方和盲目扩大训练步数；A/B/C 也不能并行改多个变量。截至 2026-11-05，P0 和
P1 已在服务器完成正式 screen；P0 为 `no attributable signal`，P1 为
`no_path_attribution`，路径分支停止；P2 随后完成，encoded-history DADE 判为
`no_signal`，后续保留 raw-history `A0_stride1`；A/B/C 已完成，A 无因果信号、
B10 为效率非劣、B5 失败、C=`C_not_triggered`，最终冻结 `A0_stride1`。S16-V0 随后
执行并失败，V 分支按协议停止；完整结论见 `doc/goals/goal9_final_report.md`。

#### 前置工作：P0/P1/P2/P3（必须先于 A/B/C）

前置组只回答三个问题：actor 是否缺少路径信息、历史 token 是否需要先编码、奖励是否在
阻碍已有信息的使用。它不回答 A/B/C 的时间尺度问题，也不把“输入、DADE、奖励”混在
同一轮实验里。每一步只改一个变量；未过门禁就停止该分支，不用加步数、换评估 seed、
调阈值或加入第二个变量来救结果。

**共同契约**

- 主环境固定为 `odarrl_control_pc_c3_random_realistic_v1_rewardscale001.json`；C2 mild
  只做回归，不作为主判据。
- P0/P1/P2/P3 都是低预算筛查：1 个训练 seed × 20 个固定评估 seed（`1000–1019`）。
  只有最终唯一候选才进入 3 个训练 seed × 32 个评估 seed 和 CPU PVS 64-seed。
- P0/P1/P2/P3 之间保持同 seed、同训练物理秒数、同更新次数和同评估 seed；每个 arm
  必须有独立输出目录，不覆盖 A/B/C 或历史结果。
- actor 只能使用可部署观测。路径特征必须由 `sensor_output.p_nav` 投影得到；禁止把
  真实海流、故障标签、真值 `ground_velocity_body_mps` 或仿真内部状态送入 actor。
- P0/P1/P2/P3 的任何 Torch 结果只能作为预筛，不能写成正式性能结论；最终仍必须经过
  同口径 CPU PVS。
- 新产物根目录：`artifacts/lowlevel/pc_abc_prework_20261105/`；A/B/C 正式结果仍写入
  `artifacts/lowlevel/pc_abc_temporal_20261105/`。两套目录不能互相覆盖。

**P0：A 组短筛查（2026-11-05 已完成，无可归因信号，P1 使用 A0）**

目的：先判断 1 s 历史是否明显不足，同时保留后续 P1/P2/P3 的干净基线。P0 只使用
`14.4.18` 后面 A 组定义的 `A0_stride1`、`A1_stride5`、`A2_stride10`，不启动 B/C。

- [x] 运行 A0/A1/A2 契约 smoke：1 个训练 seed × 8 个评估 seed，检查历史 stride、物理
  时长、动作频率、reward 时间尺度、no-privilege-leak 和有限性；
- [x] 运行 A0/A1/A2 的 1 个训练 seed × 20 个评估 seed，逐 seed 配对比较 success、
  strict125、timeout、3D RMSE、safety 和 material saturation；
- [x] 若 A1/A2 没有达到 `+0.05` success/strict125 或 `-0.10 m` RMSE，记录负结果，
  后续以 `A0_stride1` 为基线；不要为了“再看看”直接跑 3×32；
- [x] 若 A1/A2 有可解释信号，只保留最佳 stride 进入 P1/P2/P3；schema 冻结后仍要用
  同 seed 重跑 `A0` 与该最佳 stride，不能跨 schema 直接比较；
- [x] 产物写入 `artifacts/lowlevel/pc_abc_prework_20261105/P0_a_screen/`，包含
  `summary.json`、`summary.md`、manifest、配置/checkpoint hash 和 finite audit。

结果：服务器 smoke 与 screen 的 training audit 均 `passed=true`，三组各完成
`32 × 128 × 256 = 1,048,576` 环境步，评估 seed 为 `1000–1019`。A1 的 default success
delta 为 `0.0000`、3D RMSE 改善 `0.0175 m`；A2 的 default success delta 为 `0.0500`、
3D RMSE 改善 `0.4271 m`，但 `history-zero` 和 `history-reverse` 都没有显示可解释退化，
因此 A2 只能登记为相关信号，不能据此宣称时间感受野的因果收益。按预注册规则 P0 判为
**no attributable signal**，P1 基线使用 `A0_stride1`。首次 screen 的两次 strict125
历史消融因 CPU 绑定编号超出 384 核失败；修复 runner 槽位分配后只重跑同一命令，科学
预算和评估 seed 未改变。

**P1：路径观测 pilot（2026-11-05 screen 完成，`no_path_attribution`）**

目的：只检验“actor 缺少可部署的路径相对信息”是否是瓶颈。P1 不改变 DADE，不改变
奖励，不改动作频率；基线是 P0 选出的最佳 A 配置（无信号时用 A0）。

建议先加 6–8 维，不要一次扩到十几维：

- signed cross-track error：`projection.horizontal_error_m`；
- signed vertical error：`projection.vertical_error_m`；
- 路径切向 heading error：`wrap(psi_tangent - psi)`；
- 路径切向 pitch error；
- along-track progress 或剩余弧长；
- curvature × lookahead，或 lookahead tangent 的 heading/pitch 变化量；
- 可选路径坐标系 DVL 速度，但必须由 `dvl_ground_body + attitude` 计算，不能使用真值
  `ground_velocity_body_mps`。

实现契约：

- **路径特征必须作为 `o_t^a` 的一部分送入 actor encoder，并编码进 `z_t^a`：**
  ```text
  o_t^a = [o_base_t, path_t]
  z_t^a = E_a(o_t^a)
  ```
  路径列复用编码器预留输入槽位，或作为一个独立的 encoder 输入 block；无论采用哪种
  布局，最终 actor 只能使用编码后的 `z_t^a`，不能把 `path_t` 直接拼到 actor 顶层，
  也不能让 `path_t` 绕过 encoder 单独进入 DADE。可选的实现方式是 encoder 内部融合
  `z_t^a = f_base(o_base_t) + alpha * f_path(path_t)`；`alpha` 用很小的非零初值
  （例如 `0.01`），或让 `f_path` 最后一层零初始化但保持 `alpha=1`。不要设置
  `alpha=0`，否则路径编码器初始阶段没有梯度，容易重演 DADE 零初始化死梯度；
  不要把路径列直接 append 到 12D state，因为现有 `actor_state_mode='pose'` 和 DADE
  `pose` 模式会按前 6 列截断，路径列会被当成速度列丢掉（见
  `src/auv_lowlevel/batch/pc_policy.py:531` 和 `pc_policy.py:548`）；
- P1 只改 actor；critic 保持现有 15D + true current，避免把 value 函数改动混入路径输入
  的归因；若以后要测 critic 路径输入，另立 P1b；
- 路径特征必须在 `pc_env.py` 的 ALOS 投影之后构造，并记录归一化尺度；新增字段必须
  更新 schema hash、checkpoint metadata 和单测，不能只改运行时张量；
- 评估时增加 `path-zero` 消融：在送入 encoder 前把 `o_t` 的路径列置零，并同时作用于
  当前编码和 DADE 历史编码；不能只在最终 `z_t` 上做减法。若指标与完整路径输入几乎
  相同，则不能宣称“路径信息有效”，只能判为未使用或无关。

- [ ] 实现路径观测分支、配置、归一化、schema/checkpoint 兼容和单测；
- [x] 验证 `o_t^a` 的新宽度、预留槽位和 `z_t^a` 的维度；确认 `path_t` 没有任何绕过
  encoder 的 actor/DADE 直连路径；
- [x] 1 个训练 seed × 20 个评估 seed，与 P0 最佳 A 逐 seed 配对；
- [x] 运行 `path-zero` 消融并记录动作敏感度；
- [x] 通过阈值与 P0 相同：success 或 strict125 至少 `+0.05`，或 3D RMSE 至少改善
  `0.10 m`；safety/material saturation 增幅不超过 `0.02`，timeout 增幅不超过 `0.05`；
  `path-zero` 必须显示可解释退化；
- [x] 若 P1 fail，停止路径特征分支并保留负结果；不通过继续加特征、改 scale 或换 seed
  来救结果。P2 只有在另有明确的历史表征瓶颈证据时才继续。

结果：P1 smoke 的 `P1_full` 与 `P1_curriculum` 均完成 32 iterations，training audit
均通过，最终 schema 相同，四类路径消融齐全。正式 screen 使用训练 seed `91430`、评估
seeds `1000–1019` 和 `32 × 128 × 256` 预算；`P1_full` 的 training audit 为
`passed=true`。相对 `A0_stride1`，default success delta 为 `+0.0500`，3D RMSE 改善
`0.4308 m`，timeout delta 为 `-0.0500`，safety delta 为 `0`，material saturation
delta 为 `+0.0053`；strict125 success delta 为 `0`。primary signal 和 guardrails 都
通过，但 default `path-zero` 的 success 下降为 `-0.0`、3D RMSE 仅恶化 `3.13e-05 m`，
动作 L2 平均变化约 `4.70e-05`，`path-geom-zero` 与 `path-context-zero` 也未显示可解释
退化。因此路径归因为 **false**，stage decision 为 `no_path_attribution`、
`selected_arm=null`、`next_action=stop_path_branch_and_write_negative_result`。
`P1_curriculum` screen 因 `P1_full` 未通过路径归因门禁而按 runner 规则主动跳过，不是
curriculum 训练失败。完整证据见
`doc/goals/goal9_p1_path_observation_report.md`。训练预算、训练 seed、评估 seed 和阈值
均未因结果调整。P2 基线回到 P0 的 `A0_stride1`，不再继续增加路径变量。

**P1 的模块化观测与课程学习（实现契约，不新增强制门禁）**

编码后的 `z_t` 是固定宽度接口，因此 P1 必须把观测按“可开关的 block”实现，而不是
把路径特征做成一个不可拆的整体张量。建议至少分为：

- `state_base`：机器人状态，始终启用；
- `path_geom`：signed cross-track、signed vertical、路径切向 heading/pitch error；
- `path_context`：unwrapped progress/remaining arc、曲率、lookahead 切向变化；
- `path_velocity`：可选的路径坐标系 DVL 速度，默认关闭。

每个 block 都要有独立的 `active_mask`、归一化参数、schema hash 和固定的 `z_t` 投影
槽位；禁用某个 block 时，应在送入 encoder 前置零并屏蔽梯度，不能改变 actor 顶层
宽度，也不能改变 DADE 的 token 宽度。这样可以在同一套 checkpoint metadata 下做
`block-zero`、路径消融和课程式启用。

**P1 允许两种执行模式，二者使用相同的最终观测 schema 和相同总训练预算：**

- `P1-full`：从第一步起启用完整路径 block，作为归因基线；
- `P1-curriculum`：按下面的阶段逐步启用路径 block，作为阶段性训练候选。

课程阶段为：

1. 先用 `state_base` 做短程热启动；
2. 启用 `path_geom`，学习横向/垂向误差和切向方向；
3. 启用 `path_context`，加入进度、剩余距离、曲率和 lookahead；
4. 最后再考虑 `path_velocity`；没有明确收益就保持关闭。

`P1-curriculum` 与 `P1-full` 必须使用相同的总物理步数、训练 seed、评估 seed 和更新
次数；阶段切换点必须预注册，不能看评估结果临时延长或跳过阶段。不能因为分阶段训练而
获得额外预算。阶段切换只通过 `active_mask` 完成，最终阶段必须启用与 `P1-full` 相同的
block 集合；记录每个阶段的 active blocks、归一化、checkpoint hash 和评估结果。

如果资源只够跑一个 P1 arm，先跑 `P1-full`，因为它回答“路径输入本身有没有用”；如果
资源允许，再跑同预算 `P1-curriculum`，回答“分阶段启用是否更容易收敛”。课程学习不能
与 P2 的 DADE-z 重构或 P3 的奖励改动同时进行，否则无法归因。

**P2：DADE-on-z_t 消融**

目的：把历史 token 从 raw state 改成编码后的 `z_t` 序列，验证统一表征是否优于原始
历史 token。P2 不是性能提升的默认假设，必须证明相对 P1 有增量；仅仅“不退化”不算
通过。基线是 P1 通过后的配置；P1 fail 时以 P0 最佳 A 为基线，且不继续加路径变量。

目标接口：

```text
z_t = E(o_t)                         # o_t 包含 base state；P1 通过时也包含 path_t
token_t = [z_t, side_t]              # side_t 保持原始估计海流，不编码
DADE(short, long) -> c_hat_t
actor input = [z_t, command_t, c_hat_t]
```

P1 通过后，`path_t` 已经是 `o_t` 的组成部分，因此路径信息会随 `z_t` 一起进入当前
actor 输入和历史 token；DADE 不得再额外接收未经编码的 `path_t` 旁路。P2b 比较的是
“raw state history”与“包含路径信息的编码 `z_t` history”，不是只编码基础状态。

- [x] P2a：保留 raw-history DADE，作为同 seed 基线；
- [x] P2b：把 short/long 历史中的 state 过共享 actor encoder，side channel 保持原值；
- [x] P2c：在 P2b 上增加 `history-zero` 消融，证明收益来自历史而不是新增参数；
- [x] 历史编码在 policy 前向内部完成，环境仍缓存 raw token；这样不改变 `pc_vec_env`
  的 observation buffer 布局；
- [x] 记录 DADE 输入宽度、checkpoint schema 变化和 p50/p95 决策时延；DADE 需要重新
  训练，不能假装旧 DADE 权重可直接严格加载。

注意：当前 `DualHorizonEncoder.forward` 会直接 `del current`（见
`src/auv_lowlevel/batch/encoded_policy.py:177`），所以 P2b 的准确含义是“编码历史
token”，不是“让 current z_t 参与 DADE 融合”。如果要做 current-z 作为 query，必须
另立 P2d，不能把它混进 P2b。现有 `dade_state_mode='pose'` 依赖原始列的截断语义；
改成 `z_t` 后不能再用同一套按列丢弃规则，必须在编码前做 base-state mask，或明确在
P2 关闭 pose 模式并记录原因。

- [x] P2b 相对 P2a：success 或 strict125 至少 `+0.05`，或 3D RMSE 至少改善 `0.10 m`
  （已评估：未达阈值，判 `no_signal`）；
  safety/material saturation 增幅不超过 `0.02`，timeout 增幅不超过 `0.05`；
- [x] p95 决策时延必须在当前 50 Hz 预算内；若超预算，直接 fail，不靠减少评估 seed
  或换 tag 掩盖；
- [x] 若 P2b 没有稳定增量，保留 raw-history DADE，停止 DADE-z 重构；不继续加层、加
  head 或换共享/独立编码器来挑结果。

**P2 执行结果（2026-11-05，服务器完成，`no_signal`）**

- P2b smoke 与 screen 均使用训练 seed `91430`、评估 seed `1000–1019`；screen 训练为
  `32 × 128 × 256`，训练审计、finite audit、schema/hash 和四项评估 job 全部通过；
- 相对 P2a `A0_stride1`，default success delta `0.000000`、3D RMSE 改善 `0.002811 m`；
  strict125 success delta `0.000000`、3D RMSE 改善 `0.018726 m`；safety/timeout 增量
  均为 `0.0`，saturation 增幅 `0.001035`，未达到任何 primary 阈值；
- `history-zero` 在 default/strict125 的 success drop 均为 `0.0`，RMSE 变化分别为
  `-0.000648 m` 和 `-0.002151 m`，没有可解释历史归因；p95 latency `1.066 ms`，
  latency gate 通过；
- 判定 `status=no_signal`、`selected_arm=A0_stride1`、
  `next_action=keep_raw_history_and_continue_to_formal_A`。不追加训练步数、seed、
  编码器或 head；P2 的负结果保留，不能写成 DADE-z 收益。
- P3 随即正式记录为 `not_applicable`：P1 `no_path_attribution`，P2b 使用
  `path_dim=0`，路径奖励的前置证据不成立。

**P3：奖励 pilot（最后做；当前默认不适用）**

P3 只在 P1/P2 已显示“路径信息被使用但仍有明确路径误差”时启动。奖励改动代码少，但会
改变训练目标，必须作为独立 arm，不能静默修改 `pc_command_track` 基线。P1 已判为
`no_path_attribution`，因此当前默认前置条件不成立；只有 P2 产生新的、可审计的路径使用
证据时，才允许重新评估是否启动 P3。否则必须在 P2 后正式记录 P3 为“不适用/前置条件
不满足”，不能静默跳过，也不能为了跑满阶段而加入路径奖励。

- 不加入 goal-distance、path-progress bonus 或 timeout bonus；这些项容易诱发冲终点、
  抄近路或忽略 ALOS 指令；
- 只做一个固定小权重的 clipped 路径误差惩罚，公式固定为
  `r_path = -w_path_pc * mean(clip(e_lat/5, -1, 1)^2 + clip(e_vert/5, -1, 1)^2)`；
  `w_path_pc` 初值固定为 `0.05`，不做权重网格搜索；若 smoke 出现 command tracking
  明显退化即停止；
- P3 的基线是 P2 胜出配置（P2 无增量时用 P1 配置）；除奖励外，输入、DADE、seed、
  训练预算和评估 seed 全部保持不变；
- 低层奖励仍以 command tracking 为主；若未来要做 path-aware objective，必须另立
  “end-to-end guidance+control”臂，重新定义基线和门禁。

- [ ] 新增独立 reward arm 配置、reward breakdown 和单测，保留原 `pc_command_track`
  配置完全不变；
- [ ] 1 个训练 seed × 20 个评估 seed，与基线逐 seed 配对；报告 command error、路径
  误差、success、strict125、timeout、safety、saturation 和动作平滑度；
- [ ] 通过阈值与 P1 相同；若 reward 只改善路径误差但 success/strict125/RMSE 没有
  改善，判 fail，不能把诊断指标改善写成控制能力提升；
- [ ] 若 P3 fail，停止奖励分支，不再调 `w_path_pc` 或增加第二个路径奖励项。

**前置组退出条件**

- [x] P0 完成：得到 P1/P2/P3 的统一基线，且没有用旧 schema 的全量 A 结果替代；
- [x] P1 完成：路径输入通过并证明 `path-zero` 有因果退化，或明确记录路径分支 fail；
- [x] P2 完成：DADE-z 相对 raw-history 有稳定增量，或明确保留 raw-history；
- [x] P3 完成：奖励 arm 通过，或明确保留原 `pc_command_track`；
- [x] schema、归一化、checkpoint metadata 和产物目录冻结后，才允许启动 A/B/C 正式组；
- [x] 任一 P 阶段 fail 都写负结果报告，不把失败分支带入下一阶段，也不通过加训练步数
  改变判定。

**产物与 Git**

```text
artifacts/lowlevel/pc_abc_prework_20261105/
  P0_a_screen/
  P1_path_obs/
  P2_dade_z/
  P3_path_reward/
  smoke/
  report/
```

每个阶段完成后按文件级命令提交，不使用 `git add -A`；大 checkpoint、完整 rollout 和
event 文件只留服务器，Git 只提交配置、脚本、小型 summary 和报告。示例：

```bash
cd /home/bubble/水下机器人/settled_局部规划器_反应式
git status --short
git add doc/goals/goal9_odarrl_control_reproduction.md \
        doc/goals/goal9_pc_server_validation_plan.md \
        configs/lowlevel/<prework-config>.json \
        src/auv_lowlevel/<changed-source>.py \
        tests/<changed-test>.py
git diff --cached --check
git commit -m "feat(pc): add pre-ABC observation and reward gates"
git push origin "$(git branch --show-current)"
```

如果当前分支没有 remote，只记录本地提交并明确说明未 push；不要伪造远端状态。任何
P0/P1/P2/P3 的结果都必须先写 `summary.json`、`summary.md`、manifest 和哈希，再决定
哪些小型摘要进入 Git。

#### 动机与要回答的问题

当前 PC 环境固定为 `dt_s=0.02 s`、`control_hz=50`。因此 50 帧历史只有 `1 s`，而 C3
时变海流的配置周期约为 `8 s`。同时，当前 DADE 已经使用 10 帧短窗加 50 帧长窗的
Transformer 历史编码器，再输出特征给 MLP actor；所以“再加一个 50 帧 Transformer”
不是单纯增加上下文，而是替换 actor 的信息通路。

本组只回答三个互相独立的问题：

1. **A：时间感受野** 在不降低 50 Hz 控制频率、不更换 actor 结构的前提下，把长历史从
   `1 s` 扩到 `5 s` 或 `10 s`，能否改善 C3 realistic？
2. **B：决策频率** 在上下文物理时长和输入采样率相同的条件下，把 actor 决策从 `50 Hz`
   降到 `10 Hz`（必要时再试 `5 Hz`），能否保持闭环性能并降低决策/推理成本？
3. **C：actor 信息通路** 在控制频率和历史采样相同的条件下，用“当前状态 + 命令 + 50
   帧历史 token”的直接 Transformer actor 替换 `DADE + MLP`，是否存在可验证收益？

#### 固定契约

- 主环境：`odarrl_control_pc_c3_random_realistic_v1_rewardscale001.json`；C2 mild 只做
  回归，不作为主判据，因为 BC 在 C2 已接近成功率上限。
- 物理积分、传感器生成、安全终止和动作速率限制保持 50 Hz；降低频率只降低 actor
  决策频率，不降低物理/安全环路。
- 每个 episode 保持相同物理时长（当前 2500 × 0.02 s = 50 s）。10 Hz 决策对应 500 个
  决策点，5 Hz 对应 250 个；禁止只改 `control_hz` 而保留 2500 个决策点造成时长变化。
- 低频动作使用零阶保持，并继续经过现有速率限制/饱和检查；reward 和 discount 必须按
  物理时间累加/换算，不能把同一段奖励重复累加。
- actor 只能使用可部署观测；真实海流、故障标签和仿真内部状态仍只能进入训练动力学、
  critic 或评估记录。
- 所有 arm 使用相同的训练数据、物理秒数、更新次数和评估 seed；不能给某个 arm 额外
  预算后把差异归因于模型结构。
- 不以 Torch 预筛代替最终 CPU PVS；A/B/C 的正式结论必须经过同口径闭环复核。

#### A 组：只改历史时间跨度

保持当前 DADE/MLP actor、50 Hz 动作频率和动作包线不变，只改变历史 buffer 的采样
stride。短窗和长窗使用同一 stride，保持双层时间尺度一致：

| 组 | 控制频率 | 历史采样 | 短窗跨度 | 长窗跨度 | 与 A0 的差异 |
|---|---:|---:|---:|---:|---|
| `A0_stride1` | 50 Hz | 50 Hz | 0.2 s | 1 s | 当前基线 |
| `A1_stride5` | 50 Hz | 10 Hz | 1 s | 5 s | 只扩长时间感受野 |
| `A2_stride10` | 50 Hz | 5 Hz | 2 s | 10 s | 更长，但时间分辨率更低 |

执行项：

- [x] 实现/确认 `short_stride=5/10`、`long_stride=5/10` 的配置路径，不修改物理 `dt_s`；
- [x] 从同一 BC/教师数据和同一初始化协议训练 A0/A1/A2；
- [x] 加入 `history-zero` 或 `history-reverse` 作为机制检查，证明收益确实来自历史；
- [x] 报告不同 stride 的动作敏感度、推理时延、成功率、严格成功率、timeout、RMSE、
  safety 和 material saturation；
- [x] 若 A1/A2 都不优于 A0，不再继续调更长窗口；B 只能作为效率实验启动。

#### B 组：只改 actor 决策频率

物理/安全环路保持 50 Hz，actor 每 5 或 10 个物理步决策一次；动作在决策间隔内零阶
保持。B 组与 A 组配对，确保历史物理跨度和 token 采样率一致：

| 组 | actor 决策 | 物理步 | 短窗 | 长窗 | 配对基线 |
|---|---:|---:|---:|---:|---|
| `B10` | 10 Hz | 50 Hz | 1 s | 5 s | `A1_stride5` |
| `B5` | 5 Hz | 50 Hz | 2 s | 10 s | `A2_stride10` |

执行项：

- [x] 先跑 `B10`；只有 B10 通过非劣门禁，才允许跑 `B5`；
- [x] 实现独立决策计数器，禁止改变 `dt_s` 或让安全终止延迟到下一个决策点；
- [x] 报告动作保持造成的舵面速率、饱和、roll/heading 振荡和 timeout 变化；
- [x] 报告 p50/p95 推理时延和单位物理时间计算量；
- [x] 若 B10 明显退化，停止降频，不通过增加训练步数补偿。

**B 组执行结果（2026-11-05，服务器完成）**

- B10 正式 screen：`pass_non_inferior`。相对 `A1_stride5`，default success/strict
  降幅为 `0.0`、3D RMSE ratio `1.0014`、timeout 增幅 `0.0`、material saturation 增幅
  `0.0009`；latency gate 通过，p95 `0.8907 ms`、compute ratio `0.1186`。B10 是效率
  候选，不是相对 A0 的性能提升。
- B5 正式 screen：`fail_non_inferior`。相对 `A2_stride10`，default/strict125 success
  均下降 `0.05`、3D RMSE ratio `1.1153`、timeout 增加 `0.05`；按预注册门禁停止
  5 Hz 分支。不得重跑、加训练步数或换 seed 救结果。

#### C 组：直接 Transformer actor

只在 A 或 B 证明“时间尺度/历史采样确实有影响，但 DADE+MLP 通路仍限制性能”后启动。
当前 A 无历史因果归因、B10 仅效率非劣、B5 fail，因此默认记录
`C_not_triggered`；除非已有 artifact 明确满足该条件，否则不训练 C。
Transformer 输入必须包含当前状态 token、命令 token 和历史 token，不能只喂历史而丢掉
当前指令；使用与最佳 A/B 组相同的控制频率和历史采样率。

- [x] C 触发判定完成：正式记录 `C_not_triggered`；以下条件执行项均未启动。

- [ ] 先对当前 BC/最佳基线动作做监督蒸馏，保证 Transformer 初始闭环不劣化；
- [ ] 再做一次有界微调；训练数据和物理秒数与最佳 A/B 组一致；
- [ ] 参数规模和 p95 推理时延与 DADE+MLP 保持同一量级，避免把“模型变大”误写成
  “架构更好”；
- [ ] 与相同输入、相同频率、相同训练预算的最佳 A/B 组配对比较；
- [ ] 若参数量/时延明显更大而没有稳定收益，C 判 fail，不继续加层或加 head。

#### 训练与评估阶梯

- **契约 smoke：** 1 个训练 seed × 8 个固定评估 seed；检查物理时长、动作保持、reward
  时间尺度、无特权泄漏、无非有限值。
- **首轮筛查：** 1 个训练 seed × 20 个固定评估 seed（建议 `1000–1019`），与同 seed 的
  `A0` 或配对基线比较。
- **正式比较：** 3 个训练 seed（建议 `94030–94032`）× 32 个评估 seed（建议 `0–31`），
  逐 seed 配对，报告 mean/std/worst 和失败尾部。
- **最终复核：** 只有正式比较通过的单一候选进入 CPU PVS 64-seed；不对多个候选做
  seed 挑选。
- 主要 tag：`default`、`strict125`、`grid_hs`；`h1/h3/h4` 只作为饱和和执行器压力
  回归，不能用来挑 win。

#### 预注册门禁

**A 组相对 A0：**

- success 至少 `+0.05`，或 strict125 至少 `+0.05`，或 3D RMSE 至少改善 `0.10 m`；
- safety 增幅不超过 `0.02`；material saturation 只报告，不作为硬门禁；
- timeout 增幅不超过 `0.05`；
- 历史消融必须显示可解释的因果贡献；否则只能算相关性。

**B 组相对匹配的 A 组：**

- success 和 strict125 的下降都不超过 `0.02`；
- 3D RMSE 不超过 `1.05×`，timeout 增幅不超过 `0.02`；saturation 只报告；
- p95 推理时延明显低于决策周期，且计算量/时延收益可复现；
- 只有同时满足非劣和效率收益，B 才能登记为正结果；性能提升不是必要条件。

**C 组相对最佳匹配 A/B：**

- 达到与 A 组相同的性能提升阈值（success/strict `+0.05` 或 RMSE `-0.10 m`）；
- 不能靠更多参数、更长训练或不同评估 seed 取得优势；
- 若无稳定收益，C 判 fail，并停止继续堆 Transformer 层数/head。

正式比较通过后，仍必须遵守 Goal9 原有的 CPU PVS、同训练 seed、固定评估 seed 和
安全/timeout 硬门禁；material saturation 仅作为风险信息报告。任何 Torch 预筛结果都
不能直接写成最终结论。

#### 执行顺序与停止规则

1. 先完成 14.4.18 顶部的 P0/P1/P2/P3 前置门禁；P0/P1/P2/P3 未完成或未记录负结果前，
   不进入 A/B/C 的 3×32 正式比较。
2. 只执行 A 组契约 smoke；任何契约错误先修，不扩大实验。
3. A 组 1×20 筛查；无论正负，保留结果并决定是否进入 3×32。
4. 仅在 A1/A2 没有显著退化时执行 B10；B10 通过非劣门禁后才执行 B5。
5. 仅在 A/B 证明时间尺度有影响但通路仍有瓶颈时执行 C；C 只做一次有界训练，
   不追加重训练 seed。
6. 任一组未过门禁，记录负结果并停止该分支；禁止用加步数、换评估 seed、调阈值或
   同时混入第二个变量来“救”结果。

#### 产物与 Git

- 服务器根目录：`artifacts/lowlevel/pc_abc_temporal_20261105/`；
- 每个 arm/seed 独立子目录，包含 `summary.json`、`summary.md`、manifest、配置/checkpoint
  hash 和有限性审计；
- 大 checkpoint、完整 rollout 和 event 文件不进 Git；只提交配置、脚本、小型 summary
  和报告；
- 文档或代码提交示例：

```bash
cd /home/bubble/水下机器人/settled_局部规划器_反应式
git add doc/goals/goal9_odarrl_control_reproduction.md \
        doc/goals/goal9_pc_server_validation_plan.md
git diff --cached --check
git commit -m "docs(pc): prioritize temporal-receptive-field and actor-rate experiments"
git push origin main
```

如果当前分支不是 `main`，先确认 remote 和分支，不能伪造 push。没有 remote 时只记录本地
提交；不要使用 `git add -A`，不要提交 artifacts、checkpoint 或凭据。

### 14.4.19 ALOS-free 速度学习 V0/V1/V1b/V2/V3（2026-11-05，预注册，依赖 P/A/B/C）

状态：**已完成 V0 并因失败停止。** V 组是 P0/P1/P2/P3 和 A/B/C 之后的新阶段，只在
schema 和最佳 A/B/C 配置冻结后启动。V 之前 ALOS 仍是唯一路径制导层；V 才移除 ALOS
的 heading/pitch 制导，让 actor 由包含路径信息的 `z_t` 和速度信息输出执行器。V0 使用
修正后的 `path_encoder_alpha=1.0`、actor LR `1e-4` 完成正式 screen：default、
strict125、grid_hs 的 success 均为 `0/20`，三组共 `60/60` timeout，判定
`fail_v0`。按预注册规则，V1/V1b/V2/V3 未启动，S16 候选为空，保留 ALOS 外置 +
S15 `A0_stride1`。material saturation 只报告、不参与 V0 判定。V 不重写任何历史结果，
也不把 V0 的失败解释成 ALOS 外置方案失败。

#### 全局顺序与关联

```text
P0/P1/P2/P3  路径观测、DADE-z、路径奖励的前置门禁
      ↓
A → B → C    时间感受野、决策频率、actor 信息通路
      ↓
V0           ALOS-free + 固定速度参考，先验证纯路径控制
      ↓
V1           删掉显式速度参考，actor 通过 RPM 自己学速度
      ↓
V1b          仅在 V1 没学会转弯减速时，加轻微曲率软约束（可选）
      ↓
V2           仅在 V1 速度行为特别糟糕时，考虑 speed head（可选，不默认）
      ↓
V3           正式 3×32 + CPU PVS，与 ALOS teacher 和固定速度基线比较
```

- **P 组决定 V 的输入。** P1 决定路径特征是否作为 `o_t` 的一部分编码进 `z_t`；P2
  决定 DADE 是否使用 \(z_t\) 历史；P3 决定是否保留路径奖励。P1/P2/P3 失败时，V 使用
  对应的 P0 基线配置，不把失败分支硬带进 V。
- **A/B/C 决定 V 的时间尺度、决策频率和 actor 结构。** V0/V1 使用 A/B/C 选出的最佳
  历史 stride、actor 频率和策略结构；若 C 失败，则使用最佳 A/B 的 DADE+MLP 配置。
- **V 不能提前到 A/B/C 之前。** V 改变了 actor 的动作目标和奖励语义；提前启动会让
  A/B/C 的“只改一个变量”失效，也会让 V0/V1 失去可比基线。
- **代码可以提前准备，训练不能提前。** P 组 schema 冻结后可以并行写 V 的配置、奖励
  和评测工具，但 V0/V1 的训练必须等待 A/B/C 胜出配置确定。
- **V 的失败退出规则：** V0 失败则停止 V，不进入 V1；V1 失败则保留 ALOS 外置方案并
  记录负结果；V2 只有在 V1 速度行为特别糟糕且经复核后才允许启动。

#### 共同契约

- 主环境仍为 `odarrl_control_pc_c3_random_realistic_v1_rewardscale001.json`；V0/V1
  先做 1 个训练 seed × 20 个评估 seed（`1000–1019`）筛查，最终唯一候选才进入 3×32
  和 CPU PVS 64-seed。
- V0/V1 使用独立输出目录 `artifacts/lowlevel/pc_alos_free_v_20261105/`，不覆盖
  P0/P1/P2/P3、A/B/C 或历史结果。
- **BC 数据来源：** V0/V1 初期允许二选一或混采 `ALOS + PD/PI` teacher 的闭环轨迹与
  中低速 teacher/策略轨迹；无论哪种来源，都必须覆盖非零初始速度、非零初始 RPM、曲率段
  和恢复状态。若教师轨迹只有零初速/零 RPM 起步，必须用中低速或非零初速课程补采，不能
  把教师数据的初始分布直接当成学生部署分布。BC 标签是执行器动作
  `[rudder, stern, rpm]`，不是 ALOS 的中间量。
- **初始状态：** 不使用“速度=0、RPM=0”的默认假设。复用并扩展 `initial_speed_ratio`
  和 `initial_propeller_rpm_ratio`，允许非零初速和非零起步 RPM；课程至少覆盖
  路径上对准起步、带横向/垂向偏差起步、带航向/俯仰偏差起步和低速/中速起步。训练和
  评估矩阵必须实际包含非零初速与非零起步 RPM 的 episode，不能因为旧配置默认值就省略；
  初值范围先按已有 R 组验证过的 `initial_speed_ratio=0.75/1.0/1.10/1.25` 和
  `initial_propeller_rpm_ratio=0/0.25/0.75/1.0` 作为候选网格，再由 PC 本体标定裁剪。
- **奖励优先级：** 安全/timeout/速度包线是硬约束；路径精度和任务时间是主目标；速度
  平滑是高优先级次级目标；能耗尽量优化；动作平滑略低于速度平滑；material saturation
  仅报告，不作为约束。
- **终端规则：** 不要求终端减速，不加入过冲惩罚。终端仍用捕获半径、success、
  timeout 和安全终止判定；过冲只作为诊断指标记录，不进入奖励。
- **路径与 DADE：** V0/V1 的 `o_t` 包含路径特征，当前 actor 使用编码后的 `z_t`；
  若启用 DADE，历史 token 也只能使用同一套包含路径的 `z_t`，不得另开 raw `path_t`
  旁路；路径特征不直接拼接进 actor 顶层。
- **模块化观测复用：** V0/V1 必须复用 P1 的 block 划分和 `active_mask`；`path_geom`、
  `path_context` 和可选的 `path_velocity` 的启用状态要写入 checkpoint metadata。
  V0 与 V1 的最终观测 schema 必须一致，V1 只删除 `u_d` 这个命令输入，不能同时删改
  路径 block；若要测试“少看一部分路径”，必须另立 arm，并按配对预算评估。
- **速度控制方式：** V0 保留固定速度参考；V1 删除显式 `u_d`，actor 仍直接输出 RPM，
  速度通过奖励和路径/曲率/剩余距离观测隐式学习。**默认不引入 speed head。**

**V0：ALOS-free + 固定速度参考**

目的：先隔离“去掉 ALOS 的 heading/pitch 制导”这一件事，不把速度学习混进来。

- actor 输入：`z_t = E(o_t)`，其中 `o_t` 包含机器人状态和路径特征；固定 `u_ref`
  作为显式输入；actor 输出：`[rudder, stern, rpm]`；
- 不接收 ALOS 生成的 `psi_d/theta_d`，路径偏差、切向、曲率和剩余距离由路径观测提供；
- 奖励：路径误差 + 进度/时间为主，固定 `u_ref` 的速度跟踪、速度平滑、能耗、动作平滑和
  安全项为辅；不加过冲惩罚；
- BC 可以用 ALOS teacher 或中低速轨迹，但学生观测必须与 V0 部署接口一致；
- 门禁：相对最佳 ALOS-external P/A/B/C 配置，success 或 strict125 至少 `+0.05`，或
  3D RMSE 至少改善 `0.10 m`；time-to-goal 不超过 `1.05×`；safety 增幅不超过 `0.02`，
  timeout 增幅不超过 `0.05`；material saturation 只报告。

**V1：删除显式速度参考**

目的：在 V0 证明纯路径控制成立后，再让 actor 自己决定速度。

- actor 输入：`z_t = E(o_t)`，其中 `o_t` 包含机器人状态和路径特征；不含 `u_d`；
  actor 输出：`[rudder, stern, rpm]`；
- 奖励：路径误差 + 进度/时间为主；速度包线作为硬约束；速度平滑优先，能耗次要，动作
  平滑略后置；不加过冲惩罚；
- 不硬编码“曲率→目标速度”，只把曲率和前视路径放入 `o_t` 并编码进 `z_t`，让策略
  自己学转弯减速；
- BC 可以使用 ALOS 轨迹或中低速轨迹；没有 `u_d` 时，教师速度行为通过 RPM 标签隐式
  蒸馏，后续再用 PPO 微调；
- 速度包线必须固定 `u_min/u_max`、RPM 上限；material saturation 继续测量并报告，
  但不能因为“自己学速度”而放弃最低舵效速度或安全上限。

**V1b：可选曲率软约束**

只有 V1 在转弯段明显不减速，且不是路径精度/时间目标导致的合理高速时，才允许加一个
很小的曲率软约束：

```text
r_turn = -w_turn * relu(u - u_turn_max(kappa))
```

`w_turn` 必须小，不能压过路径精度和时间目标；V1b 必须与 V1 同 seed 配对，若只降低
速度但 success/strict/RMSE/time 退化，则判 fail。

**V2：可选 speed head（默认不做）**

只有 V1 出现严重速度振荡、速度不可解释、速度包线频繁越界或路径/速度控制互相干扰，
且复核确认不是奖励或观测问题时，才允许测试 speed head：

```text
actor -> [rudder, stern, u_cmd] -> speed controller -> rpm
```

speed head 会改变动作空间、控制器、BC 标签和 checkpoint，属于中等架构改动；在 V1
没有“特别糟糕”之前，不把它作为默认路线，也不和 V1 同时改。

**V3：正式比较**

最终至少比较：

- `ALOS + PD/PI` teacher；
- V0：ALOS-free + 固定速度参考；
- V1：ALOS-free + RPM 隐式速度；
- V1b：仅当 V1 未学会转弯减速时加入；
- V2：仅当 V1 速度行为特别糟糕且复核通过时加入。

正式门禁沿用 3 个训练 seed × 32 个评估 seed 和 CPU PVS 64-seed。主指标为 success、
strict125、路径横向/垂向 RMSE、time-to-goal、timeout 和 safety；material saturation
作为信息性风险指标记录；次级指标记录速度标准差、`Δu` p95、RPM²/能耗、动作变化率、
转弯段速度-曲率关系和过冲诊断。速度平滑/能耗的改善不能以 primary 指标退化为代价。

#### 产物与 Git

```text
artifacts/lowlevel/pc_alos_free_v_20261105/
  V0_fixed_speed/
  V1_learned_speed/
  V1b_turn_soft/
  V2_speed_head/
  V3_formal/
  smoke/
  screen/
  report/
```

- [x] P0/P1/P2/P3 和 A/B/C 的胜出配置已冻结；
- [x] V0 的固定速度参考、路径奖励、初始状态分布和 BC 数据契约已实现；
- [x] V0 未通过，已按预注册规则停止 V 分支；V1/V1b 未启动；
- [x] V2 未启动：V1 前置分支不成立，没有“V1 特别糟糕”的证据；
- [x] V3 未启动：V0=`fail_v0`，没有通过的单一候选，因此未跑 3×32 或 CPU PVS；
- [x] 每阶段写 `summary.json`、`summary.md`、manifest、配置/checkpoint hash 和 finite
  audit；大 checkpoint/rollout/event 不进 Git。

每阶段完成后按文件级命令提交，不使用 `git add -A`；示例：

```bash
cd /home/bubble/水下机器人/settled_局部规划器_反应式
git status --short
git add configs/lowlevel/<v-arm-config>.json \
        src/auv_lowlevel/<changed-source>.py \
        tests/<changed-test>.py \
        doc/goals/goal9_odarrl_control_reproduction.md \
        doc/goals/goal9_pc_server_validation_plan.md
git diff --cached --check
git commit -m "feat(pc): add ALOS-free speed-learning V arm"
git push origin "$(git branch --show-current)"
```

没有 remote 时只记录本地提交；不要伪造 push，不要提交 artifacts、checkpoint 或凭据。

### 14.5 资源规则

- 主链 I 臂优先占用资源；O 臂只有在 GPU/CPU 不超卖、不影响 I 臂时才并行；
- R 臂等待 G2/G3A 的 MPC 专家，不与 I 臂争抢关键路径；
- 每次正式运行使用独立 GPU（如有）、CPU 核范围和输出目录；
- 单次运行优先增加 PVS worker，而不是强上单任务多卡；
- 未来 IsaacLab 原生 GPU 批量 PPO 在显存和 CPU 允许时以 `num_envs=4096` 为首选；当前一环境一进程的 PVS 后端不适用该数字，资源不足时保持 64–128 workers；
- 任何新依赖只装到隔离 venv，安装前后比较 `pip freeze` SHA-256；
- 默认不修改 PVS/IsaacLab/rsl_rl 源码。
- **2026-11-02 起 PC 臂拥有最高资源优先级**：PC 的契约、PD/PI BC 热启动、传感器真实化和 C0/C1/C2 分层 smoke 先占资源；旧 I/O/R 只做兼容、对照和已有证据，不因旧主链状态阻塞 PC。PC 达到正式评估条件后，再决定是否回补旧臂对照。
- **2026-11-05 起 14.4.18/14.4.19 的 P、A/B/C、V 组是 PC 臂内的最高优先级**：先完成
  P0（A 短筛查）、P1（路径观测）、P2（DADE-z）和 P3（奖励）；schema 冻结后按 A→B→C
  串行执行；A/B/C 胜出配置冻结后再按 V0→V1→V1b→V2→V3 执行。未完成当前组的预注册
  门禁前，不启动新的 DADE/DAgger 调参、S14 重复轮次或大规模扩训。实验资源按
  P0→P1→P2→P3→A→B→C→V 串行占用；V2 默认不启动。
