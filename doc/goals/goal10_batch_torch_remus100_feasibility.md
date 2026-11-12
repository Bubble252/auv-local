# 任务：批量 Torch REMUS100 + 4096 环境训练可行性评估

记录日期：2026-10-30

更新日期：2026-10-31（锁定 26D 为长期统一 schema；补记 P2.5 吞吐、reward-scale 修复、canonical 26D 配置、26D PVS 50-seed 门禁、P4 compile 验证、长训练、等预算规模对照和最终报告）

状态：**P0–P4 全部完成。** P0 批量动力学、P1 完整批量环境、P2 的 `BatchTorchVecEnv`/配置驱动 `num_obs`/26D-27D 接口/rsl_rl 2.3.3 适配、P2.5 的 `4096×256` 26D GPU pilot 均通过。P3 的直接 26D 核心门禁已通过：canonical v2 配置和 checkpoint schema/dim 校验已落地，CPU PVS 50-seed 成功率 `50/50`，三维 RMSE `1.51546 m`（B0 `1.51475 m`，差 `0.00071 m`），material saturation 为 `0`。P4 完成了 profile、compile 数值一致性、4096×256 长训练、候选择优、4096/1024/256 等预算规模对照和最终报告。长训练按 20-seed PVS 选出的 `model_25.pt` 在 50-seed PVS 上成功率 `50/50`、三维 RMSE 差 `+0.00804 m`、material saturation `0`，通过控制门禁，但相对 B0 没有精度收益。等预算规模对照中，4096/1024/256 的三维 RMSE 差分别为 `+0.01804/+0.02265/+0.03229 m`，4096 吞吐最高、质量最接近 B0，但三者都没有正收益。低层观测长期统一为 26D；27D 只保留给旧 checkpoint、历史数据、迁移 teacher 和前后端回归，不再作为新实验默认。复用旧策略时使用 BC/DAgger/动作蒸馏得到 26D student；直接训练 26D 是默认主路线。没有修改 PVS、IsaacLab 或 rsl_rl 源码。**

关联文档：`doc/goals/goal9_odarrl_control_reproduction.md`

## 1. 结论

**技术上可行，但必须作为独立子项目完成，不能简单把现有 `PvsVecEnv` 的 `num_envs` 改成 4096。**

目标不是训练一个替代 PVS 的神经网络近似模型，而是把 **同一套 PVS REMUS100 六自由度 Fossen 方程、前向 Euler 积分和三个执行器的一阶动态** 改写成 batched Torch 实现，再把它与 ALOS、路径投影、观测、奖励、终端条件和 reset 一起全向量化。

建议最终形态：

- 训练：单进程、单张 RTX 4090、`num_envs=4096` 的 GPU 批量环境；
- PPO：继续使用 rsl_rl，不修改 rsl_rl 源码，只提供一个兼容的 `VecEnv` 适配器；
- 评估：批量 Torch 环境用于训练和快速筛查，正式 50-seed 结果仍用 CPU PVS 复核；
- 初始配置：`4096 × 256` 而不是 `4096 × 1024`，先控制 rollout 显存和 PPO 更新成本。
- 观测 schema：P1/P2 可以保留 v1 27D 作为兼容回归；多环境基础设施完成后，P2.5 的主 pilot 和 P3/P4 的新实验都直接使用 v2 26D。27D 不再承担新训练或正式结果，只用于旧 checkpoint、历史数据、迁移 teacher 和前后端对照。
- 切换顺序：先把多环境运行基础设施（`BatchTorchVecEnv`、配置驱动的 `num_obs`、GPU rollout、auto-reset 和 rsl_rl 适配）做稳；随后把新实验默认切到 26D。26D 可以直接从在线 PD/PI 基础策略或从零开始训练；若要复用旧 27D 策略的能力，则用 BC/DAgger/动作蒸馏得到 26D student。两条路线都必须经过同 seed 的 50-seed PVS 闭环门禁，不能把“删掉一列”直接当成等价迁移。
- canonical 配置：环境侧使用 `configs/lowlevel/odarrl_control_v2_26d.json`（canonical hash `c19a5867…0f72e7`，仅把 base 的 `observation.dim` 从 27 改为 26）；批量训练侧使用新增的 `configs/lowlevel/odarrl_control_batch4096_v2_26d.json`，它显式指向上述 26D 环境配置。历史 pilot 配置和 checkpoint 保留原哈希，不回写。
- 长期口径已经执行：新实验只使用 26D；27D 的 `pi0`、r1–r4 和旧数据继续作为 v1 历史结果保存，但不启动新的 27D 长训练。若未来确实需要旧 27D 策略的能力，再单独做 BC/DAgger/动作蒸馏迁移，并重新通过 50-seed PVS 门禁。

如果物理一致性门禁无法通过，批量环境只能用于预训练或课程学习，不能替代 PVS 作为正式结论来源。

## 2. 为什么可行

PVS REMUS100 是明确写出的六自由度 Fossen 模型，不是 CFD。其一步计算由以下部分组成：

- 刚体质量、附加质量、Coriolis/centripetal 项和阻尼项；
- 螺旋桨推力、扭矩和前进比插值；
- 舵角、艉水平舵和螺旋桨的一阶执行器动态；
- 升阻力、20 段横流阻力、恢复力；
- 水平海流相对速度；
- `nu`、执行器和 Euler 姿态角的前向 Euler 积分。

这些运算都是矩阵乘、逐元素非线性、`atan2`、`sqrt`、`exp` 和有限维求和，可以对环境维 `N` 直接广播。PVS 的单个环境对象、Python 循环、Pipe IPC 和同步 barrier 才是当前吞吐瓶颈。

4096 个环境、50 Hz 聚合需要：

```text
4096 × 50 = 204,800 env-steps/s
```

这个计算量对 RTX 4090 的 FP32 算力很小。真正的难点是：

- 不能逐环境调用 Python；
- 不能每一步做 GPU→CPU 同步；
- 路径投影、reset、终止和 PPO rollout 必须批处理；
- 需要控制 Python/CUDA kernel launch 开销；
- 需要证明批量实现与 PVS 的长期闭环结果一致。

## 3. 必须向量化的范围

| 模块 | 当前实现 | 批量实现 | 风险 |
|---|---|---|---|
| REMUS100 动力学 | 单环境 Python/NumPy | `N×6` 状态、固定 6×6 常数矩阵和 batched force | 方程抄写错误、float32 漂移 |
| 执行器 | 单环境一阶动态 | `N×3` 批量更新和限幅 | 限幅时序必须与 PVS 相同 |
| 海流 | 单环境 `UniformCurrent` | 每环境二维常值向量 | episode 内常值、方向采样 |
| 路径投影 | 逐段 Python 循环 | `N×M` 段点积/argmin，或局部窗口搜索 | tie-break 和曲率索引 |
| ALOS | 单环境 NumPy | 全批量 atan2/三角/向量运算 | 终端捕获分支 |
| 观测 | 单环境 dataclass | `N×D` GPU tensor，`D` 由 schema 决定 | 归一化、schema 版本和历史残差索引 |
| BC `pi0` | 逐环境 CPU MLP | `N×D` 一次 MLP 前向，D 由 schema 决定 | 训练/评估数值一致性、checkpoint 维度匹配 |
| 奖励和终止 | 单环境 scalar | `N` 维 tensor，mask 处理 reset | 终止与 bootstrap 时序 |
| reset/随机化 | 逐环境 Python RNG | 固定形状 GPU buffer + mask | seed 可复现性 |
| PPO 适配 | 64 个 PVS worker | 单 `VecEnv` + GPU tensor | rsl_rl 的 dtype/device 契约 |

建议新增独立包，例如 `src/auv_lowlevel/batch/`，不要改写现有 `env.py`、`path.py` 和 `guidance.py`。旧实现继续作为 reference backend。

### 3.1 26D 长期 schema 与能力迁移（已决定）

观测维度不能写死为 27。当前 v1 是 27 维，是因为 I 臂教师是带速度积分的 PI 控制器，索引 26 保存 `speed_integral_rpm`；这不是所有控制器的通用要求。用户已决定后续统一为 26D。批量后端从配置读取 `observation_schema` 和实际维度 `D`，并在批量张量中保持 schema 版本、归一化参数和 controller context 的对应关系。

统一范围仅限 ODARRL/PVS 的低层执行器级控制观测。`goal8_ppo_isaaclab.md` 的 P1 二维避障使用独立的 17 维感知契约，不属于本 schema，不应因本决定改成 26D。26D 指策略的 canonical 单帧观测；若后续启用观测历史栈或循环网络，原始单帧仍是 26D，实际网络输入形状必须单独记录。

- v1（兼容）：保留 27 维和索引 26，用于现有 I 臂 checkpoint、已采数据和前后端回归，不再作为新实验的长期默认；
- v2（目标）：26 维 core，索引 `0–25` 与 v1 保持一致，删除索引 26；后续新控制器、新策略和新实验默认使用 v2；
- 26D 不是简单删一列就结束。旧 PI `pi0` 去掉积分输入后，动作不再是严格的当前观测函数；如果复用旧策略，必须用 BC/DAgger/蒸馏重新学习并在闭环中验证；
- 直接训练 26D 是合法主路线：可以让在线 PD/PI 继续在控制器内部维护积分，只把 26D 状态给学习策略；也可以从零训练 26D 策略。此时不要求先构造 27D teacher；
- 复用旧 27D 策略是可选迁移路线：从现有 27D `pi0`、残差策略或 current-aware 教师生成闭环数据，只保留 26D core 输入训练 student；naive drop-column BC 只作为基线；
- 对未来带其他内部状态的控制器：仍使用版本化 schema 或独立的 controller-context 编码，不把不同语义硬塞进第 27 维；若增加观测历史或循环网络，必须显式记录网络输入形状，单帧 canonical schema 仍是 26D。

**统一规则（本轮冻结）：**

- `26D = v1 的索引 0–25`，是所有新控制器、新 checkpoint、新 PPO/残差实验的正式输入；不得再为某个控制器临时增加第 27 维。
- `27D` 只是 legacy compatibility schema：只允许用于旧 `pi0`/r1–r4、历史数据、前后端回归和迁移 teacher，不允许新增长训练或新实验组。
- 多环境基础设施本身必须与维度解耦：`observation_schema`、`observation.dim`、`env.num_obs` 和网络输入维度由同一份配置驱动，禁止在 `BatchTorchVecEnv` 或 PPO 适配层写死 `27`。P2 的 27D smoke 只用于兼容验证；P2.5 主 pilot 直接使用 26D。
- 切到 26D 是正式目标，但不是强制要求“先有一个 27D teacher”。P3 允许两条路线：直接训练 26D，或按需把旧 27D 策略用 BC/DAgger/动作蒸馏迁移到 26D。只有选择迁移时，才需要固定 teacher 和闭环 DAgger 数据。
- 26D 部署路径不能偷偷读取在线 PI 积分、海流真值或其他 privileged state；需要时序信息时，只能使用显式的 26D 观测历史或循环网络。
- 26D 通过门禁后，配置文件、vec-env 的 `num_obs`、checkpoint 元数据和报告都写 `schema_version=v2`；旧 27D 结果单独标注为兼容回归，不混入正式 26D 结论。

阶段边界必须明确：

| 阶段 | 默认 schema | 用途 | 是否可作为正式新实验结果 |
|---|---|---|---|
| P1 | v1 27D（仅兼容） | 对照标量环境，验证批量动力学、制导、观测、奖励和 reset 的逐元素一致性 | 否，仅后端回归 |
| P2 | v1 27D + v2 26D（接口 smoke） | 验证维度可配置的 rsl_rl 接口、26D/27D shape、reset 和 PPO 更新；27D 只做兼容 | 否，仅接口 smoke |
| P2.5 | v2 26D（主 pilot） | 用 26D 跑 `4096×256` 短程 pilot，验证吞吐、显存、损失和 reset；27D 只保留独立兼容 smoke | 否，仅短程 pilot |
| P3 | v2 26D | 将 26D 设为新实验默认；直接训练 26D，或按需做 27D → 26D 的 BC/DAgger/蒸馏和 50-seed 闭环门禁 | 是，门禁通过后进入正式训练 |
| P4 | v2 26D | 26D 正式训练、PVS 50-seed 评估和论文结果 | 是 |

26D 策略落地与可选迁移路线：

1. **直接 26D 路线（默认可用）**：在线 PD/PI 基础控制器可以在内部维护积分，但学习策略只接收 26D 观测；冻结基座后学习残差 PPO，或直接从零训练 26D。该路线不需要构造或保留 27D teacher。
2. **复用旧策略路线（可选）**：固定一个 27D teacher，可以是现有 `pi0`、r4 类残差策略或 current-aware 教师；teacher 的配置、归一化统计、启动 checkpoint 和评估 seed 全部冻结并记录哈希。
3. 选择迁移时，在相同路径、海流和 reset seed 下跑闭环，teacher 使用完整 27D，student 只使用索引 `0–25`；在线收集 DAgger 数据，标签是 teacher 的执行器动作，样本必须包含 student 自己访问到的状态，不能只使用 teacher 的离线轨迹。
4. 迁移先跑 naive drop-column BC 作为下界，再用 BC/DAgger 或动作蒸馏。若需要，可把 27D 网络的输入权重除最后一列外复制给 26D 网络、将删除列置零，作为热启动；这只是一次初始化，不能代替闭环蒸馏和验证。
5. 26D 部署路径不能偷偷读取在线 PI 积分、海流真值或其他 privileged state；需要时序信息时，只能使用显式的 26D 观测历史或循环网络，并在 checkpoint 元数据中记录历史窗口/循环结构。
6. 不论直接训练还是迁移，先冻结 26D student 做 50-seed PVS 评估，再决定把它作为 PPO 初始化或继续训练。若一次离线 BC 在闭环失败，不能回到 27D；应增加 DAgger 轮数、闭环采样和蒸馏损失。

## 4. 吞吐可行性

### 4.1 计算量

单个环境的 Fossen 一步估计为几千到一万次浮点运算。按 204,800 env-steps/s 计算，批量动力学本身所需算力只有数个 GFLOP/s，远低于 RTX 4090 的 FP32 上限。

因此不能把“4096 环境”理解成 GPU 算力不够；4096 设计的核心是 **避免 4096 个 Python 进程**，把状态合并成一个 batch tensor。

### 4.2 一次批量 step 的时间预算

```text
4096 env 的一步 wall time = 4096 / 目标 env-steps/s
目标 204,800 env-steps/s 时，一步允许约 20 ms
目标 500,000 env-steps/s 时，一步允许约 8.2 ms
目标 1,000,000 env-steps/s 时，一步允许约 4.1 ms
```

动力学和 ALOS 的 GPU 计算很可能低于 1 ms；主要时间会落在 Python 调度、kernel launch、路径投影、reset mask 和 PPO 数据管理。第一版接受 200k–500k env-steps/s，不把 1M 作为硬门禁。

### 4.3 可能需要 `torch.compile`

第一版用 eager PyTorch 做正确性和 benchmark。只有在 profiling 证明 Python/kernel launch 是瓶颈时，才对固定形状的 `step_core` 使用：

- `torch.compile(mode="reduce-overhead")`；
- 或 CUDA Graph；
- 或去掉逐环境 Python 分支，改为 `torch.where`/mask。

不建议一开始就上 CUDA Graph，因为 reset、终止和调试会明显变难。

## 5. 显存可行性

以 `num_envs=4096`、`num_steps_per_env=256` 估算，每个 iteration 有 1,048,576 个 transition。rsl_rl 需要保存 observation、action、value、reward、log-prob、advantage 等：

- observation：v1 的 `1.05M × 27 × 4 B ≈ 113 MB`；v2 的 26D 约为 `109 MB`，以下按 27D 上限估算；
- 其余 rollout 字段保守估计再增加 `0.3–0.8 GB`；
- 一个 iteration 的 PPO rollout 总量预计 `0.5–1.2 GB`。

如果 `num_steps_per_env=1024`，transition 数变成 4,194,304，rollout 约 `2–5 GB`，仍然可能放进 24 GB，但 PPO minibatch、激活和临时 buffer 会显著增加风险。

建议：

- 第一轮正式实验：`4096 × 256`；
- 吞吐和显存稳定后再测 `4096 × 512`；
- 不把 `4096 × 1024` 作为第一目标；
- `num_mini_batches` 从 64 开始，根据显存和更新速度再调。

## 6. 物理保真与 PVS 一致性

### 6.1 必须完全相同的内容

- `dt=0.02 s` 和 forward Euler；
- 状态定义：位置、Euler 角、体轴速度、体轴角速度；
- 刚体/附加质量、Coriolis、阻尼、恢复力；
- 螺旋桨前进比和正/反推力分支；
- 舵与艉水平舵的 `T_delta=0.1 s`、螺旋桨 `T_n=1.0 s`；
- 执行器限幅顺序；
- 20 段横流阻力的离散和；
- 水平海流的相对速度定义；
- 观测 schema：v1 的维度 `D=27`、索引顺序、归一化、索引 `23:26` 的残差历史和索引 26 的 PI 速度积分；v2 的 `D=26` 保留索引 `0–25` 并删除索引 26；
- 路径投影 tie-break、ALOS lookahead 和终端捕获；
- 奖励、终止、timeout 和 reset 时序。

### 6.2 保真门禁

在批量环境被允许用于正式训练前，至少完成以下测试：

| 测试 | 数据 | 建议门禁 |
|---|---|---|
| 单步导数一致性 | 随机 10k 状态/动作 | float64 最大相对误差 `<1e-6`；float32 `<1e-4` |
| 观测逐元素一致性 | 随机 10k 状态/参考/海流/执行器/残差/积分，批量与标量 builder 对照 | shape 为 `(N,D)`；float64 最大绝对误差 `<1e-6`，float32 `<1e-5`；无非有限值 |
| v1 schema 契约 | v1 配置和构造后的观测 | `D=27`；索引 `17:20` 为海流策略、`20:23` 为上一执行器、`23:26` 为上一残差、索引 26 为 `speed_integral_rpm / max_speed_integral_rpm`；D 或 schema 不匹配时立即报错 |
| v2 schema 契约 | v2 配置和构造后的观测 | `D=26`；索引 `0–25` 与 v1 完全相同；不存在索引 26；任何仍读取积分的分支必须立即报错 |
| reset 历史一致性 | 批量 reset 前后、首个 step | 索引 `23:26` 残差历史清零；v1 的索引 26 积分清零；I/R 臂未暴露海流时索引 `17:20` 为零；不得残留上一 episode 的值 |
| 100 步开环一致性 | 固定舵/RPM 序列 | 位置和速度最大绝对误差 `<1e-3` 量级 |
| 500 步开环一致性 | 多组动作和当前 | 误差不爆炸，报告增长曲线 |
| 2500 步闭环一致性 | 与 PVS 同一控制器/seed | 成功率和 RMSE 分布无系统偏差 |
| 50-seed B0 对比 | 同一 B0 控制器 | 成功率差 `≤0.02`，RMSE 差 `≤0.02 m` 目标 |
| I0/BC 对比 | 同一 pi0 checkpoint | 轨迹与终止原因可解释 |

长期闭环不会 bitwise 相同，因为非线性系统对浮点误差敏感；门禁应同时看单步误差和分布统计，不能只看一条轨迹是否完全重合。

### 6.3 科学结论策略

批量环境训练出的最终 policy 必须回到 CPU PVS 做 50-seed 评估。正式论文结果不混用两个后端的数字；如果要声称二者等价，必须给出跨后端对照表和误差范围。

## 7. 推荐实施阶段

### P0：动力学 spike（已完成）

- [x] 新增独立包 `src/auv_lowlevel/batch/`；在当前 worktree 内实现，不新建分支、不改 PVS/现有后端；
- [x] 实现 batched REMUS100 dynamics + actuator + Euler attitude；
- [x] 用随机 10k 状态与 PVS 单步对照，并完成 500 步开环对照；
- [x] 测量 `N=1/256/4096` 的 eager 和 compiled 吞吐；
- [x] 决定采用 float64 物理 + 网络边界 float32，优先保证物理保真。

阶段门禁：单步误差达标，`N=4096` 的动力学-only 吞吐至少达到 `500k env-steps/s`。

**P0 结果（2026-10-31，本机 CPU，单线程）：**

| 项目 | 实测 | 门禁 | 结论 |
|---|---:|---:|---|
| float64 单步 `nu_dot` 最大相对误差 | `4.47e-12` | `<1e-6` | 通过 |
| float64 单步 `nu_dot` 最大绝对误差 | `8.88e-15` | - | 通过 |
| float64 500 步开环位置最大绝对误差 | `4.44e-16` | 不爆炸 | 通过 |
| float32 单步 `nu_dot` 最大绝对误差 | `8.24e-6` | - | 可用于网络侧，不作为物理积分精度 |
| `N=4096` float64 eager | `1.30M env-steps/s` | `≥500k` | 通过 |
| `N=4096` float64 compiled | `1.74M env-steps/s` | `≥500k` | 通过 |
| `N=4096` float32 eager | `1.83M env-steps/s` | `≥500k` | 通过 |
| `N=4096` float32 compiled | `2.49M env-steps/s` | `≥500k` | 通过 |

float32 的严格相对误差会被接近零的加速度分量放大；虽然绝对误差很小，但本阶段不据此放宽物理门禁。最终选择 **float64 物理积分 + float32 网络输入/输出边界**，后续在 GPU 上复核吞吐。

证据文件：

- `artifacts/lowlevel/batch4096/p0_pvs_validation_float64.json`
- `artifacts/lowlevel/batch4096/p0_pvs_validation_float32.json`
- `artifacts/lowlevel/batch4096/p0_benchmark_cpu_float64.json`
- `artifacts/lowlevel/batch4096/p0_benchmark_cpu_float32.json`
- `artifacts/lowlevel/batch4096/p0_benchmark_cpu_float32_compiled.json`
- `artifacts/lowlevel/batch4096/p0_benchmark_cpu_float64_compiled.json`

提交方式（在本地 Git 根目录人工审阅后执行）：

```bash
git add src/auv_lowlevel/batch/dynamics.py tests/test_batch_remus100.py
git commit -m "feat: add batched Torch REMUS100 dynamics"
git push origin goal10-batch-torch-remus100
```

### P1：完整批量环境

- [x] batched path projection、ALOS、terminal capture；
- [x] batched observation、reward、termination、reset，并完成 v1 27D schema 和逐元素一致性测试；
- [x] batched observation builder 按 `ObservationSchema`/`D` 参数化，并验证 v2 前 26 列与 v1 完全一致；配置、vec-env 和网络端的端到端 `D` 切换归入 P2/P3；
- [x] batched frozen BC 推理；
- [x] 固定 episode seed 和 mask reset；
- [x] 与 PVS 500/2500 步对照。

P1/P2 使用 27D 的目的是兼容现有 I 臂和验证批量基础，不是重新确认 27D 为长期方案；P2.5 主 pilot 已切换到 26D。P1/P2 的实现若只能支持 27D，视为接口未完成。

阶段门禁：`N=4096` 完整环境 step 达到 `200k env-steps/s`，50-seed B0 统计接近 PVS。

**P1 结果（2026-10-31，本机 CPU）：**

| 项目 | 实测 | 门禁 | 结论 |
|---|---:|---:|---|
| `tests/test_batch_*.py` | `25 passed` | 全部通过 | 通过 |
| `N=4096` 完整环境，单线程稳态 | `333,971 env-steps/s` | `≥200k` | 通过 |
| `N=4096` 完整环境，8 线程稳态 | `462,308 env-steps/s` | 参考 | 通过 |
| `N=4096` 单线程 step p50 / p95 | `11.84 / 14.24 ms` | 记录项 | 通过 |
| 50-seed B0 成功率 | `1.00`，50 个 goal | 与 PVS 差 `≤0.02` | 通过 |
| 50-seed B0 三维 RMSE 差 | `-2.22e-16 m` | `≤0.02 m` | 通过 |
| 50-seed B0 水平 / 垂直 RMSE 差 | `0 / 1.67e-16 m` | 记录项 | 通过 |
| 50-seed B0 saturation / constraint 差 | `0 / 0` | 不上升 | 通过 |

实现要点：

- 路径投影使用“先相减再点积”，避免代数展开在共享顶点处的 ULP 差异；
- 闭环步骤用上一段索引做局部窗口投影（默认半径 8 段），reset 使用全路径投影；局部窗口有与全局投影的回归测试；
- 完整环境基准脚本：`scripts/benchmark_batch_lowlevel.py`；
- 50-seed 批量评估脚本：`scripts/evaluate_batch_lowlevel.py`。

证据文件：

- `artifacts/lowlevel/batch4096/p1_benchmark_cpu_float64.json`
- `artifacts/lowlevel/batch4096/p1_benchmark_cpu_float64_8threads.json`
- `artifacts/lowlevel/batch4096/p1_b0_50seed/summary.json`
- `artifacts/lowlevel/batch4096/p1_b0_50seed/summary.md`

```bash
git add src/auv_lowlevel/batch/env.py tests/test_batch_closed_loop.py
git commit -m "feat: add vectorized REMUS100 control environment"
git push origin goal10-batch-torch-remus100
```

### P2：rsl_rl 适配

- [x] 实现 `BatchTorchVecEnv`，保持 rsl_rl 2.3.3 接口；rsl_rl 源码未修改；
- [x] observation/action/reward 全程使用 tensor；网络侧 float32，物理内部默认 float64；
- [x] `env.num_obs`、配置中的 `D` 和 rsl_rl 实际 observation tensor 的最后一维由同一个 `observation_schema` 驱动；v1 27D 和 v2 26D 均通过接口 smoke，代码不写死 `27`；
- [x] mask reset 已移除显式 `.item()`、`nonzero()` 和布尔索引；CUDA 同步 debug 审计在 64 env × 100 step、v2 26D 下为 `sync_warning_count=0`，step 热路径不再触发可检测的 host 同步；
- [x] 本机 CPU 完成 `4096×64`、2 iteration 的 26D/27D 完整脚本 smoke；正式 GPU `4096×256` 归入 P2.5；
- [x] manifest 记录 backend、dtype、schema、seed、batch size、配置哈希、PVS 对照版本、总步数和吞吐；
- [x] P2.5 前补齐 step p50/p95、rollout/PPO 时间、显存、KL、clip fraction、entropy、loss、reset 原因和 NaN/Inf 计数。

配置文件口径：现有 `configs/lowlevel/odarrl_control_batch4096.json` 是 v1 27D 兼容 smoke，不得直接作为正式训练配置。历史 P2.5 pilot 使用 `configs/lowlevel/odarrl_control_batch4096_v2.json`，该文件为了保留历史哈希仍指向原 27D base，但通过 `observation_schema: "v2"` 驱动出 26D 策略输入。从现在起，新实验统一使用 `configs/lowlevel/odarrl_control_batch4096_v2_26d.json`，它显式指向 `configs/lowlevel/odarrl_control_v2_26d.json`；两者除观测维度外与 base 完全一致。旧配置不覆盖、不回写。

**P2 当前证据（2026-10-31，本机 CPU 与小规模 rsl_rl smoke）：**

| 项目 | 结果 | 说明 |
|---|---:|---|
| `tests/test_batch_*.py` + `test_lowlevel_ppo_config.py` | `48 passed` | 26D/27D、auto-reset、收敛和配置校验；服务器同步后关键子集 `22 passed` |
| rsl_rl 2.3.3 接口 | 4 env × 8 steps × 1 iteration | 26D/27D 均完成，未修改 rsl_rl 源码 |
| 本机 4096 env smoke | 4096 × 64 × 2 iteration | 26D/27D 均完成；CPU 吞吐约 `311k env-steps/s` |
| CUDA sync 审计 | 64 env × 100 steps，v2 26D | `sync_warning_count=0`；证据 `artifacts/lowlevel/batch4096/cuda_sync_audit_gpu5_64env_100steps_after_fix.json` |
| 正式 GPU 证据 | P2.5 已完成 | 服务器 RTX 4090 的 `4096×256` 26D pilot 已通过接口/显存/NaN 检查；学习质量和 PVS 门禁见 P2.5/P3 |

```bash
git add src/auv_lowlevel/batch/vec_env.py configs/lowlevel/odarrl_control_batch4096.json
git commit -m "feat: adapt batched REMUS100 env to rsl_rl"
git push origin goal10-batch-torch-remus100
```

### P2.5：26D 主 pilot 与后端短程试跑

这一阶段只回答“批处理后端是否值得进入正式训练”，不用于判断最终控制效果。**主 pilot 直接使用 v2 26D**；27D 只做一次独立兼容回归，不参与吞吐或学习信号的主结论。正式训练必须等试跑通过后再启动。

证据有效性说明（2026-10-31）：首次服务器 GPU pilot 虽然完成了 `4096×256` 的 5 个 iteration，但 `BatchTorchVecEnv` 的 episode 级累计指标错误地除以 episode 数而不是已完成步数，导致 residual、RMSE、material saturation 和 constraint violation 数值失真，因此该次运行的 episode 指标不作为验收证据。修复后，`server_gpu5_4096x256_10iter_metricslogfix_20261031_1531` 修正了指标分母和 rsl_rl 日志聚合，但审计又发现新批量后端把原始 `reward.total` 而不是 `reward.ppo_scaled` 交给 PPO，导致 value loss 从应有量级放大到数千。修复后，PPO 使用归一化奖励、日志仍保留原始 return；学习质量与最终 50-seed 预筛以 `server_gpu4_4096x256_10iter_rewardscale_20261031_1540` 为准，gpu5 运行只保留吞吐和接口参考。

- [x] `4096×64` 接口 smoke：本机 CPU 已运行 2 个 iteration，检查 `observation.shape == (num_envs, D)`、v2 的 `D=26`、索引 `23:26` 残差历史、reset mask、终止原因和 PPO 更新无异常；v1 的 `D=27` 兼容 smoke 也已通过，不作为新实验路线；
- [x] `4096×256` 26D 短程 pilot：服务器 GPU4 已用 reward-scale 修复版运行 10 个 iteration，固定 seed、配置和 PVS 对照版本；基础策略使用在线 PD/PI，不依赖 BC checkpoint；
- [ ] 可选迁移对照：若已有可用旧 27D teacher，再单独跑 26D student 的 BC/DAgger/蒸馏 pilot；它是能力迁移实验，不是批量后端通过门禁的前置条件；
- [x] 记录每个 iteration 的 `env-steps/s`、单步 p50/p95、rollout 耗时、PPO 更新耗时和峰值显存；
- [x] 记录 `approx_kl`、clip fraction、entropy、value loss、policy loss、reward、episode length、reset 原因和 NaN/Inf 计数；
- [x] 与 256/1024/2048 环境的小规模吞吐对照，确认 4096 的收益不是完全被 PPO 更新成本抵消；
- [x] 在 manifest 中写清 backend、`schema_version=v2`、`D=26`、`num_envs`、`num_steps_per_env`、dtype、seed、git commit、配置哈希和试跑起止时间。

**P2.5 本机 CPU pilot 证据（2026-10-31）：**

- 路径：`artifacts/lowlevel/batch4096_smoke_v2/local_cpu_4096x64_r1`；
- 规模：`4096` env × `64` steps × `2` iterations，共 `524,288` env-steps；
- collection 吞吐约 `196k env-steps/s`，含 PPO 更新的端到端吞吐约 `38.8k env-steps/s`；
- step p50/p95 约 `13.3–13.8 / 14.3 ms`；
- KL 约 `0.0010 / 0.0007`，clip fraction 约 `0.0049 / 0.0027`，loss 有限，NaN/Inf 为 0；
- 该结果只证明指标链和 26D 训练接口可用，不能替代服务器 RTX 4090 的 `4096×256` 正式 pilot。

**P2.5 服务器 GPU pilot 证据（2026-10-31，reward-scale 修复版）：**

- 路径：`artifacts/lowlevel/batch4096_smoke_v2/server_gpu4_4096x256_10iter_rewardscale_20261031_1540`；
- 规模：`4096` env × `256` steps × `10` iterations，共 `10,485,760` env-steps；schema `v2`、`D=26`、float64 物理、seed `30000`；
- `BatchTorchVecEnv` 纯 step 计数约 `292.3k env-steps/s`；逐 iteration 的 rollout collection（含策略前向和 buffer 写入）为 `157k–187k env-steps/s`，平均约 `173k`；含 PPO 更新的端到端吞吐为 `112k–128k env-steps/s`。三者必须分开报告，不能把纯 step 吞吐当作完整训练吞吐；
- step p50 为 `13.36–13.60 ms`，p95 为 `15.51–15.88 ms`；
- GPU peak allocated `424.1 MiB`，reserved `518.0 MiB`；
- value loss 为 `0.008–0.508`，KL 为 `0.00037–0.00108`，clip fraction 为 `0.0011–0.0130`，所有 loss 有限，NaN/Inf 计数为 0；
- 第 5/6 iteration 的 episode 指标分别为 success `1.000/1.000`、3D RMSE `2.842/1.648 m`、residual abs mean `0.121/0.121`、material saturation `0.178/0.133`；
- 第 10 iteration 出现短程波动：success `0.480`、timeout `0.520`、3D RMSE `8.596 m`、material saturation `0.220`；10 个 iteration 只有少量 episode 完成，不能据此判断最终控制质量；
- `training_manifest.json` 和 `final.pt` 均记录 `observation_schema=v2`、`observation_dim=26`、`vector_env_backend=batch_torch`、`physics_dtype=float64`。

**P2.5 规模对照（2026-10-31，`256 steps × 3 iterations`，只用于吞吐/显存，不用于学习质量）：**

路径：`artifacts/lowlevel/batch4096_smoke_v2/scaling_{256,1024,2048,4096}env_256steps_3iter_20261031_1540`。

| `num_envs` | rollout collection env-steps/s | 端到端 env-steps/s | step p50 ms | step p95 ms | peak allocated MiB | peak reserved MiB |
|---:|---:|---:|---:|---:|---:|---:|
| 256 | 12,157.9 | 8,596.7 | 14.449 | 16.175 | 104.5 | 122.0 |
| 1024 | 50,213.3 | 35,559.9 | 13.380 | 15.647 | 155.2 | 212.0 |
| 2048 | 101,794.9 | 70,711.5 | 13.528 | 15.456 | 223.3 | 321.3 |
| 4096 | 198,788.6 | 145,253.5 | 13.563 | 16.208 | 423.9 | 517.3 |

4096 环境相较 256 环境的 rollout collection 收益接近线性，但 PPO 更新仍占约四分之一 wall-clock；显存远低于 24 GB。规模对照运行发生在 reward-scale 修复前，因此表中只采用吞吐、延迟和显存，不把其 value loss 或 episode 指标当成学习证据。

**26D 批量 50-seed 预筛（2026-10-31）：**

- 路径：`artifacts/lowlevel/batch4096_smoke_v2/eval_rewardscale_26d_50seed_20261031_1548`；
- checkpoint：`server_gpu4_4096x256_10iter_rewardscale_20261031_1540/final.pt`，SHA256 `ec4b0c8a…63ad9a`，schema `v2`、`D=26`；
- 批量 Torch 后端成功率 `50/50 = 1.00`，三维 RMSE `1.51546 m`，水平 RMSE `1.46950 m`，垂向 RMSE `0.32364 m`，平均到达 `27.93 s`；
- material saturation 为 `0`，但 `saturation_rate_mean` 为 `0.2086`、constraint violations 为 `325.48` 次/episode；同 seed 的 B0 PVS 参照均为 `0`。后续逐 seed 诊断见 P3，确认它主要是舵角顶到 15° 边界后的微小幅值裁剪，不是材料级饱和；
- 该结果只用于批量后端预筛。正式结论以 P3 的 CPU PVS 50-seed 复核为准；批量预筛不再单独承担 26D 通过/失败判定。

结论：P2.5 的后端接口、显存、指标链和规模扩展已验证；纯 step 吞吐通过 `200k`，但完整 rollout/端到端仍低于该目标，P4 需要继续 profile。26D 批量预筛和后续 PVS 复核的成功率/RMSE 均与 B0 接近；short pilot 第 10 iteration 的波动说明它不能代替正式长训练，而不是 26D schema 本身失败。

建议门禁：

- 正确性：所有 smoke 无 NaN/Inf、无异常 reset、无 PPO 维度错误；
- 性能：26D `4096×256` 完整环境达到 `200k env-steps/s` 才进入 P3；`100k–200k env-steps/s` 只允许继续 profile，不直接开长训练；
- 稳定性：峰值显存低于单卡可用显存的 70%，且至少 5 个 iteration 的吞吐和损失量级稳定；
- 学习信号：entropy 不在一开始塌缩，clip fraction 和 `approx_kl` 不持续异常，reward/episode 指标至少不出现系统性恶化；短程试跑不以成功率提升作为门禁。

试跑失败时先降级到 2048 或 1024 环境定位问题，不通过缩短 episode、改变物理方程或减少观测来“跑通”。

### P3：26D 正式化与可选策略迁移（强制门禁）

- [x] `OBSERVATION_SCHEMA_V2`（`D=26`）已接入标量 `ObservationBuilder`、config loader、批量 observation、vec-env 和 PPO 配置，并通过 26D shape smoke；不另建第二套 26D schema；
- [x] 在训练配置、vec-env 配置和 checkpoint 元数据中显式记录 `schema_version` 与 `dim`，并提供加载时一致性校验；新 batch 配置缺省 v2，旧 pvs 配置缺省 v1；
- [x] 完成 v2 的观测逐元素一致性、reset 历史、`N=4096` shape、标量 v2 为 v1 精确前 26 列，以及与 v1 前 26 列完全相同的回归测试；
- [x] 先确定落地路线：**直接 26D 为默认主路线**（在线 PD/PI 基座 + 26D 残差 PPO，或从零训练 26D）；只有在需要复用旧 27D 策略能力时，才执行下面的迁移步骤；
- [x] 已完成批量 Torch 的 26D checkpoint 50-seed 预筛：成功率 `1.00`、三维 RMSE `1.51546 m`、material saturation `0`；但 constraint violations 为 `325.48` 次/episode，且该结果不是 CPU PVS 正式门禁；
- [x] 新增 canonical v2 环境配置 `configs/lowlevel/odarrl_control_v2_26d.json`：与 `odarrl_control_base.json` 除 `observation.dim=26` 外完全一致；canonical hash 为 `c19a5867ae298aaf373339e172be3d78e492c0b62c821d303e2ca2b60e0f72e7`；
- [x] 新增 canonical 26D 批量训练配置 `configs/lowlevel/odarrl_control_batch4096_v2_26d.json`，显式指向上述环境配置；历史 `odarrl_control_batch4096_v2.json` 保留原哈希；
- [x] 给 PVS 评估器补上 training/environment schema、`D=26` checkpoint metadata 和 actor warm-start 维度校验，防止 27D checkpoint 被误用于 26D 配置；批量评估器已有同类校验；
- [x] 26D 部署路径没有偷偷读取在线教师积分：策略 actor 输入为 26 维，测试覆盖 `D=26` 和 checkpoint metadata；需要时序信息时只能显式增加历史或循环结构；
- [x] 与 B0 做同 seed 配对的 PVS 50-seed 评估，报告成功率、三维 RMSE、material saturation、约束计数和推理延迟；
- [x] 所有新增训练配置默认使用 26D；27D 仅保留在旧实验配置、旧 checkpoint、历史数据和显式兼容回归中，不再新增 27D 训练分支；
- [ ] **可选迁移（不阻塞直接 26D）**：若未来要复用旧 27D teacher，先做 naive drop-column BC 基线，把 27D 数据的第 27 列删除训练 26D 学生并记录性能损失；
- [ ] **可选迁移**：采用 BC/DAgger/蒸馏，固定 27D teacher，在闭环中采集 student 实际访问的状态，以 teacher 动作为标签训练 26D student；naive BC 只作为对照；
- [ ] **可选迁移**：补齐 27D → 26D 的数据采集、BC/DAgger 训练和动作蒸馏入口；当前仓库尚未提供 `scripts/distill_control_policy_26d.py`，在该脚本实现并通过 smoke 前，不把迁移路线写成已就绪；
- [ ] **可选迁移**：复制 27D 网络除最后输入列外的权重，将删除列置零作为热启动，并与随机初始化 26D student 对照；权重移植不能单独作为迁移成功证据。

**P3 PVS 50-seed 正式复核（2026-10-31）：**

- 结果路径：`artifacts/lowlevel/batch4096_smoke_v2/pvs_eval_rewardscale_26d_50seed_20261031_1600`；
- checkpoint：`server_gpu4_4096x256_10iter_rewardscale_20261031_1540/final.pt`，SHA256 `ec4b0c8a28837cdf47484923af57a743b9003201ad3548d4a23d20f35c63ad9a`，metadata 为 schema `v2`、`D=26`；
- 成功率：B0 `50/50`，26D residual `50/50`，差 `0`；
- 三维 RMSE：B0 `1.514748 m`，26D residual `1.515458 m`，劣化 `0.000710 m`；水平差 `0.000825 m`，垂向改善 `0.000286 m`；
- material saturation：两者均为 `0`；residual clip rate 为 `0`；平均到达时间 `27.94 → 27.93 s`；
- 控制环 p95 延迟的均值为 `10.07 ms`（B0）和 `9.42 ms`（26D residual），满足 50 Hz 时序；PVS 单步 p95 均值为 `0.240 ms` 和 `0.238 ms`；
- 历史 pilot 的 checkpoint metadata 记录的是旧 batch 配置和 base 环境文件哈希；PVS 复核显式 override 到 canonical 26D 环境配置。两者物理参数相同、仅观测维度不同，且 schema/dim 均为 v2/26。新配置 `odarrl_control_batch4096_v2_26d.json` 已消除这一配置来源歧义，后续新训练必须使用它；
- `constraint_violations=325.48` 的拆解：50 seeds 合计 `16,274` 次，其中幅值裁剪 `15,559` 次、速率裁剪 `715` 次；最大舵角请求偏差为 `0.0502 deg`，最大 RPM 偏差为 `0`。原因是基座舵角经常已经贴近 `15 deg` 上限，残差再次向外请求时被微小裁剪；它远低于 `0.01 × full_scale = 0.15 deg` 的 material threshold，因此不代表材料级饱和或部署不可用。后续报告应同时给出 `material_saturation_rate` 与微裁剪计数，不能只把任何非零 clip 都解释为控制失败；
- 本地校验：`uv run pytest -q tests/test_lowlevel_*.py tests/test_batch_*.py` 为 `81 passed`，`ruff check` 通过；canonical 环境配置 hash 为 `c19a5867…0f72e7`，批量 26D 训练配置 hash 为 `5f866208…e75678`；
- 判定：**直接 26D 的 P3 核心门禁通过**。成功率下降 `0`、RMSE 劣化 `0.00071 m`、material saturation 不上升，均满足暂定门禁；BC/DAgger/蒸馏迁移没有执行，也不再是 26D 正式化的前置条件。

建议门禁（已用于本次判定）：26D 相对 B0 的成功率下降不超过 `0.02`，三维 RMSE 劣化不超过 `0.05 m`，material saturation 不上升；26D 策略还必须经过至少一次短程 PPO 训练，确认 actor loss、entropy、clip fraction 和 episode 指标没有因 schema 切换出现异常。迁移路线若未来启用，若未通过先改蒸馏数据、损失和闭环采样，不通过隐藏积分特征“补回”第 27 维。

```bash
git add src/auv_lowlevel/batch/observation.py src/auv_lowlevel/config.py src/auv_lowlevel/batch/env.py \
  scripts/evaluate_odarrl_control_residual.py scripts/train_odarrl_control_residual.py \
  tests/test_batch_observation_actions_reward.py tests/test_batch_closed_loop.py \
  tests/test_lowlevel_config.py tests/test_lowlevel_residual_eval.py tests/test_lowlevel_ppo_config.py \
  configs/lowlevel/odarrl_control_v2_26d.json configs/lowlevel/odarrl_control_batch4096_v2_26d.json
git commit -m "feat: make 26D observation schema the default"
git push origin goal10-batch-torch-remus100
```

### P4：性能、正式训练与报告

- [x] 对 `step_core` 做 profile；证据为 `artifacts/lowlevel/batch4096_smoke_v2/step_core_profile_gpu0_20261031_1640/summary.json`，4096 env、26D、float64、GPU0 的 mean 为 `23.71 ms`、p95 为 `28.52 ms`、纯 step 为 `172,755 env-steps/s`；profiler 显示每步约 `1,775` 次 CUDA kernel launch，主要瓶颈是 CPU launch/synchronization；
- [x] 只在必要时启用 `torch.compile`/CUDA Graph；已实现默认关闭的 `vector_env.compile_step`/`compile_mode="default"`，CUDA Graph 明确禁用。`scripts/verify_batch_compile.py` 在 GPU5、64 env、40 步、episode 11、192 次 reset 下得到 observation 最大差 `8.56e-18`、reward `0`、done `0`、actuator `4.26e-14`，全部低于 `1e-10`；证据为 `artifacts/lowlevel/batch4096_smoke_v2/compile_verify_gpu5_20261031/summary.json`；
- [x] 比较 256/1024/2048/4096 env 的吞吐、延迟和显存；
- [x] 比较不同环境规模下的最终控制质量；4096/1024/256 使用同一训练 seed `40000`、同一 `26,214,400` env-steps 预算和共同 `model_100.pt` 完成 20-seed PVS 对照；
- [x] 从通过 P3 门禁的 26D 策略继续训练，或以同一 26D 配置从零训练；本轮选择从零训练，使用 `odarrl_control_batch4096_v2_26d_p4.json` 完成 `200 × 4096 × 256 = 209,715,200` env-steps，没有新建 27D 分支；
- [x] 对 P4 的正式长训练策略做 PVS 50-seed 评估；20-seed 筛选显示 `model_25` 优于 `model_50`、`model_100` 和 `final`，`model_25` 的 50-seed 结果为成功率 `1.00`、三维 RMSE 差 `+0.00804 m`、material saturation `0`，通过门禁但没有超过 B0；
- [x] 写跨后端一致性和性能报告，并明确区分 v1 27D 兼容结果与 v2 26D 正式结果；报告见 `doc/goals/goal10_batch_torch_remus100_report.md`。

**P4 等预算规模对照结果（2026-10-31）：**

P4 配置中的 `experiment.arm="I"` 是历史标签，但 `base_policy.mode="online_pd_pi"`；这些结果在语义上属于 O 臂 residual-only 预检，不是 I1/I2。后续配置必须让 `arm` 与基础策略来源一致。

原设计预算是 `400 iterations × 262,144 = 104,857,600` env-steps。为了让 4096/1024/256 在“同更新次数、同总样本量”下比较，控制质量统一取共同拥有的 `model_100.pt`，即 `100 × 262,144 = 26,214,400` env-steps。4096 和 1024 之后跑完了 400 iterations；256 在 `model_100` 后主动停止，因为相同 env-step 预算需要多得多的 Python/vec-env step 调用。停止原因记录在 `artifacts/lowlevel/p4_26d_scale_256_20261031/stop_note.json`。

| envs | 前 100 轮墙钟 env-steps/s | 20-seed 三维 RMSE 差 | material saturation | residual abs mean | constraint violations/ep |
|---:|---:|---:|---:|---:|---:|
| 4096 | `115,034.7` | `+0.018037 m` | `0 → 0` | `0.05734` | `0 → 86.8` |
| 1024 | `66,599.3` | `+0.022648 m` | `0 → 0` | `0.08458` | `0 → 1.6` |
| 256 | `25,918.4` | `+0.032291 m` | `0 → 0` | `0.12097` | `0 → 1.15` |

三组成功率均为 `1.00 → 1.00`，共同 B0 三维 RMSE 为 `1.565020 m`。4096 的端到端吞吐约为 1024 的 `1.73×`、256 的 `4.44×`；PPO 学习时间近似不变，差异主要来自环境 step 调用次数。4096 的质量最接近 B0，但三种规模都没有超过 B0。

4096 规模对照跑到 400 轮后的 `final` 反而变差：20-seed 三维 RMSE 差 `+0.034527 m`、material saturation `0.000321`、residual abs `0.17070`。因此本轮不是“训练越久越好”，正式候选必须按 PVS 闭环结果选择。

```bash
git add doc/goals/goal10_batch_torch_remus100_feasibility.md \
  doc/goals/goal10_batch_torch_remus100_report.md \
  doc/goals/goal9_odarrl_control_reproduction.md
git commit -m "docs: report batched REMUS100 training results"
git push origin goal10-batch-torch-remus100
```

## 8. 工作量和风险

一个熟悉当前代码和 PyTorch 的工程师，建议预算：

- P0：3–5 个工作日；
- P1：5–8 个工作日；
- P2：2–3 个工作日；
- P2.5：1–2 个工作日；
- P3：3–5 个工作日；
- P4：3–5 个工作日；
- 总计：**4–6 周**，不含论文级超参搜索。

主要风险：

1. **长期漂移**：非线性系统对 float32 和积分顺序敏感，必须用 PVS 分布级对照，而不是单条轨迹；
2. **路径投影开销**：全量 `N×M` 搜索可先实现，后续再优化为局部窗口；不能为了快而改变 tie-break；
3. **reset 分支导致 GPU 同步**：应使用 mask 和预分配 buffer，避免 Python 逐环境 reset；
4. **rsl_rl 契约**：4096 环境是一次进程内的 4096，不是自动 8 卡 32768；先做单卡；
5. **显存**：先用 `4096×256`，不要直接上 `4096×1024`；
6. **可调试性下降**：batch backend 出错时，先用小 N 和逐环境 reference 做二分定位。
7. **26D 迁移不是无损删列**：删除积分后教师动作不再是严格的当前观测函数，必须用 BC/DAgger/蒸馏和闭环评估，不能把 naive drop-column 结果当成等价。
8. **约束计数口径**：`constraint_violations` 会统计任何超过数值容差的幅值/速率裁剪，而 `material_saturation_rate` 只统计超过 1% 满量程的裁剪。26D pilot 的 325.48 次/episode 已确认主要是顶到 15° 边界后的亚 0.051° 微裁剪；报告必须同时给出两者和最大偏差，不能把非零微裁剪直接等同于材料饱和。

## 9. 推荐决策

- 批量 Torch REMUS100 路径可行，`4096×256` 是单卡 RTX 4090 的默认训练配置；资源不足时 1024 可降级，但必须把 PPO 更新和 Python 调度成本算入端到端预算；
- 26D 是长期正式 schema。新实验默认直接训练 26D；27D 只保留 legacy checkpoint、历史数据和兼容回归，不再新增 27D 训练分支；
- 若后续需要复用旧 27D 策略，使用 BC/DAgger/动作蒸馏得到 26D student；这是可选的迁移实验，不是 26D 新实验的前置条件；
- 批量环境负责吞吐和候选筛选，CPU PVS 负责正式 50-seed 结论；任何批量后端数字都不能直接替代 PVS 报告；
- 本轮 PPO 没有超过 B0。不要把“4096 环境跑起来”或“训练 reward 更高”写成控制增益；后续提升精度应另开奖励、探索、学习率和 early-stop 实验；
- 本轮正式候选按 PVS 选 `model_25.pt`，不默认使用 final。规模对照也显示 4096 的 400 轮 `final` 差于共同 `model_100.pt`。

## 10. 关闭情况与剩余风险

P4 完成后，goal10 的可行性问题已经关闭。以下项目要么有直接证据，要么被明确划出本 goal 范围；它们不再是批量后端启动前的阻塞项。

| 项目 | 结论 | 证据或边界 | 后续 |
|---|---|---|---|
| 实验范围 | 已关闭 | 新实验统一 26D；P1/P2 的 v1 27D 只用于兼容；旧配置不回写 | 只维护显式 `schema_version=v1` 的 legacy 回归 |
| 26D 落地 | 已关闭 | P3 pilot 50-seed 三维 RMSE 差 `+0.00071 m`；P4 `model_25` 50-seed 差 `+0.00804 m`，成功率 `1.00`，material saturation `0` | 新实验直接 26D；需要旧能力时另做迁移 |
| 环境规模与吞吐 | 已关闭 | 等预算前 100 轮：4096/1024/256 分别为 `115,034.7/66,599.3/25,918.4` env-steps/s；4096 长训练墙钟 `267k` env-steps/s | 默认 4096；资源不足再降级 |
| 规模控制质量 | 已关闭，负结果 | 共同 `model_100` 的 20-seed RMSE 差分别为 `+0.018037/+0.022648/+0.032291 m`；4096 最接近 B0 但都未超过 | 不再靠增加环境数追求控制增益 |
| PPO 后期漂移 | 已关闭，负结果 | 4096 长训练 `model_25` 优于 final；规模对照 `model_100` 优于 400 轮 final | 正式候选按 PVS 选择，另做 early-stop/超参消融 |
| rsl_rl 接口 | 已关闭 | 2.3.3 完整跑完 200 iterations，无 NaN/Inf；未修改 rsl_rl 源码 | 后续保持适配器边界，不把训练参数写进 rsl_rl |
| reset/RNG、路径和 ALOS | 已关闭 | P1 固定 seed、mask reset、局部窗口路径投影、终端捕获和 50-seed B0 对照均通过；B0 三维 RMSE 差 `-2.22e-16 m` | 修改物理或路径逻辑时继续跑同一回归 |
| 数值精度 | 已关闭 | float64 物理 + float32 网络边界；单步、500 步开环和 compile 一致性均低于门禁 | 全 float32/混合精度只作为独立消融 |
| BC/DAgger/蒸馏迁移 | 可选，未执行 | 当前仓库没有 `scripts/distill_control_policy_26d.py`；直接 26D 不依赖它 | 仅在需要复用旧 27D 策略时单独立项 |
| 规模结论的统计强度 | 剩余风险 | 每个环境规模只有一个训练 seed `40000`；20-seed 指评估 seed，不是 20 次独立训练 | 若要写成论文级“规模因果”，至少重复 3 个训练 seed |
| 多卡和超参搜索 | 本 goal 外 | 第一版固定单进程、单卡；本轮没有做多卡 PPO 或系统化 HP 搜索 | 另开性能或调参 goal，避免混入后端可行性结论 |

goal10 到此关闭。任何后续“让 PPO 真正超过 B0”的工作，应作为新的控制优化实验处理，并继续使用 CPU PVS 50-seed 作为正式门禁。
