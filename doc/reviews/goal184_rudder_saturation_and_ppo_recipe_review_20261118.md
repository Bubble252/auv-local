# Goal184：舵饱和控制与 BCFT / BCRES 配方评审

日期：2026-11-18。状态：只读评审与建议，不是新 Goal、冻结协议或正式实验结论。
范围：Goal182 D13 reward、Goal183 empirical BC、Goal184 PE2E Wave 1，以及 Wheel-Legged-Lab@e61bfe1。
本次仅新增本文；没有修改训练代码、配置、阈值，没有停止或重启训练，也未 commit/push。

## 1. 判断与优先级

目标是减少无效的大幅舵动作和不可执行的高速反转，同时保留抗流、转弯和路径跟踪能力；不把饱和率压到零作为独立优化目标。

建议保留现有 BCFT / BCRES 作为完整对照，不把两套配方直接平均。BCFT 保护已有 BC 行为，BCRES 允许独立残差学习，两者承担不同任务。当前训练快照呈现 BCRES 跟踪误差较小、执行器裁剪较多的权衡，尚不能宣布赢家。

改动优先级：先解释饱和口径 → 区分随机探索与确定性策略 → 根据原因只改一个因素。优先利用已有记录和既定评价，不另开超参扫描，不因单个软指标停掉现有实验；完整 BCFT 仍必做。

## 2. 当前饱和率到底统计什么

代码：`src/auv_lowlevel/batch/actions.py` 的 `BatchedPureActionPipeline.apply()`；`batch/pc_env/core.py` 使用该 pipeline；`batch/pc_vec_env.py` 聚合 episode 日志。

当前 PE2E 配置：舵/艉翼幅值各 ±20°，速率 150°/s，dt=0.02 s，所以每步最多变化 3°；材料容差是满量程 1%，即每片舵 0.2°。

`material_saturated` 是任一通道发生超过材料容差的幅值裁剪或速率裁剪的布尔 OR。它不是每个舵的占满比例，也不是三通道饱和比例的平均。

尤其要注意当前执行顺序：

1. 归一化动作先 clamp 到 [-1,1]。
2. 转换为物理舵角/RPM。
3. 检查物理幅值限幅。
4. 舵角变化限制到每步 3°，计算材料速率裁剪。

第 1 步已经把请求映射进物理幅值包络，因此当前纯控制 pipeline 的后续幅值裁剪通常为零；归一化越界另记 `normalized_magnitude_clipped`，不直接并入 `material_saturated`。由此，PE2E 的材料饱和率主要反映舵/艉翼的速率裁剪，不能解释成“有这么多时间舵角在 ±20°”。恰好稳定保持 20°，没有新裁剪，也可能不触发该指标。

当前 reward 的 requested 输入是 `applied.normalized`，已经过 clamp。未来即使启用 legacy clip-attempt 项，也必须先核对输入位置；只把权重从 0 改为非零，并不能保证惩罚到原始越界动作。

最小诊断应分别看：

| 量 | 用途 |
|---|---|
| 舵 / 艉翼各自的速率裁剪率及请求-执行差 | 判断抖动或不可执行纠偏 |
| 舵 / 艉翼各自的近限位占用与连续停留时间 | 判断持续打满；例如 95% 满量程只作诊断分界，不是新 Gate |
| 原始动作越界率、幅值超出量 | 判断采样或均值落在可执行范围外 |
| 实际舵角变化、反转次数、RMS | 判断实际执行器负担 |
| BC 基座、残差均值、探索 std | 区分来源；std 按通道看，不能只看全通道 mean |
| 同条件路线 cross/vertical、completion、safety | 防止通过不转舵或走不动降低饱和 |

已有 info 提供 `material_rate_clipped`、`material_magnitude_clipped`、`requested_action`、`limited_action`、`actual_actuator`；优先复用。是否已经保存到具体 run 的 trace 需要按产物核验，不能把运行内字段当成磁盘证据。缺可选遥测记缺失，不阻塞当前实验。

## 3. 奖励目前能约束什么

配置：`configs/lowlevel/goal184/env_path_fixed_speed_formal_empirical.json`；实现：`batch/pc.py`。

| 项 | 当前值与作用 | 局限 |
|---|---|---|
| 材料饱和 | -0.02 × 任一通道材料裁剪事件 | 不区分舵/艉翼、超出多少、持续多久 |
| 一阶平滑 | -0.1 × 实际归一化动作变化平方和 | 对实际执行动作计算；限速已经削弱了输入变化 |
| 二阶平滑 | -0.005 × 实际舵/艉翼二阶差分平方和 | 不罚 RPM；也不能单独辨识裁剪前的大请求 |
| clip attempt | 权重 0 | 当前未提供学习约束 |
| cross / vertical | 3.5 / 1.5 的负 Huber | 跟踪任务优先，可能接受更多执行器裁剪换取误差下降 |

以上总奖励再乘 `pc_ppo_scale=0.01` 送入 PPO；饱和事件的单步 PPO 贡献仅 -0.0002。以 cross 误差 1 m、Huber delta=0.5 m 为例，cross 原始惩罚为 -1.3125，明显大于 -0.02。这个量级比较说明饱和项较弱，但不等于最优权重已知，也不能据此直接放大几十倍。

不要只增大实际动作平滑：它可能让纠偏迟钝，却没有让网络看清裁剪前请求。不要放宽物理限幅、提高容差、减少统计通道来制造“饱和改善”。不要把参考机器人的关节位置越界终止直接搬成 AUV 舵打满终止：机械关节越界与正常执行器达到额定舵角的语义不同。

## 4. 从参考代码借鉴什么

参考 `mdp/rewards.py:soft_joint_limits_penalty()` 在离软关节边界 0.15 rad 时开始线性扣分，权重 -10；注释写二次但实际运行代码是一次。四个腿关节越过软限位会终止；基础 flat 任务非超时终止权重 -100，经 dt=0.01 后事件贡献 -1。力矩平方另有惩罚；力矩饱和率仅作 telemetry，控制层直接限幅。

可借鉴的是提前、连续、按物理量定义代价；不照抄权重或终止。

如果确定性策略确有持续满舵，候选方案是在实际舵角接近边界时加入弱的缓冲区代价。例如令 u=|δ|/δmax，候选起点 ρ=0.85：

`C_margin = [max(0, (u - ρ)/(1 - ρ))]^2`

`r_margin = -λ_margin * C_margin`

ρ=0.85 对 ±20° 舵意味着从 17° 起扣分；只是未验证候选，不是已批准阈值。舵与艉翼独立计算，不把 RPM 混进同一指标。λ 必须结合现有分项贡献选定，先做只读尺度审计，不通过压倒路线奖励来禁止正常抗流。

如果主要问题是速率裁剪，更直接的候选是请求与可执行量的连续差：

`C_rate = [(δ_req - δ_feasible)/δmax]^2`

这里 δ_req 是物理幅值限幅后的请求，δ_feasible 是基于上一执行值与速率上限算出的本步可执行值，二者单位必须一致。幅值越界另从 clamp 前动作计算。只针对已定位原因选择其中一项；不要同时堆上 margin、rate、clip-attempt 和大幅平滑权重。平方候选与参考的线性函数不是同一设计，需独立验证。

## 5. 当前两条 run：本地与服务器配置一致

2026-11-18 服务器只读核验：两进程均使用 `20261118-goal184-64d8225` worktree，seed86201、2600 iterations、相同 PE2E 冻结阈值，分别输出到：

`/data/yhx/underwater-robotics/runs/auv-local/goal184/20261118-goal184-64d8225/full_wave1/pe2e_{bcft,bcres}_seed86201`

训练 JSON 的本地/远端 SHA256 一致；manifest 记录也匹配。共同 env SHA 是 `db4de5ff1ed1d5cb9b4b37da6d04df95c82855d61d609cc6e0e5d61512cc9729`。

| 项目 | BCFT | BCRES |
|---|---|---|
| train JSON SHA256 | `0266775a3e0b7dc9d174dc08ffffe8d5a8bff1b969d4dafe2be03e57259670f4` | `7e117ba5be1e76f9c84d561e124abc0f9b023998a3fdc03d2bc71c349b157e2e` |
| 学习方式 | 加载 v4 BC 后微调 actor | 冻结同一 v4 BC，另学残差 |
| actor LR | 固定 1e-5 | 初始 3e-4，adaptive |
| critic LR | 固定 3e-4 | 初始 3e-4，随共享 optimizer adaptive |
| 初始 std | 0.05 | 0.3 |
| residual | 无冻结基座叠加 | scale=1.0，mask=[1,1,1] |
| clip / epochs / minibatches | 0.2 / 5 / 4 | 相同 |
| desired KL / entropy / gamma / lambda | 0.01 / 0.005 / 0.995 / 0.95 | 相同 |
| actor / critic MLP | [512,256,128] | 相同 |
| advantage | minibatch normalization | 相同 |
| profile / env / reward / seed | 4096×48、同 env、同 reward、86201 | 相同 |

截图的超参数基本准确，但“同课程”仅指相同课程机制与目标 mix，晋级与实际采样会由各自轨迹决定，不保证 episode 配对。

BCRES 默认 `output_mode=composed`：`mean=BC+scale*mask*residual_mean`，随后采样 `Normal(mean,std)`。探索噪声是在组合均值之后加入。因此降低 residual_scale 并不自动按比例降低 std；这也是不能用缩小残差权限替代探索治理的原因。当前 mean/std 细节见 `pc_policy.py:_actor_mean()`、`residual_action()`、`update_distribution()`。

此外，当前 `_configure_optimizer_learning_rates()` 明确要求不同 actor/critic LR 使用 fixed schedule。把“BCFT 分离 LR + BCRES adaptive”写到同一 JSON 不是现成可用的混合配方，需要先改优化器调度语义。

## 6. 训练中途快照：收益与负担并存

服务器时间字段为 2026-11-18T20:40:20（服务器输出未附时区）。来源为上述 run 的 `iteration_records.jsonl` 与 `curriculum_route_gate.jsonl`。两条均 running，BCFT it323、BCRES it324，均 C5；rollout mix 为 C5 主导并混入其他阶段。这是在线随机策略数据，不是纯 C5 的配对确定性评价。

| 指标 | BCFT | BCRES |
|---|---:|---:|
| 当前 actor LR | 1e-5 | 0.00129746 |
| 当前 critic LR | 3e-4 | 0.00129746 |
| 当前 mean noise std | 0.05889 | 0.46130 |
| 当前 approx KL | 0.00348 | 0.01314 |
| 当前 clip fraction | 0.0304 | 0.1711 |
| 最近20轮 cross RMSE，m | 1.0400 | 0.5408 |
| 最近20轮 vertical RMSE，m | 1.8040 | 1.3785 |
| 最近20轮 completion | 0.97735 | 0.97700 |
| 最近20轮材料饱和率 | 0.32616 | 0.85553 |
| gate窗口 safety rate | 0 | 0 |

“最近20轮”数值是各轮指标的等权平均，不是重新按 episode/步数加权的总体统计。两条最新 losses finite，最新 nonfinite action/observation/reward 均为 0。最新 timeout rate 均为 0；success 为 0.79365 / 0.76866，仅作 telemetry。gate 因尚未达到 C5 soft iteration cap=336 而未晋级，不能写成 Gate FAIL。

初始 std 不能代表当前探索幅度：0.3 已涨到约 0.46，且 adaptive LR 会升也会降，不是自动保守。按 ±20° 动作映射，若某一舵通道 std 恰为全通道均值，0.46 对应裁剪前约 9.2° 标准差，而每步可执行变化只有 3°；这解释了速率裁剪风险，但通道真实 std、动作相关性和裁剪后的分布仍需核验，不能据此精确预测饱和率。

## 7. 哪套更好，最小修改应落在哪里

本次判断：BCFT 更适合保护已有 BC 和控制探索负担；BCRES 当前显示较强跟踪改善潜力，但随机探索与执行器负担较大。现有证据支持保留两条，而不支持简单选一条替代另一条；在线单 seed 快照不能成为 formal winner。

三种选择：

1. **现有两条保持协议，完成既定评价（当前推荐）**：保留证据与可比性；同步复用已有遥测解释来源。reward 不变时仍以完整路线表现、completion、safety 和执行器负担一起判断。
2. **BCRES 的探索做单因素治理（有条件的首选后续修改）**：若确定性均值动作跟踪良好、材料速率裁剪明显低于随机 rollout，保留 BCRES 架构/authority/LR/reward，单独约束舵通道探索 std 的上界。仅把 init std 改小不够，当前已有 std 自行增长证据；需确认底层策略支持该约束再实现。候选物理起点可按单步 3°/20°=0.15 的量级讨论，但它不是最佳 std 或安全保证，不直接作为批准值，也不扩成扫描。
3. **对确定性持续满舵或速率裁剪做连续代价（第二分支）**：若无采样噪声仍高，则单纯降 std 无法解决；先分辨 BC 基座、残差均值或路线/流场的必要控制需求，再选择第4节的一项代价。不要同时改 LR、std、reward、authority，否则无法归因。

不建议现在把 BCFT actor LR 提到 3e-4、std 提到 0.3；那会把 BC 微调改成强探索，失去原来的保护作用。也不建议把 BCRES 的 LR 直接压成 1e-5 并同步缩小 residual_scale，避免重新落入 Goal177/178 已记录的残差权限过小问题。

如果未来证据显示主因是 optimizer 更新导致均值突变而不是采样噪声，再单独考虑限制 adaptive LR 上界；不能仅因为观察到 1.30e-3 就认定它错误。当前 KL、clip 和 route 需要共同解释。

## 8. 后续执行与验证边界

- [x] 读取 Goal182–184、参考奖励和本地执行器/策略实现；只读核对远端进程、配置 SHA、manifest 与已有迭代指标。Git：无需提交证据日志。
- [x] 新增本文，核对公式、单位和相对链接；运行 `git diff --check`。Git：本轮不 commit/push。
- [ ] 在既定 evaluation 中按相同 checkpoint、seed、stage、control 对照确定性策略和现有随机 rollout，优先重用 trace；不启动 held-out 或新矩阵。报告按通道速率裁剪/近限位占用/路线指标。Git：只提交必要分析代码和报告，产物留 artifacts；提交前 `git diff --cached --check`，用户明确要求后才 commit/push。
- [ ] 只有归因成立且确定需要改协议，再在新 Goal/批准的 amendment 写 Research Contract、单因素变更、运行和验证命令；新 launcher 复用现有 profile/protocol/auditor。不能沿用旧 run root 静默续跑改 reward/std 的模型。Git：按变更文件显式 add，提交前检查，得到明确要求后 commit/push。

本评审未新开实验、不宣告 Goal184 完成；未发现需要凭本快照触发 autoresearch 的 hard blocker。若后续满足根 AGENTS 的触发条件，按 research/ 的 blocker、hypothesis、reading note、最小证伪与 ledger 流程执行，不用反复改参数替代研究。

## 9. 证据入口

- [Goal182 奖励与协议](../goals/goal182_ppo_stack_governance_20261114.md)
- [Goal183 BC_ONLY 结果](../goals/goal183_bc_only_result_20261118.md)
- [Goal184 主计划与 Wave 1](../goals/goal184_empirical_full_arm_training_20261118.md)
- [BCFT 配置](../../configs/lowlevel/goal184/train_formal_mod4_pe2e_bcft_env_empirical_5d_m_seed86201.json)
- [BCRES 配置](../../configs/lowlevel/goal184/train_formal_mod4_pe2e_bcres_env_empirical_5d_m_seed86201.json)
- [执行器限制](../../src/auv_lowlevel/batch/actions.py)
- [奖励实现](../../src/auv_lowlevel/batch/pc.py)
- [策略均值与采样](../../src/auv_lowlevel/batch/pc_policy.py)
- [优化器分组与训练入口](../../scripts/train_batch_lowlevel_ppo.py)
- 参考：`/home/bubble/港中深暑研/Wheel-Legged-Lab/source/wheel_legged_robot/wheel_legged_robot/tasks/manager_based/wheel_legged_robot/{mdp/rewards.py,wheel_legged_flat_env_cfg.py}`。

## 10. 补充：逐执行器归因（复用已有 trace）

2026-11-18 后续只读检查使用两条 run 已有的 `curriculum_evals/C4_049938432/default/trace.npz` 与 `curriculum_evals/C5_066453504/default/trace.npz`，没有新跑 rollout。各评价为 40 episodes、seeds 20–59、action_mode=policy（act_inference 确定性推理）、absolute 接口、decision_stride=1；这是课程边界诊断，不是正式冻结矩阵。两组 C4/C5 各自 checkpoint 的训练步数对应一致。

重算方法：trace 的 requested_action 是归一化动作，先 clamp 到 [-1,1]，乘 20° 得两片舵请求；按 episode_id 与 step 对齐上一时刻 actual_actuator。本步材料速率裁剪条件为 `abs(δ_req,t - δ_actual,t-1) > 3° + 0.2°`。每 episode 首步缺前态，排除，共排除40步；只保留相邻 step 连续记录。不能直接用本步 requested 与本步 actual 的差代替裁剪量，因为后者还包含动力学中的执行器响应。

以下每一行重算的两通道 OR 均与原 material_saturated 逐步完全一致，mismatch=0：

| 阶段 / arm | 有效步数 | 方向舵事件率 | 艉翼事件率 | 两者同时 | 总事件率 |
|---|---:|---:|---:|---:|---:|
| C4 BCFT | 54836 | 5.93% | 23.88% | 2.22% | 27.58% |
| C4 BCRES | 56913 | 26.58% | 28.57% | 9.92% | 45.23% |
| C5 BCFT | 61138 | 11.17% | 23.60% | 2.90% | 31.87% |
| C5 BCRES | 61430 | 39.01% | 33.21% | 15.84% | 56.38% |

比例均按该行有效步数加权；事件有重叠，不能直接将两通道比例相加。C5 BCFT 仅方向舵=8.26%、仅艉翼=20.70%；BCRES 仅方向舵=23.17%、仅艉翼=17.37%。当前 pipeline 的 RPM 不受舵速率限幅，且这批 trace 的两舵事件已完全解释总材料饱和，不存在需由 RPM 解释的剩余事件。这不表示 RPM 从未达到满转速。

判断：BCFT 的确定性材料饱和主要来自艉翼；BCRES 在 C5 是方向舵略占主导、艉翼也显著，C4 则两者接近。不能把此前提出的“主要关注方向舵”当成已验证的通用事实；治理必须同时保留艉翼分项。

训练 iteration_records 仍只有总材料饱和率，没有逐通道随机 rollout 事件；因此第6节约85.6%的随机训练饱和率不能直接按本表比例分摊。本表也表明 BCRES 确定性执行仍有明显速率裁剪，不能声称饱和全由探索噪声造成。下一步应同时检查确定性残差均值/基座和探索，不应直接断言只限制 std 就能解决。


## 11. 补充：速度跟踪诊断

复用第10节 C5_066453504 的两份 trace/summary，各40 episodes、seeds20–59，确定性课程边界评价；不是训练最新权重，也不是完整正式评价。没有新增 rollout。

首先区分口径：summary 的 speed_error 是 command.speed - ground_velocity_body.x；训练奖励则用目标路径速度向量旋转到机体系后的 x 分量作为目标。PE2E 的 command heading/pitch 由奖励所用投影切向构造，因此可从 command 三列和 eta 的 pitch/yaw 恢复该目标：tangent=(cos(pitch_cmd)cos(yaw_cmd), cos(pitch_cmd)sin(yaw_cmd), -sin(pitch_cmd))；body_forward=(cos(pitch)cos(yaw), cos(pitch)sin(yaw), -sin(pitch))；u_ref=command_speed*dot(tangent,body_forward)。这比使用 trace 中真值位置重投影的 route_tangent 更直接对应 reward 目标。参考与实际都为地速。

令 e=实际机体前向地速-u_ref，正值代表偏快。以下按有效 trace 全部时间步聚合，BCFT 61178步、BCRES 61470步：

| reward目标口径 | BCFT | BCRES |
|---|---:|---:|
| 速度 RMSE，m/s | 0.5661 | 0.5075 |
| MAE，m/s | 0.4416 | 0.4169 |
| 平均偏差，m/s | +0.3995 | +0.3952 |
| 绝对误差 p95，m/s | 1.1642 | 0.9148 |
| 绝对误差≤0.25 m/s占比 | 34.15% | 28.61% |
| 偏快超过0.5 m/s占比 | 31.40% | 30.79% |
| 偏慢超过0.5 m/s占比 | 0.94% | 0.41% |
| 原始速度奖励均值（最差趋近-2.5） | -1.7555 | -1.8410 |

现成 summary 的标量目标口径：RMSE 0.4492/0.4311 m/s，MAE 0.3276/0.3439 m/s；summary 的误差聚合为 episode 等权，不与上述时间步加权表混算。两者没有全面占优关系。

这批目标标量速度约1.1001–1.3982 m/s，每个episode内恒定，不能用于证明训练中每5秒换速或完整1.0–2.2 m/s包络的动态跟踪能力。时间步加权实际前向地速均值1.5121/1.5543 m/s；实际地速模长均值1.7028/1.7377 m/s，后者包含横向/垂向分量，不能当作前向速度误差。

沿真值路径切向的推进速度相对标量目标，偏差+0.2540/+0.3529 m/s，RMSE 0.4510/0.4940 m/s。BCRES在这个任务推进口径反而更偏快。实际RPM达到95%上限的时间占比6.64%/17.43%，说明“RPM不贡献材料速率裁剪”不等于推进器无高输出行为。

结论：当前能够推进，但精确速度跟踪不足，系统性偏快比偏慢更突出。BCRES改善了一部分大误差，却没有增加进入±0.25 m/s区间的比例；指数速度项按当前sigma=0.25，偏离较大时可能区分度较弱，但本数据不能单独证明其为根因。不据此静默改速度权重、std或训练协议。


## 12. PPO 对 BC 的同协议诊断计划（2026-11-18）

用户要求列表比较 PPO 相对 BC 的改善。旧 BC_ONLY 是 C3/seeds20–23、速度范围1.2–2.0；现有 PPO stage-boundary 是40 seeds，不能直接作为配对改善率。

- [x] 复用现有 evaluate_pc_ppo.py，在CPU运行一次C5/seeds20–59、float64、absolute接口、默认 nominal start/endpoint-stop 的 BC baseline；与已有两条 C5_066453504 评价对齐。使用 BCRES 的 model_initial.pt + action-mode=frozen-base，保留训练前 BC 权重及归一化统计，避免把训练后的归一化改变当作原始 BC。已有PPO只读，不新跑。输出单独 diagnostic_bc_initial_c5_20261118，不覆盖旧结果。
- [x] 启动前核对训练 JSON/env SHA、初始 BC identity，复用 evaluator 自带 checkpoint/schema/normalization 检查；结束运行 audit_goal182_artifacts.py --run-dir <output> --expected-episodes 40。验证seeds、stage、逐episode场景、真值路径RMSE口径与终止契约，统计3D/horizontal/vertical、速度、饱和、completion、success/timeout/safety。不是新正式矩阵或 formal winner。
- [x] 写回结果与局限；Git只涉及本文，本次不commit/push；如后续要求提交，只显式add本文、git diff --cached --check后提交，产物不入Git。

训练继续原协议；不占用已有GPU任务，不调参、不打开held-out。若环境/协议无法对齐，则标为不可比较，不拼接改善率。


### 12.1 配对结果

CPU baseline 已完成，输出 `.../20261118-goal184-64d8225/eval/diagnostic_bc_initial_c5_20261118/`；使用原有 evaluator，不修改远端源码。运行 `audit_goal182_artifacts.py --run-dir <该目录> --expected-episodes 40` 返回 artifact audit OK。

核验：BCRES初始与C5 checkpoint的24个frozen_bc state_dict字段逐元素完全一致；baseline使用初始checkpoint归一化状态，训练后的PPO使用各自checkpoint状态，因而比较覆盖整个训练栈的变化，不单独归因为网络权重。三组seeds、C5、env、nominal start、endpoint-stop、decision_stride、absolute接口、传感器、plant randomization均一致，40个seed逐episode curriculum字典完全一致。PPO为同一训练步数66,453,504（iteration338）的C5 checkpoint，非最新C6权重。单个训练seed，40个评价seed，诊断结果不作为formal winner。

| 指标 | 初始BC | BCFT | 相对BC变化 | BCRES | 相对BC变化 |
|---|---:|---:|---:|---:|---:|
| 路线3D RMSE，m | 2.4951 | 2.5528 | +2.31% | 1.7752 | -28.85% |
| 横向RMSE，m | 1.4321 | 1.3779 | -3.79% | 0.8150 | -43.09% |
| 纵向RMSE，m | 2.0432 | 2.1489 | +5.18% | 1.5770 | -22.81% |
| 速度RMSE，m/s | 0.8400 | 0.4492 | -46.52% | 0.4311 | -48.68% |
| 速度MAE，m/s | 0.6718 | 0.3276 | -51.24% | 0.3439 | -48.81% |
| 材料饱和率 | 43.40% | 31.91% | -11.49个百分点 | 56.40% | +13.01个百分点 |
| completion | 91.11% | 90.36% | -0.75个百分点 | 89.89% | -1.22个百分点 |
| success（telemetry） | 30.0% | 62.5% | +32.5个百分点 | 60.0% | +30.0个百分点 |
| timeout | 0 | 0 | 无变化 | 0 | 无变化 |
| safety终止 | 0 | 0 | 无变化 | 0 | 无变化 |
| 地速模长超过2.5m/s占比 | 20.96% | 4.56% | -16.40个百分点 | 3.73% | -17.23个百分点 |

所有数值来自同一 evaluator 的 summary.metrics；路线RMSE是episode等权MSE再开方，不能混用stage gate解析的episode RMSE均值。速度使用该summary的标量command减前向地速口径，不与第11节reward投影目标误差混算。BCFT有22/40个seed的路线3D RMSE降低，BCRES有33/40个seed降低；不据此声称统计显著。

结论：有改善，但改善内容不同。BCRES主要改善路线误差和速度，代价是更高执行器速率裁剪；BCFT改善速度与执行器负担，但当前C5 checkpoint的总体路线3D并未优于初始BC。两者completion略降，success提高只是辅助现象。此前“速度跟踪仍不足”与本表“相对BC明显改善”并不矛盾：基线速度误差更大。
