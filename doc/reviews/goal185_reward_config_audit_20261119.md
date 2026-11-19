# Goal185：reward 配置审计与逐项讨论记录

日期：2026-11-19。状态：`ALL_DECIDED / READY_TO_PLAN / NO_CODE_CHANGE`。
范围：PE2E BCRES/BCFT 的 Goal184 reward 配置、Goal185 复用配置，以及执行器
请求-可执行 gap 候选。本文只记录审计、结论和待讨论项；没有改 reward、配置、
训练代码或阈值，没有启动队列或训练。

## 1. 边界与结论摘要

- 当前配置约 70 个 reward 字段，C5 确定性 trace 上真正产生明显预算的约 10 项。
- 速度项 `cmd_speed` 占 BCRES 活跃惩罚预算约 49.5%，且 p95 已到上限 2.5。
- 执行器相关项合计不到 0.4%；`w_action_clip_attempt_pc` 与 `w_sat_pc` 的幅值
  部分因输入已被 clamp 而实际失效。
- 最值得先讨论的新增项是“请求-本步可执行 gap”连续代价；离线定标建议
  `lambda=5`（2026-11-19 用户定夺），只写进规划，尚未实现或启动训练。
- 所有配置项都要逐项讨论，包括本文标注“暂不建议动”的项。

## 2. 证据来源

- 配置：`configs/lowlevel/goal184/env_path_fixed_speed_formal_empirical.json`
- 实现：`src/auv_lowlevel/batch/pc.py`、`src/auv_lowlevel/batch/pc_env/core.py`
- 执行器限幅：`src/auv_lowlevel/batch/actions.py`
- 评审：
  - `doc/reviews/goal184_rudder_saturation_and_ppo_recipe_review_20261118.md`
  - `doc/reviews/goal184_strengths_tradeoffs_autoresearch_20261118.md`
- 只读 trace（服务器）：
  - `/data/yhx/underwater-robotics/runs/auv-local/goal184/20261118-goal184-64d8225/full_wave1/pe2e_bcft_seed86201/curriculum_evals/C5_066453504/default/trace.npz`
  - `/data/yhx/underwater-robotics/runs/auv-local/goal184/20261118-goal184-64d8225/full_wave1/pe2e_bcres_seed86201/curriculum_evals/C5_066453504/default/trace.npz`
  - 同目录 C4 trace；`eval/diagnostic_teacher_matched_c5_20261118` 与
    `eval/diagnostic_bc_initial_c5_20261118`
- 本审计没有新增 rollout；C5 是课程边界诊断，不是正式冻结矩阵。

## 3. gap 代价的审计方法

只对两片舵计算，单位统一为满量程 20°：

```text
a_req      = clamp(requested_action[:2], -1, 1) * 20°
a_prev     = actual_actuator[t-1][:2]
a_feasible = a_prev + clip(a_req - a_prev, -3°, +3°)
g          = mean_over_fins( ((a_req - a_feasible) / 20°)^2 )
r_gap      = -lambda * g
```

每个 episode 首步不参与差分；`valid` 只保留同一 episode 的相邻 step。
该式只罚“本步执行不了的部分”，不罚大而可执行的舵请求。

## 4. 审计结果

### 4.1 gap 分布与 lambda

| 指标 | BCFT C5 | BCRES C5 |
|---|---:|---:|
| 有效步数 | 61138 | 61430 |
| gap 均值 | 0.01427 | 0.03626 |
| gap p50 | 0.00000 | 0.00102 |
| gap p90 | 0.03230 | 0.11280 |
| gap p95 | 0.09319 | 0.19912 |
| gap p99 | 0.26674 | 0.42382 |
| 跟踪项均值（cross+vertical+speed） | 4.160 | 3.237 |
| cross p95 | 5.320 | 2.782 |

预注册定标规则：

- 平均 gap 代价占跟踪项均值的 5–10%；
- p95 gap 代价不超过 p95 cross 代价；
- 不扫参，不再按结果调 lambda。

得到的 lambda 区间：

| 口径 | BCFT C5 | BCRES C5 |
|---|---:|---:|
| 平均 5% | 14.58 | 4.46 |
| 平均 10% | 29.16 | 8.93 |
| p95 上限 | 57.09 | 13.97 |

2026-11-19 用户定夺首轮取 `lambda=5`：

- BCRES 平均 gap 代价 `0.181`，占跟踪项 `5.6%`，占全部活跃项 `4.9%`；
- BCRES p95 gap 代价 `0.996`，低于 cross p95 `2.782`；
- BCRES p99 gap 代价 `2.119`，也低于 cross p95；
- BCFT 平均 gap 代价 `0.071`，占跟踪项 `1.7%`。

该值落在预注册 5–10% 区间的下沿，比 `lambda=6` 更保守。

### 4.2 当前活跃 reward 预算

C5 确定性 trace，逐步加权，未乘 `pc_ppo_scale=0.01`：

| 分项 | BCFT 均值 | BCRES 均值 | BCRES 占比 |
|---|---:|---:|---:|
| `cmd_speed` | 1.75555 | 1.84110 | 49.5% |
| `path_cross` | 1.33612 | 0.67939 | 18.3% |
| `path_vertical` | 1.06850 | 0.71700 | 19.3% |
| `path_heading` | 0.24231 | 0.23127 | 6.2% |
| `path_pitch` | 0.36251 | 0.23448 | 6.3% |
| `path_energy` | 0.00419 | 0.00441 | 0.12% |
| `material_sat` | 0.00637 | 0.01128 | 0.30% |
| `action` | 0.00006 | 0.00009 | 0.002% |
| `action_smooth` | 0.00001 | 0.00001 | ~0 |
| `roll` | 0.00003 | 0.00029 | 0.008% |
| `roll_rate` | 0.00001 | 0.00001 | ~0 |
| `path_speed_smooth` | 0.00000 | 0.00000 | 0 |
| 合计（上述项） | 4.77566 | 3.71932 | 100% |

结论：执行器项合计 `<0.4%`；速度项接近一半预算；加大实际动作平滑不会
让策略看到裁剪前的大请求。

## 5. 全部 reward 配置项逐项结论

状态标记：

- `KEEP`：保持，不进入本轮变更。
- `CAND`：候选，需逐项讨论后再决定。
- `DEAD`：当前输入或量级导致实际无效，建议记录并在后续清理。
- `TBD`：需要用户定夺，不在本文自行改。

### 5.1 全局与模式

| 字段 | 当前值 | 结论 | 说明 |
|---|---:|---|---|
| `mode` | `pc_path_track` | KEEP | 任务模式，不是调参项。 |
| `pc_ppo_scale` | 0.01 | KEEP | 等比例缩放全部项；单独改会同时改变有效 LR/熵平衡。 |

### 5.2 速度与路径跟踪

| 字段 | 当前值 | 结论 | 说明 |
|---|---:|---|---|
| `w_cmd_speed` | 2.5 | CAND/TBD | 最大项；BCRES 平均偏差 `+0.395 m/s`，p95 已到上限。先讨论“改形状”而不是加权重。 |
| `cmd_speed_std_mps` | 0.25 | CAND/TBD | 高斯宽度；候选 0.4 或 Huber 长尾，但不能与 gap 项同批改。 |
| `w_path_cross` | 3.5 | KEEP | BCRES 平均 0.679，主要路线项。 |
| `path_cross_huber_delta_m` | 0.5 | KEEP | 超 0.5 m 后线性，梯度稳定。 |
| `w_path_vertical` | 1.5 | CAND/TBD | BCRES 平均 0.717，与 cross 接近；是否提高取决于 2 m 半径下的失败构成。 |
| `path_vertical_huber_delta_m` | 0.5 | KEEP | 与 cross 对称。 |
| `w_yaw` | 1.0 | KEEP | 路径航向项，平均 0.231，有效。 |
| `path_heading_std_rad` | 0.5 | KEEP | 同时用于路径航向与路径俯仰。 |
| `w_path_pitch` | 1.0 | KEEP | 路径俯仰项，平均 0.234，有效。 |
| `w_path_energy` | 0.01 | KEEP/DEAD-ish | 平均 0.0044，仅 0.12%；没有省推力目标就保持。 |
| `w_path_speed_smooth` | 0.025 | DEAD | 50 Hz 下地速逐步变化极小，实测为 0；建议后续清理或标 telemetry。 |
| `w_path_progress` | 0.0 | KEEP | 当前 completion 高；开启易奖励劫持。 |
| `w_prog` | 0.0 | KEEP | 同上，旧接口项。 |
| `w_path_3d` | 0.0 | KEEP | cross+vertical 已覆盖，避免重复计罚。 |
| `w_3d` | 0.0 | KEEP | 同上。 |
| `route_middle_gate` | 0.0 | KEEP | 中段不加权，保持。 |
| `w_u` | 0.0 | KEEP | `path_fixed_speed` 额外速度项，与 `cmd_speed` 重叠。 |

### 5.3 指令跟踪与 hold

| 字段 | 当前值 | 结论 | 说明 |
|---|---:|---|---|
| `w_cmd_yaw` | 0.0 | KEEP | 指令航向项关闭；路径航向已生效。 |
| `w_cmd_pitch` | 0.0 | KEEP | 同上。 |
| `cmd_heading_std_rad` | 0.5 | KEEP | 指令航向宽度；当前权重为 0。 |
| `hold_bonus` | false | KEEP | 未启用。 |
| `w_hold` | 5.0 | KEEP | 仅 `hold_bonus=true` 时生效。 |
| `hold_yaw_tolerance_rad` | 0.0873 | KEEP | 悬停容差，未启用。 |
| `hold_pitch_tolerance_rad` | 0.0873 | KEEP | 同上。 |
| `hold_speed_tolerance_mps` | 0.1 | KEEP | 同上。 |
| `hold_time_s` | 2.0 | KEEP | 同上。 |

### 5.4 执行器与姿态

| 字段 | 当前值 | 结论 | 说明 |
|---|---:|---|---|
| `w_action_pc` | 0.1 | KEEP | 作用在实际动作一阶差分，均值 0.00009；不建议加大。 |
| `w_action_smooth_pc` | 0.005 | KEEP/DEAD-ish | 实际动作二阶差分，均值 0.00001。 |
| `w_action_clip_attempt_pc` | 0.0 | DEAD/TBD | 输入已被 clamp，幅值越界恒为 0；要启用必须先改输入位置。 |
| `w_sat_pc` | 0.0 | DEAD/TBD | 幅值部分同样失效；速率部分只有非零时才生效。 |
| `w_material_sat_pc` | 0.02 | KEEP | 布尔事件，BCRES 均值 0.0113；保留作旧结果对照。 |
| `w_action_feasible_gap_pc`（候选新字段） | 未实现 | CAND/DECIDED | 已定夺 lambda=5，只算两片舵，按 mean；只记录，未实现。 |
| `w_phi` | 3.0 | KEEP | 横滚姿态惩罚，均值 0.0003；作为安全边界保留。 |
| `phi_dead_rad` | 0.0873 | KEEP | 横滚死区 5°。 |
| `phi_soft_rad` | 0.1745 | KEEP | 横滚软阈 10°。 |
| `phi_strong_rad` | 0.5236 | KEEP | 横滚强阈 30°。 |
| `w_p` | 0.02 | KEEP | 横滚角速率惩罚，均值 0.00001。 |
| `p_dead_rad_s` | 0.1745 | KEEP | 角速率死区 10°/s。 |
| `p_soft_rad_s` | 0.5236 | KEEP | 角速率软阈 30°/s。 |
| `w_action` | 0.0 | DEAD | 旧的非 PC 动作项。 |
| `w_sat` | 0.0 | DEAD | 旧的非 PC 饱和项。 |

### 5.5 终端、接近与势函数

| 字段 | 当前值 | 结论 | 说明 |
|---|---:|---|---|
| `goal_bonus` | 0.0 | TBD | 当前完全无终点奖励；completion 高，且终点不作评价判据，建议先保持。 |
| `timeout_penalty` | 0.0 | TBD | 同上。 |
| `out_of_bounds_penalty` | 0.0 | TBD | 安全另有 `safety_terminal_penalty=-50`。 |
| `terminal_progress_weight` | 0.0 | TBD | 不按终点距离做 potential；避免终点奖励劫持。 |
| `w_approach_vertical` | 0.0 | KEEP | 终点不是评价标准，保持关闭。 |
| `w_approach_pitch` | 0.0 | KEEP | 同上。 |
| `approach_enter_m` | 5.0 | KEEP | 仅接近项开启时生效。 |
| `approach_full_m` | 2.0 | KEEP | 同上。 |
| `w_vertical_potential` | 0.0 | KEEP | dense vertical 已够，保持关闭。 |
| `vertical_potential_huber_delta_m` | 0.25 | KEEP | 仅开启时生效。 |
| `potential_gamma` | 0.995 | KEEP | 势函数折扣，当前未用。 |
| `w_heave_potential` | 0.0 | KEEP | 未启用。 |
| `heave_potential_huber_delta_mps` | 0.05 | KEEP | 仅开启时生效。 |

### 5.6 路线速度与停滞

| 字段 | 当前值 | 结论 | 说明 |
|---|---:|---|---|
| `w_route_speed_pc` | 0.0 | KEEP/TBD | 可能防“贴线但走不动”；当前 completion 高，先不启用。 |
| `route_speed_floor_ratio` | 0.75 | KEEP | 仅 `w_route_speed_pc>0` 时生效。 |
| `w_route_ground_speed_stall` | 0.0 | KEEP/TBD | 停滞项未启用；与 `w_route_speed_pc` 功能重叠，不应同时开。 |
| `route_ground_speed_stall_threshold_mps` | 0.25 | KEEP | 仅停滞项开启时生效。 |
| `route_ground_speed_stall_gate_low_mps` | 0.2 | KEEP | 同上。 |
| `route_ground_speed_stall_gate_high_mps` | 0.4 | KEEP | 同上。 |

### 5.7 邻接 termination 项

| 字段 | 当前值 | 结论 | 说明 |
|---|---:|---|---|
| `safety_terminal_penalty` | -50 | KEEP | 硬安全项，不在 reward 调参范围内。 |

## 6. 候选变更（仅候选，未批准）

### CAND-1：请求-可执行 gap 代价

- 新字段建议名：`w_action_feasible_gap_pc`
- 首轮值：`5.0`（2026-11-19 用户定夺；只写进规划，不启动训练）
- 范围：只算两片舵，不含 RPM；按 `mean` 聚合。
- 不改：`w_material_sat_pc`、速度项、std、LR、课程、架构、残差权限。
- 需要代码管线：把 `applied.total`（速率限制后的可执行指令）传入 reward；
  当前 reward 只收到 `requested_normalized`（已 clamp）和实际执行器。
- 预注册验证：用现有 C5 trace 复算 `g` 分布；训练后看确定性 gap、材料饱和、
  路线 cross/vertical、速度偏差、completion；不以饱和率单项判优。

### CAND-2：速度项形状

- 2026-11-19 定夺：本轮保持不动；`std=0.4` 仅记录为后续候选。
- 只在 CAND-1 之后单独讨论；不与 gap 同批改。
- 现状：`cmd_speed` 占 BCRES 预算 49.5%，p95 到上限，平均偏差 `+0.395 m/s`。
- 2026-11-19 补充只读诊断（C5 trace，逐步加权）：

| 指标 | BCFT C5 | BCRES C5 |
|---|---:|---:|
| 速度偏差均值 | +0.400 m/s | +0.396 m/s |
| 速度 RMSE | 0.566 m/s | 0.508 m/s |
| `corr(e, current_body_x)` | 0.538 | 0.311 |
| 当前机体前向流均值 | +0.046 m/s | +0.069 m/s |
| RPM 平均占比 | 0.585 | 0.574 |
| RPM > 95% 时间占比 | 6.7% | 17.4% |
| RPM > 99% 时间占比 | 1.1% | 8.1% |
| 高斯惩罚 mean（std=0.25） | 0.702 | 0.736 |
| std=0.3 时 mean | 0.640 | 0.673 |
| std=0.4 时 mean | 0.532 | 0.552 |
| std=0.5 时 mean | 0.444 | 0.450 |

判断：
- BCRES 的速度偏差不是单纯“权重不够”；它在 17% 时间把 RPM 用到 95% 以上。
- 海流前向分量只能解释一部分偏差（相关 0.31），不是全部。
- 高斯 std=0.25 在 `e=0.4 m/s` 时已达 92% 上限，尾部梯度衰减；但直接把权重加大只会压一个已饱和的项。

候选方向：
- A（推荐先保持）：本轮不改速度项；先完成 gap 单因素实验。
- B（后续单独 arm）：`cmd_speed_std_mps=0.25 → 0.4`，权重暂保持 2.5。平均惩罚降到约 1.38，但 `e=0.4` 的梯度更强，相当于把压力从小误差移向大误差。
- C（后续单独 arm）：改用 Huber/线性长尾；需重新定标权重，不能直接套用 2.5。
- D（不建议）：只提高 `w_cmd_speed`；尾部已饱和，可能只增加整体压力而不改善大误差梯度。
- 评价口径：summary 的 `speed_error` 与 reward 的投影目标不完全一致；正式比较前应统一到 reward 口径，避免“指标偏差”被误判成策略偏差。

### CAND-3：垂直项权重

- 当前 `w_path_vertical=1.5`，BCRES 平均 0.717，与 cross 0.679 接近。
- 是否提高应看全程轨迹误差，而不是末态成功/失败；终点情况不作为好坏判据。
- 2026-11-19 修正：此前用“末态 2 m 失败构成”的论证已撤回。`-0.122 m`
  是带符号的末态横向误差均值，负号只表示偏向一侧，均值会抵消，不能当作
  横向误差大小；末态也不应作为权重依据。
- 2026-11-19 全程轨迹只读诊断（C4/C5，逐步加权）：

| 指标 | BCFT C5 | BCRES C5 | BCFT C4 | BCRES C4 |
|---|---:|---:|---:|---:|
| cross 带符号均值 | +0.460 | -0.058 | +0.550 | +0.053 |
| cross 平均绝对误差 | 0.973 | 0.577 | 1.009 | 0.538 |
| cross RMS | 1.400 | 0.848 | 1.539 | 0.767 |
| cross p95 | 3.289 | 1.839 | 3.754 | 1.646 |
| vertical 带符号均值 | -0.003 | +0.199 | +0.350 | +0.800 |
| vertical 平均绝对误差 | 1.649 | 1.168 | 1.214 | 1.118 |
| vertical RMS | 2.179 | 1.690 | 1.554 | 1.504 |
| vertical p95 | 4.627 | 4.088 | 3.095 | 3.213 |
| `|vertical| > |cross|` 时间占比 | 63.3% | 68.4% | 57.3% | 68.7% |
| `path_error_3d > 2 m` 时间占比 | 43.0% | 23.7% | 36.0% | 22.3% |

判断：
- 垂向误差在全程都明显大于横向：BCRES C5 平均绝对误差 1.168 vs 0.577，
  p95 4.088 vs 1.839，68.4% 时间垂向更大。这不是末态统计造成的假象。
- 因此 `w_path_vertical=1.5` 相对 `w_path_cross=3.5` 可能确实偏低；但也可能是
  垂向控制权限/制导能力不足，不能只凭误差大小就加权重。
- 候选值：`w_path_vertical` 1.5 → 2.0（温和）或 2.5（更强）。按当前均值，
  2.0 会让垂向项从 0.717 升到约 0.956，占活跃项约 24%；2.5 会升到约 1.195，
  占约 29%。
- 仍应作为 gap 之后的单独 arm；不与 gap、速度项同批改。
- 若加权重后垂向误差不降、RPM/俯仰先饱和，则说明瓶颈在控制权限，应回到
  制导/控制而不是继续加 reward。

#### D-3 定夺与影响范围（2026-11-19）

用户定夺：`w_path_vertical` 1.5 → **2.5**，不单独跑 arm，并入下一轮主配方。

预算影响（C5 确定性 trace，未乘 `pc_ppo_scale`）：

| 分项 | 旧均值 | 旧占比 | 新均值 | 新占比 |
|---|---:|---:|---:|---:|
| `cmd_speed` | 1.841 | 49.5% | 1.841 | 43.9% |
| `path_cross` | 0.679 | 18.3% | 0.679 | 16.2% |
| `path_vertical` | 0.717 | 19.3% | 1.195 | 28.5% |
| `path_heading` | 0.231 | 6.2% | 0.231 | 5.5% |
| `path_pitch` | 0.234 | 6.3% | 0.234 | 5.6% |
| 其他/执行器 | 0.017 | 0.4% | 0.017 | 0.4% |
| 合计 | 3.719 | 100% | 4.197 | 100% |

若同时加入 gap λ=5（BCRES 均值约 0.181），下一轮主配方的预算约为：

| 分项 | 均值 | 占比 |
|---|---:|---:|
| `cmd_speed` | 1.841 | 42.1% |
| `path_vertical`（2.5） | 1.195 | 27.3% |
| `path_cross` | 0.679 | 15.5% |
| `path_heading` | 0.231 | 5.3% |
| `path_pitch` | 0.234 | 5.4% |
| `action_feasible_gap`（λ=5） | 0.181 | 4.1% |
| 其他/执行器 | 0.017 | 0.4% |
| 合计 | 4.378 | 100% |

影响范围：

- **BC 数据与 BC 模型：不受影响。** BC 由 teacher demo 做行为克隆，reward 不参与
  BC 损失；改 `w_path_vertical` 不会改变已有 BC 数据或 BC checkpoint。
- **teacher / ALOS / 洋流补偿：不受影响。** 它们不使用 PPO reward。
- **PPO 训练：受影响。** 垂向项从 19.3% 升到 28.5%；总奖励尺度上升约 12.9%。
  这会改变策略在垂向、速度、横向之间的取舍。
- **课程晋级：阈值不变，但轨迹可能改变。** 阈值文件、stage mix、soft/hard
  budget 都不动；reward 变了，实际晋级时机会随之变化。
- **评价指标：定义不变。** cross/vertical/3D 误差仍按原口径计算；但新 reward
  训练出的策略不能当作旧协议的同配方结果。
- **配置 SHA / manifest：会变。** 必须使用新 run root，不能静默续跑旧 run；
  旧 BCRES 只能作为“旧配方对照”，不能宣称同配方可比。
- **归因强度：下降。** gap λ=5 与 vertical 2.5 同批生效，无法单独归因；
  这是用户明确选择的效率优先方案，需在结果文档中写清。

配置落地方式：

- **不要直接改** `configs/lowlevel/goal184/env_path_fixed_speed_formal_empirical.json`
  这份共享文件；复制成新的 reward 版本（例如
  `env_path_fixed_speed_formal_empirical_reward_v2.json`），再让下一轮训练 JSON
  指向新文件。
- 这样旧 BC checkpoint、旧 run 的 `environment_config_sha256` 和 manifest 保持
  不变；新配方有独立 SHA，不会把旧结果混成同一协议。
- 当前 PPO 加载 BC checkpoint 只校验 observation schema/dim，不校验 env SHA；
  但 manifest 会记录 env SHA，仍应保持旧文件不动以保证 provenance 清晰。
- BC 行为克隆损失不使用 reward；改 reward 不需要重采 BC 或重训 BC。只有未来
  用新策略重新生成 teacher demo 时，才需要重新采集。

### CAND-4：终端奖励

- 当前 `goal_bonus/timeout_penalty/out_of_bounds_penalty/terminal_progress_weight`
  全为 0；`goal_plane_miss_penalty=0`、`goal_plane_miss_distance_weight=0`。
- 安全终止另有 `safety_terminal_penalty=-50`，覆盖 OUT_OF_BOUNDS、NAN、
  INVALID_STATE、DEPTH_LIMIT、SAFETY_ROLL_ANGLE/RATE。
- 2026-11-19 C5 trace 终止构成（40 episodes）：

| terminal_kind | BCFT C5 | BCRES C5 |
|---|---:|---:|
| GOAL（1） | 25 | 24 |
| GOAL_PLANE_MISS（9） | 15 | 16 |
| TIME_LIMIT（2） | 0 | 0 |

判断：
- C5 没有 timeout；主要非成功终止是 goal-plane miss，占 37.5%–40%。
- 当前 miss 没有终端惩罚，策略不会因为“越过终点平面但偏出捕获半径”而收到
  额外信号；但这是训练信号问题，不等于应把终点当评价判据。
- 加 `goal_bonus` 风险最高：容易奖励劫持、鼓励抢终点；且与“终点不作好坏判据”
  的立场冲突。
- 加 `goal_plane_miss_penalty` 是更针对性的候选，但需要定标；在
  `pc_ppo_scale=0.01` 下，-5 只相当于 -0.05，太小；-50 又会和安全项同级。
- `timeout_penalty` 当前为 0 可以接受，因为 C5 没有 timeout；若后续 timeout
  上升，再单独加小惩罚。
- `out_of_bounds_penalty=0` 不建议动：OUT_OF_BOUNDS 已在安全终止集合内，
  已有 -50。

候选：
- A（推荐）：下一轮主配方保持终端奖励全 0，不改。
- B（后续单独讨论）：只加小的 `goal_plane_miss_penalty`，先做尺度审计。
- C（不建议）：加 `goal_bonus`。
- D（条件触发）：只有 timeout 明显上升时，才加 `timeout_penalty`。

### CAND-5：`w_action_clip_attempt_pc` / `w_sat_pc` 的输入与去留

代码核对：

- `w_action_clip_attempt_pc` 在 reward 计算中没有直接使用；它只在
  `config.py` 解析时作为 `w_sat_pc` 的别名/覆盖值：
  `clip_attempt_weight = w_action_clip_attempt_pc if present else w_sat_pc`。
  两个键同时存在时，`w_action_clip_attempt_pc` 会覆盖 `w_sat_pc`。
- `w_sat_pc` 在 `pc.py` 中计算：
  - `magnitude_excess = mean(max(|requested|-1, 0))`；但 `requested` 是
    `applied.normalized`，已经 clamp 到 [-1,1]，所以这一项恒为 0；
  - `rate_excess = mean(max(|requested_delta|/action_rate_norm - 1, 0))`；
    这一项可以非零，但当前 `w_sat_pc=0`。
- 因此当前两个键都为 0 时没有行为；`w_action_clip_attempt_pc` 是一个
  “看起来独立、实际是别名”的配置陷阱。

候选：

- A（推荐）：下一轮主配方保持两个键都为 0，不改行为。
- B（推荐，仅配置清理）：在新 v2 env 配置中只保留 canonical 的 `w_sat_pc=0.0`，
  删除或显式置零 `w_action_clip_attempt_pc`；行为不变，避免别名覆盖。
- C（不建议现在做）：把 clip-attempt 改接 clamp 前的原始动作并启用；这会与
  gap 项功能重叠，且需要新的尺度审计。
- D（不建议现在做）：只启用 `w_sat_pc` 的 rate-excess 部分；同样与 gap 重叠。

2026-11-19 定夺：**A+B**。保持权重 0；新 v2 env 配置显式写
`w_sat_pc: 0.0`，删除或置零 `w_action_clip_attempt_pc`；不改代码行为。

### CAND-6：路线速度与停滞项

- 当前 `w_route_speed_pc=0`、`w_route_ground_speed_stall=0`，两项都未启用。
- `w_route_speed_pc`：当 route ground speed 低于
  `route_speed_floor_ratio × reference_speed`（0.75×）时线性扣分；是“最低前进速度”项。
- `w_route_ground_speed_stall`：只在 reference speed 较低（0.2–0.4 m/s 的
  smooth gate）时生效，低于 0.25 m/s 的停滞按平方扣分；是“低速停滞”项。
- 两项功能重叠，不应同时启用。
- 当前证据：BCRES 速度偏差为 **正**（平均偏快 `+0.40 m/s`），不是走不动；
  启用任一项都会进一步推高速度，可能加重超速和 RPM 饱和。
- 因此下一轮主配方保持两项为 0。
- 若未来出现“贴线但走不动”或 timeout 上升，再单独选择一个启用：
  - 优先 `w_route_speed_pc`（连续最低速度，语义简单）；
  - 或 `w_route_ground_speed_stall`（只针对低速指令段，范围更窄）；
  - 不同时开，不做扫参。
- `route_speed_floor_ratio=0.75`、三个 stall 阈值目前只影响未启用项，保持原值。

候选：

- A（推荐）：保持两项为 0，不启用。
- B（条件触发）：未来出现停滞时，单独启用 `w_route_speed_pc`。
- C（条件触发）：未来只在低速段停滞时，单独启用 `w_route_ground_speed_stall`。
- D（不建议）：从配置中删除；保留 dormant 项成本很低，未来可能用到。

### CAND-7：dormant 配置项与命名清理

关键发现：**很多字段的代码默认值不是 0，删掉配置行会悄悄改行为。**

| 字段 | 当前值 | 代码默认值 | 删掉后的后果 |
|---|---:|---:|---|
| `w_cmd_yaw` | 0.0 | 1.5 | 指令航向惩罚被打开 |
| `w_cmd_pitch` | 0.0 | 0.5 | 指令俯仰惩罚被打开 |
| `w_cmd_speed` | 2.5 | 1.0 | 速度惩罚被削弱 |
| `w_action_pc` | 0.1 | 0.5 | 动作惩罚增强 5 倍 |
| `w_sat_pc` | 0.0 | 0.1 | 饱和惩罚被打开 |
| `w_phi` | 3.0 | 0.05 | 横滚惩罚削弱 60 倍 |
| `w_path_cross` | 3.5 | 0.0 | 横向路径惩罚被关闭 |
| `w_path_vertical` | 1.5→2.5 | 0.0 | 垂向路径惩罚被关闭 |
| `w_path_pitch` | 1.0 | 0.0 | 路径俯仰惩罚被关闭 |

另外 `w_prog`、`w_3d`、`w_yaw`、`w_u`、`w_action`、`w_sat`、
`terminal_progress_weight`、`goal_bonus`、`timeout_penalty`、
`out_of_bounds_penalty` 是 dataclass 必填字段，删掉会直接报错或依赖解析路径。

结论：

- 不建议为了“干净”删除 dormant 键；保留显式值最安全。
- v2 配置应复制全部字段，只改本轮已定夺的三处：
  `w_path_vertical=2.5`、新增 gap 项、`w_sat_pc=0.0`（并按 D-5 处理别名）。
- 可以用分组/注释标明 `active`、`dormant`、`deprecated alias`，但不要依赖代码默认值。
- 唯一建议删除/置零的别名是 `w_action_clip_attempt_pc`，因为它与
  `w_sat_pc` 共用开关，且已在 D-5 定夺中处理。

候选：

- A（推荐）：v2 配置保留全部字段显式值，只加分组/注释，不删除。
- B：只删除“默认值等于当前值”的字段；收益很小，仍有漏判风险，不推荐。
- C：做 schema 级重命名/清理；属于代码重构，另开任务，不与本轮实验混。
- D：做“最小配置”，只写 active 字段；不建议，会触发大量非零默认值。

## 6.9 最终决策汇总（2026-11-19）

| 编号 | 决策 | 内容 |
|---|---|---|
| D-1 | 采用 | 新增请求-可执行 gap 项，`lambda=5`，只算两片舵，按 mean；只写进规划。 |
| D-2 | 保持 | 速度项本轮不动；`std=0.4` 仅记录为后续候选。 |
| D-3 | 采用 | `w_path_vertical` 1.5 → 2.5；与 gap 同批进入下一轮主配方。 |
| D-4 | 保持 | 终端奖励全部为 0，不引入 goal bonus / timeout / miss 惩罚。 |
| D-5 | 采用 | `w_sat_pc=0.0` 显式保留；`w_action_clip_attempt_pc` 别名删除或置零；不启用。 |
| D-6 | 保持 | 近零平滑/能量项全部保持原值，标记为弱项/近似 telemetry。 |
| D-7 | 保持 | `w_route_speed_pc` / `w_route_ground_speed_stall` 保持 0。 |
| D-8 | 采用 | v2 配置保留全部字段显式值，只加分组/注释；不删除 dormant 键。 |

下一轮主配方的 reward 变更只有三项：

1. 新增 `w_action_feasible_gap_pc=5.0`；
2. `w_path_vertical=2.5`；
3. `w_sat_pc=0.0` 明确保留，`w_action_clip_attempt_pc` 置零（别名）。

其他 reward 字段、速度项、终端项、std、LR、课程、架构、残差权限均不变。
归因说明：gap 与 vertical 同批生效，不能做单因素归因；旧 BCRES 作为对照。

## 7. 逐项讨论队列

- [x] D-1：CAND-1 gap 代价与 lambda 是否批准为预注册候选。
  - 2026-11-19 定夺：采用 `lambda=5`；只写进规划，不实现、不启动训练。
- [x] D-2：速度项形状（高斯 vs Huber 长尾；std/权重如何成对定标）。
  - 2026-11-19 定夺：本轮保持不动；`std=0.4` 仅记录为后续候选，不与 gap 同批。
- [x] D-3：cross/vertical 权重与 2 m 半径失败构成。
  - 2026-11-19 定夺：`w_path_vertical` 1.5 → 2.5；不单独开 arm，并入下一轮主配方。
  - 归因说明：与 gap λ=5 同批生效，不再宣称单因素归因；旧 BCRES 作为对照。
- [x] D-4：终端奖励是否继续保持全 0。
  - 2026-11-19 定夺：A，继续保持 `goal_bonus/timeout_penalty/out_of_bounds_penalty/terminal_progress_weight` 全 0；不引入终点奖励。
- [x] D-5：`w_action_clip_attempt_pc` / `w_sat_pc` 的输入位置修复或删除。
  - 2026-11-19 定夺：A+B。保持权重 0；新 v2 env 配置显式写 `w_sat_pc: 0.0`，删除或置零别名键；不改代码行为。
- [x] D-6：`w_path_speed_smooth`、`w_action_smooth_pc`、`w_path_energy` 等
  “量级近零”项是保留、删除还是转 telemetry。
  - 2026-11-19 定夺：A，全部保持原值；在文档和新配置说明中标记为弱项/近似 telemetry。
- [x] D-7：路线停滞项 `w_route_speed_pc` / `w_route_ground_speed_stall` 是否
  需要在新一轮课程中启用。
  - 2026-11-19 定夺：A，两项保持 0，不启用；未来出现停滞再单独选一个。
- [x] D-8：所有 dormant 项在下一版配置中的去留与命名清理。
  - 2026-11-19 定夺：A，v2 配置保留全部字段显式值，只加分组/注释；不删除 dormant 键。

## 8. 待用户定夺

1. CAND-1 已定夺：`lambda=5`，只写进规划；尚未批准实现或运行。
2. gap 项实现时是否保留 `w_material_sat_pc=0.02`（本文建议保留）。
3. 是否允许把 `applied.total` 作为新 reward 输入；这会改 reward 接口，但不改
   动作管线语义。
4. 讨论顺序是否按 D-1 → D-8；还是先讨论速度项 D-2。

## 9. 复现与验证

只读复算口径：

```bash
# 服务器
/data/yhx/underwater-robotics/envs/venvs/isaaclab/bin/python - <<'PY'
# 读取 trace.npz，按 episode_id/step 排序；
# 只用相邻 step；首步排除；
# 计算 gap、cross/vertical Huber、speed penalty 与分项均值。
PY
```

验证要求：

- 不新增 rollout、不启动训练；
- 复算结果与本文表一致才进入配置改动；
- 配置改动必须另开 goal/amendment，写 Research Contract、单因素变更、
  运行命令、评价矩阵与审计器；
- 不用旧 run root 静默续跑改 reward 的模型。

## 10. Git

本文只新增文档。提交时：

```bash
git add doc/reviews/goal185_reward_config_audit_20261119.md
git diff --cached --check
git commit -m "docs(goal185): audit reward config and record discussion queue"
```

是否 commit 由用户明确要求后执行；本文不自动 commit/push。
