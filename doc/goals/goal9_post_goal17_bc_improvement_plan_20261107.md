# Goal9 后续规划：Goal17 之后的 BC 提升路线

日期：2026-11-07

状态：**P1/P2/P2.1 已完成；P2A/P2B 已登记但未实现；不改变 Goal17 的负结果和已冻结门禁。**

关联文档：

- `doc/goals/goal17_v0d_current_crab_factorial_plan_20261107.md`
- `doc/goals/goal17_v0d_current_crab_factorial_report_20261107.md`
- `doc/goals/goal18_v0d_sign_magnitude_bc_report_20261107.md`
- `doc/goals/goal19_v0d_weighted_bc_targeted_dagger_report_20261107.md`
- `doc/goals/goal9_execution_state.md`

## 1. 背景与约束

Goal17 的 2×2 current/crab BC 因子实验已经收口：

- `A10` default RMSE ratio `1.2019×`，FAIL；
- `A11` default RMSE ratio `1.0139×`，FAIL；
- `A11` success 有 `+0.05` 的正向信号，current 消融也证明 current 分支被使用；
- 但没有形成相对 `A00` 的 3D RMSE 净收益，因此没有进入残差 PPO。

后续提升不能把 Goal17 的门禁或结论改写成成功，也不能用 held-out 反选方案。所有新
方向都必须满足：

- 主指标仍是 3D RMSE；success、timeout、safety、strict125 和 path-zero 作护栏；
- dev 使用 `20–39`；held-out `100–149` 只在配方冻结后评估一次；
- 不混用历史 `0–19`、dev `20–39` 和 held-out `100–149`；
- 新功能默认关闭，不破坏旧 checkpoint、旧配置和旧评估结果；
- 每个方向先写预注册门禁，再训练和评估；
- 不自动 commit/push，不使用 `git add -A`。

## 2. 候选方向

| 编号 | 方向 | 主要目的 | 预期收益 | 成本/风险 | 初步优先级 |
|---|---|---|---|---|---|
| P1 | 统一复评已有 BC | 找出当前最强、最稳定的 BC 底座 | 高信息量，避免跨 seed 误比较 | 低 | **最高** |
| P2 | 审计 3D RMSE 构成 | 确认横向、垂直、纵向/进度和投影口径的贡献，并建立终点诊断指标 | 防止优化错指标 | 低 | **最高** |
| P2B | 终点切平面未捕获终止 | 失败即停止并给终端惩罚，禁止继续飞行污染指标 | 与任务语义一致，降低 RMSE 方差 | 中 | 高，依赖 P2.1 |
| P3 | Default + strict125 双场景 DAgger | 修复两个工作点的失败分布 | 有望同时改善 default 和 strict125 | 中 | 高 |
| P4 | Action chunking / 降低控制频率 | 改善动作时间一致性和 rate-limit | 可能减少抖动并改善饱和动作 | 中高 | 中高 |
| P5 | 残差 PPO | 在强 BC 上学习小残差修正 | 可能修正 BC 的系统偏差 | 高 | 低，最后做 |
| P6 | 在更强 BC 上复测 current/crab | 验证 Goal17 的负结果是否受弱底座限制 | 可能让 current 分支产生价值 | 中 | 中，依赖 P1 |

## 3. P1：统一复评已有 BC

### 目的

现有 BC 结果来自不同轮次和不同 seed 集，不能直接横比。例如 Goal19 的 `6.6236 m`
来自 test seeds `0–19`，Goal17 的 `8.1176 m` 来自 dev seeds `20–39`。必须先放到同一
张表上。

### 候选 checkpoint

结果读取前冻结以下候选，不根据复评结果增删：

| ID | checkpoint | 来源 |
|---|---|---|
| `g19_dagger` | `artifacts/lowlevel/pc_v0d_r2b_weighted_bc_dagger_20261107/bc_satw_dagger2/pc_bc_actor.pt` | Goal19 targeted-DAgger |
| `g19_satw` | `artifacts/lowlevel/pc_v0d_r2b_saturation_bc_20261107/bc_satw/pc_bc_actor.pt` | Goal19 baseline weighted BC |
| `g18_s0` | `artifacts/lowlevel/pc_v0d_sign_magnitude_bc_20261107/s0_continuous/pc_bc_actor.pt` | Goal18 连续动作对照 |
| `g17_a00` | `artifacts/lowlevel/pc_v0d_current_crab_factorial_20261107/bc_A00/pc_bc_actor.pt` | Goal17 基线 |
| `g17_a10` | `artifacts/lowlevel/pc_v0d_current_crab_factorial_20261107/bc_A10/pc_bc_actor.pt` | current only |
| `g17_a01` | `artifacts/lowlevel/pc_v0d_current_crab_factorial_20261107/bc_A01/pc_bc_actor.pt` | crab only |
| `g17_a11` | `artifacts/lowlevel/pc_v0d_current_crab_factorial_20261107/bc_A11/pc_bc_actor.pt` | current + crab |

所有候选统一使用 Goal17 的共同评估环境，避免把训练配置差异混入复评：

- default/path-zero：`configs/lowlevel/odarrl_control_pc_c3_random_realistic_v0d_teacher_r2b_current.json`；
- strict125：`configs/lowlevel/odarrl_control_pc_c3_random_realistic_v0d_teacher_r2b_current_strict125.json`；
- 统一 `path_dim=8`、`device=cpu`、`dtype=float64`。

历史 `0–19` 结果只作来源说明，不进入这张统一 dev 表。

### 评估口径

- 固定 dev seeds `20–39`；
- default、strict125、path-zero；
- CPU、float64、`path_dim=8`；
- 记录 3D RMSE、水平/垂直误差、success、timeout、safety、saturation；
- 每个 checkpoint 记录 SHA256 和训练数据来源。

### 产出

- 一张统一 BC 排行榜；
- 一个冻结的“当前最强底座”候选；
- 明确指出不同 seed 集导致的旧结论偏差。

### 停止条件

- 不训练新模型；
- 不在 P1 内改变网络、数据、奖励或评估门禁；
- 不把 dev 结果当作 held-out 结论。

## 4. P2：审计 3D RMSE 构成

### 目的

当前 `A11` 的 default 指标约为：

- 3D RMSE `8.1176 m`；
- 水平误差 `5.3974 m`；
- 垂直误差 `1.0596 m`。

三者不是简单的平方和关系，说明 3D 误差还包含纵向/进度或投影口径相关成分。必须先
确认“3D RMSE 到底在惩罚什么”，否则后续训练可能在优化错误目标。

### 审计内容

- 分解横向、垂直、纵向/progress 和沿路径滞后误差；
- 比较全局最近投影与带 segment hint 的投影；
- 检查 success 与 timeout 回合的误差构成是否不同；
- 检查 strict125 的 timeout 是否主要由终点附近误差、速度或转弯造成；
- 对 `A11`、`A00` 和 Goal19 最强 BC 各抽样若干 seed 做 trace 对比。

P2 结果读取前固定审计对象与 seed：

- `g19_dagger`、`g17_a00`、`g17_a11`；
- dev seeds `20–39` 全部 20 个；
- default 环境；
- 只做诊断，不据此改主指标或选择 checkpoint。

### 产出

- 误差分解表和典型 trace；
- 判断后续应优先优化横向跟踪、纵向进度、速度控制还是终点行为；
- 如果需要改评估口径，必须先写成新的诊断指标，不直接覆盖历史主指标。

### P2 新增诊断指标（不替换历史主指标）

历史主指标 `path_three_d_rmse_m` 的计算、投影方式和汇总范围全部保持不变，继续作为
与 Goal17/18/19 结果比较的一号指标。新增以下四项诊断指标，与主指标并列输出；它们
用于解释误差来源和验证终点修复，不得回填、覆盖或重新命名历史结果。

| 指标 | 统计范围 | 用途 |
|---|---|---|
| `success_only_3d_rmse_m` | 仅 `success=True` 的 episode，沿用主指标的 3D 路径距离和时间步聚合口径 | 衡量真正完成任务时的跟踪质量，排除 timeout 过冲污染 |
| `pre_goal_3d_rmse_m` | 从 episode 开始到第一次满足终点捕获条件的时刻之前；不含终点帧及其后时间步 | 衡量到点前的正常路径跟踪质量 |
| `post_goal_overshoot_m` | 从第一次越过终点切平面开始，到 episode 结束期间的 `max(signed_goal_tangent_offset_m)`，只取正值；只统计实际越界回合 | 衡量越过终点后继续前飞的幅度 |
| `terminal_longitudinal_rmse_m` | 从第一次越过终点切平面到 episode 结束的沿末端切向误差 RMSE | 衡量终点后的纵向回收/制动能力 |

统计与报告约束：

- 四项指标都固定使用 dev seeds `20–39`，并与 P1 主指标逐 seed 对齐；
- 每项必须同时报告 episode 数、均值、标准差和计算分母，避免空集合产生伪结果；
- `post_goal_overshoot_m` 和 `terminal_longitudinal_rmse_m` 只对越界回合统计；未越界
  回合不按 0 混入均值，单独报告 `N/A` 和数量；
- `success_only_3d_rmse_m` 只作跟踪质量诊断，不能通过筛掉 timeout 回合来宣称主指标
  改善；主指标仍是完整的 `path_three_d_rmse_m`；
- 四项指标先由统一后处理脚本从已有 trace 和 summary 重算，不重新 rollout；
- 新指标在报告中标记为 `diagnostic_only=true`，不参与 checkpoint 排序；后续只能作为
  预注册门禁的解释项或护栏项。

P2.1 的历史诊断仍按 `legacy` 主指标解释；新的 `endpoint_stop` 主排序与门禁见 P2B
和 P5。终点相关的 `post_goal_overshoot_m` 和 `terminal_longitudinal_rmse_m` 用于
解释旧轨迹中的过冲，不得混入 `endpoint_stop` 的主排序。

### P2A：终点上下文观测块（新增，尚未实现）

现有 8D 路径特征在终点饱和后无法区分“刚到终点”和“已越过终点”。P2A 必须把终点
相对几何显式提供给 actor，并编码进 `z_t`；不能只给 critic，也不能直接扩大 actor
MLP 的顶层输入宽度。

必选特征：

| 特征 | 定义 | 用途 |
|---|---|---|
| `goal_distance_3d_m` | `||p_nav - p_goal||_2` | 原始距离，用于奖励、终止和诊断；actor 输入使用下方的单调接近度 |
| `signed_goal_tangent_offset_m` | `dot(p_nav - p_goal, end_tangent)`，越过终点为正 | 判断是否已经越过终点切平面 |
| `goal_bearing_heading_error_rad` | 当前位置到终点的水平方位角减去当前 heading，wrapped 到 `[-pi, pi]`；编码进 actor 前乘终点平滑门控 | 只在临近终点时判断水平方向是否朝向终点 |
| `goal_bearing_pitch_error_rad` | 当前位置到终点的俯仰方位角减去当前 pitch，wrapped；编码进 actor 前乘终点平滑门控 | 只在临近终点时判断垂向是否朝向终点 |

可选特征：

| 特征 | 定义 | 用途 |
|---|---|---|
| `endpoint_active_flag` | 当 `goal_distance_3d_m <= terminal_capture_distance_m` 时为 1 | 作为终点附近奖励和分支的门控 |
| `terminal_capture_ratio` | `goal_distance_3d_m / goal_radius_m`，在预注册范围内裁剪 | 提供归一化的捕获接近程度 |
| `goal_radius_excess_m` | `max(goal_distance_3d_m - goal_radius_m, 0)` | 直接度量离捕获球的径向超出量 |

实现约束：

- 所有特征都从可部署的 `p_nav`、参考路径终点和当前姿态计算，不使用真实海流、仿真
  真值或绝对位置特权信息；
- 终点块单独编码后写入 `z_t`，actor 顶层宽度保持不变；
- 旧 8D path schema 和旧 checkpoint 继续可加载，新块默认关闭并带 schema/version；
- 编码前做固定裁剪和尺度记录，禁止根据评估集调整归一化；
- actor 和 critic 都可以使用该块，但 actor 的版本必须全部可部署；
- 终点距离、方位误差既是观测，也是 P5 终端接近/对准奖励的来源；但它们首先是观测，
  不能只靠奖励替代。

#### 终点邻近门控

`goal_distance_3d_m` 原始值应全程计算，用于奖励、终止和诊断；但 actor 不应在长距离
路径跟踪阶段直接看到大尺度米数，两个 bearing error 也不应与 `w_yaw`、`w_3d` 同权。
初版采用“全程计算、临近终点才增强”的平滑门控，而不是在阈值处硬切换：

```text
s = clip((d_enter - goal_distance_3d_m) / (d_enter - d_full), 0, 1)
g_endpoint = s * s * (3 - 2 * s)
```

终点距离也不能直接把原始米数乘 `g_endpoint`：这样在远处会变成 0，临近终点时又会
先增大后减小，不是单调信号。actor 使用单独的单调接近度：

```text
d_ratio = clip(goal_distance_3d_m / d_enter, 0, 1)
goal_distance_proximity = g_endpoint * (1 - d_ratio)
```

它在远处为 0，在终点处为 1，中间单调上升；原始 `goal_distance_3d_m` 仍保留给奖励、
终止和诊断。

- 预注册初值：`d_enter = 20 m`，`d_full = 5 m`；训练前冻结并写入配置，不能根据
  dev 或 held-out 结果调整；
- `goal_bearing_heading_error_rad`、`goal_bearing_pitch_error_rad` 作为 actor 输入时
  乘以 `g_endpoint`；原始值只用于奖励、终止和诊断，不直接进入 actor 顶层；
- `goal_distance_3d_m` 不直接进入 actor；actor 输入使用 `goal_distance_proximity`，
  远处为 0，临近终点平滑增强到 1，避免早期与路径跟踪特征抢权重；
- `signed_goal_tangent_offset_m` 的原始值用于终止判断，送入 actor 的版本同样乘
  `g_endpoint`，避免远距离时产生无意义的过冲信号；
- 门控必须连续可导，避免在 `20 m` 附近突然改变策略行为；若后续需要调整，只允许在
  预注册的 screen 中作为一个独立变量测试。

### P2B：终点切平面未捕获终止（新增，尚未实现）

P2.1 证明 timeout 回合会在越过终点切平面后继续运行约 `1100–1224` 步，最大过冲
`20–31 m`。这不是“到点成功后继续仿真”，而是没有进入终点捕获球、又没有被判失败，
于是跑到时间上限。新协议要在第一次越过终点切平面且仍未捕获终点时立即终止。

#### 终止定义

新增一个默认关闭的终止模式 `endpoint_stop`，旧模式保持为 `legacy`。同一步的优先级：

1. `goal_distance_m <= goal_radius_m`：`GOAL`，成功；
2. 否则若 `signed_goal_tangent_offset_m >= 0`：新的 `GOAL_PLANE_MISS`，立即终止；
3. 否则继续执行现有的 timeout、safety、out-of-bounds 等终止逻辑。

实现时新增独立 terminal kind，不复用 `TIME_LIMIT`，避免把“切平面漏检”和“真的跑满
时间”混在一起。配置开关建议为：

```text
pc_goal_plane_miss_termination_enabled = false  # legacy 默认
```

启用后不保留越过切平面后的长轨迹；终止步之后不再推进。旧 checkpoint、旧配置和旧
`summary.json` 必须继续可加载，旧结果继续按 `legacy` 解释。

#### 训练与评估

- PPO 训练环境：启用 `endpoint_stop`；
- 新的 BC/DAgger 数据采集：默认使用 `endpoint_stop`，不再用长时间飞过终点后的状态；
- 新模型评估：主评估使用 `endpoint_stop`；旧 BC 在 `legacy` 下保留历史对照；
- 需要比较新旧模型时，双方必须在同一终止模式下评估，不能一边 `legacy`、一边
  `endpoint_stop`；
- 旧 artifacts 不覆盖，所有新结果写入新的版本化目录。

#### 奖励

- `GOAL` 保留现有成功奖励；
- `GOAL_PLANE_MISS` 给一次固定终端负奖励，参数名建议为
  `goal_plane_miss_penalty`；
- 可选增加一个固定的 miss-distance 比例项，距离定义为切平面处的
  `max(goal_distance_m - goal_radius_m, 0)`，参数名建议为
  `goal_plane_miss_distance_weight`；
- 不再使用“越界后每一步继续惩罚”，因为新模式已经不会产生越界后的长轨迹；
- 惩罚量级、是否启用距离项必须在 PPO 训练前预注册一次，不并行扫权重；
- 奖励不能替代观测：actor 仍然需要终点上下文才能判断自己是否即将越过切平面。

#### 新评估口径

新实验不再把旧的完整 RMSE 作为唯一主排序指标。固定输出三层结果：

| 层级 | 指标 | 用途 |
|---|---|---|
| 任务护栏 | `goal_capture_rate`、`goal_plane_miss_rate`、`miss_distance_at_plane_m`、`timeout_rate`、`safety_rate` | 先判断是否完成任务、是否在切平面失败 |
| 路径主指标 | `pre_goal_3d_rmse_m` | 到终止前的路径跟踪质量；新实验的主排序指标 |
| 历史回归 | `legacy_full_3d_rmse_m` | 与旧结果比较，防止新协议掩盖回归 |

补充约束：

- `post_goal_overshoot_m` 和 `terminal_longitudinal_rmse_m` 只在 `legacy` 轨迹上
  有意义；在 `endpoint_stop` 下应输出 `N/A`，不能填 0；
- `success_only_3d_rmse_m` 仍作辅助诊断，但必须和 `goal_capture_rate` 一起看；
- `miss_distance_at_plane_m` 记录首个切平面失败点到终点中心的 3D 距离；
  可同时记录 `goal_radius_excess_m = max(distance - goal_radius_m, 0)`；
- `time_to_goal_s` 或 `time_to_terminal_s` 作为效率护栏，不用它替代 RMSE；
- `legacy_full_3d_rmse_m` 和新的 `pre_goal_3d_rmse_m` 不可混为同一列。

P2B 的预注册门禁：

- `endpoint_stop` 的实现必须通过“目标球捕获优先于切平面失败”的单元测试；
- 旧 `legacy` 配置下所有历史 summary 和 checkpoint 仍可加载；
- 同一条轨迹在 `legacy` 与 `endpoint_stop` 下的成功/失败分类只在切平面漏检处不同；
- 新模式上线前必须验证：成功回合不因新终止逻辑被提前打断；
- `GOAL_PLANE_MISS` 的 episode 不得继续推进，也不得生成越界后的伪“过冲”指标。

## 5. P3：Default + strict125 双场景 DAgger

### 触发条件

P1/P2 显示主要瓶颈是**训练分布覆盖不足**，而不是动作时间一致性。

### 做法

- 从 P1 选出的最强 BC 出发；
- 同时在 default 和 strict125 的失败 seed 上采集教师纠正；
- 失败筛选至少覆盖：
  - default timeout；
  - strict125 timeout；
  - 高横向误差；
  - 终点附近失败；
  - 高饱和或高反转状态；
- 新数据与原始数据合并时保留 episode 来源、seed 和 teacher 标签；
- 先做一轮小规模 DAgger，不直接扩大采样量；
- BC 训练沿用当前连续动作头，除非 P4 已单独证明 action chunking 必要。

### 预注册门禁

- 同一 dev `20–39` 上，default `pre_goal_3d_rmse_m` 相对 P1 最强底座至少改善 `5%`；
- `legacy_full_3d_rmse_m` 不得出现超过 `5%` 的回归；
- `goal_capture_rate` 不下降，`goal_plane_miss_rate` 不上升；
- success 不下降超过 `0.05`；
- strict125 `pre_goal_3d_rmse_m` 不超过最强底座的 `1.10×`，success 不低于 `0.10`；
- safety 不增加超过 `0.02`；
- path-zero 仍应产生可解释的恶化；
- 未通过时不增加 DAgger 轮数、不改变 seed 列表、不放宽门禁。

## 6. P4：Action chunking / 降低控制频率

### 触发条件

P1/P2 显示主要瓶颈是**动作抖动、反转、rate-limit 或单步动作时间一致性**，而不是
数据覆盖不足。

### 候选实现

优先做一个最小版本，不直接引入复杂 Transformer：

- 一次输出 `H=5` 步动作，执行前 `H` 步或执行到新观测触发重规划；
- 如 `H=5` 无改善，再预注册 `H=10` 或 10Hz 决策频率，不并行扫多个频率；
- 评估时保持同一环境和同一动作解码规则；
- 旧单步 checkpoint 继续可加载，chunking 新功能默认关闭；
- 记录动作块内连续性、反转率、rate-limit 命中率和饱和恢复率。

### 预注册门禁

- default `pre_goal_3d_rmse_m` 相对 P1 最强底座改善至少 `5%`；
- `legacy_full_3d_rmse_m` 不得出现超过 `5%` 的回归；
- 动作反转率或 rate-limit 命中率至少改善 `10%`；
- success 不下降超过 `0.05`；
- strict125 `pre_goal_3d_rmse_m` 和 safety 不出现灾难性退化；
- 未通过时停止 chunking，不继续扫 `H` 或叠加 mixture-density。

## 7. P5：残差 PPO

### 前置条件

只能在 P2A 终点观测、P2B `endpoint_stop` 终止协议和 P3 终点定向 BC 都完成后启动。
禁止用 PPO 掩盖 BC 的动作表示问题，也禁止在没有终点观测时靠加大奖励强行训练。

### 做法

- 冻结强 BC；
- 只学习小幅残差动作，残差尺度先固定为小值；
- DADE 保持关闭；
- 训练环境启用 `endpoint_stop`，actor 使用编码进 `z_t` 的终点上下文；
- `GOAL` 成功奖励、`GOAL_PLANE_MISS` 终端惩罚和 timeout 惩罚在同一份预注册配置中
  固定，不通过扫奖励权重寻找结果；
- 先跑 zero-residual 基线，确认冻结 BC 的动作不被新训练器改变；
- 再做 PPO screen；
- 训练和评估使用独立 seed，held-out 仍保持未使用。

#### P5.1 稠密奖励范围

当前 `pc_path_track` 已经包含以下稠密项，第一版 PPO 不重复增加同类奖励：

- 路径进度 `w_prog`；
- 横向/垂向误差 `w_3d`；
- 路径切向航向误差 `w_yaw`；
- 速度平滑 `w_path_speed_smooth`；
- 能耗 `w_path_energy`；
- 动作变化、饱和、material saturation。

只保留以下两个“终点附近”候选，并且都必须按单变量、预注册方式加入：

| 优先级 | 候选 | 条件 | 目的 |
|---|---|---|---|
| 1 | 终端接近奖励 | 在 `terminal_capture_distance_m` 内使用现有 `terminal_progress_weight` 或等价项 | 让策略在终点前继续缩短到终点距离，而不是只追求路径进度 |
| 2 | 终点通道横向对准惩罚 | 仅在终点保护区，使用与 P2A 相同的平滑 `g_endpoint` 门控；第一版 PPO 仍有过冲时启用 | 减少临近切平面时的横向漏检 |

约束：

- `goal_plane_miss_penalty` 是终端奖励，不算稠密奖励；
- 不在第一版同时加入时间惩罚、额外速度惩罚、额外动作惩罚或全局过冲惩罚；
- 新增稠密项只用于 PPO，不改变 BC/DAgger 的监督标签和教师奖励；
- 每次 screen 只改一个奖励项；未通过就停止该项，不扫权重；
- 奖励候选必须先通过 P2A/P2B/P3 的契约和 BC 门禁，不能反过来用奖励掩盖缺失观测。

终点奖励与观测的对应关系：

- `goal_distance_3d_m`：原始值用于终端接近奖励和 `terminal_capture_ratio`；actor
  使用其单调门控版本 `goal_distance_proximity`；
- `goal_bearing_heading_error_rad` / `goal_bearing_pitch_error_rad`：用于终点通道对准
  惩罚，并乘以同一个平滑 `g_endpoint`；远距离时权重接近 0，不在路径跟踪阶段抢占奖励；
- `goal_radius_excess_m`：用于 `GOAL_PLANE_MISS` 的 miss-distance 比例项；
- `signed_goal_tangent_offset_m`：主要作为终止条件，不作为连续奖励重复惩罚。

### 预注册门禁

- `pre_goal_3d_rmse_m` 相对冻结 BC 改善至少 `3%`；
- `goal_capture_rate` 不下降，`goal_plane_miss_rate` 不上升；
- `legacy_full_3d_rmse_m` 不得出现超过 `5%` 的回归；
- success 不下降超过 `0.05`；
- safety 不增加超过 `0.02`；
- 先用 1 个训练 seed 做 screen；通过后再扩到 3 个训练 seed；
- 不通过时停止残差 PPO，不连续调 residual scale、reward 或学习率。

## 8. P6：在更强 BC 上复测 current/crab

### 前置条件

P1 已经确定一个强底座，且 P2 证明 current/crab 的失败不是评估口径造成的。

### 做法

- 保留 Goal17 的 2×2 结构，但只在新底座上重做最小对照：
  - 最强 BC baseline；
  - baseline + encoded current；
  - baseline + crab teacher；
  - baseline + encoded current + crab；
- 使用同一训练 seed、同一 dev 评估 seed、同一 BC 配方；
- 不重跑所有历史组合，只验证 current/crab 是否在新底座上出现可迁移收益；
- current 仍只输入 3D estimate，不直接拼接 valid/age。

### 预注册门禁

- 与 Goal17 保持一致：default `pre_goal_3d_rmse_m` 相对同训练 seed 的 baseline
  `<=0.95×`；
- `legacy_full_3d_rmse_m` 不出现超过 `5%` 的回归；
- success 不下降超过 `0.05`；
- safety 不增加超过 `0.02`；
- current-zero/current-shuffle 必须造成可解释退化；
- 如果再次失败，停止 current/crab 方向，不再用更弱或更强的数据掩盖。

## 9. 推荐执行顺序

### 阶段 A：只做诊断，不训练

1. P1 统一复评；
2. P2 误差分解；
3. P2.1 终点诊断指标；
4. 输出选择：数据覆盖问题还是动作时间一致性问题。

### 阶段 A2：终点协议与观测

1. P2B：实现默认关闭的 `endpoint_stop` 和 `GOAL_PLANE_MISS`；
2. P2A：把终点距离、带符号切向偏移、终点方位误差编码进 `z_t`；
3. 用 `legacy` 与 `endpoint_stop` 双模式复评最强 BC，确认口径切换没有掩盖回归；
4. 两项都通过契约测试后，才允许进入 P3 DAgger。

### 阶段 A3：nominal BC 重采与重训（Goal22，必须先于 P3）

1. 在 `realistic_calibrated` nominal 传感器域和 `endpoint_stop` 下重采 PC 教师示教；
2. 固定 N8/N12 两臂：N8 为 8D 控制臂，N12 为 12D 终点上下文臂；
3. 两臂使用相同 seeds、教师、预算和 BC 配方；
4. 先在 dev `20–39` 比较 default/strict125/path-zero，不碰 held-out；
5. N12 通过预注册门禁才作为 P3 输入，否则保留 N8 或停止；
6. Goal22 不包含 P3 DAgger、P5 PPO 或超参数扫参。

提示词和门禁见 `doc/goals/goal22_nominal_bc_rebase_prompt_20261107.md`。

### 阶段 B：只选一个主修复

- 如果 P2 指向分布覆盖：做 P3；
- 如果 P2 指向动作抖动/rate-limit：做 P4；
- 不并行扫 P3 和 P4 的参数。

### 阶段 C：条件性扩展

- 如果 P3/P4 产生强底座：先做 P6 验证 current/crab，或者直接进入 P5；
- P5 只能在 P2A、P2B 和 BC 门禁都通过后做；
- 任何失败方向都停止，不通过继续扫 seed、学习率或门禁“救”结果。

## 10. 不做清单

- 不继续在 Goal17 的 `A00/A11` 上扫 current/crab；
- 不重复 Goal16 的 rate-limit 教师扫描；
- 不重复 Goal18 的 sign/magnitude 动作头；
- 不在 BC 门禁失败时直接用残差 PPO；
- 不用 held-out `100–149` 选择 checkpoint 或超参数；
- 不把不同 seed 集的 RMSE 直接横比；
- 不把 path-zero、teacher 指标或诊断指标包装成主指标改善。
- 不用 `endpoint_stop` 结果覆盖旧 `legacy` 历史指标；
- 不在没有终点观测和 `GOAL_PLANE_MISS` 终止的情况下靠加大奖励强行训练 PPO；
- 不把 `post_goal_overshoot_m` 混进 `endpoint_stop` 的主排序。

## 11. 产物与提交

建议后续实验根目录：

```text
artifacts/lowlevel/pc_v0d_post_goal17_improvement_20261107/
```

建议文档：

```text
doc/goals/goal9_post_goal17_bc_improvement_plan_20261107.md
doc/goals/goal9_post_goal17_bc_improvement_report_20261107.md
```

每个阶段必须记录：

- checkpoint、dataset、config 的 SHA256；
- 训练 seed、评估 seed、seed role；
- device、dtype、num_envs、path_dim、command_dim；
- 完整命令行和开始/结束时间；
- 门禁判定和失败原因。

提交时只添加明确相关文件，不自动 commit/push：

```bash
git status --short
git diff -- <明确文件列表>
git add <明确文件列表>
git commit -m "docs(goal9): register post-goal17 BC improvement roadmap"
git push origin <当前分支>
```

## 12. 执行状态（2026-11-07）

P1/P2/P2.1 已按本规划完成，报告见
`doc/goals/goal9_post_goal17_bc_improvement_report_20261107.md`。

- [x] P1：7 个冻结 BC 的 default、strict125、path-zero 统一复评；
- [x] P1：从旧训练服务器恢复 `g19_dagger` 并校验 SHA256；
- [x] P1：统一排行榜、A 矩阵复评和 path-zero 比值已输出；
- [x] P2：default/strict125 的横向、法向、纵向误差分解；
- [x] P2：hint 投影与全局最近投影对比；
- [x] P2：success/timeout 分组、终点过冲和典型 trace；
- [x] P2：A11−A00 同 seed 配对；
- [x] P2.1：补齐 `success_only_3d_rmse_m`、`pre_goal_3d_rmse_m`、
  `post_goal_overshoot_m`、`terminal_longitudinal_rmse_m` 四项诊断指标；
  - 产物：`artifacts/lowlevel/pc_v0d_p2_1_diagnostic_metrics_20261107/`；
  - 脚本：`scripts/audit_pc_v0d_p2_diagnostic_metrics_20261107.py`；
  - 测试：`tests/test_pc_v0d_p2_diagnostic_metrics.py`。
- [x] P2A：终点上下文观测块（Goal20 完成）；
- [x] P2B：`endpoint_stop` 切平面未捕获终止协议（Goal20 完成）；
- [ ] Goal22：nominal 传感器域 BC 重采与重训（P3 前置）；
- [ ] P3：default + strict125 终点失败定向 DAgger；
- [ ] P6：在通过门禁的新底座上复测 current/crab。

直接结论：default 最强底座为 `g18_s0`（`6.3695 m / 0.70`），strict125 最强底座为
`g19_dagger`（`7.9071 m / 0.50`）。当前 3D RMSE 主要由 timeout 回合越过终点后的
纵向误差决定；`A11−A00` 的 default 改善不显著。P2.1 进一步表明 success-only 和
pre-goal RMSE 稳定在低值，而 post-goal overshoot 与 terminal longitudinal 是主要
差距来源。下一阶段先完成 P2B 终止协议、P2A 终点可观测性和定向 DAgger，再复测 A
矩阵；PPO 只有在 P2A/P2B/BC 门禁都通过后才考虑。
