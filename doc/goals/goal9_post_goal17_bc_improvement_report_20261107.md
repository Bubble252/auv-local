# Goal9 后续报告：P1 统一复评与 P2 路径误差审计

日期：2026-11-07

状态：**P1、P2、P2.1 完成；“在 A 矩阵上继续提升”的下一阶段方案已给出，尚未启动训练。**

关联文档：

- `doc/goals/goal9_post_goal17_bc_improvement_plan_20261107.md`
- `doc/goals/goal17_v0d_current_crab_factorial_plan_20261107.md`
- `doc/goals/goal17_v0d_current_crab_factorial_report_20261107.md`
- `doc/goals/goal9_execution_state.md`

## 1. 结论摘要

- P1 统一复评后，default 最强 BC 是 `g18_s0`：
  - 3D RMSE `6.3695 m`；
  - success `0.70`；
  - timeout `0.30`；
  - safety `0`。
- strict125 最强 BC 是 `g19_dagger`：
  - 3D RMSE `7.9071 m`；
  - success `0.50`；
  - timeout `0.50`；
  - safety `0`。
- 旧结论中基于 test seeds `0–19` 的排序不能直接用于 dev `20–39`。例如 `g18_s0`
  在旧 test 集为 `8.9129 m`，在统一 dev 复评中为 `6.3695 m`，成为 default 第一。
- P2 证明当前 3D RMSE 的主要组成部分不是垂向误差，也不是 hint/global 投影口径，
  而是**终点之后的沿路径纵向误差**：
  - `g18_s0` 的纵向误差占总 MSE `74.1%`；
  - `g19_dagger` 占 `72.9%`；
  - `g19_satw` 占 `63.6%`；
  - `A00` 占 `70.9%`；
  - `A11` 占 `48.1%`。
- success 回合的纵向 RMSE 基本为 `0`；timeout 回合的纵向 RMSE 约为 `8–10 m`。
  因此当前分数主要由少数越过终点后继续前飞的 timeout episode 决定。
- hint 投影与全局最近投影只差 `0.012–0.034 m` RMSE，segment disagreement 不超过
  `0.9%`。主误差不是投影切换或评估口径造成的假象。
- `A11 - A00` 在 default 上平均改善 `0.5193 m`，但 95% CI 为
  `[-2.3682, +1.3297]`，不显著；横向误差反而平均恶化 `+0.5057 m`，主要收益来自
  纵向误差平均减少 `1.0111 m`。A11 当前仍然是在“换成功/超时分配”，不是稳定解决瓶颈。
- P2.1 用同一批 frozen traces 重算了四项诊断指标。success-only 与 pre-goal RMSE
  都处于正常跟踪水平，而完整 RMSE 的增量集中在 post-goal overshoot 和 terminal
  longitudinal；这进一步确认瓶颈发生在终点之后，而不是全程路径跟踪。

## 2. 固定评估口径

- 开发集 seeds：`20–39`，共 20 个 episode；
- default/path-zero 配置：
  `configs/lowlevel/odarrl_control_pc_c3_random_realistic_v0d_teacher_r2b_current.json`；
- strict125 配置：
  `configs/lowlevel/odarrl_control_pc_c3_random_realistic_v0d_teacher_r2b_current_strict125.json`；
- CPU、float64、`path_dim=8`；
- 没有训练新模型；
- 没有使用 held-out `100–149`；
- 没有把历史 `0–19` 结果混入新排行榜。
- 每个 BC 方向只取一个已冻结 checkpoint 做统一复评；Goal17 四个 A 格都是训练 seed
  `91430` 的 checkpoint，不是历史三训练 seed 聚合值。

## 3. P1 统一 BC 复评

### 3.1 候选完整性

`g19_dagger` 的 checkpoint 原先只在训练服务器上，本地缺失。本次从旧训练服务器拉回并
校验为报告中的 SHA256：

```text
acf690eae0829ee3f8fff0ced9613d989dc58f8c8975ff709f89aeece5151ce3
```

因此七个冻结候选都完成了统一复评，没有用历史结果静默替代缺失 checkpoint。

### 3.2 default 排行榜

| 排名 | BC | 3D RMSE | 水平 | 垂向 | success | timeout | safety | saturation |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | `g18_s0` | **6.3695** | 3.1363 | 0.8181 | **0.70** | 0.30 | 0 | 0.1455 |
| 2 | `g19_satw` | 6.7822 | 3.9818 | 0.9435 | **0.70** | 0.30 | 0 | 0.1161 |
| 3 | `g19_dagger` | 6.8613 | 3.4613 | 0.8862 | 0.65 | 0.35 | 0 | 0.1357 |
| 4 | `g17_a11` | 7.7864 | 5.5226 | 0.9720 | 0.55 | 0.45 | 0 | 0.1480 |
| 5 | `g17_a00` | 8.0384 | 4.2451 | 0.9008 | 0.50 | 0.50 | 0 | 0.1501 |
| 6 | `g17_a01` | 9.7369 | 5.6478 | 1.1340 | 0.45 | 0.55 | 0 | 0.1578 |
| 7 | `g17_a10` | 10.1432 | 6.6797 | 1.0437 | 0.35 | 0.65 | 0 | 0.1800 |

所有候选 safety 都是 `0`。default 排序按 3D RMSE；success 只作护栏。

### 3.3 strict125 与 path-zero

| BC | strict125 RMSE | strict125 success | strict125 timeout | path-zero/default |
|---|---:|---:|---:|---:|
| `g18_s0` | 9.4646 | 0.45 | 0.55 | 2.020 |
| `g19_satw` | 8.8176 | 0.50 | 0.50 | 1.699 |
| `g19_dagger` | **7.9071** | **0.50** | 0.50 | 1.793 |
| `g17_a11` | 11.2302 | 0.20 | 0.80 | 1.083 |
| `g17_a00` | 10.4970 | 0.20 | 0.80 | 1.053 |
| `g17_a01` | 10.8659 | 0.35 | 0.65 | 0.827 |
| `g17_a10` | 10.2598 | 0.30 | 0.70 | 0.930 |

解释：

- default 和 strict125 的第一名不是同一个 checkpoint；
- `g18_s0` 的 default 最强，但 strict125 明显不如 `g19_dagger`；
- `g19_dagger` 是当前更均衡的 strict125 候选；
- Goal17 A 矩阵的 path-zero 比值不稳定，甚至 `A10/A01` 小于 1。该指标可以保留为
  诊断，但不能单独证明 current/crab 更好。

### 3.4 Goal17 A 矩阵复评

| cell | 3D RMSE | success | timeout | path-zero/default |
|---|---:|---:|---:|---:|
| `A00` | 8.0384 | 0.50 | 0.50 | 1.053 |
| `A10` | 10.1432 | 0.35 | 0.65 | 0.930 |
| `A01` | 9.7369 | 0.45 | 0.55 | 0.827 |
| `A11` | 7.7864 | 0.55 | 0.45 | 1.083 |

统一复评确认：

- `A11` 是四格中 default 最好的；
- `A11` 也仍然比 `g18_s0` 差 `1.4169 m`，比 `g19_dagger` 差 `0.9251 m`；
- current-only `A10` 是四格最差；
- current/crab 的交互仍存在，但不能替代更强的 BC 底座。

## 4. P2 路径误差审计

### 4.1 误差组成

`path_three_d_rmse_m` 使用真实 `eta` 到路径投影点的三维距离。P2 把投影残差分解为：

- 横向：`dot(position - projection_point, lateral_axis)`；
- 法向/垂向：`dot(position - projection_point, vertical_axis)`；
- 纵向：`dot(position - projection_point, tangent)`。

default 结果：

| BC | 3D RMSE | 水平 RMSE | 法向 RMSE | 纵向 RMSE | 水平 MSE 占比 | 法向 MSE 占比 | 纵向 MSE 占比 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `g18_s0` | 6.3695 | 3.1363 | 0.8181 | 5.4831 | 0.242 | 0.016 | **0.741** |
| `g19_satw` | 6.7822 | 3.9818 | 0.9435 | 5.4086 | 0.345 | 0.019 | **0.636** |
| `g19_dagger` | 6.8613 | 3.4613 | 0.8862 | 5.8577 | 0.254 | 0.017 | **0.729** |
| `g17_a11` | 7.7864 | 5.5226 | 0.9720 | 5.4022 | 0.503 | 0.016 | 0.481 |
| `g17_a00` | 8.0384 | 4.2451 | 0.9008 | 6.7664 | 0.279 | 0.013 | **0.709** |

垂直/法向误差只占 1.3%–1.9%，不是当前主矛盾。

### 4.2 success 与 timeout 的分布完全不同

| BC | 组 | episode | 3D RMSE | 水平 | 法向 | 纵向 |
|---|---|---:|---:|---:|---:|---:|
| `g18_s0` | success | 14 | 2.6827 | 2.6362 | 0.4879 | 0.0940 |
| `g18_s0` | timeout | 6 | 10.8831 | 4.0708 | 1.2945 | 10.0097 |
| `g19_dagger` | success | 13 | 2.0411 | 1.9922 | 0.4438 | 0.0163 |
| `g19_dagger` | timeout | 7 | 11.2593 | 5.1826 | 1.3704 | 9.9012 |
| `g17_a00` | success | 10 | 2.8605 | 2.8258 | 0.4299 | 0.1123 |
| `g17_a00` | timeout | 10 | 11.0023 | 5.2968 | 1.1992 | 9.5685 |
| `g17_a11` | success | 11 | 2.4197 | 2.3882 | 0.3892 | 0.0186 |
| `g17_a11` | timeout | 9 | 11.2948 | 7.7978 | 1.3837 | 8.0530 |

因此：

- success 回合本身的跟踪质量并不差，`g19_dagger` success 回合 3D RMSE 只有 `2.04 m`；
- timeout 回合的 3D RMSE 约 `10.9–11.3 m`，其中纵向误差约 `8–10 m`；
- 当前总分主要是在惩罚终点之后继续航行，而不是惩罚深度跟踪或普通横向跟踪。

### 4.3 不是 hint/global 投影造成的

| BC | 全局投影 3D RMSE | 全局−hint | segment disagreement | progress disagreement p95 | 越过终点步数 | 终点饱和且过冲 |
|---|---:|---:|---:|---:|---:|---:|
| `g18_s0` | 6.3531 | -0.0164 | 0.005 | 0.7194 | 0.131 | 0.131 |
| `g19_satw` | 6.7486 | -0.0336 | 0.009 | 1.2857 | 0.142 | 0.141 |
| `g19_dagger` | 6.8298 | -0.0315 | 0.007 | 0.7050 | 0.163 | 0.163 |
| `g17_a11` | 7.7684 | -0.0180 | 0.006 | 0.6844 | 0.203 | 0.203 |
| `g17_a00` | 8.0261 | -0.0124 | 0.005 | 0.7166 | 0.222 | 0.222 |

全局投影只把 RMSE 改善 `0.012–0.034 m`，远小于 `5–7 m` 的纵向误差；因此不应把
历史主指标改成全局最近投影。当前带 hint 的投影是连续的，主误差是真实过冲。

### 4.4 A11−A00 的配对解释

default 同 seed 配对：

| 指标 | A11−A00 均值 | 95% CI |
|---|---:|---:|
| 3D RMSE | -0.5193 | [-2.3682, +1.3297] |
| 水平 RMSE | +0.5057 | [-1.1436, +2.1549] |
| 法向 RMSE | -0.0483 | [-0.2708, +0.1741] |
| 纵向 RMSE | -1.0111 | [-2.7763, +0.7541] |
| speed error | -0.0621 | [-0.1248, +0.0006] |

strict125 配对：

| 指标 | A11−A00 均值 | 95% CI |
|---|---:|---:|
| 3D RMSE | -0.1285 | [-3.9979, +3.7409] |
| 水平 RMSE | +0.9327 | [-2.6219, +4.4874] |
| 法向 RMSE | +0.2271 | [-0.2756, +0.7297] |
| 纵向 RMSE | -1.1436 | [-3.3246, +1.0374] |
| speed error | -0.0901 | [-0.1956, +0.0154] |

success/timeout 迁移也不稳定：default 中 `A11` 相对 `A00` 有 3 个
timeout→success，同时有 2 个 success→timeout；strict125 中有 4 个
timeout→success，同时有 4 个 success→timeout。

结论：当前 `A11` 的收益是纵向过冲减少，但代价是横向变差；净收益不显著，不能据此
把 current/crab 升级为稳定主方案。

### 4.5 strict125

| BC | 3D RMSE | success | timeout | 水平 | 法向 | 纵向 | 纵向 MSE 占比 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `g19_dagger` | **7.9071** | 10/20 | 10/20 | 4.0888 | 0.9788 | 6.6967 | 0.717 |
| `g18_s0` | 9.4646 | 9/20 | 11/20 | 4.7534 | 0.8501 | 8.1401 | 0.740 |
| `g17_a00` | 10.4970 | 4/20 | 16/20 | 6.3501 | 0.9504 | 8.3042 | 0.626 |
| `g17_a11` | 11.2302 | 4/20 | 16/20 | 8.6033 | 1.5217 | 7.0559 | 0.395 |

strict125 同样由纵向过冲主导。`A11` 的纵向 MSE 占比低一些，但水平 RMSE 增至
`8.60 m`，总 RMSE 反而最差，说明 current/crab 没有解决 strict125 的核心失败模式。

### 4.6 P2.1 终点诊断指标

新增的四项指标只作为诊断输出，不替换历史主指标 `path_three_d_rmse_m`。所有指标
固定使用 dev seeds `20–39`，并由 frozen P2 traces 后处理得到；没有重新训练或
rollout。`hint_rmse_3d_m` 和 `success_group.hint_rmse_3d_m` 的重算误差均为 `0`。

| BC | 场景 | 完整 RMSE | success-only | pre-goal | post-goal overshoot | terminal longitudinal | 最终切向偏移 | success/timeout |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `g18_s0` | default | 6.3695 | 2.6827 | 2.5805 | 6.5456 | 7.9721 | 4.6025 | 14/6 |
| `g18_s0` | strict125 | 9.4646 | 3.1774 | 2.5643 | 13.1783 | 11.5700 | 11.5559 | 9/11 |
| `g19_satw` | default | 6.7822 | 2.7318 | 2.5975 | 6.5164 | 7.7368 | 4.3957 | 14/6 |
| `g19_dagger` | default | 6.8613 | 2.0411 | 2.4969 | 8.1072 | 8.4068 | 5.4404 | 13/7 |
| `g19_dagger` | strict125 | 7.9071 | 1.5872 | 2.4847 | 10.8258 | 9.6393 | 8.3033 | 10/10 |
| `g17_a00` | default | 8.0384 | 2.8605 | 2.7198 | 9.9257 | 9.7930 | 7.5421 | 10/10 |
| `g17_a00` | strict125 | 10.4970 | 4.1322 | 2.7079 | 15.5091 | 12.0484 | 12.0589 | 4/16 |
| `g17_a11` | default | 7.7864 | 2.4197 | 2.7705 | 7.5704 | 7.8462 | 4.4978 | 11/9 |
| `g17_a11` | strict125 | 11.2302 | 1.4375 | 2.7609 | 13.2171 | 10.2555 | 7.1521 | 4/16 |

解释：

- success-only 和 pre-goal RMSE 大致在 `1.44–4.13 m` 与 `2.48–2.77 m`，说明模型在
  真正完成任务的回合、以及到点前的路径跟踪并不差；
- 完整 RMSE 与上述两项之间的差距，主要由 post-goal overshoot 和 terminal
  longitudinal 解释；
- `A11` 没有消除终点过冲：default overshoot 仍为 `7.57 m`，strict125 为
  `13.22 m`；
- 这支持下一步优先补终点可观测性和终点纠偏数据，而不是先做残差 PPO。

完整产物：`artifacts/lowlevel/pc_v0d_p2_1_diagnostic_metrics_20261107/`。

## 5. 根因判断

当前 actor 路径输入只有 8 维：

```text
signed_cross_track_m
signed_vertical_error_m
tangent_heading_error_rad
tangent_pitch_error_rad
progress_ratio
remaining_ratio
curvature_1_per_m
lookahead_heading_delta_rad
```

其中 `progress_ratio` 在路径末端被裁剪到 `1.0`，`remaining_ratio` 随之变成 `0.0`；
`lookahead` 也被裁剪到终点。结果是：

- “刚好到达终点”；
- “已经越过终点 20 m”；

在路径特征中几乎不可区分。actor 顶层输入又没有绝对位置，因此 BC 策略在越过终点后
缺少明确信号去转向或回收，容易继续沿末端切向前飞。P2 的纵向过冲不是单纯奖励不够，
而是**终点状态可观测性存在别名**。

教师本身也没有完全解决该问题：default teacher 为 `5.62 m / 0.75`，strict125 为
`7.74 m / 0.45`。因此后续不能假设“只要蒸馏教师就会消除 timeout”，教师数据也需要
覆盖已经越过终点的纠偏状态。

## 6. 在 A 矩阵上继续提升的方案

### 原则

- 不先做残差 PPO；
- 不继续在旧 `A00/A11` 上扫 current/crab 超参数；
- 不加过冲惩罚；
- 不把 current 直接裸拼进 actor；
- 不改变旧 8D checkpoint 的加载语义；
- 新路径/终点特征仍编码到 `z_t`，actor 顶层宽度保持不变；
- 先在 dev `20–39` 验证，再冻结配方后才允许 held-out。

### 优先级

| 优先级 | 改动 | 目的 | 为什么现在做 |
|---|---|---|---|
| 1 | 增加“终点上下文”观测块并编码进 `z_t` | 区分到点与越过终点 | 直接修复 P2 证明的状态别名 |
| 2 | 对 timeout / 越过终点状态做双场景 DAgger | 给新观测提供教师纠偏标签 | 只有特征、没有纠正数据仍然不够 |
| 3 | 在新底座上重做 2×2 current/crab | 判断 current/crab 是否有可迁移价值 | Goal17 的负结果被弱底座和终点失败污染 |
| 4 | 仅在新 BC 过门禁后考虑残差 PPO | 修正剩余系统偏差 | 不能再用 PPO 掩盖状态不可观测 |

### 6.1 终点上下文块

建议新增一个默认关闭、版本化的终点观测块，特征全部由可部署的 `p_nav` 和参考路径
计算，不使用真实海流或仿真真值。候选 4 维：

```text
goal_distance_3d_m
signed_goal_tangent_offset_m     # 沿末端切向，越过终点为正
goal_bearing_heading_error_rad
goal_bearing_pitch_error_rad
```

可选的替代/补充特征：

```text
endpoint_active_flag             # 是否已进入终点保护区
terminal_capture_ratio           # goal_distance / goal_radius，便于归一化
```

实现约束：

- 新特征作为独立 block 编码后加到 `z_t`，不扩大 actor MLP 输入宽度；
- 旧 8D 路径 schema 继续可加载；
- 新特征必须先写 schema hash、归一化尺度和回归测试；
- 不把“越过终点”直接写成惩罚项，只作为观测。

### 6.2 双场景终点 DAgger

从 P1 的两个底座分别做最小 screen：

- `g18_s0`：default 主底座；
- `g19_dagger`：strict125 主底座。

采集状态必须覆盖：

- default timeout；
- strict125 timeout；
- 已越过终点且 `goal_tangent_offset_m > 0` 的状态；
- 终点附近高横向误差；
- 终点附近高饱和/rate-limit 状态。

标签仍由教师给出；新增数据保留 seed、scenario、teacher 和失败原因。不要把历史
`0–19` 与新 dev 数据混在一起做选择。

### 6.3 预注册门禁

终点观测 + DAgger 的 screen 只有同时满足以下条件才进入 A 矩阵复测：

- default 3D RMSE 相对对应 P1 底座至少改善 `5%`；
- default success 不下降超过 `0.05`；
- default timeout 至少减少 `1` 个 episode，优先看纵向 RMSE；
- strict125 RMSE 不超过对应 P1 底座的 `1.10×`；
- safety 不增加超过 `0.02`；
- path-zero 仍应出现可解释退化；
- 同一配方至少通过两个训练 seed，不能只靠一个训练 seed 的偶然结果。

如果终点块单独通过而 DAgger 没有增量，保留终点块并停止扩展 DAgger；如果两者都不通过，
停止“加观测/加数据”路线，不要继续扫 block 维度或特征尺度。

### 6.4 A 矩阵复测

只在新底座通过上述门禁后做最小 2×2：

| cell | current encoder | crab |
|---|---|---|
| `A2-00` | off | off |
| `A2-10` | on | off |
| `A2-01` | off | on |
| `A2-11` | on | on |

门禁沿用 Goal17：

- default RMSE ratio `<=0.95×`；
- success 绝对下降不超过 `0.05`；
- safety 绝对增加不超过 `0.02`；
- current-zero/current-shuffle 必须造成可解释退化；
- 三训练 seed 聚合通过后才允许进入 PPO；
- 仍然不使用 held-out 选模型。

## 7. 明确不优先做的方向

- **不优先做垂向误差优化**：垂向 MSE 占比只有 `1.3%–1.9%`。
- **不优先做 speed head**：P2 的 timeout 主要由终点后纵向位移造成，speed error 不是
  第一主因；只有在终点问题解决后仍然存在明显速度跟踪瓶颈，才重新评估。
- **不优先做 action chunking**：P2 显示 success 回合可以稳定到 `2.0–2.9 m`，当前更像
  终点状态不可观测，而不是单步动作时间一致性。
- **不直接做 current/crab 的再扫参**：A11 的收益不显著，且被终点过冲污染。
- **不直接做残差 PPO**：在状态不可区分的情况下，PPO 只能拟合评估分布，不能提供新的
  终点可观测性。

## 8. 产物与复现命令

### P1

```text
artifacts/lowlevel/pc_v0d_p1_unified_bc_eval_20261107/
  <bc>/<default|strict125|path_zero>/summary.json
  report/summary.json
  report/summary.md
logs/goal_p1p2_unified_bc/
```

汇总命令：

```bash
.venv/bin/python scripts/summarize_pc_v0d_p1_unified_bc_20261107.py \
  --root artifacts/lowlevel/pc_v0d_p1_unified_bc_eval_20261107 \
  --output artifacts/lowlevel/pc_v0d_p1_unified_bc_eval_20261107/report
```

### P2

```text
artifacts/lowlevel/pc_v0d_p2_path_error_audit_20261107/
  <bc>/<default|strict125>/summary.{json,md}
  <bc>/<default|strict125>/traces.npz
  <bc>/<default|strict125>/selected_traces/
  report/summary.json
  report/summary.md
logs/goal_p2_path_audit_20261107/
```

审计命令模板：

```bash
.venv/bin/python scripts/audit_pc_bc_path_error_20261107.py \
  --config configs/lowlevel/odarrl_control_pc_c3_random_realistic_v0d_teacher_r2b_current.json \
  --checkpoint <checkpoint> \
  --output <output> \
  --seed-start 20 \
  --seed-count 20 \
  --device cpu \
  --dtype float64 \
  --path-dim 8
```

汇总命令：

```bash
.venv/bin/python scripts/summarize_pc_v0d_p2_path_audit_20261107.py \
  --root artifacts/lowlevel/pc_v0d_p2_path_error_audit_20261107 \
  --output artifacts/lowlevel/pc_v0d_p2_path_error_audit_20261107/report
```

## 9. 完成审计

- [x] P1 的 7 个冻结 checkpoint × 3 个场景全部完成；
- [x] P1 的 21 份 summary 都使用 dev `20–39`、`path_dim=8`；
- [x] `g19_dagger` 缺失本地 checkpoint 已从服务器恢复并校验 SHA256；
- [x] P2 完成 `g18_s0`、`g19_satw`、`g19_dagger`、`A00`、`A11` 的 default 审计；
- [x] P2 完成 `g18_s0`、`g19_dagger`、`A00`、`A11` 的 strict125 审计；
- [x] P2 的 hint RMSE 与 P1 主指标逐 seed 对齐到机器精度；
- [x] A11−A00 配对差值、成功/超时分组、终点过冲和投影差异均已输出；
- [x] held-out `100–149` 未使用；
- [x] 没有训练新模型、没有修改历史主指标；
- [ ] 终点上下文块 + 双场景 DAgger 尚未实现（下一阶段，需要用户确认后启动）。

## 10. 提交建议

本报告和新增分析脚本不自动提交。若用户要求提交，只添加明确相关文件：

```bash
git status --short
git diff --check
git add doc/goals/goal9_post_goal17_bc_improvement_report_20261107.md
git add scripts/audit_pc_bc_path_error_20261107.py
git add scripts/summarize_pc_v0d_p1_unified_bc_20261107.py
git add scripts/summarize_pc_v0d_p2_path_audit_20261107.py
git commit -m "docs: report P1/P2 BC and path-error diagnostics"
```

禁止使用 `git add -A`，也不要在未检查前自动 push。
