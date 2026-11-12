# Goal20 V-0D：P2B `endpoint_stop` + P2A 终点上下文观测

日期：2026-11-07

状态：**已执行；结论 `PASS_CONTRACT`，见
`goal20_v0d_p2b_p2a_endpoint_protocol_report_20261107.md`。** 执行后根据用户裁决，
`p_nav` 为权威 success，真实 `eta` 为诊断；历史
`paired_classification_only_plane_miss` 检查降级为 `DIAG`，不再作为硬门禁。

用途：这是下一个 Goal 的完整提示词。目标是先修终点终止协议和终点可观测性，
再用 `legacy` / `endpoint_stop` 双模式复评当前两个最强 BC，确认指标口径切换没有
掩盖回归。本 Goal 不启动 DAgger、不训练 PPO、不重采 BC。

## 0. 与 Goal9 的关系

`goal9_post_goal17_bc_improvement_plan_20261107.md` 是 umbrella 规划：P2A → P2B →
P3 DAgger → P5 残差 PPO。Goal20 只是其中 **P2B + P2A + 双模式复评** 这一段执行
提示词，不是新的总规划，也不替代 P3/P5。

因此：

- Goal20 里的“不训练 PPO”只是本阶段的边界，不是取消残差训练；
- Goal20 通过后，仍回到 Goal9 规划，先决定是否进入 P3 DAgger；
- P3 完成后，按 Goal9 第 7 节进入 P5 残差 PPO；
- 如果 Goal20 的契约测试失败，先修终止协议或观测接口，不应绕过它直接做残差训练；
- 残差 PPO 的前置条件、冻结 BC、DADE 关闭、`endpoint_stop` 和终点奖励规则，仍以
  Goal9 第 7 节为准。

关联文档：

- `doc/goals/goal9_post_goal17_bc_improvement_plan_20261107.md`
- `doc/goals/goal9_post_goal17_bc_improvement_report_20261107.md`
- `doc/goals/goal9_execution_state.md`

## 1. 任务目标

在 `/home/bubble/水下机器人/settled_局部规划器_反应式` 中，以非破坏方式完成：

1. 实现默认关闭的 `endpoint_stop` 终止模式；
2. 新增 `GOAL_PLANE_MISS` 终止类型和对应评估指标；
3. 新增版本化的终点上下文路径观测块，编码进 `z_t`；
4. 对 `g18_s0` 和 `g19_dagger` 做 `legacy` / `endpoint_stop` 双模式复评；
5. 输出测试证据、对比表、结论和下一阶段建议。

本 Goal **不做**：

- 不启动 P3 DAgger；
- 不训练 BC、PPO 或残差 PPO；
- 不改动作头、奖励主尺度、教师、海流、crab、plant 或传感器；
- 不调整 dev/held-out 划分；
- 不覆盖历史 artifacts、summary 或 checkpoint；
- 不自动 commit/push，不使用 `git add -A`。

## 2. 先读和先核对

执行前必须先读：

1. `doc/goals/goal9_post_goal17_bc_improvement_plan_20261107.md` 的 P2A、P2B、
   P5.1、执行顺序和门禁；
2. `doc/goals/goal9_post_goal17_bc_improvement_report_20261107.md` 的 P2/P2.1 表；
3. `doc/goals/goal9_execution_state.md` 的最新状态。

先核对：

- 当前没有教师、BC、PPO 或评估进程在运行；
- 默认配置：
  `configs/lowlevel/odarrl_control_pc_c3_random_realistic_v0d_teacher_r2b_current.json`；
- strict125 配置：
  `configs/lowlevel/odarrl_control_pc_c3_random_realistic_v0d_teacher_r2b_current_strict125.json`；
- `g18_s0` checkpoint：
  `artifacts/lowlevel/pc_v0d_sign_magnitude_bc_20261107/s0_continuous/pc_bc_actor.pt`；
- `g19_dagger` checkpoint：
  `artifacts/lowlevel/pc_v0d_r2b_weighted_bc_dagger_20261107/bc_satw_dagger2/pc_bc_actor.pt`；
- dev seeds 固定为 `20–39`；本 Goal 不使用 held-out `100–149`；
- 旧评估报告中的 `g18_s0` 和 `g19_dagger` 是当前 default/strict125 两个不同口径的
  最强底座，不能只复评一个。

若任何路径、SHA256 或配置与报告不一致，先写 `baseline_mismatch.md` 并停止，不要静默
替换 checkpoint。

## 3. 非破坏性要求

- 所有新行为必须默认关闭；旧配置和旧 checkpoint 的加载、评估、summary 语义不能改变；
- `legacy` 模式下旧终止类型、旧指标和旧轨迹必须可复现；
- 新增配置字段必须是可选的，缺省值等价于旧行为；
- 新路径观测必须使用版本化 schema，不能直接改写 `pc_path_v1` 的 dim、feature names
  或 hash；
- 旧 8D path checkpoint 必须继续可加载；新 12D endpoint block 只在新配置中启用；
- 评估器新增字段只能追加，不删除或重命名旧字段；
- 不覆盖 `artifacts/lowlevel/pc_v0d_p1_unified_bc_eval_20261107/` 和
  `artifacts/lowlevel/pc_v0d_p2_path_error_audit_20261107/`；
- 不修改历史报告中的数字，只追加新报告或新章节。

## 4. 阶段 1：P2B `endpoint_stop`

### 4.1 终止类型

在 `src/auv_lowlevel/batch/pc.py` 中新增独立终止类型：

```text
PCTerminalKind.GOAL_PLANE_MISS
```

它必须与 `TIME_LIMIT` 和 safety 类型分开，不能被统计成 timeout 或 safety。

建议新增配置字段，缺省值保持 legacy：

```text
termination.pc_goal_plane_miss_termination_enabled = false
termination.goal_plane_miss_penalty = 0.0
termination.goal_plane_miss_distance_weight = 0.0
```

若沿用其他命名，必须在报告中记录精确字段名。默认值必须让旧配置行为完全不变。

### 4.2 判定定义

在训练/评估环境中，对每个 env 计算：

```text
goal_distance_m = ||p_nav - p_goal||_2
signed_goal_tangent_offset_m = dot(p_nav - p_goal, end_tangent)
```

其中 `end_tangent` 是参考路径最后一段的单位切向量，不能用真实海流或仿真特权量。
`endpoint_stop` 启用时：

1. `goal_distance_m <= goal_radius_m`：`GOAL`；成功优先；
2. 否则若 `signed_goal_tangent_offset_m >= 0`：`GOAL_PLANE_MISS`，立即终止；
3. 否则继续现有 timeout、safety、out-of-bounds 等逻辑。

`GOAL_PLANE_MISS` 的返回语义：

- `terminated=True`；
- `truncated=False`；
- `safety_event=False`；
- 终止后该 env 不再推进，不生成越界后的长轨迹，也不生成伪 `post_goal_overshoot_m`。

`legacy` 模式必须完全保留当前行为：越过终点切平面但未进入捕获球时继续运行到 timeout。

### 4.3 奖励接口

本 Goal 只实现奖励接口，不进行 PPO 调参：

- `GOAL` 保留现有成功奖励；
- `GOAL_PLANE_MISS` 可给一次固定 `goal_plane_miss_penalty`；
- 可选的 miss-distance 项使用
  `max(goal_distance_m - goal_radius_m, 0)`；
- 默认关闭时两项都为 0，且不改变旧评估和旧 checkpoint；
- 具体权重留到 P5 预注册，本 Goal 不扫权重。

### 4.4 必须新增的测试

在现有 termination/reward 测试文件中增加：

1. goal 与 plane miss 同一步时，结果是 `GOAL`，不是 `GOAL_PLANE_MISS`；
2. 未进入捕获球但越过切平面时，结果是 `GOAL_PLANE_MISS`；
3. `GOAL_PLANE_MISS` 后 environment 不再推进；
4. `legacy` 下同一状态仍得到原来的 timeout/继续行为；
5. 成功 episode 不因新逻辑被提前终止；
6. `GOAL_PLANE_MISS` 不计入 safety，不计入 timeout；
7. 旧配置缺省字段时行为逐位不变。

## 5. 阶段 2：P2A 终点上下文观测

### 5.1 编码原则

终点上下文不能在 actor 顶层扩宽 MLP，也不能直接塞进原始 history token。它必须作为
独立 path block，经 path encoder 编码后写入 `z_t`。旧 8D path schema 保留为
`pc_path_v1`；新块使用新的 schema version/hash。

### 5.2 新特征

新增 `endpoint_context` block，actor 侧必选 4 个特征：

| 特征 | 定义 | 说明 |
|---|---|---|
| `goal_distance_proximity` | `g_endpoint * (1 - clip(d / d_enter, 0, 1))` | 远处 0，终点 1，必须单调 |
| `goal_tangent_offset_gated` | `g_endpoint * signed_goal_tangent_offset_m` | 只有临近终点才进入 actor |
| `goal_heading_error_gated` | `g_endpoint * wrap(goal_bearing_heading - heading)` | 只有临近终点才增强 |
| `goal_pitch_error_gated` | `g_endpoint * wrap(goal_bearing_pitch - pitch)` | 只有临近终点才增强 |

门控定义：

```text
s = clip((d_enter - goal_distance_3d_m) / (d_enter - d_full), 0, 1)
g_endpoint = s * s * (3 - 2 * s)
```

初版固定 `d_enter = 20 m`、`d_full = 5 m`，作为配置字段写入，训练/评估前冻结。

不要直接把 `goal_distance_3d_m * g_endpoint` 送进 actor：该函数不单调，远处为 0，
临近终点会先升后降。actor 侧必须使用 `goal_distance_proximity`。

原始 `goal_distance_3d_m`、`signed_goal_tangent_offset_m` 和两个 bearing error 仍保留
给终止、奖励、诊断和评估；终止判断始终使用未门控的原始值。

### 5.3 可选特征

以下可选，默认不启用，避免第一版无谓扩维：

- `endpoint_active_flag`；
- `terminal_capture_ratio`；
- `goal_radius_excess_m`。

如果加入，必须同样版本化、固定裁剪、记录尺度和测试旧 schema 兼容性。

### 5.4 实现位置

优先检查并最小修改：

- `src/auv_lowlevel/batch/pc_path.py`：path feature names、scales、block indices、
  schema hash、feature builder；
- `src/auv_lowlevel/batch/pc_env.py`：从 `p_nav`、`path.goal_ned_m`、终点切向量和姿态
  构建 endpoint block；
- `src/auv_lowlevel/batch/pc_policy.py`：path encoder 输入宽度和 metadata；
- `src/auv_lowlevel/batch/pc_vec_env.py`：observation layout/normalization；
- `scripts/evaluate_pc_bc.py`：`--path-dim` 兼容旧 `0/8` 和新 endpoint 维度，默认仍为
  旧 8D。

### 5.5 必须新增的测试

1. 远处 `goal_distance_proximity == 0`；
2. `goal_distance_proximity` 从远到近单调不减；
3. 到达终点时接近 1；
4. 门控连续，在 `d_enter` 和 `d_full` 两侧没有跳变；
5. `legacy` 路径下四个 endpoint 特征不进入 actor；
6. `pc_path_v1` hash 不变，旧 8D checkpoint 可加载；
7. 新 schema 的 feature names、scales、hash 稳定；
8. 终点特征只依赖 `p_nav`、路径终点和姿态，不读取真实海流或绝对真值；
9. path zero 时 endpoint block 也不泄漏；
10. actor 顶层输入宽度与旧模型保持一致，扩维只发生在 path encoder 内部。

## 6. 阶段 3：双模式复评

### 6.1 固定矩阵

必须跑完以下 8 个组合，不能只跑有利组合：

| BC | 场景 | 终止模式 |
|---|---|---|
| `g18_s0` | default | `legacy` |
| `g18_s0` | default | `endpoint_stop` |
| `g18_s0` | strict125 | `legacy` |
| `g18_s0` | strict125 | `endpoint_stop` |
| `g19_dagger` | default | `legacy` |
| `g19_dagger` | default | `endpoint_stop` |
| `g19_dagger` | strict125 | `legacy` |
| `g19_dagger` | strict125 | `endpoint_stop` |

固定条件：

- seeds `20–39`；
- CPU、`float64`；
- 旧 BC 使用 `--path-dim 8`；
- 同一 checkpoint、同一配置、同一 seeds 下比较 `legacy` 与 `endpoint_stop`；
- 新 P2A 还没有训练新 BC，因此本阶段不把 endpoint 特征接入旧 BC 的闭环评估；
- 评估主指标按新口径分层：
  - 任务护栏：`goal_capture_rate`、`goal_plane_miss_rate`、
    `miss_distance_at_plane_m`、`timeout_rate`、`safety_rate`；
  - 路径主指标：`pre_goal_3d_rmse_m`；
  - 历史回归：`legacy_full_3d_rmse_m`；
- `post_goal_overshoot_m` 和 `terminal_longitudinal_rmse_m` 在 `endpoint_stop` 下输出
  `N/A`，不能填 0。

### 6.2 对比要求

输出逐 seed 表和汇总表，至少回答：

1. `endpoint_stop` 是否消除了越过切平面后的长尾；
2. 原 timeout 中有多少变成 `GOAL_PLANE_MISS`；
3. 原 success 是否有任何一个被新终止逻辑误伤；
4. `legacy` 结果是否与旧 summary 逐位或按容差一致；
5. `pre_goal_3d_rmse_m` 是否与历史诊断一致；
6. `g18_s0` 和 `g19_dagger` 在 default/strict125 上是否保持原排序；
7. 新模式是否改变 safety、capture 或 timeout 的统计口径。

允许的结论只有：

- `PASS_CONTRACT`：实现契约和双模式一致性通过；
- `FAIL_CONTRACT`：旧行为、捕获优先、停止后不推进或指标口径不通过；
- `INCONCLUSIVE`：证据不足或检查未完成。

不要把“endpoint_stop 下完整 RMSE 变小”误写成策略改善；它只是去掉了失败后的长尾，
路径质量仍看 `pre_goal_3d_rmse_m`。

## 7. 产物与报告

新 artifacts 目录固定为：

```text
artifacts/lowlevel/pc_v0d_p2b_p2a_endpoint_20261107/
```

至少包含：

```text
contract_checks.json
legacy/g18_s0/default/summary.json
legacy/g18_s0/strict125/summary.json
legacy/g19_dagger/default/summary.json
legacy/g19_dagger/strict125/summary.json
endpoint_stop/g18_s0/default/summary.json
endpoint_stop/g18_s0/strict125/summary.json
endpoint_stop/g19_dagger/default/summary.json
endpoint_stop/g19_dagger/strict125/summary.json
comparison.json
comparison.md
```

新增报告：

```text
doc/goals/goal20_v0d_p2b_p2a_endpoint_protocol_report_20261107.md
```

报告必须包含：

- 改动文件清单；
- 配置字段和缺省值；
- 终止类型的优先级和测试证据；
- 终点观测公式、门控值、schema/hash；
- 8 个组合的对比表；
- 是否误伤 success；
- 是否保留 legacy 兼容；
- pytest 和 ruff 结果；
- 明确的 `PASS_CONTRACT` / `FAIL_CONTRACT` / `INCONCLUSIVE`；
- 下一阶段建议，但不要自动开始 P3 或 PPO。

同时更新 `doc/goals/goal9_execution_state.md`，追加本次状态和产物路径。

## 8. 验收门禁

本 Goal 通过必须同时满足：

- 相关 unit tests 全绿；
- `ruff check` 和格式检查通过；
- `legacy` 旧配置行为不变；
- 旧 8D checkpoint 仍可加载并评估；
- goal 捕获优先于 plane miss；
- `GOAL_PLANE_MISS` 后不再推进；
- `GOAL_PLANE_MISS` 不被统计为 timeout 或 safety；
- success 回合没有被新逻辑提前终止；
- `g18_s0` / `g19_dagger` 双模式评估完成，且比较口径相同；
- 新报告明确区分 `pre_goal_3d_rmse_m`、`legacy_full_3d_rmse_m` 和任务护栏；
- 没有使用 held-out seeds 选择实现或调参。

任一条件不满足，结论写 `FAIL_CONTRACT` 并停止，不继续启动 P3、DAgger 或 PPO。

执行后的用户裁决补充：`p_nav` 是可部署的主成功口径，真实 `eta` 只作诊断；历史
`paired_classification_only_plane_miss` 的字面检查保留为 `DIAG`，不再据此判
`FAIL_CONTRACT`。因此本次最终结论按更新后的契约写为 `PASS_CONTRACT`。

## 9. 执行顺序

严格按以下顺序，不要并行跳过：

1. 核对 baseline、进程和文件；
2. 实现 P2B 配置、终止类型、终止逻辑和测试；
3. 跑 P2B 定向测试和全量相关测试；
4. 实现 P2A 版本化 endpoint block、门控和测试；
5. 跑 P2A schema、单调性、旧 checkpoint 兼容测试；
6. 跑 `legacy` / `endpoint_stop` 8 组合复评；
7. 生成 comparison 和报告；
8. 更新执行状态；
9. 停止并等待用户决定是否进入 P3 DAgger。

## 10. 最终回答格式

完成后只汇报：

- 当前结论：`PASS_CONTRACT` / `FAIL_CONTRACT` / `INCONCLUSIVE`；
- P2B/P2A 是否实现；
- 8 个组合的关键结果；
- success 是否被误伤；
- 旧 checkpoint / 旧配置是否兼容；
- tests 和 ruff 结果；
- 报告与 artifacts 路径；
- 下一步建议，但不要自动执行。


