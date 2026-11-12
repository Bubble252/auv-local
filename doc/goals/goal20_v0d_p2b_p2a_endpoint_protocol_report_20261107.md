# Goal20 V-0D：P2B `endpoint_stop` + P2A 终点上下文观测报告

日期：2026-11-07

状态：**PASS_CONTRACT**

实现本身、legacy 兼容、8 组复评、`GOAL_PLANE_MISS` 隔离、`endpoint_stop` 长尾指标和
静态检查均通过。seed 级别仍有 5 个 legacy/endpoint 分类迁移；它们来自
`endpoint_stop` 使用 `p_nav`、legacy 使用真实 `eta` 的位置口径差异，不是终止后继续
推进造成的伪长尾。

用户已裁决：`p_nav` 是可部署的主判定口径，真实 `eta` 只作为诊断；历史
`paired_classification_only_plane_miss` 检查保留为 `DIAG`，不再是硬门禁。本 Goal
因此重判为 `PASS_CONTRACT`。方案 2（goal capture 使用真实 `eta`、plane miss 使用
`p_nav`）没有被混入主实验，也没有重跑；它只作为后续备选臂记录。

## 1. 与 Goal9 的关系

- Goal9 是总规划：P2A → P2B → P3 DAgger → P5 残差 PPO；
- Goal20 只执行 P2B + P2A + 双模式复评，不启动 P3、DAgger、BC 重采或 PPO；
- 本报告不改变 Goal9 的残差训练路线；P5 仍在 Goal9 第 7 节；
- 本 Goal 结论为 `PASS_CONTRACT`，但仍不自动进入 P3 或 P5；下一步由用户明确启动。

## 2. 改动文件

核心实现：

- `src/auv_lowlevel/config.py`：新增终止和终点观测配置字段；
- `src/auv_lowlevel/batch/pc.py`：新增 `PCTerminalKind.GOAL_PLANE_MISS`、终止优先级和
  终端奖励接口；
- `src/auv_lowlevel/batch/pc_path.py`：新增 12D `endpoint_context` path block、schema
  hash 和门控特征；
- `src/auv_lowlevel/batch/pc_env.py`：构造 endpoint block、使用 `p_nav` 判定
  endpoint 终止、支持 `advance_mask`；
- `src/auv_lowlevel/batch/pc_policy.py`：支持 path encoder 输入 8D/12D，actor 顶层
  输入宽度不变；
- `src/auv_lowlevel/batch/pc_vec_env.py`：observation layout、归一化和 path block mask；
- `scripts/evaluate_pc_bc.py`：新增 `--path-dim {0,8,12}`、
  `--termination-mode {legacy,endpoint_stop}` 和终点诊断指标；
- `scripts/summarize_pc_v0d_p2b_p2a_endpoint_20261107.py`：生成本报告的
  `comparison.json/md` 和 `contract_checks.json`。

测试：

- `tests/test_lowlevel_pc_reward_termination.py`
- `tests/test_lowlevel_pc_path_observations.py`
- `tests/test_lowlevel_pc_vec_env.py`
- `tests/test_lowlevel_pc_bc_eval_records.py`
- `tests/test_lowlevel_pc_eval_contract.py`

## 3. 配置字段与缺省值

| 字段 | 缺省值 | 作用 |
|---|---:|---|
| `termination.pc_goal_plane_miss_termination_enabled` | `false` | 启用后使用 `endpoint_stop`；缺省完全保持 legacy |
| `termination.goal_plane_miss_penalty` | `0.0` | `GOAL_PLANE_MISS` 的固定终端惩罚接口 |
| `termination.goal_plane_miss_distance_weight` | `0.0` | miss-distance 比例项接口，本 Goal 不调权重 |
| `observation.path_endpoint_context_enabled` | `false` | 是否启用 12D endpoint path block |
| `observation.path_endpoint_enter_m` | `20.0` | 终点门控开始距离 |
| `observation.path_endpoint_full_m` | `5.0` | 终点门控完全打开距离 |

`endpoint_stop` 只在显式配置或评估器 `--termination-mode endpoint_stop` 时启用；
旧配置、旧 checkpoint 和旧 summary 不改变语义。

## 4. P2B：`endpoint_stop` 与终止优先级

新增终止类型：

```text
PCTerminalKind.GOAL_PLANE_MISS = 9
```

`endpoint_stop` 的位置口径：

- `goal_distance_m = ||p_nav - p_goal||_2`；
- `signed_goal_tangent_offset_m = dot(p_nav - p_goal, end_tangent)`；
- `end_tangent` 来自参考路径最后一段，不使用真实海流或仿真特权量；
- 终止判断使用未门控原始值，门控只作用于 actor 的 path block。

启用 `endpoint_stop` 时的优先级：

```text
GOAL > GOAL_PLANE_MISS > timeout / safety / out-of-bounds
```

这个优先级是本次复核后修正的。实现中先把 timeout/safety/bounds 写入 `kind`，再写
`GOAL_PLANE_MISS`，最后写 `GOAL`；因此 `GOAL_PLANE_MISS` 不会在同一帧被 safety 或
timeout 覆盖。legacy 分支保持原来的赋值顺序，行为与旧版本一致。

返回语义：

- `GOAL_PLANE_MISS`：`terminated=True`、`truncated=False`、`safety_event=False`；
- 终止后通过 `advance_mask` 冻结该行，不再推进，也不生成越界后的伪
  `post_goal_overshoot_m` 或 `terminal_longitudinal_rmse_m`；
- `advance_mask=None` 时完全保持旧 `step()` 行为。

定向测试：

- `test_goal_plane_miss_is_opt_in_and_goal_capture_wins`
- `test_endpoint_stop_uses_nav_position_and_goal_capture_has_priority`
- `test_endpoint_stop_plane_miss_precedes_safety_and_timeout`
- `test_endpoint_stop_goal_precedes_invalid_state`
- `test_legacy_ignores_optional_nav_position_when_not_enabled`
- `test_pc_env_advance_mask_freezes_terminated_rows`

## 5. P2A：终点上下文观测

旧 8D schema 保留为 `pc_path_v1`，hash 不变：

```text
be882d27f3aa48fe7495c4bfaf27cd5062be588a9e350f495c8f3887c7dc4c64
```

新 12D schema 为 `pc_path_v2_endpoint`：

```text
1f746c2088172883d5d9489b5f025269650efc1dca40d2135326c1e839f0a847
```

新增四个 actor 侧特征：

| 特征 | 公式 | scale |
|---|---|---:|
| `goal_distance_proximity` | `g_endpoint * (1 - clip(d / d_enter, 0, 1))` | `1.0` |
| `goal_tangent_offset_gated` | `g_endpoint * signed_goal_tangent_offset_m` | `5.0` |
| `goal_heading_error_gated` | `g_endpoint * wrap(goal_bearing_heading - heading)` | `pi` |
| `goal_pitch_error_gated` | `g_endpoint * wrap(goal_bearing_pitch - pitch)` | `pi/2` |

门控：

```text
s = clip((d_enter - d) / (d_enter - d_full), 0, 1)
g_endpoint = s * s * (3 - 2 * s)
```

固定 `d_enter=20 m`、`d_full=5 m`。四个特征作为独立 `endpoint_context` block 经过
path encoder 写入 `z_t`，不进入原始 history token，不扩大 actor 顶层输入宽度；旧 8D
checkpoint 仍可加载。path-zero 时 endpoint block 也随 path 一起归零。

## 6. 8 组复评结果

固定条件：dev seeds `20–39`，CPU，`float64`，旧 BC 使用 `--path-dim 8`，同一
checkpoint、同一配置、同一 seeds 比较 `legacy` 与 `endpoint_stop`。

| BC | 场景 | 模式 | capture p_nav (seen) | capture eta (seen) | plane miss | timeout | safety | pre-goal 3D RMSE | full 3D RMSE | post overshoot | terminal longitudinal |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `g18_s0` | default | legacy | 0.10 | 0.70 | 0.00 | 0.30 | 0.00 | 2.6525 | 6.3695 | 6.5456 | 14.7857 |
| `g18_s0` | default | endpoint_stop | 0.70 | 0.65 | 0.30 | 0.00 | 0.00 | 2.6448 | 2.5716 | N/A | N/A |
| `g18_s0` | strict125 | legacy | 0.05 | 0.45 | 0.00 | 0.55 | 0.00 | 2.6341 | 9.4646 | 13.1783 | 15.7541 |
| `g18_s0` | strict125 | endpoint_stop | 0.45 | 0.45 | 0.55 | 0.00 | 0.00 | 2.6245 | 2.5545 | N/A | N/A |
| `g19_dagger` | default | legacy | 0.20 | 0.65 | 0.00 | 0.35 | 0.00 | 2.5662 | 6.8613 | 8.1072 | 14.3130 |
| `g19_dagger` | default | endpoint_stop | 0.70 | 0.50 | 0.30 | 0.00 | 0.00 | 2.5592 | 2.4888 | N/A | N/A |
| `g19_dagger` | strict125 | legacy | 0.15 | 0.50 | 0.00 | 0.50 | 0.00 | 2.5466 | 7.9071 | 10.8258 | 13.7465 |
| `g19_dagger` | strict125 | endpoint_stop | 0.50 | 0.40 | 0.50 | 0.00 | 0.00 | 2.5385 | 2.4780 | N/A | N/A |

`endpoint_stop` 下的 `legacy_full_3d_rmse_m` 是在“已停止轨迹”上计算的，不是策略
质量改善；路径质量主指标是 `pre_goal_3d_rmse_m`。终点后长尾指标在 `endpoint_stop`
下必须是 `N/A`，不能填 0。`endpoint_stop` 的 success 主口径是 `capture p_nav`；
`capture eta` 用真实位置复核，只作诊断，不参与门禁。

`legacy` 的 `capture p_nav (seen)` 只表示 legacy 因真实 `eta` 捕获而终止之前，导航
位置是否曾经进入捕获球。它不是 legacy success，也不是与 endpoint success 对称的
指标：legacy 不会等 `p_nav` 进入捕获球，所以这一列天然偏低，不能解释成策略退化。

主要结论：

- `endpoint_stop` 消除了 legacy timeout 回合越界后的长尾；4 个组合的 timeout 都降到
  `0.00`，原 timeout 主要转为 `GOAL_PLANE_MISS`；
- `pre_goal_3d_rmse_m` 与 legacy 非常接近，最大绝对差 `0.0096 m`，说明终止协议没有
  改变到点前的正常路径跟踪；
- `safety` 仍为 `0.00`；
- legacy 的 `p_nav` 终止前捕获率只有 `0.10/0.05/0.20/0.15`，而同一回合的真实
  `eta` 捕获率是 `0.70/0.45/0.65/0.50`；差值说明 legacy 往往在估计位置尚未进球时
  就因真实位置进球而终止，这正是不能把 legacy 的 `eta` success 当作可部署
  `p_nav` success 的原因；
- endpoint 模式下 `p_nav` 捕获率分别是 `0.70/0.45/0.70/0.50`，与各自 success 完全
  一致；真实 `eta` 诊断是 `0.65/0.45/0.50/0.40`；
- default 排序发生变化：legacy full RMSE 仍是 `g18_s0` 更好，但 endpoint 的 pre-goal
  RMSE 是 `g19_dagger` 更好（2.5592 对 2.6448）。这是去掉了 timeout 长尾后排序
  翻转，不能当成同一指标下的退化；
- strict125 排序不变：`g19_dagger` 的 pre-goal 和 legacy full 都优于 `g18_s0`；
- endpoint 双口径下 `g19_dagger/default` 和 `g19_dagger/strict125` 的 `p_nav` 捕获率
  都高于 `eta`；而 legacy 因提前终止，`p_nav` 捕获率远低于 `eta`。两组现象共同说明：
  `p_nav` 必须是 endpoint 协议的主判定，legacy 的真实 `eta` 只能作为历史对照。

口径审计：初版 evaluator 曾把 legacy 的 `goal_terminations` 直接记为 `p_nav`，但
legacy 的 goal 终止实际使用真实 `eta`。复核后已新增独立的
`pnav_goal_capture_seen`，逐步分别记录 `p_nav` 与真实 `eta` 是否进入捕获球，并重跑
8 组。修正前的产物备份在
`artifacts/lowlevel/pc_v0d_p2b_p2a_endpoint_20261107_pre_pnav_metric_fix_20261107/`，
本报告和 `comparison.md` 均使用修正后的双口径结果。

## 7. success 迁移与契约裁决

legacy 与 endpoint 的逐 seed 迁移如下。legacy outcome 是真实 `eta` goal termination，
endpoint outcome 是 `p_nav` goal/plane termination，两者不是同一测量口径：

| BC | 场景 | 迁移 |
|---|---|---|
| `g18_s0` | default | `success->success` 14，`timeout->plane_miss` 6 |
| `g18_s0` | strict125 | `success->plane_miss` 1，`success->success` 8，`timeout->plane_miss` 10，`timeout->success` 1 |
| `g19_dagger` | default | `success->success` 13，`timeout->plane_miss` 6，`timeout->success` 1 |
| `g19_dagger` | strict125 | `success->plane_miss` 1，`success->success` 9，`timeout->plane_miss` 9，`timeout->success` 1 |

具体有 5 个 legacy/endpoint 分类迁移。下表同时给出两套模式下的真实 `eta` 诊断：

| BC | 场景 | seed | legacy | endpoint | legacy eta | endpoint eta | legacy steps | endpoint steps | legacy overshoot | endpoint miss distance |
|---|---|---|---|---|---|---|---:|---:|---:|---:|
| `g18_s0` | strict125 | 22 | timeout | success | no | no | 2500 | 1191 | 25.5739 | N/A |
| `g18_s0` | strict125 | 23 | success | plane_miss | yes | yes | 1356 | 1405 | 0.0000 | 1.3362 |
| `g19_dagger` | default | 20 | timeout | success | no | no | 2500 | 1240 | 15.0674 | N/A |
| `g19_dagger` | strict125 | 22 | success | plane_miss | yes | yes | 1161 | 1240 | 0.0000 | 2.3766 |
| `g19_dagger` | strict125 | 31 | timeout | success | no | no | 2500 | 1258 | 27.9133 | N/A |

解释：

- `endpoint_stop` 用 `p_nav` 判定 goal/plane，legacy 用真实 `eta` 判定 goal；两者在
  估计误差边界上会把“真实 eta 已进球但 p_nav 未进球”的回合判成不同结果；
- 2 个 `success->plane_miss` 的 legacy `post_goal_overshoot_m` 为 0，说明 legacy 的成功
  发生在切平面之前，不是“越界后才成功”；endpoint 没有在 legacy 成功前终止
  （endpoint steps 分别为 1405/1240，均大于 legacy 的 1356/1161），所以不是
  “成功被提前打断”，而是 `p_nav` 没有在同一时刻识别成功；
- 3 个 `timeout->success` 是 `p_nav` 在 legacy timeout 之前进入了捕获球；
- 因此 `success_not_prematurely_terminated` 检查通过。历史
  `paired_classification_only_plane_miss` 的字面检查不通过，但用户已裁决该检查只作
  `DIAG`：它与 `endpoint_stop` 使用 `p_nav` 的目标判定存在口径冲突，不能再用真实
  `eta` 的 legacy 分类作为唯一成功语义；
- 采用双口径后，5 个差异都能被明确归因于 `p_nav` 与真实 `eta` 的估计边界，而不是
  终止后继续推进、提前终止或 safety/timeout 重叠造成的伪差异。

## 8. legacy 兼容与历史指标

- 4 个 `legacy` 组合的所有 student 数值字段与 P1 unified evaluation 在 `1e-12` 内
  完全一致；`g18_s0/default` 的 `path_three_d_rmse_m` 仍为 `6.369455348762715`；
- 旧 8D checkpoint 在 `--path-dim 8` 下正常加载并完成 8 组复评；
- 新 12D schema 没有改变 `pc_path_v1` 的 feature names、scales 或 hash；
- P2.1 的逐 seed `pre_goal_3d_rmse_m` 与本次 legacy 逐 seed 完全一致；
- P2.1 的 aggregate 使用 episode 等权 MSE，本次评估器使用时间步加权 RMSE，所以
  汇总值有约 `2.5–2.8%` 的差异；这是聚合口径不同，不是逐 seed 重算不一致。
  新评估器保留时间步加权，以和主路径指标一致。

## 9. 测试与静态检查

运行：

```bash
uv run pytest -q \
  tests/test_lowlevel_pc_path_observations.py \
  tests/test_lowlevel_pc_policy.py \
  tests/test_lowlevel_pc_vec_env.py \
  tests/test_lowlevel_pc_bc.py \
  tests/test_lowlevel_pc_contract.py \
  tests/test_lowlevel_pc_env.py \
  tests/test_lowlevel_pc_reward_termination.py \
  tests/test_lowlevel_pc_eval_contract.py \
  tests/test_lowlevel_pc_bc_eval_records.py
```

结果：`136 passed`。

格式与 lint：

```bash
uv run ruff check <goal20 changed files>
uv run ruff format --check <goal20 changed files>
```

结果：`All checks passed!`、`13 files already formatted`。

## 10. 契约检查

`artifacts/lowlevel/pc_v0d_p2b_p2a_endpoint_20261107/contract_checks.json`

通过的检查：

- 8 个组合齐全；
- seeds `20–39`、`path_dim=8`、终止模式字段正确；
- legacy 与旧 P1 数值逐字段一致；
- legacy 没有 plane miss；
- endpoint timeout、safety 都是 0；
- endpoint 的 post-goal overshoot / terminal longitudinal 为 `N/A`；
- endpoint success 不与 plane miss/timeout/safety 重叠；
- GOAL_PLANE_MISS 不被统计为 timeout 或 safety；
- 2 个 lost legacy success 都不是 endpoint 提前终止造成的；
- legacy 逐 seed pre-goal 与 P2.1 一致；
- endpoint pre-goal 与 legacy 最大差 `0.0096 m`；
- 每个 endpoint cell 都记录了 `p_nav` 成功率和真实 `eta` 诊断成功率；
- endpoint success / `goal_capture_rate` 与 `p_nav` 捕获集合严格一致；
- 5 个分类差异都带有 legacy/endpoint 两套 `eta` 诊断。

诊断检查，不计入失败门禁：

- `paired_classification_only_plane_miss_diagnostic`：5 个 seed 的分类迁移不是纯粹的
  timeout→plane_miss 或“越界后的 legacy success”。保留该结果用于解释
  `p_nav` 与真实 `eta` 的测量边界，不再据此判 FAIL。

## 11. 结论与下一步

结论：**PASS_CONTRACT**。

最终契约：

- `p_nav` 是主成功口径，endpoint success、`goal_capture_rate` 和主门禁都以它为准；
- 真实 `eta` 同步记录为 `goal_capture_rate_eta`，用于诊断估计误差边界，不参与主门禁；
- 历史 `paired_classification_only_plane_miss` 降级为 `DIAG`，不再要求逐 seed 分类
  与 legacy 完全一致；
- 方案 2（goal capture 用真实 `eta`、plane miss 用 `p_nav`）保留为未来备选臂，本次
  没有重跑，也没有混入主实验结果。

这个调整符合可部署观测：actor 只能使用导航位置 `p_nav`，真实 `eta` 是评估侧特权量。
本次没有启动 P3 DAgger，没有训练 BC/PPO/残差 PPO；后续是否进入 P3 或 P5 由用户明确
启动。

## 12. 产物

- `artifacts/lowlevel/pc_v0d_p2b_p2a_endpoint_20261107/legacy/g18_s0/default/summary.json`
- `artifacts/lowlevel/pc_v0d_p2b_p2a_endpoint_20261107/legacy/g18_s0/strict125/summary.json`
- `artifacts/lowlevel/pc_v0d_p2b_p2a_endpoint_20261107/legacy/g19_dagger/default/summary.json`
- `artifacts/lowlevel/pc_v0d_p2b_p2a_endpoint_20261107/legacy/g19_dagger/strict125/summary.json`
- `artifacts/lowlevel/pc_v0d_p2b_p2a_endpoint_20261107/endpoint_stop/g18_s0/default/summary.json`
- `artifacts/lowlevel/pc_v0d_p2b_p2a_endpoint_20261107/endpoint_stop/g18_s0/strict125/summary.json`
- `artifacts/lowlevel/pc_v0d_p2b_p2a_endpoint_20261107/endpoint_stop/g19_dagger/default/summary.json`
- `artifacts/lowlevel/pc_v0d_p2b_p2a_endpoint_20261107/endpoint_stop/g19_dagger/strict125/summary.json`
- `artifacts/lowlevel/pc_v0d_p2b_p2a_endpoint_20261107/comparison.json`
- `artifacts/lowlevel/pc_v0d_p2b_p2a_endpoint_20261107/comparison.md`
- `artifacts/lowlevel/pc_v0d_p2b_p2a_endpoint_20261107/contract_checks.json`
- `artifacts/lowlevel/pc_v0d_p2b_p2a_endpoint_20261107/validation.json`
