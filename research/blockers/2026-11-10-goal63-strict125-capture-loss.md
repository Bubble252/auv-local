# Blocker：Goal63 strict125 true capture 丢失
日期：2026-11-10<br>
关联 Goal：Goal63<br>
状态：`resolved_by_goal64`，`GEOMETRY_SUPPORTED`
上游：Goal62 `INVALID_COUNTERFACTUAL_DEGENERATE_BOUNDARY`

## 1. 现象

Goal63 的 eval-only 终止抑制在 default 上通过主 Gate：

- false-success removal `0.621`，CI `[0.493, 0.745]`；
- true-capture retention `0.907`，CI `[0.821, 0.980]`；
- 4/4 checkpoint 通过。

但 strict125 只通过 removal，不通过 true retention：

- removal `0.611`，CI `[0.448, 0.769]`；
- retention `0.739`，CI `[0.545, 0.909]`；
- 0/4 checkpoint 通过。

zero-residual 在 strict125 通过方向检查，但 default retention 只有 `0.750`。
两个场景 timeout/safety 都不增加，plane-miss 分别增加 `+0.062/+0.088`，
extra steps 很小。说明大部分丢失发生在终点几何附近，而不是“继续飞很久后没捕获”。

## 2. 证据

- `doc/goals/goal63_result_20261110.md`
- `artifacts/lowlevel/goal63_terminal_suppression_20261110/goal63_terminal_suppression.json`
- `artifacts/lowlevel/goal63_terminal_suppression_20261110/goal63_terminal_suppression.md`
- 服务器 candidate runs
  `/data/yhx/underwater-robotics/src/plan_p1/goal60_action_reward_terminal_readonly_audit_20261110/artifacts/goal63_terminal_suppression/`

## 3. 已排除项

| 解释 | 状态 | 理由 |
|---|---|---|
| NoisyNet 主导 | 已排除 | deterministic eval；Goal59 mean 8/8 饱和；zero-residual 也复制 terminal 偏差 |
| 样本效率/优化不足 | 未支持 | 未训练；失败表现为 plane-miss 增加，不是 residual 学不动 |
| 数据损坏 | 已排除 | seeds、SHA、finite、radius metadata 全部通过 |
| timeout 主导 | 已排除 | candidate timeout 增量为 0 |
| 全局终止方向无效 | 已排除 | default policy 主 Gate 与 4/4 checkpoint 通过 |

## 4. 新 hypothesis

### H64-A：strict125 的 true loss 由 goal-plane crossing 早于更紧半径触发

机制：strict125 半径只有 `1.25 m`，收紧到 `1.00 m` 后，策略在接近目标时先穿过
goal plane；plane-miss guard 用原半径，因此 crossing 后不能再进入更紧球。

可证伪预测：被丢失的 baseline true-success episode 中，大多数 terminal kind 为
`GOAL_PLANE_MISS`，且 plane-miss 前 `p_nav` 尚未达到 `r_tight`；若丢失 episode
大多 timeout、safety 或已经在 `r_tight` 内但被错误分类，则 H64-A 不成立。

最便宜检验：只读记录 suppression candidate 的 terminal 前窗口，统计 plane
offset、`p_nav` 距离、到 plane 的时间；不训练。

### H64-B：strict125 的丢失不是 residual 幅度或动作饱和增加

机制：如果 true loss 是 endpoint geometry/approach 问题，被丢失 episode 的
residual L2、动作饱和和 terminal 前动作幅值不应显著高于保留 episode。若 residual
在丢失 episode 更大且更饱和，则应优先回到动作参数化/限幅。

可证伪预测：丢失组与保留组的 residual/action 指标差小于 seed 噪声，或效应方向
不能解释 plane miss；若丢失组显著更饱和且更晚减速，则 H64-B 被证伪。

最便宜检验：复用 Goal63 candidate 的 terminal 窗口 trace；必要时新 Goal64 只加
trace，不改 controller/reward。

### H64-C：default zero-residual 的 retention 失败也指向通用 geometry

机制：default forced zero-residual 仍有 `1/4` baseline true success 未保留，说明
问题不是 policy residual 特有，而是 `p_nav` 估计/终端接近几何或半径缩放。

可证伪预测：zero-residual 丢失 episode 与 policy 丢失 episode 有相同的
plane-crossing 顺序和 terminal offset 分布；若 zero-residual 的丢失完全由不同机制
造成，则 H64-C 不成立。

## 5. 排查顺序与决策

按 `数据 → 标签 → 评测 → baseline → 架构 → 文献`：

1. 数据/标签：Goal63 的 paired seed、SHA、finite 已通过；
2. 评测：确认 candidate success/capture/plane-miss 三种标签不混淆；
3. baseline：zero-residual 同向显示不是 residual 特有；
4. 架构：先记录 terminal 前窗口，再判断动作参数化/reward 是否仍是主因；
5. 文献：见
   `research/paper_reading/2026-11-10-goal63-terminal-suppression-partial.md`；
6. 最便宜 falsification：Goal64 只读 terminal-window trace，不训练。

决策：`pivot`。不回头调 Goal63 的 `0.25 m` margin，不放宽 retention 门槛，不启动
SAC/REDQ/DroQ，不打开 held-out `100–149`。

## 6. 算力

Goal64 可用 Goal63 的 10 个 candidate run 复用 checkpoint，按 scenario/checkpoint
分片并行记录 terminal trace；评估仍为 CPU/float64，保持 dev `20–59` 和 seed 层
合并。若进入训练分支，另行预注册更大 `num_envs`/batch，不静默改正式 Gate。

## 7. Goal64 关闭证据（2026-11-10）

Goal64 已用 40 env、dev `20–59`、同 checkpoint/SHA/半径只读重跑 10 个 candidate
并记录 terminal 前窗口。结果 `GEOMETRY_SUPPORTED`：

- strict125 policy 的 6 个 baseline true-success 丢失 episode 全部是
  `GOAL_PLANE_MISS`，且全部从未进入 `1.00 m` 的 candidate 半径；
- 丢失组 minimum `p_nav` 中位数 `1.172 m`，perpendicular offset 中位数
  `1.065 m`；plane crossing 前已经存在约一米的横向偏差；
- 丢失组与保留组 last-100-step residual L2 中位数分别为 `0.351/0.352`，
  material saturation `0.210/0.340`，scale/rate clip 均为 `0`；
- forced zero-residual 的 5 个丢失 episode 同样全部是 plane miss；default
  zero 与 policy 的 perpendicular offset 中位数只差 `0.091 m`。

因此 blocker 的根因不是 residual 幅度、动作饱和差异或 NoisyNet，而是
plane-miss guard 先于收紧 GOAL 半径触发，加上 terminal approach 的横向几何
偏差。下一步是新预注册 terminal rule 诊断，不重跑 Goal63 margin，不启动
SAC/REDQ/DroQ。

证据：

- `doc/goals/goal64_result_20261110.md`
- `artifacts/lowlevel/goal64_terminal_window_20261110/goal64.json`
- `artifacts/lowlevel/goal64_terminal_window_20261110/goal64.md`
- `research/paper_reading/2026-11-10-goal64-terminal-geometry.md`
