# Goal28：任务级终点捕获 / 过平面失败诊断

**日期**：2026-11-09<br>
**状态**：`DIAG_COMPLETE`

## Research Contract

```yaml
autoresearch:
  required: true
  triggers:
    - gate_fail_twice
    - no_metric_movement_after_3_fixes
    - heldout_collapse
    - seed_noise_ambiguous
  ledger: research/LEDGER.jsonl
  blockers_dir: research/blockers/
  reading_dir: research/paper_reading/
  max_outer_loops: 3
  check_command: python ../scripts/check_research_ledger.py --goal Goal28 --goal-doc doc/goals/goal28_terminal_capture_plane_miss_20261109.md --strict
```

本 Goal 首先是**只读诊断**，不是新的 PPO/residual 训练。只有在诊断完成并通过
预注册判定后，才允许另写 Goal29 或本 Goal 的追加实验来测试干预。

## 1. 背景与证据

Goal27 已停止命令残差和 supervised residual 主线。C1/R1/H1/G1/A1/H3-SR 都没有
产生可部署的路径跟踪收益；继续扫 reward、gamma、seed 或 residual 预算没有证据支持。

现有 Goal26/Goal27 的 40-seed dev 结果已经暴露一个更基础的问题：

- `endpoint_stop` 的终止位置使用 `p_nav`，不是 true `eta`；
- 因此现有 `success` 是“导航估计命中半径/过平面”的成功，不等于真实状态捕获；
- D3 default 的 `success_rate` 为 `0.725`，但 `goal_capture_eta_rate` 只有 `0.425`；
- D3 strict125 的 `success_rate` 为 `0.500`，`goal_capture_eta_rate` 只有 `0.250`；
- plane miss 在 D3 default/strict125 分别为 `0.275/0.500`，过平面时的平均 miss
  distance 为 `3.153/2.551 m`，明显大于对应 goal radius。

这说明“路径 RMSE 不够好”与“任务终点没有被真实捕获”可能是两个不同问题。继续
优化路径误差或 residual 之前，必须先回答：失败到底来自估计偏差、终端几何误差、
终端速度/纵向过冲，还是执行器权限不足。

### 1.1 已有 dev 证据（测量前已知）

| arm | scenario | success | eta capture | plane miss | miss distance (m) | final goal distance (m) | pre-goal RMSE (m) |
|---|---|---:|---:|---:|---:|---:|---:|
| D3 | default | 0.725 | 0.425 | 0.275 | 3.153 | 2.289 | 2.849 |
| D3 | strict125 | 0.500 | 0.250 | 0.500 | 2.551 | 1.899 | 2.829 |
| G26-P2 | default | 0.725 | 0.425 | 0.275 | 3.117 | 2.280 | 2.833 |
| G26-P2 | strict125 | 0.475 | 0.250 | 0.525 | 2.512 | 1.906 | 2.812 |
| A1 | default | 0.700 | 0.450 | 0.300 | 2.994 | 2.271 | 2.812 |
| A1 | strict125 | 0.525 | 0.250 | 0.475 | 2.573 | 1.867 | 2.792 |
| H3-SR | default | 0.650 | 0.350 | 0.350 | 3.049 | 2.346 | 2.826 |
| H3-SR | strict125 | 0.350 | 0.225 | 0.650 | 2.400 | 1.989 | 2.806 |

这些数字只作为诊断输入，不作为新的性能结论。A1/H3-SR 已经失败，不能因为本 Goal
重新解释为“值得继续 residual”。

## 2. 假设

每个假设都必须能在不训练新策略的情况下被支持或证伪。

### H1：导航估计把一部分未捕获 episode 计成了 success

机制：`endpoint_stop` 用 `p_nav` 判断半径和过平面；导航误差可能让真实状态尚未
进入目标半径，或因估计偏差提前/延后终止。

若成立应看到：

- `success_protocol` 明显高于 `success_eta`；
- `pnav_only_success / success_protocol >= 0.20`（至少一个 scenario）；
- p_nav-only episode 的 `nav_position_error` 或终端 p_nav-eta 差值大于 eta-captured
  episode。

若失败应看到：p_nav-only 比例低于 `0.10`，且估计误差不能区分 success/plane miss。

最便宜证伪：直接重算现有 40-seed summary 的 `success & !goal_capture_eta`，并比较
两类 episode 的 `nav_position_error_mean_m`/`p95`。不训练。

### H2：过平面失败主要是终端横向/垂向几何误差

机制：策略在接近终点时已经无法把真实位置压回目标半径；过平面事件只是暴露了终端
横/垂向偏差，而不是速度或终止语义的问题。

若成立应看到：plane miss 的终端偏移在垂直于 goal tangent 的方向上显著大于
goal radius，且横向或垂向至少一个分量解释了大部分 miss distance。

若失败应看到：终端垂直/横向误差接近零，或 miss 主要由 p_nav-eta 估计偏差造成。

最便宜证伪：对 D3/P2 的 plane miss 和 matched success 运行固定 terminal trace，
把 termination offset 分解为 tangent、horizontal-normal、vertical 三个分量。

### H3：终端速度/纵向过冲是主因

机制：接近目标时速度仍过高，纵向制动/转向不足，导致过早穿过目标平面或在平面外
错过半径。

若成立应看到：plane miss 的 terminal speed ratio、终端纵向 offset 或接近终点时的
速度衰减显著差于 matched success；差距应达到下文 Gate 的数值门槛。

若失败应看到：success 与 miss 的 terminal speed ratio 分布接近，且纵向 offset
不能解释分类。

最便宜证伪：用已有 trace 的 `guidance_terminal_phase`、terminal speed ratio、
`goal_distance_m` 和 `signed_goal_tangent_offset` 做 matched comparison；不训练。

### H4：终端执行器权限/饱和限制捕获

机制：终端需要较大的 rudder/stern 修正，但 saturating action 或 rate clip 使策略
无法完成最后几十厘米到数米的修正。

若成立应看到：plane miss 在 terminal window 的 material saturation、rudder/stern
limit 或 yaw/pitch error 明显高于 matched success。

若失败应看到：两组的 terminal saturation 与 actuator error 无实质差异。

最便宜证伪：trace 级比较 terminal window 的 saturation、action rate clip、rudder/
stern fraction 和姿态误差；不训练。

### H5：当前 success 定义需要同时报告协议成功和真实状态捕获

机制：任务的部署可用性依赖 p_nav，但研究结论必须同时报告 true-state capture，否则
会把估计器行为误写成控制能力。

若成立应看到：两类 success 的差距在多个 arm/scenario 中稳定存在，并且 p_nav-only
episode 的终止距离分布与真实捕获 episode 不同。

若失败应看到：两类 success 基本一致，且 p_nav-only episode 可以忽略。

最便宜证伪：对 8 个已有 summary 逐 episode 统计，不需要新 rollout。

## 3. 数据与隔离

- 主数据：D3、G26-P2、A1、H3-SR 在 `default` / `strict125` 的 40-seed dev summary；
- seeds：固定 `20–59`；每个 summary 必须恰好包含这 40 个 seed；
- held-out：`100–149` 完全冻结，本 Goal 不读取、不评估；
- 评估协议：`endpoint_stop`、`realistic_calibrated`、CPU/float64、`path_dim=8`；
- baseline：D3 和 G26-P2 为主对照；A1/H3-SR 只作失败控制；
- 禁止：训练新策略、改 checkpoint、改 eval split、改 goal radius、改 success 阈值；
- 禁止：用 held-out 或 best-checkpoint 事后挑选；
- oracle：本 Goal 不需要 privileged controller；任何 p_nav/eta 比较都只作诊断，不作
  部署方案。

## 4. 方法

### Step 1：summary 级只读分解

新增只读脚本 `scripts/analyze_pc_goal28_terminal.py`，读取 8 个 summary，输出：

- `success_protocol`、`success_eta`、`pnav_only_success`；
- `plane_miss`、`timeout`、`safety` 计数及互斥分类；
- `miss_distance_at_plane_m` 分布（mean / median / p90 / min / max）；
- `final_goal_distance_m` 按 outcome 分组；
- pre-goal/full RMSE、material saturation、terminal speed ratio；
- 每个 seed 的 outcome 转移，与 D3/P2 paired comparison；
- 每个 arm/scenario 的配置、checkpoint hash 和 summary SHA256。

该步骤不运行环境、不训练、不改任何现有 summary。

### Step 2：terminal trace 级分解

只对 D3 和 G26-P2 的 `default` / `strict125` 运行固定 dev sweep；每个 seed 一条
trace，trace 只写新的 Goal28 目录。trace 至少包含：

- `eta`、`p_nav`、`goal_ned_m`、`goal_tangent_ned`；
- `signed_goal_tangent_offset`（p_nav 和 eta 各一份）；
- `goal_distance`（p_nav 和 eta 各一份）；
- `terminal_kind`、`endpoint_entry_reached`；
- 终端 2 s 的 speed ratio、yaw/pitch error、rudder/stern/rpm、material saturation；
- 现有 `guidance_terminal_phase` 和 `guidance_terminal_speed_ratio`。

所有 trace 字段必须 finite；若某个旧 checkpoint 不兼容新 trace 字段，只允许增加
只读字段，不允许改策略或评估语义。

### Step 3：matched comparison

对每个 scenario 固定以下比较：

1. `plane_miss` vs `success_eta`；
2. `pnav_only_success` vs `success_eta`；
3. D3 vs G26-P2 的同 seed、同 outcome 配对；
4. A1/H3-SR 只做 summary 级失败控制，不进入 trace matched comparison。

比较量包括 terminal tangent offset、horizontal-normal offset、vertical offset、
speed ratio、saturation、姿态误差和 p_nav-eta 误差。不允许用 outcome 重新选 seed；
seed 集合在 trace 前固定为完整 `20–59`。

### Step 4：机制裁决

按“数据 → 标签 → 评测 → baseline → 架构/执行器 → 文献”的顺序解释结果。若 H1–H4
没有任何一个达到 material 门槛，则写 `DIAG_INCONCLUSIVE`，并按 AGENTS.md 进入
autoresearch blocker，而不是直接开新训练。

外部论文只在出现明确机制假设后补 reading note；本 Goal 不先下载无关文献来替代
本地证据。

## 5. 执行计划

- [x] Step 0：记录 8 个 summary 的路径、SHA256、checkpoint SHA、seeds 和协议；
- [x] Step 1：实现 `analyze_pc_goal28_terminal.py` 与最小单测；
- [x] Step 2：在 `/tmp` 或新 `artifacts/lowlevel/goal28_terminal_*` 目录汇总 8 个
      summary，检查 40/40 seeds 与互斥 outcome；
- [x] Step 3：给 `evaluate_pc_ppo.py` 增加只读 terminal trace 字段，并做 1-seed smoke；
- [x] Step 4：在隔离目录运行 D3/P2 的 `default`/`strict125` 40-seed trace sweep；
- [x] Step 5：运行 matched decomposition，输出 JSON/Markdown；
- [x] Step 6：按 §6 Gate 写 result、更新 `research/STATE.md`/ledger；
- [x] Step 7：运行 checker；只在用户明确要求时 commit/push。

每步的验证和提交要求：

- Step 1：`pytest -q tests/test_pc_goal28_terminal.py`；只 add 脚本、测试和 goal 文档；
- Step 4：检查每个 trace 的 seed/场景/长度和 finite；不提交 `.npz`、日志或大型输出；
- Step 6：运行 `check_research_ledger.py` 和 `git diff --cached --check`；
- 任何 commit 只 add 本次相关文件，禁止 `git add -A`。

## 6. Gate

### 6.1 数据与评测有效性

| check | 通过条件 |
|---|---|
| seed 集合 | 8 个 summary 均为 `20–59`，无 held-out |
| 协议 | 全部 `endpoint_stop`、`realistic_calibrated`、CPU/float64 |
| outcome 互斥 | 每个 episode 恰好属于 success/plane_miss/timeout/safety 之一 |
| 数值 | 主指标 finite；允许 `miss_distance`/`terminal_longitudinal` 对非对应 outcome 为 null |
| trace 完整性 | D3/P2 的 4 个 scenario 各 40 条 trace，seed 与 summary 对齐 |
| 无训练 | 0 个新策略、0 次 checkpoint 更新、0 次 held-out 评估 |

### 6.2 机制门槛（测量前冻结）

以下门槛用于判断机制是否 material；不是“候选策略通过/失败”的性能门禁。

| 假设 | material 条件 |
|---|---|
| H1 估计/协议错配 | 任一 scenario 的 `pnav_only_success / success_protocol >= 0.20` |
| H2 终端几何 | plane miss 的 median miss distance `>= 1.25 * goal_radius`，且 horizontal-normal 或 vertical 分量占主导 |
| H3 速度/纵向 | plane miss 的 median terminal speed ratio 比 matched success 高 `>= 0.10`，或 median tangent offset 比 matched success 高 `>= 0.5 * goal_radius` |
| H4 执行器权限 | terminal window 的 material saturation 或 rudder/stern limit 在 miss 组比 success 组高 `>= 0.05`，并伴随姿态误差差 `>= 0.05 rad` |
| H5 指标有效性 | H1 成立且至少两个 arm/scenario 出现同方向差异 |

### 6.3 裁决

| 观察 | 判读 | 下一步 |
|---|---|---|
| 数据/协议/trace 不完整 | `INVALID` | 只修数据链路并重跑同一诊断，不改阈值 |
| 至少一个 H1–H4 material | `DIAG_COMPLETE` | 写 result，并按机制预注册 Goal29 干预；不得直接训练 |
| 仅 H5 material | `DIAG_COMPLETE_METRIC` | 先统一 success/eta 报告口径，再重新诊断 |
| 没有 H1–H5 material，但数据完整 | `DIAG_INCONCLUSIVE` | 写 blocker、至少 2 个可证伪新假设、读 1 篇相关论文，再决定 pivot/stop |
| 任何 held-out 被读取或检查点被改写 | `INVALID` | 停止本 Goal，恢复隔离协议 |

本 Goal 不产生“策略 PASS”。如果后续要测试 terminal capture 控制器或任何 residual，
必须在 Goal29 中重新预注册候选、阈值、seed 和 held-out 规则。

## 7. Git

禁止 `git add -A`。只 add 本 Goal 相关文件，提交前运行：

```bash
cd /home/bubble/水下机器人/settled_局部规划器_反应式
git add doc/goals/goal28_terminal_capture_plane_miss_20261109.md \
  scripts/analyze_pc_goal28_terminal.py \
  tests/test_pc_goal28_terminal.py
git diff --cached --check
.venv/bin/python ../scripts/check_research_ledger.py \
  --goal Goal28 \
  --goal-doc doc/goals/goal28_terminal_capture_plane_miss_20261109.md --strict
```

如果 Step 3 修改了 `scripts/evaluate_pc_ppo.py` 或 trace 测试，再单独 add 这些文件。
不要提交 `artifacts/`、trace `.npz`、日志、checkpoint 或大数据文件。

## 8. Goal 提示词

```text
你现在执行 Goal28：任务级终点捕获 / 过平面失败诊断。

先读：
- doc/goals/goal28_terminal_capture_plane_miss_20261109.md
- doc/goals/goal27_result_20261109.md
- doc/goals/goal27_h3_sr_result_20261109.md
- research/STATE.md

硬约束：
1. 本 Goal 先做只读诊断，不训练、不 warm-start、不改 checkpoint、不碰 held-out 100–149；
2. 固定 dev 20–59、endpoint_stop、realistic_calibrated、CPU/float64、path_dim=8；
3. 不把 p_nav success 当成 true-state success；两者必须同时报告；
4. 不因诊断结果直接开 residual PPO；若需要干预，另写 Goal29 预注册；
5. 失败/不明确时按 AGENTS.md 进入 autoresearch blocker，不换 seed 救结果。

执行顺序：
- 先完成 summary 级分解和数据有效性检查；
- 再补 terminal trace 字段并做 1-seed smoke；
- 最后跑 D3/P2 的固定 trace sweep 和 matched comparison；
- 按 §6 Gate 写 result、ledger、STATE，并运行 checker。
```
