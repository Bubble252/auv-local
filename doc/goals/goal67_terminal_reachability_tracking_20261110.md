# Goal67：terminal reachability 与 command-to-velocity tracking 审计

**日期**：2026-11-10<br>
**状态**：`COMPLETED_TERMINAL_AUTHORITY_LIMITED`<br>
**上游**：Goal27 `H3_SR_FAIL_STOP`、Goal32 `AUTHORITY_LIMITED_SUPPORTED`、Goal64 `GEOMETRY_SUPPORTED`、Goal65 `DWELL_NOT_SUFFICIENT`、Goal66 `TERMINAL_GEOMETRY_UNRESOLVED`

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
  check_command: python ../scripts/check_research_ledger.py --goal Goal67 --goal-doc doc/goals/goal67_terminal_reachability_tracking_20261110.md --strict
```

## 1. 背景与问题

Goal32 在旧 checkpoint 的 20 m terminal entry 上发现“当前闭环展示的垂直闭合权限
不足”，但它没有覆盖当前 Goal59 checkpoint、最近 true-`eta` 点，也没有拆开
command 与实际地速跟踪。Goal66 又显示当前 lost 组的 corrected command 已高度对准
goal，所以“command 方向错”不是主因。

本 Goal 只问：

**在最近 true-`eta` 点之后，失败是因为所需闭合速度超出当前执行器/动力学实际展示的
authority，还是 command 可达但实际地速没有跟上，还是捕获边界/估计器几何？**

本 Goal 不训练、不扫 reward/gamma/seed/scale、不改 controller 或 checkpoint。
唯一新增计算是给当前 frozen-checkpoint 评估加只读 rich trace 字段，并重跑与
Goal65 完全相同的 10 个 candidate 评估，以获得真实 `nu`、地速、姿态和执行器状态。

## 2. 可证伪假设

### H67-A：terminal 所需闭合速度超出实际展示 authority

对 lost true 中 true `eta` 最小距离仍大于 `r_tight` 的 episode（`eta-outside lost`）：

```text
T_rem_s = max((terminal_step - i*) * dt_s, 0.20)
d_rem_m = max(eta_distance(i*) - r_tight, 0.0)
required_closure_mps = d_rem_m / T_rem_s

closing_k_mps = -(eta_distance[k+1] - eta_distance[k]) / dt_s
demonstrated_closure_p95_mps = percentile_95(closing_k_mps in final 51 steps)
closure_deficit_mps = required_closure_mps - demonstrated_closure_p95_mps
```

支持条件（pooled lost/retained，仅 `eta-outside lost` 参与 A 的支持判定）：

```text
n_eta_outside_lost >= 3
lost deficit fraction (closure_deficit_mps > 0) >= 0.75
lost closure_deficit median >= 0.10 m/s
lost closure_deficit median - retained closure_deficit median >= 0.10 m/s
```

若 true `eta` 已经进入 `r_tight`，该 episode 不属于 H67-A 的可证伪集合；它留给
H67-C 或估计器/终止边界解释。

### H67-B：command 可达但实际地速方向没有跟上

用 rich trace 的实际 ground velocity NED、command physical direction、true
eta-to-goal LOS 计算：

```text
actual_goal_cosine = cos(actual_ground_velocity_ned, LOS(eta, goal))
command_goal_cosine = cos(command_direction, LOS(eta, goal))
actual_command_cosine = cos(actual_ground_velocity_ned, command_direction)
```

支持条件（pooled lost/retained，最近 true-`eta` 点）：

```text
lost command_goal_cosine median >= 0.95
lost actual_goal_cosine median < 0.90
lost actual_ground_speed median >= 0.10 m/s
retained actual_goal_cosine median - lost median >= 0.05
```

若实际地速方向已经和 command/goal 一致，H67-B 不支持，不能把失败写成速度跟踪滞后。

### H67-C：捕获边界/估计器只差一个薄壳

支持条件（pooled lost）：

```text
p_nav_only_capture_fraction >= 0.40
abs(eta_min_distance_median_m - r_tight) <= 0.15 m
```

该假设只说明 true `eta` 已接近捕获边界但 `p_nav`/终止几何截断，不单独证明估计器
校准误差；若 H67-A 或 H67-B 同时支持，按 A → B → C 的因果优先级报告。

## 3. 数据与隔离

| 项目 | 固定值 |
|---|---|
| baseline | Goal60 的 10 个 trace，dev `20–59` |
| candidate checkpoint | Goal59 `240/752/880/1008` |
| candidate runs | 4 checkpoint × default/strict125 policy + 2 zero-residual |
| evaluation | 与 Goal65 `_dwell50_m025` 完全相同：CPU/float64、`endpoint_stop`、dwell=50、`r_tight = r0 - 0.25 m`、plane guard `r0` |
| new outputs | `artifacts/goal67_terminal_reachability/`（服务器）与对应本地分析目录 |
| rich trace | 新增只读字段：time、eta RPY、nu、ground velocity body/NED、actual actuator、command physical、command reachability、terminal phase、crab angle |
| held-out | `100–149` 关闭 |

rich trace 不改变动作、reward、终止、reset、seed 或 controller。评估后必须与 Goal65
candidate 逐 seed 做等价性检查：

```text
success, terminal_kind, steps, goal_plane_miss, timeout, safety,
terminal_p_nav_distance_m, terminal_eta_distance_m, terminal_signed_goal_offset_m
```

任一字段 mismatch 则整体 `INVALID_DIAGNOSTIC`，不得继续用部分 run 下结论。

## 4. 计算方法

对每个 candidate seed：

1. 从 Goal60 baseline 选出 `success & terminal_eta_capture` 的 true success；
2. 按 seed 对齐 Goal67 rich candidate episode；
3. 在 candidate 最后 51 行中找 true `eta` 到 goal 的最小距离点 `i*`；
4. 计算 `T_rem`、`d_rem`、`required_closure_mps`；
5. 从 rich trace 的 true eta distance 序列计算 `demonstrated_closure_p95_mps`；
6. 在 `i*` 读取实际地速 NED、command physical、true LOS；
7. 对 lost/retained 分别汇总中位数、计数和逐 seed 值；
8. 用预注册阈值输出 H67-A/B/C 与总体 classification。

实际地速方向的单位向量由 rich trace 的 `ground_velocity_ned_mps` 恢复；若速度小于
`0.10 m/s`，方向余弦记为 `NaN` 并从对应 median 中排除，同时单独报告低速计数。
不能把静止时的不稳定方向当作 tracking 证据。

## 5. 执行计划

- [x] Step 0：复核 Goal32/65/66 数据、config、trace schema 和服务器状态；
- [x] Step 1：给 evaluator 加 `--rich-trace` 只读字段，默认关闭；
- [x] Step 2：写 Goal67 runner，复用 Goal65 的 10 个 frozen-checkpoint 设置；
- [x] Step 3：实现只读 analyzer 与合成 trace unit tests；
- [x] Step 4：跑 10 个 rich candidate，检查 Goal65 逐 seed 等价性；
- [x] Step 5：生成 H67 JSON/Markdown，写 result/blocker/ledger/STATE；
- [x] Step 6：strict checker、全量 pytest、Ruff、`git diff --cached --check`；
- [ ] Step 7：只 add 相关文件；用户明确要求后才 commit/push。

运行命令（服务器）：

```bash
cd /data/yhx/underwater-robotics/src/plan_p1/goal60_action_reward_terminal_readonly_audit_20261110
export PYTHONPATH=src:/data/yhx/underwater-robotics/src/plan_p1/settled_全局规划器/src
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=/data/yhx/underwater-robotics/envs/conda_envs/oceansim/bin/python \
ROOT=$PWD OUT_ROOT=$PWD/artifacts/goal67_terminal_reachability \
  bash scripts/run_pc_goal67_terminal_reachability_20261110.sh
```

分析命令：

```bash
python scripts/analyze_pc_goal67_terminal_reachability.py \
  --baseline-root artifacts/traces \
  --candidate-root artifacts/goal67_terminal_reachability \
  --reference-root artifacts/goal65_plane_miss_dwell \
  --output artifacts/goal67_terminal_reachability/analysis/goal67.json \
  --markdown artifacts/goal67_terminal_reachability/analysis/goal67.md
```

## 6. Gate（测量前冻结）

| 观察 | 判定 | 下一步 |
|---|---|---|
| H67-A 支持 | `TERMINAL_AUTHORITY_LIMITED` | 预注册固定参数的速度/舵权限与时间预算候选；不改 reward/算法 |
| A 不支持且 H67-B 支持 | `TERMINAL_TRACKING_LAG` | 预注册 command-to-velocity tracking / controller bandwidth 诊断 |
| A/B 不支持且 H67-C 支持 | `CAPTURE_REGION_BOUNDARY` | 预注册 capture-region/estimator 边界诊断 |
| A/B/C 均不支持 | `TERMINAL_MECHANISM_UNRESOLVED` | 转更细的动力学辨识或 terminal target 几何，不扫训练参数 |
| 数据/等价性失败 | `INVALID_DIAGNOSTIC` | 只修测量实现，不改阈值 |

若 default 与 strict125 分类不同，总状态记为 `MIXED_TERMINAL_MECHANISM`，优先采用
strict125 机制，并保留 default 反例。

## 7. 算力策略

- 10 个 candidate 全部并行，`MAX_PARALLEL=10`，每 run `num_envs=40`；
- 每进程 `OMP_NUM_THREADS=1`、`MKL_NUM_THREADS=1`；rich trace 只增加输出字段；
- 不增加 seed、batch、UTD 或训练预算；不占用 GPU 训练；
- 若后续进入训练，按 seed 分空闲 GPU 并行；扩大 `num_envs`/batch 必须另开预注册，
  固定总 transitions、UTD、replay 比例、checkpoint 选择和评估协议；
- 不因 rich trace 改变 dtype/device，正式 Gate 仍为 CPU/float64。

## 8. 明确禁止

- 不重开 Goal27 的 residual/reward/gamma/seed 搜索；
- 不训练，不改 controller/reward/config/checkpoint；
- 不扫 authority、speed profile、capture radius 或 reward 权重；
- 不打开 held-out `100–149`；
- 不启动 SAC/REDQ/DroQ；
- 不 `git add -A`，不自动 commit/push。

## 9. Goal 提示词

```text
你现在执行 Goal67：terminal reachability 与 command-to-velocity tracking 审计。

先读：
- doc/goals/goal67_terminal_reachability_tracking_20261110.md
- doc/goals/goal66_result_20261110.md
- doc/goals/goal32_result_20261109.md

硬约束：
1. 复用 Goal59 frozen checkpoint 和 Goal65 的 10 个评估设置，不训练、不扫参数；
2. rich trace 只增加只读字段，必须与 Goal65 逐 seed 等价；
3. dev 20-59，held-out 100-149 关闭；
4. 先冻结 H67-A/B/C 阈值，运行后不回改；
5. 不把实际展示 authority 写成硬件绝对上限；不启动 SAC/REDQ/DroQ；
6. 禁止 git add -A；不自动 commit/push。

执行顺序：
1. 加 rich trace flag 和 runner；
2. 实现 analyzer/unit tests；
3. 服务器并行跑 10 个 candidate；
4. 检查 Goal65 等价性，生成 H67 判读；
5. 写 result/blocker/ledger/STATE，跑 pytest/checker/diff check。
```
