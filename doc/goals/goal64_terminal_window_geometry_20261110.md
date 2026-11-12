# Goal64：Goal63 candidate 的 terminal-window 几何诊断

**日期**：2026-11-10<br>
**状态**：`GEOMETRY_SUPPORTED`<br>
**上游**：Goal63 `PARTIAL_TERMINAL_SUPPRESSION`<br>
**服务器**：`yhx@152.136.183.51 -p 43127`，8×RTX 4090，384 CPU

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
  check_command: python ../scripts/check_research_ledger.py --goal Goal64 --goal-doc doc/goals/goal64_terminal_window_geometry_20261110.md --strict
```

## 1. 背景与问题

Goal63 已经在 frozen Goal59 checkpoint 上证明：把 GOAL 半径从 `r0` 收紧到
`r0 - 0.25 m`、同时保留原 `r0` 的 plane-miss guard，可以在 default 上显著
减少 false success，但 strict125 的 true-success retention 只有 `0.739`，且
4 个 checkpoint 全部未通过。零残差对照也出现同类 terminal capture 损失。

本 Goal 不是重跑 Goal63，也不是救 Gate。它只回答一个机制问题：

**strict125 的 true capture 是在哪里丢掉的？**

重点区分三种解释：

1. 先穿过 goal plane，且横向偏移大于收紧后的半径余量；
2. `p_nav` 与 `eta` 的 terminal 口径偏差；
3. residual 幅度或动作饱和在丢失组显著更大。

### 1.1 NoisyNet 先验结论

NoisyNet 不是当前主因，不能作为本 Goal 的开放解释：

- Goal59 的正式评估使用 `evaluate=True`，即 NoisyNet 的 mean-weight 网络；
- Goal59 G12 的 8/8 deterministic action saturation 均为 `1.0`，`noise/det`
  中位数只有 `0.0193`，`noise>det` 只有 seed `1008` 明显升高；
- 关闭噪声的 N12 仍然在 `99.9%` 状态饱和；
- Goal63 的 forced zero-residual 也复现了 terminal capture 损失，说明损失不依赖
  residual 网络，更不可能由噪声头单独解释。

因此本 Goal 不新增噪声开关、不改 eval-time noise scale，也不启动 SAC/REDQ/DroQ。

## 2. 可证伪假设

### H64-A：strict125 的 true loss 由 goal-plane crossing 几何触发

若成立应看到：

- strict125 policy 的 baseline true-success 丢失 episode 中，至少 `75%` 的
  candidate terminal kind 是 `GOAL_PLANE_MISS`；
- 这些 episode 在 terminal 前的 plane crossing 处，垂直于 goal tangent 的偏移
  中位数至少 `0.20 m`；
- 大多数丢失 episode 的 minimum `p_nav` 距离大于 candidate GOAL 半径
  `r_tight = 1.00 m`，且 terminal 前没有进入过 `r_tight`。

若失败应看到：丢失主要来自 timeout、safety、out-of-bounds，或 terminal 时
`p_nav` 已经进入 `r_tight` 却被错误标记。

最便宜 falsification：只读 candidate terminal trace，不训练、不改控制。

### H64-B：丢失组的 residual/action 饱和没有显著高于保留组

若成立应看到：

- 丢失组与保留组的最后 `100` 步（约 `2 s`）residual L2 ratio、action saturation
  fraction、material saturation rate 的差异小于 seed 噪声，或方向不能解释 plane
  crossing；
- residual scale clip / rate clip 在丢失组没有系统性增加；
- forced zero-residual 的丢失 episode 仍表现为 plane crossing。

若丢失组显著更饱和、更早打满执行器，或 residual clip 与 plane miss 强相关，则
H64-B 被证伪，下一步应查动作参数化、reward 和 terminal capture 顺序，而不是
继续训练更大的 residual。

最便宜 falsification：复用同一 trace 的 last-100-step 窗口；不新增训练。

### H64-C：default zero-residual 的丢失与 policy 丢失共享 terminal geometry

若成立应看到：

- zero-residual 的 baseline true-success 丢失 episode 同样以
  `GOAL_PLANE_MISS` 为主；
- zero-residual 与 policy 丢失组的 perpendicular offset、`p_nav/eta` 距离差、
  到 plane 的时间步分布有实质重叠；
- 丢失不是只出现在某一个 checkpoint 或单一 seed。

若 zero-residual 的丢失由完全不同的 terminal kind 或明显不同的几何造成，则
H64-C 不成立，不能把 policy 失败简单归因于通用 terminal geometry。

## 3. 数据与隔离

| 项目 | 固定值 |
|---|---|
| baseline | Goal60 的 10 个 trace run，dev `20–59` |
| candidate | Goal63 的 4 checkpoint × 2 scenario policy + 2 zero-residual |
| checkpoint | Goal59 `240/752/880/1008` 的 `final.pt` |
| scenario | default r0 `2.00 m`；strict125 r0 `1.25 m` |
| candidate GOAL | `r0 - 0.25 m`：`1.75/1.00 m` |
| plane-miss guard | 原 `r0`：`2.00/1.25 m` |
| action mode | policy + forced zero-residual |
| 精度 / 终止 | CPU / float64；`endpoint_stop` |
| 训练 / held-out | 不训练；`100–149` 关闭 |

baseline trace 直接复用 Goal60；candidate 只新增 `--trace-output` 的只读 per-step
记录并重新 rollout。reward、observation、controller、checkpoint、seed、dtype、
device、GOAL 半径和 plane-miss guard 均与 Goal63 完全相同。

## 4. 方法

### 4.1 Trace 记录

复用 `scripts/evaluate_pc_command_residual_offpolicy.py` 的 `--trace-output`，
不另造轨迹格式。每个 candidate run 输出：

```text
trace.npz
trace_meta.json
episodes.json
summary.json
```

需要字段：

- `p_nav_ned`、`eta_ned`、`goal_ned`、`goal_tangent_ned`；
- `goal_distance`、`signed_offset`、`terminal_kind`、`done`；
- `action`、`residual`、`gate`、`scale_clipped`、`rate_clipped`、
  `material_saturated`；
- reward 分量，尤其是 `reward_terminal_progress`、`reward_goal_plane_miss`、
  `reward_saturation`、`reward_terminal`、`reward_total`；
- `step`、`seed`、`episode_id`。

### 4.2 Terminal-window 指标

对每个 seed，按 baseline 的 `success & terminal_eta_capture` 定义 baseline true
success。candidate 是否保留只看 GOAL success 标签，不改定义。

在 candidate terminal 前的最后 `100` 步和 terminal row 上计算：

- terminal kind 名称、`p_nav`/`eta` 到 goal 的距离、两者之差；
- signed plane offset 和垂直于 goal tangent 的 offset；
- minimum `p_nav` 距离、是否曾经进入 `r_tight`；
- 从最后一次负 offset 到 terminal 的步数；
- 由相邻 `p_nav` 差分估计的速度方向，以及相对 `goal_tangent` 的 approach
  cosine；
- residual/action 的 mean L2、饱和比例、gate 活跃比例、clip 次数；
- terminal reward 分量和 last-100-step mean reward。

这些量只用于诊断，不反馈到 reward、controller 或 checkpoint 选择。

## 5. 判读规则（测量前冻结）

本 Goal 没有“救 Goal63”的正式成功率 Gate。以下规则只决定机制归因和下一步。

| hypothesis | 支持条件 | 失败/反向条件 |
|---|---|---|
| H64-A | strict125 policy 丢失组 `GOAL_PLANE_MISS >= 0.75`，perpendicular offset 中位数 `>= 0.20 m`，最低 `p_nav > r_tight` | plane-miss 比例 `< 0.50`，或丢失由 timeout/safety 主导 |
| H64-B | 丢失组与保留组的 last-100-step residual/action 指标差异不能解释 plane crossing；zero-residual 也同向 | 丢失组 residual L2 或 action saturation 中位数 `>= 1.25×` 保留组，且与 plane miss 同向 |
| H64-C | zero-residual 丢失组 `GOAL_PLANE_MISS >= 0.75`，且其 perpendicular offset 与 policy 丢失组中位数差 `<= 0.10 m` | zero-residual 由不同 terminal kind 主导，或几何分布明显不重叠 |

所有比例同时给出 seed-level bootstrap 95% CI；样本数很小，点估计和逐
checkpoint 方向必须一起报告，不能只报 pooled 均值。

## 6. 裁决与下一步门闩

| 观察 | 状态 | 下一步 |
|---|---|---|
| H64-A 支持且 H64-B 不显示 residual 主导 | `GEOMETRY_SUPPORTED` | 新 Goal 预注册 terminal plane-aware/`eta`-aware 捕获规则；仍先 eval-only，不训练 |
| H64-B 被证伪，丢失组明显更饱和 | `ACTION_PARAMETERIZATION_SUSPECTED` | 先查动作参数化、reward 和 terminal capture 顺序，不换算法 |
| H64-A/C 都不支持，terminal kind 混杂 | `GEOMETRY_NOT_SUFFICIENT` | 写 blocker，重新做 estimator/guidance 诊断；不在本 Goal 内改阈值 |
| trace/标签/metadata 不完整 | `INVALID` | 只修测量实现，不修改判读阈值 |

## 7. 算力与并行合同

本 Goal 是 eval-only，不需要 GPU 训练，但不能串行等待：

- 10 个 candidate run 全部并行；每个 run 内部 `num_envs=40`，一次覆盖完整 dev seed；
- `MAX_PARALLEL=10`，每进程 `OMP_NUM_THREADS=1`、`MKL_NUM_THREADS=1`；
- 服务器有 384 CPU、503 GiB 内存，10 个 CPU/float64 进程不会造成资源瓶颈；
- 每个 run 写独立 trace 目录，最后按 `(scenario, checkpoint, action_mode, seed)` 合并；
- baseline trace 复用 Goal60，不重复生成；
- 不在 eval 中增加 seed 重复、不改变 dtype/device/终止规则；
- GPU `0–3` 空闲也不把 float64 CPU 评估迁到 GPU，避免数值协议变化。

后续训练若需要提高迭代速度，按 seed 分 GPU 并行；探索型 arm 可预注册更大的
`num_envs` 和 batch（例如每卡 128/256 env、总 batch 1024），但必须固定总
transitions、UTD、replay 比例、评估协议和 checkpoint 选择，不能静默改变正式
Gate。该规则只适用于新预注册，不追溯修改 Goal63/64。

## 8. 执行计划

- [x] Step 0：复核 Goal63 result/blocker、trace 字段和服务器资源；
- [x] Step 1：写 Goal64 runner，复用 Goal63 evaluator 并开启 `--trace-output`；
- [x] Step 2：本地单测、Ruff、smoke，确认 trace schema 和 metadata；
- [x] Step 3：服务器同步代码，1 个 checkpoint × 40 seed 复现检查；
- [x] Step 4：10 路并行运行 10 个 candidate run；
- [x] Step 5：写 terminal-window 分析器，输出 JSON/MD；
- [x] Step 6：按 H64-A/B/C 判读，写 result、blocker/reading note、LEDGER、STATE；
- [x] Step 7：运行 strict checker、全量测试、Ruff、`git diff --check`；
- [ ] Step 8：只 add 本次相关文件；用户明确要求后才 commit/push。

运行命令：

```bash
export PYTHONPATH=src:/data/yhx/underwater-robotics/src/plan_p1/settled_全局规划器/src
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
MAX_PARALLEL=10 bash scripts/run_pc_goal64_terminal_window_20261110.sh
```

验证命令：

```bash
python3 scripts/analyze_pc_goal64_terminal_window_geometry.py \
  --baseline-root artifacts/traces \
  --candidate-root artifacts/goal64_terminal_window \
  --output artifacts/goal64_terminal_window/analysis/goal64.json \
  --markdown artifacts/goal64_terminal_window/analysis/goal64.md
```

commit / push 只使用显式文件列表，禁止 `git add -A`：

```bash
git add doc/goals/goal64_terminal_window_geometry_20261110.md \
  doc/goals/goal64_result_20261110.md \
  scripts/run_pc_goal64_terminal_window_20261110.sh \
  scripts/analyze_pc_goal64_terminal_window_geometry.py \
  tests/test_pc_goal64_terminal_window.py \
  research/blockers/2026-11-10-goal64-terminal-geometry.md \
  research/paper_reading/2026-11-10-goal64-terminal-geometry.md \
  research/LEDGER.jsonl research/STATE.md ../research/STATE.md
git diff --cached --check
git commit -m "research: diagnose goal64 terminal-window geometry"
git push
```

## 9. 明确禁止

- 不训练、不改 reward/observation/controller/config/checkpoint；
- 不重跑 Goal63 的 margin 扫描，不放宽 retention 阈值；
- 不把 Goal64 结果写成新的正式成功率 PASS；
- 不启动 SAC/REDQ/DroQ，不重开 Goal27/58/59；
- 不打开 held-out `100–149`；
- 不 `git add -A`，不自动 commit/push。

## 10. 后续算法升级门闩

当前证据不支持换算法。若未来出现新的、可复现的失败，再按下面规则开新 Goal：

- 若失败明确是 optimization/exploration（deterministic policy 未收敛、动作分布探索
  不足、critic 稳定但 actor 不改进），新开预注册 Goal，用 SAC 做同协议对照；
- 若失败明确是 sample efficiency（同总 transitions 下学习曲线显著慢于 baseline，
  且增加并行 env 后仍不改善），再考虑 REDQ/DroQ；
- 任何算法对照都必须保持同一 checkpoint 选择、seed 集、评估 split、终止协议和
  总训练预算，不得用 Goal64 的失败直接授权换算法。

## 11. Goal 提示词

```text
你现在执行 Goal64：Goal63 candidate 的 terminal-window 几何诊断。

先读：
- doc/goals/goal64_terminal_window_geometry_20261110.md
- doc/goals/goal63_result_20261110.md
- research/blockers/2026-11-10-goal63-strict125-capture-loss.md
- scripts/evaluate_pc_command_residual_offpolicy.py

硬约束：
1. 只读 trace；不训练、不改 reward/observation/controller/checkpoint；
2. 复用 Goal63 的 r0-0.25 m 与 plane-miss guard，不扫 margin；
3. dev 20-59，held-out 100-149 关闭；
4. 10 个 candidate run 并行，CPU/float64/endpoint_stop；
5. 重点是 plane crossing 几何、p_nav/eta 偏差和 residual/action saturation；
6. 不启动 SAC/REDQ/DroQ，不重开 Goal27/58/59。

执行顺序：
- runner + smoke；
- 10 路并行 trace；
- terminal-window 分析；
- 按 H64-A/B/C 判读，写 result/ledger/STATE；
- strict checker、测试、Ruff、diff check；
- 不自动 commit/push。
```

## 12. 执行结果（2026-11-10）

最终状态 `GEOMETRY_SUPPORTED`。10 个 candidate run 用 `num_envs=40` 全并行完成，
单 run 与 Goal63 逐 seed terminal 结果完全一致（0 mismatch）。

- H64-A `SUPPORTED`：strict125 policy 的 6 个 true-loss 全部为
  `GOAL_PLANE_MISS`，minimum `p_nav` 中位数 `1.172 m` 高于 `r_tight=1.00 m`，
  perpendicular offset 中位数 `1.065 m`；
- H64-B `NOT_FALSIFIED`：lost/retained 的 last-100-step residual L2 中位数为
  `0.351/0.352`，material saturation `0.210/0.340`，scale/rate clip 均为 `0`；
- H64-C `SUPPORTED`：5/5 zero-residual 丢失为 plane miss，default zero 与
  policy 的 perpendicular offset 中位数只差 `0.091 m`。

因此根因是 terminal 接近几何与 plane-miss guard 的交互，不是 residual 大小、
NoisyNet 或样本效率。未训练、未改 margin、未打开 held-out。下一步是新预注册
eval-only plane-aware/`eta`-aware terminal rule 诊断；不启动 SAC/REDQ/DroQ。

详见：

- `doc/goals/goal64_result_20261110.md`
- `research/blockers/2026-11-10-goal63-strict125-capture-loss.md`
- `research/paper_reading/2026-11-10-goal64-terminal-geometry.md`
