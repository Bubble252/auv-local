# Goal74：terminal capture margin 与 p_nav-eta 方向偏差的只读审计

**日期**：2026-11-10<br>
**状态**：`MARGIN_UNUSABLE_BIAS_SUPPORTED`<br>
**上游**：Goal73 `COMMAND_DEMAND_NOT_SUPPORTED` / `PIVOT_TO_CAPTURE_AUDIT`

结果：`doc/goals/goal74_result_20261110.md`

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
  check_command: .venv/bin/python ../scripts/check_research_ledger.py --goal Goal74 --goal-doc doc/goals/goal74_terminal_capture_margin_bias_20261110.md --strict
```

## 1. 背景

Goal73 Step 1 证伪了 command-demand/曲率假设，并发现 false-success 的
共同结构：

- 所有 false-success 都是 `sphere` 捕获；
- `p_nav` 距离刚好在 tight radius 内，余量只有 `0.01–0.04 m`；
- true `eta` 距离仍在半径外；
- terminal `p_nav-eta` 误差中位 default `0.4475 m`、strict125 `0.6131 m`。

这说明 capture 边界附近的 `p_nav` 估计误差可以翻转 success 标签。当前
需要先回答：这是**边界 margin 问题**，还是**方向性 bias 问题**，还是两者
都有。答案决定下一步是改 capture 口径、做估计偏差修正，还是回去查动作
参数化/奖励。

### 1.1 口径修正记录（正式结果重跑前）

第一次 provisional run 暴露了 analyzer 的半径混用：`summary["eval_goal_radius_m"]`
是 tight radius（`r0 - 0.25 m`），而 `terminal_eta_capture` 使用的是
`environment_config.plant.goal_radius_m`（protocol radius）。因此 provisional
的 `true_success` 被低估，`false_success` 被高估；该次 JSON/MD 不追认为
Goal74 结果，也不据此改主线。

正式重跑前固定以下修正：

- `r_tight = summary["eval_goal_radius_m"]`，用于 GOAL sphere success；
- `r_protocol = environment_config.plant.goal_radius_m`，并要求其与
  `summary["eval_goal_plane_miss_guard_radius_m"]` 一致，用于 true capture；
- `success_margin(m)` 只由 sphere 判定；plane crossing 仅作诊断，不再混入
  success；
- margin 只有同时保留至少 `50%` 的 margin=0 true success 才可称为可用。

这是一次测量口径纠错，不改训练算法、checkpoint、seed、eval split 或 held-out。

## 2. 假设与最便宜 falsification

### H74-A：margin 收紧会按边界余量减少 false-success

机制：false-success 的 `p_nav` 距离只比半径小 `0.01–0.04 m`；把 sphere
capture 改为 `d_p <= r - margin` 会让这些边界样本退出 success。

支持条件（测量前冻结）：

1. `margin = 0.10/0.25/0.50 m` 的 false-success 计数单调不增；
2. `margin=0.25 m` 相对 `margin=0` 至少减少 `2` 个 false-success；
3. `margin=0.25 m` 的 true success 保留率至少 `50%`，并同时报告绝对损失，
   不能只报 false-success 下降。

最便宜 falsification：只读重分类现有 Goal71 episodes，不新增 rollout。

### H74-B：p_nav-eta 误差有沿 goal 方向的系统偏差

机制：如果 `p_nav` 在 goal 方向上系统性偏乐观，则方向性 bias correction
可以在较少损失 true success 的情况下减少 false-success；如果只是零均值
噪声，margin 收紧会近似对称地损失 true success。

支持条件（测量前冻结）：

1. terminal `p_nav - eta` 在 goal 方向投影的中位数 `>= +0.10 m`
   （正号表示 p_nav 更靠近 goal）；
2. false-success 组的方向投影中位数比 true-success 组高 `>= +0.05 m`；
3. 该结果在 default 与 strict125 同向；若只有一个 scenario 支持，只记为
   描述性证据，不单独通过。

最便宜 falsification：同一批 trace 的方向投影统计，不训练、不新增 rollout。

### H74-C：如果 margin/bias 都不能解释，转动作参数化/奖励

支持条件：H74-A 与 H74-B 都不成立，或 margin 收紧只减少 false-success
但 true success 等量下降，说明不是可修正的估计偏差。

## 3. 数据与隔离

| 项目 | 固定值 |
|---|---|
| candidate | Goal71 limiter-on `_dwell50_m025_cmdlimit` |
| checkpoints | frozen Goal59 `s240/s752/s880/s1008` + `zero` |
| scenarios | `default`, `strict125` |
| seeds | dev `20–59` |
| eval | CPU/float64、`endpoint_stop`、dwell=50、`r_tight=r0-0.25 m` |
| held-out | `100–149` 关闭 |
| training | 无 |
| new rollouts | 无 |

## 4. 方法

对每个 episode 计算：

```text
d_p = ||p_nav - goal||
d_e = ||eta - goal||
e = p_nav - eta
u = (goal - eta) / ||goal - eta||
bias_projection = e · u
```

重分类：

```text
success_margin(m) = d_p <= r_tight - m
true_capture      = d_e <= r_protocol
false_success(m)  = success_margin(m) and not true_capture
true_success(m)   = success_margin(m) and true_capture
```

其中 `signed_offset >= 0` 只作为 terminal plane 诊断，不参与 success；
这与实际 evaluator 的 `PCTerminalKind.GOAL` 语义一致。

`margin` 固定为 `0 / 0.10 / 0.25 / 0.50 m`，不做事后扫描。报告每个
margin 的 success、true capture、false-success、plane success、true success
损失，以及 false-success 与 true-success 的 bias projection 分布。

## 5. 执行计划

- [x] Step 0：Run Contract、路径/SHA/配置审计（复用 Goal71/73 冻结审计）；
- [x] Step 1：实现 analyzer + 单元测试；
- [x] Step 2：本地定向测试、Ruff；
- [x] Step 3：服务器 `--jobs 10` 只读运行，拉回 JSON/MD；
- [x] Step 4：按 Gate 判 H74-A/H74-B/H74-C；
- [x] Step 5：写 result、blocker/ledger、STATE、strict checker、只 add 相关文件。

## 6. Gate

| check | 阈值 |
|---|---:|
| false-success margin 单调性 | `0.10/0.25/0.50` 单调不增 |
| margin=0.25 false-success 减少 | `>= 2` 个 |
| margin=0.25 true success 保留率 | `>= 50%`（否则 margin 不可用） |
| bias projection median | `>= +0.10 m` |
| false-success minus true-success bias | `>= +0.05 m` |
| scenario 一致性 | default 与 strict125 同向 |

失败裁决：

| 观察 | 判读 | 下一步 |
|---|---|---|
| margin 收紧减少 false-success 但 true 等量下降 | 边界噪声，不是方向 bias | 转动作参数化/奖励 |
| margin 减少 false-success 但 true 保留率 `<50%` | `MARGIN_UNUSABLE`；若 bias 同时通过则 `MARGIN_UNUSABLE_BIAS_SUPPORTED` | 不采用 margin 收紧；预注册 bias correction |
| margin 收紧几乎不减少 false-success | capture 不是主因 | 转动作参数化/奖励 |
| bias projection 明显为正且 false-success 更高 | 方向性估计偏差 | 预注册 bias correction 单因素 arm |
| 数据不有限/不一致 | INVALID | 修数据，不训练 |

## 7. 算力

- 只读分析，`--jobs 10`，每进程单线程 BLAS；
- 不占 GPU，不训练，不新增 rollout；
- 服务器使用隔离目录，不覆盖 Goal67/71/72/73 结果。

## 8. Git

禁止 `git add -A`。只 add 本 Goal 相关文件，提交前运行：

```bash
.venv/bin/python ../scripts/check_research_ledger.py \
  --goal Goal74 \
  --goal-doc doc/goals/goal74_terminal_capture_margin_bias_20261110.md \
  --strict
git diff --cached --check
```

不自动 commit/push，除非用户明确要求。

## 9. Goal 提示词

```text
你现在执行 Goal74：terminal capture margin 与 p_nav-eta 方向偏差的只读审计。

先读：
- doc/goals/goal74_terminal_capture_margin_bias_20261110.md
- doc/goals/goal73_result_20261110.md
- research/blockers/2026-11-10-goal73-command-demand-not-discriminative.md

硬约束：
1. 只读复用 Goal71 episodes/trace，不训练、不新增 rollout、不打开 held-out；
2. margin 固定 0/0.10/0.25/0.50 m，不做事后扫描；
3. 同时报告 false-success 与 true success 损失，不能只报下降；
4. 先判 H74-A/H74-B，再决定是否预注册 bias correction 或 capture arm；
5. 不换 SAC/REDQ/DroQ，不加控制 authority；
6. 禁止 git add -A；只 add 本次相关文件。
```
