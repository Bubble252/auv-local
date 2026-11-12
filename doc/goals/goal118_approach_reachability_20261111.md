# Goal118：2 m entry reachability 与 terminal-credit 分解（只读）

**日期**：2026-11-11<br>
**状态**：`PROTOCOL_DRAFTED_BEFORE_MEASUREMENT`<br>
**上游**：Goal117 `MIXED_TERMINAL_MECHANISM`；Goal85/85b/86 的位置、plant mismatch 与 BC 先验契约<br>
**性质**：eval-only 只读 screen；不训练、不改 checkpoint、不改 reward/terminal/success/goal radius、不打开 held-out、不启动 SAC。

## Research Contract

```yaml
autoresearch:
  required: false
  triggers:
    - gate_fail_twice
    - no_metric_movement_after_3_fixes
    - heldout_collapse
    - seed_noise_ambiguous
  ledger: research/LEDGER.jsonl
  blockers_dir: research/blockers/
  reading_dir: research/paper_reading/
  max_outer_loops: 3
  check_command: python3 ../scripts/check_research_ledger.py --goal Goal118 --goal-doc doc/goals/goal118_approach_reachability_20261111.md --strict --require-result
```

这是冻结的预注册 screen。记录前不得改 success envelope 定义、p95 分位、8/11
阈值、primary 11 seed 集合或分类规则。

## 1. 问题

Goal117 已经证明：

```text
11/11 个 primary failure 在 2 m entry 时 geometry-bad
success 只有 3/20 在 2 m entry geometry-bad
failure 最后 2 s terminal share = -0.230，success = 0.594
failure dense/terminal = 3.28，success = 0.68
```

但还没有回答：

```text
这 11 个 failure 在 2 m entry 时，按剩余时间看，
vertical/pitch 修正是“物理上已经来不及”，还是“仍然可达但策略没有做到”？
```

这决定下一步是改 approach-phase 的 vertical/pitch 收敛，还是改 terminal
credit / reward，而不是继续换 RL 算法。

## 2. 固定输入

只读 Goal117 的：

```text
artifacts/lowlevel/goal117_terminal_closure_attribution_20261111/
    R100_RTRACE/summary.json
    R100_RTRACE/trace.npz
    phase0_summary.json
    parity.json
```

不新增 rollout，不重新训练，不改变 Goal116 R100 / Goal117 R100_RTRACE 的协议。

## 3. 方法

### 3.1 2 m entry 状态

对每个 episode，取首次 `goal_distance_eta_m <= 2.0 m` 的 index：

```text
e_vertical_m   = |eta_z - goal_z|
e_pitch_rad    = |pitch_error_rad|
e_heading_rad  = |heading_error_rad|
T_to_closest_s = closest_time_s - entry2_time_s
```

若没有 2 m entry，标 `NO_2M_ENTRY`，不参与 reachability 统计。

### 3.2 成功包络

只用 20 个 success control 的 2 m entry 构造经验包络：

```text
vertical_rate_req = e_vertical_m / max(T_to_closest_s, dt)
pitch_rate_req    = e_pitch_rad / max(T_to_closest_s, dt)

success_vertical_p95 = p95(vertical_rate_req among success)
success_pitch_p95    = p95(pitch_rate_req among success)
```

`dt` 取 trace 的决策步长中位数。p95 而不是 max，避免单个 success 的离群点把
包络抬得过高。

### 3.3 分类

```text
margin_vertical = vertical_rate_req / success_vertical_p95
margin_pitch    = pitch_rate_req / success_pitch_p95
margin_max      = max(margin_vertical, margin_pitch)
```

- `UNREACHABLE_AT_2M`：`margin_max > 1.0`
- `REACHABLE_AT_2M`：`margin_max <= 1.0`
- `NO_2M_ENTRY`：没有 2 m entry

这只是“超出 success 已展示过的修正速度包络”，不是严格物理不可达；因此结论
只能用于决定下一轮优先级，不能写成动力学不可能。

### 3.4 terminal credit 组合

对每个 primary failure 读取 Goal117 的：

```text
terminal_share_last2s
dense_terminal_ratio_last2s
```

组合标签：

- `APPROACH_GEOMETRY_PRIMARY`：`UNREACHABLE_AT_2M`
- `TERMINAL_CREDIT_PRIMARY`：`REACHABLE_AT_2M` 且 `terminal_share_last2s < 0.2`
- `REACHABLE_CONTROL_LIMITED`：`REACHABLE_AT_2M` 且 `terminal_share_last2s >= 0.2`

## 4. 主指标

对 11 个 primary failure 报告：

```text
UNREACHABLE_AT_2M
REACHABLE_AT_2M
NO_2M_ENTRY
margin_vertical median / p95
margin_pitch median / p95
margin_max median / p95
terminal_share_last2s median
dense_terminal_ratio_last2s median
```

同时报告 20 个 success control 和 20 个 failure 全集。

## 5. 预注册 Gate

| 观察 | 状态 | 下一步 |
|---|---|---|
| `UNREACHABLE_AT_2M >= 8/11` | `APPROACH_GEOMETRY_PRIMARY` | 新 Goal 做 approach-phase vertical/pitch 收敛；不改 RL 算法 |
| `REACHABLE_AT_2M >= 8/11` 且 `TERMINAL_CREDIT_PRIMARY >= 8/11` | `TERMINAL_CREDIT_PRIMARY` | 新 Goal 预注册 potential-based terminal shaping / dense 退火 |
| `REACHABLE_AT_2M >= 8/11` 且 `REACHABLE_CONTROL_LIMITED >= 8/11` | `REACHABLE_CONTROL_LIMITED` | 新 Goal 查动作参数化 / controller authority |
| 没有类别达到 `8/11` | `MIXED_REACHABILITY` | 写 blocker，进入 autoresearch loop |
| parity、seeds、summary 任一不通过 | `INVALID` | 先修运行协议 |

不把 20 个 success 的 p95 当成严格物理上界；它只是“success 已展示过的包络”。
若 failure 的 margin 在 `1.0–1.2` 之间，判 `AMBIGUOUS`，不强行归类。

## 6. 执行计划

- [x] Step 0：写 Goal118 预注册并冻结输入/阈值；
- [x] Step 1：实现 `scripts/analyze_pc_goal118_reachability.py`；
- [x] Step 2：本地 smoke：读取 Goal117 summary/trace，验证 parity 与 20/40；
- [x] Step 3：计算 success envelope、margin 和分类；
- [x] Step 4：Gate 判定：`MIXED_REACHABILITY`；
- [x] Step 5：写 result/blocker/decision/LEDGER/STATE；
- [ ] Step 6：运行 checker 和 `git diff --check`；不自动 commit/push。

## 7. Git

禁止 `git add -A`。只 add 本次相关文件，提交前运行：

```bash
python3 ../scripts/check_research_ledger.py \
  --goal Goal118 \
  --goal-doc doc/goals/goal118_approach_reachability_20261111.md \
  --strict --require-result
git diff --cached --check
```
