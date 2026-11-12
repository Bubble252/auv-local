# Goal61：Goal59 checkpoint 的 p_nav/eta terminal 对齐只读审计

日期：2026-11-10<br>
状态：`COMPLETED_DIAG_H_C1_C2_C3_SUPPORTED`<br>
父结果：Goal60 `DIAG_COMPLETE_H_C_SUPPORTED_H_B_WEAK_H_A_REJECTED`<br>
服务器：`yhx@152.136.183.51 -p 43127`，8×RTX 4090

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
  check_command: python ../scripts/check_research_ledger.py --goal Goal61 --goal-doc doc/goals/goal61_terminal_pnav_eta_alignment_20261110.md --strict
```

## 1. 背景

Goal60 在 Goal59 的 4 个 final checkpoint 上发现：

- default success/capture discordance `0.381`，strict125 `0.269`；
- success_no_capture 的 true `eta` terminal distance 明显更大；
- H-A（动作饱和）不是 episode 级失败判别器；
- H-B（reward composition）是弱/部分信号；
- H-D/H-E 不支持换 SAC/REDQ/DroQ。

Goal41/42 已在 Goal38 trace 上证明 `p_nav` 比 true `eta` 更接近 goal，且误差方向
指向 goal。Goal61 只做一件事：在 **Goal59 checkpoint 的 Goal60 trace** 上复现
该机制，并区分“策略放大”还是“零残差基线也存在”。

不训练、不新增 rollout、不改控制器、不打开 held-out。

## 2. 可证伪假设

### H-C1：success_no_capture 中 p_nav 系统性更接近 goal

在 protocol success 但 true capture 为 false 的 episode 中：

- `p_nav_goal_distance < eta_goal_distance` 的比例；
- `p_nav_goal_distance - eta_goal_distance` 的中位数（负值表示 p_nav 更近）；
- 误差向量 `p_nav - eta` 指向 goal 的方向余弦。

支持条件（至少一个 scenario）：

```text
p_nav closer fraction >= 0.80
median (p_nav - eta) <= -0.15 m
direction cosine > 0 fraction >= 0.80
```

### H-C2：success 在 true capture 之前触发

在 success_no_capture 中：

```text
p_nav_goal_distance <= goal_radius
eta_goal_distance > goal_radius
```

支持条件：满足该顺序的比例 `>=0.80`（至少一个 scenario）。

### H-C3：偏差来自环境/估计器，不是 residual 策略特有

对同场景的 forced zero-residual trace 做同样统计：

- 若 zero-residual 也出现 success_no_capture 或 p_nav closer fraction `>=0.80`，
  则偏差主要是环境/估计器/终止口径，不是 residual 策略造成；
- 若 zero-residual 没有偏差、只有 policy 有，则 residual 放大了偏差。

支持条件：zero-residual 的 p_nav closer fraction `>=0.80` 或
success_no_capture count `>0`。

## 3. 数据与隔离

| 项目 | 固定值 |
|---|---|
| 输入 | Goal60 的 10 个 trace run |
| policy checkpoint | Goal59 `240/752/880/1008` 的 `final.pt` |
| scenario | `default`、`strict125` |
| dev seeds | `20–59` |
| zero-residual | 同 scenario 的 forced zero-residual trace |
| 训练 | 不运行 |
| held-out | `100–149` 关闭 |
| 终止 | `endpoint_stop`，CPU/float64 |

Goal radius 从 environment config 读取：default `2.0 m`，strict125 `1.25 m`。

## 4. 指标合同

每个 episode 的 terminal row（`done=true`）提取：

- `eta_ned`、`p_nav_ned`、`goal_ned`；
- `eta_goal_distance`、`p_nav_goal_distance`；
- `gap = p_nav_goal_distance - eta_goal_distance`；
- `bias_vector = p_nav_ned - eta_ned`；
- `toward_goal_cosine = dot(bias_vector, goal - eta) / (|bias_vector| |goal - eta|)`；
- `p_nav_signed_offset`、`eta_signed_offset`；
- protocol `success`、true `capture`、`terminal_kind`。

输出：

```text
goal61_terminal_alignment.json
goal61_terminal_alignment.md
```

数据完整性：40 seeds、40 terminal 行、finite、checkpoint SHA、无 held-out。

## 5. Gate 与决策

| 结果 | 决策 |
|---|---|
| H-C1/H-C2 支持，H-C3 支持 | 偏差主要是环境/估计器/终止口径；下一 Goal 修 calibration/termination |
| H-C1/H-C2 支持，H-C3 不支持 | residual 放大偏差；下一 Goal 同时查 policy 与 estimator |
| H-C1/H-C2 不支持 | 写 blocker，重新提出 hypothesis；不训练 |
| 任何情况 | 不启动 SAC/REDQ/DroQ，不打开 held-out |

## 6. 执行计划

- [x] Step 0：复核 Goal60 数据完整性、Goal41/42 结论和 Goal radius；
- [x] Step 1：写 `scripts/analyze_pc_goal61_terminal_alignment.py`；
- [x] Step 2：本地测试 + Ruff；
- [x] Step 3：服务器只读运行，输出 JSON/MD；
- [x] Step 4：写 result、blocker/decision、LEDGER、两份 STATE，运行 strict checker；
- [ ] Step 5：只 add 相关文件，`git diff --cached --check`；用户要求后才 commit/push。

## 7. 明确禁止

- 不训练、不新增 rollout、不改 controller/termination/config；
- 不打开 held-out；
- 不把只读审计写成性能 PASS；
- 不用换 seed、加预算或放宽阈值救结果；
- 不 `git add -A`。

## 8. 执行结果

见 `doc/goals/goal61_result_20261110.md`。摘要：

- default policy pooled success_no_capture `58` 个，p_nav closer/direction/order
  均为 `1.000`，中位 gap `-0.200 m`（95% CI `[-0.329, -0.163]`）；
- strict125 policy pooled success_no_capture `36` 个，p_nav closer/direction/order
  均为 `1.000`，中位 gap `-0.242 m`（95% CI `[-0.321, -0.175]`）；
- forced zero-residual 在两场景各有 `13` 个 success_no_capture，同样全部
  closer/direction/order `1.000`；
- H-C1/H-C2/H-C3 均支持；H-C2 在当前标签定义下接近恒等，不能单独当作因果证据；
- NoisyNet 不是主导：Goal59 的 8/8 deterministic mean 已饱和，`noise/det`
  中位数 `0.0193`；本 Goal 不改变这一判断；
- 决策 `pivot` 到 terminal calibration/termination；不启动 SAC/REDQ/DroQ，
  不打开 held-out。
