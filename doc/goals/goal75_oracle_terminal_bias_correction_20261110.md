# Goal75：oracle terminal p_nav bias correction 只读审计

**日期**：2026-11-10<br>
**状态**：`PARTIAL_SUPPORT`<br>
**上游**：Goal74 `MARGIN_UNUSABLE_BIAS_SUPPORTED`<br>
**结果**：`doc/goals/goal75_result_20261110.md`

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
  check_command: .venv/bin/python ../scripts/check_research_ledger.py --goal Goal75 --goal-doc doc/goals/goal75_oracle_terminal_bias_correction_20261110.md --strict
```

## 1. 背景与证据

Goal74 已证明：

- post-hoc isotropic capture margin 不可用：`margin=0.10 m` 会把
  false-success 和 true-success 一起清零；
- false-success 与 true-success 的 terminal `p_nav-eta` 沿 goal 方向投影
  明显不同：default `+0.5591 / -0.0155 m`，strict125
  `+0.5513 / +0.0760 m`；
- 这不是 NoisyNet 主导：正式评估使用 deterministic mean；zero-residual
  仍复现 terminal capture 失败。

因此下一步不是继续加控制 authority，也不是换 RL 算法，而是先做只读的
oracle bias correction，回答两个问题：

1. 沿 goal 方向扣掉乐观偏差，是否真的能消除 false-success？
2. 这种修正是否会误伤 true-success，尤其是 strict125？

## 2. 假设与最便宜 falsification

### H75-A：full oracle 沿 goal 方向偏差修正可以消除 false-success 且保留 true-success

机制：如果 false-success 主要由 `p_nav` 沿 goal 方向的乐观偏差造成，那么把
`p_nav` 沿 true goal 方向推回 `p_nav-eta` 的投影量，应让这些 false-success
退出 tight sphere，同时几乎不影响 true-success。

支持条件（测量前冻结）：

1. default 与 strict125 的 false-success 都下降至少 `50%`；
2. 两个 scenario 的 true-success 保留率都至少 `80%`；
3. 重分类 mismatch 不超过 `2%`。

最便宜 falsification：复用 Goal71 limiter-on episodes/trace，不训练、不新增
rollout。

### H75-B：只修正强正偏差的 deadband 规则可以更安全地分离 false-success

机制：Goal74 的 true-success 组仍有少量正偏差；full correction 可能误伤它们。
只对 `b > b0` 的部分做修正，可以保留正常 true-success，同时移除强乐观的
false-success。固定 `b0 = 0.25 m`，位于 Goal74 true 组中位
`-0.0155 / +0.0760 m` 与 false 组中位 `+0.5591 / +0.5513 m` 之间。

支持条件（测量前冻结）：

1. default 与 strict125 的 false-success 都下降至少 `50%`；
2. 两个 scenario 的 true-success 保留率都至少 `95%`；
3. 修正量只使用 `max(b - 0.25, 0)`，不对负偏差或弱正偏差做补偿。

### H75-C：如果两种 oracle 修正都失败，转动作/奖励/terminal reference

支持条件：H75-A 与 H75-B 都不成立；此时不再继续做 `p_nav` 偏差拟合，
回到动作参数化、reward 或 terminal reference point。

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

不覆盖 Goal67/71/74 的 trace、JSON 或 MD。服务器只读运行，结果写入 Goal75
隔离目录。

## 4. 方法

对每个 terminal snapshot 计算：

```text
e = p_nav - eta
d_p = ||p_nav - goal||
d_e = ||eta - goal||
u = (goal - eta) / ||goal - eta||
b = dot(e, u)

p_nav_none      = p_nav
p_nav_full      = p_nav - max(b, 0) * u
p_nav_deadband  = p_nav - max(b - 0.25, 0) * u
```

重分类使用与 Goal74 相同的真实协议：

```text
r_tight    = summary["eval_goal_radius_m"]              # 1.75 / 1.00
r_protocol = environment_config.plant.goal_radius_m     # 2.00 / 1.25
success    = ||p_nav_arm - goal|| <= r_tight
true_capture = d_e <= r_protocol
true_success = success and true_capture
false_success = success and not true_capture
```

`full` 和 `deadband` 都是 oracle 诊断，不是可部署策略；`u` 和 `b` 使用 true
`eta`，只用于判断偏差是否可分离。若 deadband 通过，下一 Goal 才允许预注册
从可观测特征预测 `b` 的单因素 arm。

固定报告：

- no-correction baseline 的 success/true/false；
- full 与 deadband 的 success/true/false；
- false-success reduction fraction；
- true-success retention；
- false 组与 true 组的 `b` 中位数、`p_nav-eta` 误差、tight margin。

## 5. 执行计划

- [x] Step 0：Run Contract、路径/SHA/配置审计；
- [x] Step 1：实现 analyzer + 单元测试；
- [x] Step 2：本地定向测试、Ruff；
- [x] Step 3：服务器 `--jobs 10` 只读运行，确认 8×4090 环境但不占 GPU，拉回 JSON/MD；
- [x] Step 4：按 Gate 判 H75-A/H75-B/H75-C；
- [x] Step 5：写 result、blocker/ledger、STATE、strict checker、只 add 相关文件。

## 6. Gate

| check | 阈值 |
|---|---:|
| H75-A false-success reduction | 两个 scenario 都 `>= 50%` |
| H75-A true-success retention | 两个 scenario 都 `>= 80%` |
| H75-B false-success reduction | 两个 scenario 都 `>= 50%` |
| H75-B true-success retention | 两个 scenario 都 `>= 95%` |
| reclassification mismatch | `<= 2%` |

失败裁决：

| 观察 | 判读 | 下一步 |
|---|---|---|
| full 与 deadband 都通过 | oracle 偏差可分离 | 预注册可观测 `b` predictor |
| 只有 deadband 通过 | full 修正过强，阈值化可分离 | 预注册可观测 `b` predictor + deadband |
| 两个都失败 | 沿 goal 投影不足以解释 false-success | 回动作参数化/reward/terminal reference |
| mismatch 超限 | 数据/半径口径错误 | `INVALID`，只修数据后重跑 |

## 7. 算力

- 只读分析，`--jobs 10`，每进程单线程 BLAS；
- 不训练、不新增 rollout、不打开 held-out；
- 该阶段不需要 GPU；8×4090 留给后续可部署 predictor/training arm；
- 服务器使用隔离目录，不覆盖 Goal67/71/74 结果。

## 8. Git

禁止 `git add -A`。只 add 本 Goal 相关文件，提交前运行：

```bash
.venv/bin/python ../scripts/check_research_ledger.py \
  --goal Goal75 \
  --goal-doc doc/goals/goal75_oracle_terminal_bias_correction_20261110.md \
  --strict
git diff --cached --check
```

不自动 commit/push，除非用户明确要求。

## 9. Goal 提示词

```text
你现在执行 Goal75：oracle terminal p_nav bias correction 只读审计。

先读：
- doc/goals/goal75_oracle_terminal_bias_correction_20261110.md
- doc/goals/goal74_result_20261110.md
- research/blockers/2026-11-10-goal74-capture-margin-unusable-bias-supported.md

硬约束：
1. 只读复用 Goal71 episodes/trace，不训练、不新增 rollout、不打开 held-out；
2. full 与 deadband 两条规则固定，不扫额外阈值；
3. 同时报告 false-success 下降与 true-success 保留率；
4. oracle correction 只作诊断，不得写成可部署收益；
5. 不换 SAC/REDQ/DroQ，不加控制 authority；
6. 禁止 git add -A；只 add 本次相关文件。
```
