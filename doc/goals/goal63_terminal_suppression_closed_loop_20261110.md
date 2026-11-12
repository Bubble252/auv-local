# Goal63：frozen checkpoint 的闭环 terminal suppression

**日期**：2026-11-10<br>
**状态**：`PARTIAL_TERMINAL_SUPPRESSION`<br>
**上游**：Goal62 `INVALID_COUNTERFACTUAL_DEGENERATE_BOUNDARY`<br>
**服务器**：`yhx@152.136.183.51 -p 43127`，8×RTX 4090

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
  check_command: python ../scripts/check_research_ledger.py --goal Goal63 --goal-doc doc/goals/goal63_terminal_suppression_closed_loop_20261110.md --strict
```

## 1. 背景与问题

Goal61 证明 protocol success 使用的 `p_nav` 半径与 true capture 使用的 `eta`
半径存在系统性错位；Goal62 试图用 Goal60 terminal row 做静态 radius margin
反事实，但 trace 在首次穿过原半径时停止，任意正 margin 都会删除全部 baseline
success，得到 `INVALID_COUNTERFACTUAL_DEGENERATE_BOUNDARY`。

本 Goal 只回答：**在 frozen checkpoint 上抑制 GOAL 终止并继续 rollout，是否能
减少 false success，同时保留 true capture？** 不训练、不换算法、不扫 reward。

## 2. 可证伪假设

### H63-A：闭环抑制 GOAL 终止能区分 false success 与 true capture

候选规则：`p_nav` 达到 `r_tight = max(r0 - 0.25 m, 0)` 才触发 GOAL；timeout、
safety 和 plane-miss 仍可终止。若 baseline false-success episode 只是“过早按
原半径结束”，抑制后应有一部分继续进入 true capture，或至少不再被记为 false
success；baseline true-success episode 应大部分继续成功。

若成立应看到 policy pooled 在两个 scenario 同时满足：

```text
false_success_removed_fraction >= 0.50
true_success_retention >= 0.90
```

若失败应看到：抑制后 false success 仍大量保留，或 true success 大量变成
timeout/plane-miss/safety。此时不能把它归因于 NoisyNet、残差大小或样本效率。

最便宜 falsification：本 Goal 的 frozen-checkpoint 闭环 eval；不训练。

### H63-B：效果跨 checkpoint 稳健，不是单个 seed 的偶然

若 H63-A 成立，且每个 scenario 至少 `3/4` 个 policy checkpoint 同时满足
`false_success_removed_fraction >= 0.40`、`true_success_retention >= 0.85`，
则 H63-B 支持。若只有 1–2 个 checkpoint 通过，则只记为局部效应，不授权改正式
终止协议。

### H63-C：zero-residual 同向复现

若 forced zero-residual 在两个 scenario 都满足
`false_success_removed_fraction > 0` 且 `true_success_retention >= 0.80`，则说明
终止抑制不是 residual 策略特有；否则候选规则只能作为 policy-specific
diagnostic。

### H63-D：若闭环抑制仍失败，问题转向 estimator/geometry

若 H63-A 失败，但抑制 arm 产生大量 timeout/plane-miss，且 terminal `p_nav`
与 `eta` 的距离/方向仍不收敛，则下一步应检查 `p_nav` 估计器、终端几何和
controller，而不是继续调 residual 或换 SAC/REDQ/DroQ。

## 3. 数据与隔离

| 项目 | 固定值 |
|---|---|
| baseline | Goal60 的 10 个 trace run，dev `20–59` |
| policy checkpoint | Goal59 `240/752/880/1008` 的 `final.pt` |
| scenario | `default` r0 `2.00 m`；strict125 r0 `1.25 m` |
| candidate radius | `r0 - 0.25 m`：`1.75/1.00 m` |
| plane-miss guard | 原 `r0`：`2.00/1.25 m` |
| action mode | policy + forced zero-residual |
| 评估精度 | CPU / float64 |
| 终止模式 | `endpoint_stop` |
| 训练 / held-out | 不训练；`100–149` 关闭 |

baseline 直接复用 Goal60 trace，但必须校验 checkpoint SHA、seeds、finite、
`endpoint_stop`、policy/zero radius。candidate 重新 rollout，但只改 eval-only
GOAL 半径和 plane-miss guard；reward、observation、controller、checkpoint、
seed、dtype、device 不变。

## 4. Eval-only 终止语义

不改 `plant.goal_radius_m`。只把 termination manager 的 GOAL 判定半径设为
`r_tight`，并显式保留 plane-miss guard 为原 `r0`：

```text
GOAL:              p_nav_distance <= r_tight
GOAL_PLANE_MISS:   p_nav_distance > r0 and signed_plane_offset >= 0
TIME_LIMIT/SAFETY: 与原协议相同
```

这样做是因为如果把 plane-miss 的参考半径也缩小，原有正常接近会在原半径处被
误判为 plane-miss，无法继续 rollout 到更紧半径。默认参数不传时，终止行为与
原实现逐元素一致。

## 5. 指标合同

每个 run 按 seed 与 baseline 配对：

- `baseline_success`：Goal60 `success`；
- `baseline_capture`：Goal60 `terminal_eta_capture`，使用原 `r0`；
- `candidate_success`：candidate terminal kind 为 GOAL；
- `candidate_capture`：candidate terminal `eta` distance <= 原 `r0`；
- `baseline_false_success = baseline_success & ~baseline_capture`；
- `candidate_false_success = candidate_success & ~candidate_capture`；
- `false_success_removed_fraction`：baseline false-success episode 中，candidate
  不再属于 false-success 的比例；
- `true_success_retention`：baseline true-success episode 中，candidate 仍为
  true-success 的比例；
- `false_to_true_fraction`：baseline false-success 转成 candidate true-success
  的比例；
- `candidate_precision = candidate_true / candidate_success`；
- 同时报告 success、true capture、plane-miss、timeout、safety、extra steps、
  terminal `p_nav`/`eta` distance。

分母为零时输出 `null`。所有比例用 10,000 次 seed-level bootstrap 95% CI。

## 6. Gate（测量前冻结）

### 6.1 数据有效性

| check | 通过条件 |
|---|---|
| seeds | 每个 run 恰好 `20–59`，无 held-out |
| baseline/candidate | 同 seed、同 checkpoint、同 scenario、同 action mode |
| checkpoint | SHA 与 Goal60 一致 |
| 数值 | 所有 episode finite；40 个 episode |
| 隔离 | 无训练、无 reward/observation/controller/config 改动 |

### 6.2 主候选 Gate

Policy pooled 在 default 和 strict125 同时满足：

| check | 阈值 |
|---|---:|
| baseline false success | `> 0` |
| false success removed fraction | `>= 0.50` |
| true success retention | `>= 0.90` |

### 6.3 Checkpoint 稳健性

每个 scenario 至少 `3/4` 个 policy checkpoint 满足：

```text
false_success_removed_fraction >= 0.40
true_success_retention >= 0.85
```

### 6.4 Zero-residual 方向

每个 scenario 的 forced zero-residual 满足：

```text
false_success_removed_fraction > 0
true_success_retention >= 0.80
```

### 6.5 稳定性护栏

- 所有 run finite；无 NaN/Inf/OOM；
- safety rate 相对 baseline 增加不超过 `0.05`；
- 报告但不单独 gate plane-miss/timeout/extra steps，因为它们可能是终止抑制的
  预期代价；只有主指标和 true-retention 同时通过才允许进入下一阶段。

## 7. 裁决

| 观察 | 判读 | 下一步 |
|---|---|---|
| §6.2 + §6.3 + §6.4 全通过 | `PASS_TERMINAL_SUPPRESSION` | 新 Goal 预注册正式终止规则与 held-out 方案；仍不直接改训练 |
| 主候选通过但 checkpoint/zero 不稳健 | `PARTIAL_TERMINAL_SUPPRESSION` | 只做机制诊断，不推广 |
| false success 降不下或 true retention 崩 | `FAIL_TERMINAL_SUPPRESSION` | 写 blocker；按 H63-D 转 estimator/geometry |
| 数据或实现不完整 | `INVALID` | 只修测量/实现，不改阈值 |

任何结果都不能把 Goal63 写成训练算法 PASS，也不能重开 Goal27/58/59 或打开
held-out。

## 8. 算力与并行合同

本 Goal 是 eval-only，不增加训练预算，但充分并行：

- 10 个 candidate run：4 checkpoint × 2 scenario policy，加 2 个 zero-residual；
- 每个 run 内部 `num_envs=40`，一次覆盖完整 dev seed 集；
- 最多 8 个进程并行，`OMP_NUM_THREADS=1`/`MKL_NUM_THREADS=1`，避免 CPU 线程
  过度订阅；
- 每片写独立目录，最后按 `(scenario, checkpoint, action_mode, seed)` 合并；
- baseline 复用 Goal60，不重复消耗算力；
- 不改变正式 Gate 的 seed、dtype、device、终止规则或 checkpoint 选择。

后续若进入训练分支，按 seed 分 GPU 并行；探索型 arm 可以预注册更大
`num_envs` 和 batch（例如 256/512），但必须保持总 transitions、UTD 和评估协议
可比，且不能静默改正式 Gate。

## 9. 执行计划

- [x] Step 0：复核 Goal60/62、termination 代码和服务器空闲资源；
- [x] Step 1：实现 eval-only GOAL radius/plane-miss guard override；
- [x] Step 2：本地单测、Ruff、smoke；
- [x] Step 3：服务器同步代码，1 checkpoint × 2 seed smoke；
- [x] Step 4：8 路并行运行 10 个 candidate run；
- [x] Step 5：写配对分析器，输出 JSON/MD，按 Gate 裁决；
- [x] Step 6：写 result/blocker/reading note、LEDGER、两份 STATE，运行 strict checker；
- [ ] Step 7：只 add 相关文件，`git diff --cached --check`；用户要求后才 commit/push。

## 10. 明确禁止

- 不训练、不改 reward/observation/controller/config；
- 不直接改 `plant.goal_radius_m` 全局字段；
- 不把候选半径用于 reward 或 plane-miss guard；
- 不打开 held-out `100–149`；
- 不启动 SAC/REDQ/DroQ；
- 不重开 Goal27/58/59；
- 不 `git add -A`。

## 11. Goal 提示词

```text
你现在执行 Goal63：frozen checkpoint 的闭环 terminal suppression。

先读：
- doc/goals/goal63_terminal_suppression_closed_loop_20261110.md
- doc/goals/goal62_result_20261110.md
- doc/goals/goal61_result_20261110.md
- src/auv_lowlevel/batch/pc.py（terminal manager）
- research/STATE.md

硬约束：
1. 只做 eval-only GOAL radius override，不改全局 goal_radius_m、reward、
   observation、controller 或 checkpoint；
2. plane-miss guard 固定为原 r0，避免正常接近被误判；
3. policy/zero-residual 同 seed、同协议；
4. dev 20-59，held-out 100-149 关闭；
5. 10 个 candidate run 最多 8 路并行，seed 层合并；
6. 先 smoke，再完整 run；不训练、不启动 SAC/REDQ/DroQ。

执行顺序：
- 单测 override 默认行为不变；
- 服务器 smoke；
- 并行完整 run；
- 配对分析、Gate、result、ledger、STATE、strict checker；
- 不自动 commit/push。
```

## 12. 执行结果（2026-11-10）

服务器 smoke 通过后，10 个 candidate run 在 8 路并行下完成。最终状态
`PARTIAL_TERMINAL_SUPPRESSION`：

- default policy：false-success removal `0.621`，true retention `0.907`，4/4
  checkpoint 通过；
- strict125 policy：removal `0.611`，但 true retention `0.739`，0/4 checkpoint
  通过；
- zero-residual：strict125 方向通过，default retention `0.750` 未过；
- timeout/safety 不增，plane-miss 分别 `+0.062/+0.088`，extra steps 很小。

因此 default 支持终止抑制，strict125 更像 terminal geometry/plane-crossing
问题。未训练、未改 reward/observation/controller、未打开 held-out；不启动
SAC/REDQ/DroQ，不回头调 `0.25 m` margin。

详见：

- `doc/goals/goal63_result_20261110.md`
- `research/blockers/2026-11-10-goal63-strict125-capture-loss.md`
- `research/paper_reading/2026-11-10-goal63-terminal-suppression-partial.md`
