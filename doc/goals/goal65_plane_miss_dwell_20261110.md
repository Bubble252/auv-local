# Goal65：terminal plane-miss 的短 dwell 反事实

**日期**：2026-11-10<br>
**状态**：`DWELL_NOT_SUFFICIENT`（见 `doc/goals/goal65_result_20261110.md`）<br>
**上游**：Goal27 `H3_SR_FAIL_STOP`、Goal64 `GEOMETRY_SUPPORTED`<br>
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
  check_command: python ../scripts/check_research_ledger.py --goal Goal65 --goal-doc doc/goals/goal65_plane_miss_dwell_20261110.md --strict
```

## 1. 背景与问题

Goal27 的 command/action residual 主线已经停止：C1/R1/H1/G1/A1 以及 H3-SR
都没有产生可用收益。Goal27 的 decision 明确要求下一步先处理任务级 terminal
capture/plane miss，而不是继续扫 residual reward、gamma 或训练 seed。

Goal64 已经把这一分支定位到几何：strict125 policy 的 6 个 baseline true-success
丢失 episode 全部是 `GOAL_PLANE_MISS`，minimum `p_nav` 中位数 `1.172 m` 高于
candidate GOAL 半径 `1.00 m`，perpendicular offset 中位数 `1.065 m`。这说明
plane-miss guard 在收紧 GOAL 半径之前先终止了 episode。

本 Goal 只问一个更便宜的问题：

**如果 plane crossing 后允许 50 步（1 s）继续闭环，strict125 的 true capture
能否恢复？**

这是 Goal64 之后的单因素反事实。它不改 GOAL 半径、不改 reward、不改
controller/checkpoint，也不训练。

## 2. 可证伪假设

### H65-A：50 步 dwell 能恢复 strict125 的 true capture

机制：当前 plane-miss 在第一次穿过原半径 plane 时立即终止；若控制器只是“在
plane 处掠过”，短暂继续可能让它进入 `r_tight` 或至少回到原 `r0` 的 true
capture 区域。

支持条件（policy pooled）：

```text
strict125 true_success_retention >= 0.85
default  true_success_retention >= 0.90
strict125 false_success_removed_fraction >= 0.50
default  false_success_removed_fraction >= 0.50
```

若 strict125 retention 仍低于 `0.80`，则短 dwell 不是有效修复；下一步应查
estimator/calibration，而不是继续增加 dwell 或扫 margin。

### H65-B：dwell 不会用 false success 换 true capture

支持条件：

- candidate false-success count 不高于 Goal63 immediate suppression 的
  `+2` 个 seed；
- candidate 的 success 中 true capture 比例至少 `0.85`；
- safety/timeout 增量不超过 `0.05`；
- 至少 `3/4` checkpoint 的 retention 不低于 `0.80`。

若 dwell 增加了 false success，或把大量 episode 变成 timeout/safety，则
H65-B 被证伪，dwell 不能作为候选终止规则。

### H65-C：zero-residual 同向

若 forced zero-residual 在 strict125 的 true retention 也不低于 `0.80`，则
dwell 不是 residual 策略特有；若只有 policy 有效，则只能作为 policy-specific
诊断，不能推广。

## 3. 数据与隔离

| 项目 | 固定值 |
|---|---|
| baseline | Goal60 的 10 个 trace run，dev `20–59` |
| immediate reference | Goal63 的 10 个 suppression candidate |
| candidate | 4 checkpoint × 2 scenario policy + 2 zero-residual |
| checkpoint | Goal59 `240/752/880/1008` 的 `final.pt` |
| scenario | default r0 `2.00 m`；strict125 r0 `1.25 m` |
| candidate GOAL | `r0 - 0.25 m`：`1.75/1.00 m` |
| plane-miss guard | 原 `r0`：`2.00/1.25 m` |
| dwell | 首次 plane crossing 后最多 `50` 步（`1.0 s`） |
| action mode | policy + forced zero-residual |
| 精度 / 终止 | CPU / float64；`endpoint_stop` |
| 训练 / held-out | 不训练；`100–149` 关闭 |

baseline 和 immediate reference 都复用已有结果；candidate 只新增 eval-only dwell
和只读 trace，不改变 action、reward、observation、controller 或 checkpoint。

## 4. 方法

终止语义：

```text
GOAL:              p_nav_distance <= r_tight
GOAL_PLANE_MISS:   p_nav_distance > r0 and signed_offset >= 0
                   但首次满足 plane-miss 后，先累计 50 步；第 51 步仍满足才终止
TIME_LIMIT/SAFETY: 与原协议相同
```

`dwell=0` 时行为必须与 Goal63 的 immediate plane-miss 逐元素一致。dwell 计数器
在 episode reset、reset_mask、离开 plane-miss 条件或 GOAL 触发时清零。

## 5. Gate（测量前冻结）

本 Goal 是诊断型 Gate，不是正式部署协议。主判读：

| check | threshold |
|---|---:|
| strict125 policy true retention | `>= 0.85` |
| default policy true retention | `>= 0.90` |
| strict125 policy false removal | `>= 0.50` |
| default policy false removal | `>= 0.50` |
| checkpoint pass count | `>= 3/4` |
| zero-residual true retention | `>= 0.80` both scenarios |
| safety/timeout delta | `<= 0.05` |

失败裁决：

| 观察 | 判定 | 下一步 |
|---|---|---|
| H65-A/B/C 全部通过 | `DWELL_PASS_DIAGNOSTIC` | 另开预注册正式 termination 规则；仍不直接改训练 |
| H65-A 失败，但 H65-B 通过 | `DWELL_NOT_SUFFICIENT` | 停止短 dwell；转向 p_nav/eta estimator/calibration |
| false success 增加或 timeout/safety 增加 | `DWELL_INVALID_FALSE_SUCCESS` | 不推广 dwell；保留 Goal63 immediate suppression |
| 数据/实现不完整 | `INVALID` | 只修测量实现，不改阈值 |

## 6. 算力与并行合同

- 10 个 candidate run 全部并行，`MAX_PARALLEL=10`；
- 每个 run 内部 `num_envs=40`，覆盖完整 dev seed；
- 每进程 `OMP_NUM_THREADS=1`、`MKL_NUM_THREADS=1`；
- CPU / float64，保持 Goal63/64 数值协议；
- 生成只读 trace，便于确认 dwell 后 terminal kind 和 extra steps；
- 不为了加速改 dtype/device/终止规则，不增加重复 seed。

后续训练若开启，按 seed 分 GPU 并行；更大的 `num_envs`/batch 必须在新预注册中
固定总 transitions、UTD、replay 比例和评估协议。

## 7. 执行计划

- [x] Step 0：复核 Goal27/64、terminal manager 和服务器资源；
- [x] Step 1：实现 eval-only `goal_plane_miss_dwell_steps`，默认 0 等价；
- [x] Step 2：定向测试、Ruff、零残差/默认行为 smoke；
- [x] Step 3：服务器 1 个 checkpoint × 40 seed 复现检查；
- [x] Step 4：10 路并行运行 10 个 dwell candidate；
- [x] Step 5：新增 Goal65 paired analyzer，写 JSON/MD 和 H65 判读；
- [x] Step 6：写 result/blocker/reading note、LEDGER、两份 STATE；
- [x] Step 7：strict checker、全量测试、Ruff、`git diff --check`；
- [ ] Step 8：只 add 相关文件；用户明确要求后才 commit/push。

运行与验证命令：

```bash
export PYTHONPATH=src:/data/yhx/underwater-robotics/src/plan_p1/settled_全局规划器/src
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
MAX_PARALLEL=10 bash scripts/run_pc_goal65_plane_miss_dwell_20261110.sh

python3 scripts/analyze_pc_goal63_terminal_suppression.py \
  --baseline-root artifacts/traces \
  --candidate-root artifacts/goal65_plane_miss_dwell \
  --candidate-suffix _dwell50_m025 \
  --output artifacts/goal65_plane_miss_dwell/analysis/goal65.json \
  --markdown artifacts/goal65_plane_miss_dwell/analysis/goal65.md
```

## 8. 明确禁止

- 不训练、不改 reward/observation/controller/checkpoint/config；
- 不重跑 Goal27 的 C1/R1/H1/G1/A1，不扫 reward/gamma/seed；
- 不扫多个 dwell 值；本 Goal 只测预注册的 50 步；
- 不打开 held-out `100–149`；
- 不启动 SAC/REDQ/DroQ；
- 不 `git add -A`，不自动 commit/push。

## 9. Goal 提示词

```text
你现在执行 Goal65：Goal64 之后的 terminal plane-miss 短 dwell 反事实。

先读：
- doc/goals/goal65_plane_miss_dwell_20261110.md
- doc/goals/goal64_result_20261110.md
- doc/goals/goal27_result_20261109.md
- research/LEDGER.jsonl

硬约束：
1. 只测 50 步 dwell，不改 GOAL 半径/plane guard/reward/controller/checkpoint；
2. dwell=0 必须与 Goal63 immediate plane-miss 等价；
3. dev 20-59，held-out 100-149 关闭；
4. 10 个 candidate run 并行，CPU/float64/endpoint_stop；
5. 失败写 blocker/decision；不启动 SAC/REDQ/DroQ。

执行顺序：
- 实现 dwell + 单测 + smoke；
- 40-seed 复现检查；
- 10 路并行 candidate；
- paired analysis + H65-A/B/C；
- result/ledger/STATE/checker；
- 不自动 commit/push。
```
