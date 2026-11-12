# Goal141：终止条件与 success 契约只读审计

**日期**：2026-11-11<br>
**状态**：`COMPLETED_WITH_RESULT`<br>
**性质**：eval-only 只读审计；不训练，不改 reward、termination、success、goal radius，不打开 held-out。

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
  check_command: python ../scripts/check_research_ledger.py --goal Goal141 --goal-doc doc/goals/goal141_terminal_contract_audit_20261111.md --strict
```

## 1. 背景与证据

当前 PC success 主要由几何 `GOAL` 决定；`BatchedPCTerminationManager` 的
endpoint-stop 分支注释明确写着：

```text
GOAL > GOAL_PLANE_MISS > timeout/safety/bounds
```

现有 safety kind 只包含越界、NaN/Inf、invalid state、深度限制、横滚角和
横滚角速度；没有俯仰角、俯仰角速度、垂向速度或 terminal vertical-error
包络。Goal117/131 又显示 nominal failure 主要是 `GOAL_PLANE_MISS` 和
approach geometry，而不是 safety/timeout。

因此必须把两个问题分开：

1. **success 定义太弱**：已经标成 `GOAL` 的 episode，terminal 姿态/速度是否
   已经超出可接受包络；
2. **终止优先级吞掉 safety**：同一 terminal step 已满足 roll/depth/bounds
   等 safety 条件时，是否仍被 `GOAL` 或 `GOAL_PLANE_MISS` 覆盖。

本机没有 Goal95 `P_D3/final.pt`，也没有 Goal117 原始 trace。可用同族资产是
Goal86 P-A residual BC checkpoint 与 strict125 环境配置；本 Goal 的闭环数字
严格是 **BC-residual proxy**，不能替代 PPO final 的正式结论。

## 2. 假设

### H141-A：当前 GOAL 携带不可接受的 terminal 包络

固定候选阈值（测量前固定，不用结果调参）：

| 量 | 候选不可接受阈值 |
|---|---:|
| `abs(pitch)` | `>= 45 deg` |
| `abs(pitch_rate q)` | `>= guidance.pitch_rate_limit_rad_s`（15 deg/s） |
| `abs(heave w)` | `>= 0.50 m/s` |
| `abs(vertical_error)` | `>= 1.00 m` |

- 成立：至少 3/40 个 `GOAL` episode 命中任一候选条件；
- 失败：`GOAL` episode 命中数 `<= 1/40`；
- 中间：`2/40`，记为 `WEAK_CONTRACT_GAP`。

### H141-B：GOAL / plane-miss 覆盖已触发的 safety

- 成立：至少 1 个 terminal episode 同时满足 `kind in {GOAL, GOAL_PLANE_MISS}`
  与 raw safety predicate；
- 失败：0 个优先级吞并；
- 同时报告 roll dwell、roll-rate dwell、depth、bounds、invalid-state 分项。

### H141-C：缺失的 terminal 条件能解释当前 failure

- 成立：多数 failure（`>= 50%`）在终止时也命中候选包络，且这些 failure
  不是单纯几何 miss；
- 失败：多数 failure 是 `GOAL_PLANE_MISS` 且 terminal 包络正常，或 failure
  已在 2 m entry 被 geometry 解释。

## 3. 数据与隔离

- 环境：`configs/lowlevel/pc_goal85_unbiased_strict125_candidate_20261110.json`
- 训练契约：`configs/lowlevel/goal86/pc_goal86_p-a_ppo_formal_20261110.json`
- checkpoint：`artifacts/lowlevel/goal86/pa_residual_bc_path12_cmd3_current_20261110/pc_bc_actor.pt`
- dev seeds：`20-59`；held-out `100-149` 关闭；
- 对照臂：`policy`（BC residual）与 `zero-residual`（ALOS/PD-PI base）；
- 不新增训练，不选 checkpoint，不改 success frame 或 goal radius。

## 4. 执行计划

- [x] Step 0：写本 Run Contract，固定候选阈值；
- [x] Step 1：实现只读审计脚本，记录每个 terminal 的 kind、eta/nu、路径几何、
      roll/rate dwell、distance-to-bounds 与 raw safety predicate；
- [x] Step 2：运行 termination manager 单元测试，验证 endpoint-stop 优先级；
- [x] Step 3：跑 `policy` 与 `zero-residual` 各 40 dev seeds；
- [x] Step 4：按 A/B/C 判读，写 result md；
- [x] Step 5：更新 `research/STATE.md` 与 ledger；不自动 commit/push。

## 5. 运行命令

```bash
PYTHONPATH=src .venv/bin/python scripts/audit_pc_terminal_contract.py \
  --training-config configs/lowlevel/goal86/pc_goal86_p-a_ppo_formal_20261110.json \
  --environment-config configs/lowlevel/pc_goal85_unbiased_strict125_candidate_20261110.json \
  --checkpoint artifacts/lowlevel/goal86/pa_residual_bc_path12_cmd3_current_20261110/pc_bc_actor.pt \
  --action-mode policy \
  --output artifacts/lowlevel/goal141_terminal_contract_20261111/policy \
  --seed-start 20 --seed-count 40 --device cpu --dtype float64

PYTHONPATH=src .venv/bin/python scripts/audit_pc_terminal_contract.py \
  --training-config configs/lowlevel/goal86/pc_goal86_p-a_ppo_formal_20261110.json \
  --environment-config configs/lowlevel/pc_goal85_unbiased_strict125_candidate_20261110.json \
  --action-mode zero-residual \
  --output artifacts/lowlevel/goal141_terminal_contract_20261111/zero \
  --seed-start 20 --seed-count 40 --device cpu --dtype float64

PYTHONPATH=src .venv/bin/python -m pytest -q \
  tests/test_pc_goal141_terminal_contract.py
```

## 6. Gate

| check | 阈值 |
|---|---:|
| episode 完整终止 | `40/40`，无超时未记录 |
| 非有限 terminal 字段 | `0` |
| H141-A | 报告 `GOAL` 命中数与 seed 列表 |
| H141-B | 报告优先级吞并与分项 safety 计数 |
| H141-C | 报告 failure kind 分布及候选包络命中率 |

失败裁决：

| 观察 | 判读 | 下一步 |
|---|---|---|
| GOAL terminal 大量越出候选包络 | `SUCCESS_CONTRACT_GAP` | 预注册新的 terminal envelope，不改 Goal140 reward |
| GOAL/plane-miss 覆盖 safety | `TERMINATION_PRIORITY_GAP` | 单独 Goal 修 priority，并补 co-occurrence 测试 |
| failure 主要是 plane-miss 且 terminal 包络正常 | `MISSING_TERMINAL_NOT_PRIMARY` | 不改 success；继续 approach geometry / credit |
| 两者都弱 | `CONTRACT_GAP_NOT_PRIMARY` | 回到 Goal140 的 tracking reward |

## 7. Git

禁止 `git add -A`。只 add 本 Goal 的 goal/result 文档、审计脚本、测试和
`research/STATE.md`；不提交 `artifacts/`、checkpoint、日志或缓存。提交前运行：

```bash
python ../scripts/check_research_ledger.py --goal Goal141 \
  --goal-doc doc/goals/goal141_terminal_contract_audit_20261111.md --strict
git diff --cached --check
```

用户明确要求前不 commit / push。

## 8. Goal 提示词

```text
你现在执行 Goal141：只读审计 PC termination/success 契约。

硬约束：
1. 不训练，不改 reward/terminal/success/goal radius；
2. 不打开 held-out；
3. P_D3 不可用时只报告 BC-residual proxy，不写成 PPO 结论；
4. 先测 GOAL 覆盖 safety，再测 GOAL terminal envelope，最后测 failure 包络；
5. 结果只用于决定是否新开 terminal-contract Goal，不直接改 Goal140。
```

