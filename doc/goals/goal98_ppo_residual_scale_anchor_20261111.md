# Goal98：PPO residual scale 锚点修正与双侧网格

**日期**：2026-11-11<br>
**状态**：`SCALE_PARTIAL / TERMINAL_DIRECTION_REMAINS_BOTTLENECK / NO_SAC`<br>
**上游**：Goal97 `INVALID_SCALE_ANCHOR / NO_GATE_RECORD`<br>
**性质**：eval-only residual scale grid；不训练、不改 reward/terminal/success/goal radius，不打开 held-out。

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
  check_command: python ../scripts/check_research_ledger.py --goal Goal98 --goal-doc doc/goals/goal98_ppo_residual_scale_anchor_20261111.md --strict
```

本 Goal 的 Phase 0 是冻结的只读 screen。记录前不得修改 scale 集合、dev seed、
success frame、scenario、checkpoint 选择规则或 Gate 阈值。

## 1. 背景与错误修正

Goal97 预注册假定 `s=1.00` 是训练基线。实际 Goal95 配置：

```text
configs/lowlevel/goal95_20261111/p_d3_ppo.json:  pc.actuator_residual_scale = 1.5
configs/lowlevel/goal95_20261111/p_bc15_ppo.json: pc.actuator_residual_scale = 1.5
```

补跑锚点验证：

| arm/scenario | no override | override 1.50 |
|---|---:|---:|
| P_D3 default | 30/40 | 30/40 |
| P_D3 strict125 | 19/40 | 19/40 |
| P_BC15 default | 30/40 | 30/40 |
| P_BC15 strict125 | 19/40 | 19/40 |

所以 `1.50` 才是 no-op 基线，`1.00` 等价于把残差缩小 33%。Goal97 的
`0.25–1.25` 全部低于训练 scale，不能回答“PPO 残差是否需要放大/保持”。

## 2. 假设

### H98-A：1.50 是局部最优，scale 单因素不能恢复 frozen

如果成立：

- `1.50` 复现 D3/BC15 的 30/19；
- 所有 `s < 1.50` 和 `s > 1.50` 都不超过 30/19；
- 最佳 scale 仍是 `1.50`，frozen 31/20 更好。

则判 `SCALE_NOT_SUPPORTED`：下一步查 terminal residual 方向/信用，不进入
prior-preserving 训练，不开 SAC。

### H98-B：PPO 残差被训练/部署 scale 低估，放大到 1.75 或 2.00 有增益

如果成立：

- 某个 `s > 1.50` 让两个 arm 同时达到 default `>=31`、strict `>=20`；
- path 3D RMSE 平均恶化 `<=5%`；
- timeout/safety 不增加，trace 有限。

则判 `SCALE_SUPPORTED`：先记录 dev40 机制证据，再预注册 formal/held-out
复现；仍不直接换算法。

### H98-C：只有单臂或非 frozen 阈值改善

如果成立：

- 只有 D3 或 BC15 达到 31/20；
- 或两臂都超过 1.50 但仍低于 frozen；
- 或 success 改善但 path/closure 明显退化。

则判 `SCALE_PARTIAL`：只做归因，不进入训练，不扩 seed，不开 SAC。

## 3. 数据与隔离

- 训练：Phase 0 不训练；
- eval：dev seeds `20–59`，每 run 40 episodes；
- scenario：Goal85 unbiased default 与 strict125；
- checkpoint：Goal95 `P_D3/final.pt`、`P_BC15/final.pt`；
- checkpoint SHA 必须匹配 Goal95 result：
  - D3 `2d57745a811268553041c561f2ae1ad590dced472b712efd90cdd9c011c1a616`
  - BC15 `1a6a153d0d2e9548c2e14ebbe1b0fd33b65188a51be5958565b59885f494fe43`
- held-out `100–149` 禁止打开；
- success frame 固定 `both`；goal radius、terminal capture、reward 不改。

## 4. 预注册 scale grid

固定绝对 scale：

```text
s ∈ {0.75, 1.00, 1.25, 1.50, 1.75, 2.00}
```

`1.50` 是基线，必须复现 `30/19`；`1.25` 和 `0.75` 是缩幅侧，`1.75` 和
`2.00` 是放大侧。`0.00` 由 Goal96 已记录的 frozen-base `31/20` 代表，不重复跑。

## 5. 实验矩阵

```text
2 arms (P_D3, P_BC15)
× 2 scenarios (default, strict125)
× 6 scales
= 24 eval runs
```

每个 run：

- `--seed-start 20 --seed-count 40`；
- `--success-frame both`；
- CPU/float64；
- 单线程 BLAS，最多 24 个进程并行；
- 写 `summary.json/md` 和 `trace.npz`；
- 记录实际写回的 `actuator_residual_scale_override` 和 training config scale。

## 6. 指标

主指标：

- `success`（both-frame）；
- `success_pnav_tight`、`success_eta_tight_ever`、`success_eta_tight_at_done`；
- `timeout`、`safety`、`goal_plane_miss_rate`。

机制指标：

- path 3D/horizontal/vertical RMSE 相对 `1.50` 的比值；
- residual magnitude/rate/scale clip；
- terminal window `3–2 m`、`2–1.25 m` 的 radial closure；
- `actuator_residual_applied` 的 L2 和相对 `1.50` 的 ratio；
- 同 seed gained/lost 集合。

## 7. Gate

### 7.1 `INVALID`

任一条件成立：

- training config 的 `pc.actuator_residual_scale` 不等于 `1.50`；
- `1.50` 没有复现 D3/BC15 的 `30/19`；
- checkpoint SHA 不匹配；
- trace 缺字段、NaN/Inf、success 口径改变；
- held-out 被读取。

`INVALID` 不写成 `FAIL`。

### 7.2 `SCALE_SUPPORTED`

存在同一个 `s`，同时满足：

- D3 与 BC15 都达到 default `>=31`、strict `>=20`；
- 相对 `s=1.50` 的 path 3D RMSE 平均恶化 `<=5%`；
- timeout/safety 不增加；
- trace 有限且 scale 写回一致。

则记录 dev40 支持，下一步预注册 formal/held-out 复现；仍不换算法。

### 7.3 `SCALE_PARTIAL`

只有单 arm 达到 31/20，或两臂都超过 `1.50` 但低于 frozen，或 success
改善但 path/closure 明显退化。只做只读归因，不进入训练。

### 7.4 `SCALE_NOT_SUPPORTED`

所有 scale 都不超过 `1.50` 的 30/19，或最佳点仍是 frozen 31/20。则判
`PPO_RESIDUAL_SCALE_LOCAL_OPTIMUM`：下一步查 terminal residual 方向/信用，
不进入幅度约束训练，不开 SAC/REDQ/DroQ。

## 8. 执行计划

- [x] Step 0：写 Goal98 预注册；
- [x] Step 1：写 runner/analyzer，断言 config scale=1.50，加入单测；
- [x] Step 2：单 seed smoke，确认 1.50 与 no override 一致；
- [x] Step 3：24 个 dev40 summary 并行跑完；
- [x] Step 4：验证 1.50 复现 30/19，再计算 Gate；
- [x] Step 5：补 trace/closure 机制审计；
- [ ] Step 6：写 result/blocker/decision、ledger、STATE，并运行 checker。

## 9. 不做什么

- 不训练 PPO/SAC/REDQ/DroQ；
- 不改 reward、terminal、success、goal radius；
- 不扫 lr/gain/seed；
- 不扩 Goal95 seed；
- 不打开 held-out；
- 不用 Goal97 的无效 sweep 代替本 Goal 的 Gate；
- 不把 `1.25` 的改善写成“放大有效”。

## 10. Git

禁止 `git add -A`。只 add 本 Goal 相关文件；提交前运行：

```bash
python ../scripts/check_research_ledger.py --goal Goal98 \
  --goal-doc doc/goals/goal98_ppo_residual_scale_anchor_20261111.md --strict
git diff --cached --check
```

## 11. Goal 提示词

```text
你现在执行 Goal98：修正 PPO residual scale 锚点并跑双侧网格。

先读：
- doc/goals/goal97_result_20261111.md
- research/blockers/2026-11-11-goal97-scale-anchor-invalid.md
- research/decisions/2026-11-11-goal97-invalid-open-goal98.md
- doc/goals/goal98_ppo_residual_scale_anchor_20261111.md

硬约束：
1. 基线必须是 config 的真实 scale=1.50，不是 1.00；
2. grid 固定 0.75/1.00/1.25/1.50/1.75/2.00；
3. dev seeds 20–59，held-out 100–149 禁止打开；
4. 不改 reward/terminal/success/goal radius；
5. 不训练，不开 SAC/REDQ/DroQ；
6. 先验证 1.50 复现 30/19，否则直接 INVALID。

执行顺序：
- 写 runner/analyzer 和单测；
- 单 seed smoke；
- 24 个 dev40 summary；
- Gate 与 trace 审计；
- result/ledger/STATE/checker。
```
