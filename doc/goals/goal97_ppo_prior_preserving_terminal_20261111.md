# Goal97：PPO 先验保持与 terminal residual 幅度预注册

**日期**：2026-11-11<br>
**状态**：`INVALID_SCALE_ANCHOR / NO_GATE_RECORD`<br>
**上游**：Goal96 `DIRECTION_NOT_SUPPORTED / PPO_DEGRADES_PRIOR / NO_SAC`<br>
**性质**：Phase 0 为 eval-only residual scale sweep；在 Phase 0 记录前不训练、不改 reward/terminal/success/goal radius，不打开 held-out。

> 结果：预注册把 `s=1.00` 当成基线，但 Goal95 配置的真实
> `pc.actuator_residual_scale` 是 `1.50`。`1.50` no-op 复现 Goal95 的
> 30/19，而 `1.00` 只有 29/16。本次 sweep 全部低于训练 scale，记为
> `INVALID`，不产生 Gate；修正版见
> `doc/goals/goal98_ppo_residual_scale_anchor_20261111.md`。

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
  check_command: python ../scripts/check_research_ledger.py --goal Goal97 --goal-doc doc/goals/goal97_ppo_prior_preserving_terminal_20261111.md --strict
```

本 Goal 的 Phase 0 是冻结的只读 screen。记录前不得修改 scale 集合、dev seed、
success frame、scenario、checkpoint 选择规则或 Gate 阈值。

## 1. 背景与证据

Goal96 已完成：

- terminal-window direction 的 12/12 run CI 跨 0 -> `DIRECTION_NOT_SUPPORTED`；
- terminal authority survival 最低 `0.9192` -> `AUTHORITY_SUFFICIENT`；
- 名义 terminal positive share 约 `1.00`，形式
  `REWARD_CREDIT_MISALIGNED`，但折扣后 terminal 只占 `1.2%–1.4%`；
- initial 31/19、final 30/19、frozen 31/20；
- final/initial residual mean cosine `0.907–0.969`，L2 ratio
  `1.10–1.52`，closure delta `-0.035–0.000`；
- default 唯一丢失 seed 26；D3 的 L2 放大 `2.29×`，closure 从 `1.2066`
  降到 `0.8846`。

所以当前最具体的机制假设是：PPO 把已有监督残差放大，但没有学出新的终端
方向。先做最便宜的 eval-only 反事实：把同一个 final checkpoint 的 residual
按不同 scale 缩放，观察是否能恢复 frozen BC 的 31/20，同时改善 terminal
closure。这个实验不需要训练，不会改变 checkpoint。

## 2. 假设

### H97-A：幅度放大是主要原因

如果成立：

- 某个 `s < 1` 的 scale 在同一 final checkpoint 上把 success 从 30/19
  提升到至少 frozen 的 31/20；
- residual L2 降低，terminal radial closure 改善；
- path 3D RMSE 不明显恶化。

如果失败：

- 所有 `s < 1` 都不超过 final 的 30/19；或 success 改善但 closure 不改善。

### H97-B：方向/信用分配仍是主要瓶颈

如果成立：

- 只改幅度最多恢复 boundary seed，strict 仍停在 19/20；
- `action_effect_score` 仍不随 scale 出现 success/failure 分离；
- frozen BC 仍优于所有 PPO scale。

如果失败：

- 一个固定 scale 同时改善 default 和 strict，并让 PPO 超过 frozen。

### H97-C：PPO residual 净有害

如果成立：

- 最佳 scale 位于 `0.0`（frozen BC）；
- 所有 `s > 0` 都不超过 `31/20`；
- final residual 只在放大噪声/先验误差。

如果失败：

- 任一 `s > 0` 超过 frozen 且 path RMSE 不退化。

## 3. 数据与隔离

- 训练：Phase 0 不训练；
- eval：`dev seeds 20–59`，共 40 个 episode/run；
- scenario：Goal85 unbiased default 与 strict125；
- checkpoint：`P_D3/final.pt`、`P_BC15/final.pt`；
- held-out：`100–149` 禁止打开；
- 不改 success frame（`both`）、goal radius、terminal capture 或 reward；
- 不把 val/dev 结果直接写成 formal；Phase 0 仅作为机制 screen。

## 4. Phase 0：eval-only scale sweep

### 4.1 scale 集合

固定：

```text
s ∈ {0.25, 0.50, 0.75, 1.00, 1.25}
```

`0.0` 由已有 frozen-base 评价代表，不重复跑。`1.0` 是同一个 final checkpoint
的基线；`1.25` 是负控，用来检查继续放大是否更差。

### 4.2 arm × scenario × scale

```text
2 arms (P_D3, P_BC15)
× 2 scenarios (default, strict125)
× 5 scales
= 20 eval runs
```

每个 run：

- `--seed-start 20 --seed-count 40`；
- `--success-frame both`；
- CPU/float64，按 8 进程并行；
- 记录 `summary.json/md`；
- 记录 trace 以便计算 terminal closure 和 residual L2。

### 4.3 命令模板

```bash
/data/yhx/underwater-robotics/envs/venvs/isaaclab/bin/python \
  scripts/evaluate_pc_ppo.py \
  --training-config configs/lowlevel/goal95_20261111/p_d3_ppo.json \
  --checkpoint artifacts/lowlevel/goal95_supervised_residual_20261111/P_D3/final.pt \
  --environment-config configs/lowlevel/pc_goal85_unbiased_default_candidate_20261110.json \
  --output artifacts/lowlevel/goal97_prior_preserving_20261111/scale_sweep/P_D3_default_s050 \
  --seed-start 20 --seed-count 40 --success-frame both --device cpu \
  --actuator-residual-scale-override 0.50 \
  --trace-output artifacts/lowlevel/goal97_prior_preserving_20261111/scale_sweep/P_D3_default_s050/trace.npz
```

## 5. 指标

主指标：

- `success`（summary 的 both-frame success）；
- `goal_capture_pnav_rate`、`goal_capture_eta_rate`（诊断）；
- `success_pnav_tight`、`success_eta_tight_ever`、`success_eta_tight_at_done`。

机制指标：

- residual L2 相对 scale 1.0 的变化；
- terminal window `3–2 m`、`2–1.25 m` 的 mean radial closure 和
  `action_effect_score`；
- residual magnitude/rate clip、survival；
- path 3D RMSE、plane-miss、timeout、safety。

每个 scale 与 scale 1.0 做同 seed 配对，报告 gained/lost seed 集合。

## 6. Gate

Phase 0 只决定是否进入 Phase 1 训练，不直接产生 formal 结论。

### 6.1 `SCALE_SUPPORTED`

同时满足：

- 两个 arm 在同一个 `s ∈ {0.50, 0.75}` 下达到 default `>=31`、
  strict `>=20`；
- 相对 scale 1.0 的 success 不下降；
- path 3D RMSE 平均恶化 `<=5%`；
- timeout/safety/NaN 不增加。

则进入 Phase 1：预注册训练时的 terminal prior-preserving 约束。

### 6.2 `SCALE_PARTIAL`

只有单个 arm 达到 `31/20`，或 success 改善但 terminal closure 没改善。

则先做只读归因，不直接扩 seed；可以预注册一个更窄的 prior-preserving
screen，但不得直接开 formal。

### 6.3 `SCALE_NOT_SUPPORTED`

所有 `s > 0` 都不超过 scale 1.0 的 30/19，或最佳点是 frozen 的 31/20。

则判 `PPO_RESIDUAL_NET_HARMFUL`：下一步查 terminal residual 的方向/信用，
不进入幅度约束训练，不开 SAC。

### 6.4 `INVALID`

checkpoint/config hash 不匹配、success 口径改变、held-out 被读取、trace 缺字段
或 NaN/Inf，记为 `INVALID`，不写成 FAIL。

## 7. 执行计划

- [x] Step 0：写 Goal97 预注册；
- [x] Step 1：写 scale sweep 汇总器与单测；
- [x] Step 2：单个 1-seed smoke 验证命令与输出；
- [x] Step 3：远端 20 个 dev40 eval 并行跑完；
- [x] Step 4：审计发现 scale 锚点错误，按 `INVALID` 记录，不记录 Gate；
- [ ] Step 5：写 result/blocker/decision、ledger、STATE，并运行 checker。

## 8. 不做什么

- 不训练 PPO/SAC/REDQ/DroQ（Phase 0）；
- 不改 reward、terminal、success、goal radius；
- 不扩 Goal95 seed；
- 不扫 lr/gain/seed；
- 不打开 held-out；
- 不把 eval-only scale override 写成训练时已经修好。

## 9. Git

禁止 `git add -A`。只 add 本 Goal 相关文件；提交前运行：

```bash
python ../scripts/check_research_ledger.py --goal Goal97 \
  --goal-doc doc/goals/goal97_ppo_prior_preserving_terminal_20261111.md --strict
git diff --cached --check
```

## 10. Goal 提示词

```text
你现在执行 Goal97：PPO prior-preserving terminal residual。

先读：
- doc/goals/goal96_result_20261111.md
- doc/goals/goal97_ppo_prior_preserving_terminal_20261111.md
- research/blockers/2026-11-11-goal96-terminal-credit-prior-degradation.md

硬约束：
1. Phase 0 只做 eval-only scale sweep，不训练；
2. dev seeds 固定 20–59，held-out 100–149 禁止打开；
3. success frame、goal radius、reward、terminal 均不改；
4. 不换 SAC/REDQ/DroQ；
5. 先跑最便宜的 1-seed smoke，再跑 20 个 dev40 run；
6. Gate 记录前不改 scale 集合和阈值。

执行顺序：
- 写 scale sweep 汇总器和单测；
- 远端 smoke；
- 8 进程并行跑 20 个 eval；
- 按第 6 节裁决，写 result/ledger/STATE；
- 只有 SCALE_SUPPORTED 才预注册 Phase 1 训练。
```
