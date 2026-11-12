# Goal102：现有 PPO checkpoint 的剂量-反应审计

**日期**：2026-11-11<br>
**状态**：`COMPLETE / PPO_DOSE_HARM / EVAL_ONLY / NO_TRAINING`<br>
**上游**：Goal101 `BC_PRIOR_BEATS_ALOS`<br>
**性质**：eval-only、冻结 Gate；不训练、不改 reward/terminal/success/goal radius，不打开 held-out。

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
  check_command: python ../scripts/check_research_ledger.py --goal Goal102 --goal-doc doc/goals/goal102_ppo_dose_response_20261111.md --strict
```

本 Goal 记录前不得修改 checkpoint 集合、剂量标签、dev seed、success frame、
scenario、scale 或 Gate 阈值。

## 1. 背景

Goal101 已确认：

```text
B20 = frozen BC only + actuator_residual_scale 2.00
default 33/40, strict125 20/40
ALOS baseline: default 27/40, strict125 14/40
```

但 Goal100 的 `P0_20`（PPO 全开）只有：

```text
default 31/32, strict125 18/18
```

这说明“PPO 最终 checkpoint”有害，但还没有回答一个更窄的问题：

> 在同一个 Goal95 训练轨迹上，是否存在某个更早的 PPO 剂量，既保留
> B20 的 33/20，又把 strict 提到 21+？

如果存在，就不需要新训练；如果不存在，才值得实现真正的有界 PPO / KL-to-BC
训练。这个 Goal 先用现成 checkpoint 做最便宜的剂量-反应，不新开算法。

## 2. 数据与隔离

- 训练：不训练；
- action mode：`policy`（完整 PPO 增量）；
- eval：dev seeds `20–59`，每 run 40 episodes；
- scenario：Goal85 unbiased default、strict125；
- success frame：`both`；
- residual scale：`2.00`；
- checkpoint：Goal95 `model_initial.pt`、`model_10.pt`、`model_20.pt`、
  `model_30.pt`、`final.pt`；
- 两臂：`P_D3`、`P_BC15`；
- 参照：Goal100 `B20`、Goal101 `ALOS`；
- held-out `100–149` 禁止打开。

## 3. 剂量矩阵

| ID | checkpoint | 训练迭代 | 近似 env transitions | 说明 |
|---|---|---:|---:|---|
| `I0` | `model_initial.pt` | 0 | 0 | supervised residual prior |
| `D25` | `model_10.pt` | 10 | 250k | 最小 PPO 微调剂量 |
| `D50` | `model_20.pt` | 20 | 500k | 中剂量 |
| `D75` | `model_30.pt` | 30 | 750k | 高剂量 |
| `D100` | `final.pt` | 40 | 1M | Goal95 最终 PPO |

新跑矩阵：

```text
2 arms × 2 scenarios × 5 checkpoints = 20 eval runs
```

## 4. 假设

### H102-A：存在有界 PPO 窗口

如果成立：

- 某个 `D*`（不是 `D100`）在两臂都达到 default `>=33`、strict `>=21`；
- path 3D RMSE 相对 B20 不恶化超过 `5%`；
- timeout/safety 全 0；
- `D100` 仍低于该 checkpoint。

判读：当前训练轨迹上已有可用的有界 PPO 剂量；下一 Goal 只需做 checkpoint
选择/保守微调，不需要重训新算法。

### H102-B：所有 PPO 剂量都不超过 B20

如果成立：

- 所有 `D25/D50/D75/D100` 都至少有一个主场景低于 B20；
- 没有 checkpoint 达到 strict `>=21` 且 default `>=33`；
- `I0` 可能接近 B20，但一旦更新就退化。

判读：不是“训练不够久”，而是当前 PPO objective / credit assignment 与
terminal capture 冲突；下一 Goal 必须实现硬约束的 prior-preserving PPO，
不能继续扫 checkpoint。

### H102-C：早期剂量只改善 default、牺牲 strict

如果成立：

- 某些 checkpoint default `>=33`，但 strict `<=20`；
- 或 strict `>=21`，但 default `<33`。

判读：存在 trade-off，不是单调剂量问题；需要 terminal-aware bound 或
reward/terminal capture 审计，不能只选 checkpoint。

### H102-D：评价链无效

如果成立：

- checkpoint SHA 不匹配；
- B20 或 ALOS 参照不能复现；
- seed 集合不是 `20–59`；
- NaN/Inf 或 summary/trace 缺失。

判读：`INVALID`，先修评价链。

## 5. Gate

### 5.1 `INVALID`

- 任一 checkpoint SHA 与冻结值不一致；
- B20 不能复现两臂 `33/20`；
- ALOS 不能复现两臂 `27/14`；
- 缺 summary/trace；
- seed 集合不是 `20–59`；
- NaN/Inf 或读取 held-out。

### 5.2 checkpoint 合格条件

对 arm `a`、checkpoint `d`：

```text
default_success >= 33
strict125_success >= 21
path_3d_rmse <= 1.05 * B20_path_3d_rmse
timeout == 0
safety == 0
```

### 5.3 裁决

| 观察 | 裁决 |
|---|---|
| 同一个 `d` 在两臂都合格 | `BOUNDED_PPO_SUPPORTED` |
| 只有一臂有合格 checkpoint | `SINGLE_ARM_SIGNAL` |
| 没有合格 checkpoint，但至少一个 checkpoint 保留 B20 的 `33/20` | `PPO_DOSE_NEUTRAL` |
| 所有 checkpoint 都低于 B20 的 `33/20` | `PPO_DOSE_HARM` |

`PPO_DOSE_NEUTRAL` 不等于支持 PPO；它只说明某些剂量没有破坏先验。

## 6. 执行计划

- [x] Step 0：确认 checkpoint SHA 与参照结果；
- [x] Step 1：实现 20-run runner/analyzer；
- [x] Step 2：单测 + 1 seed smoke；
- [x] Step 3：远端 20 个 dev40 eval；
- [x] Step 4：按冻结 Gate 判断；
- [x] Step 5：补 result/decision、LEDGER、STATE，并运行 checker。

## 7. 不做什么

- 不训练 PPO/SAC/REDQ/DroQ；
- 不改 reward、terminal、success、goal radius；
- 不扩 seed，不打开 held-out；
- 不把 `model_initial` 当作“零 PPO”；它含 supervised residual prior；
- 不用最终 checkpoint 的结果代替剂量-反应。

## 8. Goal 提示词

```text
你现在执行 Goal102：现有 PPO checkpoint 的剂量-反应审计。

先读：
- doc/goals/goal102_ppo_dose_response_20261111.md
- doc/goals/goal101_result_20261111.md
- doc/goals/goal100_result_20261111.md

硬约束：
1. 固定 Goal95 的 5 个 checkpoint、Goal85 unbiased 环境、dev seeds 20-59、success frame both；
2. action_mode 固定 policy，residual scale 固定 2.00；
3. 不训练、不扩 seed、不打开 held-out、不换算法；
4. 不改 reward/terminal/success/goal radius；
5. 不把 model_initial 写成“零 PPO”；
6. 失败后先归因，不直接改阈值。

执行顺序：
- 校验 checkpoint SHA 与 B20/ALOS 参照；
- 跑 20 个 dev40 eval；
- 按冻结 Gate 判 BOUNDED_PPO_SUPPORTED / SINGLE_ARM_SIGNAL / PPO_DOSE_NEUTRAL / PPO_DOSE_HARM / INVALID；
- 写 result/decision、LEDGER、STATE，并运行 checker。
```

## 9. 自主迭代契约

执行 `research/AUTORESEARCH_LOOP.md`。失败后先归因、写至少两个可证伪
hypothesis、查论文/代码、跑最便宜 falsification、写 LEDGER decision；不自动
commit/push，不打开 held-out，不改变任务成功定义或主算法族。
