# Goal124：constant stern bias 的 default guardrail 与 strict 复现

**日期**：2026-11-11<br>
**状态**：`PROTOCOL_DRAFTED_BEFORE_MEASUREMENT / EVAL_ONLY / NO_TRAINING`<br>
**上游**：Goal123 `BIAS_IMPROVES_CAPTURE`（stern-025/030 24/40）<br>
**性质**：冻结的 eval-only guardrail；不训练、不改 reward/terminal/success/goal
定义，不打开 held-out。

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
  check_command: python3 ../scripts/check_research_ledger.py --goal Goal124 --goal-doc doc/goals/goal124_bias_default_guardrail_20261111.md --strict --require-result
```

记录前不得修改 profile、seed、scenario、success frame、checkpoint、Gate 阈值。

## 1. 背景

Goal123 在 strict125 + 15 deg/s pitch limit 下发现：

```text
stern-025  24/40  +29,31,33,56  0 lost
stern-030  24/40  +29,31,33,56  0 lost
```

在把它作为 PPO 先验之前，必须先确认它不伤害 default（2.0 m）场景。

## 2. 固定输入

| 项目 | 值 |
|---|---|
| training config | Goal119 `C3_s86201.json` |
| checkpoint | Goal119 `C3_s86201/final.pt` |
| reward | 与 Goal119 C3 完全一致（terminal_progress=8.0、approach_enter=5.0） |
| default env | C3 配置仅把 `plant.goal_radius_m` 改为 `2.0` |
| strict env | Goal119 `C3.json`（`goal_radius_m=1.25`） |
| eval seeds | dev 20–59，40 episodes |
| success frame | `both` |
| action mode | `injected-residual` |
| held-out | `100–149` 关闭 |

## 3. 实验矩阵

| profile | scenario | residual |
|---|---|---|
| `default_zero` | default 2.0 | `(0,0,0)` |
| `default_stern-025` | default 2.0 | `(0,-0.25,0)` |
| `default_stern-030` | default 2.0 | `(0,-0.30,0)` |
| `strict_zero` | strict125 | `(0,0,0)` |
| `strict_stern-025` | strict125 | `(0,-0.25,0)` |
| `strict_stern-030` | strict125 | `(0,-0.30,0)` |

## 4. 预注册假设

### H124-A：stern bias 伤害 default

- 成立时应看到：default stern 比 default zero 低 `>=2` success。
- 失败时应看到：default 下降 `<=1`。

### H124-B：strict +4 可复现

- 成立时应看到：strict_stern-025/030 再次 `>=23/40`，且 gained 集合与
  Goal123 接近。
- 失败时应看到：strict `<23/40` 或 gained 集合完全改变。

### H124-C：bias 是场景特定的

- 成立时应看到：default 与 strict 的 optimal bias 不同，或 default 对 bias
  不敏感。
- 失败时应看到：同一 bias 在两个场景都不伤害且 strict 提升。

## 5. Gate

| 观察 | 状态 | 下一步 |
|---|---|---|
| default 下降 `<=1`，strict 复现 `>=23/40` | `BIAS_GUARDRAIL_PASS` | 预注册 bias-initialized PPO（10M） |
| default 下降 `>=2` | `BIAS_DEFAULT_HARM` | 缩小 bias 或做场景门控，不训练 |
| strict `<23/40` | `BIAS_STRICT_NOT_REPRODUCED` | 重查 batch/seed/执行链，不训练 |
| safety>0、NaN/Inf、zero 未复现 | `INVALID` | 修复执行链后重跑 |

## 6. 执行计划

- [x] Step 0：写预注册并冻结 profile/seed/Gate；
- [ ] Step 1：default/strict 6 臂 × dev40 并行 eval；
- [ ] Step 2：汇总 default 损失、strict 复现、gained/lost；
- [ ] Step 3：Gate 判定并写 result/blocker/decision/LEDGER/STATE；
- [ ] Step 4：通过后预注册 bias-initialized PPO；不自动 commit/push。

## 7. 后续训练设计（Gate 通过才执行）

```text
base = frozen BC + constant stern bias (-0.25)
actor output = bias + learned residual
training = 10M PPO, same C3 reward, seeds 86201/86202
control = Goal120 P0 zero-bias PPO
```

先在 eval 中确认 bias 不伤 default，再训练。SAC 仍不启动。

## 8. Git

禁止 `git add -A`。只 add 本次相关文件，提交前运行：

```bash
python3 ../scripts/check_research_ledger.py \
  --goal Goal124 \
  --goal-doc doc/goals/goal124_bias_default_guardrail_20261111.md \
  --strict --require-result
git diff --cached --check
```

## 9. 自主迭代契约

本 Goal 执行 `research/AUTORESEARCH_LOOP.md`：

1. 失败后先归因，不直接换算法或扫 seed；
2. 每次 blocker 至少写两个可证伪 hypothesis；
3. 需要外部方法时使用 `paper-search` / `deep-research`，写 reading note；
4. 先跑最便宜的 falsification，再决定 `continue` / `pivot` / `stop`；
5. `pivot` 时写新 Goal，不修改失败 Gate 的阈值；
6. 持续迭代，直到出现可复现的 success 提升或触发停止条件；
7. 不自动 commit/push，不打开 held-out，不改变任务成功定义或主算法族。
