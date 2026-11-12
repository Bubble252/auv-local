# Goal122：PPO residual gain sweep

**日期**：2026-11-11<br>
**状态**：`PROTOCOL_DRAFTED_BEFORE_MEASUREMENT / EVAL_ONLY / NO_TRAINING`<br>
**上游**：Goal121 `AUTHORITY_PRESENT / PPO_RESIDUAL_NEAR_ZERO`<br>
**性质**：冻结的 eval-only falsification；不训练、不改 reward/terminal/success/
goal radius，不打开 held-out。

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
  check_command: python3 ../scripts/check_research_ledger.py --goal Goal122 --goal-doc doc/goals/goal122_ppo_residual_gain_sweep_20261111.md --strict --require-result
```

记录前不得修改 gain 集合、seed、scenario、success frame、checkpoint、Gate 阈值。

## 1. 背景

Goal121 证明：

```text
动作权限存在：rpm- Δeta 0.334 m；stern- 22/40
PPO residual 只有 0.007–0.016 归一化单位，约为 BC base 的 3%
ppo-4x 仍 20/40，Δeta 只有 0.085 m
```

现在需要回答一个更便宜的问题：PPO 学到的方向是否只是“幅度太小”，还是
方向本身没有可用信号？

## 2. 固定输入

| 项目 | 值 |
|---|---|
| training config | Goal119 `C3_s86201.json` |
| checkpoint | Goal119 `C3_s86201/final.pt` |
| environment | Goal119 `C3.json`（strict125 + 15 deg/s pitch limit） |
| eval seeds | dev 20–59，40 episodes |
| success frame | `both` |
| eval scale / gains | 2.00 / `1,1,1` |
| action mode | `injected-residual` |
| held-out | `100–149` 关闭 |

`zero` profile 必须复现 B20 20/40；`ppo` profile 必须复现 Goal120 P0 20/40；
`ppo` 与原生 `policy` 的 residual 必须逐位一致（Goal121 已验证 `maxdiff=0.0`）。

## 3. Gain 矩阵

| profile | residual |
|---|---|
| `zero` | 0 |
| `ppo` | 1x |
| `ppo-4x` | 4x |
| `ppo-8x` | 8x |
| `ppo-16x` | 16x |
| `ppo-32x` | 32x |
| `ppo-64x` | 64x |

所有 gain 经过同一个 terminal gate、actuator scale 和 envelope；允许被 clamp，
clamp 本身作为权限/饱和证据记录。

## 4. 预注册假设

### H122-A：只是幅度太小

- 成立时应看到：success 或 reachable Δeta 随 gain 单调改善；8x/16x 至少
  达到 22/40 或 Δeta `>=0.25 m`。
- 失败时应看到：4x–64x 全部 20/40，Δeta 不随 gain 单调增长。

### H122-B：方向没有可用信号

- 成立时应看到：所有 gain 的 success `<22/40`，且 Δeta `<0.25 m`；
  放大只增加饱和/扰动，不增加 capture。
- 失败时应看到：任一 gain 达到 Gate。

### H122-C：放大后有权限但无 capture

- 成立时应看到：Δeta `>=0.25 m`，但 success `<22/40` 或下降。
- 失败时应看到：Δeta 和 success 同步改善。

### H122-D：放大造成伤害

- 成立时应看到：任一 gain success `<18/40`、safety>0 或 timeout 明显增加。
- 失败时应看到：timeout/safety 全 0，success 不低于 18。

## 5. Gate

| 观察 | 状态 | 下一步 |
|---|---|---|
| 任一 gain `>=8x` 的 success `>=22/40`，或 Δeta `>=0.25 m` | `GAIN_RECOVERS_CAPTURE` | 优先修 residual gain/scale，不换算法 |
| 任一 gain 的 Δeta `>=0.25 m`，但 success `<22/40` | `GAIN_AUTHORITY_WITHOUT_CAPTURE` | 查 reward/terminal/direction，不直接 SAC |
| 所有 gain success `<22/40` 且 Δeta `<0.25 m` | `PPO_DIRECTION_UNUSABLE` | 预注册 SAC 同协议对照 |
| 任一 gain success `<18/40`、safety>0 | `GAIN_HARM` | 查饱和/动作参数化 |
| 缺失 trace、NaN/Inf、zero/ppo 未复现 | `INVALID` | 修复执行链后重跑 |

## 6. 执行计划

- [x] Step 0：写预注册并冻结 gain/seed/Gate；
- [ ] Step 1：smoke `ppo-8x`；
- [ ] Step 2：7 profile × dev40 并行 eval；
- [ ] Step 3：汇总 success、Δeta、applied/effective、clip；
- [ ] Step 4：Gate 判定并写 result/blocker/decision/LEDGER/STATE；
- [ ] Step 5：checker 与 `git diff --check`；不自动 commit/push。

## 7. 与 SAC 的关系

只有 `PPO_DIRECTION_UNUSABLE` 才预注册 SAC。SAC 必须使用 Goal86 P-A 架构：

```text
bc_path_dim = 8
path endpoint = 4 -> 32 -> 64
加载 BC encoder
state/path/current -> z_t -> actor
不使用 Goal108 的 path12 随机初始化
```

若 `GAIN_RECOVERS_CAPTURE`，优先修 residual gain/scale；若
`GAIN_AUTHORITY_WITHOUT_CAPTURE`，先查 reward/terminal，不直接 SAC。

## 8. Git

禁止 `git add -A`。只 add 本次相关文件，提交前运行：

```bash
python3 ../scripts/check_research_ledger.py \
  --goal Goal122 \
  --goal-doc doc/goals/goal122_ppo_residual_gain_sweep_20261111.md \
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
