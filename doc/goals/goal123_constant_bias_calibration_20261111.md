# Goal123：constant stern/rpm bias 标定

**日期**：2026-11-11<br>
**状态**：`PROTOCOL_DRAFTED_BEFORE_MEASUREMENT / EVAL_ONLY / NO_TRAINING`<br>
**上游**：Goal122 `GAIN_HARM / GAIN_AUTHORITY_WITHOUT_CAPTURE`<br>
**性质**：冻结的 eval-only 标定；不训练、不改 reward/terminal/success/goal
radius，不打开 held-out。

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
  check_command: python3 ../scripts/check_research_ledger.py --goal Goal123 --goal-doc doc/goals/goal123_constant_bias_calibration_20261111.md --strict --require-result
```

记录前不得修改 bias 集合、seed、scenario、success frame、checkpoint、Gate 阈值。

## 1. 背景

Goal121/122 已证明：

```text
动作权限存在；
常数 stern -0.20 给 22/40（+2，无丢失）；
PPO 方向放大到 16x/32x 有 Δeta，但 success 不升反降；
64x 掉到 13/40。
```

因此下一步不是 SAC，而是把常数 `stern-` 方向做细标定，找是否存在更稳定的
净增益 bias。常数 bias 也是后续 bias-initialized PPO/SAC 的先验。

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

`zero` 必须复现 B20 20/40。

## 3. Bias 矩阵

| profile | residual `(rudder, stern, rpm)` |
|---|---|
| `zero` | `(0,0,0)` |
| `stern-005` | `(0,-0.05,0)` |
| `stern-010` | `(0,-0.10,0)` |
| `stern-015` | `(0,-0.15,0)` |
| `stern-020` | `(0,-0.20,0)` |
| `stern-025` | `(0,-0.25,0)` |
| `stern-030` | `(0,-0.30,0)` |
| `rpm-010` | `(0,0,-0.10)` |
| `rpm-020` | `(0,0,-0.20)` |
| `stern-020_rpm-010` | `(0,-0.20,-0.10)` |

所有 bias 经过同一个 terminal gate、actuator scale 和 envelope。

## 4. 预注册假设

### H123-A：存在稳定净增益 stern bias

- 成立时应看到：某个 stern bias success `>=23/40`，lost `<=1`。
- 失败时应看到：所有 stern bias `<=22/40`，或 lost `>=2`。

### H123-B：rpm bias 也能稳定改善

- 成立时应看到：`rpm-010` 或 `rpm-020` success `>=22/40` 且 lost `<=1`。
- 失败时应看到：rpm bias 只做 seed 交换或下降。

### H123-C：stern + rpm 可叠加

- 成立时应看到：combined 比最佳单通道高 `>=1` success。
- 失败时应看到：combined 不高于最佳单通道。

### H123-D：+2 只是边界噪声

- 成立时应看到：更细 bias 扫描没有 `>=23/40`，成功集合不稳定。
- 失败时应看到：至少一个 bias `>=23/40` 且 lost `<=1`。

## 5. Gate

| 观察 | 状态 | 下一步 |
|---|---|---|
| 任一 bias success `>=23/40`，lost `<=1`，safety=0 | `BIAS_IMPROVES_CAPTURE` | 预注册 bias-initialized PPO / bias + RL residual |
| 最佳 bias 21–22/40 | `BIAS_WEAK` | 保守保留方向，继续诊断 reward/terminal |
| 最佳 bias `<=20/40` | `BIAS_NONE` | 不再追常数 bias，考虑 SAC 或其他动作参数化 |
| safety>0、NaN/Inf、zero 未复现 | `INVALID` | 修复执行链后重跑 |

## 6. 执行计划

- [x] Step 0：写预注册并冻结 bias/seed/Gate；
- [ ] Step 1：10 profile × dev40 并行 eval；
- [ ] Step 2：汇总 success、gained/lost、Δeta、applied/effective；
- [ ] Step 3：Gate 判定并写 result/blocker/decision/LEDGER/STATE；
- [ ] Step 4：checker 与 `git diff --check`；不自动 commit/push。

## 7. 与后续 RL 的关系

若 `BIAS_IMPROVES_CAPTURE`：

- 优先预注册 **bias-initialized PPO**：actor 输出层初始化为最佳常数 bias，
  再训练 10M，检验 RL 是否能在 bias 上继续提升；
- 或预注册 **bias + residual RL**：base = BC + constant bias，RL 只学剩余残差；
- 仍不换算法，不打开 held-out。

若 `BIAS_NONE`：

- 常数方向不成立，再考虑 SAC 同协议对照；
- SAC 必须使用 Goal86 P-A 架构（`bc_path_dim=8`、endpoint `4→32→64`、
  加载 BC encoder、z_t 融合）。

## 8. Git

禁止 `git add -A`。只 add 本次相关文件，提交前运行：

```bash
python3 ../scripts/check_research_ledger.py \
  --goal Goal123 \
  --goal-doc doc/goals/goal123_constant_bias_calibration_20261111.md \
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
