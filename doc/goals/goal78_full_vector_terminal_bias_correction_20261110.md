# Goal78：full-vector terminal bias correction 只读审计

**日期**：2026-11-10<br>
**状态**：`FULL_VECTOR_DEADBAND_SUPPORTED`<br>
**上游**：Goal77 `GEOMETRY_GATE_NOT_SUPPORTED`<br>
**结果**：`doc/goals/goal78_result_20261110.md`

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
  check_command: .venv/bin/python ../scripts/check_research_ledger.py --goal Goal78 --goal-doc doc/goals/goal78_full_vector_terminal_bias_correction_20261110.md --strict
```

## 1. 背景与证据

Goal75 的 deadband 只沿 goal 方向修正 `p_nav-eta` 的投影 `b`。Goal76 在
tight 标签下，deadband 后仍剩 default `34`、strict125 `17` 个 false；
Goal77 证明这些 false 和 true 都大多停在 goal plane 前，离线 geometry gate
无法区分。

但 Goal75/76 的剩余 false 仍有：

```text
p_nav-eta error median: 0.3252 / 0.2925 m
along-goal b median:     0.1319 / 0.1456 m
```

这说明 `p_nav-eta` 误差并非全部沿 goal 方向；Goal75 的修正可能只处理了其中
一部分。Goal77 的 proper goal-frame 分解也显示，剩余 false 的 lateral 为
`0.3404 / -0.0021 m`、vertical 为 `-0.4242 / 0.0225 m`，并非纯沿-goal bias。

## 2. 假设与最便宜 falsification

### H78-A：full-vector deadband 能保留 tight true 并消除 tight false

对每个 terminal snapshot 计算完整误差向量 `e = p_nav - eta`，并做固定阈值
`b0=0.25 m` 的向量 deadband：

```text
e_norm = ||e||
correction = max(e_norm - 0.25, 0)
p_corr = p_nav - correction * e / e_norm
```

支持条件（测量前冻结）：default 与 strict125 都满足

1. tight false reduction `>=50%`；
2. tight true retention `>=95%`；
3. mismatch / schema `=0`。

证伪：任一 scenario 的 false 降幅或 true retention 未过线。

最便宜 falsification：只读复用 Goal71 episodes/trace，不训练、不新增 rollout。

### H78-B：只沿 goal 修正是不对称残差的主因

比较：

- A0：no correction；
- A1：Goal75 的 along-goal deadband；
- A2：full-vector deadband。

预测：A2 相对 A1 在两个 scenario 至少一个 false 降幅提高 `>=25` 个百分点，
且 true retention 下降不超过 `5` 个百分点。

证伪：A2 没有比 A1 多去掉 false，或 true 损失超过 `5` 个百分点。

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

不覆盖 Goal75/76/77 的 trace、JSON 或 MD。

## 4. 方法

对 terminal snapshot：

```text
d_p = ||p_nav - goal||
d_e = ||eta - goal||
e = p_nav - eta
u = (goal - eta) / ||goal - eta||
b = dot(e, u)
```

三个 arm：

```text
A0 none:        p0 = p_nav
A1 along:       p1 = p_nav - max(b - 0.25, 0) * u
A2 full-vector: p2 = p_nav - max(||e|| - 0.25, 0) * e / ||e||
```

标签仍分开报告：

```text
protocol_capture = d_e <= r_protocol
tight_capture    = d_e <= r_tight
success_arm      = ||p_arm - goal|| <= r_tight
tight_true_arm   = success_arm and tight_capture
tight_false_arm  = success_arm and not tight_capture
```

同时报告 A2 的全局 correction norm 中位数、与 A1 的 correction 差异、
每个 checkpoint 的 remaining false，以及 coordinate / schema mismatch。

## 5. 执行计划

- [x] Step 0：Run Contract、路径/SHA/checkpoint 审计；
- [x] Step 1：实现 analyzer + 单元测试；
- [x] Step 2：本地定向测试、Ruff；
- [x] Step 3：服务器 `--jobs 10` 只读运行，拉回 JSON/MD；
- [x] Step 4：按 Gate 判 H78-A/H78-B；
- [x] Step 5：写 result、blocker/ledger、STATE、strict checker、只 add 相关文件。

## 6. Gate

| check | 阈值 |
|---|---:|
| H78-A tight false reduction | 两个 scenario 都 `>=50%` |
| H78-A tight true retention | 两个 scenario 都 `>=95%` |
| H78-B A2 vs A1 false 增益 | 至少一个 scenario `>=25pp` |
| H78-B A2 true retention 损失 | 两个 scenario 都 `<=5pp` |
| mismatch / schema | 0 |

失败裁决：

| 观察 | 判读 | 下一步 |
|---|---|---|
| H78-A 通过 | 完整向量修正是可行的离线方向 | 预注册可观测 correction 或闭环 terminal capture arm |
| A2 好于 A1 但未过 Gate | 修正方向对，阈值/可观测性不足 | 回 terminal capture/reward，不扫阈值 |
| A2 不如 A1 | 误差不是简单 full-vector bias | 停止 bias correction，转动作参数化/terminal reference |
| mismatch / schema | `INVALID` | 只修数据后重跑 |

## 7. 算力

- 只读 10 个 run，`--jobs 10`，每进程单线程 BLAS；
- 0 GPU、不训练、不新增 rollout、不打开 held-out；
- 实测墙钟约 `1.77 s`、CPU 约 `870%`、RSS 约 `36 MB`，不为占满 384 CPU
  再拆 seed；
- 只有进入训练型 Goal79 才做 `num_envs 128/256/512` 与 batch 256/512
  的吞吐基准；
- SAC/REDQ/DroQ 继续关闭，除非出现独立的 optimization/exploration 或
  sample-efficiency 证据。

## 8. Git

禁止 `git add -A`。只 add 本 Goal 相关文件，提交前运行：

```bash
.venv/bin/python ../scripts/check_research_ledger.py \
  --goal Goal78 \
  --goal-doc doc/goals/goal78_full_vector_terminal_bias_correction_20261110.md \
  --strict
git diff --cached --check
```

不自动 commit/push，除非用户明确要求。
