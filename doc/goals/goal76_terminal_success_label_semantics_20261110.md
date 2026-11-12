# Goal76：terminal success label semantics 与 deadband 重评估

**日期**：2026-11-10<br>
**状态**：`LABEL_CONFLATION_CONFIRMED_BUT_DEADBAND_FAIL`<br>
**上游**：Goal75 `PARTIAL_SUPPORT` / strict125 label conflation<br>
**结果**：`doc/goals/goal76_result_20261110.md`

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
  check_command: .venv/bin/python ../scripts/check_research_ledger.py --goal Goal76 --goal-doc doc/goals/goal76_terminal_success_label_semantics_20261110.md --strict
```

## 1. 背景与证据

Goal75 的正式 Gate 是 `PARTIAL_SUPPORT`：deadband `b0=0.25 m` 在两个
scenario 都把 false-success 降到 0，但 strict125 的 true retention 只有
`93.18%`，差 3 个样本没过预注册 `95%`。

逐样本诊断发现这 3 个 strict125 “true loss”中：

- 2 个 true `eta` 在 tight radius 外，只在 protocol radius 内；
- 1 个 true `eta` 在 tight radius 内（seed 57，`d_e≈0.956 m`）。

根因是标签定义混用：

```text
success            = ||p_nav - goal|| <= r_tight       # 1.75 / 1.00 m
protocol_capture   = ||eta - goal|| <= r_protocol      # 2.00 / 1.25 m
perfect_tight      = ||eta - goal|| <= r_tight         # 完美估计器下的 tight success
```

Goal74/75 的 `true_success` 使用 `protocol_capture`，因此会把“协议半径内、
tight 半径外”的样本算成 true。default 有 38/97、strict125 有 19/44 属于
这种 protocol-only 情况。

## 2. 假设与最便宜 falsification

### H76-A：在 tight 标签下，deadband 修正能保留 tight true 且消除 tight false

机制：如果 Goal75 的 strict125 失败主要是标签混淆，而不是修正方向错误，
那么把 true 标签改成 `d_e <= r_tight` 后，deadband 的 tight true retention
应该回到 `>=95%`，同时 tight false 仍下降至少 `50%`。

支持条件（测量前冻结）：

1. default 与 strict125 的 tight false-success 都下降至少 `50%`；
2. 两个 scenario 的 tight true-success 保留率都至少 `95%`；
3. 不改变 deadband `b0=0.25 m`，不扫新阈值。

最便宜 falsification：只读复用 Goal71 episodes/trace，重算 tight 标签。

### H76-B：protocol 标签系统性高估 tight true success

机制：protocol radius 比 tight radius 宽 `0.25 m`，在 capture 边界附近会
把大量 protocol-only 样本算成 true。若这种高估显著，Goal75 的 strict125
失败就不能直接解释为 deadband 方向错误。

支持条件（测量前冻结）：

1. `base_tight_true / base_protocol_true <= 0.70` 在两个 scenario 都成立；
2. 两个 scenario 的 protocol-only 样本数都 `>=10`；
3. 该结论不依赖 deadband 修正，只看 no-correction 标签。

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

不覆盖 Goal67/71/74/75 的 trace、JSON 或 MD。服务器只读运行，结果写入
Goal76 隔离目录。

## 4. 方法

对每个 terminal snapshot 计算：

```text
d_p = ||p_nav - goal||
d_e = ||eta - goal||
u = (goal - eta) / ||goal - eta||
b = dot(p_nav - eta, u)
p_nav_deadband = p_nav - max(b - 0.25, 0) * u
```

标签：

```text
protocol_capture = d_e <= r_protocol
tight_capture    = d_e <= r_tight
success_none     = d_p <= r_tight
success_deadband = ||p_nav_deadband - goal|| <= r_tight
```

统计：

```text
protocol_true = success and protocol_capture
tight_true    = success and tight_capture
tight_false   = success and not tight_capture
```

同时报告：

- no-correction 的 protocol/tight 计数；
- deadband 的 protocol/tight 计数；
- tight false reduction fraction；
- tight true retention；
- protocol-only 样本数；
- 高 `b` true/false 的已有 trace 特征对比（描述性，不作 Gate）。

## 5. 执行计划

- [x] Step 0：Run Contract、路径/SHA/配置审计；
- [x] Step 1：实现 analyzer + 单元测试；
- [x] Step 2：本地定向测试、Ruff；
- [x] Step 3：服务器 `--jobs 10` 只读运行，拉回 JSON/MD；
- [x] Step 4：按 Gate 判 H76-A/H76-B；
- [x] Step 5：写 result、blocker/ledger、STATE、strict checker、只 add 相关文件。

## 6. Gate

| check | 阈值 |
|---|---:|
| H76-A tight false reduction | 两个 scenario 都 `>= 50%` |
| H76-A tight true retention | 两个 scenario 都 `>= 95%` |
| H76-B tight/protocol true ratio | 两个 scenario 都 `<= 0.70` |
| H76-B protocol-only samples | 两个 scenario 都 `>= 10` |
| schema / mismatch | 0 mismatch，字段完整 |

失败裁决：

| 观察 | 判读 | 下一步 |
|---|---|---|
| H76-A 通过 | deadband 在正确标签下可用 | 预注册可观测 `b` predictor 或闭环 arm |
| H76-A 失败但 H76-B 通过 | 标签混淆解释不了全部损失 | 回动作/reward/terminal reference |
| 两个都失败 | 标签不是主因 | 回动作/reward/terminal reference |
| mismatch / 数据不一致 | `INVALID` | 只修数据后重跑 |

## 7. 算力

- 只读分析，`--jobs 10`，每进程单线程 BLAS；
- 正式审计墙钟约 `2.79 s`、CPU 约 `922%`、RSS 约 `36 MB`、0 张 GPU；
- 追加的按 checkpoint 残差分解墙钟约 `4.03 s`，同为 10 进程、0 张 GPU；
- 不训练、不新增 rollout、不打开 held-out；
- 该阶段只有 10 个 run 级任务，继续增加并发没有收益；耗时更长的分析才按
  run×seed 继续分片；
- 8×4090 留给后续 predictor/training arm，不因 GPU 空闲而启动新算法；
- 服务器使用隔离目录，不覆盖 Goal67/71/74/75 结果。

## 8. Git

禁止 `git add -A`。只 add 本 Goal 相关文件，提交前运行：

```bash
.venv/bin/python ../scripts/check_research_ledger.py \
  --goal Goal76 \
  --goal-doc doc/goals/goal76_terminal_success_label_semantics_20261110.md \
  --strict
git diff --cached --check
```

不自动 commit/push，除非用户明确要求。

## 9. Goal 提示词

```text
你现在执行 Goal76：terminal success label semantics 与 deadband 重评估。

先读：
- doc/goals/goal76_terminal_success_label_semantics_20261110.md
- doc/goals/goal75_result_20261110.md
- research/blockers/2026-11-10-goal75-deadband-strict125-label-conflation.md

硬约束：
1. 只读复用 Goal71 episodes/trace，不训练、不新增 rollout、不打开 held-out；
2. protocol_capture 与 perfect_tight_success 必须分开报告；
3. 不调 deadband 阈值，不扫新阈值；
4. 只有 H76-A 通过才允许讨论可观测 predictor；
5. 不换 SAC/REDQ/DroQ，不加控制 authority；
6. 禁止 git add -A；只 add 本次相关文件。
```
