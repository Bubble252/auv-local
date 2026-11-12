# Goal116：guidance pitch rate-limit 因果筛选

**日期**：2026-11-11<br>
**状态**：`COMPLETED_EVAL_ONLY / FILTER_NONCAUSAL / NO_SAC / HELDOUT_CLOSED`<br>
**上游**：Goal115 `REFERENCE_FILTER_LIMITED`；Goal114 `MIXED_PITCH_HANDOFF`<br>
**性质**：eval-only 反事实筛选；不训练、不改 checkpoint、不改 reward/terminal/success/goal radius，不打开 held-out。

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
  check_command: python3 ../scripts/check_research_ledger.py --goal Goal116 --goal-doc doc/goals/goal116_reference_filter_causality_screen_20261111.md --strict --require-result
```

这是冻结的预注册 Gate。记录前不得修改 arms、seeds、scenario、checkpoint、
success frame、主指标或判读阈值。

## 1. 要回答的问题

Goal115 证明 2 m 前失效链路经过 guidance command rate limiter：

- 11 个原 failure 中 `REFERENCE_FILTER_LIMITED=8`；
- 8 个 mixed 子集中 filter `6/8`；
- failure 的 `limited_at_limit_fraction=0.853`，success control `0.495`；
- 但 success control 的 `clip_fraction=0.931`，说明高频 target 和 clipping 并非
  failure 独有。

因此必须回答：

```text
15 deg/s 的 pitch command rate limiter 是否对闭环结果有因果影响？
  A. 提高限幅改善：raw target 的方向信息被 filter 削掉；
  B. 降低限幅改善：raw target 的高频抖动是主要危害；
  C. 关闭限幅最好：filter 是多余的限制；
  D. 四臂无差异：filter 不是主要因果瓶颈，回到 plant/action/reward。
```

## 2. 固定协议

| 项目 | 值 |
|---|---|
| checkpoint | `artifacts/lowlevel/goal95_supervised_residual_20261111/P_D3/final.pt` |
| checkpoint SHA256 | `2d57745a811268553041c561f2ae1ad590dced472b712efd90cdd9c011c1a616` |
| training config | `configs/lowlevel/goal95_20261111/p_d3_ppo.json` |
| environment | `configs/lowlevel/pc_goal85_unbiased_strict125_candidate_20261110.json` |
| scenario | strict125 / C0 |
| action mode | `frozen-base` |
| eval scale | `2.00` |
| channel gains | `1,1,1` |
| dev seeds | `20–59`，40 episodes/arm |
| success frame | `both` |
| held-out | `100–149` 关闭 |
| device | CPU / float64（eval-only） |
| workers | 4 个独立进程并行 |

只改 strict125 配置中的：

```text
guidance.command_rate_limit_enabled
guidance.pitch_rate_limit_rad_s
```

heading/speed rate limit、reward、termination、success、goal radius 全部冻结。

## 3. 四个预注册 arms

| arm | enabled | pitch rate limit | 解释 |
|---|---:|---:|---|
| `R050` | true | `0.1308996939 rad/s`（7.5 deg/s） | 更平滑，检验高频 target 噪声 |
| `R100` | true | `0.2617993878 rad/s`（15 deg/s） | Goal115 control |
| `R200` | true | `0.5235987756 rad/s`（30 deg/s） | 放宽 filter，检验方向信息 |
| `ROFF` | false | 保留配置值但忽略 | raw target 上界/安全对照 |

每个 arm 生成独立 environment JSON，在 `run_manifest.json` 记录原 config
SHA256、派生 config SHA256 和实际 limit。`R100` 还必须与 Goal115 strict C0
的 40-seed trace 在 `time_s/eta/nu/command/actuator_total_physical/terminal_kind`
上逐 seed 相等；不相等则整个 Goal116 判 `INVALID`。

## 4. 主指标

每个 arm 输出：

```text
success_count / 40
timeout_count
safety_count
goal_plane_miss_count
other_terminal_count
closest_approach_median_m
closest_approach_p05_m
```

原 11 个 failure 子集单独报告：

```text
seed 25,26,27,29,31,33,36,49,50,51,54
new_success_count
terminal_kind_counts
min_distance_median_m
PREENTRY_GEOMETRY_BAD_count（用 Goal111 的冻结 2 m 几何判据复算）
```

## 5. 预注册判读

与 `R100` 对照，只有满足“效率收益”且“安全不退化”才算支持：

```text
efficiency_gain(arm) =
    success_count(arm) >= success_count(R100) + 3

safety_guard(arm) =
    timeout_count(arm) + safety_count(arm)
    <= timeout_count(R100) + safety_count(R100) + 1
```

主判读：

| 观察 | 状态 | 下一步 |
|---|---|---|
| `R200` 或 `ROFF` 同时满足 gain + safety，且原 11 子集至少多 2 个 success | `FILTER_CAUSAL_UP` | 新 Goal 调 reference/limit 设定，不换算法 |
| `R050` 同时满足 gain + safety，且原 11 子集至少多 2 个 success | `REFERENCE_NOISE_DOMINANT` | 新 Goal 做 reference smoothing |
| 所有 arms 相对 `R100` success 差绝对值 <= 2 | `FILTER_NONCAUSAL` | 回 plant/action/reward，不开 SAC |
| 某 arm success 更高但 `safety_guard` 失败 | `HARM` | 淘汰该 arm，保留数据，不救指标 |
| R050 与 R200/ROFF 同时满足 gain + safety | `NONMONOTONIC_FILTER_RESPONSE` | 不选方向，下一 Goal 做连续 limit sweep |
| 任一 arm 缺失 40 seeds、checkpoint SHA 不符或 summary 非有限 | `INVALID` | 先修运行协议，不分析机制 |

不把 3 个 episode 的差异解释成统计显著性；它只是下一轮是否值得投入的
预注册筛选门槛。若结果落在噪声区，必须按 `FILTER_NONCAUSAL` 处理，不能换 seed。

## 6. 执行计划

- [x] Step 0：写 Goal116 预注册并冻结 arms/阈值；
- [x] Step 1：实现派生 config 与 4-worker eval runner；
- [x] Step 2：本地 smoke 单 seed，确认只改 pitch rate limit；
- [x] Step 3：远端跑 4 arms × 40 dev seeds；
- [x] Step 4：校验 manifest、summary、SHA、seeds 和 terminal counts；
- [x] Step 5：对原 11 failure 子集复算 Goal111 几何分类；
- [x] Step 6：写 result/decision/reading note/LEDGER/STATE；
- [x] Step 7：运行 checker 和 `git diff --check`；不自动 commit/push。

## 7. Git

禁止 `git add -A`。只 add 本次相关文件。提交前运行：

```bash
python3 ../scripts/check_research_ledger.py \
  --goal Goal116 \
  --goal-doc doc/goals/goal116_reference_filter_causality_screen_20261111.md \
  --strict --require-result
git diff --cached --check
```

不自动 commit/push；不提交 `artifacts/`、checkpoint、日志、缓存。

## 8. Goal 提示词

```text
你现在执行 Goal116：guidance pitch rate-limit 因果筛选。

先读：
- doc/goals/goal116_reference_filter_causality_screen_20261111.md
- doc/goals/goal115_result_20261111.md
- doc/goals/goal114_result_20261111.md
- doc/goals/goal108_network_and_noisenet_audit_20261111.md
- research/STATE.md
- research/LEDGER.jsonl

协议：
- 同 checkpoint、strict125 C0、frozen-base、dev seeds 20–59、40 episodes/arm；
- 只扫 guidance pitch rate limit：R050=7.5 deg/s、R100=15 deg/s control、
  R200=30 deg/s、ROFF=disabled；
- 不训练、不改 reward/terminal/success/goal radius、不打开 held-out；
- 4 个 eval run 并行；结果按预注册的 success + safety + 原 11 failure 子集判读。

必须先保留 R100 的 Goal107/Goal115 parity 基线。若任一 arm 只靠放宽安全或
增加 timeout 换 success，判 HARM，不得进入下一轮。若四臂在 seed 噪声内，
判 FILTER_NONCAUSAL，回到 plant/action/reward，不开 SAC/REDQ/DroQ。
```
