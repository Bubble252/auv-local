# Goal71：启用现有 command rate limiter 的单因素闭环对照

**日期**：2026-11-10<br>
**状态**：`MEASURED_TRACKING_IMPROVES_OUTCOME_UNCHANGED`<br>
**上游**：Goal70 `COMMAND_RATE_INFEASIBLE`<br>
**基线**：Goal67 limiter-off rich trace / Goal60 true-reference

**结果**：`doc/goals/goal71_result_20261110.md`

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
  check_command: python ../scripts/check_research_ledger.py --goal Goal71 --goal-doc doc/goals/goal71_command_rate_limit_control_20261110.md --strict
```

## 1. 背景与研究问题

Goal70 在同一批 Goal67 rich trace 上发现：

- default lost heading command rate P90 为 `1.457 rad/s`，retained 为
  `0.677 rad/s`；exceed fraction 为 `0.961 vs 0.196`；
- strict125 lost 为 `3.489 vs 0.677 rad/s`，exceed fraction 为
  `0.667 vs 0.176`；
- `command_rate_limit_enabled=false`，配置中的 rate limit 实际没有生效；
- fin rate at limit 为 0，material saturation 在 lost 反而更低，speed 负控通过；
- NoisyNet/residual 与 phase/bandwidth 都不是 lost/retained 的判别因素。

本 Goal 只回答一个问题：

> 在不改 checkpoint、reward、seed 和 controller 的前提下，启用现有
> guidance command rate limiter，能否降低 crossing 前 command demand、改善
> tracking，并转化为 true retention？

这是观测性 Goal70 的最小因果 falsification，不是算法切换。

## 2. 单因素定义

candidate config 相对同 scenario 的 Goal59 environment config 只允许以下差异：

```text
guidance.command_rate_limit_enabled: false -> true
experiment.name / experiment.output: 仅用于隔离输出
```

以下字段必须逐项相同：

```text
guidance.heading_rate_limit_rad_s = 0.5236
guidance.pitch_rate_limit_rad_s   = 0.2618
guidance.speed_rate_limit_mps2    = 0.25
guidance.saturation_governor_enabled = false
plant.goal_radius_m / dt_s / max_episode_steps
controller / reward / observation / current / actuator
```

允许的 metadata 差异必须在 analyzer 中显式列出；其余 JSON 递归比较必须完全一致。
limiter 位于 `_limit_guidance_command`，只改变 ALOS base guidance command；policy
residual 仍按原 wrapper 和 residual rate limit 执行。

## 3. 可证伪假设

### H71-A：rate limiter 把 command feasibility 转化为 true retention

在 baseline-true 且有 Goal67 trace 的 original-lost 子集上：

```text
recovered_original_lost >= ceil(0.40 * original_lost)
newly_lost_vs_limiter_off <= 1
true_success_delta = recovered - newly_lost >= 3 per scenario
timeout_delta <= +0.02
safety_delta <= +0.02
pre_goal_rmse 恶化不超过 3%
```

default 的 original lost 为 9，strict125 为 7；因此恢复门槛为 default `>=4`、
strict125 `>=3`。两个 scenario 都满足才算总 H71-A 支持。

证伪：恢复数不足、newly-lost 超过 1、true delta 不足 3，或 timeout/safety/RMSE
恶化超过上述上限。

### H71-B：limiter 生效并改善 tracking，但闭环结果没有转化

支持条件：

```text
limiter_effective == true
and (
  heading abs tracking error 在 original-lost 子集下降 >= 0.10 rad
  or pitch abs tracking error 下降 >= 0.05 rad
)
and H71-A == false
```

`limiter_effective` 的冻结定义：

```text
candidate config 中 command_rate_limit_enabled == true
and original-lost subset 的 heading command rate P90 <= 0.75 rad/s
and heading exceed fraction <= 0.20
```

证伪：limiter 开启后 command rate 仍超出上述范围，或 tracking error 没有下降。

### H71-C：command feasibility 不是 causal bottleneck

支持条件：

```text
limiter_effective == true
and tracking error 未达到 B 的下降门槛
and H71-A == false
```

证伪：limiter 生效后 tracking error 和 true retention 至少一个明显改善。

## 4. 数据、隔离与对照

| 项目 | 固定值 |
|---|---|
| checkpoint | Goal59 frozen `s240/s752/s880/s1008` |
| baseline | Goal67 limiter-off `_dwell50_m025` |
| true-reference | Goal60 `artifacts/traces` 同 checkpoint/scenario |
| candidate | 新建 Goal71 limiter-on `_dwell50_m025_cmdlimit` |
| scenarios | `default`, `strict125` |
| action modes | 4 policy + 1 zero-residual per scenario |
| seeds | dev `20–59`，40 seeds/run |
| eval | CPU/float64、`endpoint_stop`、dwell=50、`r_tight = r0 - 0.25 m` |
| trace | rich trace，字段与 Goal67 相同 |
| new rollout | 仅 limiter-on 评估；无训练 |
| held-out | `100–149` 关闭 |

paired 统计以同一 scenario/checkpoint/action-mode/seed 为单位；不允许跨 seed
配对。default 与 strict125 分开裁决，任一 scenario 失败不能由另一个 scenario
掩盖。

## 5. 指标

### 5.1 结果指标

- `true = success and terminal_eta_capture`；
- `original_lost = goal60_true and not limiter_off_true`；
- `recovered_original_lost = original_lost and limiter_on_true`；
- `newly_lost_vs_limiter_off = limiter_off_true and not limiter_on_true`；
- `true_success_delta = recovered_all - newly_lost_all`；
- success、eta capture、plane miss、timeout、safety、pre-goal 3D RMSE；
- false-success count 不得以牺牲 true capture 换取。

### 5.2 tracking 指标

复用 crossing 前 1 s 主窗口，2 s 只作一致性检查：

- heading/pitch absolute tracking error；
- command heading/pitch rate P90 和 exceed fraction；
- actual yaw/pitch rate、dynamic gain、phase lag/correlation；
- material saturation、fin rate at limit；
- speed error 作为负控。

### 5.3 analyzer 口径（测量前固定）

- H71-A 的 `original_lost`、`recovered_original_lost` 严格限定在
  `Goal60 true and not Goal67 true` 的子集；恢复率门槛为
  `ceil(0.40 * original_lost)`。
- `true_success_delta = recovered_all - newly_lost_all`，其中
  `recovered_all = not Goal67 true and Goal71 true`，
  `newly_lost_all = Goal67 true and not Goal71 true`。这与
  `newly_lost_vs_limiter_off` 使用同一全量配对；恢复率门槛仍只在
  original-lost 子集上判断。
- timeout/safety delta 为 Goal71 减 Goal67 的全量配对率；pre-goal RMSE 为同一
  seed 配对后的均值相对变化，门槛 `<= +3%`。
- `limiter_effective` 使用 original-lost 子集的逐 episode heading command-rate
  P90 中位数 `<= 0.75 rad/s`，且 exceed fraction 中位数 `<= 0.20`；不把
  residual 之后的实际 yaw rate 混入该定义。
- exceed fraction 对限速边界使用 `1e-6` 相对数值容差；命令恰好等于
  `0.5236/0.2618/0.25` 时不算超限。这个容差只修正浮点边界，不改变任何数值阈值。
- H71-B 的 tracking 改善是同一 original-lost 子集上
  `Goal67 - Goal71` 的 abs heading/pitch error 中位数差；heading `>= 0.10 rad`
  或 pitch `>= 0.05 rad`。
- `COMMAND_RATE_LIMITER_HURTS` 的“明显变差”固定为
  `true_success_delta <= -3`；unsafe 优先级高于 helps/hurts。
- 配置递归比较只允许 `experiment.name`、`experiment.output`、
  `guidance.command_rate_limit_enabled` 三条变化；任何其他差异直接
  `INVALID_IMPLEMENTATION`。

## 6. 执行计划

- [x] Step 0：复核 Goal67 baseline、Goal60 true-reference、checkpoint SHA 和 config diff；
- [x] Step 1：生成 limiter-on configs，写 config parity tests；
- [x] Step 2：本地 smoke 验证 limiter 确实改变 heading command，且只有单因素差异；
- [x] Step 3：服务器跑 10 个 limiter-on runs，`--jobs 10`，rich trace；
- [x] Step 4：paired analyzer 判 H71-A/B/C，写 result/blocker/ledger/STATE；
- [x] Step 5：strict checker、pytest、Ruff、diff check；只 add 相关文件。

## 7. Gate（测量前冻结）

| 观察 | 判读 | 下一步 |
|---|---|---|
| limiter effective + H71-A | `COMMAND_RATE_LIMITER_HELPS` | 保留 limiter，检查是否还有 residual 收益 |
| limiter effective + tracking 改善但 H71-A false | `TRACKING_IMPROVES_OUTCOME_UNCHANGED` | 转 controller bandwidth / terminal capture，不换算法 |
| limiter effective + tracking/outcome 都不动 | `COMMAND_RATE_LIMITER_NO_EFFECT` | command feasibility 不是 causal bottleneck，转动力学/带宽 |
| outcome 明显变差 | `COMMAND_RATE_LIMITER_HURTS` | 停止 limiter 轴，保留负结果 |
| limiter 未生效 | `INVALID_IMPLEMENTATION` | 只修配置/实现并重跑，不改阈值 |
| timeout/safety 恶化 | `COMMAND_RATE_LIMITER_UNSAFE` | 不追认收益，先查 terminal/安全 |
| default/strict125 不同 | `MIXED_COMMAND_RATE_LIMITER` | 报告 strict125 主判定并保留 default 反例 |

## 8. 算力与禁止

- 服务器：`8 × RTX 4090`，384 CPU，约 503 GB RAM；当前至少 7 张 GPU 空闲；
- 本 Goal 是 CPU/float64 闭环评估，不占 GPU；10 个 run 用 `--jobs 10`，
  `OMP_NUM_THREADS=1`、`MKL_NUM_THREADS=1`；
- 若后续转训练型 arm，再按 seed 分 GPU；扩大 `num_envs`/batch 必须另开预注册 arm，
  固定总 transitions、UTD、replay 比例、checkpoint 和评估协议；
- 不训练、不换 SAC/REDQ/DroQ、不扫 reward/gamma/seed、不打开 held-out；
- 不 `git add -A`；不自动 commit/push。

## 9. Git

```bash
git diff --cached --check
.venv/bin/python ../scripts/check_research_ledger.py \
  --goal Goal71 \
  --goal-doc doc/goals/goal71_command_rate_limit_control_20261110.md \
  --strict
```

只 add Goal71 相关文件；提交前确认 config diff 只有 limiter 和 metadata。
