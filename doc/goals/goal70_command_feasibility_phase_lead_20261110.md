# Goal70：command feasibility / phase-lead 只读归因审计

**日期**：2026-11-10<br>
**状态**：`PROTOCOL_DRAFTED_BEFORE_MEASUREMENT`<br>
**上游**：Goal69 `PRECROSSING_MULTICHANNEL_TRACKING_LAG`<br>
**关联决策**：`research/decisions/2026-11-10-goal69-multichannel-tracking-pivot.md`

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
  check_command: python ../scripts/check_research_ledger.py --goal Goal70 --goal-doc doc/goals/goal70_command_feasibility_phase_lead_20261110.md --strict
```

## 1. 背景与证据

Goal69 在 plane crossing 前 1 s 发现 frozen controller 的 heading/pitch tracking
失败：lost 的 heading error 为 `0.930/0.364 rad`，pitch error 为
`0.618/0.596 rad`，而 speed 通道不支持。command-goal cosine 仍有
`0.980/0.983`，actual-command cosine 只有 `0.294/0.659`，同时闭环地速约
`1.9–2.0 m/s`，不是低速卡死。

Goal69 同时排除了 NoisyNet/residual 主导：

- eval 使用 deterministic mean weights，不采样 NoisyNet 噪声；
- residual/base 在 default lost 为 `0.109`、retained 为 `0.200`，strict125 为
  `0.181/0.176`，lost 并不更高；
- 完全关闭 residual 后仍丢 default 4 个、strict125 1 个 true episode；
- scale/rate clip 为 0，Goal59 deterministic saturation 均为 1.0。

因此本 Goal 不再检查 residual/NoisyNet，也不训练、不启动 SAC/REDQ/DroQ。要回答：

> crossing 前的 heading/pitch tracking lag，是 command rate/feasibility 超过
> shaping envelope，还是 controller 的 phase lead/bandwidth 不足？
> default 的 terminal guidance 是否额外增加了最后 1 s 的需求？

## 2. 数据与隔离

| 项目 | 固定值 |
|---|---|
| trace | Goal67 的 10 个 rich candidate |
| baseline/reference | Goal60 / Goal65 的相同 dev 结果 |
| candidate | 4 个 Goal59 checkpoint × default/strict125 policy + 2 个 zero-residual |
| seeds | dev `20–59`，40 seeds/run |
| eval | CPU/float64、`endpoint_stop`、dwell=50、`r_tight = r0 - 0.25 m` |
| candidate suffix | `_dwell50_m025` |
| 主窗口 | crossing 前 1 s；2 s 只作一致性检查 |
| crossing | 第一个 `signed_offset >= 0` 的行；窗口截止该行 |
| new rollout / 训练 | 无 |
| held-out | `100–149` 关闭 |

必须复用 Goal65 的逐 seed 等价性检查、Goal68 的 crossing 定义和 Goal69 的
lost/retained 分组。summary、trace metadata、checkpoint SHA、dtype/device、
dwell、radius 任一项不一致都写 `INVALID_DIAGNOSTIC`，不改阈值救结果。

## 3. 指标定义

所有时间导数都按 trace 的真实 `time_s` 计算，不用 step index 近似。

```text
c_h = d(unwrap(command_heading))/dt
c_p = d(command_pitch)/dt
c_v = d(command_speed)/dt
a_yaw   = nu[:, 5]
a_pitch = nu[:, 4]
a_surge = d(nu[:, 0])/dt
```

- `cmd_*_p50/p90`：command rate 的绝对中位数/P90；
- `cmd_*_exceed_fraction`：`abs(c_*) > guidance.*_rate_limit_rad_s` 的比例；
- `actual_*_p50/p90`：actual rate 的绝对分位数；
- `dynamic_gain`：detrended actual rate 与 command rate 的标准差之比；
- `phase_lag_s`：在 `[-0.10, +0.30] s` 内使 detrended cross-correlation 最大
  的 lag；正值表示 actual 落后 command；
- `phase_corr`：上述最佳 lag 的 Pearson 相关系数；
- `fin_rate_at_limit_fraction`：`actual_actuator` 的 rudder/stern 差分达到
  plant fin rate limit 的比例；
- `command_reachability_infeasible_fraction`：trace 中
  `command_reachability_feasible == false` 的比例；
- `terminal_phase_fraction`：`guidance_terminal_phase == true` 的比例；
- `late_minus_early_*_p90`：窗口后半段与前半段 command rate P90 的差。

配置参考值从当前 environment config 读取，不在代码里硬编码正式阈值：

```text
heading command rate limit = 0.5236 rad/s
pitch command rate limit   = 0.2618 rad/s
speed command rate limit   = 0.25 m/s^2
plant fin rate limit       = 150 deg/s = 2.618 rad/s
```

注意：guidance command rate limiter 当前实际关闭，因此“超过 limit”表示需求
超出原本设计的 shaping envelope，不等于执行器已经饱和。执行器饱和必须由
`material_saturated`、`fin_rate_at_limit_fraction` 和 actual actuator 证据另行判断。

## 4. 可证伪假设与冻结 Gate

所有比较都在同一 scenario 内按 lost/retained 分组；`gap = lost - retained`，
正值表示 lost 更差。主窗口为 1 s；2 s 只检查方向是否一致，不扩大正式阈值。

### H70-A：command rate / translational feasibility 是主因

支持条件（任一通道）：

```text
heading:
  lost cmd_heading_rate_p90 >= 1.25 * 0.5236 rad/s
  and lost cmd_heading_rate_exceed_fraction >= 0.20
  and gap(cmd_heading_rate_p90) >= 0.10 rad/s

pitch:
  lost cmd_pitch_rate_p90 >= 1.25 * 0.2618 rad/s
  and lost cmd_pitch_rate_exceed_fraction >= 0.20
  and gap(cmd_pitch_rate_p90) >= 0.05 rad/s

or translational:
  lost command_reachability_infeasible_fraction >= 0.10
  and gap(command_reachability_infeasible_fraction) >= 0.05
```

证伪条件：需求未超过上述 envelope，或 lost/retained 无上述差距；此时不能把失败
归因于 command feasibility。

### H70-B：controller phase lead / bandwidth 不足

对 heading 或 pitch 分别计算。通道支持条件：

```text
phase_corr >= 0.50
and (
  phase_lag_s >= 0.08 s
  or dynamic_gain <= 0.60
)
and (
  phase_lag_gap >= 0.03 s
  or dynamic_gain_retained_minus_lost >= 0.15
)
and same-channel tracking error gap:
  heading abs error gap >= 0.10 rad
  or pitch abs error gap >= 0.05 rad
```

证伪条件：actual rate 已跟上 command rate（lag < 0.08 s 且 gain > 0.60），或
lost 与 retained 无差异。若 A、B 同时成立，正式状态必须是组合，不能只报一个。

### H70-C：terminal guidance 在最后 1 s 增加额外需求

仅对 default 作正式判定，strict125 只作反例描述。支持条件：

```text
lost terminal_phase_fraction >= 0.80
and (
  lost late_minus_early_heading_rate_p90 >= 0.20 rad/s
  or lost late_minus_early_pitch_rate_p90 >= 0.10 rad/s
)
and lost 的同通道 late-minus-early 比 retained 高至少 0.10 rad/s
```

证伪条件：terminal phase 不占优、后半段需求没有上升，或 retained 同样上升。

### 负控与饱和守卫

- speed 是负控：若 speed 通道显示同样强的 rate/phase 差异，必须显式记录为
  `NEGATIVE_CONTROL_FAILED`，不能把 heading/pitch 解释当成唯一机制；
- `material_saturation_fraction` 或 `fin_rate_at_limit_fraction` 在 lost 只允许
  作为描述量；只有 actual actuator 同时到限速且方向一致，才可写“执行器 rate
  saturation”；
- `phase_corr < 0.50` 的 lag 不参与 B 的正式判定。

### 优先级与状态名

| A | B | C | 状态 |
|---|---|---|---|
| true | true | any | `COMMAND_RATE_AND_BANDWIDTH_BOTH` |
| true | false | any | `COMMAND_RATE_INFEASIBLE` |
| false | true | any | `CONTROLLER_BANDWIDTH_LIMITED` |
| false | false | true | `TERMINAL_GUIDANCE_EXTRA_DEMAND` |
| false | false | false | `COMMAND_FEASIBILITY_BANDWIDTH_UNRESOLVED` |

default 与 strict125 状态不同则总状态为 `MIXED_COMMAND_FEASIBILITY`，优先保留
strict125，并同时报告 default 反例。数据完整性问题写 `INVALID_DIAGNOSTIC`。

## 5. 执行计划

- [ ] Step 0：复核 Goal67 trace、Goal65 等价性和当前配置；
- [ ] Step 1：实现 `scripts/analyze_pc_goal70_command_feasibility.py` 与合成测试；
- [ ] Step 2：本地 smoke，确认 crossing、lag 符号、负控和 INVALID 路径；
- [ ] Step 3：服务器用 `--jobs 10` 跑 10 个 run 的只读分析；
- [ ] Step 4：按冻结 Gate 判 H70-A/B/C，写 result/blocker/ledger/STATE；
- [ ] Step 5：strict checker、pytest、Ruff、diff check；只 add 相关文件。

## 6. Gate（测量前冻结）

| 观察 | 判读 | 下一步 |
|---|---|---|
| A 支持、B 不支持 | command rate/shaping 需求超限 | 另开预注册单因素 command-rate-limit 闭环 arm；不改算法 |
| B 支持、A 不支持 | controller phase/bandwidth | 另开预注册 controller 带宽/前馈诊断；不改 checkpoint |
| A、B 都支持 | 需求与响应共同限制 | 先做 2×2 单因素 command shaping × controller bandwidth 的预注册对照 |
| 只有 C 支持 | terminal guidance 额外需求 | 预注册 terminal path/command 分层诊断 |
| 都不支持 | 机制未定 | 转 command/actuator 动力学辨识；不扫 reward/gamma/seed |
| speed 负控失败 | 通道不特异 | 暂停 heading/pitch 归因，先查测量或共同工况 |
| 数据/等价性失败 | `INVALID_DIAGNOSTIC` | 只修测量实现，不改阈值 |

## 7. 算力协议

- 复用已有 trace，不新增 rollout；10 个 run 用 `--jobs 10` 并行；
- 每进程 `OMP_NUM_THREADS=1`、`MKL_NUM_THREADS=1`，避免 CPU 线程过度订阅；
- 记录墙钟、CPU%、RSS、GPU 使用；本 Goal 预期不占 GPU；
- 若之后转为训练型对照，按 `research/decisions/2026-11-10-compute-utilization-protocol.md`
  先做吞吐基准，再按 seed 分 GPU；扩大 `num_envs`/batch 必须另开预注册 arm，
  固定总 transitions、UTD、replay 比例、checkpoint 选择和评估协议。

## 8. 禁止项

- 不重开 Goal27，不改 reward、gamma、seed、checkpoint、controller 或 eval split；
- 不把 command rate 超限写成 actuator saturation；
- 不启动 SAC；没有 optimization/exploration 失败证据前不换算法；
- 没有 sample-efficiency 失败证据前不启动 REDQ/DroQ；
- 不打开 held-out `100–149`；不 `git add -A`；不自动 commit/push。

## 9. Git

本 Goal 只应涉及 Goal70 文档、analyzer、测试、result、blocker/ledger/STATE。
提交前运行：

```bash
git diff --cached --check
.venv/bin/python ../scripts/check_research_ledger.py \
  --goal Goal70 \
  --goal-doc doc/goals/goal70_command_feasibility_phase_lead_20261110.md \
  --strict
```

用户明确要求后才 commit/push；禁止 `git add -A`。
