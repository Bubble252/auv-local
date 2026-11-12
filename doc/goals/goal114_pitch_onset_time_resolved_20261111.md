# Goal114：2 m 锚定的 pitch tracking 时间分辨 mixed-mechanism 审计

**日期**：2026-11-11<br>
**状态**：`COMPLETED_AUDIT_ONLY / MIXED_PITCH_HANDOFF / EVAL_ONLY`<br>
**上游**：Goal113 `PITCH_TRACKING_AMBIGUOUS`；Goal112 `TRACKING_LIMITED`；Goal111 pre-entry geometry；Goal108/89 网络 parity<br>
**性质**：只读 Goal107 dev40 trace；不训练、不 rollout、不改 reward/terminal/success/goal radius、不打开 held-out。

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
  check_command: python3 ../scripts/check_research_ledger.py --goal Goal114 --goal-doc doc/goals/goal114_pitch_onset_time_resolved_20261111.md --strict --require-result
```

测量前不得修改主分析子集、主窗口、阈值规则、seed、scenario、trace 或 Gate 判读规则。

## 1. 背景与问题

Goal113 在冻结的 3.0 m 主窗口没有得到单一机制：

```text
AUTHORITY_LIMITED        = 0/11
COMMAND_RATE_UNREACHABLE = 2/11
SPEED_CURRENT_COUPLED    = 3/11 (mechanism)
PREENTRY_PITCH_LAG       = 0/11
NO_PITCH_LAG             = 8/11
MIXED_OR_AMBIGUOUS       = 2/11
```

但描述层显示 pitch error 越界数从 3.0 m 的 3/11 升到 2.5 m 的 4/11、
2.0 m 的 7/11。因此现在要回答的是时间顺序问题，而不是再做一次静态窗口投票：

```text
谁先离开 success envelope？
  - pitch command rate / rate error
  - actual pitch rate deficit
  - stern near-limit / material saturation
  - speed error / current magnitude
  - pitch error 本身
```

如果多个 onset 接近或交替，就必须判 `MIXED_PITCH_HANDOFF`，不能强行归因给单一
authority/rate/current。Goal114 不训练、不调参、不改动作参数化。

## 2. 数据与隔离

- 输入：`artifacts/lowlevel/goal107_terminal_reward_credit_20261111/`
- 主 run：`P_D3_strict125_C0`
- 重复对照：`P_D3_strict125_U050`（只作描述，不进入主 Gate）
- default 只作描述；
- seeds：`20–59`，每 run 40 episodes；
- 主分析子集：Goal111 判为 `PREENTRY_GEOMETRY_BAD` 的 strict C0 failure（11 个）；
- 成功对照：strict C0 的 success episodes，只用于构造 success envelope，不参与
  failure 分类；
- held-out `100–149` 保持关闭；
- 不训练、不 rollout、不改任何配置；
- 不读取或修改 policy/checkpoint，只读 trace。

trace 字段契约：

```text
eta                     [x, y, z, roll, pitch, yaw]
nu                      [u, v, w, p, q, r]  (body axes)
command                 [heading, pitch, speed]
actuator_base_physical  [rudder, stern_plane, rpm]
actuator_residual_effective
actuator_total_physical
current_ned_mps
speed_error_mps
ground_speed_mps
material_saturated
guidance_terminal_phase
goal_distance_eta_m
time_s
```

pitch 通道：

```text
pitch_cmd    = command[:, 1]
pitch_actual = eta[:, 4]
pitch_rate   = nu[:, 4]
stern_actual = actuator_total_physical[:, 1]
stern_norm   = abs(stern_actual) / radians(20)
```

## 3. 冻结窗口与 onset 规则

### 3.1 时间锚点

- 主锚点：首次进入 `2.0 m`；
- onset 搜索区间：首次进入 `3.0 m` 到 `2.0 m entry + 1.0 s`；
- `3.0/2.5/2.0/1.5 m` 入口只用于分段快照和斜率描述；
- 若 episode 未进入 2.0 m，主记录为 `NO_2M_ENTRY`，不进入机制 Gate；
- 主 Gate 分母仍固定为 11；未进入 2.0 m 的样本计入 `not_evaluable`，并要求
  `evaluable_n >= 9` 才允许判读；否则状态为 `INSUFFICIENT_2M_EVENTS`。

### 3.2 信号

```text
pitch_error_abs      = abs(pitch_cmd - pitch_actual)
pitch_cmd_rate_abs   = abs(d(pitch_cmd)/dt)
pitch_rate_abs       = abs(pitch_rate)
rate_error_abs       = abs(d(pitch_cmd)/dt - pitch_rate)
rate_deficit         = abs(d(pitch_cmd)/dt) - abs(pitch_rate)
stern_norm           = abs(stern_actual) / radians(20)
speed_error_abs      = abs(speed_error_mps)
current_mps          = ||current_ned_mps||
```

`pitch_cmd_rate` 使用 `np.gradient(pitch_cmd, time_s)`；所有信号必须有限。

### 3.3 success envelope 阈值

阈值只由 strict C0 的 success episodes 在 `3.0→2.0 m` 区间决定，不能看 failure
结果后再改。对每个 success episode 和每个信号，取该区间内的 95th percentile；
阈值取这些 per-episode p95 的中位数：

```text
threshold_signal = median_over_success_episodes(p95_within_3m_to_2m(signal))
```

最终阈值再做物理下限/上限约束：

| 信号 | 最终阈值 |
|---|---:|
| `pitch_error_abs` | `max(0.45 rad, envelope)` |
| `pitch_cmd_rate_abs` | `max(0.20 rad/s, envelope)` |
| `rate_error_abs` | `max(0.10 rad/s, envelope)` |
| `rate_deficit_abs` | `max(0.10 rad/s, envelope)` |
| `stern_norm` | `min(0.95, envelope)` |
| `speed_error_abs` | `max(0.25 m/s, envelope)` |
| `current_mps` | `max(0.30 m/s, envelope)` |

若某信号在 success envelope 下没有任何 failure onset，则允许一次固定回退阈值：

```text
pitch_error_abs >= 0.45
pitch_cmd_rate_abs >= 0.20
rate_error_abs >= 0.10
rate_deficit_abs >= 0.10
stern_norm >= 0.95
speed_error_abs >= 0.40
current_mps >= 0.35
```

回退只改变该信号的 onset 搜索阈值，不改变主窗口、主子集或 Gate 分母。

### 3.4 持续越界与 onset

- 一个信号必须连续 `3` 帧（20 Hz 下约 `0.15 s`）超过阈值，才算 onset；
- onset 时间记为连续区间的第一帧；
- 相对时间：`onset_rel_s = time_s[onset] - time_s[2m_entry]`；
- 只记录 `3.0 m entry` 到 `2.0 m entry + 1.0 s` 内的第一次持续越界；
- `pitch_cmd_rate`、`rate_error`、`rate_deficit` 属于 rate-demand 组；
- `stern_norm` 属于 authority 组；
- `speed_error_abs`、`current_mps` 属于 speed/current 组；
- `pitch_error_abs` 是 tracking-error 终点信号，不用于循环定义 rate/authority。

### 3.5 onset 分类

对每个可评估 failure，先取各组最早 onset：

```text
rate_onset      = earliest(pitch_cmd_rate, rate_error, rate_deficit)
authority_onset = stern_norm
coupling_onset  = earliest(speed_error_abs, current_mps)
tracking_onset  = pitch_error_abs
```

分类规则：

| 分类 | 条件 |
|---|---|
| `MIXED_PITCH_HANDOFF` | 至少两个组的 onset 相差 `<=0.20 s`；2 m 前后出现不同组的最早 onset 只作描述，不单独判 mixed |
| `RATE_DEMAND_FIRST` | rate 组最早，且比其余组至少早 `0.20 s` |
| `AUTHORITY_FIRST` | authority 组最早，且比其余组至少早 `0.20 s` |
| `SPEED_CURRENT_FIRST` | coupling 组最早，且比其余组至少早 `0.20 s` |
| `TRACKING_ERROR_FIRST` | pitch error 最早，且比其余组至少早 `0.20 s` |
| `NO_ONSET_ENVELOPE` | 所有组都没有持续越界，且 2 m 时 pitch error 仍 >=0.45 |
| `NO_2M_ENTRY` | 未进入 2.0 m |
| `ONSET_AMBIGUOUS` | 其余情况 |

`tracking_onset` 只作为“谁先坏”的终点信号；如果它总是最早，不能据此说 residual
方向正确，只能说明当前 pitch tracking 已经在 2 m 前形成。

### 3.6 分段快照

在首次进入 `3.0/2.5/2.0/1.5 m` 时，各取入口前 4 帧与入口帧的中位数：

```text
snapshot(shell, signal) = median(signal[entry-4 : entry+1])
```

描述 `3.0→2.5`、`2.5→2.0`、`2.0→1.5` 的每米变化量；这些分段只用于解释 onset，
不改变主 Gate 分类。

## 4. 假设

### H114-A：rate demand 先坏

机制：pitch command 在接近 2 m 时快速变化，actual pitch rate 长期跟不上；pitch
error 是 rate 不足的结果。

成立：`RATE_DEMAND_FIRST >= 6/11`，且中位 rate onset 早于 tracking onset 至少
`0.20 s`。

失败：rate 组不是多数，或 rate onset 与 tracking onset 相差小于 3 帧。

最便宜证伪：本 Goal 的逐帧 onset 排序；不训练。

### H114-B：authority 先坏

机制：stern 先接近限幅，随后 pitch error 才越界；authority 是上游限制。

成立：`AUTHORITY_FIRST >= 6/11`，且 authority onset 早于 tracking onset。

失败：authority 不是多数，或 `material_saturated` 很少。

### H114-C：speed/current 先坏

机制：speed error 或 current magnitude 先离开 success envelope，随后 pitch tracking
恶化；这是外部扰动/速度耦合，不是单一 fin authority。

成立：`SPEED_CURRENT_FIRST >= 6/11`，且 coupling onset 早于 tracking onset。

失败：coupling 不是多数，或 Spearman 相关性在窗口内不稳定。

### H114-D：mixed handoff

机制：rate、authority、coupling 中至少两个在 2 m 附近交接，任何单机制分类都会
丢失信息。

成立：`MIXED_PITCH_HANDOFF >= 6/11`，或前两类各 >=3/11 且中位 onset 差 <=0.20 s。

失败：存在单一类别 >=6/11，且 onset 间隔稳定大于 3 帧。

### H114-E：pitch command 可达、residual 方向正确但样本效率不足

成立条件必须同时满足：

- 没有 majority authority/rate/coupling onset；
- pitch command 与实际 pitch 的符号/方向在 2 m 前后一致；
- residual direction 审计显示有效 residual 与所需 pitch 修正同号；
- 只是闭环 success 不改善。

本 Goal 只读 trace，不训练，因此 H114-E 最多只能判 `NOT_TESTED`，不能直接开 SAC。

## 5. 执行计划

- [x] Step 0：写 Goal114 预注册与 Research Contract；
- [ ] Step 1：实现只读 onset 分析器与单测；
- [ ] Step 2：在本地合成 trace 上跑 smoke；
- [ ] Step 3：在远端 Goal107 冻结 trace 上跑正式只读审计；
- [ ] Step 4：写 `goal114_result_20261111.md`、decision、reading note、LEDGER、STATE；
- [ ] Step 5：运行 checker、`git diff --check`；不自动 commit/push。

## 6. Gate

| check | 阈值 |
|---|---:|
| 主分析子集 | strict C0 的 11 个 Goal111 `PREENTRY_GEOMETRY_BAD` failure |
| 可评估数 | `evaluable_n >= 9` |
| rate majority | `RATE_DEMAND_FIRST >= 6` |
| authority majority | `AUTHORITY_FIRST >= 6` |
| coupling majority | `SPEED_CURRENT_FIRST >= 6` |
| mixed majority | `MIXED_PITCH_HANDOFF >= 6` |
| 稳定性 | 0 NaN / 0 非有限 trace / 0 held-out seed |
| 负控 | success envelope 阈值只由 success 构造，failure 不参与 |

失败裁决：

| 观察 | 判读 | 下一步 |
|---|---|---|
| 单一类别 >=6/11 且 onset 稳定 | 对应机制优先 | 转向该机制的 reference/controller/plant 审计 |
| 两类各 >=3/11 或 onset 差距 <=0.20 s | `MIXED_PITCH_HANDOFF` | 设计 joint handoff 审计，不直接换算法 |
| 所有类别 <6/11 且 `NO_ONSET_ENVELOPE` 多 | `PITCH_ONSET_AMBIGUOUS` | 检查阈值/信号定义，写 blocker |
| authority/rate 均可达但 residual 方向错误 | `RESIDUAL_DIRECTION_LIMITED` | 回到 residual direction/reward credit，不开 SAC |
| 只有明确证明探索/样本效率不足 | `SAC_ELIGIBLE` | 才允许新开预注册 SAC 同协议对照 |

## 7. Git

禁止 `git add -A`。只 add 本次相关文件，提交前运行：

```bash
python3 ../scripts/check_research_ledger.py \
  --goal Goal114 \
  --goal-doc doc/goals/goal114_pitch_onset_time_resolved_20261111.md \
  --strict --require-result
git diff --cached --check
```

不自动 commit/push；不提交 `artifacts/`、checkpoint、日志、缓存。

## 8. Goal 提示词

```text
你现在执行 Goal114：2 m 锚定的 pitch tracking 时间分辨 mixed-mechanism 审计。

先读：
- doc/goals/goal114_pitch_onset_time_resolved_20261111.md
- doc/goals/goal113_result_20261111.md
- research/decisions/2026-11-11-goal113-pitch-tracking-ambiguous-continue.md

硬约束：
1. 只读 Goal107 dev40 trace；不训练、不 rollout、不改 reward/terminal/success/goal radius；
2. 不打开 held-out 100–149；不换 PPO/SAC/REDQ/DroQ/DADE；
3. 主窗口固定首次进入 2.0 m，3.0/2.5 m 只作 onset 描述；
4. success envelope 阈值只由 success episodes 构造，failure 不参与；
5. 逐帧 onset 必须使用连续 3 帧持续越界；
6. 多机制接近或交替时判 MIXED_PITCH_HANDOFF，不强行归因。

执行顺序：
- 实现 scripts/analyze_pc_goal114_pitch_onset.py 和单测；
- 先跑合成 trace smoke；
- 再在远端 Goal107 trace 上跑正式审计；
- 写 result、decision、reading note、LEDGER、STATE；
- 运行 checker 和 git diff --check；不自动 commit/push。
```
