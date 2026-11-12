# Goal113：pitch tracking、authority 与 rate 的只读归因审计

**日期**：2026-11-11<br>
**状态**：`COMPLETED_AUDIT_ONLY / PITCH_TRACKING_AMBIGUOUS / EVAL_ONLY`<br>
**上游**：Goal112 `TRACKING_LIMITED`；Goal111 pre-entry geometry；Goal109 residual direction；Goal108/89 网络 parity<br>
**性质**：只读 Goal107 dev40 trace；不训练、不改 reward/terminal/success/goal radius、不打开 held-out。

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
  check_command: python ../scripts/check_research_ledger.py --goal Goal113 --goal-doc doc/goals/goal113_pitch_tracking_authority_audit_20261111.md --strict --require-result
```

测量前不得修改主分析子集、窗口、阈值、seed、scenario、trace 或 Gate 规则。

## 1. 背景与问题

Goal112 对 strict125 C0 的 11 个 `PREENTRY_GEOMETRY_BAD` failure 做了只读分类：

```text
reference involved = 3/11 = 27.3%
tracking involved  = 8/11 = 72.7%
tracking pitch bad = 7/11
tracking heading bad = 3/11
reference pitch bad = 0/11
material saturation positive = 0/11
```

这只能说明“实际姿态没有跟上 command，且 pitch 是最强通道”。现在要回答更具体的
机制问题：

```text
A. pitch command 本身是否变化过快/不可达？
B. actual pitch rate 是否长期不足，还是只是相位滞后？
C. stern/fin authority、低速、speed error、heave、current 是否限制 tracking？
D. 3.0 m 已坏的 6 个样本，是否在更早阶段就已经有稳定 pitch lag？
E. 这是姿态环问题，还是深度/速度/洋流耦合问题？
```

本 Goal 不训练、不调参、不改动作参数化。它只决定下一步应该修 controller/
reference/plant mismatch，还是才有资格考虑 SAC 同协议对照。

## 2. 数据与隔离

- 输入：`artifacts/lowlevel/goal107_terminal_reward_credit_20261111/`
- 主 run：`P_D3_strict125_C0`
- 重复对照：`P_D3_strict125_U050`
- default 只作描述；
- seeds：`20–59`，每 run 40 episodes；
- 主分析子集：Goal111 判为 `PREENTRY_GEOMETRY_BAD` 的 strict C0 failure（11 个）；
- 成功对照：strict C0 的 20 个 success，只作描述分布，不参与 failure 分类；
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

其中 pitch 通道：

```text
pitch_cmd    = command[:, 1]
pitch_actual = eta[:, 4]
pitch_rate   = nu[:, 4]
stern_actual = actuator_total_physical[:, 1]
```

## 3. 冻结窗口与指标

### 3.1 窗口

以首次进入 3.0 m、2.5 m、2.0 m、1.5 m 的帧为锚点，每个锚点取：

- 锚点前 `0.5 s` 到锚点后 `0.1 s`；
- 若 episode 在该锚点前结束，则该锚点记为 `not_entered`；
- 主窗口固定为 `3.0 m entry`，因为 Goal112 显示 6/11 在 3.0 m 已坏；
- `2.5/2.0/1.5 m` 只作 onset/持续性描述。

### 3.2 基础量

```text
pitch_err          = pitch_cmd - pitch_actual
pitch_cmd_rate     = d(pitch_cmd)/dt        (np.gradient, time_s)
pitch_rate         = nu[:,4]
rate_error         = pitch_cmd_rate - pitch_rate
rate_deficit       = abs(pitch_cmd_rate) - abs(pitch_rate)
rate_sign_align    = sign(pitch_cmd_rate) * sign(pitch_rate)
rate_error_abs_med = median(abs(rate_error))
stern_norm         = abs(stern_actual) / radians(20)
surge              = nu[:,0]
heave              = nu[:,2]
current_mag        = ||current_ned_mps||
speed_err          = speed_error_mps
```

窗口统计使用 median；rate sign alignment 只在 `abs(pitch_cmd_rate) >= 0.05 rad/s`
的帧上统计，避免 command 静止时符号无意义。

### 3.3 机制判定

对每个主分析 failure，在 3.0 m 主窗口计算：

```text
pitch_tracking_bad = abs(median(pitch_err)) >= 0.45 rad
rate_deficit_bad   = median(rate_deficit) > 0
rate_align_bad     = median(rate_sign_align) < 0.5
rate_unreachable_bad = median(abs(rate_error)) >= 0.05 and rate_align_bad
authority_bad      = any(material_saturated) or median(stern_norm) >= 0.95
coupling_strong    = abs(spearman(pitch_err, speed_err)) >= 0.5
                  or abs(spearman(pitch_err, current_mag)) >= 0.5
                  or abs(spearman(pitch_err, heave)) >= 0.5
preexisting_lag    = abs(median(pitch_err)) >= 0.45 at 3.0 m entry
```

互斥分类按以下优先级：

| 分类 | 条件 |
|---|---|
| `AUTHORITY_LIMITED` | `authority_bad=true` 且 `pitch_tracking_bad=true` |
| `COMMAND_RATE_UNREACHABLE` | `rate_unreachable_bad=true` 且 `pitch_tracking_bad=true` |
| `SPEED_CURRENT_COUPLED` | `coupling_strong=true` 且 `pitch_tracking_bad=true`，且不满足前两类 |
| `PREENTRY_PITCH_LAG` | `preexisting_lag=true` 且不满足前三类 |
| `NO_PITCH_LAG` | `pitch_tracking_bad=false` |
| `MIXED_OR_AMBIGUOUS` | 其余 |

若多个机制同时成立，记录 `mechanisms` 列表，但互斥分类只取优先级最高者；
另报 `multi_mechanism_n`，不得把 mixed 样本强行写成单一原因。

### 3.4 描述指标

- 3.0/2.5/2.0/1.5 m 各窗口的 `pitch_err`、`pitch_cmd_rate`、`pitch_rate`、
  `rate_deficit`、`stern_norm`、`surge`、`heave`、`speed_err`、`current_mag`；
- `pitch_err` 与各自变量的 Spearman rho；
- command pitch 与 actual pitch 的互相关峰值滞后（只作描述）；
- success 对照的同一组指标分布（median 和 5–95% 区间）；
- strict U050 的 9 个 pre-entry failure 只作重复性描述。

## 4. 假设

### H113-A：command rate 不可达

pitch command 在接近 goal 时变化过快，actual pitch rate 长期落后；tracking error
在 command 快速变化后累积。

成立：`COMMAND_RATE_UNREACHABLE` 至少 6/11；或 `rate_unreachable_bad` 至少 6/11。

### H113-B：fin / actuator authority 限制

低速、speed error 或海流下 stern/fin 权限不足；表现为 `material_saturated`、
`stern_norm >= 0.95` 或 pitch rate 无法跟随。

成立：`AUTHORITY_LIMITED` 至少 6/11。

### H113-C：深度/速度/海流耦合

pitch tracking error 与 heave、speed error 或 current magnitude 的相关性高于与
command rate 的相关性；失败不是纯姿态环。

成立：`SPEED_CURRENT_COUPLED` 至少 6/11，或至少 6/11 样本出现
`abs(rho) >= 0.5` 的速度/海流耦合。

### H113-D：pre-entry pitch lag 已经形成

3.0 m 已坏的样本在 3.0 m 窗口就有稳定 pitch lag；2 m 只是记录点，不是起点。

成立：`PREENTRY_PITCH_LAG` 至少 6/11，或 3.0 m 窗口 `pitch_tracking_bad` 至少 7/11。

### H113-E：pitch command 可达，问题在 residual/探索

pitch command 的 rate/authority 都可达，tracking 方向正确但长期不足，且没有
速度/洋流耦合；这才支持“探索或样本效率不足”。

成立：`NO_PITCH_LAG` 至少 6/11，且 `AUTHORITY_LIMITED`、`COMMAND_RATE_UNREACHABLE`、
`SPEED_CURRENT_COUPLED` 都 `<3/11`。

## 5. Gate

### 5.1 `INVALID`

以下任一失败判 `INVALID`：

- strict C0 success 不是 `20/40`，或 default C0 不是 `33/40`；
- Goal111 的 strict C0 分类不能复现为 `7/11/2`；
- 主分析子集不是 11；
- trace 缺字段、形状错误、非有限值，或 `eta`/`nu`/`command` 时间轴不一致；
- seed 不是 `20–59`，或 checkpoint/scale/gains 不匹配 Goal107；
- 读取 held-out；
- 任何训练、rollout、reward/terminal/success/goal radius 修改。

### 5.2 主判读

若多个机制同时达到 6/11，优先判 `MIXED_PITCH_TRACKING`，不得按单行优先级
挑一个写成单一原因。

| 观察 | 状态 | 允许的下一步 |
|---|---|---|
| `AUTHORITY_LIMITED` >=6/11 | `PITCH_AUTHORITY_LIMITED` | 新预注册 controller/actuator/plant mismatch 审计；不改 RL 算法 |
| `COMMAND_RATE_UNREACHABLE` >=6/11 | `PITCH_RATE_UNREACHABLE` | 新预注册 command filter/rate limit/reference 审计；不改 RL 算法 |
| `SPEED_CURRENT_COUPLED` >=6/11 | `PITCH_SPEED_CURRENT_COUPLED` | 新预注册 current/speed-aware reference 与 plant mismatch 审计 |
| `PREENTRY_PITCH_LAG` >=6/11 | `PREENTRY_PITCH_LAG` | 新预注册更早 homing/vertical handoff 审计 |
| 多个机制都 >=6/11 或 mixed >=6/11 | `MIXED_PITCH_TRACKING` | 写 joint pre-registration；禁止单通道改动 |
| `NO_PITCH_LAG` >=6/11 且其余 <3/11 | `PITCH_COMMAND_REACHABLE` | 只有此时才允许新预注册 SAC 同协议对照 |
| 其他 | `PITCH_TRACKING_AMBIGUOUS` | 补 reading note 与更便宜 falsification，不训练 |

## 5A. 正式结果

正式判读：

```text
PITCH_TRACKING_AMBIGUOUS
reason = no_mechanism_at_least_6_of_11_in_frozen_3m_window
```

strict C0 的 11 个 `PREENTRY_GEOMETRY_BAD` failure，在 3.0 m 主窗口：

| 机制 | 数量 |
|---|---:|
| `AUTHORITY_LIMITED` | 0 |
| `COMMAND_RATE_UNREACHABLE` | 2 |
| `SPEED_CURRENT_COUPLED` | 3 |
| `PREENTRY_PITCH_LAG` | 0 |
| `NO_PITCH_LAG` | 8 |

互斥分类：

| 分类 | 数量 |
|---|---:|
| `AUTHORITY_LIMITED` | 0 |
| `COMMAND_RATE_UNREACHABLE` | 0 |
| `SPEED_CURRENT_COUPLED` | 1 |
| `PREENTRY_PITCH_LAG` | 0 |
| `NO_PITCH_LAG` | 8 |
| `MIXED_OR_AMBIGUOUS` | 2 |

描述层：

```text
pitch error >= 0.45:
  3.0 m = 3/11
  2.5 m = 4/11
  2.0 m = 7/11
```

因此 3 m 作为 onset 窗口过早；pitch lag 主要从 3 m 到 2 m 形成，且 2 m 时机制
已经 mixed（2 m：authority 3、rate 3、coupling 7、mixed class 5）。这不能事后改写
Goal113 的主判读，但足够支持新开 Goal114，把主窗口固定到 2 m 并做逐帧 onset 排序。

`decision = continue`：不训练、不换算法、不开 SAC；下一步为 Goal114
2 m-anchored time-resolved mixed-mechanism 审计。详细结果见
`doc/goals/goal113_result_20261111.md`。

## 6. 执行计划

- [x] Step 0：冻结主分析子集、窗口、指标、阈值与 Gate；
- [x] Step 1：实现 `scripts/analyze_pc_goal113_pitch_tracking_authority.py` 与单元测试；
- [x] Step 2：本地 synthetic smoke，覆盖 authority/rate/coupling/preexisting/no-lag/mixed 六类；
- [x] Step 3：远端只读运行 Goal107 trace，生成 JSON/Markdown；
- [x] Step 4：写 result、decision、reading note 和 LEDGER；
- [x] Step 5：运行 checker；不自动 commit/push。

## 7. 复现命令

```bash
cd /home/bubble/水下机器人/settled_局部规划器_反应式
PYTHONPATH=src ../.venv/bin/python -m pytest -q tests/test_pc_goal113_pitch_tracking_authority.py
PYTHONPATH=src ../.venv/bin/python scripts/analyze_pc_goal113_pitch_tracking_authority.py \
  --root artifacts/lowlevel/goal107_terminal_reward_credit_20261111 \
  --output artifacts/lowlevel/goal113_pitch_tracking_authority_20261111/phase0_summary.json \
  --markdown artifacts/lowlevel/goal113_pitch_tracking_authority_20261111/phase0_summary.md
python3 ../scripts/check_research_ledger.py \
  --goal Goal113 \
  --goal-doc doc/goals/goal113_pitch_tracking_authority_audit_20261111.md \
  --strict --require-result
```

## 8. 不做什么

- 不训练 PPO/SAC/REDQ/DroQ/DADE；
- 不修改 reward、terminal、success、goal radius、dev seeds；
- 不打开 held-out `100–149`；
- 不先改 residual scale、动作参数化或 controller；
- 不把 Spearman 相关性写成因果；
- 不因为 Goal112 的 tracking 多数就直接开 SAC；
- 不用 strict U050/default 的事后结果改写 strict C0 主判决。

## 9. Goal 提示词

```text
你现在执行 Goal113：只读审计 Goal112 判为 tracking-limited 的 11 个 strict C0
pre-entry failure，分解 pitch tracking 失败是 command rate 不可达、stern/fin
authority 不足、深度/速度/洋流耦合，还是更早阶段已经形成的 pitch lag。

先读 doc/goals/goal113_pitch_tracking_authority_audit_20261111.md、
Goal112 result/decision、Goal111 result、Goal109 result、Goal108 network audit、
Goal85/86/89/104 契约和 Goal107 trace 字段契约。

硬约束：不训练；不改 reward/terminal/success/goal radius/dev seeds；不打开
held-out；不改动作参数化、controller 或 residual scale；不事后修改窗口、阈值、
互斥分类或 Gate。

执行顺序：实现并测试分析器 -> 远端只读运行 Goal107 trace -> 按冻结 Gate 写
result/decision/reading note/LEDGER -> 只有单一机制明确 >=6/11 才写对应下一步；
如果是 mixed/ambiguous，只写 joint pre-registration。只有在 pitch command 可达、
direction 正确且没有 authority/rate/coupling 证据时，才允许新开 SAC 同协议对照。
```

## 10. 自主迭代契约

本 Goal 执行 `research/AUTORESEARCH_LOOP.md`。失败后先归因，至少写两个可证伪
hypothesis，使用 `paper-search` 或 `deep-research` 查最接近的 2–5 篇论文并写 reading note，
先跑最便宜 falsification，再在 `research/LEDGER.jsonl` 写 `continue` / `pivot` / `stop`。
不自动 commit/push。
