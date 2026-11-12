# Goal115：joint pitch handoff 的 guidance / filter / plant / residual 审计

**日期**：2026-11-11<br>
**状态**：`COMPLETED_AUDIT_ONLY / REFERENCE_FILTER_LIMITED / NO_SAC / HELDOUT_CLOSED`<br>
**上游**：Goal114 `MIXED_PITCH_HANDOFF`；Goal113 `PITCH_TRACKING_AMBIGUOUS`；Goal107 冻结 P_D3 trace<br>
**性质**：只读 + 行为不变的 trace instrumentation；不训练、不改 reward/terminal/success/goal radius，不打开 held-out。

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
  check_command: python3 ../scripts/check_research_ledger.py --goal Goal115 --goal-doc doc/goals/goal115_joint_pitch_handoff_20261111.md --strict --require-result
```

记录前不得修改主分析子集、机制分类阈值、seed、scenario、checkpoint、success frame 或 Gate 判读规则。

## 1. 背景与问题

Goal114 在 strict C0 的 11 个 `PREENTRY_GEOMETRY_BAD` failure 上得到：

```text
MIXED_PITCH_HANDOFF = 8/11
RATE_DEMAND_FIRST   = 2/11
SPEED_CURRENT_FIRST = 1/11
AUTHORITY_FIRST     = 0/11
TRACKING_ERROR_FIRST = 0/11
```

中位 onset 相对 2 m：rate/speed-current `-0.84 s`、authority/tracking `-0.66 s`，
只差约 `0.18 s`，落在 `0.20 s` mixed 窗口内。Goal114 还发现
`pitch_cmd_rate_abs`、`rate_error_abs`、`rate_deficit_abs` 分别有 `10/11`、`6/11`、
`5/11` 触发固定 fallback，因此必须把 filter 前目标与 filter 后命令拆开。

Goal115 要回答的机制问题是：

```text
2 m 前的 pitch command rate 到底来自：
  A. guidance/reference 本身要求快速变 pitch；
  B. command rate limiter 削掉了更大的 guidance 目标；
  C. limiter 输出可达，但 plant 的 actual pitch rate 跟不上；
  D. command/plant 都可达，但 residual 的 stern 修正方向不对。
```

Goal114 只能看到经过 `_limit_guidance_command()` 的 `command`，无法区分 A/B/C。
本 Goal 只增加行为不变的诊断字段，随后用同 checkpoint、同 config、同 seeds
复跑 eval-only，并与 Goal107 trace 做逐帧 parity。

## 2. 数据与隔离

- checkpoint：`artifacts/lowlevel/goal95_supervised_residual_20261111/P_D3/final.pt`；
- checkpoint SHA256：`2d57745a811268553041c561f2ae1ad590dced472b712efd90cdd9c011c1a616`；
- 训练配置：`configs/lowlevel/goal95_20261111/p_d3_ppo.json`；
- 环境：Goal85 `unbiased_noise_v1` default / strict125 candidate；
- action mode：`frozen-base`；
- eval scale：`2.00`；
- dev seeds：`20–59`，每 run 40 episodes；
- success frame：`both`；
- held-out `100–149`：关闭；
- 主分析 run：`strict125 / C0`；
- 描述性对照：`default / C0`、`strict125 / U050`；不参与主 Gate；
- 不训练、不改 checkpoint、不改 reward/terminal/success/goal radius；
- 旧 Goal107 trace 只读，不能覆盖或重写。

primary set 固定为 Goal111/114 的 11 个 strict C0 `PREENTRY_GEOMETRY_BAD`
failure；其中 8 个 Goal114 mixed 样本单独列出，不改变 11 个样本的 Gate 分母。

## 3. 行为不变 instrumentation

在 `BatchedPCLowLevelEnv._alos_command()` 的 `_limit_guidance_command()` 前后记录：

```text
guidance_target_heading_rad
guidance_target_pitch_rad
guidance_target_speed_mps
guidance_limited_heading_rad
guidance_limited_pitch_rad
guidance_limited_speed_mps
guidance_pitch_rate_clipped
guidance_heading_rate_clipped
guidance_speed_rate_clipped
```

要求：

- 只新增 trace/infos 字段，不改变 command、controller、plant、reward、termination 或 success；
- `guidance_limited_pitch_rad` 必须等于当前步实际送入控制器的 pitch command；
- `guidance_pitch_rate_clipped` 只在 limiter 启用且 raw delta 超过 `limit * dt` 时为真；
- 复跑 trace 必须和 Goal107 trace 的 success、terminal kind、command、eta、nu、
  actuator 序列逐 seed 相等；若不相等，状态为 `INVALID_INSTRUMENTATION_PARITY`。

## 4. 指标与分段

### 4.1 2 m 锚定窗口

- 首次进入 `3.0 m` 到首次进入 `2.0 m + 1.0 s`；
- 主 onset/统计窗口与 Goal114 保持一致；
- 3.0→2.5→2.0 m 分段快照只作解释，不改变 Gate。

### 4.2 Pitch 信号

```text
pitch_error       = command_pitch - eta_pitch
target_pitch_rate = d(guidance_target_pitch)/dt
limited_pitch_rate = d(guidance_limited_pitch)/dt
actual_pitch_rate = nu[:, 4]
target_excess_rate = max(abs(target_pitch_rate) - pitch_rate_limit, 0)
filter_attenuation = target_pitch_rate - limited_pitch_rate
tracking_deficit = abs(limited_pitch_rate) - abs(actual_pitch_rate)
plant_gain = actual_pitch_rate / limited_pitch_rate  (|limited| > 0.05 rad/s)
stern_norm = abs(actuator_total_physical[:, 1]) / radians(20)
residual_stern = actuator_residual_applied[:, 1]
```

所有 rate 使用 `np.gradient(..., time_s)`；连续 3 帧才算持续越界。

注意：Goal107/114 使用 `action_mode=frozen-base`，`residual_action` / PPO 增量是
严格零；这里的 `residual_stern` 指冻结 BC 的 `actuator_residual_applied`，不是
PPO 学习残差。本 Goal 不评价 PPO 增量方向。

### 4.3 每 episode 机制判读

对每个 failure 先计算 3.0→2.0 m 窗口内：

- `clip_fraction`：`guidance_pitch_rate_clipped` 的比例；
- `median_target_excess_rate`；
- `median_limited_rate`；
- `median_tracking_deficit`；
- `stern_norm_p95`；
- `actuator_residual_applied[:, 1]`（冻结 BC stern 残差）与 `pitch_error` 的
  Spearman 相关；
- success control 的同类统计，用于确定 BC residual 方向参考。

`residual_action` 是 PPO 增量，在 `frozen-base` 下为零，不用于本 Goal。

分类优先级：

| 分类 | 冻结条件 |
|---|---|
| `REFERENCE_FILTER_LIMITED` | `clip_fraction >= 0.25` 且 `median_target_excess_rate >= 0.05 rad/s`，且 limiter 输出已接近上限；说明 guidance 目标被 filter 截断 |
| `PLANT_RESPONSE_LIMITED` | `clip_fraction < 0.25` 或 `median_target_excess_rate < 0.05`，但 `median_tracking_deficit >= 0.05 rad/s`；说明 limited command 可达而 plant 跟不上 |
| `RESIDUAL_DIRECTION_LIMITED` | 前两类都不成立，且 failure 的 BC residual-stern/pitch-error 相关方向与 success control 相反，或 `abs(correlation) < 0.10` |
| `MIXED_HANDOFF_CONTINUE` | 多条条件同时成立或都不稳定 |
| `NOT_EVALUABLE` | 字段缺失、非有限或未进入 2 m |

`clip_fraction`、`target_excess_rate`、`tracking_deficit` 的阈值在测量前固定；不得
在看过结果后移动。若某个字段在 success control 中缺失，整个 run 判
`INVALID_INSTRUMENTATION`。

## 5. 假设

### H115-A：filter 是瓶颈

机制：guidance 的 raw pitch target 变化快于 `pitch_rate_limit_rad_s`，limiter 把
command 削到上限，plant 实际能跟上 limited command，但 limited command 不足以
消除 pitch error。

成立：11 个 failure 中 `>=6` 判 `REFERENCE_FILTER_LIMITED`，且 success control
的 clip fraction 明显更低。

失败：clip fraction 很低，或 raw target 没有明显超出 limit。

最便宜证伪：同 checkpoint、同 seeds 的 instrumentation 复跑；不需要训练。若 H115-A
成立，下一 Goal 才做 rate-limit/reference 的 eval-only screen。

### H115-B：plant response 是瓶颈

机制：guidance target 和 limiter 都在可达范围内，但 actual pitch rate 长期低于
limited command；stern 可能接近限幅。

成立：`>=6` 判 `PLANT_RESPONSE_LIMITED`，且 `median_tracking_deficit` 在 failure
中稳定大于 success control。

失败：limited command 与 actual rate 的差很小，或 stern 未接近限幅。

最便宜证伪：同一 trace 的 limited-vs-actual gain/lag 计算；不训练。

### H115-C：BC residual 方向是瓶颈

机制：pitch command 可达、plant 跟得上，但冻结 BC 的 stern 修正与 pitch error
方向不一致，导致闭环不改善。

成立：`>=6` 判 `RESIDUAL_DIRECTION_LIMITED`，且 failure 的相关方向与 success
control 相反。

失败：BC residual 方向与 success 一致，或 residual 幅度接近零。

注意：本 Goal 的 trace 是 `frozen-base`，PPO 增量严格为零；不能把 BC residual
的符号问题写成 PPO 残差问题。

最便宜证伪：同 trace 的 residual-stern/pitch-error 对齐；不训练。

### H115-D：仍是 joint handoff

成立：没有单一分类 `>=6/11`，或 `MIXED_HANDOFF_CONTINUE >=6/11`。

处理：不换 PPO/SAC/REDQ/DroQ；写 blocker 和至少两个新的可证伪 hypothesis，
再决定是否进入 reward/terminal/action parameterization。

## 6. 执行计划

- [x] Step 0：写 Goal115 预注册与 Research Contract；
- [x] Step 1：增加 behavior-preserving trace fields 和单测；
- [x] Step 2：本地合成测试，确认 limiter on/off 字段语义；
- [x] Step 3：远端同 checkpoint、同 config、同 dev seeds 复跑 4 个 eval run；
- [x] Step 4：与 Goal107 trace 做逐 seed parity；不通过则 `INVALID`；
- [x] Step 5：跑 Goal115 分析器，写 result/decision/reading note/LEDGER/STATE；
- [x] Step 6：运行 checker 和 `git diff --check`；不自动 commit/push。

## 7. Gate

| check | 阈值 |
|---|---:|
| primary failure set | 11 个 strict C0 `PREENTRY_GEOMETRY_BAD` |
| instrumentation parity | success/terminal/command/eta/nu/actuator 逐 seed 相等 |
| 可评估数 | `>=9/11` |
| 单机制多数 | `>=6/11` |
| 稳定性 | 0 NaN / 0 非有限 / 0 held-out seed |
| 负控 | success control 不参与 failure 分类，只用于阈值/方向参考 |

失败裁决：

| 观察 | 判读 | 下一步 |
|---|---|---|
| filter 截断占多数 | `REFERENCE_FILTER_LIMITED` | 新 Goal 做 rate-limit/reference eval-only screen，不换算法 |
| limited 命令可达但 actual 跟不上 | `PLANT_RESPONSE_LIMITED` | 查 stern authority/plant response，不换算法 |
| command/plant 可达但 residual 方向错 | `RESIDUAL_DIRECTION_LIMITED` | 回到 reward/action parameterization |
| 混合且无多数 | `MIXED_HANDOFF_CONTINUE` | 写 blocker，继续 joint audit |
| parity 或字段失败 | `INVALID` | 先修 instrumentation，不分析机制 |

## 8. Git

禁止 `git add -A`。只 add 本次相关文件，提交前运行：

```bash
python3 ../scripts/check_research_ledger.py \
  --goal Goal115 \
  --goal-doc doc/goals/goal115_joint_pitch_handoff_20261111.md \
  --strict --require-result
git diff --cached --check
```

不自动 commit/push；不提交 `artifacts/`、checkpoint、日志、缓存。

## 9. Goal 提示词

```text
你现在执行 Goal115：joint pitch handoff 的 guidance/filter/plant/residual 审计。

先读：
- doc/goals/goal115_joint_pitch_handoff_20261111.md
- doc/goals/goal114_result_20261111.md
- doc/goals/goal107_result_20261111.md

硬约束：
1. 固定 P_D3 checkpoint、Goal85 unbiased、dev seeds 20–59、strict125 C0 主分析；
2. 不训练、不改 reward/terminal/success/goal radius、不打开 held-out；
3. 只增加行为不变的 trace instrumentation；先做 goal107 parity；
4. 不换 PPO/SAC/REDQ/DroQ/DADE，不根据失败结果移动阈值；
5. filter、plant、residual 三类机制必须分开判，不用单一 success 猜测。

执行顺序：
- 增加 guidance target/limited/clipped 字段并写单测；
- 本地 smoke；
- 远端同 checkpoint/config/seeds 复跑并验证 parity；
- 跑分析器，写 result、decision、reading note、LEDGER、STATE；
- 运行 checker 和 git diff --check；不自动 commit/push。
```
