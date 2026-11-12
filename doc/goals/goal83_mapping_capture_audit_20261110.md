# Goal83：residual → actuator → state → p_nav/eta → capture 的只读映射与捕获帧审计

**日期**：2026-11-10<br>
**状态**：`RESULT_RECORDED / CAPTURE_FRAME_AMBIGUITY`<br>
**上游**：Goal71 `goal71_command_rate_limit`、Goal80 `TERMINAL_REFERENCE_HURTS`、Goal81 `GUARD_NOT_PREEMPTIVE`、Goal82 `CHANNEL_GAIN_WITHOUT_OUTCOME`<br>
**性质**：只读 trace / episodes / config / code 审计；不训练、不新增 rollout、不改 reward、controller、checkpoint、seed 或 held-out。

**结果文档**：`doc/goals/goal83_result_20261110.md`

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
  check_command: python ../scripts/check_research_ledger.py --goal Goal83 --goal-doc doc/goals/goal83_mapping_capture_audit_20261110.md --strict --require-result
```

`required: true`：这是诊断型 goal。若 mapping 与 capture frame 都不能解释失败，必须写
blocker、至少两个可证伪 hypothesis 和 ledger decision，不能直接改 reward 或换 SAC。

## 1. 背景与证据

Goal82 已记录：

- `H82-A` 支持，但支持来自 command tracking error，不是 residual 三通道幅度；
- `H82-B` 不支持：scale clip / rate clip 全为 0，material saturation 和 residual
  direction 没有稳定方向；
- `H82-C` 支持：terminal outcome 只在 done 行激活，command tracking 是主导 reward group；
- `H82-D` 支持，但主要来自 strict125：policy 的 heading tracking 优于 zero，
  true success 反而更低（baseline `-0.069`，candidate `-0.106`）；
- 所有 lost 都是 `GOAL_PLANE_MISS`，不是 guard 抢跑，也不是 NoisyNet 或 residual
  幅度主导。

因此本轮不再问“reward 该不该改”，而是问更上游的三件事：

1. residual 的 base、时间对齐、符号、尺度和 rate limit 是否与 trace / code 一致；
2. guidance、reward、termination、success 是否使用同一个 `p_nav` / `eta` 和
   goal-tangent / path-local frame；
3. policy 的 tracking 增益是在 actuator、state、`p_nav/eta` 还是 capture 环节丢失。

## 2. 固定数据与隔离

| 项目 | 固定值 |
|---|---|
| 数据索引 | `artifacts/lowlevel/goal82_action_reward_attribution_20261110/goal82.json` |
| baseline | Goal71 `goal71_command_rate_limit`，extension `0.0`，10 runs |
| candidate | Goal80 `goal80_terminal_reference_20261110`，extension `1.0`，10 runs |
| 每个 root | 4 policy checkpoints × 2 scenarios + 2 zero-residual |
| seeds | dev `20–59`，40 seeds/run |
| episodes | `20 × 40 = 800` |
| eval | CPU/float64、`endpoint_stop`、dwell=50、`r_tight=r0-0.25 m`、guard=`r0` |
| held-out | `100–149` 关闭 |
| 训练 / 新 rollout | 无 |
| 算法切换 | 禁止；SAC/REDQ/DroQ 均不启动 |

`goal82.json` 只作为 run 索引和 label 来源。所有数值都必须回到同一 run 的
`trace.npz`、`episodes.json`、`summary.json` 和 environment config 重新计算。

## 3. 映射定义

固定语义：

```text
base_command[t]      = ALOS 在当前步 residual 施加前的命令
residual[t]          = 经 gate / scale clamp / rate limit 后的物理 command delta
corrected_command[t] = base_command[t] + residual[t]
command_physical[t]  = 执行动作后下一步观测到的命令
```

因此必须满足：

```text
heading: wrap_pi(command_physical[t] - base_command[t+1]) = 0
pitch:   command_physical[t] - base_command[t+1]        = 0
speed:   command_physical[t] - base_command[t+1]        = 0
```

容差统一为 `1e-8`（角度用 `wrap_pi`）。若 trace 不支持该关系，记
`INVALID_DIAGNOSTIC`，不能继续把 `command_physical[t]` 当作当前 residual 的基座。

residual 物理量必须满足配置中的 scale 和 rate limit：

```text
abs(residual[t, c]) <= scale[c] + 1e-8
abs(residual[t, c] - residual[t-1, c]) <= rate[c] * dt_s + 1e-8
```

其中 `t=0` 的 previous delta 固定为 0。若 `active[t]` 为假，按环境的 hold 语义
跳过该步的 rate 检查，并显式报告跳过步数。

## 4. 捕获帧定义

当前 `endpoint_stop` 下，终止和 success 使用 `p_nav`；`terminal_eta_capture` 是
真实状态 `eta` 的诊断量。必须逐 episode 验证：

> 实现口径澄清（结果记录后补充，不改假设或阈值）：trace_meta 的
> `eval_goal_radius_m` 是 tight radius，用于 `p_nav` 的 GOAL/success；
> `plant.goal_radius_m` 与 plane-miss guard 相等，是 protocol radius，用于
> `terminal_eta_capture` 标签。H83-B 的同帧比较统一使用 tight radius，避免把
> protocol 标签半径误当成正式 success 半径。

```text
d_pnav[t]       = ||p_nav[t] - goal||
signed_pnav[t]  = (p_nav[t] - goal) · goal_tangent
d_eta[t]        = ||eta[t] - goal||
signed_eta[t]   = (eta[t] - goal) · goal_tangent
```

终止规则固定为：

```text
goal_reached   = d_pnav <= r0
plane_active   = (d_pnav > guard) and (signed_pnav >= 0)
dwell_count    = consecutive plane_active steps
plane_miss     = plane_active and dwell_count > dwell_steps
terminal_kind  = GOAL if goal_reached else GOAL_PLANE_MISS if plane_miss else ...
```

`terminal_kind == GOAL` 和 `terminal_kind == GOAL_PLANE_MISS` 必须能被上述 p_nav
规则在 done 行复现；否则记 `CAPTURE_FRAME_MISMATCH`。`eta` 不用于正式 success，
只用于判断 p_nav/eta 帧差异是否把 true episode 判成 lost。

## 5. 可证伪 hypothesis

### H83-A：residual 命令映射存在可量化缺陷

检查 base timing、residual scale/rate、`goal_distance` 与 `signed_offset` 的
p_nav 定义、done `terminal_kind` 复现。

- 支持：任一检查在至少一个 run 中超过容差，并且该不一致能够改变至少一个 lost
  episode 的 terminal 判定或使终端 `p_nav` 偏差下降 `>= 0.10 m`。
- 不支持：全部检查在 `1e-8` 容差内通过，且反事实重算不改变任何 episode 的类别。
- 最便宜 falsification：只读现有 800 episodes，不新增 rollout。

### H83-B：p_nav/eta 捕获帧不一致解释一部分 lost

- 支持：至少一个 group 中 `>=10%` 的 lost 在 done 或首次 crossing 处出现
  `p_nav capture != eta capture`，或 p_nav/eta 的 plane sign 相反；同时报告该
  frame 规则会新增多少 false-success。
- 不支持：所有 lost 的 p_nav/eta capture 和 plane sign 在 done 与 crossing 都
  一致，或差异只出现在不改变结论的 retained episode。
- 该 hypothesis 只判断“帧差异是否解释失败”，不把 eta 诊断量改写成正式 success。

### H83-C：mapping 和 capture frame 一致，增益在 terminal chain 中丢失

- 支持：H83-A/B 不支持；至少一个 scenario 中 policy 相对 zero 改善 heading 或
  pitch tracking（paired median improvement，95% CI 不跨零，`relative_diff <= -0.10`），
  但 terminal `p_nav` distance 或 `eta` capture 没有改善。
- 不支持：policy 连 tracking 都没有改善，或 tracking 改善同时带来 terminal
  `p_nav`/`eta` 改善。
- 此时下一步查 actuator 响应、terminal approach 或 controller bandwidth，不先改
  reward，也不开 SAC。

## 6. 指标

完整性：

- run 数、episode 数、checkpoint SHA、seeds、r0/tight/guard、dwell、action mode；
- 唯一 done 行、step 连续、时间严格递增、trace finite、无 NaN/Inf；
- missing channel / frame 字段记 `INVALID_DIAGNOSTIC`，不能静默跳过。

映射：

- `base_timing_*_max_abs_error` 和 violation 数；
- `residual_scale_violation_count`、`residual_rate_violation_count`；
- `goal_distance_frame_max_abs_error`、`signed_offset_frame_max_abs_error`；
- `terminal_kind_replay_match_fraction`。

捕获帧：

- done 和首次 p_nav crossing 的 `d_pnav`、`d_eta`、`signed_pnav`、`signed_eta`、
  perp offset、p_nav/eta gap；
- `pnav_capture != eta_capture` 的 episode 数；
- `pnav_plane_sign != eta_plane_sign` 的 episode 数；
- 若用 eta 规则替代 p_nav 规则，true recovery 和 additive false-success 各多少。

terminal chain：

- policy vs zero 的 paired heading/pitch tracking error；
- terminal `p_nav` distance、`eta` distance、signed offset、lateral/vertical offset；
- `terminal_eta_capture`、true/false success、lost/retained 计数；
- 每个 group 的 seed 数和中位数；所有 paired effect 报 95% CI，不只报 p 值。

## 7. 预注册决策矩阵

| 观察 | 状态 | 下一步 |
|---|---|---|
| 任一映射/终止复现检查失败 | `MAPPING_OR_FRAME_DEFECT` | 先修实现，不训练、不改 reward |
| H83-B 支持且 frame 规则能恢复 lost 且 false-success 可接受 | `CAPTURE_FRAME_AMBIGUITY` | 预注册单因素 eval-only capture-frame arm |
| H83-A/B 不支持，H83-C 支持 | `TERMINAL_CHAIN_LOSS` | 查 actuator / terminal approach，不先改 reward |
| H83-A/B/C 都不支持 | `UNRESOLVED_CHAIN` | 写 blocker，进入 autoresearch loop |
| 任一完整性失败 | `INVALID_DIAGNOSTIC` | 只修解析链路，重跑同一审计 |

“false-success 可接受”固定为 `additive false-success <= 0.05`，与 Goal81 的预注册
阈值一致。任何阈值、seed、eval split 或 held-out 状态在正式结果记录后都不能修改。

## 8. 执行计划

- [x] Step 0：冻结本预注册；核对 20 个 run 的路径、schema、配置和 trace 字段；
- [x] Step 1：实现 `scripts/analyze_pc_goal83_mapping_capture.py`；
- [x] Step 2：写 `tests/test_pc_goal83_mapping_capture.py`，覆盖 base timing、
      scale/rate、goal frame、terminal replay、p_nav/eta frame 和缺失字段；
- [x] Step 3：在远端 20 个已记录 run 上只读并行解析（`--jobs 20`，每进程单线程 BLAS）；
- [x] Step 4：按 H83-A/B/C 和 §7 矩阵判定；
- [x] Step 5：写 `doc/goals/goal83_result_20261110.md` 和
      `research/decisions/2026-11-10-goal83-mapping-capture-decision.md`；
- [x] Step 6：更新 `research/LEDGER.jsonl`、`research/STATE.md`，运行 ledger checker；
- [x] Step 7：运行定向 pytest、ruff、`git diff --check`；
- [ ] Step 8：用户明确要求后才 commit/push。

## 9. 算力与隔离

本 Goal 只读 20 个已记录 run，约 800 episodes；不做新 rollout，不占 GPU。
远端使用 20 个进程并行读取 `trace.npz`，每进程单线程 BLAS。若分析规模超过
run×seed 分片能力，再按 run 分片，不改变数据或假设。

训练型 arm 的前置条件仍是：

```text
num_envs 128/256/512 × batch_size 256/512
单卡 throughput 基准 -> 按 seed 分 8 卡 -> 独立预注册
```

## 10. Git

禁止 `git add -A`。只 add 本 Goal 相关文件，提交前运行：

```bash
python ../scripts/check_research_ledger.py \
  --goal Goal83 \
  --goal-doc doc/goals/goal83_mapping_capture_audit_20261110.md \
  --strict --require-result
git diff --cached --check
```

不提交 `artifacts/`、checkpoint、日志、trace 或缓存。未经用户明确要求，不 commit、
不 push。

## 11. Goal 提示词

```text
你现在执行 Goal83：residual → actuator → state → p_nav/eta → capture 的只读映射与
捕获帧审计。

先读：
- doc/goals/goal83_mapping_capture_audit_20261110.md
- doc/goals/goal82_result_20261110.md
- doc/goals/goal81_result_20261110.md
- research/STATE.md
- AGENTS.md

硬约束：
1. 只读复用 Goal71 baseline、Goal80 candidate、Goal60 reference；
2. 不训练、不新增 rollout、不改 reward/controller/checkpoint/seed/held-out；
3. 不启动 SAC/REDQ/DroQ；
4. 先冻结本预注册，再实现 analyzer；
5. 缺失字段记 INVALID_DIAGNOSTIC，不静默跳过；
6. 结果、ledger、STATE 全部中文；只 add 本次相关文件；
7. 不自动 commit/push。

执行顺序：
- Step 0：核对 20 runs、800 episodes、trace 字段和配置；
- Step 1：实现 mapping / capture-frame analyzer；
- Step 2：写合成测试；
- Step 3：远端 20 进程并行只读解析；
- Step 4：按 H83-A/B/C 判定；
- Step 5：写 result、decision、LEDGER、STATE，运行 checker。
```

## 12. 结果摘要

- 正式状态：`CAPTURE_FRAME_AMBIGUITY`；
- H83-A 不支持：base timing、scale/rate、goal-frame projection、terminal replay
  全部通过；
- H83-B 支持：done 时刻无分歧，但首次 p_nav crossing 的 plane-sign 分歧出现在
  四个 group；candidate default 有 `4/15` 的 crossing capture 分歧；
- true eta tight-radius oracle 恢复 `12/29` lost，四个 group 的 additive
  false-success 均不超过 `0.05`；eta 仍是 oracle，不可直接部署；
- H83-C 总体不支持（预注册要求 H83-A/B 都不支持），但 strict125 有次级信号：
  heading tracking 改善 `-0.1305` / `-0.1151`，terminal distance 改善的 CI
  跨零，true diff 为负；
- 不修改 reward、动作参数化、controller、checkpoint、seed 或 held-out，不启动
  SAC/REDQ/DroQ；
- 下一步：预注册 eval-only capture-frame 单因素审计。

