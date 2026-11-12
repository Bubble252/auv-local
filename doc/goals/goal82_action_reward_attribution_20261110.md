# Goal82：action parameterization 与 reward 的只读归因审计

**日期**：2026-11-10<br>
**状态**：`RESULT_RECORDED / CHANNEL_GAIN_WITHOUT_OUTCOME`<br>
**上游**：Goal27 `H3_SR_FAIL_STOP`、Goal80 `TERMINAL_REFERENCE_HURTS`、Goal81 `GUARD_NOT_PREEMPTIVE`<br>
**性质**：只读 trace / episodes 审计；不训练、不新增 rollout、不改 reward、controller、checkpoint、seed 或 held-out。

**结果文档**：`doc/goals/goal82_result_20261110.md`

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
  check_command: python ../scripts/check_research_ledger.py --goal Goal82 --goal-doc doc/goals/goal82_action_reward_attribution_20261110.md --strict --require-result
```

`required: true`：这是诊断型 goal。若 action 与 reward 审计不能区分失败机制，必须写 blocker、
至少两个可证伪 hypothesis 和 ledger decision，不能直接换 SAC/REDQ/DroQ。

## 1. 背景与当前证据

Goal81 已记录：

- 所有 lost 都是 `GOAL_PLANE_MISS`，`100%` 未进入 `r_tight`；
- lost 在终止前约 `0.88–1.36 s` 已过最近点，随后外离 `0.60–1.25 m`；
- `dwell=50` 已等满，plane-miss guard 不是抢跑；
- deterministic residual L2 的 lost/retained 比值约 `1.17`，zero-residual 仍有
  `14` 个 lost；NoisyNet / residual magnitude 不是 lost 的稳定判别因素；
- default 存在 `p_nav/eta` 捕获帧差异，但 strict125 不能外推。

因此下一步不是继续扫 termination、capture radius、dwell 或 residual scale，也不是
直接换 SAC。需要先回答：

1. residual 在 heading、pitch、speed 三个可部署 command 通道里究竟修正了什么？
2. 它是在正确通道产生有效修正，还是在补偿、抖动、饱和或顶住 rate limit？
3. terminal 前窗口 reward 主要奖励什么？success/plane-miss 是否太弱、太晚，或被
   command tracking / saturation / energy 淹没？
4. policy 相对 zero-residual 是否改善了关键通道但没有改善 true outcome？
5. 如果 action 和 reward 都不能解释，是否应回到估计器、控制器带宽或 pre-crossing？

## 2. 固定数据与隔离

| 项目 | 固定值 |
|---|---|
| baseline | Goal71 `goal71_command_rate_limit`，extension `0.0`，10 runs |
| candidate | Goal80 `goal80_terminal_reference_20261110`，extension `1.0`，10 runs |
| reference | Goal60 `artifacts/traces` 对应 run 的 `episodes.json` |
| 每个 root | 4 policy checkpoints × 2 scenarios + 2 zero-residual |
| seeds | dev `20–59`，40 seeds/run |
| episodes | `20 × 40 = 800` |
| eval | CPU/float64、`endpoint_stop`、dwell=50、`r_tight=r0-0.25 m`、guard=`r0` |
| held-out | `100–149` 关闭 |
| 训练 / 新 rollout | 无 |
| 算法切换 | 禁止；SAC/REDQ/DroQ 均不启动 |

`reference_true(seed, key, scenario) = success and terminal_eta_capture`，来自 Goal60
对应 run。对每个 policy episode：

```text
lost     = reference_true and not candidate_true
retained = reference_true and candidate_true
```

所有比较必须按同一 root、scenario、checkpoint/key 和 seed 对齐；不能把不同
checkpoint 的 episode 当成独立样本。

## 3. 通道定义与消歧

主假设只使用三个可部署 residual 通道：

| 主通道 | trace 列 | 物理单位 | 归一化尺度 |
|---|---|---|---:|
| heading residual | `residual[:, 0]` | rad | `scale_heading_rad` |
| pitch residual | `residual[:, 1]` | rad | `scale_pitch_rad` |
| speed residual | `residual[:, 2]` | m/s | `scale_speed_mps` |

`residual` 是经过 gate、scale clamp 和 rate limit 后的 **物理 command delta**，不是
normalized actor output；分析器必须先通过 `pc_command_residual.py` 的语义和
`base_command` / `command_physical` 校验这一点。

提示词中“四个通道”的解释：

- 第四个不是第二个可学习 residual，而是 supplementary authority channel：
  `actual_actuator` 的 rudder / stern / rpm，加上 `scale_clipped`、`rate_clipped`、
  `material_saturated`；
- 它只解释 residual 是否真的转化为执行器权限、是否被限幅/饱和，不并入 H82-A 的
  三通道主判定；
- 如果 trace 或配置显示 residual 不是 heading/pitch/speed 三通道，停止解释并记
  `INVALID_DIAGNOSTIC`，不得静默改变定义。

### 3.1 trace 对齐说明（实现审计修正，未改假设/阈值）

Goal71/Goal80 的 recorded trace 中，`command_physical[t]` 与 `base_command[t+1]`
逐元素相等（100% 精确）。代码审计确认：`base_command[t]` 是 residual[t] 施加前的
当前步 ALOS 命令，`command_physical[t]` 是执行动作后下一步观测到的命令。因此
residual 的基座必须使用 `base_command[t]`，而不是 `command_physical[t]`：

```text
corrected_command[t] = base_command[t] + residual[t]
```

其中 heading 按 `wrap_pi` 处理。`command_physical[t] == base_command[t+1]` 只作为
trace 时序一致性检查；若该关系不成立，记 `INVALID_DIAGNOSTIC`。此修正只改变
分析器的字段选择，不改变 H82-A/B/C/D/E、窗口、阈值或 eval split。

## 4. 窗口定义

对每个 episode 的 done 行定义两个窗口，二者都必须报告：

1. `pre2s`：`time_s >= t_done - 2.0 s`；
2. `last20m`：最后一次连续满足 `goal_distance <= 20.0 m` 的区段。

每个窗口再分成：

- `preterminal`：窗口内、done 行之前的行；用于 action、tracking、饱和和 reward 主导分析；
- `full`：窗口包含 done 行；只用于 outcome 激活时刻、done 动作和终态诊断。

若 `preterminal` 少于 `5` 行、episode 起点已在 20 m 内但无法形成连续段、或窗口
不足，报告中记 `insufficient_window`，不插补、不改变主统计。对 `last20m` 的连续性
定义是：从最后一次从 `>20 m` 进入 `<=20 m` 的行开始，直到 done。

## 5. Action 指标

每个窗口、每个 episode 计算以下指标。连续量先按通道计算，再按 seed-cluster
bootstrap 汇总。

### 5.1 residual 通道

- `residual_signed_mean[c]`；
- `residual_abs_mean[c]`；
- `residual_std[c]`；
- `residual_p95_abs[c]`；
- `residual_norm_mean`、`residual_norm_p95`；
- `gate_mean`、`gate_active_fraction`。

### 5.2 clip / saturation

- `scale_clip_fraction[c]`、`any_scale_clip_fraction`；
- `rate_clip_fraction[c]`、`any_rate_clip_fraction`；
- `material_saturation_fraction`；
- supplementary `actual_actuator` 的 rudder / stern / rpm 均值和 p95。

### 5.3 command tracking

用 `command_physical`、`base_command`、`eta_rpy`、`ground_velocity_body_mps` 计算：

- heading tracking error：`abs(wrap(command_heading - eta_yaw))`；
- pitch tracking error：`abs(command_pitch - eta_pitch)`；
- speed tracking error：`abs(command_speed - ground_surge)`；
- 对 base command 计算同一组误差，并报告
  `tracking_improvement = base_error - corrected_error`；
- residual 对 terminal-LOS 的方向：
  - `los_heading = atan2(goal_east - p_nav_east, goal_north - p_nav_north)`；
  - `los_pitch = atan2(-(goal_down - p_nav_down), horizontal_distance)`；
  - `heading_error = wrap(los_heading - base_heading)`；
  - `pitch_error = los_pitch - base_pitch`；
  - `direction_cosine_hp = dot(residual_hp, error_hp) / (norm(residual_hp) * norm(error_hp))`；
  - speed 只报告 supplementary `speed_sign_agreement`，使用
    `sign(residual_speed) * sign(reference_speed - base_speed)`；若 reference speed
    不可解析，记 `INVALID_DIAGNOSTIC`，不把它塞进 heading/pitch cosine。

### 5.4 command-to-actuator 代理

映射固定为 `heading -> rudder`、`pitch -> stern`、`speed -> rpm`。对通道差分：

```text
delta_residual[c](t) = residual[c](t) - residual[c](t-1)
delta_actual[c](t)  = actual_actuator[c](t) / actuator_scale[c]
                      - actual_actuator[c](t-1) / actuator_scale[c]
```

在 `-0.20 s` 到 `+0.20 s` 的 lag 网格上计算 Pearson correlation，取绝对值最大的
lag；报告 `lag_s`、`correlation`、`gain`（actual delta 对 residual delta 的回归斜率）
和 `direction_cosine`（差分向量夹角余弦）。这是控制器/执行器响应的代理指标，不是
严格系统辨识；若任一侧方差接近零，记 `None` 并报告 coverage。

## 6. Reward 字段语义与重建

### 6.1 关键限制

trace 中以下字段是 **原始诊断量**，不是已经加权的 reward 贡献：

```text
reward_path_cross
reward_path_vertical
reward_path_heading
reward_path_progress
reward_speed_smooth
reward_energy
```

分析器必须从 run 的 environment config 读取权重，重建 path reward，并逐行验证
重建总和与 `reward_total` 一致。若不一致超过 `1e-8 * max(1, |reward_total|)`，
记 `INVALID_DIAGNOSTIC`，不得继续做占比结论。

重建公式：

```text
w3d = reward.w_path_3d                    if experiment.pc_control_mode == "alos"
      reward.w_3d                         otherwise

path_cross_vertical = -w3d * sqrt(reward_path_cross^2
                                  + reward_path_vertical^2) / 5.0
path_heading        = -reward.w_yaw * abs(reward_path_heading) / pi
path_progress       =  reward.w_prog * reward_path_progress
speed_smooth        = -reward.w_path_speed_smooth * reward_speed_smooth
energy              = -reward.w_path_energy * reward_energy

if experiment.pc_control_mode == "path_fixed_speed":
    path_speed = -reward.w_u * abs(command_speed - ground_surge) / 2.5
```

真实 PC reward 分量使用：

```text
reward_command_yaw, reward_command_pitch, reward_command_speed,
reward_hold, reward_roll_angle, reward_roll_rate, reward_action,
reward_saturation, reward_route_speed, reward_material_saturation,
reward_safety, reward_goal_plane_miss, reward_terminal_progress,
reward_terminal
```

### 6.2 归并

| 组 | 组成 |
|---|---|
| command_tracking | command_yaw + command_pitch + command_speed |
| path_tracking | path_cross_vertical + path_heading + path_speed（若适用） |
| progress | path_progress |
| terminal_progress | terminal_progress |
| terminal_outcome | terminal + goal_plane_miss |
| saturation | saturation + material_saturation |
| effort_regularization | action + speed_smooth + energy + roll_angle + roll_rate + route_speed + hold + safety |
| other | 未映射的 reward_* 字段；必须列出字段名 |

每个窗口先对每个 episode 求各组的带符号和，再求
`share_g = abs(sum_g) / sum_h abs(sum_h)`。`reward_total`、`reward_ppo_scaled` 和
`reward`（PPO scaled 日志）不进入分母。

### 6.3 outcome 激活

分别记录：

- `reward_terminal`、`reward_goal_plane_miss`、`reward_terminal_progress` 首次非零的
  行、时间和 signed offset；
- 首次非零是否在 done 行、是否在 crossing（`signed_offset >= 0`）之后；
- pre-terminal 行中的 `terminal + terminal_progress + goal_plane_miss` 占比；
- pre-terminal 行中的 `command_tracking + path_tracking + saturation + effort`
  占比。

## 7. 统计与 effect size

- 单位是 seed，不是 episode；同一 seed 的 4 个 checkpoint 必须一起重采样；
- lost vs retained：在 seed-cluster 内重采样 2000 次，计算两组中位数差；
- policy vs zero：先按 seed 对 4 个 policy checkpoint 求均值，再与同 seed 的 zero
  配对，重采样 seed 计算 paired median difference；
- 报告 `median_a`、`median_b`、`median_diff`、`relative_diff`、95% percentile CI；
- 连续通道支持条件：CI 不跨零，且 `relative_diff >= 0.10`，同时满足最小物理效应
  `heading/pitch >= 0.01 rad` 或 `speed >= 0.01 m/s`；
- 比例/饱和支持条件：CI 不跨零，且绝对差 `>= 0.10`；
- cosine 支持条件：CI 不跨零，且中位数差 `<= -0.10`；
- bootstrap seed 固定为 `20261110`，draws 固定为 `2000`；
- 不因 p 值、单次 episode 或个别 checkpoint 下结论；所有 effect size 和 CI 必须
  与 n 一起报告。

## 8. 可证伪 hypothesis

### H82-A：至少一个 action channel 区分 lost 与 retained

主指标：三通道 `residual_abs_mean`；辅助指标：`scale_clip_fraction`、
`rate_clip_fraction`、`command_tracking_error`、`direction_cosine_hp`。

- 支持：同一通道在至少两个 group（`baseline/candidate × default/strict125`）中满足
  §7 的效应和 CI 条件，且方向一致；至少一个 group 必须来自 candidate。
- 不支持：所有三通道和辅助指标都没有可复现差异。
- 最便宜 falsification：只读分析现有 800 episodes 的两个窗口，不新增 rollout。

### H82-B：residual 在应修正通道上饱和、rate-clipped 或方向不一致

- 支持：lost 相对 retained 的 `scale_clip_fraction`、`rate_clip_fraction` 或
  `material_saturation_fraction` 绝对差 `>= 0.10` 且 CI 不跨零；
  或 lost 的 `direction_cosine_hp` 中位数 `<= -0.10` 且 CI 上界 `< 0`；
  或 lost 的 `direction_cosine_hp` 相对 retained 的差 `<= -0.10` 且 CI 不跨零。
- 不支持：饱和/限幅和方向都没有稳定差异，或差异方向与“lost 更差”相反。
- 支持后动作参数化只是候选，下一 Goal 才能预注册单因素 action arm。

### H82-C：terminal reward 被 command tracking / saturation / energy 主导

- 支持：`pre2s` 的 preterminal 窗口中，
  `terminal + terminal_progress + goal_plane_miss` 的绝对占比 `< 0.10`；
  或 `reward_terminal` / `reward_goal_plane_miss` 首次非零只发生在 done 行或
  crossing 之后；
  或 lost 与 retained 的 reward group share 有 §7 的稳定差异。
- 不支持：terminal/progress 在 pre-terminal 窗口已有实质占比，且 outcome 项不是
  只在 crossing 后激活；或 lost/retained 的 reward profile 无差异。
- 支持后 reward 只是候选，下一 Goal 才能预注册单因素 reward arm。

### H82-D：policy 改善关键通道但没有改善 true outcome

- 关键通道：heading/pitch/speed tracking error、terminal LOS cosine、
  terminal horizontal/vertical offset；
- 支持：policy 相对 zero 至少一个关键通道达到 §7 的改善条件，且
  `policy_true_rate - zero_true_rate <= 0.0`；
- 不支持：policy 在关键通道也不优于 zero，或 true outcome 反而更好。
- 支持后先查 mapping、capture frame、terminal capture，不急着改 reward 或换算法。

### H82-E：action 与 reward 都没有区分度

- 支持：H82-A、H82-B、H82-C 均不支持，且 H82-D 不显示“通道改善但 outcome 不改善”。
- 此时失败更可能在估计器、控制器带宽或 pre-crossing approach；不得用“换 SAC”
  代替这个结论。

## 9. 判定与下一步

判定优先级固定为：

1. 完整性失败 -> `INVALID_DIAGNOSTIC`，只修解析链路，重跑同一审计；
2. H82-D 支持 -> `CHANNEL_GAIN_WITHOUT_OUTCOME`，转 mapping / terminal capture；
3. H82-B 支持 -> `ACTION_PARAMETERIZATION_CANDIDATE`，预注册单因素 action arm；
4. H82-C 支持 -> `REWARD_CANDIDATE`，预注册单因素 reward arm；
5. H82-A 支持但 B/C 不支持 -> `CHANNEL_DIFFERENCE_WITHOUT_MECHANISM`，
   先做 targeted channel diagnostic，不直接训练；
6. H82-A/B/C 均不支持 -> `ESTIMATOR_CONTROLLER_PRECROSSING`；
7. 只有明确证明“residual 方向正确但探索/样本效率不足” -> 才允许新开 SAC 对照。

若 H82-B 和 H82-C 同时支持，不得把 action 和 reward 混在一个 arm；按
“candidate 两个 scenario 都支持的那一个”优先，否则先做更便宜的单因素审计。

## 10. Gate（审计完整性，不是方法效果 Gate）

| check | 阈值 |
|---|---:|
| 20 个 run 的 schema/checkpoint SHA/action mode/seeds/radius/dwell | 全部通过 |
| baseline extension / candidate extension | `0.0` / `1.0` |
| 每个 episode 唯一 done 行且为最后一行 | 100% |
| trace / episodes 的 terminal kind 和 steps 一致 | 100% |
| `reward_total` 与重建 reward 的逐行误差 | `<= 1e-8 * max(1, abs(total))` |
| NaN/Inf | 0 |
| missing channel / reward 字段 | `INVALID_DIAGNOSTIC`，不得静默跳过 |
| 新 rollout / 训练 | 0 |
| held-out seeds | 0 |

## 11. 执行计划

- [x] Step 0：写本预注册并冻结；核对本地/远端路径、run 名称、schema 和配置权重；
- [x] Step 1：实现 `scripts/analyze_pc_goal82_action_reward_attribution.py`；
- [x] Step 2：写 `tests/test_pc_goal82_action_reward_attribution.py`，覆盖窗口、通道、
      reward 重建、bootstrap、缺失字段和 INVALID；
- [x] Step 3：在远端 20 个已记录 run 上并行只读解析（建议 `--jobs 20`，
      `OMP_NUM_THREADS=1`、`MKL_NUM_THREADS=1`、`OPENBLAS_NUM_THREADS=1`）；
- [x] Step 4：写 `doc/goals/goal82_result_20261110.md` 和
      `research/decisions/2026-11-10-goal82-action-reward-decision.md`；
- [x] Step 5：更新 `research/LEDGER.jsonl`、`research/STATE.md`，
      运行 ledger checker；
- [x] Step 6：运行定向 pytest、ruff、`git diff --check`；
- [ ] Step 7：用户明确要求后才 commit/push。

## 12. 算力与隔离

本 Goal 只读 20 个已记录 run，约数百 MB trace；不做新 rollout，不占 GPU。远端用
20 个进程并行读取 `trace.npz`，每进程单线程 BLAS；输出只写本 Goal 的
`artifacts/lowlevel/goal82_action_reward_attribution_20261110/`。

这不是样本效率实验；不得用 8×RTX 4090 的算力为理由扩大采样、batch 或重跑
Goal27 失败臂。训练型 arm 的前置条件仍是：

```text
num_envs 128/256/512 × batch_size 256/512
单卡 throughput 基准 -> 按 seed 分 8 卡 -> 独立预注册
```

## 13. Git

禁止 `git add -A`。只 add 本 Goal 相关文件，提交前运行：

```bash
python ../scripts/check_research_ledger.py \
  --goal Goal82 \
  --goal-doc doc/goals/goal82_action_reward_attribution_20261110.md \
  --strict --require-result
git diff --cached --check
```

不提交 `artifacts/`、checkpoint、日志、trace 或缓存。未经用户明确要求，不 commit、
不 push。

## 14. Goal 提示词

```text
你现在执行 Goal82：action parameterization 与 reward 的只读归因审计。

先读：
- doc/goals/goal82_action_reward_attribution_20261110.md
- doc/goals/goal81_result_20261110.md
- doc/goals/goal80_result_20261110.md
- doc/goals/goal27_result_20261109.md
- research/STATE.md
- AGENTS.md

硬约束：
1. 不训练、不新增 rollout、不改 reward/controller/checkpoint/seed/held-out；
2. 只读 Goal71 baseline、Goal80 candidate、Goal60 reference 的 trace/episodes；
3. 不启动 SAC/REDQ/DroQ，不换算法；
4. 先完成并冻结本预注册，再实现 analyzer；
5. 缺失 channel/reward 字段记 INVALID_DIAGNOSTIC，不静默跳过；
6. 结果、ledger、STATE 全部中文；只 add 本次相关文件；
7. 不自动 commit/push。

执行顺序：
- Step 0：核对 20 个 run 的 schema、SHA、seeds、radius、dwell、配置权重；
- Step 1：实现 action/reward 归因分析器；
- Step 2：合成测试覆盖窗口、通道、reward 重建、bootstrap、缺失字段；
- Step 3：远端 20 进程并行只读解析；
- Step 4：按 H82-A/B/C/D/E 和 §9 优先级判定；
- Step 5：写 result、decision、LEDGER、STATE，运行 checker。
```

## 15. 结果摘要

- 正式状态：`CHANNEL_GAIN_WITHOUT_OUTCOME`；
- H82-A：支持，但只来自 `tracking_error_heading` / `tracking_error_pitch`，
  residual 三通道幅度不区分 lost/retained；
- H82-B：不支持；scale/rate clip 全为 0，material saturation 和 direction
  cosine 没有稳定方向；
- H82-C：支持；terminal outcome 在 100% episode 中只在 done 行激活，
  terminal_progress 在 preterminal 仍然活跃且 lost 更低；
- H82-D：支持但主要是 strict125-specific；policy 的 heading tracking 改善，
  true success 反而更低；
- H82-E：不支持；
- trace 对齐修正：residual 的基座应为 `base_command[t]`，不是
  `command_physical[t]`；修正后主结论未翻转；
- 下一步：mapping / capture frame / terminal capture 审读，不启动 reward arm、
  SAC/REDQ/DroQ。
