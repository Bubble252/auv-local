# Goal153：奖励 + 课程学习 + 多环境测评联合 screen（C1 主线）

**日期**：2026-11-11<br>
**状态**：`TRAIN_SCREEN_RUNNING / 200K_SCRATCH_NO_REWARD_EFFECT / BC_INIT_1M_RUNNING / HELDOUT_CLOSED`<br>
**上游**：Goal136 课程计划、Goal140 纵向 reward 方案、Goal144 C1 瓶颈、Goal151/152 residual authority 证据<br>
**优先级**：`P1`。用户明确要求：奖励与课程学习优先，且任何训练范式都必须经过
“多方案训练 + 多环境测评”；本 Goal 是这条 standing rule 的第一个完整实例。
**Eval suite**：`configs/lowlevel/eval/pc_standing_eval_suite_v1.json`（从 Phase 3 起强制执行）。

**速度口径（2026-11-12 用户确认）**：REMUS100 最大期望地速为 `2.5 m/s`，任何
课程采样或评测档位都不得出现 `>2.5 m/s`。本 Goal 已冻结的 C1 训练仍为 `1.2 m/s`，
不得在本 Gate 中途改写；高速课程只能在新预注册 Goal 中启用。`2.5 m/s` 端点只允许与
零/弱海流及零/可忽略速度波动同时采样，否则必须标成 OOD 压力档并报告不可达、饱和和
timeout。

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
  check_command: .venv/bin/python ../scripts/check_research_ledger.py --goal Goal153 --goal-doc doc/goals/goal153_reward_curriculum_multienv_20261111.md --strict --require-result
```

## 1. 背景

Goal144 已证明课程 Gate 严格生效，但 C1 是瓶颈：

| arm | C0 | C1 | 判读 |
|---|---:|---:|---|
| PPO command/actuator | `0.925` -> promote | `0.65` -> rollback | 残差没有闭环效果 |
| SAC command | `0.925` -> promote | `0.15–0.225` -> rollback | command residual 主动破坏 |
| SAC actuator | `0.925` -> promote | `0.625` -> rollback | actuator 有局部作用，但 C1 不达标 |

C1 失败是 `goal_plane_miss`，不是 timeout/safety；失败集中在末段 3D capture。
Goal140 的 `F_z / F_zdot` 方案此前因“纵向未主导”没有启动；现在按用户要求，把它
作为 reward 维度正式纳入联合 screen，而不是继续等纵向单独主导。

Goal151/152 又证明 residual 权限位置会显著改变 path 代价。因此本 Goal 固定
**actuator residual + full authority**，只研究 reward 与 curriculum；权限 schedule
放到 Goal154，避免两个变量混在一起。

## 2. 假设

### H153-A：R1/R2 能在不伤 path 的前提下改善 C1 capture

成立应看到：

- R1 或 R2 的 C1 default success 比 R0 至少 `+4/40`，或 `eta_done_distance` 改善
  `>=0.05 m`；
- path 3D RMSE 不比 R0 差 `+0.05 m` 以上；
- vertical RMSE 改善 `>=10%`；
- strict125 success 不下降超过 `2/40`。

失败应看到：success 不动、path 变差，或 vertical RMSE 不动。此时不能继续扫权重；
写 blocker，转动作参数化/terminal capture 审计。

### H153-B：C-full 比 C-fixed 更适合 C1

成立应看到：同 reward、同预算下，C-full 的 C1 success 比 C-fixed 至少 `+2/40`，
plane miss 更低，且 C0 不退化超过 `2/40`。

失败应看到：C-full 仍停在 C0 回滚，或 C1 与 C-fixed 持平。此时判
`CURRICULUM_ORDER_NO_EFFECT`，不把课程顺序写成收益。

### H153-C：reward 收益不是 residual 幅度膨胀带来的

成立应看到：

- residual L2 ratio 不比 R0 高 `50%` 以上；
- action saturation / residual clipping 不高于 R0；
- success 提升伴随 terminal vertical/horizontal error 下降，而不是单纯更大动作。

失败应看到：residual 幅度显著变大、饱和升高，success 只靠更激进动作。此时判
`REWARD_HACKING`，停止该 reward arm。

### H153-D：多环境成立，而不是 nominal 过拟合

成立应看到：

- 至少 3/4 个压力档（current10、noise N2、plant20、combined20）相对 zero-residual
  success 不为负；
- current15 不出现从 nominal 到 `<=0.25` 的系统性崩溃；
- timeout/safety 不高于 zero。

失败应看到：nominal 改善但 current/plant 崩溃。此时判 `ENV_OVERFIT`，不 promotion。

### H153-E：如果 R1/R2 + C-full 仍不能救 C1，则瓶颈不是 reward

成立应看到：所有 reward/curriculum 组合的 C1 都停在 `0.60–0.70`，且 residual
方向/幅度没有随 reward 改变。此时按预注册停止 reward 方向，转 Goal154 的 residual
authority schedule 与 terminal capture 审计；不换算法、不扫 seed。

## 3. 训练矩阵

### 3.1 Phase 0：只读审计（不训练）

复用 Goal144/151/152 traces，检查：

- C1 失败与成功在 2 m entry 前的 vertical/horizontal error 分离；
- residual 幅度按 `<=2 m / 2–10 m / >10 m` 分箱；
- reward 分量在 terminal 前 32/64 步的激活；
- action saturation 与 plane miss 的关系。

输出：`artifacts/lowlevel/goal153_reward_curriculum_20261111/phase0.md`。

### 3.2 范式矩阵（必须全部登记）

“范式”是本 Goal 的独立维度，不是只在 SAC 上做完就算。每个范式至少跑
`R0 × C1-fixed`；有正向信号的范式再扩到 R1/R2 与 C-full。

| paradigm | base | residual | 算法 | 最低 screen |
|---|---|---|---|---|
| `T0`（ALOS-COMP） | 纯 ALOS + estimated-current `compensate_current` | 无 | 传统控制器 | nominal + current10 + plant20 |
| `T0-NONE` | 纯 ALOS，无 crab、无 compensate | 无 | 传统控制器 | 同 `T0`，作为洋流补偿消融 |
| `BC-PPO` | frozen BC | actuator | PPO | R0 C1-fixed |
| `BC-SAC` | frozen BC | actuator | SAC | R0 C1-fixed |
| `ALOS-PPO` | 纯 ALOS | actuator | PPO | R0 C1-fixed |
| `ALOS-SAC` | 纯 ALOS | actuator | SAC | R0 C1-fixed |
| `future-REDQ/DroQ` | 待定 | actuator | off-policy | 先另开预注册，不在本 Goal 偷跑 |

硬约束：所有范式共用同一 success/terminal/radius、dev seeds、环境档位和指标表；
不能因为某个范式难训就少报环境或换 split。

**2026-11-12 用户追加裁决**：ALOS 洋流补偿只保留两组，`ALOS-NONE`（crab 与
compensate 都关）和 `ALOS-COMP`（只开 estimated-current `compensate_current`）。
`ALOS-COMP` 是默认主基线，`ALOS-NONE` 是对照；`current_crab_enabled=true` 只保留
历史复现，不进入新训练/评价矩阵。详见
`doc/goals/goal174_alos_current_compensation_paradigm_20261112.md`。

### 3.3 Phase 1：reward pilot（1 seed × 200k）

| arm | reward | curriculum | authority | 预算 |
|---|---|---|---:|---:|
| P-R0 | R0 baseline | C1 fixed | actuator full | 200k |
| P-R1 | R1 `F_z` | C1 fixed | actuator full | 200k |
| P-R2 | R2 `F_z + F_zdot` | C1 fixed | actuator full | 200k |

只检查 reward finite、动作不饱和、vertical RMSE 方向、C1 success 方向；pilot 不作为
正式 Gate。

### 3.4 Phase 2：1M screen

| arm | paradigm | reward | curriculum | authority | seed | 预算 |
|---|---|---|---|---|---:|---:|
| S-T0 | T0 | R0 | C1 fixed | zero | 86201 | 1M |
| S-BC-PPO | BC-PPO | R0 | C1 fixed | actuator full | 86201 | 1M |
| S-BC-SAC | BC-SAC | R0 | C1 fixed | actuator full | 86201 | 1M |
| S-ALOS-PPO | ALOS-PPO | R0 | C1 fixed | actuator full | 86201 | 1M |
| S-ALOS-SAC | ALOS-SAC | R0 | C1 fixed | actuator full | 86201 | 1M |
| S-R1 | ALOS-SAC | R1 | C1 fixed | actuator full | 86201 | 1M |
| S-R2 | ALOS-SAC | R2 | C1 fixed | actuator full | 86201 | 1M |
| S-R1C | ALOS-SAC | R1 | C-full | actuator full | 86201 | 1M |
| S-R1-full | ALOS-SAC | R1 | C1 fixed | actuator full（不 gate） | 86201 | 1M |
| S-R1-mid | ALOS-SAC | R1 | C1 fixed | only 2–10 m | 86201 | 1M |

S-T0 是传统控制器基线；S-ALOS-SAC 是 residual-only 对照。S-R1/S-R2/S-R1C 只在
Phase 1 显示 reward 方向为正时启动；PPO/BC 行用于验证“不是某个范式特例”。
S-R1-full/S-R1-mid 专门检验“纵向 reward 能否把近段优势扩展到中/远段”：
如果 R1-full 能追平 near gate，说明 reward 可以抵消远段 residual 的伤害；
如果只有 near/mid 有效，则权限位置仍是主约束。

### 3.5 Phase 3：多环境测评

对 Phase 2 每个范式至少保留 1 个 survivor（最多 2 个/范式）跑：

| 环境 | 设置 | 目的 |
|---|---|---|
| nominal | 当前默认 | 主结果 |
| current05 | `0.25–0.45 m/s` + shear `0.5` | 中等洋流 |
| current10 | `0.30–0.55 m/s` + shear `1.0` | 强洋流 |
| current15 | `0.375–0.675 m/s` + shear `1.5` | 饱和边界 |
| noise N2 | `0.50–1.00×` 零均值噪声 | 感知压力 |
| plant20 | ±20% plant mismatch | 动力学误差 |
| combined20 | current10 + noise N2 + plant20 | 联合压力；旧 Goal134 current20 版本标 `legacy`，不静默合并 |

以上 current 档位以 `configs/lowlevel/eval/pc_standing_eval_suite_v1.json` 为唯一 canonical 定义；旧文中的绝对区间不再单独解释。

每个 arm × 环境 × `default/strict125`，dev40；policy/zero 配对。

Phase 3 起不再只报 success。每个 survivor 还必须生成统一 `grid_summary.md`，包含：
路线健康度、终点附近分箱、方向性失败、policy-head/effective/actuator 饱和、
residual L1/L2 与 gate-active 比例。该要求不改变本 Goal 已冻结的 Gate 阈值，只增加
风险指标和失败归因。

### 3.6 Phase 4：5M formal

只有 Phase 2/3 都通过的 arm 才进入：3 seeds × 5M × 多环境。formal 另开预注册。

### 3.7 统一测评契约（固化）

完整规则见 `research/EVALUATION_PROTOCOL.md`，本 Goal 至少要覆盖：

| 轴 | 档位 | 备注 |
|---|---|---|
| 压力 | nominal、current05/07/10/15、noise N2、plant20、combined20 | 见 §3.5；current ladder 采用 Goal149 |
| 路线 | `legacy_sine / P0 / P1 / P2 / P3 / P4` | 不再只用一条 sinusoidal bank |
| 长度 | `30 / 35 / 45 m` | 长路径与 timeout 必须同时报 |
| 速度 | `0.9 / 1.2 / 1.4 / 1.8 / 2.2 / 2.5 m/s` | REMUS100 上界为 `2.5`；训练侧 `CurriculumStageSpec.speed_range_mps` 同样硬校验 `<=2.5`；速度收益不能牺牲 path/饱和 |
| 课程 | `C-fixed / C-full / C-hard` | C-full 与 C-fixed 同预算 |
| 范式 | `T0 / BC / BC-PPO / BC-SAC / ALOS-PPO / ALOS-SAC`；每条 ALOS-based arm 登记 `ALOS-COMP` 或 `ALOS-NONE` | 默认 ALOS-COMP；ALOS-NONE 为对照；crab 只做 legacy；REDQ/DroQ 另开预注册 |
| 残差权限 | zero、near、mid、far、near10_2、full、scale 0.5/0.75 | 至少覆盖 mid/far/full |

测评分层：

| tier | 覆盖 | 用途 |
|---|---|---|
| `A_screen` | nominal + current10 + noise + plant | 快速筛选 |
| `B_robustness` | 7 pressure + P2/P3 + 35/45 m + 1.2/1.4/2.5 m/s | 压力与泛化；2.5 必须保留，不可因不可达删除 |
| `C_full_geometry` | 6 routes + 3 lengths + 6 speeds（0.9/1.2/1.4/1.8/2.2/2.5 m/s） | 几何审计；含 2.5 上界 |
| `D_formal` | 全部轴，3 training seeds | 正式结论 |

关键口径：

- 主 success 仍是 `eta_tight_at_done`；`p_nav/eta` 位置差暂不作性能轴。
- standing eval 的 C0/C1 必须关闭 direct-command 训练脚手架；三类 evaluator 都由
  runner 强制传 `--disable-curriculum-direct-command`，否则 zero-residual 会退化成
  随机 direct command，而不是沿路线跟踪。
- `path3d` 的 episode-mean 与 pooled active-step RMSE 必须分开写。
- 速度档是期望地速，训练/课程采样必须满足
  `v_cmd + v_current + v_variation <= 2.5 m/s`；`command_speed_range_mps` 已在
  parse 层校验，逐 episode speed 已在 sampler 层按最坏情况
  `v_current + v_variation` 收缩电流项。`2.5` 端点不得与强流独立
  采样，只能用于零/弱流或标注清楚的 OOD 压力档。
- `policy-head saturation` 使用全局 residual multiplier 前的原始策略输出；
  `effective residual saturation` 使用 multiplier 后动作；两者不能混用。
- 旧 trace 没有 `policy_action_raw` 时，策略头饱和显示 `NA`，不能把缺失当 0%。
- 结果必须报告 terminal kind、终点分箱、2 m entry geometry、signed vertical/N/E bias。

## 4. 数据与隔离

| 项目 | 值 |
|---|---|
| base controller | 纯 ALOS；K4 只作为 C1 nominal 的上界参考，不混入主臂 |
| residual | actuator，scale `0.25`，full authority |
| 训练 seeds | screen 86201；formal 86201–86203 |
| dev seeds | `20–59`，每 run 40 episodes |
| success | `eta_tight_at_done`，default 2.0 m / strict 1.25 m |
| held-out | `100–149` 关闭 |
| reward / terminal / radius | R0/R1/R2 只改 shaping；terminal/success/radius 不变 |
| 算法 | SAC screen；PPO 只在 Phase 2 后按需复制一个同协议 arm |

禁止：在同一 arm 中途改 reward、curriculum 或 authority；用 dev 选 checkpoint 后报
formal；为单个 arm 增加预算；打开 held-out。

## 5. Gate

Phase 2 screen 的人工 Gate（1 seed，只作 screen）：

| check | 阈值 |
|---|---|
| C1 success | 至少一个 reward arm 比 S-R0 `+4/40` |
| path | 不比 S-R0 差 `+0.05 m` |
| vertical | RMSE 改善 `>=10%` 或 terminal vertical error 下降 |
| residual | 不比 S-R0 膨胀 `50%` 以上 |
| C-full | 若启用，C1 比 C-fixed 至少 `+2/40` |
| safety | timeout/safety 不高于 zero |

裁决：

| 观察 | 判读 | 下一步 |
|---|---|---|
| reward arm 过 success/path/vertical | `SCREEN_PASS_TO_MULTIENV` | 进入 Phase 3 |
| reward 不动 | `REWARD_NO_EFFECT` | 转 Goal154/terminal audit，不扫权重 |
| success 升但 path/饱和升 | `REWARD_HACKING` | 停该 arm |
| C-full 无增益 | `CURRICULUM_ORDER_NO_EFFECT` | 保留 C-fixed，不写课程收益 |
| nominal 升、压力崩 | `ENV_OVERFIT` | 不 promotion |

## 6. 执行计划

- [x] Step 0：只读 Phase 0 trace 审计；
- [x] Step 1：实现 R1/R2 reward 配置与 unit test，默认权重 0 时与 R0 parity；
- [x] Step 2a：远端 200k from-scratch reward pilot（已记录为样本效率证据）；
- [~] Step 2b：BC-init 1M 主实验 + ALOS-only 课程对照（运行中）；
- [~] Step 3：1M reward × authority 扩展 screen（至少 mid/far/full），统一用新分析器；
- [ ] Step 4：多环境 Phase 3（A/B tier），不只看 success；
- [ ] Step 4b：几何 Phase 3b（P0–P4、30/35/45 m、0.9/1.2/1.4/1.8/2.2/2.5 m/s；2.5 为硬上界）；
- [ ] Step 5：写 result、blocker/decision、LEDGER、STATE；
- [ ] Step 6：通过 checker 后才考虑 5M formal。

## 7. 纵向奖励说明

R1/R2 直接引用 Goal140 §4.2：

```text
F_z     = -w_z     * Huber(vertical_error)
F_zdot  = -w_zdot  * Huber(heave_rate_error)
F_predz = -w_predz * Huber(vertical_error + tau_z * heave_rate_error)
R1      = baseline + F_z
R2      = R1 + F_zdot
```

三组共同保留 C1 原有的 `w_approach_vertical=1.0`。R0/R1/R2 的差异只有
`F_z`/`F_zdot` 是否加入，避免把“删掉旧纵向奖励”误当成“新增 potential 的效果”。

先只跑 R1/R2；`F_predz` 只在 R2 显示方向后作为第三档，不在同一 Gate 内扫权重。

## 8. 2026-11-11 执行记录

### 8.1 200k from-scratch pilot（已完成的样本效率证据）

首轮 200k pilot 使用 `bc_checkpoint=null`，结果为 dev40：

| arm | default success | strict success | path3d default |
|---|---:|---:|---:|
| zero-residual | 26/40 | 14/40 | 2.3375 |
| R0 | 27/40 | 15/40 | 2.3198 |
| R1 | 25/40 | 14/40 | 2.3539 |
| R2 | 25/40 | 14/40 | 2.3521 |

residual RMS 只有约 `0.06–0.11` 归一化，说明 200k from scratch 基本没有学到
有效 residual；R1/R2 相对 zero 的 success 变化为 `-1/40`，不能作为 reward 结论。
同一批 checkpoint 在课程内 C1 评估为 `0/40`、path 约 `9.5 m`，说明 C1 课程分布
（depth start 20 m）显著难于 nominal Goal85 default（depth start 10 m），
不能用 nominal 结果替代课程 Gate。

### 8.2 BC 初始化修正

Goal144 的 1M SAC 主协议使用 `checkpoints/goal26/d3/pc_bc_actor.pt` 做 BC 初始化；
首轮 pilot 把它置空，等于同时改变了初始化与预算。现已恢复：

- R0/R1/R2 三组都保留 BC checkpoint；
- 三组都保留 `w_approach_vertical=1.0`，只让 R1/R2 额外加 `F_z`/`F_zdot`；
- 1M BC-init 主实验正在跑：`pilot_bc_1m`；
- 另起 ALOS-only 课程对照：`pilot_bc_1m_alos`，训练与评估都加
  `--curriculum-disable-direct-command`，用于排除 C0/C1 direct-command 混杂。

### 8.3 authority 结论对 reward 的约束

Goal154 Phase 1b 已确认：terminal-only residual（<=2 m）与 zero 等价，有效区段是
2–10 m shell。因此 R1/R2 的 authority 扩展实验必须至少包含
`mid_only`、`far_off_only`、`full` 三档；若 R1/R2 只在 `mid_only` 有纵向收益，
就判 `SHELL_LOCAL_REWARD`，不能写成“reward 把优势扩展到全路段”。

### 8.4 多指标分析器修复（2026-11-11）

旧 `analyze_pc_goal153_eval_grid.py` 有三个会误导判断的问题，已修复：

- `terminal_kind` 是数字编码，旧表把 `GOAL/PLANE_MISS` 全统计成 0；现优先读
  `summary.json.summary.terminal_kind_counts`，回退时才显示 `KIND_n`；
- `Route health` 的 pooled active-step RMSE 与 episode-mean path3d 不是同一口径；现明确分栏，
  不再混用；
- 旧 `action sat` 是乘以 global residual multiplier 后的有效动作，`scale 0.5/0.75`
  自然显示 0%；现拆分“策略头 pre-scale”和“有效残差 post-scale”，并新增
  actuator/material sat、scale/rate clip。

分析器现自动发现任意 `<tag>/<scenario>` 或 `<tag>/<seed_tag>/<scenario>`，
并把 env SHA、goal radius、route family、length、speed、current、noise、plant、
curriculum stage 写进 scenario axes 表。
