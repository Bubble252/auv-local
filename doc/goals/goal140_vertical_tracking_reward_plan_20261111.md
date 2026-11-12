# Goal140：全程纵向跟踪的奖励与约束设计

**日期**：2026-11-11<br>
**状态**：`STANDBY / K4_CONTROLLER_BOTTLENECK_CONFIRMED / REWARD_PILOT_NOT_TRIGGERED / NO_WEIGHT_SWEEP / HELDOUT_CLOSED`<br>
**上游**：Goal139 `mixed-horizon/credit-horizon`、Goal138 `terminal curriculum`、Goal137 `latent probe`、Goal118 `reachability`、Goal117 `terminal geometry/credit`、Goal34/35 `early vertical diagnostics`<br>
**优先级**：`P2`。当前 `P1` 是 Goal136 C-line 的 C0/C1 控制语义修复
（Goal143：同一 checkpoint 在 direct command 下 `0/20`，ALOS 下 `18/20`）。
Goal140 必须等 Goal143 的 C0 Gate 与后续 C-line screen 裁决后再启动；本文件当前
只是排队，不启动训练。
**性质**：奖励审计 + SAC reward ablation；先只读排序指标，再训练少量 reward/constraint arm。不改变 success、terminal、goal radius、sensor、评估起点或 held-out。

## 0. 候选触发记录（2026-11-11）

用户要求：如果后续实验发现效果不理想主要来自**纵向误差**，优先回看本文件的奖励修缮
方案并重跑实验；当前只记录思想，不在已预注册的 Goal144 / Goal136 C-line Gate 中途
改 reward。

**Goal145 后更新（2026-11-11）**：C1 已确认是 near-goal 纵向末端捕获瓶颈，但失败
episode 同时出现 pitch/speed guidance rate clipping 显著升高；现有 residual 有数值但
没有闭环效果。因此本 Goal 仍不启动权重扫描。必须先由 Goal146 的 eval-only 单变量
证明终端 pitch/speed 权限是否足以改变 success；只有当“权限足够、policy/reward 没有
利用它”时才进入本文件的 potential-based `R1/R2` pilot。若 Goal146 表明权限本身不足，
继续修 terminal guidance，不把 reward 当替代品。

**Goal147 后更新（2026-11-11）**：`K4`（`base_controller.pitch_gain 2 -> 4`）在固定
`C1_000300000`、dev40 上把 success 从 `26/40` 提到 `40/40`，7 个近终点失败全部转化，
400k guard 也保持。K4 的 policy 与 zero-command-residual 完全相同，说明瓶颈是传统
pitch 跟踪增益，不是 reward，也不是 residual。因此本 Goal **仍不启动**权重扫描。

**Goal150 后更新（2026-11-11）**：current07 K4 残差 screen 失败，但**不是纵向误差
主导**。PPO 的 vertical RMSE 比 zero 低 `0.0028 m`，success 反而从 `37/40` 掉到
`35/40`；SAC 是 31/40 `GOAL_PLANE_MISS`，横向 RMSE 恶化约 `0.16 m`，heading 残差
均值 `-0.092 rad`、pitch `+0.044 rad`。因此本 Goal **不启动** `F_z` / `F_zdot` /
`F_predz` reward pilot，也不扫权重。下一步先做 residual authority、channel mask、
reward credit 与 terminal capture 的只读审计；只有新的预注册明确证明“纵向主导、
残差方向正确但幅度不降”时，才允许重跑本文件的 reward 方案。

**Goal155 后更新（2026-11-12）**：Goal155 的 terminal-near 复核已满足本文件的
“纵向主导”触发条件：失败几乎全是 `PLANE_MISS`，default 失败纵向 `1.83–1.98 m`，
进入 2 m 球时纵向 p90 `1.76–1.87 m`；`near10_2` 只收窄 final-distance 尾部、
不降纵向，`mid_only` 关掉 `<=2 m` residual 反而降低 success。当前仍**不启动**
reward pilot：先等 Goal156 的 5M full-training Gate 记录；若 Goal156 失败，先做
terminal vertical credit / 动作参数化的最便宜证伪。原因是 Goal155 的 2–10 m
residual 幅度并不小（`mean_abs` 约 `0.34–0.45`），但它与纵向误差/heave 的相关性
跨 seed 符号不一致，不能直接支持“方向正确、幅度不够”的 R1/R2 重跑。只有在新的
只读/短 pilot 证明信用或动作参数化不是主因后，才开新的预注册 Goal140-R1/R2
pilot（1 seed、200k，先 `F_z`，再加 `F_zdot`），并与同预算 control 配对，
不在 Goal156 内改 reward。

用户要求保留一个常驻提醒：后续如果出现“残差方向/符号看起来正确，但纵向误差持续
不降”的情况，先回看本文件的 `F_z` / `F_zdot` / `F_predz` 方案，重新预注册一个
reward pilot 并重跑实验；不要在同一 Gate 内临时改 reward，也不要先扫权重。

常驻触发条件（满足任一条即写新预注册，不直接改当前 arm）：

- K4 基座上的 policy 与 zero 仍有稳定纵向分离，且失败集中在中远段
  `vertical_error` / `heave_rate_error`，不是 terminal 权限；
- residual 符号与纵向误差方向一致，但幅度长期接近 0，提示探索或 credit 不足；
- current/plant stress 下传统 K4 已饱和，而 residual 有可测的方向性，但闭环收益
  仍被 reward 的 3D 混合项掩盖。

不触发的情况：terminal pitch 仍跟不上 target、stern 已饱和、timeout/safety 主导，
或失败主要是横向/路径误差。此时继续修 action parameterization / controller / terminal
capture，不把 reward 当替代品。

触发条件（必须同时满足）：

这些条件现在由 `goal145_terminal_capture_path_vertical_audit_20261111.md` 的
`VERTICAL_DOMINANT_CONFIRMED` Gate 统一检验；Goal145 未确认前，本文件只保留方案，
不启动 reward arm。

- 失败集中在 2 m / goal-plane entry 的 `vertical_error`、`heave_rate_error` 或
  `predicted_vertical_error`，而不是 timeout、safety、横向误差或纯探索；
- 现有 trace 显示成功与失败在 entry 前已有稳定纵向分离；
- 当前 Gate 已记录，允许开新 reward arm，而不是救当前 arm。

启用后的首选方案（来自本文件 §4.2）：

- `R1 = baseline + F_z`，`F_z = -w_z * Huber(vertical_error)`；
- `R2 = baseline + F_z + F_zdot`，加入 heave-rate potential；
- `F_predz = -w_predz * Huber(vertical_error + tau_z * heave_rate_error)` 作为
  predicted-error 版本；
- potential-based shaping，不直接把绝对纵向误差当 raw dense reward；
- 先 1 seed、200k transitions 的 `R1/R2` pilot，检查纵向 RMSE/p95、strict success、
  timeout 和动作饱和；通过后再按本文件预注册扩 seed。

硬约束：不在同一 Gate 内同时改 reward、动作接口和算法；不因 pilot 不好就扫权重；
不打开 held-out；PPO 不混入本 Goal 的 SAC reward  ablation。

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
  check_command: .venv/bin/python ../scripts/check_research_ledger.py --goal Goal140 --goal-doc doc/goals/goal140_vertical_tracking_reward_plan_20261111.md --strict --require-result
```

`required: true`：本 Goal 会训练新的 SAC reward arm。Step 0 冻结基线 checkpoint、
reward arm、seed、预算、评估 bins 和 Gate 阈值；Gate 记录前不得改动。若需要外部
reward-shaping / constrained-RL 方法，先写 reading note，再实现最小可迁移组件。

## 1. 背景与问题

当前目标主要奖励终端 success。`pc_command_track_v2` 的 pitch/heading 指令跟踪项
比 `w_path_3d` 更直接，而 `w_path_3d` 又把横向和纵向误差混在一个 3D 范数里。
这会带来两个问题：

1. 路径中段可以“沿路走、但不贴深度参考”，只要最后有机会进入捕获球就不被强惩罚；
2. 纵向误差、heave rate、pitch 误差和横向误差的梯度混在一起，无法判断哪个 reward
   项真正改善了控制，哪个只是改变了轨迹风格。

Goal137 已证明 `z_t` 保留纵向信息，BC head 也响应，不需要再回到 encoder/freeze
问题。Goal117/118/131 则表明失败在 2 m 入口前已经形成，且可达/不可达失败并存。
Goal119/132/133 又证明单纯加 reward scale、terminal shaping、fixed bias 或
residual gain 不会自动提高 success。

因此本 Goal 不再问“再调大哪个权重”，而是问：

> 哪些可观测、可控制、领先于失败的指标，适合做全程纵向跟踪的 potential 或
> constraint，并且不会把终端 success 和横向控制带坏？

## 2. 研究问题与假设

### H140-A：指标分层后，只有少数指标值得进入 reward

对现有 traces 计算候选指标的 success 判别力、提前量、可控性和与 terminal
geometry 的相关性。预期 `vertical_error`、`heave_rate_error` 和
`entry feasibility margin` 比 `terminal closure`、`base saturation`、`residual
magnitude` 更适合作为训练信号。

成立应看到：

- 至少 2 个候选指标在控制 2 m entry geometry 后仍与 failure 有稳定关联；
- 这些指标在失败前至少一个时间窗开始偏离 success；
- 候选指标不是同一个量的线性复制。

失败应看到：所有候选指标都被 2 m entry geometry 或 seed 方差解释。此时不进入 reward
训练，先回到数据/几何审计。

最便宜证伪：只读 Goal117/118/131/133/138/139 traces，计算分箱 AUC、Cliff's delta、
提前量和简单偏相关；不训练。

### H140-B：potential-based 纵向项能改善全程 tracking 而不伤 success

加入 `vertical_error` 和 `heave_rate_error` 的 potential-based shaping，而不是
直接把绝对误差当 raw reward；预期各路径段的纵向 RMSE/p95 下降，terminal success
不退化。

成立应看到：

- 至少 3/5 个路径 bin 的 `vertical_rmse` 改善 `>=10%`；
- 整体 `vertical_rmse` 和 `vertical_p95` 各改善 `>=10%`；
- strict/default success 不低于 paired control -1/40；
- 2/3 training seeds 方向一致。

失败应看到：纵向 RMSE 不动，或 RMSE 改善但 strict success 掉超过 2/40。后者判
`TRACKING_SUCCESS_TRADEOFF`，不能当部署收益。

最便宜证伪：1 seed、200k transitions 的 `R1/R2` pilot，只检查 reward 数值有限、
动作未饱和、纵向 RMSE 方向和 timeout；pilot 不作为正式 Gate。

### H140-C：2 m entry feasibility 比 terminal closure 更适合做 gate/constraint

把“剩余时间是否足够修正纵向/pitch”归一化成 `entry_margin`，作为辅助 critic 或
Lagrangian cost，而不是直接奖励闭合速度。预期它能减少 2 m 时仍 geometry-bad 的
比例，并避免 Goal132 中“提高 closure 反而伤边界样本”的问题。

成立应看到：

- 2 m entry 的 `vertical_error`、`pitch_error` 中位数下降；
- 5/11 可达 failure 子集至少有一半转化为 success 或在 1.25 m 壳层内；
- timeout/safety 不超过 control +2/120。

失败应看到：entry margin 改善但 success 不动，或 terminal closure 被推高而 success
下降；此时停止 closure/entry 约束，转回纵向位置与 heave rate 的基本跟踪。

最便宜证伪：先用只读 traces 重放 entry-margin 与 2 m geometry 的关系；再跑 1 seed
的 entry-cost pilot。

### H140-D：奖励项不是越多越好

同时加入 vertical、heave、pitch、cross-track、closure、entry gate 会产生尺度冲突和
reward hacking。预期 2–3 个核心项加 1 个约束，优于 5 个以上 dense 项。

成立应看到：`R1/R2/R4` 中至少一个 arm 同时满足 tracking 和 success guards；
继续叠加 pitch/cross/closure 没有额外 tracking 改善。

失败应看到：carrier arm 越多，success 或 path RMSE 越差。此时按预注册规则去掉
低贡献项，而不是继续扫权重。

## 3. 数据、基线与隔离

| 项目 | 规定 |
|---|---|
| 只读审计数据 | Goal117/118/131/133/138/139 已记录 traces 和 summary |
| 训练 seeds | `86201–86203`，除非 Step 0 冻结为 Goal139 指定的 3-seed winner |
| 训练预算 | 每 arm `1,000,192` transitions（SAC） |
| 评估 scenarios | `strict/default` |
| 评估 levels | `nominal/current15/current20/plant20/combined20` |
| dev seeds | `20–59`，每 run 40 episodes |
| held-out | `100–149` 关闭 |
| 评估起点 | 强制完整 35 m，`reference.start_progress_m=0` |
| success 契约 | Goal85 `eta_tight_at_done`，不改 success/terminal/goal radius |
| 基线选择 | Step 0 固定；优先 Goal139 预注册规则选出的 3-seed SAC 基线；若 Goal139 未完成，本 Goal 不启动训练 |

禁止：

- 用 held-out 选 reward 或 Gate；
- 用单 seed 86203 作为训练基线或正式结论；
- 为 reward arm 增加 transitions 或 seed；
- 在同一 Gate 内同时更换 SAC 主算法、动作接口和 reward。

## 4. 候选指标与奖励设计

### 4.1 指标分层

#### A. 直接预测 success 的几何指标

- `2m_entry_vertical_error`
- `2m_entry_pitch_error`
- `entry_feasibility_margin`
- `terminal_closure`

这些指标适合做 gate、辅助 critic 或 terminal 约束，不适合全部做 dense reward。

#### B. 可在线控制、领先于失败的中间指标

- `vertical_error`
- `heave_rate_error = w - w_ref`
- `tangent_pitch_error`
- `signed_pitch_rate_error = q - q_ref`
- `cross_track_error`
- `predicted_vertical_error = vertical_error + tau_z * heave_rate_error`
- `predicted_pitch_error = tangent_pitch_error + tau_theta * signed_pitch_rate_error`

其中 `tau_z/tau_theta` 是 p90 跟踪时滞的先验值，必须先由 Step 0 审计确定；不能根据
测试结果反复扫。

#### C. 约束/诊断指标

- `base_saturation_fraction`
- `residual_high_frequency_ratio`
- `action_rate_clip_fraction`
- `roll_rate/safety/timeout`
- `terminal_share_last2s`
- `dense_terminal_ratio_last2s`

这些指标不能单独作为主 reward；它们用于约束、失败归因和解释。

### 4.2 reward 候选

所有位置/速度项使用与 observation 一致的物理尺度，先做 deadband/Huber，再进入
potential。`F` 只依赖状态，训练奖励为：

```text
r_shape = gamma * F(s_{t+1}) - F(s_t)
```

候选 potential：

```text
F_z     = -w_z     * Huber(vertical_error)
F_w     = -w_w     * Huber(heave_rate - heave_rate_ref)
F_predz = -w_predz * Huber(vertical_error + tau_z * heave_rate_error)
F_theta = -w_theta * Huber(tangent_pitch_error)
F_predt = -w_predt * Huber(tangent_pitch_error + tau_theta * pitch_rate_error)
F_cross = -w_cross * Huber(cross_track_error)
```

推荐初始组合：

- `R1 = F_z`
- `R2 = F_z + F_w`
- `R3 = F_z + F_w + F_predz`
- `R4 = R2 + pitch potential`
- `R5 = R2 + cross-track potential`

不把 `terminal closure` 直接塞进 `R1–R5`。它只作为 entry-feasible 子集的弱 potential
或诊断项，避免重现 Goal132 的 closure 反噬。

### 4.3 约束候选

先只采用一个约束，避免多约束互相掩盖：

```text
C_entry = mean(max(0, entry_margin - 1)^2)
entry_margin = max(
    |vertical_error| / (v_z_available * T_2m),
    |pitch_error| / (q_available * T_2m)
)
```

只对 `2m <= goal_distance <= 10m` 的 step 生效。`T_2m` 使用当前闭合速度和剩余距离
估计；若闭合速度小于阈值，则使用 conservative time-to-go。`v_z_available` 和
`q_available` 用同一 plant envelope 计算，不能逐 run 调参。

另外保留 `C_u`（action rate/saturation/high-frequency）作为安全约束，但不先启用
多个 action regularizer，避免重复 Goal135 的 authority 正则伤害。

### 4.4 不推荐的奖励

| 项 | 原因 |
|---|---|
| 直接放大 `w_path_3d` | 横向/纵向混合，且无法区分跟踪质量与路径速度 |
| 直接奖励 `terminal closure` | 失败 closure 低是结果；Goal132 已证明强行加速会伤边界样本 |
| 直接奖励 terminal share/dense ratio | Goal119 只移动了 share，success 不动 |
| 直接惩罚 base saturation 或 residual magnitude | 成功/失败都接近饱和，失败 residual 反而更小 |
| 直接惩罚所有 residual 高频 | 可能压制必要修正；应先作为平滑约束而非主 reward |
| 多通道 fixed bias | 已经到上限，不能替代 tracking 目标 |

## 5. 执行计划

- [ ] Step 0：读取 `research/STATE.md`、Goal139 结果和最近 traces；冻结基线、bins、`tau`、reward 配置和 Gate；
- [ ] Step 1：写只读审计脚本，计算分箱指标、success 判别力、提前量、共线性和 entry-margin；
- [ ] Step 2：写审计结果；仅当至少两个中间指标独立领先失败时才进入训练；
- [ ] Step 3：实现 `vertical_error`/`heave_rate`/`predicted_error` potential，默认权重 0 时与旧 reward 逐位 parity；补单元测试；
- [ ] Step 4：实现 entry-margin cost/gate 与对应配置校验；保留旧配置默认关闭；
- [ ] Step 5：本地 smoke：1 env、少量 step，检查 reward finite、动作不饱和、旧行为 parity；
- [ ] Step 6：远端 1 seed × 200k pilot：`R1/R2/R3` 只检查 finite、tracking 方向和 timeout，不用于挑正式 arm；
- [ ] Step 7：正式训练 `R0–R5`，每 arm 3 seeds × 1M；SAC 单独裁决，不与 PPO 混算；
- [ ] Step 8：跑 5 levels × strict/default × dev40 评估，并记录分箱 tracking 指标；
- [ ] Step 9：按 Gate 写 result、blocker/decision、LEDGER、STATE；运行 checker；
- [ ] Step 10：只 add 本 Goal 相关文档、脚本、测试和配置；不自动 commit/push。

### 运行与验证命令（实现后替换脚本名）

```bash
cd settled_局部规划器_反应式

PYTHONPATH=src .venv/bin/python -m pytest -q tests/test_pc_goal140_vertical_reward.py

PYTHONPATH=src .venv/bin/python scripts/analyze_pc_goal140_vertical_reward.py \
  --inputs artifacts/lowlevel/goal117_terminal_closure_attribution_20261111 \
  --output artifacts/lowlevel/goal140_vertical_reward_20261111/audit.json

bash scripts/run_goal140_vertical_reward_sac_train.sh
bash scripts/run_goal140_vertical_reward_sac_eval.sh

PYTHONPATH=src .venv/bin/python scripts/analyze_pc_goal140_vertical_reward.py \
  --inputs artifacts/lowlevel/goal140_vertical_reward_20261111 \
  --output artifacts/lowlevel/goal140_vertical_reward_20261111/analysis.json
```

## 6. Gate

### 6.1 Tracking Gate

以 paired control 为参照，至少 2/3 个 training seeds 同时满足：

| check | 阈值 |
|---|---:|
| overall vertical RMSE | 改善 `>=10%` |
| vertical p95 | 改善 `>=10%` |
| segment vertical RMSE | 至少 3/5 个 progress bins 改善 `>=10%` |
| 2 m entry vertical median | 下降 `>=10%` 或保持 success 改善 |
| path 3D RMSE | 不超过 control `1.05x` |
| residual L2 ratio | 不超过 control `1.50x` |
| safety/timeout | 不超过 control +2/120 |

### 6.2 Success Guard

| check | 阈值 |
|---|---:|
| strict success | 不低于 paired control -1/40 |
| default success | 不低于 paired control -1/40 |
| 至少一个 arm | strict `>= control +2/40`，且 default 不低于 control -1/40 |
| 2 m entry geometry | 不劣于 control |

若 tracking gate 过但 success guard 失败，判 `TRACKING_SUCCESS_TRADEOFF`；若 success
上升但 tracking 不改善，判 `SUCCESS_NOT_TRACKING`，不能宣称完成“全程纵向跟踪”。

### 6.3 裁决表

| 观察 | 判读 | 下一步 |
|---|---|---|
| tracking 和 success 都过 | `VERTICAL_REWARD_SIGNAL` | 扩 8 seeds / formal |
| tracking 过、success 掉 | `TRACKING_SUCCESS_TRADEOFF` | 降低权重/改成 entry constraint，不换算法 |
| success 过、tracking 不动 | `SUCCESS_NOT_TRACKING` | 检查 reward 是否只改 terminal 行为 |
| tracking 不动 | `NO_VERTICAL_REWARD_EFFECT` | 先查 reward scale/parity，再查动作接口 |
| reward 数值异常/饱和 | `INVALID` | 修复实现，不分析算法 |
| PPO residual 仍近零 | `PPO_RESIDUAL_COLLAPSE` | 不在本 Goal 混入 PPO；另开信用/优化 Goal |

## 7. 论文/代码要求

进入 reward 实现前，至少完成 2–5 篇最接近工作的 reading note，并写最小可迁移组件：

- potential-based reward shaping 的理论与边界；
- constrained MDP / Lagrangian reward-cost；
- hierarchical residual control 或 reference/velocity action interface；
- off-policy actor-critic 的 BC regularization / replay prioritization。

优先检查是否有可运行代码、license 和输入输出能否映射到当前 PC observation/action。
不把整篇方法直接搬成黑盒；`research/paper_reading/` 只提交 note，不提交 PDF。

## 8. Git

禁止 `git add -A`。规划阶段只 add：

```bash
git add doc/goals/goal140_vertical_tracking_reward_plan_20261111.md research/STATE.md
git diff --cached --check
git commit -m "docs: add goal140 vertical tracking reward plan"
# 仅在用户明确要求时：
git push
```

实现后按实际改动只 add 对应脚本、测试、配置和 goal 文档；不提交
`artifacts/`、checkpoint、日志、缓存、数据集或大 PDF。

完成前运行：

```bash
.venv/bin/python ../scripts/check_research_ledger.py \
  --goal Goal140 \
  --goal-doc doc/goals/goal140_vertical_tracking_reward_plan_20261111.md \
  --strict --require-result
```

## 9. Goal 提示词

```text
你现在执行 Goal140：全程纵向跟踪的奖励与约束设计。

先读：
- doc/goals/goal140_vertical_tracking_reward_plan_20261111.md
- doc/goals/goal139_mixed_horizon_credit_20261111.md
- doc/goals/goal138_result_20261111.md
- doc/goals/goal137_result_20261111.md
- doc/goals/goal118_result_20261111.md
- doc/goals/goal117_result_20261111.md

硬约束：
1. 先只读审计，再训练；没有指标领先证据不进入 reward ablation；
2. 不改 success、terminal、goal radius、sensor、评估起点或 held-out；
3. 不改变主算法族；Phase 1 只做 SAC reward ablation，PPO residual collapse
   另行归因，不混在同一个 Gate；
4. 所有位置/速度项用 potential-based shaping，不直接堆 raw dense penalty；
5. 一次只加一个核心项或一个约束，保留 R0 paired control；
6. 不扫 seed、不加预算、不换 held-out、不因为单 seed 好就扩 formal；
7. 不自动 commit/push。

执行顺序：
- 审计分箱 tracking 指标和 2 m entry 关系；
- 实现 reward parity 与 smoke；
- 1 seed 200k pilot；
- R0–R5 × 3 seeds × 1M；
- dev40 × 5 levels × strict/default；
- 按 Gate 写 result/blocker/decision/LEDGER/STATE。
```

## 10. 未决问题与默认假设

1. `tau_z/tau_theta` 的具体数值由 Step 0 的 p90 时滞审计决定，不在训练中扫。
2. `heave_rate_ref` 的符号和参考坐标系必须通过 parity/静态测试；在符号 Gate 通过前
   不启用该项。
3. 如果 Goal139 的 SAC 训练尚未完成，Goal140 只保留为预注册规划，不启动正式训练。
4. 如果 Step 0 发现所有候选指标都被 2 m entry geometry 解释，则停止 reward ablation，
   回到 approach geometry 审计；这不是 reward 失败，而是问题层级不在这里。
