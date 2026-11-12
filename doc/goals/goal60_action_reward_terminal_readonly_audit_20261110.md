# Goal60：动作参数化、奖励方向与终端捕获的只读诊断

日期：2026-11-10<br>
状态：`COMPLETED_DIAG_PIVOT`<br>

> 2026-11-10 预分析修正（读取任何 trace 统计之前）：H-A 的“饱和导致误差”应为
> 正相关 `rho >= +0.30`；H-B 的“奖励方向冲突”应为 path/command 奖励越高、
> terminal eta distance 越大，即 `rho >= +0.30`。负相关表示对齐，不作为冲突证据。

父结论：Goal27 已 `H3_SR_FAIL_STOP`；Goal59 已 `FAIL_BOUNDED_RESIDUAL` /
`pivot`<br>
服务器：`yhx@152.136.183.51 -p 43127`，8×RTX 4090

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
  check_command: python ../scripts/check_research_ledger.py --goal Goal60 --goal-doc doc/goals/goal60_action_reward_terminal_readonly_audit_20261110.md --strict
```

## 1. 背景与边界

Goal27 的提示词已经按原预注册执行完毕：命令残差 PPO、reward-v2、长 horizon、
supervised residual 都没有通过效果门禁，最终为 `FAIL / PIVOT`。因此本 Goal
**不重开 Goal27**，不重跑 C1/R1/H1/G1/A1，也不改它的 seed、阈值或 eval split。

Goal59 用 softsign、terminal gate 和 12D endpoint context 继续测试 residual，8 个
formal seed × 5M 仍为 `FAIL_BOUNDED_RESIDUAL`：

- 8/8 deterministic 和 raw saturation 都是 `1.0`；
- 只有 752/880 的 strict125 capture 恰好 `+0.05`，default 不改善，success/plane
  变差；
- 752/880 的 default path-zero 没有效应，记为 `INVALID_ATTRIBUTION`；
- NoisyNet 不是全局主因，mean-weight 策略已经饱和。

本 Goal 只做**只读诊断**：不训练新策略，不扫 reward/gamma/seed/scale，不打开
held-out `100–149`。目标是分清动作参数化、reward 方向和 terminal capture 三个
解释，决定下一个修复 Goal 到底该改什么。

## 2. 可证伪假设

### H-A：动作参数化/饱和方向是主因

机制：softsign + scale 只改变边界形状，actor 仍可以把三个归一化动作推到边界；
饱和的维度与 terminal eta 误差方向不一致，导致 residual 很大但没有闭环收益。

可证伪预测：

- 失败 episode 的 action saturation / residual magnitude 高于成功 episode；
- 每 episode 的饱和比例或 residual magnitude 与 terminal eta distance 有
  `Spearman rho >= +0.30`，且 bootstrap 95% CI 下界大于零（至少一个场景）；
- 饱和维度与 terminal 误差向量的方向关系在多数失败 episode 中同向，而不是
  随机。

若饱和与 terminal 误差无关，或成功/失败 episode 的饱和无差异，则 H-A 不成立，
不能继续把“换 squash 函数”当主线。

### H-B：reward 方向与 terminal capture 冲突

机制：dense path/command reward 奖励了“沿路径前进/压低命令误差”，但这条方向
在终端阶段与真实 goal capture 不一致；residual 优化了错误的代理目标。

可证伪预测：

- 在失败 episode 的终端窗口内，path-progress / command reward 为正，而
  terminal capture 为负；
- episode return 的 path/command 分量与 terminal eta distance 负相关，而
  terminal reward 分量没有提供同向信号；
- 这种冲突在 default 或 strict125 至少一个场景中达到预注册比例。

若 reward 分量与 terminal capture 同向，或失败 episode 的 path 分量并不为正，
则 H-B 不成立。

### H-C：terminal capture 口径不一致

机制：终止平面、协议 success、true-state eta capture 和 `terminal_kind` 存在
口径差异；策略可能“按协议成功”但没有真正捕获，或 true capture 却被记为 miss。

可证伪预测：

- protocol success 与 true-state `terminal_eta_capture` 的 discordance rate
  `>=0.10`（至少一个场景）；
- 或者 `terminal_kind` 与 true capture 在失败 episode 中系统性不一致；
- 终端 signed/vertical/horizontal offset 能解释 discordance。

若两者在 40 个 dev seed 上完全一致，H-C 不成立。

### H-D：优化/探索是主因（SAC 分支）

机制：residual 很小、策略没有离开零附近，或训练曲线仍在探索阶段，导致闭环
不动。

可证伪预测：final checkpoint 的 residual saturation 低、residual 幅度接近零，
且训练曲线仍明显改善。Goal59 现有证据已经显示 residual 非零且饱和；若 trace
再次确认，则 H-D 不成立，不启动 SAC。

### H-E：样本效率是主因（REDQ/DroQ 分支）

机制：同样 5M transitions 下更新不足，增加 UTD/Q ensemble 才能学会 residual。

可证伪预测：training_curve 在 final checkpoint 仍明显上升，或同一 checkpoint
在更多更新后动作/指标持续移动。若训练曲线已平台化且失败是方向错误，则 H-E
不成立，不启动 REDQ/DroQ。

## 3. 固定数据与隔离

| 项目 | 固定值 |
|---|---|
| 诊断 checkpoint | Goal59 `240/752/880/1008` 的 `final.pt` |
| 选择理由 | 240 为典型失败；752/880 为唯一 strict 主效果；1008 为噪声异常 seed |
| 场景 | `default`、`strict125` |
| dev seeds | `20–59`，40 seeds |
| 评估精度 | CPU / float64 |
| 终止方式 | `endpoint_stop` |
| 对照 | 同 seeds 的 forced `zero-residual` |
| 训练 | 不运行 |
| held-out | `100–149` 关闭 |
| checkpoint 选择 | 只用预先声明的 `final.pt`，不选 best |

Goal59 训练配置和环境配置：

```text
configs/lowlevel/pc_goal59_offpolicy_g12_20261109.json
configs/lowlevel/pc_goal59_terminal_bounded_default_20261109.json
configs/lowlevel/pc_goal59_terminal_bounded_strict125_20261109.json
```

checkpoint 绝对路径：

```text
/data/yhx/underwater-robotics/src/plan_p1/goal59_bounded_terminal_residual_20261109/artifacts/formal_s240/final.pt
/data/yhx/underwater-robotics/src/plan_p1/goal59_bounded_terminal_residual_20261109/artifacts/formal_s752/final.pt
/data/yhx/underwater-robotics/src/plan_p1/goal59_bounded_terminal_residual_20261109/artifacts/formal_s880/final.pt
/data/yhx/underwater-robotics/src/plan_p1/goal59_bounded_terminal_residual_20261109/artifacts/formal_s1008/final.pt
```

## 4. 只读 trace 合同

在 `scripts/evaluate_pc_command_residual_offpolicy.py` 增加可选的
`--trace-output`，默认关闭；关闭时 summary 和既有行为逐元素不变。trace 只记录
评估过程，不参与训练。

每个 run 输出：

```text
trace.npz
trace_meta.json
summary.json
episodes.json
```

`trace.npz` 至少包含以下逐 step 字段：

- `step`、`env_index`、`episode_id`、`seed`、`active`；
- `action`（3D normalized）、`residual`（3D physical）、`base_command`（3D）、
  `gate`、`scale_clipped`、`rate_clipped`；
- `reward`、`reward_total`、`reward_command_yaw/pitch/speed`、`reward_hold`、
  `reward_action`、`reward_saturation`、`reward_route_speed`、
  `reward_material_saturation`、`reward_safety`、`reward_goal_plane_miss`、
  `reward_terminal_progress`、`reward_terminal`、`reward_path_cross`、
  `reward_path_vertical`、`reward_path_progress`、`reward_path_heading`；
- `goal_distance`、`signed_offset`、`endpoint_now`、`material_saturated`；
- `path_error_3d`、`path_horizontal_error`、`path_vertical_error`、
  `path_progress`；
- `eta_ned`（3D）、`p_nav_ned`（3D）、`goal_ned`（3D）、
  `goal_tangent_ned`（3D）；
- `done`、`terminal_kind`。

`trace_meta.json` 记录 schema、checkpoint、checkpoint SHA256、training config、
environment config、action mode、path-zero、seeds、字段列表、finite 检查。

数据完整性门禁（不通过则 `INVALID`，不是 `FAIL`）：

- 每个 trace 的 seeds 恰好是 `20–59`；
- 不含 `100–149`；
- 所有数组 finite；
- policy 与 zero-residual 使用同一 seeds、场景和终止模式；
- checkpoint SHA256 与 `summary.json` 一致；
- 每个 episode 至少有一个 terminal 行。

## 5. 诊断 Gate（测量前冻结）

### 5.1 H-A 判定

对每个 scenario，分别计算：

- 每 episode 的 `action_saturation_fraction`（3 维中 `|action| >= 0.9*scale`
  的比例）和 mean residual L2；
- 与 `terminal_eta_distance_m` 的 Spearman 相关及 10,000 次 bootstrap CI；
- 成功 vs 失败 episode 的饱和中位数差。

H-A 支持条件（至少一个 scenario）：

```text
rho >= +0.30 且 bootstrap 95% CI 下界 > 0
并且 failed - success 的 saturation 中位数差 > 0.05
```

### 5.2 H-B 判定

在 terminal window（最后 2 s 或最后 100 step，取较小者）内：

- 计算 path-progress / command reward 的 episode 和；
- 计算同 episode 的 terminal capture 和 terminal reward；
- 统计“path/command 为正、capture 为负”的失败 episode 比例；
- 计算 path/command 分量与 terminal eta distance 的 Spearman 相关。

H-B 支持条件（至少一个 scenario）：

```text
失败 episode 中 “path/command 为正且 capture 为负” 的比例 >= 0.20
或 path/command 分量与 terminal eta distance 的 rho >= +0.30 且 CI 下界 > 0
```

### 5.3 H-C 判定

按 episode 对齐：

- protocol success；
- true-state `terminal_eta_capture`；
- `terminal_kind`；
- terminal signed/vertical/horizontal offset。

H-C 支持条件（至少一个 scenario）：

```text
protocol success 与 true capture 的 discordance rate >= 0.10
或 terminal_kind 与 true capture 的 discordance rate >= 0.10
```

### 5.4 H-D / H-E 判定

- H-D 支持：final residual saturation `<0.50` 且 residual L2 ratio `<0.05`，
  同时训练曲线仍明显改善；否则 H-D 不成立。
- H-E 支持：training_curve 在 final 附近仍单调上升，或同 checkpoint 增加更新后
  动作/指标持续移动；否则 H-E 不成立。

H-D/H-E 的现有 Goal59 证据已偏向“不成立”，本 Goal 只做只读复核，不训练。

## 6. 决策规则

| 诊断结果 | 下一步 |
|---|---|
| H-A 支持 | 新 Goal 只改动作参数化/限幅/rate 语义，不换算法 |
| H-B 支持 | 新 Goal 只改 reward 方向/终端项，先做 1-seed falsification |
| H-C 支持 | 新 Goal 先修 terminal capture 口径，再谈 residual |
| H-A/H-B/H-C 都不支持 | 写 blocker + reading note + LEDGER；重新提出至少 2 个 hypothesis |
| H-D 支持 | 另开预注册 SAC 同协议 Goal |
| H-E 支持 | 才考虑 REDQ/DroQ |

任何情况下都不打开 held-out、不扩 seed 救 PASS、不把 `INVALID` 写成 `FAIL` 或
`PASS`。

## 7. 执行计划

### Step 0：复核与冻结

- [x] 复核 Goal27 `H3_SR_FAIL_STOP`、Goal59 `FAIL_BOUNDED_RESIDUAL` 和两份 STATE；
- [x] 确认本 Goal 不改 Goal27/59 的预注册参数、seed、阈值和 held-out；
- [x] 记录服务器 `nvidia-smi`、`ps`、`df -h /` 和隔离目录。

### Step 1：实现只读 trace

- [x] 在 off-policy env 暴露 `last_reward_breakdown`，默认行为不变；
- [x] 在 evaluator 增加可选 `--trace-output`，默认关闭时 summary 不变；
- [x] 增加 trace schema/finite 检查；
- [x] 运行 1 checkpoint × 2 seeds × 2 scenarios 的 smoke，确认 policy/zero
      的 seeds、字段和 terminal 行完整。

### Step 2：同步服务器与 smoke

- [x] 同步 `src/`、`scripts/`、`configs/`、`tests/` 到隔离目录；
- [x] 在服务器运行 smoke，记录 SHA256、GPU/CPU、耗时和 load；
- [x] smoke 通过前不启动完整 trace。

### Step 3：完整 trace

- [x] 4 个 checkpoint × 2 scenarios 的 policy trace；
- [x] 2 个 scenario 的 zero-residual trace；
- [x] 最多 4 路并发，训练优先；每轮检查 load、内存和 finite；
- [x] 不使用 held-out，不选 best checkpoint。

### Step 4：分析、结果与收口

- [x] 写 `scripts/analyze_pc_goal60_trace_audit.py`，实现 H-A/H-B/H-C/H-D/H-E
      的预注册统计和 bootstrap；
- [x] 输出 `goal60_trace_audit.json`、`goal60_trace_audit.md`；
- [x] 写 `doc/goals/goal60_result_20261110.md`、blocker/decision（如需要）；
- [x] 更新 `research/LEDGER.jsonl`、`research/STATE.md`、`../research/STATE.md`；
- [x] 运行 strict checker。

### Step 5：Git

- [ ] 只 add 本 Goal 的文档、脚本、测试和 research 记录；
- [ ] `git diff --cached --check`；
- [ ] 用户明确要求后才 commit/push；本回合不自动提交。

## 8. 服务器命令

同步：

```bash
sshpass -p 'Bq9068w1' rsync -az \
  -e 'ssh -p 43127 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null' \
  --include='src/***' --include='scripts/***' --include='configs/***' \
  --include='tests/***' --exclude='*' \
  /home/bubble/水下机器人/settled_局部规划器_反应式/ \
  yhx@152.136.183.51:/data/yhx/underwater-robotics/src/plan_p1/goal60_action_reward_terminal_readonly_audit_20261110/
```

smoke：

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src \
/data/yhx/underwater-robotics/envs/venvs/isaaclab/bin/python \
  scripts/evaluate_pc_command_residual_offpolicy.py \
  --config configs/lowlevel/pc_goal59_offpolicy_g12_20261109.json \
  --environment-config configs/lowlevel/pc_goal59_terminal_bounded_default_20261109.json \
  --checkpoint /data/yhx/underwater-robotics/src/plan_p1/goal59_bounded_terminal_residual_20261109/artifacts/formal_s752/final.pt \
  --output artifacts/traces/smoke_default_policy \
  --seed-start 20 --seed-count 2 --device cpu --dtype float64 \
  --action-mode policy --trace-output artifacts/traces/smoke_default_policy
```

完整 trace 用相同模板替换 `scenario`、`checkpoint`、`action-mode` 和 output；
每个 run 独立目录，不覆盖旧结果。并发最多 4 路，训练/其他用户任务优先。

## 9. 明确禁止

- 不重开 Goal27 的 PPO、reward、gamma/lambda、supervised residual 或 seed 搜索；
- 不启动 SAC/REDQ/DroQ；
- 不打开 held-out `100–149`；
- 不把只读 trace 当作训练结果；
- 不因为诊断结果不理想就换 seed、加预算或放宽阈值；
- 不 `git add -A`，不自动 commit/push。
