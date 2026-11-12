# Goal87 terminal/capture 审计结果：捕获边界毫米级移动，不改变终止类别

**日期**：2026-11-10<br>
**状态**：`RESULT_RECORDED / TERMINAL_CAPTURE_MILLIMETER_ONLY`<br>
**上游**：Goal81 `GUARD_NOT_PREEMPTIVE`、Goal82 `CHANNEL_GAIN_WITHOUT_OUTCOME`、Goal84 `FRAME_ACCOUNTED`、Goal85 `POSITION_SOURCE_ETA`<br>
**性质**：只读/回放既有 checkpoint 的 eval-only 审计；不训练、不新增训练型 rollout、不改 reward / terminal / goal radius / action 参数化 / seed / held-out。

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
  check_command: python ../scripts/check_research_ledger.py --goal Goal87 --goal-doc doc/goals/goal87_residual_credit_assignment_20261110.md --strict
```

本审计只回答一个问题：在 Goal85 的 `position_source=eta` 契约下，冻结 BC 与 PPO 残差
的 terminal/capture 结果是否出现“reward 改善但终止捕获不变”的错位。

## 1. 固定协议

| 项目 | 固定值 |
|---|---|
| 基座 | D3 BC，`pc_bc_actor.pt`，path8 / command0 / current encoder off |
| 兼容 checkpoint | path12 + command3 + zero current encoder，SHA `80024ef5...ff9dd` |
| 残差臂 | `pa_bc_frozen_residual`（零初始化，8 iter） |
| 残差强臂 | `pa_bc_frozen_residual_scale1_upd25x40`（scale=1.0，40 iter，seed 86201） |
| 评价 | CPU/float64，dev seeds 20–59，每臂 40 episodes |
| 场景 | default（goal radius 2.0 m）、strict125（goal radius 1.25 m） |
| 训练 | 无 |
| held-out | `100–149` 关闭 |
| 成功语义 | 终止帧 `eta` 进入 goal radius；`p_nav` 只作为观测/诊断 |

本轮的 trace 中 `goal_distance_pnav_m` 与 `goal_distance_eta_m` 逐行完全相等，说明
`termination.position_source=eta` 已生效；旧的 p_nav/eta capture-frame 差异不再
进入终止判定。summary 中的 `p_nav_done_distance_m` 是传感器 p_nav 的 done 诊断，
不能当作终止帧的 eta 距离。

## 2. 主要结果

### 2.1 default：30/40 成功，全部停在 2.0 m 边界内

| 指标 | BC base | scale1 residual | 差值 |
|---|---:|---:|---:|
| success | 30/40 = 0.750 | 30/40 = 0.750 | 0 |
| plane miss | 10/40 = 0.250 | 10/40 = 0.250 | 0 |
| timeout | 0/40 | 0/40 | 0 |
| capture-ever @1.25 m | 0/40 | 0/40 | 0 |
| capture-ever @1.50 m | 0/40 | 0/40 | 0 |
| capture-ever @2.00 m | 30/40 | 30/40 | 0 |
| capture-ever @2.50 m | 35/40 | 35/40 | 0 |
| success 终止帧距离均值 | 1.9877 m | 1.9861 m | -0.0016 m |
| success 终止帧余量均值 | +0.0123 m | +0.0139 m | +0.0016 m |
| plane-miss 终止距离均值 | 2.6928 m | 2.6951 m | +0.0023 m |

终止距离分桶（eta，done 帧）：

| 区间 | BC base | scale1 residual |
|---|---:|---:|
| `[0, 1.25)` | 0 | 0 |
| `[1.25, 1.50)` | 0 | 0 |
| `[1.50, 2.00)` | 30 | 30 |
| `[2.00, 2.50)` | 5 | 5 |
| `[2.50, ∞)` | 5 | 5 |

30 个成功 episode 的终止余量只有 `0.8–31 mm`，其中 14 个小于 `1 cm`、24 个小于
`2 cm`。这不是“残差把载具带进目标中心”，而是“载具刚好越过 2.0 m 球壳就被
终止”；因此成功计数对厘米/毫米级轨迹变化极其敏感。

### 2.2 strict125：17/40 成功，13 个失败卡在 1.25–2.0 m 近失壳

| 指标 | BC base | scale1 residual | 差值 |
|---|---:|---:|---:|
| success | 17/40 = 0.425 | 17/40 = 0.425 | 0 |
| plane miss | 23/40 = 0.575 | 23/40 = 0.575 | 0 |
| capture-ever @1.25 m | 17/40 | 17/40 | 0 |
| capture-ever @1.50 m | 21/40 | 21/40 | 0 |
| capture-ever @2.00 m | 30/40 | 30/40 | 0 |
| capture-ever @2.50 m | 35/40 | 35/40 | 0 |
| success 终止帧距离均值 | 1.2376 m | 1.2361 m | -0.0015 m |
| success 终止帧余量均值 | +0.0124 m | +0.0139 m | +0.0015 m |
| plane-miss 终止距离均值 | 2.0933 m | 2.0918 m | -0.0015 m |

终止距离分桶（eta，done 帧）：

| 区间 | BC base | scale1 residual |
|---|---:|---:|
| `[0, 1.25)` | 17 | 17 |
| `[1.25, 1.50)` | 4 | 4 |
| `[1.50, 2.00)` | 9 | 9 |
| `[2.00, 2.50)` | 5 | 5 |
| `[2.50, ∞)` | 5 | 5 |

strict125 的 13 个近失 episode 在 `1.25–2.0 m` 内结束，说明失败不是“完全没有
接近”，而是停在 capture 球外壳；但 scale1 没有改变任何一个桶的计数。

### 2.3 逐 seed 配对：毫米级移动，零类别翻转

| 配对指标 | default | strict125 |
|---|---:|---:|
| `d_eta_done` 均值差 | -0.00065 m | -0.00150 m |
| `d_eta_done` 中位数差 | -0.00067 m | -0.00124 m |
| `d_eta_done` 范围 | -0.0283…+0.0184 m | -0.0461…+0.0157 m |
| capture-ever @1.25/1.5/2.0/2.5 类别变化 | 0/0/0/0 | 0/0/0/0 |
| plane-miss / timeout 类别变化 | 0 / 0 | 0 / 0 |
| `mean_episode_reward` 均值差 | +0.00477 | +0.00449 |
| `path_three_d_rmse_m` 均值差 | -0.00020 m | -0.00027 m |

reward 的改善只有千分之几，且逐 seed 方向不一致；terminal 类别在 40 个 seed 上
完全不变。因此这不是“reward 变好但 terminal capture 不变”的强错位，而是
“reward 和 terminal 都几乎不变”。

## 3. 残差到底有多大

| 指标 | BC base | frozen residual | scale1 residual |
|---|---:|---:|---:|
| `residual_action_abs_mean`（归一化） | 0 | 0.000371 | 0.001938 |
| `residual_action_abs_p95` | 0 | 0.000919 | 0.003021 |
| `residual_action_l2_mean` | 0 | 0.000844 | 0.003536 |
| `residual_to_base_l2_ratio_mean` | 0 | 0.001355 | 0.005659 |
| `residual_to_base_l2_ratio_p95` | 0 | 0.003074 | 0.012724 |
| PPO Gaussian `init_noise_std` | — | 0.1 | 0.1 |

结论：

- 评估时确定性学习残差只有基座 L2 的 `0.14%–0.57%`，不是“大残差”；
- 指标里看起来很大的 `actuator_residual_abs_mean ≈ 38–50 rpm / 0.18 rad` 是
  **冻结 BC 动作相对 ALOS 基座的差**，不是 PPO 学到的残差；
- 训练时每个动作通道的 Gaussian 噪声标准差是 `0.1`，而确定性残差 L2 只有
  `0.0035`；探索噪声约为确定性残差 L2 的 `49×`，因此训练采样阶段的“残差”
  主要是策略噪声，不是已学到的均值；
- 当前配置没有 NoisyNet；NoisyNet 主导的旧问题不适用于这组实验。

## 4. 与历史网络结构/代码的比对

### 4.1 当前 z_t 结构与 Goal80 的 z_t 代码

当前 `pc_policy.py` 与 Goal80 `repo/src/auv_lowlevel/batch/pc_policy.py` 的差异
集中在三处，不是主干网络退化：

1. path12 被拆成 `bc_path_dim=8` 的 BC 前缀 + endpoint4 尾段；
2. 增加 `frozen_pc_bc_residual` 分支，冻结 BC encoder/actor/path/current；
3. 修复 frozen BC 分支在 `n_z_reserved` 上的重复补零（原逻辑会把 60 维补成
   68 维）。

state encoder `[128,128]`、`d_z_total=64`、`n_z_reserved=4`、actor `[256,128]`、
critic `[256,128]`、ELU、DADE 关闭等主干设置没有变化。

### 4.2 BC checkpoint 和兼容 checkpoint

原始 D3 BC checkpoint：

```text
path_dim=8
actor_command_dim=0
actor_current_encoder=off
actor.0.weight  (256,128)  norm 17.1671
actor.2.weight  (128,256)  norm 14.1374
actor.4.weight  (3,128)    norm 0.318514
path_encoder.0  (64,8)     norm 12.2268
path_encoder.2  (64,64)    norm 11.9113
```

兼容 checkpoint 对 path 新增的 4 列、command 新增的 3 列和 current encoder
全部零填充；actor/encoder/path 的已有权重范数逐项一致，因此初始 BC 动作等价。

### 4.3 当前 frozen residual actor 不是 BC actor 的复制

frozen residual 的可训练 actor 使用独立随机初始化的 state encoder：

```text
actor_encoder.network.0.weight  norm 6.49   (BC: 13.91)
actor_encoder.network.2.weight  norm 6.54   (BC: 17.63)
path_encoder final weight norm  0.30
current_encoder final norm      0.22
actor.4.weight                  0.0147
```

这是设计上的“冻结 BC 基座 + 新残差头”，不是权重错接；但它意味着残差头必须在
一个新的特征空间里从零学习信用分配。`pa_bc_head` 臂把 BC encoder/actor 热启动
后，actor 权重范数恢复到 BC 水平（约 17.2 / 14.1 / 0.314），闭环回到 BC 水平，
但没有稳定超过 BC。这支持“BC 先验有效，PPO 增量没有学会”，而不是“网络接错”。

### 4.4 历史 Goal59 checkpoint 不是同一网络

Goal59 的 `formal_s*/final.pt` 是：

```text
observation_dim=30
GRU history
twin critics
NoisyNet (use_noisy=true)
softsign actor output
```

它与当前 `z_t` PC actor 不是同一架构，不能把它的 success 或参数量直接当作
“同一网络的对照”。如果后续要比较传统方法，必须明确是 ALOS/PD 基座、D3 BC，
还是旧 Goal59 的 GRU/NoisyNet 策略。

## 5. 判定

```text
terminal_capture_changed = false
reward_changed_meaningfully = false
residual_dominant_by_noisenet = false
learned_residual_dominant = false
root_cause = actor update / credit assignment, not reward/terminal contract
```

- 不改 reward shape；
- 不改 terminal/capture 规则；
- 不改 action 参数化；
- 不把 scale1 的毫米级改善写成“有效残差”；
- 不因毫米级 reward 改善启动 SAC/REDQ/DroQ；
- 下一步先做 **PPO actor 更新幅度/信用分配** 的预注册诊断：减少
  `num_mini_batches`、降低 actor lr、检查零初始化输出尺度与 old-log-prob
  一致性；只有该诊断仍无法产生非零有效残差，才开 SAC 同协议对照。

## 6. 复现

远端仓库根目录：

```bash
export PYTHONPATH="src:/data/yhx/underwater-robotics/src/stable-baselines3"
PY=/data/yhx/underwater-robotics/envs/venvs/isaaclab/bin/python

# BC base（default / strict125）
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  "$PY" scripts/evaluate_pc_ppo.py \
  --training-config configs/lowlevel/goal86/ppo_diag_20261110/pa_bc_frozen_residual.json \
  --checkpoint artifacts/lowlevel/goal86/ppo_diag/pa_bc_frozen_residual_seed86201/model_initial.pt \
  --output artifacts/lowlevel/goal87_terminal_audit_20261110/bc_default \
  --seed-start 20 --seed-count 40 --device cpu --dtype float64 \
  --trace-output artifacts/lowlevel/goal87_terminal_audit_20261110/bc_default/trace.npz

# scale1 residual（default / strict125）
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  "$PY" scripts/evaluate_pc_ppo.py \
  --training-config configs/lowlevel/goal86/ppo_diag_20261110/pa_bc_frozen_residual_scale1_upd25x40.json \
  --checkpoint artifacts/lowlevel/goal86/ppo_diag/pa_bc_frozen_residual_scale1_upd25x40_seed86201/final.pt \
  --output artifacts/lowlevel/goal87_terminal_audit_20261110/scale1_default \
  --seed-start 20 --seed-count 40 --device cpu --dtype float64 \
  --trace-output artifacts/lowlevel/goal87_terminal_audit_20261110/scale1_default/trace.npz
```

strict125 评价在两条命令上增加：

```text
--environment-config configs/lowlevel/pc_goal85_unbiased_strict125_candidate_20261110.json
```

## 7. Git

本轮只新增诊断文档、blocker、reading note 和 ledger 记录；不提交 `artifacts/`、
checkpoint、trace 或日志。

```bash
git add doc/goals/goal87_terminal_capture_audit_result_20261110.md \
        research/blockers/2026-11-10-goal87-ppo-no-movement.md \
        research/paper_reading/2026-11-10-goal87-residual-ppo-credit-terminal-capture.md \
        research/LEDGER.jsonl research/STATE.md
git diff --cached --check
git commit -m "goal87: audit terminal capture and residual update magnitude"
```

禁止 `git add -A`；不自动 commit/push。
