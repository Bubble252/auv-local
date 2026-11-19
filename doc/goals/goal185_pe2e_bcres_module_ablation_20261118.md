# Goal185：PE2E × BCRES 模块集消融（MOD3 → MOD0）

**日期**：2026-11-18<br>
**状态**：`BLOCKED_PENDING_USER_DECISION / EXECUTED_2026-11-19_INVALID`<br>
**上游**：Goal184 Wave 1（MOD4）、Goal182 模块集轴注册表

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
  check_command: python3 /home/bubble/水下机器人/scripts/check_research_ledger.py --goal Goal185 --goal-doc doc/goals/goal185_pe2e_bcres_module_ablation_20261118.md --strict
```

## 1. 背景与证据

- Goal184 Wave 1 正在跑 `PE2E × BCRES × MOD4`（seed86201，2600 it）。
  2026-11-18 22:24 的 C6 窗口：completion `0.9798`、cross `0.4264 m`、
  vertical `0.7706 m`、success `0.6912`、timeout `0`、safety `0`、
  material saturation `0.8085`。
- 同窗口 `PE2E × BCFT × MOD4`：completion `0.9771`、cross `0.7500 m`、
  vertical `2.2509 m`、success `0.7698`、saturation `0.3475`。
  BCRES 的路径精度更好，但动作更激进；目前不知道收益来自哪个模块。
- 模块集轴以 `configs/governance/goal182_axis_registry.json` 为单一来源：

| 变体 | DADE 时序 | direct current encoder | privileged critic |
|---|---|---|---|
| MOD4 | 开 | 开 | 开（`critic_path_dim=30`） |
| MOD3 | 开 | 开 | 关（`critic_path_dim=0`） |
| MOD2 | 开 | 关 | 开 |
| MOD1 | 开 | 关 | 关 |
| MOD0 | 关 | 关 | 关 |

- 用户 2026-11-18 决定：本 goal 只做 `PE2E × BCRES`，不复用 ALOSB、
  不改任何非 MOD 超参。

## 2. 假设

1. **H185-MOD3-NULL**：privileged critic 只影响 value/return 与学习稳定性，
   不影响 BCRES 的路径误差。若成立，MOD3 的 C6 cross/vertical 与 MOD4
   同量级且过同一冻结阈值；若失败，MOD3 的 cross 或 vertical 明显变差或
   阈值失败。
2. **H185-CURRENT-MAIN**：current encoder 是 BCRES 路径增益的主要来源。
   若成立，MOD2（去掉 current encoder）相对 MOD3 在 cross 和/或 vertical
   上退化；若失败，MOD2 与 MOD3 同量级。
3. **H185-HISTORY**：DADE 时序对完成度和纵向误差有独立贡献。若成立，
   MOD0 相对 MOD1 在 completion 或 vertical 上退化；若失败，两者同量级。
4. **H185-SAT-COST**：模块减少会降低动作饱和度。若成立，MOD3/MOD2/MOD1/MOD0
   的 `material_saturation_rate` 低于 MOD4 的 `0.8085`；若失败，说明饱和度
   主要来自 residual 配方而不是模块集。

最便宜的 falsification：直接跑预注册的四条 2600-it 单 seed arm，不另开
超参或 seed 预实验；每条 arm 自身的 stage gate 和最终矩阵回答上述问题。

## 3. 数据与隔离

| 项目 | 冻结值 |
|---|---|
| demo 数据 | `/data/yhx/underwater-robotics/runs/auv-local/goal184/20261118-goal184-64d8225/data/demos_goal184_stage_matched_v4_teacher_alos_truth/demos.npz` |
| demo SHA256 | `cee7e4906cf526ccad6a58c6a9e203c387fb9e4d09a8fdee2d85731f879813fc` |
| demo 规模 | 593 episodes / 688,128 samples / C0–C6 |
| MOD3 BC | `pe2e_bc_stage_matched_v4_teacher_alos_truth/pc_bc_actor.pt`，SHA `86cab3fef9ee5784c0d3ef9441b1a409636df45204902713bb2fc950ebb1c086` |
| MOD2/1/0 BC | `pe2e_bc_stage_matched_v4_teacher_alos_truth_nocurrent/pc_bc_actor.pt`，SHA `f593cd9957cd245a932611b188c9fac2c08b6978910e3b22e9d580ef4b266c80`（same demo，`actor_current_encoder=none`，best epoch 16 / val 0.05238） |
| eval seeds | `20/21/22`；dev `20–23` |
| held-out | `100–149` 关闭 |
| 半径 | 只用 `2.0 m` |

- 不重采 demo；除 `actor_current_encoder_enabled` 外，BC 训练配方与 v4 完全一致。
- 不把 MOD3 的 BC 拿去跑 MOD2/1/0；也不复用 Goal182 的旧无 current BC
  `f7103f04...`（旧 demo/旧配方，会把模块轴和 BC 数据轴混在一起）。

## 4. 方法

### 4.1 四条 arm

| arm | config | 相对 MOD4 的变化 | BC |
|---|---|---|---|
| MOD3 | `train_formal_mod3_pe2e_bcres_env_empirical_5d_m_seed86201.json` | `pc.critic_path_dim: 30→0` | v4 `86cab3fe...` |
| MOD2 | `train_formal_mod2_pe2e_bcres_env_empirical_5d_m_seed86201.json` | `pc.actor_current_encoder_enabled: true→false` | 新无 current BC |
| MOD1 | `train_formal_mod1_pe2e_bcres_env_empirical_5d_m_seed86201.json` | 上两项 | 新无 current BC |
| MOD0 | `train_formal_mod0_pe2e_bcres_env_empirical_5d_m_seed86201.json` | 上两项 + `pc.dade_enabled: false`、`pc.dade_history_mode: "raw"` | 新无 current BC |

MOD0 与 MOD2/MOD1 共用同一无 current BC；DADE warm start 是零输出，
关闭 `dade_enabled` 只是不再学习该分支。若 smoke 发现 actor 输入契约不一致，
停下来补 BC，不得静默改配置。

### 4.2 除 MOD 外一律不变

- env：`configs/lowlevel/goal184/env_path_fixed_speed_formal_empirical.json`，
  reward/termination/curriculum 条目原样保留；
- `base_policy.mode=frozen_pc_bc_residual`、`residual_action_mask=[1,1,1]`、
  `residual_scale=1.0`；
- actor/critic LR `3e-4`、`schedule=adaptive`、`init_noise_std=0.3`、
  `clip_param=0.2`、`desired_kl=0.01`、`entropy_coef=0.005`、
  `gamma=0.995`、`lam=0.95`、5 epochs、4 minibatches；
- `training_profile=PPO-FLAT-4096x48-E5-MB4`、4096 envs × 48 steps、
  `runner.empirical_normalization=false`；
- 2600 iterations、seed `86201`、阈值表不重标定。

## 5. 执行计划

### 5.1 不可协商的运行契约

| 项目 | 强制要求 | 失败标签 |
|---|---|---|
| 训练课程 | env config 内 `curriculum.enabled=true`，逐 episode `curriculum_episodes.jsonl` | `INVALID_NONCURRICULUM` |
| 训练速度 | 每 stage `speed + current + variation <= 2.5 m/s` | `INVALID_SPEED_CONTRACT` |
| 正式评价 | 完整 `arms × levels × radii × zero baselines`，半径只 2.0 m | `INVALID_INCOMPLETE_EVAL` |
| 评价 seeds | `20/21/22`；held-out 关闭 | `INVALID_SEED_SPLIT` |
| 启动/结束 | 先 `check_training_profile.py`，结束跑产物审计器 | `INVALID_PROTOCOL` |
| 主指标 | cross/vertical/completion + 整条路线；success/timeout/pre-goal 只作 telemetry | — |

### 5.2 Steps

- [x] **Step 0：配置生成**
  - 新增 `scripts/build_goal185_mod_ablation_configs.py`，以 MOD4 empirical BCRES config 为模板，
    只改 §4.1 的字段与 `experiment.name/output`；
  - 每份运行 `python3 /home/bubble/水下机器人/总协调/scripts/check_training_profile.py --training-config <config>`；
  - `git add` 脚本与 4 份 config，`git diff --cached --check` 后 commit。
- [x] **Step 1：无 current encoder BC**
  - 已重训：`actor_current_encoder=none`，same demo，best epoch 16 / val `0.05238`，
    SHA `f593cd99...`；
  - BC_ONLY C3 seeds 20–23（只读诊断，不作 Gate）：
    3D `1.8718 m`、horizontal `1.4608 m`、vertical `1.1703 m`、
    success `0.25`（1/4）、timeout `0`；
    对照 v4 current BC：3D `1.9933 m`、horizontal `1.4405 m`、
    vertical `1.3778 m`、success `0.75`；
  - 判读：路径误差不差于 v4，但速度跟踪 `speed_rmse 0.81 vs 0.46`、success
    telemetry 更弱。MOD2/1/0 的结果带 BC 质量混杂，只能作描述性消融，
    不能单独把退化归因给模块。
- [x] **Step 2：合同 smoke**
  - 4 个 MOD 各跑一次 1-it 1536-env contract smoke；
  - 检查 frozen BC 加载、MOD0 的 `dade_enabled=false`、`critic_path_dim`、
    课程 provenance、manifest；
  - 不按 seed 重复 smoke。
  - 2026-11-18 23:21 已完成：`mod3/mod2/mod1/mod0` 各 1-it 1536-env，
    全部 `rc=0`；manifest 核对：
    MOD3 `ce=true/cpd=0/dade=true`、MOD2 `ce=false/cpd=30/dade=true`、
    MOD1 `ce=false/cpd=0/dade=true`、MOD0 `ce=false/cpd=0/dade=false/raw`，
    BC 分别指向 v4 与 `f593cd99...`。
- [x] **Step 3：排队训练（已挂起，等待空闲卡）**
  - 顺序 `MOD3 → MOD2 → MOD1 → MOD0`，先只跑 seed86201；
  - 每条 2600 it，`--curriculum-eval-stage-boundaries --curriculum-route-gate-mode enforce`，
    阈值表用
    `route_gate_calibration_pe2e_stage_matched_v4_100it_fixedlr/thresholds/goal182_route_gate_pe2e_20_3.thresholds.json`
    （SHA `20ab53f6...`，`status=FROZEN`，C0–C6 全覆盖）；
  - 单卡单进程，用 `wait_pid` 链式排队；不一次排满。
  - 2026-11-18 23:22 已挂起 `scripts/goal185_gpu_queue.py`：
    - server worktree：`/data/yhx/underwater-robotics/worktrees/auv-local/goal185/20261118-goal185-d7ea65d`
    - run root：`/data/yhx/underwater-robotics/runs/auv-local/goal185/20261118-goal185-d7ea65d`
    - queue：`queue.json`（4 条，`need_mb=22000`，无 deadline）
    - 当前无卡满足 22 GB + 1.5 GB margin，队列在等待；PE2E 于 23:55 停止后
      GPU1/2 释放，MOD3/MOD2 应自动启动。
- [ ] **Step 4：评价**
  - 每条 arm 跑完整 `arms × levels × radii(2.0) × zero baselines`；
  - 审计产物，缺 cell 标 `INVALID_INCOMPLETE_EVAL`。
- [ ] **Step 5：收尾**
  - 写 `goal185_result_20261118.md`；
  - `research/LEDGER.jsonl` 记 `continue/pivot/stop`；
  - 运行 Research Ledger 检查后 commit。

### 5.3 运行命令模板

```bash
W=<goal185 worktree>
PY=/data/yhx/underwater-robotics/envs/venvs/isaaclab/bin/python
R=/data/yhx/underwater-robotics/runs/auv-local/goal185/20261118-goal185-<sha>
T=/data/yhx/underwater-robotics/runs/auv-local/goal184/20261118-goal184-64d8225/route_gate_calibration_pe2e_stage_matched_v4_100it_fixedlr/thresholds
CUDA_VISIBLE_DEVICES=<gpu> PYTHONPATH=$W/src:$CORE \
AUV_GOVERNANCE_ROOT=$GOV \
$PY $W/scripts/train_batch_lowlevel_ppo.py \
  --training-config $W/configs/lowlevel/goal185/train_formal_mod3_pe2e_bcres_env_empirical_5d_m_seed86201.json \
  --iterations 2600 \
  --curriculum-eval-stage-boundaries \
  --curriculum-route-gate-mode enforce \
  --curriculum-route-thresholds $T/goal182_route_gate_pe2e_20_3.thresholds.json \
  --output $R/full_wave/mod3/pe2e_bcres_seed86201 \
  --seed 86201
```

### 5.4 GPU 安全排队（2026-11-18 用户指令）

核心约束：**不得让其他用户的实验出问题**。4090 没有 MIG，只能时间片共享，
因此用“启动前安全检查 + 进程级显存硬上限 + 运行中压力抢占”三层保护：

- 启动条件（同时满足）：
  - 目标卡 `free >= need + margin`（默认 `margin=1536 MiB`）；
  - 其他用户当前占用 `<= other_max`（默认 `4096 MiB`）；
  - 本队列在该卡的活跃任务按声明需求预留后仍有空间。
- 启动时给本进程设置 CUDA allocator 硬上限
  `cap = free - margin`（`scripts/_gpu_mem_cap.py`），本任务不可能占用
  启动时未空闲的显存。
- 运行中每 30 s 监控：若空闲低于 `512 MiB`，或其他用户占用比启动时增长
  超过 `margin`，只停止**我们自己的**任务；绝不向其他用户进程发信号。
- 硬上限在单条 run 内固定，不会随别人释放显存而长大；释放出来的空间由
  后续排队 arm 或小任务（smoke/BC，`need` 更小）使用，提高整体利用率。
- 2600-it 任务的 `need_mb=22000`；1-it smoke 约 `9000`；BC 约 `6000`。

脚本：

```bash
# 单次 dry-run，只打印每张卡是否可用
python3 scripts/goal185_gpu_queue.py --queue <queue.json> --gpus 0-7 \
  --margin-mb 1536 --other-max-mb 4096 --dry-run

# 常驻排队
nohup python3 scripts/goal185_gpu_queue.py --queue <queue.json> --gpus 0-7 \
  --margin-mb 1536 --other-max-mb 4096 \
  --state <run>/queue_state.json --log <run>/queue.log \
  > <run>/queue.out 2>&1 &
```

**Wave 1 停止安排**：现有 `PE2E BCFT/BCRES`（pid 由 `wave1_pids.txt` 记录）
最多跑到 **2026-11-18 23:55**，由 `scripts/goal185_stop_at.py` 停止：

```bash
nohup python3 scripts/goal185_stop_at.py --at 2026-11-18T23:55:00 \
  --job pe2e_bcft:<pid>:$R/full_wave1/pe2e_bcft_seed86201 \
  --job pe2e_bcres:<pid>:$R/full_wave1/pe2e_bcres_seed86201 \
  --log $R/logs/goal185_stop_at.log > $R/logs/goal185_stop_at.out 2>&1 &
```

训练器目前没有 SIGINT/SIGTERM 优雅落盘钩子，因此停止时以“最后一个
`model_*.pt`（每 100 it 保存）+ 已 flush 的 `iteration_records.jsonl` /
`curriculum_episodes.jsonl`”为准，并在 run 目录写
`stopped_by_deadline_audit.json`（记录最后 iteration、checkpoint 路径与 SHA）。
停止后 GPU1/2 释放，可直接进入 Goal185 的排队。

## 6. Gate 与判读

训练内 stage gate：窗口 20、连续 3 次通过、最低 420 it、`mode=enforce`；
阈值来自冻结的 PE2E 表（C0–C6），不因 module 变化重标定。

| 对比 | 可归因 | 注意 |
|---|---|---|
| MOD4 vs MOD3 | privileged critic | 同 BC，干净配对 |
| MOD2 vs MOD1 | privileged critic（无 current） | 同 BC，干净配对 |
| MOD1 vs MOD0 | DADE 时序 | 同 BC，干净配对 |
| MOD3 vs MOD2 | 不可单独归因 | 同时换 current encoder 和 BC |
| MOD4 vs MOD1 | 模块总贡献 | 含 BC 差异 |

失败裁决：

| 观察 | 判读 | 下一步 |
|---|---|---|
| 某 arm stage gate 连续失败 2 次 | 该模块组合不适配当前配方 | 停该 arm，记 blocker |
| MOD3 与 MOD4 同量级 | 支持 H185-MOD3-NULL | 记录 screen 结论，不据此宣称 formal 等价 |
| MOD2 明显退化 | 支持 H185-CURRENT-MAIN | 保留 current encoder 为主线 |
| MOD0 明显退化 | 支持 H185-HISTORY | 保留 DADE 为主线 |
| 所有 arm 同量级 | 模块集在该任务上边际贡献小 | 写 negative result，不扩 seed 扫参 |

## 7. 不做的实验

- 不做 ALOSB、不做 transformer actor、不做 BCFT/TRRES、不扫 seed；
- 不重标定阈值、不改 reward/termination/课程、不改 LR/std/clip/KL；
- 不为“确认退化”另开预实验；退化由预注册 arm 本身回答。

## 8. Git

禁止 `git add -A`。提交前：

```bash
python3 /home/bubble/水下机器人/scripts/check_research_ledger.py --goal Goal185 --goal-doc doc/goals/goal185_pe2e_bcres_module_ablation_20261118.md --strict
git diff --cached --check
```

## 9. Goal 提示词

```text
你现在执行 Goal185：PE2E × BCRES 模块集消融。

先读：
- doc/goals/goal185_pe2e_bcres_module_ablation_20261118.md
- doc/goals/goal184_empirical_full_arm_training_20261118.md
- configs/governance/goal182_axis_registry.json

硬约束：
1. 只做 PE2E × BCRES × MOD3/MOD2/MOD1/MOD0，seed86201，2600 it；
2. 除 MOD 字段外，任何 env/reward/LR/std/clip/课程/阈值都不改；
3. MOD3 复用 v4 BC；MOD2/1/0 复用同一份无 current encoder BC；
4. 先 4 个 1-it smoke，再按 MOD3→MOD2→MOD1→MOD0 单卡排队；
5. 评价半径只 2.0 m，success/timeout/pre-goal 只作 telemetry；
6. held-out 100–149 关闭；不扫 seed、不扩 arm、不一次排满。
```

## 10. 执行现状与待定夺修复项（2026-11-19）

### 10.1 执行结果

2026-11-19 00:05:29，队列因 `pe2e_bcres_mod0_seed86201` 退出 `rc=1`
触发 `--stop-on-failure`，daemon 退出。四臂结果：

| arm | 状态 | 说明 |
|---|---|---|
| MOD3 | `INVALID` | 120 it，`StageBudgetExhausted: stage=C0 consumed=120 hard_cap=120` |
| MOD2 | `INVALID` | 启动后 7 it 被 `other-growth` 安全停（SIGTERM），未真正训练 |
| MOD1 | `INVALID` | 120 it，同一 `StageBudgetExhausted` |
| MOD0 | `INVALID` | 120 it，同一 `StageBudgetExhausted` |

- 不是 OOM：日志 traceback 是 `StageBudgetExhausted`。
- 不是 BC 未加载：MOD3 iteration 18 的 cross 仍是 `0.375 m`；
  manifest 的 `base_checkpoint` 和 SHA 都是 v4 `86cab3fe...`。
- 不是 env 不同：Goal184/Goal185 的 `environment_config_sha256` 都是
  `db4de5ff1ed1d5cb9b4b37da6d04df95c82855d61d609cc6e0e5d61512cc9729`。
- 不是 seed 不同：都是 `86201`。
- 不是 MOD 外超参变化：MOD3 config 与 MOD4 的 diff 只有 `name`、`output`、
  `pc.critic_path_dim 30→0`。

### 10.2 根因 A：adaptive LR 在 minibatch 内连续放大

rsl_rl `ppo.py` 的 adaptive 规则：每个 minibatch 根据 KL 调 LR，
`KL > 2*desired_kl` 时 `/1.5`，`KL < desired_kl/2` 时 `*1.5`，硬上限 `1e-2`。
本 goal 配置为 `desired_kl=0.01`、初始 `lr=3e-4`。

MOD3 前 10 it：

| it | lr | KL | clip | ratio_max | cross (m) |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.00228 | 0.0018 | 0.001 | 1.20 | - |
| 2 | 0.01000 | 0.1028 | 0.614 | 8.12 | - |
| 3 | 0.00444 | 0.0417 | 0.403 | 3.07 | - |
| 6 | 0.01000 | 0.2635 | 0.763 | 13.58 | - |
| 7 | 0.00001 | 0.0607 | 0.240 | 28.38 | - |

cross-track 轨迹：17 it `0.375` → 19 it `0.624`（KL `0.0315`）→ 120 it `1.251 m`。
MOD1：17 it `0.432` → 120 it `1.292 m`。MOD0：17 it `0.478` → 120 it `0.716 m`。
C0 晋升阈值 `0.621 m`，MOD0 是擦线失败。

按 LR 分桶统计（MOD3）：

| LR 区间 | KL 中位数 | KL 最大 | clip 最大 |
|---|---:|---:|---:|
| `≤1e-3` | 0.0139 | 0.061 | 0.40 |
| `1e-3–3e-3` | 0.0077 | 0.037 | 0.32 |
| `3e-3–5e-3` | 0.0100 | 0.049 | 0.44 |
| `5e-3–1e-2` | 0.0116 | 0.264 | 0.76 |
| `=1e-2` | 0.0061 | 0.103 | 0.61 |

结论：`1e-2` 本身不是充分条件（MOD1/MOD0 在 `1e-2` 稳定），但 MOD3 的
低 KL `0.0018` 触发连续放大到 `1e-2`，下一次更新过大后崩坏。

### 10.3 根因 B：route gate 的 `min_iterations` 使用全局 iteration

代码位置：

- `src/auv_lowlevel/curriculum.py:1329`：
  `eligible = window_full and int(iteration) >= int(min_iterations)`
- `src/auv_lowlevel/curriculum_trainer.py:241`：
  `min_iterations=int(spec.soft_budget_iterations or 0)`
- `src/auv_lowlevel/curriculum_trainer.py:170-177`：
  `_stage_boundary_target(entry) = entry + soft_budget_iterations`

`iteration` 传的是全局 `completed_iterations`，不是 stage-local。因此除 C0 外，
只要全局 iteration 超过当前 stage 的 soft budget，窗口满 20 + 连续 3 次通过
就会晋升。Goal184 MOD4 的实际预算：

| stage | 实际 it | 设计 soft budget | 差 |
|---|---:|---:|---:|
| C0 | 112 | 72 | +40（正常等待 gate） |
| C1 | 22 | 108 | -86 |
| C2 | 24 | 156 | -132 |
| C3 | 60 | 216 | -156 |
| C4 | 36 | 252 | -216 |
| C5 | 84 | 336 | -252 |
| C6 | 317 | 420 | 最后阶段，未再晋升 |

这与 MOD4 的分阶段曲线一一对应：

- C0–C2：任务容易，C1/C2 只训 22/24 it，指标仍好，主要是 BC 老本。
- C3–C4：任务跳到 coupled vertical `2 m × 2 cycles`、深度 80–250 m、
  current 0.3–0.6，但只训 60/36 it，success 掉到 0.84–0.86。
- C5：最难（shear 0.5、outage、vertical `3 m × 1.5`），只训 84 it，
  success 0.749。
- C6：训了 317 it，是 C1–C5 总和的 1.4 倍，所以回升到 0.80。

结论：Goal184 MOD4 的 `655 it` 不是完整课程结果，不能当干净基线。
修成 stage-local 后，总训练量约 `1600–2600 it`，单条约 `15–25 小时`。

### 10.4 根因 C：队列与资源保护

- `--stop-on-failure` 让 MOD0 的 `StageBudgetExhausted` abort 整个队列，
  MOD2 没机会重排。
- `need_mb=22000` 偏保守；MOD3 实际峰值约 19–20 GB，容易被
  `other-growth` 误杀（MOD2 只跑 7 it）。
- daemon 退出后 `queue_state.json` 仍把 MOD3/MOD1/MOD0 标为 `running`。
- 当前无我们的训练进程；GPU1–7 空，GPU0 是 zyt。

### 10.5 待定夺清单

| ID | 项目 | 建议 | 状态 |
|---|---|---|---|
| FIX-1 | `min_iterations` 改 stage-local | 必须；否则消融仍被预算错配扭曲 | `[ ]` 待定夺 |
| FIX-2 | adaptive LR 上限 `1e-2 → 3e-3` | 必须；同时限制每 iteration 放大次数或加 warmup | `[ ]` 待定夺 |
| FIX-3 | 队列 `stop-on-failure` / `need_mb` / state | 必须 | `[ ]` 待定夺 |
| FIX-4 | C0 hard cap `120 → 160` | 建议；当前只剩 8 it 余量 | `[ ]` 待定夺 |
| FIX-5 | MOD3/MOD4 actor 初始化 RNG 配对 | 建议；否则不是严格同初始化 | `[ ]` 待定夺 |
| FIX-6 | `StageBudgetExhausted` 优雅保存 checkpoint | 建议；避免没有 `final.pt` | `[ ]` 待定夺 |
| FIX-7 | 重跑 MOD4 干净基线 | 依赖 FIX-1 | `[ ]` 待定夺 |
| FIX-8 | blocker / LEDGER / 文档 | 已记录 blocker，LEDGER 待定夺后写 | `[x]` 已记录 |

### 10.6 未决问题

1. LR 上限选 `3e-3` 还是 `1e-3`？是否加每 iteration 放大次数限制或 warmup？
2. 是否修 `min_iterations` 并重跑 MOD4？修后 4 条约 25 小时。
3. MOD2/1/0 复用无 current BC 的质量混杂是否接受？
4. C0 阈值 `0.621` 是否继续沿用（不重标定）？
5. 队列资源策略是否调整（单臂失败继续、`need_mb` 降低）？

### 10.7 证据路径

- 队列日志：`/data/yhx/underwater-robotics/runs/auv-local/goal185/20261118-goal185-d7ea65d/logs/queue.log`
- MOD 日志：`/data/yhx/underwater-robotics/runs/auv-local/goal185/20261118-goal185-d7ea65d/full_wave/mod{3,2,1,0}/pe2e_bcres_mod*_seed86201.log`
- 队列状态：`/data/yhx/underwater-robotics/runs/auv-local/goal185/20261118-goal185-d7ea65d/queue_state.json`
- 训练记录：`/data/yhx/underwater-robotics/runs/auv-local/goal185/20261118-goal185-d7ea65d/full_wave/mod*/pe2e_bcres_seed86201/iteration_records.jsonl`
- 冻结阈值：`/data/yhx/underwater-robotics/runs/auv-local/goal184/20261118-goal184-64d8225/route_gate_calibration_pe2e_stage_matched_v4_100it_fixedlr/thresholds/goal182_route_gate_pe2e_20_3.thresholds.json`
- Goal184 MOD4：`/data/yhx/underwater-robotics/runs/auv-local/goal184/20261118-goal184-64d8225/full_wave1/pe2e_bcres_seed86201/iteration_records.jsonl`

### 10.8 Blocker 记录

`research/blockers/2026-11-19-goal185-c0-and-stage-budget.md`
