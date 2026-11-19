# Goal186：PE2E BCRES reward v2（请求-可执行 gap + 垂向权重）

**日期**：2026-11-19<br>
**状态**：`TRAINING_RUNNING`

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
  check_command: python ../scripts/check_research_ledger.py --goal Goal186 --goal-doc doc/goals/goal186_pe2e_bcres_reward_v2_20261119.md --strict
```

## 1. 背景与证据

上游结果：

- Goal184 C5 确定性 BCRES：材料饱和率 `56.4%`，方向舵 `39.0%`、艉翼 `33.2%`。
- 请求每步变化均值 `4.02°/2.50°`，物理上限 `3°/步`。
- 请求与本步可执行量 gap：均值 `1.83°/1.59°`，p95 `9.50°/9.17°`。
- 当前执行器 reward 合计 `<0.4%` 预算；`cmd_speed` 占约 `49.5%`。
- C5 全程垂向平均绝对误差 `1.168 m`，横向 `0.577 m`；垂向 p95 `4.088 m`，
  横向 p95 `1.839 m`。

决策记录：

- `doc/reviews/goal185_reward_config_audit_20261119.md`
- D-1：gap `lambda=5`；D-3：`w_path_vertical=2.5`；
  D-5：`w_sat_pc=0.0` 显式保留，别名置零；
  D-2/D-4/D-6/D-7/D-8：其他项保持不动或显式保留。

为什么现在做：执行器项几乎不参与优化，而 BCRES 的确定性饱和与垂向误差都
不是探索噪声能解释的；需要一次最小 reward 变更验证机制。

本轮用户明确要求把 reward v2 落实到同一次 MOD4–MOD0 队列，而不是只跑
MOD4 单臂。五条 arm 共用同一份 reward v2 env，只保留各自原有的模块开关、
BC checkpoint、seed 和训练协议差异。

## 2. 假设

1. **H1（gap）**：加入 `w_action_feasible_gap_pc=5.0` 后，确定性 gap 与材料
   饱和下降；如果成立，应看到 gap、速率裁剪、请求每步变化下降，而 cross/
   vertical/3D 不出现超过 10% 的退化。若 gap 下降但路线明显变差，H1 失败。
2. **H2（vertical）**：`w_path_vertical=2.5` 后，全程垂向平均绝对误差和 p95
   下降；如果成立，应看到垂向误差下降且 RPM 饱和不显著恶化。若垂向不降、
   RPM 先饱和，H2 失败，瓶颈在控制权限/制导。
3. **H3（falsification）**：若饱和下降但路线、速度或 completion 明显变差，
   说明“减少不可执行请求”与任务跟踪存在强冲突，不能继续加 reward 权重，
   应回到执行器权限/制导诊断。

## 3. 数据与隔离

- 训练数据：现有 stage-matched BC demo 与 BC checkpoint，不重采 BC。
- BC checkpoint：
  `/data/yhx/underwater-robotics/runs/auv-local/goal184/20261118-goal184-64d8225/bc/pe2e_bc_stage_matched_v4_teacher_alos_truth/pc_bc_actor.pt`
  SHA `86cab3fef9ee5784c0d3ef9441b1a409636df45204902713bb2fc950ebb1c086`
- seed：训练 `86201`；评价使用预注册 dev seeds，held-out 关闭。
- 禁止：用评价 trace 调 reward/权重；用 held-out 调参；静默续跑旧 run。
- oracle fallback：禁止。

## 4. 方法

Reward v2 只改三项：

1. 新增 `w_action_feasible_gap_pc = 5.0`：

```text
a_req      = applied.normalized[:2]
a_feasible = applied.total[:2] / action_full_scale[:2]
g          = mean_over_fins( (a_req - a_feasible)^2 )
r_gap      = -5.0 * g
```

2. `w_path_vertical: 1.5 → 2.5`。
3. `w_sat_pc: 0.0` 显式保留；`w_action_clip_attempt_pc: 0.0`（别名置零）。

其他 reward 字段全部保留显式值：

- 速度项 `w_cmd_speed=2.5`、`cmd_speed_std_mps=0.25` 不变；
- 终端奖励全 0；
- 近零平滑/能量项保持原值；
- 路线停滞项保持 0；
- std、LR、课程、架构、残差权限不变。

配置落地：

- 新建 env v2 配置，不直接改旧
  `configs/lowlevel/goal184/env_path_fixed_speed_formal_empirical.json`；
- 新建五份训练配置，分别对应 MOD4/MOD3/MOD2/MOD1/MOD0，全部指向 env v2；
- MOD4 以当前 MOD3 训练配置为模板，只把 `pc.critic_path_dim` 改回 `30`，
  其余算法、BC、课程和模块设置保持一致；
- 队列 manifest 由 `scripts/build_goal186_queue.py` 生成，顺序为
  MOD4 → MOD3 → MOD2 → MOD1 → MOD0；
- 旧 BC checkpoint 与旧 run 的 env SHA 不变。

## 5. 执行计划

### 5.1 不可协商的运行契约

| 项目 | 强制要求 | 失败标签 |
|---|---|---|
| 训练课程 | `curriculum.enabled=true`；冻结 stage mix；逐 episode `curriculum_episodes.jsonl` | `INVALID_NONCURRICULUM` |
| 训练 profile | `PPO-FLAT-4096x48-E5-MB4`；启动前运行 `check_training_profile.py` | `INVALID_PROFILE` |
| 训练速度 | 每个 stage 满足 `speed + current + variation <= 2.5 m/s` | `INVALID_SPEED_CONTRACT` |
| 正式评价 | 完整 `arms × levels × radii(2.0) × zero baselines` | `INVALID_INCOMPLETE_EVAL` |
| 评价 seeds | 预注册 dev seeds；held-out 关闭 | `INVALID_SEED_SPLIT` |
| 启动与结束 | 先协议检查，再实验，最后产物审计；失败非零退出 | `INVALID_PROTOCOL` |
| 主指标 | 全程 cross/vertical/3D、速度误差、分段/keypoint | 不能只看 success |
| 终点口径 | success/timeout/pre_goal_3d 只作 telemetry，不作好坏判据 | — |
| 执行器口径 | gap、材料饱和、RPM、近限位只作 telemetry/风险指标 | — |

### 5.2 步骤

- [x] Step 0：写 env v2 与训练 config；运行协议/profile 检查。
  - `configs/lowlevel/goal186/env_path_fixed_speed_formal_empirical_reward_v2.json`
  - `configs/lowlevel/goal186/train_formal_mod{4,3,2,1,0}_pe2e_bcres_reward_v2_seed86201.json`
  - `scripts/build_goal186_reward_v2_configs.py` 可重复生成；
  - 五份训练 config 的 `check_training_profile.py` 全部 `PASS`。
- [x] Step 1：实现 reward 字段与 `applied.total` 输入管线；补单测。
  - `RewardConfig.w_action_feasible_gap_pc`，默认 `0.0`；
  - reward 只对两片舵计算
    `mean((requested[:2] - feasible[:2])^2)`；
  - env 把 `applied.total / action_full_scale` 作为 `feasible_normalized`
    传入，因此用的是速率限制后的真实可执行指令；
  - 定向测试 `55 passed`；完整回归 `2290 passed, 11 skipped`；
  - 冻结 trace 因新增零值 breakdown 字段更新一次哈希，未改动作、观测或 RNG。
- [x] Step 2：1-it 1536-env contract smoke；核对 manifest、BC 加载、课程 provenance。
  - 2026-11-19 20:41 五臂 `mod4/mod3/mod2/mod1/mod0` 各 1-it 1536-env，全部 `rc=0`；
  - 运行根：
    `/data/yhx/underwater-robotics/runs/auv-local/goal186/20261119-goal186-a6e3b59/smoke/`；
  - 五臂均指向 v2 env（`w_action_feasible_gap_pc=5.0`、
    `w_path_vertical=2.5`、`w_sat_pc=0.0`），课程 `enabled=true`、初始 C0、
    逐 episode provenance 1536 行、route gate `enforce` 且阈值已加载；
  - MOD4 `critic_path_dim=30`、MOD3/1/0 `0`、MOD0 `dade_enabled=false`，
    BC 均为 `86cab3fe...`；
  - 发现并修复 `048fca6` 遗留问题：`learning_rate_max`/
    `learning_rate_warmup_iterations` 被透传给 rsl_rl `PPO.__init__`
    （Goal185 的 smoke 跑在加入 LR guard 之前的配置，正式队列从未启动过）；
    `a6e3b59` 在 `rsl_train_cfg()` 中过滤 runner-only 键，并补回归测试。
- [ ] Step 3：正式训练队列，2630 it，seed86201，MOD4 → MOD3 → MOD2 → MOD1 → MOD0（运行中）。
  - 2026-11-19 20:43 启动，run root
    `/data/yhx/underwater-robotics/runs/auv-local/goal186/20261119-goal186-10360ff`；
  - 队列 `queue.json` 5 条，`need_mb=22000`，`margin=1536`，
    `other_max=4096`；按 GPU 空闲情况动态取卡；
  - 20:43:48 MOD4 上 GPU1、MOD3 上 GPU2，各约 `19.7 GB`，
    显存上限 `23009 MiB`，其他用户进程未受影响；
  - 修复队列自身容量判定：运行中改为
    `total_mb - other_mb >= need_mb`，不再把自身占用算成“其他增长”
    （旧逻辑会在任务启动后误杀自己）。
  - 20:58 停止旧队列进程 `3769887`：该队列会在其他用户显存增长时主动
    SIGINT 自己的任务（MOD2 已被 `other-growth` 挤掉），与“MOD4/MOD3
    不要轻易退出”冲突。MOD4（PID `3769892`，GPU1）与 MOD3（PID
    `3769893`，GPU2）保持原进程继续跑；MOD2 状态 `stopped`，MOD1/MOD0
    `pending`，不再自动抢占。
- [ ] Step 4：完整 `2.0 m` 半径评价矩阵；审计产物。
- [ ] Step 5：写 result md、LEDGER、commit。

### 5.3 运行命令模板

```bash
W=<goal186 worktree>
PY=/data/yhx/underwater-robotics/envs/venvs/isaaclab/bin/python
R=/data/yhx/underwater-robotics/runs/auv-local/goal186/20261119-goal186-<sha>
T=/data/yhx/underwater-robotics/runs/auv-local/goal184/20261118-goal184-64d8225/route_gate_calibration_pe2e_stage_matched_v4_100it_fixedlr/thresholds
CUDA_VISIBLE_DEVICES=<gpu> PYTHONPATH=$W/src:$CORE AUV_GOVERNANCE_ROOT=$GOV $PY $W/scripts/train_batch_lowlevel_ppo.py   --training-config $W/configs/lowlevel/goal186/train_formal_mod4_pe2e_bcres_reward_v2_seed86201.json   --iterations 2630   --curriculum-eval-stage-boundaries   --curriculum-route-gate-mode enforce   --curriculum-route-thresholds $T/goal182_route_gate_pe2e_20_3.thresholds.json   --output $R/train/pe2e_bcres_reward_v2_seed86201   --seed 86201
```

队列只允许使用空闲卡；不得影响其他用户。正式启动前需用户确认。

### 5.4 断点续训与剩余 MOD（2026-11-19）

实现（commit `78bfed7`、`dbfa7bf`、`a26f3f7`、`cbc5171`、`ce39ad9`）：

- checkpoint 额外写入 `completed_iterations`、`total_env_steps`、
  `adaptive_learning_rate`、`num_envs`、`num_steps_per_env`；
- 恢复 policy / optimizer / iter / std / empirical normalization /
  curriculum controller / stage mix ramp / route-gate 20-it 窗口；
- rsl_rl 的 `iter` 是“最后完成的下标”，恢复时 `+1`，避免重跑同一
  iteration；记录文件截断到断点，重放区间不会留下重复行；
- `--resume-latest` 按 mtime 倒序试读，跳过写坏或半截的 `.pt`；
- `--save-interval N` 覆盖 config 的 100 it；新队列用 10 it（约 5.5 min）。

限制：

- IsaacLab 的 episode 状态不入 checkpoint；恢复保证“策略/优化器/课程连续”，
  不是逐 bit 复现；
- 最多重放 `save_interval` 内已完成的 iteration；
- `10360ff` 的 MOD4/MOD3 checkpoint 没有新 metadata：进度按 `iter+1`
  计算，LR 从 `iteration_records.json` 回溯，curriculum 从同目录
  `curriculum_state.json`（该文件每个 iteration 都会写）恢复；
  route-gate 窗口从 0 重新填满 20 it；
- 这两臂若被杀可以续训，最多丢 100 it（约 55 min），不会退回 LR warmup。

验证（GPU6，512 envs × 24 steps，8 GB cap；只验证路径，不作为正式指标）：

- 12 it 正常跑完 → 从 `model_6.pt` 续到 10 it：记录 `1..10` 连续、
  无重复；`resume_start_iteration=7`；LR 从 checkpoint 的 `3e-4`
  接着走；最终 `total_env_steps=122880=10×512×24`；
- 200 it 任务跑到第 30 it 被 `kill -9` → `--resume-latest` 从
  `model_28.pt` 续到 40 it：记录 `1..40` 严格单调、无重复；
  最终 `total_env_steps=491520=40×512×24`；
- 队列级 kill 测试：`--no-preempt --max-restarts 2` 下跑到第 18 it
  被 `kill -9`，队列记录 `exited rc=-9` → `requeued attempt=1/2` →
  `resume ... from latest checkpoint`，恢复后记录 `1..26` 严格单调、
  无重复；
- 定向单测 `68 passed`。

剩余 MOD（MOD2/MOD1/MOD0，默认不自动启动）：

```bash
W=/data/yhx/underwater-robotics/worktrees/auv-local/goal186/20261119-goal186-78bfed7
R=/data/yhx/underwater-robotics/runs/auv-local/goal186/20261119-goal186-resume-ce39ad9
T=/data/yhx/underwater-robotics/runs/auv-local/goal184/20261118-goal184-64d8225/route_gate_calibration_pe2e_stage_matched_v4_100it_fixedlr/thresholds
PY=/data/yhx/underwater-robotics/envs/venvs/isaaclab/bin/python
PYTHONPATH=$W/src:/data/yhx/underwater-robotics/worktrees/auv-core/g04-sync/20261114-g04-sync-0e0bfb1/src \
  $PY $W/scripts/build_goal186_queue.py \
  --worktree $W --run-root $R \
  --thresholds $T/goal182_route_gate_pe2e_20_3.thresholds.json \
  --modules mod2,mod1,mod0 --save-interval 10 \
  --need-mb 22000 --output $R/queue.json

PYTHONPATH=$W/src:/data/yhx/underwater-robotics/worktrees/auv-core/g04-sync/20261114-g04-sync-0e0bfb1/src \
  $PY $W/scripts/goal185_gpu_queue.py \
  --queue $R/queue.json --gpus 0-7 \
  --no-preempt --max-restarts 5 --restart-delay-seconds 120 \
  --margin-mb 1536 --other-max-mb 0 \
  --state $R/queue_state.json --log $R/queue.log
```

- `--no-preempt`：别人显存涨了也不主动杀自己的任务；
- `--max-restarts 5`：只在进程真的退出（含 OOM/被 kill）后，
  用 `--resume-latest` 从最新 checkpoint 重启；
- `--other-max-mb 0`：关闭“其他用户占用超过 4 GB 就不许用卡”，
  改为只要求 `free >= need + margin`，并用 allocator cap 把自己锁在
  启动时可用显存之内；
- 启动前先加 `--dry-run` 检查；当前没有 22 GB 空闲卡，队列会等待，
  不会挤占 MOD4/MOD3。

手动续训模板（必须沿用同一 output/config/seed/env 数）：

```bash
CUDA_VISIBLE_DEVICES=<gpu> PYTHONPATH=$W/src:$CORE \
  $PY $W/scripts/train_batch_lowlevel_ppo.py \
  --training-config $W/configs/lowlevel/goal186/train_formal_mod2_pe2e_bcres_reward_v2_seed86201.json \
  --iterations 2630 --output $R/full_wave/mod2/pe2e_bcres_reward_v2_seed86201 \
  --seed 86201 --resume-latest --save-interval 10 \
  --curriculum-eval-stage-boundaries --curriculum-route-gate-mode enforce \
  --curriculum-route-thresholds $T/goal182_route_gate_pe2e_20_3.thresholds.json
```

## 6. Gate

| check | 阈值 |
|---|---|
| 主指标 | 全程 cross/vertical/3D RMSE 与 p95；与旧 BCRES 同 stage/seed 对照 |
| H1 软判据 | 确定性 gap、材料饱和下降；cross/3D 退化不超过 10% |
| H2 软判据 | 垂向平均绝对误差/p95 下降；RPM >95% 占比不显著上升 |
| 速度 | 速度 RMSE/偏差不恶化超过 0.05 m/s |
| 稳定性 | 0 error / 0 NaN / 0 safety event |
| 终点 telemetry | success/timeout/pre_goal_3d 只记录，不作 Gate |
| 执行器 telemetry | gap、材料饱和、RPM、近限位只记录，不作独立 Gate |

失败裁决：

| 观察 | 判读 | 下一步 |
|---|---|---|
| gap/饱和下降，路线退化 >10% | H1 失败 | 回到执行器权限/制导诊断，不再加 reward |
| 垂向不降、RPM 先饱和 | H2 失败 | 查垂向权限/制导，不加权重 |
| 所有指标持平 | 机制未打中 | 检查 gap 输入与策略均值，不扫 lambda |
| 安全事件出现 | 立即停止 | 按安全流程处理，不继续训练 |

## 7. Git

禁止 `git add -A`。只 add 本次相关文件，提交前运行：

```bash
python ../scripts/check_research_ledger.py --goal Goal186   --goal-doc doc/goals/goal186_pe2e_bcres_reward_v2_20261119.md --strict
git diff --cached --check
```

不自动 commit/push；用户明确要求后才提交。

## 8. Goal 提示词

```text
你现在执行 Goal186：PE2E BCRES reward v2（gap λ=5 + vertical 2.5）。

先读：
- doc/goals/goal186_pe2e_bcres_reward_v2_20261119.md
- doc/reviews/goal185_reward_config_audit_20261119.md
- doc/goals/goal185_pe2e_bcres_module_ablation_20261118.md

硬约束：
1. 只改 reward 三项：新增 gap λ=5、vertical 1.5→2.5、w_sat_pc 显式 0；
2. 不改速度项、终端项、std、LR、课程、架构、残差权限；
3. 不重采 BC；使用现有 BC checkpoint；
4. 新 env v2 配置，不直接改旧 env JSON；
5. 训练必须启用课程学习并保留逐 episode provenance；
6. 评价完整 arms × levels × radii(2.0) × zero baselines；
7. success/timeout/pre_goal_3d 与饱和率只作 telemetry，不作好坏判据；
8. 启动前用户确认；不得影响其他用户的 GPU 任务。

执行顺序：
- 写 env v2 与训练 config；
- 实现 reward gap 输入与单测；
- 1-it smoke；
- 正式训练；
- 完整评价；
- result md、ledger、commit。
```
