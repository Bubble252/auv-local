# Goal182 前置核对单（临时）

> 状态：`TEMPORARY_REVIEW / NOT_GOAL / NOT_GATE / NO_TRAINING`
>
> 生成日期：2026-11-15
>
> 用途：在 Goal182 进入实现和训练前，把 Goal179–Goal182、当前代码、参考工程之间仍需用户拍板的地方逐项核对。确认前不修改 reward、termination、success、goal radius、held-out，也不启动训练。
>
> 本文件不登记到 `doc/goals/INDEX.md` 或 `doc/goals/registry/goals.jsonl`；确认后再把结论回填到正式 Goal 文档。

## 0. 使用方式

- 每项只选一个主选项；需要补充时写在最后的“补充”里。
- 推荐项只是审计建议，不代表已经冻结。
- 回复时直接按编号写，例如：

```text
R1=A
R2a=A
R2b=B
R3=A
R4a=A
R4b=A
R5=B
R6=A
R7=A
R8=A
补充：
```

## 1. 审计基线

| 检查项 | 当前结果 |
|---|---|
| Goal179 matrix checker | `PASS`，`policy=32 / zero=16 / total=48` |
| Goal182 research ledger checker | `PASS`，但目前没有结果 entry |
| `test_goal180_training_profile.py` + `test_lowlevel_ppo_config.py` | `30 passed` |
| 本地两个 `.venv` 的 `rsl_rl` | 都无法 import |
| 参考工程 | `Wheel-Legged-Lab@e61bfe1fb05aac638ba33e41f91b4eddf3c3c1e7` |

参考工程 PPO 配置已核对：

- `num_steps_per_env=48`
- `5 epochs / 4 minibatches`
- `clip_param=0.2`
- `desired_kl=0.01`
- `schedule=adaptive`
- `entropy_coef=0.005`
- `init_std=0.3`
- `clip_actions=1.0`

## 2. 已冻结事项（不要重开）

以下 D1–D18 已由用户确认，本核对单不重新讨论：

- D1：独立 `bc_authority` + 每通道 `ppo_authority`。
- D2：stern-only / rudder-only / 3D 三组，3D 为最终目标。
- D3：保留 PCNormalization，再叠加 empirical actor+critic normalization。
- D4：full-rollout advantage normalization。
- D5：`init_std=0.3`、scalar std、`clip_actions=1.0`。
- D6：`clip=0.20`、`desired_kl=0.01`、adaptive、共享 `lr=3e-4`。
- D7：actor 使用 5D 非特权 path；critic 使用独立 privileged path/actuator/disturbance/sensor-safety。
- D8：training-only 有界续路；整次 run 600–2600 iterations。
- D9：C6 arm early stop 使用 `min_iterations=600`、`max_iterations=2600`、window=20、
  consecutive=3；C0–C6 routine 晋级按 R5-D 使用训练 rollout 的 20/3。
- D10/D17：M 档 MLP `[512,256,128]`；actor/critic 同容量；旧 `[256,128]` 退役。
- D11：重跑 `P0_HISTORICAL`。
- D12：单 run C0–C6，不重置网络和 optimizer。
- D13：route-centric reward + StageGate v1 口径。
- D14：`gamma=0.995`、`lam=0.95`。
- D15：`entropy_coef=0.005`。
- D16：同一 BC 底座、fresh optimizer、iteration=0、final/early-stop checkpoint。
- D18：C6 arm early stop 只在 C6 生效；routine StageGate 在每个 stage 使用独立的 20/3
  window；stage 变化时两者都清空。
- held-out seeds `100–149` 保持关闭。

## 3. 待用户确认项

### R1. Episode 时长、续路段数与路线长度

**冲突**

- Goal182 D8.1 写 `max_episode_steps=3000`、60 s。
- 当前环境配置是 `max_episode_steps=2500`、`dt_s=0.02`，即 50 s：
  `configs/lowlevel/odarrl_control_pc_c3_random_realistic_v0d_learned_speed.json:18`。
- 当前课程路线约 35–50 m，速度约 1.1–1.4 m/s。
- 60 s 理论覆盖约 72–84 m，通常只够 1–2 个完整段。
- 若最多 4 段，路线可能达到 140–200 m，需要约 100–170 s。
- Goal182 里“最多 3–4 段”和“最多 3 次追加，即 4 段”的口径也不完全统一。

**选项**

- [ ] R1-A：保持 60 s，最多追加 1–2 段；或把追加段缩短到 15–20 m。
- [ ] R1-B：保持最多 4 段，延长 episode 总时长，使总时长和路线长度匹配。
- [ ] R1-C：保持 60 s，续路只用于防止终点附近断流，不追求跑完整条延长路线。
- [ ] 其他：________________________________

**建议**：R1-A 或 R1-C。不要同时坚持“60 s”和“4 个完整 35–50 m 段”。

**需要拍板的实质**：续路的目标是“避免 episode 太早结束”，还是“让 agent 一次跑完更长路线”。两者对应的 episode 预算和课程长度不同。

**用户确认（2026-11-15）**：采用 R1-1。保留当前 `max_episode_steps=2500`、50 s；把 Goal182 D8 的 `3000/60 s` 改成 `2500/50 s`。续路只作为训练中的防过早终止机制，不追求跑完 4 个完整 35–50 m 段。

### R2a. 课程预算按 transition 还是 iteration

**现状**

- 当前 `CurriculumStageSpec` 使用 `soft_budget_transitions` / `hard_cap_transitions`：
  `src/auv_lowlevel/curriculum.py:282`。
- 当前 C0–C6 hard cap 合计约 5,000,000 transitions。
- canonical profile 每个 iteration 是 `4096 × 48 = 196,608` transitions。
- 因此 hard cap 只能支持约 `25.4` iterations；soft cap 合计 3,500,000，约 `17.8` iterations。
- Goal182 要求整次 run 600–2600 iterations；若按 196,608 transitions/iteration 计算，2600 iterations 约 `5.11e8` transitions，和当前 5e6 hard cap 相差约 100 倍。

**选项**

- [ ] R2a-A：课程预算改成显式 iteration cap，和 Goal182 的 600–2600 iterations 对齐。
- [ ] R2a-B：保留 transition cap，但在 matrix 和 Goal182 中写清换算后的 iteration，并接受实际训练只有约 18–25 iterations。
- [ ] R2a-C：建立 profile-specific cap：screen 用 transition cap，formal 用 iteration cap。
- [ ] 其他：________________________________

**建议**：R2a-A。当前 transition cap 与 Goal182 的 formal 预算不是同一个量纲，直接混用会让“600–2600 iterations”无法成立。

**用户确认（2026-11-15）**：采用 R2a-A。课程晋级、课程 hard cap 和 early-stop 预算统一按 iteration 计算；transition 只保留为日志、资源审计和异常保护，不再作为课程主预算单位。

### R2b. hard cap 到点后的行为

**现状**

- `observe()` 到 hard cap 时返回 `STAGE_BUDGET_EXHAUSTED`，但 stage 不变、run 不停：
  `src/auv_lowlevel/curriculum.py:985`。
- 结果是训练可能在同一个 stage 无限继续，永远到不了 C6。

**选项**

- [ ] R2b-A：hard cap 到点直接终止该 arm，并记录 `ARM_BUDGET_EXHAUSTED`。
- [ ] R2b-B：hard cap 到点强制进入下一 stage，记录 `HARD_CAP_ADVANCE`，但不记作 Gate pass。
- [ ] R2b-C：只告警，继续当前 stage 到全局 2600 iterations。
- [ ] 其他：________________________________

**建议**：R2b-A 或 R2b-B。禁止保留当前“只记录、不改变状态”的行为。

**影响**：R2b-A 更干净，但会减少有效 formal 样本；R2b-B 能保证课程推进，但必须把“强制推进”和“通过 Gate 推进”严格区分。

**用户确认（2026-11-15）**：采用 R2b-A。某个 stage 的 hard cap 用完后，该 arm 直接终止并记录 `STAGE_BUDGET_EXHAUSTED`；不得继续留在原 stage，也不得把预算耗尽后的推进记作 Gate pass。

### R3. Screen 使用哪个课程 stage

**冲突**

- Goal182 screen 是 `1 seed × 5 iterations`。
- 如果从 C0 开始，5 iterations 很可能只看到早期低扰动路线，无法判断 route 行为。
- formal 仍应跑完整 C0–C6。

**选项**

- [ ] R3-A：screen 固定 C3；formal 跑完整 C0–C6。
- [ ] R3-B：screen 使用上一 Phase winner 的阶段。
- [ ] R3-C：screen 也跑完整课程，但只用于 telemetry，不用于正式判定。
- [ ] 其他：________________________________

**建议**：R3-A。5 iterations 的目的只是检查方向、instrumentation 和 crash，不适合承担课程学习结论。

**用户确认（2026-11-15）**：采用 R3-A。screen 固定从 C3 开始；formal 跑完整 C0–C6。screen 只用于检查崩溃、telemetry 和初步方向，不用于正式结论。

### R4a. canonical BC checkpoint

**现状**

- Goal180 文档建议使用 `pc_v0d_bc0d_20261106/pc_bc_actor.pt`，但本地不存在。
- 本地存在：
  `artifacts/lowlevel/pc_v0d_r2b_weighted_bc_dagger_20261107/bc_satw_dagger2/pc_bc_actor.pt`
- 该文件 SHA-256：
  `acf690eae0829ee3f8fff0ced9613d989dc58f8c8975ff709f89aeece5151ce3`
- `scripts/build_goal180_configs.py:24` 已经实际指向 DAGGER2。
- Goal182 D16 只写“同一 BC 底座”，没有写具体文件。

**选项**

- [ ] R4a-A：DAGGER2 作为统一底座。
- [ ] R4a-B：0D BC 作为 canonical，运行前从服务器恢复并校验 SHA。
- [ ] R4a-C：不同 Goal 使用不同 BC，但不得在同一矩阵中直接比较。
- [ ] 其他：________________________________

**建议（已被 R4b-B 覆盖）**：原建议为 R4a-A，但用户已选择重训 5D BC，因此 R4a 不再
在 DAGGER2 与 0D BC 之间二选一；新 5D BC 成为 canonical 底座，DAGGER2 只作历史对照。

### R4b. actor 5D path 与 frozen BC 8D 的兼容

**冲突**

- Goal182 D7.2 要求 actor 只用 5 列：cross、vertical、tangent heading、tangent pitch、lookahead heading。
- 现有 path schema 只有 8D/12D：
  `src/auv_lowlevel/batch/pc_path.py:24`。
- 现有 frozen BC checkpoint 是 `path_dim=8`，包含 progress、remaining、curvature。
- 训练代码会把前 `bc_path_dim` 列送进 frozen BC encoder：
  `src/auv_lowlevel/batch/pc_policy.py:1002`。
- 直接改 5D 会加载失败；如果保留 8D 但不 masking，actor 仍会间接看到 progress/remaining/curvature。

**选项**

- [ ] R4b-A：保留 8D BC，把 progress/remaining/curvature 三列置零，做 field audit，并比较 BC_ONLY 行为漂移。
- [ ] R4b-B：重新训练 5D BC，彻底消除三个字段。
- [ ] R4b-C：actor 保留 8D，只在 BC branch mask；PPO path branch 使用 5D。
- [ ] 其他：________________________________

**建议**：R4b-A。先以最小改动验证“mask 后是否等价”，如果 BC_ONLY 明显退化，再转 R4b-B 重训 5D BC。

**用户确认（2026-11-16）**：采用 **R4b-B**，不做 8D 置零补丁，直接重训一份
`path_dim=5` 的 BC。新 5D BC 产物作为 Goal182 的 canonical 底座；旧 8D DAGGER2
只作历史对照，不能与新 5D BC 混在同一矩阵中比较。重训时 actor path 只保留：
signed cross-track、signed vertical error、tangent heading error、tangent pitch
error、lookahead heading delta；progress/remaining/curvature 从 schema 中彻底移除。

**用户确认（2026-11-16，数据与执行位置）**：采用 A，即使用完整 DAGGER2 演示数据
`demos_merged_dagger2/demos.npz` 重训，不用本地缩水版；为提速，BC 重训和后续可并行的
ALOS 评估尽量放在 8×4090 服务器执行。执行契约：

- 数据：服务器旧路径
  `/data/yhx/underwater-robotics/src/plan_p1/settled_局部规划器_反应式/artifacts/lowlevel/pc_v0d_r2b_weighted_bc_dagger_20261107/demos_merged_dagger2/demos.npz`，
  期望 SHA-256 `080f4ac3a99ce5f41da119038f0de5a626ea8af1c6b3d9cb7de472d337169d78`；
  启动前先校验，缺失或不匹配则停止，不用本地 `demos_dagger_merged` 替代。
- 代码：先把 5D path schema、训练入口和校验器在本地仓库完成并 commit；服务器从 bare
  repo 建独立 worktree，不原地修改旧部署副本，也不用 `rsync` 覆盖源码。
- 环境：优先复用服务器已有 GPU venv（旧记录为
  `/data/yhx/underwater-robotics/envs/venvs/isaaclab/bin/python`），启动前记录 Python、
  Torch、CUDA 和实际 `git rev-parse HEAD`。
- 输出：训练产物写入服务器
  `/data/yhx/underwater-robotics/runs/auv-local/goal182/<run_id>/bc5d/`；只把
  `training_manifest.json`、`summary`、小体积 `pc_bc_actor.pt` 和评估摘要同步回本地，
  大日志/中间 checkpoint 留在服务器。
- 验收：同一 evaluation protocol 下与旧 8D DAGGER2 做 BC_ONLY paired 对照；至少记录
  cross/vertical/3D RMSE、completion、动作饱和和 safety。5D BC 行为明显漂移时，先修
  数据列映射或训练配置，不直接进入 PPO。

### R4c. 残差优先、微调必做

**用户确认（2026-11-16，覆盖 2026-11-15 的“失败后才微调”）**：保留“先 residual
screen、后 fine-tune”的串行顺序，但完整 PPO 微调组是**必做实验**，不论 residual
结果如何。residual 成功不取消微调；residual 失败也不把微调当作唯一补救。两组仍串行
运行，不并行抢同一批 baseline。

**失败判据（2026-11-16 用户已确认）**

1. **权限失败**：完成独立 authority 校准后，`residual/action_std` 仍无法进入 `[0.20, 0.35]`，或 applied share 仍接近零。
2. **因果失败**：authority 已可见，但 screen/Phase 1 的 route 改善不超过 seed 噪声，且 completion/safety 没有改善。
3. **credit 失败**：BC/PPO 明显抵消，或 reward/update 梯度方向错误。此时先修 credit，不直接转微调。

**不触发“提前切微调/额外归因”的情况**

- 依赖缺失、配置 typo、一次 smoke 崩溃；
- 只有 safety 回归，但 authority 和动作幅度尚未校准；
- 只跑了一个 seed 或 screen 未完成。

**确认口径**：以上三条只决定是否**提前结束 residual 诊断并优先排微调**，不决定微调组
是否存在。微调组 matrix 始终生成；若 residual 失败，记录
`RESIDUAL_AUTHORITY_FAIL`、`RESIDUAL_ROUTE_NO_GAIN` 或 `RESIDUAL_CREDIT_FAIL`，
并在 `research/LEDGER.jsonl` 写入 `pivot`，用于解释顺序和归因。不得因一次 smoke 失败、
配置 typo 或单 seed 波动提前写失败标签。

**微调启动条件（必做）**

- 先完成 residual screen；residual 结果不改变“微调必做”的结论；
- 微调组 matrix 无论 residual 成功或失败都生成；
- 若 residual 失败，再补失败标签和 ledger `pivot`，用于解释优先级；
- 微调组仍使用同一 BC 底座、同一 reward、同一评价协议，只改变 BC 是否冻结和 actor 可训练范围。

### R5. StageGate 评估节奏与 early-stop 数据源

**冲突**

- 训练默认每 100k transitions、40 dev seeds 评估一次：
  `scripts/train_batch_lowlevel_ppo.py:318`。
- 原 Goal182 方案要求 dev seeds 20–27、StageGate 连续 2 次通过；该方案已由 R5-D 覆盖。
- 600–2600 iterations 按默认评估间隔会产出约 1200–5200 次评估，不可接受。
- Goal182 D9.3 没有明确 route/completion/safety/KL/clip 指标到底来自训练 rollout、frozen-dev evaluation，还是两者组合。

**选项**

- [ ] R5-A：每 10 iterations 做 frozen-dev StageGate；8 seeds；C6 early stop 用训练 rollout 的 20-iteration 滑窗。
- [ ] R5-B：每 20 iterations 做 frozen-dev StageGate；其余与 R5-A 相同。
- [ ] R5-C：StageGate 和 early stop 都使用 frozen-dev evaluation。
- [ ] R5-D：照参考工程，routine 晋级用训练 rollout 的 20-iteration 滑窗、连续 3 次通过；frozen-dev 只在 stage 边界和最终 formal 评价使用。
- [ ] 其他：________________________________

**建议**：R5-D。参考工程本身使用训练 rollout 滚动指标晋级，不每 20 iterations 另开
frozen-dev 考试；固定 dev 留给阶段边界和最终 formal，既快又与参考协议一致。

**用户确认（2026-11-16）**：采用 **R5-D**，按参考工程组织。C0–C6 的 routine 晋级
使用训练 rollout 的 20-iteration 滑动平均，连续 3 次全部满足该 stage 的门槛才晋级；
每个 stage 仍保留自己的 min/max iteration。frozen-dev 只在 stage 边界和最终 formal
评价运行，固定 dev seeds `20–27`（8 seeds），不承担每 20 iterations 的 routine 晋级。
C6 的 20/3 early stop 同样使用训练 rollout 滑窗。边界评价发现 safety 回归或协议泄漏时
可以否决，但不能用边界评价结果反向调 routine 阈值。该口径在首次 formal 前冻结。

### R6. Reward 公式与精确权重（参考工程对照版）

**核对口径**

- 参考工程固定为 `Wheel-Legged-Lab@e61bfe1fb05aac638ba33e41f91b4eddf3c3c1e7`。
- 参考工程的 `RewardTermCfg.weight` 不是最终每步贡献；其环境在
  `wheel_legged_flat_env_cfg.py:228` 调用 `reward_manager.compute(dt=self.step_dt)`，
  且 `decimation=2`、`sim.dt=0.005`，所以 `step_dt=0.01`。表中的参考权重最终还会乘
  `0.01`。
- 我方当前在 `pc.py:574` 与 `pc_env/core.py:1396` 对汇总 reward 乘
  `resolved_pc_ppo_scale`，绝大多数历史配置为 `0.01`。它和参考工程的 `step_dt`
  作用位置相似，但两者不是同一个字段，不能据此认定数值语义完全相同。
- 当前代码里 cross/vertical 是 Huber，不是原始 RMSE；`w_path_pitch` 尚不存在。
  `w_cmd_pitch` 的代码默认值是 `0.5`，R6-A 历史示例是 `0.4`，最终目标表 A11 已
  改为 `1.0`；后续实现只认 A11，不再使用 `0.3–0.5` 的旧区间。
- 本轮 R6-A 已确认的形状不变：cross/vertical 保留 Huber δ=0.5；新增路线
  pitch 指数核，`std_pitch=0.5 rad`；ALOS pitch 保留线性。具体权重已由后续
  R6.5 数值筛选覆盖，不再使用 R6-A 的初始示例值。

**参考工程原式（直接摘录）**

```python
# mdp/rewards.py
# 速度跟踪
error_sq = torch.square(cmd_lin_vel_x - base_lin_vel_x)
reward = torch.exp(-error_sq / std**2)
reward *= torch.clamp(-robot.data.projected_gravity_b[:, 2], 0, 0.7) / 0.7

# yaw rate 跟踪
error_sq = torch.square(cmd_ang_vel - base_ang_vel_z)
reward = torch.exp(-error_sq / std**2)
reward *= torch.clamp(-robot.data.projected_gravity_b[:, 2], 0, 0.7) / 0.7

# heading 跟踪；heading_error 已经 wrap_to_pi
heading_error = wrap_to_pi(cmd_heading - yaw_current)
reward = torch.exp(-torch.square(heading_error) / std**2)
reward *= torch.clamp(-robot.data.projected_gravity_b[:, 2], 0, 0.7) / 0.7

# 高度/纵向位置跟踪
error = env.base_height - height_target
reward = torch.exp(-torch.square(error) / std**2)
reward *= torch.clamp(-robot.data.projected_gravity_b[:, 2], 0, 0.7) / 0.7

# 动作二阶差分；只取腿动作 0,1,3,4
diff = a_t - 2 * a_t1 + a_t2
reward = torch.sum(torch.square(diff), dim=1)
```

```python
# wheel_legged_flat_env_cfg.py:927-958
track_lin_vel   = RewardTermCfg(weight=3.0,  std=0.25)
track_ang_vel   = RewardTermCfg(weight=1.0,  std=0.5)
track_heading   = RewardTermCfg(weight=0.25, std=0.5)
base_height     = RewardTermCfg(weight=2.5,  std=0.06)
action_rate     = RewardTermCfg(weight=-0.02)   # 一阶差分 L2
action_smooth   = RewardTermCfg(weight=-0.005)  # 二阶差分 L2
termination     = RewardTermCfg(weight=-100.0)  # 再乘 step_dt，约 -1.0
```

#### R6.1 跟踪类奖励

| 用途 | 我方字段 | 我方当前/拟定公式 | 纯端到端 | ALOS | 参考工程类似项 | 参考原式 | 参考 `weight/std` | 首轮建议 |
|---|---|---|---:|---:|---|---|---|---|
| 横向路线误差 | `w_path_cross` | `-w * Huber(e_cross, 0.5)` | `3.5` | `3.5` | **无精确对应**；只能借用 `track_lin_vel` / `base_height` 的指数核形式 | `exp(-e²/std²)` | `3.0/0.25`、`2.5/0.06` | **已选 3.5**；保留 Huber；本场景 cross p90 约 3.4 m，直接套小 `std` 会饱和 |
| 纵向路线误差 | `w_path_vertical` | `-w * Huber(e_vert, 0.5)` | `1.5` | `1.5` | 最接近 `base_height_reward_simple` 的纵向误差核 | `exp(-(height-height_cmd)²/0.06²)` | `2.5/0.06 m` | 保留 Huber；vertical p90 约 0.64 m，Huber 的线性尾部更适合大误差 |
| 路线切向 heading | `w_yaw` | `-w * (1-exp(-(wrap(tangent_yaw-yaw)/std)²))` | `1.0` | `0` | `track_heading_exp` | `exp(-wrap(cmd-yaw)²/std²)` | `0.25/0.5 rad` | **已确认 H1-E**：使用指数核，`std=0.5 rad`；权重改为 1.0 |
| 路线切向 pitch | `w_path_pitch`（待新增） | 拟定 `-w * (1-exp(-(e_pitch/std_pitch)²))` | `1.0` | `0` | **无精确对应**；借用 `track_heading_exp`/高度跟踪的指数核 | `exp(-e²/std²)` | 无直接权重可抄；模板 `0.25/0.5` | 按 R6-A 新增；`std_pitch=0.5 rad`；权重改为 1.0 |
| ALOS 指令 heading | `w_cmd_yaw` | `-w * (1-exp(-(e_yaw/std)²))` | `0` | `1.0` | `track_heading` + `track_ang_vel` | `exp(-e_yaw²/std²)`；另有 yaw-rate 指数跟踪 | `0.25/0.5 rad`；`1.0/0.5 rad/s` | **已确认 H1-E**：使用指数核，`std=0.5 rad`；权重改为 1.0 |
| ALOS 指令 pitch | `w_cmd_pitch` | `-w * abs(e_pitch)/(π/2)` | `0` | `1.0` | **无直接对应**；只有姿态约束 `flat_orientation_l2` | L2 姿态惩罚 | `-4.0` | 保留线性写法；权重改为 1.0 |
| 速度跟踪 | `w_cmd_speed` | `-w * (1-exp(-((v_ref-v_ground_surge)/std)²))` | `2.5` | `2.5` | `track_lin_vel_xy_exp_wl` | `exp(-(cmd_v-v_x)²/std²)`，再乘重力对齐系数 | `3.0/0.25 m/s` | **已确认 S1-E/S-DEF-G**：使用地速前向分量 `ground_velocity_body[:,0]`、`std=0.25 m/s`、权重 2.5 |
| 路线速度地板（首轮关闭） | `w_route_speed_pc` | 旧的 `-w * max(0, 0.75v_ref-v_route)/2.5` | `0` | `0` | **无直接对应**；参考速度跟踪是双侧跟踪，没有下限保护 | 无 | 无 | **用户已确认 Q1-E**：canonical Goal182 config 固定为 0；guidance 的保底加速保留，但不再同时构造 reward 侧速度地板 |
| 低速停滞占位（首轮不生效） | `w_route_ground_speed_stall`（新增） | `-w * gate(v_ref) * clamp((0.25-v_route)/0.25,0,1)^2` | `0` | `0` | **无直接对应** | 无 | 无 | 接口、breakdown 和 telemetry 必须实现；首轮权重严格为 0；未来若启用，建议从 5.0 开始 |

参考工程的速度、heading、高度都是“命令值存在时，奖励接近命令”；它不是路径几何跟踪。
因此只有 heading、速度两项存在真正可类比的跟踪结构；cross/vertical/pitch 只能借鉴核函数，
不能声称“参考工程已有同一奖励”。

#### R6.2 平滑、约束与能耗

| 用途 | 我方字段 | 我方公式 | 目标值（E2E / ALOS） | 参考类似项 | 参考原式/权重 | 首轮建议 |
|---|---|---|---:|---|---|---|
| 动作一阶差分 | `w_action_pc` | `-w * Σ(a_norm_t-a_norm_{t-1})²` | `0.1 / 0.1` | `action_rate_l2` | `weight=-0.02`，L2 后乘 `dt` | 我方这一项实际对应参考 `action_rate`，不是 `action_smooth`；先保留 0.1 |
| 动作二阶差分（仅舵/艉翼） | `w_action_smooth_pc`（新增） | `-w * Σ_{c∈rudder,stern}(a_norm_t-2a_norm_{t-1}+a_norm_{t-2})²` | `0.005 / 0.005` | `action_smooth_l2` | `weight=-0.005`，只罚腿部关节，不罚轮子 | **用户已确认 Q4-C**：权重 0.005；只罚舵和艉翼，不罚 RPM；`a_norm` 与 `w_action_pc` 使用同一执行器来源 |
| 动作裁剪尝试（首轮关闭） | `w_action_clip_attempt_pc`（legacy `w_sat_pc`） | 未来若启用：`-w * mean(relu(|a_raw|-1)+relu(|Δa_raw|/rate_norm-1))`；必须用裁剪前动作 | `0 / 0` | 无直接对应；`soft_joint_limits_penalty` 最接近“提前约束” | `weight=-10.0`，margin 0.15 rad | **Q3-A：固定 0**；不把裁剪后动作误当成尝试量 |
| 实际材料饱和事件 | `w_actuator_material_sat_pc`（legacy `w_material_sat_pc`） | `-w * 1[actuator material saturated]` | `0.02 / 0.02` | 参考只记录 torque saturation，不把实际裁剪写成奖励 | 无 reward 对应项 | **已选 Q3-A**；权重 0.02；保留二元事件和 telemetry |
| 路线速度突变 | `w_path_speed_smooth` | `-w * ((v_t-v_{t-1})/0.25)²`（Q8-B 归一化） | `0.025 / 0.025` | 无直接对应；概念接近 `action_rate_l2` | `-0.02`（动作空间） | 保留；和参考不同量纲，不能套参考权重 |
| 能耗 | `w_path_energy` | `-w * (rpm/policy_max_rpm)²` | `0.01 / 0.01` | 最接近 `torques` / `dof_vel` 的 L2 | `-1e-4` / `-1e-4` | 保留；归一化动作平方和关节力矩平方不是同一量纲，需 scale audit |
| 横滚角 | `w_phi` | `-w * piecewise(abs(phi); 5°,10°,30°)` | `3.0` | `flat_orientation_l2` | `weight=-4.0`，roll+pitch L2 | **已选 3.0**；保留当前分段式和 deadband，不能把 -4.0 直接抄成同一公式 |
| 横滚角速率 | `w_p` | `-w * piecewise(abs(p); 10°,30°/s)` | `0.02` | `ang_vel_xy_l2` | `weight=-0.05`，x/y 角速度 L2 | **保持 0.02**；本轮只加强横滚角项，不把角速率项一起改成 3.0 |

#### R6.3 明确置零与尺度

| 项 | 纯端到端 | ALOS | 参考工程是否有对应项 | 结论 |
|---|---:|---:|---|---|
| `w_prog` | `0` | `0` | 无 | 删除 progress 奖励；参考没有路线 progress |
| `w_3d` | `0` | `0` | 无 | 3D 只作评价指标 |
| `w_path_3d` | `0` | `0` | 无 | 关闭 ALOS 旧 3D 路径项 |
| `w_path_progress` | `0` | `0` | 无 | 不重新引入路线 progress |
| `route_middle_gate` | `0` | `0` | 无 | 中间段不特殊加权 |
| `goal_bonus` | `0` | `0` | 无 | 参考没有 goal bonus；其 `termination=-100*dt` 也不是到达奖励 |
| `terminal_progress_weight` | `0` | `0` | 无 | 关闭终局 progress |
| `timeout_penalty` | `0` | `0` | 无 | timeout 不作为失败或到达激励 |
| `w_approach_vertical` / `w_approach_pitch` | `0` | `0` | 无 | 首轮关闭终点接近项 |
| `w_vertical_potential` / `w_heave_potential` | `0` | `0` | 无 | 首轮关闭势能塑形 |
| `w_route_speed_pc` | `0` | `0` | 无 | 首轮移除奖励侧速度地板；guidance floor 仍作为控制器先验保留 |
| `w_route_ground_speed_stall` | `0` | `0` | 无 | 只保留低速停滞接口、breakdown 和 telemetry；权重为 0 时不产生 reward |
| `pc_ppo_scale` | `0.01` | `0.01` | **无同名字段**；但 RewardManager 每步乘 `step_dt=0.01` | 首轮保留 0.01；不要和 reward 形状同时改，否则无法区分 100 倍尺度变化 |

#### R6.4 奖励公式拍板记录

| 编号 | 问题 | 选项 | 审计建议 |
|---|---|---|---|
| H1 | 路线切向 heading 是否改成参考式指数核 | H1-E：`-1.0*(1-exp(-(e/0.5)²))` | **用户已确认 H1-E**；纯端到端使用路线切向 heading，ALOS 使用指令 heading；两者都使用 `std=0.5 rad`、权重 1.0 |
| S1 | 速度是否改成参考式指数核 | S1-E：`-2.5*(1-exp(-(e/0.25)²))` | **用户已确认 S1-E**；这里的 speed 指速度跟踪 `w_cmd_speed`，权重 2.5、`std=0.25 m/s`；旧单侧速度地板 `w_route_speed_pc` 已改为 0 |
| Q1-S | 是否保留 reward 侧速度地板 | Q1-S-off：`w_route_speed_pc=0`，guidance 保底加速保留，另设权重为 0 的 stall 接口；Q1-S-on：恢复奖励侧地板 | **用户已确认 Q1-S-off（2026-11-15）**。首轮防停只由 `w_cmd_speed` 和 guidance 保底加速承担；stall 只做定义、接口和遥测，不参与优化 |
| P1 | 路线 pitch 梯度是否按 R6-A 实现 | P1：`-w*(1-exp(-(e/0.5)²))`，最终 `w=1.0`；P1-L：线性 `-w*abs(e)/(π/2)` | 采用 P1 形状；权重按 R6.5 A07 的 `1.0`，R6-A 的 `0.3` 只是历史起点。参考没有 pitch 跟踪，这是借用指数核而不是复刻 |
| A1 | 是否补参考的独立二阶动作平滑项 | A1-0：不加；A1-1：新增 `w_action_smooth_pc=0.005` 二阶差分 | **用户已确认 A1-1，并按 Q4-C 只作用于舵/艉翼**；不罚 RPM。必须记录二阶差分和分通道 telemetry |
| R1 | roll/roll-rate 是否改成参考 L2 | R1-K：保留分段惩罚；R1-L2：`-4.0*roll_pitch_l2 -0.05*ang_vel_xy_l2` | 保留 R1-K；分段惩罚有 deadband，和参考 L2 不可直接比较 |
| Z1 | 所有 progress/3D/goal/timeout/approach/potential 是否全部置 0 | Z1：全部置 0，字段保留 | 建议 Z1；与“减少终点依赖、只跟踪路线”一致 |
| C1 | `pc_ppo_scale` 是否首轮保留 0.01 | C1：保留；C1-off：改成 1.0 | 必须保留 C1；参考虽无同名字段，但其每步也乘 0.01。若此时去掉，会和公式改动混杂 |

#### R6.5 全量奖励数值筛选表

使用方式：先筛 A 组的主目标权重和核宽；B 组只在 A 组方向确认后筛；C 组固定为 0，
不进入筛选。候选档位是首轮建议，不是自动批准的扫参矩阵。

##### A. 主跟踪与核宽

| ID | 参数 | 纯端到端拟值 | ALOS 拟值 | 候选档位 | 说明 |
|---|---|---:|---:|---|---|
| A01 | `w_path_cross` | `3.5` | `3.5` | `2.5 / 3.5 / 4.5` | 横向路线误差主项；**已选 3.5** |
| A02 | `w_path_vertical` | `1.5` | `1.5` | `1.0 / 1.5 / 2.0` | 纵向路线误差主项；已选 1.5 |
| A03 | `path_cross_huber_delta_m` | `0.5` | `0.5` | `0.3 / 0.5 / 0.75 m` | 小于 δ 二次、大于 δ 线性 |
| A04 | `path_vertical_huber_delta_m` | `0.5` | `0.5` | `0.3 / 0.5 / 0.75 m` | 同上 |
| A05 | `w_yaw` | `1.0` | `0` | `0.5 / 1.0 / 1.5` | 纯端到端路线切向 heading；已选 1.0 |
| A06 | `path_heading_std_rad`（新增） | `0.5` | `0.5` | `0.35 / 0.5 / 0.7 rad` | ALOS 权重为 0 时不生效 |
| A07 | `w_path_pitch`（新增） | `1.0` | `0` | `0.5 / 1.0 / 1.5` | 纯端到端路线切向 pitch；已选 1.0 |
| A08 | `path_pitch_std_rad`（新增） | `0.5` | `0.5` | `0.3 / 0.5 / 0.7 rad` | ALOS 权重为 0 时不生效 |
| A09 | `w_cmd_yaw` | `0` | `1.0` | `0.5 / 1.0 / 1.5` | ALOS 指令 heading；已选 1.0 |
| A10 | `cmd_heading_std_rad`（新增） | `0.5` | `0.5` | `0.35 / 0.5 / 0.7 rad` | 纯端到端权重为 0 时不生效 |
| A11 | `w_cmd_pitch` | `0` | `1.0` | `0.5 / 1.0 / 1.5` | ALOS 指令 pitch，保持线性；已选 1.0 |
| A12 | `w_cmd_speed` | `2.5` | `2.5` | `1.5 / 2.5 / 3.5` | 跟踪 `ground_velocity_body[:,0]` 的指数核；已选 2.5 |
| A13 | `cmd_speed_std_mps`（新增） | `0.25` | `0.25` | `0.15 / 0.25 / 0.35 m/s` | `std` 越小，对速度误差越敏感 |
| A14 | `w_route_speed_pc`（legacy） | `0` | `0` | 固定 `0`，不扫参 | 用户已确认移除奖励侧速度地板；Goal182 中非零即配置错误 |
| A15 | `w_route_ground_speed_stall`（新增） | `0` | `0` | 固定 `0`，不扫参 | 仅保留低速停滞接口；未来启用建议权重 `5.0` |
| A16 | `route_ground_speed_stall_threshold_mps` | `0.25` | `0.25` | 固定 `0.25` | `v_route >= 0.25 m/s` 时不罚；权重为 0 时仅记录 |
| A17 | `route_ground_speed_stall_gate_low_mps` | `0.2` | `0.2` | 固定 `0.2` | `v_ref <= 0.2 m/s` 时 gate=0 |
| A18 | `route_ground_speed_stall_gate_high_mps` | `0.4` | `0.4` | 固定 `0.4` | `v_ref >= 0.4 m/s` 时 gate=1 |

##### B. 平滑、能耗与姿态安全

| ID | 参数 | 纯端到端拟值 | ALOS 拟值 | 候选档位 | 说明 |
|---|---|---:|---:|---|---|
| B01 | `w_action_pc` | `0.1` | `0.1` | `0.05 / 0.1 / 0.2` | 一阶动作差分 |
| B02 | `w_action_clip_attempt_pc`（legacy `w_sat_pc`） | `0` | `0` | 首轮固定 `0` | 裁剪前动作尝试项关闭；以后若启用，必须用 raw action，不能再用 `applied.normalized` |
| B03 | `w_actuator_material_sat_pc`（legacy `w_material_sat_pc`） | `0.02` | `0.02` | 首轮固定 `0.02` | 实际执行器材料饱和事件；Q3-A 唯一保留的饱和 reward |
| B04 | `w_path_speed_smooth` | `0.025` | `0.025` | `0.0125 / 0.025 / 0.05` | 路线地速突变；Q8-B 改用 `(Δv/0.25)²` 后按等强度折算 |
| B05 | `w_path_energy` | `0.01` | `0.01` | `0.005 / 0.01 / 0.02` | 归一化推进器能耗 |
| B06 | `w_phi` | `3.0` | `3.0` | 首轮固定 `3.0`；以后才考虑 `1.5 / 3.0 / 4.5` | 横滚角分段惩罚；**已选 3.0** |
| B07 | `w_p` | `0.02` | `0.02` | `0.01 / 0.02 / 0.04` | 横滚角速率分段惩罚；本轮保持 0.02，不随 `w_phi` 一起改 |
| B08 | `w_action_smooth_pc`（新增） | `0.005` | `0.005` | 首轮固定 `0.005` | 二阶动作平滑；只罚舵、艉翼，不罚 RPM |
| B09 | `phi_dead_rad` | `5°` | `5°` | `3° / 5° / 8°` | 安全阈值，建议单独审计，不与 reward 权重同扫 |
| B10 | `phi_soft_rad` | `10°` | `10°` | `8° / 10° / 15°` | 同上 |
| B11 | `phi_strong_rad` | `30°` | `30°` | 固定 `30°` | 同上 |
| B12 | `p_dead_rad_s` | `10°/s` | `10°/s` | `5° / 10° / 15° per s` | 同上 |
| B13 | `p_soft_rad_s` | `30°/s` | `30°/s` | `20° / 30° / 40° per s` | 同上 |

##### C. 固定为 0，不参与筛选

| 参数 | 纯端到端 | ALOS | 处理 |
|---|---:|---:|---|
| `w_prog` | `0` | `0` | progress 奖励关闭 |
| `w_3d` | `0` | `0` | 3D 合成奖励关闭 |
| `w_path_3d` | `0` | `0` | ALOS 旧 3D 路径项关闭 |
| `w_path_progress` | `0` | `0` | 路线 progress 关闭 |
| `route_middle_gate` | `0` | `0` | 中间段不特殊加权 |
| `w_u` | `0` | `0` | legacy 字段；Goal182 固定为 0，速度奖励统一用 `w_cmd_speed` |
| `goal_bonus` | `0` | `0` | 到达奖励关闭 |
| `terminal_progress_weight` | `0` | `0` | 终局 progress 关闭 |
| `timeout_penalty` | `0` | `0` | timeout 不塑造奖励 |
| `w_approach_vertical` | `0` | `0` | 终点纵向接近关闭 |
| `w_approach_pitch` | `0` | `0` | 终点 pitch 接近关闭 |
| `w_vertical_potential` | `0` | `0` | 纵向势能关闭 |

##### C.1 可执行 provenance 要求

- 速度上下限在运行时只能引用 `self.vehicle.min_speed_mps` 和
  `self.vehicle.max_speed_mps`，不得复制 `1.0` / `2.5`。
- ALOS 最终指令、terminal 地速和两种 path 模式共用同一来源；测试也应优先引用变量。
- 新增 `scripts/check_magic_number_provenance.py`、registry 和 baseline；Goal182 的
  速度相关 touched files 不得新增 `MAGIC_NUMBER_WITHOUT_PROVENANCE`。
- 该检查采用 changed-file ratchet；定义点、配置、历史 artifact 和带理由的独立字面量
  可豁免，但 Goal182 速度包络不得用 inline exemption 绕过。
| `w_heave_potential` | `0` | `0` | heave 势能关闭 |

##### D. 全局尺度与不参与本轮筛选的字段

| 参数 | 拟值 | 处理 |
|---|---:|---|
| `pc_ppo_scale` | `0.01` | 首轮固定，不与核函数和权重同扫 |
| `hold_bonus` | `false` | C0/C1 专用；当前主目标不启用 |
| `w_hold` | `5.0`（不生效） | `hold_bonus=false` 时无效 |
| `out_of_bounds_penalty` | 保持现状 | 越界属于安全终止协议，不属于 reward 形状筛选 |
| `termination.safety_terminal_penalty` | 保持现状 `-50.0` | 安全终止，不随 reward 权重筛选 |
| `termination.goal_plane_miss_penalty` / `goal_plane_miss_distance_weight` | **待 Q2 拍板；建议 `0 / 0`** | 现有 config 仍为 `-100 / 10`，和“取消终点到达奖励”冲突；termination 开关可保持不变 |

数值换算提醒：所有项最终乘 `pc_ppo_scale=0.01`。在 `-w*(1-exp(-(e/std)²))`
写法下，误差恰好等于 `std` 时的惩罚是 `0.632w`，不是 `0.5w`。按当前拟值：

| 情形 | 原始单步贡献 | 乘 `0.01` 后 |
|---|---:|---:|
| heading `e=std=0.5 rad`, `w=1.0` | `-0.6321` | `-0.00632` |
| speed `e=std=0.25 m/s`, `w=2.5` | `-1.5803` | `-0.01580` |
| speed `v_ref=1.2`, `v_surge=0` | `-2.5000` | `-0.02500` |
| cross `e=0.5 m`, Huber `δ=0.5`, `w=3.5` | `-0.4375` | `-0.004375` |
| cross `e≈p90 3.4 m`, Huber `δ=0.5`, `w=3.5` | `-5.5125` | `-0.055125` |
| stall `v_ref=1.2`, `v_route=0`, `w=0` | `0` | `0` |

因此首轮低速防停只由速度跟踪和 guidance 保底加速承担；stall 即使完全触发，乘
`pc_ppo_scale` 后的贡献仍严格为 0。A 组筛选不能只看权重，必须同时看误差分布、
项间相关性和贡献占比。

#### R6.6 speed 的定义与待确认项

当前代码里有三个不同的 speed：

| 名称 | 代码来源 | 含义 |
|---|---|---|
| 参考速度 `v_ref` | `observation.command.speed_mps` | 控制器/路线给出的期望速度；可能是地速，取决于 `guidance.compensate_current` |
| 前向速度 `v_surge` | `privileged_surge` | 机体 x 轴速度分量，不是三维速度模长；代码按 `velocity_contract` 选择 ground 或 water-relative |
| 路线速度 `v_route` | `_ground_route_speed()` | 世界地速在路线切线上的投影，再 `clamp(min=0)` |

当前实现的关键点：

- `pc.py:368` 仍是旧的线性 `command_speed`；指数核尚未落地。
- `pc_env/core.py:2260` 在 `velocity_contract='nu_ground'` 时把
  `privileged_surge` 设为 `water_velocity_body[:,0]`；在
  `legacy_water_nu` 时用 `ground_velocity_body[:,0]`。
- 当前 Goal171/Goal173 的 `nu_ground` 训练 config 同时设置
  `guidance.compensate_current=false` 和 `speed_feedback='ground'`，因此 `v_ref`
  是期望地速，而 `w_cmd_speed` 实际拿水体相对前向速度比较。零流时二者等价，
  有流时不等价。这个语义必须先确认。
- `w_route_speed_pc` 已按 Q1 从 Goal182 首轮 reward 中移除；guidance 保底加速仍会改变
  `command_speed_mps`，但该命令不再被复制成 reward 侧速度地板。
- `w_path_speed_smooth` 使用的是 `ground_velocity_body[:,0]`，也不是
  `privileged_surge`。

**用户确认（2026-11-15，S-DEF-G）**：`w_cmd_speed` 明确跟踪
`ground_velocity_body[:,0]`，与期望地速、路线投影速度及
`speed_feedback='ground'` 保持一致；不采用 `water_velocity_body[:,0]`。

实现时必须注意：`privileged_surge` 还被 `speed_error`、hold tracker 和 critic
observation 使用，不能在未确认影响范围时直接把全局变量改掉。建议在 reward 输入中
显式传入独立的 `reward_ground_surge = observation.ground_velocity_body_mps[:,0]`；
若决定让 hold/critic 也统一跟踪地速，则另立一条同步修改记录，不能隐式改变它们的
语义。`w_cmd_speed=2.5` 的公式只有在输入确定为 `reward_ground_surge` 后才实现。

**原 R6-A 选项与确认（历史记录，权重已被 R6.5 覆盖）**

**选项**

- [ ] R6-A：cross/vertical 保持 Huber δ=0.5；新增 `w_path_pitch`，采用参考式指数核 `-w * (1 - exp(-(tangent_pitch_error/std_pitch)²))`，初始 `w=0.3`、`std_pitch=0.5 rad`；ALOS 的 `w_cmd_pitch=0.4`。
- [ ] R6-B：cross/vertical 也改成参考式指数核；先做误差尺度审计再定 `std_cross/std_vertical`。
- [ ] R6-C：cross/vertical 改成 raw absolute 或 squared error。
- [ ] 其他精确公式：________________________________

**建议**：R6-A。参考工程的跟踪项使用 `exp(-error²/std²)`，但本场景 cross p90 约 3.4 m，直接套 `std=0.5` 会让指数项饱和并丢失梯度。因此 cross/vertical 保留 Huber 的线性尾部；pitch 用参考式指数核，写成零误差为 0 的惩罚形式，避免引入常数奖励偏移。

**历史确认（2026-11-15，已被 R6.5 覆盖，不得作为实现值）**：R6-A 最初采用
`w_path_pitch=0.3`、`std_pitch=0.5 rad`、`w_cmd_pitch=0.4`。后续 R6.5 和 Goal182 §4.9
已把这两个权重改成 `1.0 / 1.0`；实现、配置和 telemetry 只认最终目标表，不得再从本段复制
`0.3 / 0.4`。

**历史确认（2026-11-15，H1/S1）**：heading 和 speed 使用指数核。纯端到端 heading 与 ALOS 指令 heading 的权重调整为 `1.0`、`std=0.5 rad`；速度跟踪权重调整为 `2.5`、`std=0.25 m/s`。三者都采用零基线惩罚形式 `-w*(1-exp(-(e/std)²))`。其中“速度下限 `w_route_speed_pc=3.0`”已被后续 Q1-E 覆盖，首轮必须为 `0`。

#### R6.7 参考工程可借鉴点与代码歧义审计

本节只记录代码事实和待拍板项。R6.5 的数值表仍是候选值；下列 Q 项确认前，不实现
reward，也不生成训练 matrix。

**控制模式名词**

| `pc_control_mode` | heading/pitch 来源 | speed 来源与奖励 | 含义 |
|---|---|---|---|
| `alos` | ALOS 求解器输出 `desired_heading/desired_pitch` | ALOS 输出并对 current 做补偿；速度指令与终点地速统一 clamp 到 `[vehicle.min_speed_mps, vehicle.max_speed_mps]`；使用 `w_cmd_yaw`、`w_cmd_pitch`、`w_cmd_speed` | ALOS + 本体控制的模式 |
| `path_fixed_speed` | 直接取路线切向 | episode 固定速度经上下限 clamp；使用 `w_cmd_speed=2.5` 跟踪参考地速 | 纯路径朝向 + 显式固定速度目标 |
| `path_learned_speed` | 直接取路线切向 | 速度参考允许变化，但必须先经同一上下限 clamp；同样使用 `w_cmd_speed=2.5` | **次要速度参考实验**；不能理解成“没有速度命令”或“速度完全不受奖励约束” |

`path_fixed_speed` 与 `path_learned_speed` 都使用新的 `w_cmd_speed=2.5`，旧 `w_u`
在 Goal182 中固定为 0，只作兼容解析。两者差别不在速度 reward，而在速度参考是否变化：
前者保持固定 episode speed，后者允许速度参考在上下限内变化。速度项不受 Q5 的
heading/pitch 互斥规则影响。

`path_learned_speed` 本轮只作为**很次要的速度参考实验**，不进入首轮主矩阵。两种
path 模式都必须在生成 command 和 reward reference 之前应用机器人物理上下限：

```text
v_cmd_episode = _episode_command_speed_mps
v_min = vehicle.min_speed_mps
v_ref = clamp(v_cmd_episode, v_min, vehicle.max_speed_mps)
reference_ground_velocity_ned = v_ref * path_tangent_ned
```

`alos` 模式也使用同一物理包络：`command.speed_mps = clamp(solution.command_speed_mps,
vehicle.min_speed_mps, vehicle.max_speed_mps)`，终点制动后的地速也不低于
`vehicle.min_speed_mps`。因此三个模式都不再使用 `guidance.minimum_speed_ratio`；
该字段在 Goal182 中只保留 legacy 解析，不允许影响速度。该规则只修速度参考，不重新启用
`w_route_speed_pc` 或 stall。若 `v_cmd_episode`
已经位于 `[v_min, vehicle.max_speed_mps]` 内，当前任务速度不变。`w_u` 在 Goal182
必须为 0；若 parser 发现 `w_u` 非零并与 `w_cmd_speed` 同时生效，以
`DUPLICATE_SPEED_TRACKING` 拒绝，避免同一速度误差被两套项重复计权。

##### A. 可以借鉴，但不能直接照搬

| 编号 | 参考工程做法 | 对本项目的正确用法 | 不能照搬的原因 |
|---|---|---|---|
| K1 | tracking 使用 `exp(-e²/std²)` | heading、path pitch、speed 采用零基线惩罚 `-w*(1-exp(-(e/std)²))`；cross/vertical 仍用 Huber | 参考是世界坐标命令跟踪；我方 cross p90 约 3.4 m，小 `std` 指数核会饱和 |
| K2 | 一阶 `action_rate_l2` + 二阶 `action_smooth_l2` | 一阶项保留；二阶项作为独立可选字段，不塞进 `w_action_pc` | 我方动作是舵/艉翼/RPM 的归一化执行器量，参考是关节动作；权重不可直接换算 |
| K3 | RewardManager 逐项记录加权后的贡献 | 新 reward 必须同时记录误差、核值、加权贡献和 `pc_ppo_scale` 后的贡献 | 只记录总 reward 无法判断是误差变大还是权重/尺度变化 |
| K4 | 没有 goal bonus、没有 progress shaping；termination 是平直惩罚 | 支持 Z1：goal/3D/progress/timeout/approach/potential 全部置 0 | 参考的 `termination=-100*dt` 是失败代价，不是“到达终点奖励” |
| K5 | `soft_joint_limits_penalty`、姿态/角速度约束 | 保留我方 roll deadband + roll-rate deadband | 关节限位和横滚安全不是同一物理量；`-10.0`、`-4.0` 不能直接抄 |
| K6 | tracking 奖励乘 gravity-alignment gate | **不要引入** | AUV 的 pitch/roll 不是轮腿机器人的“保持直立”；加入会与下潜/俯仰跟踪冲突 |
| K7 | RewardManager 每步乘 `step_dt=0.01` | 我方继续只乘一次 `pc_ppo_scale`；不要额外再乘 `plant.dt_s` | 两边乘的位置相似但语义不同；叠加会再缩小 50 倍 |
| K8 | 参考同时奖励 heading 与 yaw rate，但 yaw-rate command 是由 heading error 经 `heading_control_stiffness=0.6` 生成（`commonds.py:71-87`） | 我方 ALOS 直接跟踪 heading/pitch 指令；只有未来显式加入 yaw-rate 参考生成器时，才可同时加 yaw-rate 跟踪 | 直接把“heading + yaw rate”当成两个独立奖励会重复塑形；参考的两项是控制器层级关系，不是同一误差抄两遍 |
| K9 | 参考在软关节限位前 `margin` 处开始惩罚，而不是只在硬裁剪后惩罚 | 若未来需要更早抑制饱和，`w_action_clip_attempt_pc` 应使用 raw/clipped 前的尝试量或“接近 envelope 的 margin”量；实际裁剪事件仍由 `w_actuator_material_sat_pc` 记录 | 我方是舵/艉翼/RPM 执行器包络，不是关节位置；margin 必须按各通道物理量纲分别定义 |

##### B. 需要用户拍板的歧义

| 编号 | 代码事实/现象 | 影响 | 选项 | 建议 |
|---|---|---|---|---|
| Q1 | `w_route_speed_pc` 当前用 `observation.command.speed_mps` 算 floor。ALOS 中该命令先经 `_route_speed_floor()` 抬升（`core.py:1201`），随后又有 `minimum_speed_ratio=0.75` 下限（`core.py:1226-1256`）。因此实际 floor 至少是 `0.75 × (0.75 × cruise)=0.5625 × cruise`，还可能更高；奖励和制导在同时抬高同一个速度目标。 | ALOS 的“75% 速度下限”不再是巡航速度的 75%，且可能形成奖励/制导反馈回路，无法区分是 reward 还是 guidance 的贡献。 | Q1-A：奖励 floor 固定用 `episode_command_speed_mps`，制导 floor 保留；Q1-B：继续用实际 command；Q1-C：关闭制导 floor，只留奖励；Q1-D：关闭奖励 floor，只留制导；Q1-E：奖励 floor 置 0，并新增权重为 0 的 stall 接口。 | **用户已确认 Q1-E（2026-11-15）**。首轮 canonical config 固定 `w_route_speed_pc=0`，guidance 的 `_route_speed_floor()` 仍保留；另新增 `w_route_ground_speed_stall=0`。stall 必须实现公式、字段、breakdown 和 telemetry，但权重严格为 0，不进入首轮优化或扫参。`w_cmd_speed` 使用 Q11-A 的补偿前地速参考，是首轮防停主项。**2026-11-16 Q5.2 已覆盖本行中的 `minimum_speed_ratio` 部分；`_route_speed_floor()` 保底加速仍保留。** |
| Q2 | 多个训练/评价 config 仍有 `goal_plane_miss_penalty=-100`、`goal_plane_miss_distance_weight=10`；`pc.py:454-458` 会按 miss 事件和超出 goal radius 的距离累加负奖励，再乘 `0.01`。这与“删除终点到达相关奖励”冲突。 | 即使 `goal_bonus=0`，终点平面 miss 仍可产生约 `-1.0` 以上的事件惩罚；距离大时可远大于普通跟踪项，重新把优化拉向终点。 | Q2-A：训练 reward 两者都设 0，termination 开关保持；Q2-B：保留现有 -100/10；Q2-C：只把 distance weight 设 0。 | **用户已确认 Q2-A（2026-11-15）**。`goal_plane_miss_penalty=0`、`goal_plane_miss_distance_weight=0`；`pc_goal_plane_miss_termination_enabled` 不变，仍用于终止和诊断，但不提供 reward。 |
| Q3 | `w_sat_pc` 的 `requested_normalized` 实际传入 `applied.normalized`，而它已在 `actions.py:243` 被 clamp 到 `[-1,1]`，所以 `magnitude_excess` 通常恒为 0；当前实际只剩“超过速率上限”的连续项。`w_material_sat_pc` 又是实际材料裁剪的二元事件，二者会重叠。 | 表格里同时开 0.05 和 0.1 看起来是两个约束，实际可能只测到同一类饱和，且幅值分支是死代码，审计会误判。 | Q3-A：只保留实际材料饱和奖励；Q3-B：把尝试项改成 raw action 的裁剪前越界；Q3-C：保持现状。 | **用户已确认 Q3-A（2026-11-15）**。canonical 命名改为 `w_action_clip_attempt_pc=0` 和 `w_actuator_material_sat_pc=0.02`；旧名 `w_sat_pc` / `w_material_sat_pc` 仅作兼容别名。首轮不计算尝试项，只保留实际材料饱和事件、breakdown 和 telemetry。 |
| Q4 | 参考有独立二阶动作平滑 `action_smooth=-0.005`；我方只有一阶 `w_action_pc`。 | 二阶项抑制动作抖动，但也会减慢急转弯/深度修正；首轮同时改核函数、权重和二阶项会混因。 | Q4-A：首轮 `w_action_smooth_pc=0`；Q4-B：加 `0.005`，三通道都罚；Q4-C：只罚舵/艉翼，不罚 RPM。 | **用户已确认 Q4-C（2026-11-15）**。`w_action_smooth_pc=0.005`，纯端到端和 ALOS 相同；只计算舵与艉翼通道的二阶差分，RPM 不参与该奖励。 |
| Q5 | 在 `path_learned_speed` / `path_fixed_speed` 下，`command.heading/pitch` 就是路线切向（`core.py:1324-1336`）；此时若 `w_yaw` 和 `w_cmd_yaw`、或 `w_path_pitch` 和 `w_cmd_pitch` 同时非零，会精确重复惩罚同一个误差。 | 同一物理误差被罚两次，权重表失去可解释性。 | Q5-A：matrix/config 检查器强制 path 模式 `w_cmd_yaw=w_cmd_pitch=0`，ALOS `w_yaw=w_path_pitch=0`；Q5-B：只在注释里提醒。 | **用户已确认 Q5-A（2026-11-15）**。parser 和 matrix checker 以 `DUPLICATE_PATH_COMMAND_TRACKING` 拒绝混用；path_* 只允许 path heading/pitch，ALOS 只允许 command heading/pitch；speed 项不受限制。 |
| Q5.1 | path 模式当前 `_path_command()` 直接把 `_episode_command_speed_mps` 当 `speed_mps`，没有物理上下限 clamp；同时旧代码把 `path_fixed_speed` 的速度奖励接在 `w_u` 上。 | 如果 fixed-speed 仍用 `w_u`，主实验会与 ALOS 使用不同速度 reward；learned-speed 则可能把不可执行命令、执行器饱和和速度误差混在一起。 | Q5.1-A：`path_fixed_speed` 与 `path_learned_speed` 都使用 `w_cmd_speed=2.5`，并统一 `v_ref=clamp(v_cmd_episode, vehicle.min_speed_mps, vehicle.max_speed_mps)`；Q5.1-B：fixed 继续用 `w_u`；Q5.1-C：不加物理上下限。 | **用户已确认 Q5.1-A（2026-11-16）**。`path_fixed_speed` 也用 `w_cmd_speed=2.5`；`w_u` 固定为 0，仅作 legacy 解析。两种 path 模式都只用 `vehicle.min_speed_mps` 和 `vehicle.max_speed_mps`，不使用 `guidance.minimum_speed_ratio`。`path_learned_speed` 只允许速度参考在物理上下限内变化，且仍是次要实验；`v_ref` 统一进入 command/reward/telemetry。 |
| Q5.2 | ALOS 当前在 `_alos_command()` 用 `max(vehicle.min_speed_mps, 0.75*cruise)` 抬高最终指令下限，guidance 的 robust terminal 又把终点地速限制在 `0.75*desired_ground_speed` 以上。 | 同一个物理速度下限会随 cruise 变化；即使目标只是防止低于机器人最低速，也会被额外抬高，与 path 模式不一致。 | Q5.2-A：ALOS 最终指令和终点制动都只使用 `vehicle.min_speed_mps`/`vehicle.max_speed_mps`；移除 `minimum_speed_ratio` 对速度的影响；Q5.2-B：ALOS 继续保留 0.75 比例。 | **用户已确认 Q5.2-A（2026-11-16）**。ALOS 与两种 path 模式统一使用 REMUS100 物理包络；`guidance.minimum_speed_ratio` 只作 legacy 解析，不再参与最终指令或 terminal 地速限制。实现时必须移除 `terminal_speed_ratio >= minimum_speed_ratio` 的旧校验依赖。 |
| Q6 | ALOS 的 `w_cmd_yaw/pitch/speed` 跟踪的是 guidance command；guidance 在终点附近会启用 terminal alignment、terminal braking/reference extension（`guidance.py:251-327`）。即使 goal bonus 为 0，这些 command 变化仍会进入 reward。 | 终点阶段仍有隐含 shaping；若训练经常跑到终点，无法声称 reward 与“到达”无关。旧配置 `terminal_alignment=50 m` 还会在 35–50 m 路线上从起点覆盖路径切向。 | Q6-A：接受该耦合，但 D8 训练路线必须保持 pre-goal/续路，matrix 记录 terminal 设置；Q6-B：训练时关闭 terminal braking/alignment；Q6-C：reward 改用纯路线目标（会偏离 ALOS 指令语义，不推荐）。 | **Q6-A**；并已于 2026-11-16 确认 `terminal_alignment_min_m=terminal_alignment_max_m=5.0 m`。前提仍是 D8 路线扩展覆盖训练 episode，终点逻辑只在最后 5 m/评价末尾出现；若训练仍频繁进入 terminal 区，再改 Q6-B。 |
| Q6.1 | 在线路线生成与 ALOS 的消费边界 | 训练 D8 是持续增长 path；部署可能只提供 rolling buffer。 | 若把 buffer 末端当 path endpoint，ALOS 会在每个 buffer 末尾触发 terminal；若路径替换后不重置 projection hint，会投影到错误局部段。 | **已确认（2026-11-16）**：terminal 只认真实 mission goal；rolling buffer 不得自动改变 terminal；同 path 追加保留 hint，替换/重规划必须全路径投影并重置 hint；buffer 前方长度至少覆盖 `max_lookahead_m + terminal margin`。 |
| Q7 | 当前 path cross/vertical 用 `observation.projection`（通常来自导航估计），而 `_ground_route_speed()` 用 `self.eta` 的真值位置/姿态重新投影（`core.py:1410-1424`）。同一个 path reward 混用了 sensor 与 true state。 | 有噪声/洋流时，两个项的路径切线、位置基准可能不同；贡献占比和失败归因会混淆。 | Q7-A：reward 全部改用 true state（cross/vertical 也从 true eta 投影）；Q7-B：reward 全部改用 observation/projection；Q7-C：保留混合并显式标注。 | **用户确认 Q7-A（2026-11-16）**。所有 reward 侧的路径误差统一从 true `eta` 重新投影；actor observation 仍只使用 sensor/projection，不获得 true state。实现时 cross、vertical、route speed、pre-goal 3D 等 path reward 必须共用同一投影入口，不能只改一半。 |
| Q8 | `w_path_speed_smooth` 当前公式是 `(Δv)²/0.25`，不是 `(Δv/0.25)²`；两者相差 4 倍，且前者量纲不是无量纲。 | 权重 0.1 的实际强度依赖这个不成文约定，scale audit 容易误读。 | Q8-A：首轮保留 legacy `Δv²/0.25`，只记录；Q8-B：改成 `(Δv/0.25)²` 并把权重降到 `0.025` 以近似等强；Q8-C：改成 `(Δv/0.25)²` 但保持 0.1（强 4 倍）。 | **用户确认 Q8-B（2026-11-16）**。公式改为 `-w * ((v_t-v_{t-1}) / 0.25)²`，权重改为 `0.025`；在阈值 `0.25 m/s` 处与旧 `0.1 * Δv²/0.25` 等强度。实现、breakdown 和 telemetry 必须记录公式版本，不能与旧 config 的 0.1 混用。 |
| Q9 | `w_path_energy` 只罚螺旋桨 `(rpm/policy_max_rpm)²`，不含舵/艉翼；它是推进能耗代理，不是总执行器能耗。 | 容易被误读为“总能耗”；也无法与参考 `torques`/`dof_vel` 权重类比。 | Q9-A：保留 0.01，明确标注 propeller-only proxy；Q9-B：改成三通道归一化 L2；Q9-C：设 0。 | **用户确认 Q9-A（2026-11-16）**。保留 `w_path_energy=0.01`，canonical 名称和 telemetry 必须标注 propeller-only proxy；舵/艉翼能耗不混入该项。若后续发现策略靠大舵角换精度，另立舵机能耗项。 |
| Q10 | `pc_ppo_scale=0.01` 仍是“暂保留”；`termination.safety_terminal_penalty=-50` 则在表外直接进 reward，再乘 0.01。 | 不显式冻结会在实现时被误当成自由参数；safety 可能和普通 reward 混在一起调。 | Q10-A：`pc_ppo_scale=0.01`、safety `-50` 都冻结；Q10-B：scale 改 1.0；Q10-C：safety 也进入本轮筛选。 | **用户确认 Q10-A（2026-11-16）**。`pc_ppo_scale=0.01` 与 `safety_terminal_penalty=-50` 首轮均冻结，不作为 treatment 或扫参项。safety 只在实际安全终止步骤计一次，乘 scale 后为 `-0.5`；若后续做尺度审计，另立因子，不能与 reward 形状混跑。 |
| Q11 | `guidance.compensate_current=false` 时，`solve(desired_ground_speed_mps=...)` 返回的 `command_speed_mps` 是期望**地速**；为 `true` 时，`_water_velocity()` 返回的是扣除海流后的**水体相对速度模长**（`guidance.py:354-373`），机体 x 轴也随之对齐水相对方向，不再对齐期望地速方向。`core.py:2298` 又把 `observation.command.speed_mps` 直接作为 `w_cmd_speed` 的参考速度。Goal175/182 的 ALOS-COMP 基线使用 `compensate_current=true`，所以这不是只影响旧配置的边缘问题。 | 同一奖励字段在不同 config 下会一边比较“地速目标 vs 地速反馈”，另一边比较“水速目标 vs 地速反馈”。更严重的是，COMP 下即使只把“补偿前地速模长”作为目标，ground surge 也只是期望地速在机体系 x 上的**投影**：迎流时 command=1.5 而真实地速=1.2；侧流时 command≈1.237 而真实 ground surge≈1.164，完美跟踪也会被误罚。 | Q11-A：保留控制用的 `command_speed_mps`，另存**补偿前的期望地面速度向量** `reference_ground_velocity_ned`；该向量来自 guidance 的 `desired_ground_speed * ground_direction`（terminal 时用 `terminal_ground_speed * terminal_ground`），不经过水速转换、也不加 current。`w_cmd_speed` 用它转到机体系后的 x 分量；权重为 0 的 stall 用它计算 `v_ref`。Q11-B：承认 command 是水速并让速度奖励也改用水速反馈。Q11-C：Goal182 首轮禁止 `compensate_current=true`。 | **用户已确认 Q11-A（2026-11-15）**。它保持 S-DEF-G 的“真实地速反馈”，不把水速 command 或 current 估计误差混进速度奖励，并能延续 Goal175/182 的 ALOS-COMP 基线。首轮固定 `current_crab_enabled=false`；crab 会再改变 command 的物理解释，另立实验。 |
| Q12 | 当前 `w_action_pc` 罚的是**实际执行器**变化：`pc.py:394-397` 用 `self.actuator` 与 `previous_actuator` 的差，而实际值已经过幅值限幅和舵/艉翼速率限制（`actions.py:243-277`）。参考 `action_rate_l2` 罚的是策略动作命令，不是执行器响应。 | 实际执行器项衡量物理平滑度，但当速率限制持续生效时，策略继续增大动作也不会继续增大该项；它和“policy action jitter”不是同一个 credit。 | Q12-A：保留实际执行器差分，telemetry 同时记录 requested/actual delta；Q12-B：改成 requested clipped action 的一阶差分，贴近参考；Q12-C：拆成 `w_action_pc`（requested）与 `w_actuator_rate_pc`（actual）两项。 | **用户确认 Q12-A（2026-11-16）**。首轮 `w_action_pc` 继续罚实际执行器差分，保持物理平滑语义；breakdown/telemetry 必须同时记录 raw action delta、clipped-requested delta 和 actual actuator delta。后续若确认抖动主要来自 policy 侧，再另立 requested 项。 |
| Q13 | 参考有 `ang_vel_xy=-0.05`，同时抑制 roll/pitch 角速度；我方只有 roll 角速率 `w_p`，没有 pitch 角速率项。 | 新增 path pitch 和 ALOS pitch 跟踪后，可能出现 pitch 振荡；但直接抑制 pitch rate 会减慢必要的下潜/上浮。 | Q13-A：首轮不加，只记录 `pitch_rate` 和其 p95；Q13-B：加入带 deadband 的 pitch-rate 惩罚（例如 10°/s 以上）；Q13-C：照参考做全 L2，不推荐。 | **用户确认 Q13-A（2026-11-16）**。首轮不新增 pitch-rate 惩罚，只记录 `pitch_rate`、p95 和与指令 pitch 的关系。只有确认振荡不是控制指令本身造成时，才另立 `w_q_track`。 |

##### B.1 当前 termination 审计（2026-11-15）

代码事实来自 `BatchedPCTerminationManager.update()`：
`src/auv_lowlevel/batch/pc.py:803-947`。canonical Goal173/175 训练配置启用了
`pc_goal_plane_miss_termination_enabled=true`，`position_source=eta`，因此下表按
“endpoint-stop 分支”的优先级列出；同一 step 同时满足多个条件时，只有最高优先级
`terminal_kind` 会被记录。

| 优先级 | `PCTerminalKind` | 触发条件（当前 canonical 口径） | `terminated` / `truncated` |
|---:|---|---|---|
| 1 | `GOAL` | `distance(termination_position, goal) <= goal_radius_m`；Goal182 只使用 default `2.0 m` | `terminated=True` |
| 2 | `GOAL_PLANE_MISS` | 启用平面终止，且越过终点切平面：`dot(pos-goal, goal_tangent) >= 0`，同时 `distance > goal_plane_miss_guard_radius_m`；默认 guard=goal radius，dwell=0，因此半径外越面立即终止 | `terminated=True` |
| 3 | `NAN_OR_INF` | `eta` 或 `nu` 任一元素非有限 | `terminated=True`，且 `safety_event=True` |
| 4 | `INVALID_STATE` | 状态有限，但 `abs(pitch) >= pi/2 - 1e-3`，即接近垂直姿态 | `terminated=True`，且 `safety_event=True` |
| 5 | `OUT_OF_BOUNDS` | 真实 `eta` 位置超出 `bounds_m`：x/y `[-200,200]`，Goal182 z 上界放宽到 `520` | `terminated=True`，且 `safety_event=True` |
| 6 | `DEPTH_LIMIT` | 真实 z `<0` 或 `>510`；Goal182 把 z bounds 上界设为 `520`，因此 `z>510` 会先记录为 `DEPTH_LIMIT`，不会被 `OUT_OF_BOUNDS` 遮蔽 | `terminated=True`，且 `safety_event=True` |
| 7 | `SAFETY_ROLL_ANGLE` | `abs(roll) >= 45 deg` 连续 `1.0 s` | `terminated=True`，且 `safety_event=True` |
| 8 | `SAFETY_ROLL_RATE` | `abs(roll_rate) >= 90 deg/s` 连续 `0.5 s` | `terminated=True`，且 `safety_event=True` |
| 9 | `TIME_LIMIT` | `step_count >= max_episode_steps=2500`，即 `dt=0.02 s` 下 `50 s` | `truncated=True`，`terminated=False` |

补充结论：

- Goal182 不再使用 `strict125`。canonical 训练和评价只保留 default `2.0 m` goal
  radius；训练 config 和 Goal175 eval 没有显式覆盖 guard radius 或 dwell，默认 guard
  radius 等于 `plant.goal_radius_m=2.0 m`，dwell 等于 `0`。
- `safety_event` 只在最终选中的 kind 属于安全集合时为真。例如同一 step 同时越过终点
  平面并出现 roll 超限，`GOAL_PLANE_MISS` 的优先级更高，该 step 不会记作 safety event。
- `success` 不是独立的终止原因；评价里的 success 是从 `GOAL` 终止和 evaluator 的
  tight-radius/frame 规则派生出来的标签，不会反过来改变环境终止。
- 材料饱和、动作幅值/速率裁剪、路径停滞、DVL/USBL 短时无效、completion 未达阈值都
  **不会**直接终止 episode；它们目前只进入 reward、诊断或 StageGate 候选指标。
- `out_of_bounds_penalty`、`safety_terminal_penalty`、`goal_plane_miss_penalty` 都是
  reward 配置，不会自行触发终止；真正触发终止的是上表的 kind 判定。
- Q2-A 只把 goal-plane miss 的 reward 置 0，不改变其终止。因此首轮训练中“半径外越过
  终点平面仍会结束 episode”这一行为保留；若后续认为它会扭曲续路或 iteration 统计，
  必须另立 termination 决策，不能通过 reward 权重隐式关闭。

##### C. 不需要用户二选一，但实现时必须修的代码缺口

- 在 `RewardConfig`、`_parse_reward()`、`PCRewardBreakdown`、step info 和 runner 中新增：
  `w_path_pitch`、`path_pitch_std_rad`、`path_heading_std_rad`、`cmd_heading_std_rad`、
  `cmd_speed_std_mps`、`w_route_ground_speed_stall`、
  `route_ground_speed_stall_threshold_mps`、`route_ground_speed_stall_gate_low_mps`、
  `route_ground_speed_stall_gate_high_mps`；std 和 stall 阈值不得写死在 reward 函数。
- `w_route_speed_pc` 保留为 legacy 解析字段，但 Goal182 canonical config 必须为 `0`；
  启动器发现非零值时直接报错，不静默把它解释成新的 stall 权重。
- stall 的 breakdown 和 telemetry 必须至少记录 `reference_ground_speed_mps`、
  `route_ground_speed_mps`、`stall_gate`、`stall_ratio`、raw penalty 和乘
  `pc_ppo_scale` 后的贡献；首轮后三项中的加权贡献必须恒为 0。
- `_path_reward()` 中旧 `w_yaw` 仍为线性 `abs(e)/π`，`w_cmd_yaw`/`w_cmd_pitch`/
  `w_cmd_speed` 仍是线性；需改为 R6.5 已确认的核函数。
- path pitch 误差当前不存在；新增时建议与 heading 一样使用 `wrap_angle()`，避免跨 `±π`
  的非物理跳变。
- `pc_env/core.py:2260-2264` 的 `privileged_surge` 在 `nu_ground` 下取
  `water_velocity_body_mps[:,0]`，与已确认的 S-DEF-G（reward 跟踪
  `ground_velocity_body_mps[:,0]`）冲突；实现时必须改成独立的
  `reward_ground_surge`，不能继续复用这个变量。
- `w_action_smooth_pc` 尚未实现；需要保存至少两步历史执行器（当前、上一步、上上步），
  在 episode reset 时清零，并且只对舵/艉翼通道计算二阶差分。RPM 必须显式排除。breakdown
  和 telemetry 至少记录分通道二阶差分、总 raw penalty 和 `pc_ppo_scale` 后贡献。
- `route_health_active` 目前只看 `w_path_cross/w_path_vertical/w_path_progress`；应把
  `w_path_pitch`、`w_path_speed_smooth`、`w_path_energy` 等 path-only 项纳入激活条件，
  否则非零项可能静默不生效。
- `BatchedPCRewardComputer` 的 class docstring 仍声称“no path-error shaping”，与
  `pc_path_track`/route-health 现状冲突；实现前更新，避免后来者按旧契约改代码。
- reward breakdown 现在混合了“误差字段”（如 `path_heading`）和“加权贡献字段”
  （如 `command_yaw`）；新增项要明确命名，并同时记录 raw 与 `pc_ppo_scale` 后贡献。
- 训练入口应拒绝“path 模式同时开 command heading/pitch”与“ALOS 同时开 path
  heading/pitch”的配置，至少输出显式 error，而不是运行时静默相加。
- step info 目前记录 `ground_surge_mps`，但没有分别记录 reward 参考地速、command
  水速、requested/raw/actual 动作差分；Q11/Q12 落地前必须补齐这些 telemetry，否则无法
  判断语义是否真的统一。

##### D. 文档修正状态（2026-11-15 复核）

下表区分“文档已经改”“只是建议”“代码仍未改”。不要把目标表当成当前 config 的实际值。

| 项目 | 文档当前口径 | 正式 Goal182 | 当前代码/config | 状态 |
|---|---|---|---|---|
| 主权重 | cross `3.5`、vertical `1.5`；E2E yaw/pitch `1.0`；ALOS cmd yaw/pitch `1.0`；E2E/ALOS speed `w_cmd_speed=2.5`（learned-speed 次要组同值）；roll angle `3.0`、roll rate `0.02`；route floor `0`；stall `0` | §4.9 已回填本轮确认值 | 绝大多数 config 尚未更新 | **文档已改，代码/config 未改** |
| 评价半径 | 只保留 default `2.0 m`；`strict125` 不再进入 Goal182 | §5.1 已收敛 | 旧历史 config/matrix 可能仍含 `1.25`，但不得进入 Goal182 active protocol | **已确认，待矩阵实现** |
| 深度终止 | `z > 510 m` 触发 `DEPTH_LIMIT`；z bounds 上界设为 `520 m`，避免先被 `OUT_OF_BOUNDS` 覆盖 | 本轮确认；核对单 B.1 已记录 | 当前 canonical config 仍是 z 上界 `300`、`max_depth_m=300` | **已确认，代码/config 未改** |
| E2E path pitch | `w_path_pitch=1.0`，`std=0.5 rad` | §4.9 已写入 | `RewardConfig` 无字段，`_path_reward()` 无该项 | **文档已改，代码未实现** |
| heading 核 | 零基线指数核 `-w*(1-exp(-(e/std)^2))` | §4.9 已写入 | `pc.py:366`、`pc.py:1368-1371` 仍是线性 | **文档已改，代码未实现** |
| speed 核 | 零基线指数核，`std=0.25 m/s` | §4.9 已写入 | `pc.py:368` 仍是线性；reward 速度源仍有 `privileged_surge` 语义问题 | **文档已改，代码未实现** |
| reward 速度源 | `ground_velocity_body_mps[:,0]` | §4.9 与 S-DEF-G 已写入 | `nu_ground` 下仍可能取 water-relative | **已确认，代码未实现** |
| 指数核数值说明 | `e=std` 时惩罚为 `0.632w`，不是 `0.5w` | 已在核对单修正 | 无代码影响 | **已修正** |
| goal-plane miss | 训练 reward `0 / 0`，termination 开关保留 | §4.9 已回填 | 41 个 config 仍启用 `pc_goal_plane_miss_termination_enabled=true`，20 个仍写 `-100`；`pc.py:454-458` 仍会累加 | **用户已确认，待实现** |
| Q11-A | 补偿前地速参考向量，控制 command 不复用 | 待回填正式 Goal | 字段/投影/telemetry 未实现 | **用户已确认，待实现** |
| Q1 | 移除 reward 侧速度地板；新增 stall 接口，权重 `0` | §4.9 已回填 | `RewardConfig` 无 stall 字段；旧 `w_route_speed_pc` 仍存在 | **用户已确认，待实现** |
| Q3 | 只保留实际材料饱和；尝试项权重 0，并改用清晰命名 | §4.9 已回填 | 旧字段名仍存在，代码未改 | **用户已确认 Q3-A，待实现** |
| Q4 | 新增二阶平滑 `0.005`，只罚舵/艉翼 | §4.9 已回填 | 字段和两步历史未实现 | **用户已确认 Q4-C，待实现** |
| Q5 | 用可执行检查禁止 path/ALOS heading/pitch 重复计权 | §4.9 已回填 | parser/matrix checker 未实现 | **用户已确认 Q5-A，待实现** |
| Q5.1 | path 模式统一 `w_cmd_speed=2.5` 与上下限 clamp；learned speed 仅次要 | §4.9 已回填 | `_path_command()` 仍直接用原始 episode command，fixed 仍走 `w_u` | **用户已确认，待实现** |
| Q5.2 | ALOS 也统一物理速度上下限，停用 `minimum_speed_ratio` | §4.9 已回填 | `_alos_command()` 仍用 `0.75*cruise`；guidance terminal 仍用 ratio | **用户已确认，待实现** |
| Magic-number guard | 速度上下限必须是变量，新增 provenance checker、registry 和 ratchet baseline | §4.9 已回填；治理契约见 `总协调/doc/governance/MAGIC_NUMBER_PROVENANCE.md` | checker 尚未实现 | **用户已确认，待实现** |
| Q6–Q10、Q12–Q13 | 已逐项确认并写入核对单/决策日志 | 待回填正式 Goal 的剩余细节 | 未实现 | **已拍板，待实现** |
| `pc_ppo_scale` | `0.01` 冻结；safety `-50` 同时冻结 | §4.9 与 Q10-A | 多数 config 为 `0.01` | **已冻结，不做首轮扫参** |

> 结论：文档层面已经统一了最终目标值和指数核数学，但“确认”不等于“已实现”。在 Q1–Q13
> 拍板并完成字段、parser、breakdown、telemetry 和 config guard 之前，不能声称 reward
> 重构完成，也不能启动 formal matrix。

##### E. Q11-A 实现口径（已确认）

核心原则是“控制用一份，评价用另一份”，字段不要复用：

| 数据 | 来源 | 用途 |
|---|---|---|
| `command_speed_mps` | `_water_velocity()` 之后、经过 guidance 速率限制 | 仍作为 ALOS 水相对速度指令，供 actor/controller 跟踪 |
| `reference_ground_velocity_ned` | `desired_ground_speed * ground_direction`；terminal 时用 `terminal_ground_speed * terminal_ground` | reward 的任务地速参考，不经过 current 补偿 |
| `ground_velocity_body_mps[:,0]` | 真实状态由 `eta/nu/current` 计算 | `w_cmd_speed` 的实际反馈 |
| `reference_ground_surge_mps` | `world_to_body(reference_ground_velocity_ned, attitude)[:,0]` | 直接作为 `w_cmd_speed` 的目标值 |

实现步骤：

1. 在 `BatchedAlosSolution` 增加可选字段 `reference_ground_velocity_ned`。在 `solve()` 里先保存
   路径/terminal 的**地面**目标，再调用 `_water_velocity()` 做水相对转换；不要在转换后用
   `command_speed + current` 反推。
2. `pc_env/observation.py` 在构造 observation 时携带该向量；path/direct 模式没有 ALOS
   solution 时，用 `command.speed_mps * command_direction` 构造同样的地速参考。
3. 在 `core.py` 计算 `reference_ground_surge_mps` 并传给 `BatchedPCRewardComputer`；
   `w_cmd_speed` 的误差改成：

   ```text
   err_speed = reference_ground_surge_mps - ground_velocity_body_mps[:, 0]
   reward = -2.5 * (1 - exp(-(err_speed / 0.25)^2))
   ```

4. Goal182 首轮不再使用奖励侧速度地板：`w_route_speed_pc=0`，且该字段标为 legacy，
   不允许把 `command.speed_mps` 重新接到 reward floor。`reference_ground_velocity_ned`
   仍用于 `w_cmd_speed`，并作为权重为 0 的 stall 接口的 `v_ref` 来源。
5. telemetry 同时记录 `command_speed_mps`、`water_surge_mps`、`reference_ground_surge_mps`
   和 `ground_surge_mps`。验收测试要覆盖迎流和侧流两组：当真实地速等于期望地速时，
   `w_cmd_speed` 必须为 0 误差，而不是因 COMP 产生假误差。

该实现不改变 actor 的 command 输入、critic 的 `speed_error` 或 hold tracker；这些若也要
改成地速口径，必须另立决策，不能在 Q11 中隐式改掉。

##### F. Q1-E 低速停滞接口（已确认，首轮权重 0）

首轮目标是“保留可审计的 anti-stall 接口，但不让第二套速度地板与 `w_cmd_speed` 重复塑形”。
因此字段、公式、breakdown 和 telemetry 必须实现，`weight` 严格为 `0`。

定义：

```text
v_ref = ||reference_ground_velocity_ned||
v_route = max(0, dot(ground_velocity_ned, path_tangent_ned))

t = clamp(
    (v_ref - route_ground_speed_stall_gate_low_mps)
    / (route_ground_speed_stall_gate_high_mps
       - route_ground_speed_stall_gate_low_mps),
    0, 1)
gate = t^2 * (3 - 2*t)                       # smoothstep；slow ref 不罚

stall_ratio = clamp(
    (route_ground_speed_stall_threshold_mps - v_route)
    / route_ground_speed_stall_threshold_mps,
    0, 1)

stall_penalty = -w_route_ground_speed_stall * gate * stall_ratio^2
```

| 字段 | 首轮值 | 语义 |
|---|---:|---|
| `w_route_ground_speed_stall` | `0.0` | 首轮不生效；未来启用建议从 `5.0` 开始 |
| `route_ground_speed_stall_threshold_mps` | `0.25` | `v_route >= 0.25 m/s` 时不罚 |
| `route_ground_speed_stall_gate_low_mps` | `0.2` | `v_ref <= 0.2 m/s` 时 gate=0 |
| `route_ground_speed_stall_gate_high_mps` | `0.4` | `v_ref >= 0.4 m/s` 时 gate=1 |
| `w_route_speed_pc` | `0.0` | legacy；Goal182 中非零即配置错误 |

验收要求：

1. `weight=0` 时，stall 的 raw penalty 可以随状态变化，但乘 `pc_ppo_scale` 后的贡献
   必须恒为 0；总 reward 不得出现可观测偏移。
2. 权重置非零时，`v_ref<0.2`、`v_route=0` 的样本必须得到 0；`v_ref>0.4`、
   `v_route=0` 的样本必须得到完整 `-w`；`v_route>=0.25` 必须得到 0。
3. 不允许把 stall 实现成 `w_route_speed_pc` 的别名，也不允许用旧 config 的 `0.75`
   比例隐式启用。
4. 首轮矩阵不增加 stall 扫参臂；它只验证接口和 telemetry，不作为 treatment。

**R6.7 回复模板**

```text
Q2=
Q3=
Q4=
Q5=
Q6=
Q7=
Q8=
Q9=
Q10=
Q11=
Q12=
Q13=
补充：
```

### R7. Completion 的分母

**现状**

- evaluator 使用 `final_progress_m / full_path_length`：
  `src/auv_lowlevel/cli/evaluate_pc_ppo/episodes.py:191`。
- 训练续路后 full path 变长，completion 会失真。
- Goal182 StageGate 没有定义 completion 到底对应“初始路线”还是“延长后的总路线”。

**选项**

- [ ] R7-A：completion 分母固定为 episode 初始路线长度；completion 退出 StageGate，只作为 summary/diagnostic。
- [ ] R7-B：StageGate 用当前 active segment completion。
- [ ] R7-C：StageGate 用 total extended route completion。
- [ ] 其他：________________________________

**建议**：R7-A。分母固定为初始路线长度（例如 40 m，不随续路变化）；StageGate 不再读取 completion。为避免策略原地不动，另设轻量 progress liveness 检查，仅判定 `STALLED`，不设置 completion 通过阈值。

**用户确认（2026-11-15）**：采用 R7-A。`completion_ratio = final_progress / initial_route_length`；该比例只写 summary/diagnostic，不进入 StageGate。StageGate 使用 safety + cross/vertical + pre-goal 3D；另加 progress liveness 诊断，防止原地不动，但不以 completion 作为通过门槛。

### R8. Goal180/181 与 Goal182 canonical stack 的边界

**冲突**

- Goal182 §4.9 声明自己是唯一 reward 目标表。
- Goal180 仍定义旧 `R0–R4`，包括 `w_prog=2.0`、`w_3d=1.0`、`w_path_progress=0.05`。
- Goal181 仍引用 Goal180 locked reward。
- Goal180/181 都仍写 formal 16 iterations。
- Goal181 仍使用 8D path 契约。

**选项**

- [ ] R8-A：Goal180 降级为 latent probe/纵向诊断和历史 reward；Goal181 直接消费 Goal182 的 5D actor、reward、PPO profile 和 600–2600 预算。
- [ ] R8-B：重写 Goal180/181，并在新 stack 下全部重跑。
- [ ] R8-C：三套 Goal 并行，但明确不能共享 baseline。
- [ ] 其他：________________________________

**建议**：R8-A。Goal182 已经是当前主栈；保留 Goal180/181 的历史价值，但避免三套 reward、两套 path schema 和两套 iteration 预算同时生效。

**用户确认（2026-11-16）**：采用 **R8-A**。Goal182 是唯一 canonical 主栈。Goal180
降级为 latent probe、纵向诊断和历史 reward 记录，不再定义正式奖励或 canonical matrix；
其 path-conditioned 消融如继续，必须复用 Goal182 的 5D actor、reward、PPO profile 和
预算。Goal181 只保留 actor 架构轴，直接消费 Goal182 的 5D actor、reward、PPO profile
和 600–2600 iteration 预算；Goal182 profile 锁定前不启动 Goal181 formal。旧
Goal180/181 的 8D path、16-iteration 结果只作历史参考，不能与新栈结果直接比较。

### R9. 先做 ALOS guidance 快速预实验

**用户决定（2026-11-16）**：PPO/reward 的修改先往后排，先用 ALOS-only 评价快速筛选
guidance。`horizontal_lookahead_m=1.5`、`vertical_lookahead_m=1.5`、
`horizontal_gain=2.5`、`vertical_gain=2.5`。本轮不训练，只做几组短评价。

**为什么先做**

- 现在 PPO 和 reward 还没有改完；如果 ALOS 本身的 lookahead/gain 明显不合适，后面的
  residual 诊断会把 guidance 问题误判成策略问题。
- 这是 eval-only 单因子筛选，不需要等 PPO authority、reward 实现或 formal matrix。
- 先确认纯 ALOS 的横向/纵向跟踪上限，后续 PPO 才有可信的对照基线。

**候选 arm**

| arm | horizontal lookahead | vertical lookahead | horizontal gain | vertical gain | 用途 |
|---|---:|---:|---:|---:|---|
| `A0_BASE` | `3.0 m` | `2.0 m` | `2.5` | `1.0` | 旧基线 |
| `A1_REQUESTED` | `1.5 m` | `1.5 m` | `2.5` | `2.5` | 用户指定组合 |
| `A2_LOOKAHEAD_ONLY` | `1.5 m` | `1.5 m` | `2.5` | `1.0` | 只看 lookahead |
| `A3_VERTICAL_GAIN_ONLY` | `3.0 m` | `2.0 m` | `2.5` | `2.5` | 只看纵向 gain |
| `A4_LOW_LOOKAHEAD` | `1.0 m` | `1.0 m` | `2.5` | `2.5` | lookahead 下边界 |
| `A5_HIGH_LOOKAHEAD` | `2.0 m` | `2.0 m` | `2.5` | `2.5` | lookahead 中间点 |

今晚跑 C3/C4 的 `A0–A5` 六组，每组固定 4 seeds（`20–23`，seed 不再扩），两档合计
48 个 seed-episode；时间不够先删 C4 的 `A4/A5`。所有 arm 固定
`terminal_alignment_min_m=terminal_alignment_max_m=5.0`，其余 guidance 参数不变。

**执行边界**

- 入口：`scripts/evaluate_pc_ppo.py --action-mode alos`。
- 必须 `--disable-curriculum-direct-command`；否则低 stage 可能不是 ALOS 在控制。
- guidance 用 `--guidance-config` 整块替换，不改默认 config，不改运行时代码。
- 同 seeds、同固定路线、同 current、同 stage、同 checkpoint；只换 guidance 四元组。
- checkpoint 必须与 training/environment config 和 stage 匹配；C0 smoke checkpoint
  只能验证机制，不能代表 C3/C4。
- 本轮不生成 formal matrix，不启动训练，不提交 artifacts。

**判读**

- 主看 `path_horizontal_rmse_m`、`path_vertical_rmse_m`、`path_three_d_rmse_m`；
  辅助看 pre-terminal 3D、completion、terminal 占比、speed error、clip/saturation、
  timeout。
- 用逐 seed paired delta、均值和 median；4 seeds 下 CI 只作诊断，不宣称显著。
  方向不一致或差异落在 seed 噪声内就写“无证据”，不扩矩阵。
- 只有横向/纵向/3D 至少一项稳定改善且没有明显安全退化，才值得把该 profile 作为后续
  PPO 的 ALOS 基线；否则先改 ALOS 消费路径或 terminal，不调 PPO。

## 4. 已发现实现缺口（确认 R 项后再处理）

### Goal182

- StageGate 仍是 success/timeout/plane-miss 驱动：`src/auv_lowlevel/curriculum.py:229`。
- `parse_eval_summary()` 没有 cross/vertical RMSE、pre-goal 3D、completion 字段。
- `w_path_pitch` 尚未进入 `RewardConfig`、parser、breakdown 和 telemetry。
- heading/speed 的指数核尚未实现；需新增 `path_heading_std_rad`、`cmd_heading_std_rad`、
  `cmd_speed_std_mps` 三个字段及其 parser、breakdown 和 telemetry。
- runner 不允许 `clip_actions`：`src/auv_lowlevel/ppo_config.py:839`。
- `empirical_normalization` 是全局开关；N2 critic-only 前必须先拆分。
- Goal182 对应的 runner、auditor、matrix、protocol checker 都还不存在。
- training-only route extension 尚未实现。
- 逐 iteration 的 route/completion/safety/KL/clip telemetry 尚未接入。
- magic-number provenance checker、registry 和 baseline 尚未实现；速度上下限必须先按
  变量引用方式实现，不能被 baseline grandfather。

### Goal179

- `bc_authority`、`ppo_head_scale`、`split_residual_mode` 尚未进入配置解析。
- `_apply_split_residual()` 仍使用同一个公共 scale。
- 需要先冻结 `r_PPO` 是否包含历史 `0.25`，否则 `s_head=1` 等价性有歧义。
- runner、auditor、analysis、stage-D 脚本不存在。
- Goal175 的远程 checkpoint 本机不存在。

### Goal180/181

- Goal180/181 matrix 不存在。
- Goal180 builder 当前实际使用 DAGGER2 和 8D path：`scripts/build_goal180_configs.py:24`。
- Goal181 的容量匹配 `±20%` 还缺参数计数口径、forward p95 的 batch/device/重复次数和 MLP hidden-dim 网格。

### 环境与资产

- `third_party/reference/wheel_legged_lab_e61bfe1/` 不存在。
- 需要为参考代码补 `SOURCE.json`、逐文件许可证和 SHA-256。
- 本地 `.venv` 都无法 import `rsl_rl`。

### 指标与注册

- summary 字段是 `path_horizontal_rmse_m`、`path_vertical_rmse_m`、`path_three_d_rmse_m`、`pre_goal_3d_rmse_m`：
  `src/auv_lowlevel/cli/evaluate_pc_ppo/summary.py:206`。
- matrix 字段是 `route_three_d_rmse_m`、`route_cross_abs_p90_m`、`route_vertical_abs_p90_m`、`preterminal_three_d_rmse_m`。
- 需要统一 alias；summary 目前没有 cross/vertical p90。
- Goal182 尚未登记到 `doc/goals/INDEX.md` 和 `doc/goals/registry/goals.jsonl`。
- 现有 WIP 只实现了 `training_profile` 配置与 manifest 传递。

## 5. 回复模板

```text
R1=
R2a=
R2b=
R3=
R4a=
R4b=
R5=
R6=
R7=
R8=
补充：
```

## 6. 确认后的动作边界

- 先把 R1–R8 的结论回填到 Goal179/180/181/182 的正式文档。
- 再按结论更新 matrix、入口检查器、auditor 和 telemetry 契约。
- 只有文档、配置、检查器都冻结后，才讨论是否生成 screen matrix。
- 本文件本身不产生训练任务，不改变任何 reward/termination/success/goal radius。
