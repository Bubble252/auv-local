# Goal177：Residual head 的三维动作接口与 head-only / last-layer pilot

**日期**：2026-11-14<br>
**状态**：`STAGE_A_COMPLETE / GATE_A_FAIL / RESIDUAL_TOO_SMALL_FOR_PPO / STAGE_B_BLOCKED / HELDOUT_CLOSED`<br>
**仓库**：`settled_局部规划器_反应式`

```yaml
goal_id: Goal177
repo: settled_局部规划器_反应式
status: STAGE_A_COMPLETE / GATE_A_FAIL / RESIDUAL_TOO_SMALL_FOR_PPO / STAGE_B_BLOCKED / HELDOUT_CLOSED
phase: P7_PLAN
created: 2026-11-14
updated: 2026-11-14
parents:
  - Goal176
matrix: configs/governance/goal177_matrix.json
legacy: false
```

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
  check_command: python ../scripts/check_research_ledger.py --goal Goal177 --goal-doc doc/goals/goal177_residual_head_action_interface_3d_20261114.md --strict
```

- `required: true`：这是动作接口诊断和最小训练 pilot；任何 Gate 失败都必须先写
  blocker、至少两个可证伪 hypothesis 和 reading note，再决定 `continue / pivot / stop`。
- 只有 `pivot` 或 `stop` 才允许改变主线；不得用扩 seed、换阈值或打开 held-out 救结果。

## 1. 背景与直接证据

Goal176 已经排除了“latent 里完全没有纵向信息”这一解释：

| probe target | held-out R² 范围（三条 COMP 臂） |
|---|---:|
| `vertical_error` | `0.9967–0.9999` |
| `pitch_error` | `0.997–0.9999` |
| `q` | `0.997–0.99996` |
| `w` | `0.997–0.9999` |

但同一批 checkpoint 的 residual action 在闭环中几乎不改变路线。Goal175 的
`P_RUD_075_COMP` 相对同场景 `zero_bc50` 的 3D route 变化只有约 `0.37 mm`，
而 residual action 的 mean absolute value 约 `0.0029`。Goal176 的通道级审计进一步
显示：

- `P_RUD_075_COMP` vertical applied share：rudder `3.27e-5`、stern `6.11e-6`、
  rpm `1.83e-4`；
- `P_ALL_075_COMP` vertical applied share：rudder `1.04e-6`、stern `2.30e-5`、
  rpm `1.23e-4`；
- residual 在进入执行器前还经过两层缩放：`residual_action()` 中的
  `frozen_bc_residual_scale=0.25`，以及 actuator wrapper 中的
  `actuator_residual_scale` 和逐通道 gain；
- residual 输出层是 zero-initialized，PPO 使用 `clip_param=0.02`、
  `num_learning_epochs=1`、`num_mini_batches=64`、`schedule=fixed`、
  `init_noise_std=0.1`；late action std 约 `0.1`，而 residual 绝对值只有
  `0.001–0.003` 量级。

因此当前最可疑的不是 encoder，而是：**head 有方向性信息，但动作接口把可用权限
压得太小，PPO 也很难从这么小的动作里获得足够 credit。**

本 goal 同时检查横向、纵向和前进/沿航迹三个方向，避免只修 vertical 而把
cross-track 或 forward progress 弄坏。

## 2. 假设与证伪

### H177-A：residual head 的动作接口没有实际权限

- 成立预期：提高 actuator authority 或放宽通道 gain 后，applied residual share、
  Jacobian 幅值和闭环路线敏感度同步上升；raw head 本身不是零。
- 失败预期：raw head 对纵向/横向/前进输入有响应，但 applied action 在 authority
  sweep 中仍被压到 `1e-5` 以下，或 apply 前后方向不一致；这说明问题在缩放、
  wrapper 或 action 参数化，而不是 encoder。
- 最便宜证伪：Stage A 的 eval-only authority × channel sweep，不训练。

### H177-B：冻结 encoder 只训 head 足够，还是需要最后一层适配

- 成立预期：`H0_head_only` 和/或 `H1_last_layer` 在相同 PPO bundle 下把 residual
  applied share 提高到可见范围，并至少改善一个方向的主指标。
- 失败预期：两者都没有改变 applied share 或闭环路线；此时“冻结与否”不是主因，
  应继续检查 action interface、reward credit 和控制器/plant 的通道耦合。
- 最便宜证伪：一个 seed、1M transitions、40 iterations 的 head-only / last-layer
  pilot；不进入正式矩阵。

### H177-C：横向、纵向和前进方向不是同一个动作瓶颈

- 成立预期：rudder-only 对 cross-track 有权限，但对 vertical 或 forward 的
  applied share 明显更低；stern/rpm 通道在 vertical/forward 上更有效。
- 失败预期：所有通道对三个方向都同样迟钝，说明是统一的 residual 输出/credit
  问题，而不是通道选择问题。
- 最便宜证伪：Stage A 的四组 channel gain，加上三方向有限差分 Jacobian。

## 3. 数据、checkpoint 与隔离

- 固定 checkpoint：Goal175 有效根
  `/data/yhx/underwater-robotics/runs/auv-local/goal175/20261114-goal175-c186c74-a2`
  中的 `P_RUD_075_COMP` final checkpoint；路径和 SHA 从 `run_manifest.json` 解析，
  不在脚本里硬编码个人目录。
- Stage A：dev seeds `20–27`；四个 level（`nominal / current10 / noise_n2 /
  plant20`）和两个 radius（`default / strict125`）。
- Stage B pilot：训练 seed 固定 `86201`；评价 dev seeds `20–59`；只跑
  `current10 × default/strict125`，明确标记为 pilot，不冒充正式矩阵。
- held-out `100–149` 全程关闭；不得读取、缓存或用于调参。
- 不修改 reward、termination、success 定义、goal radius、课程定义或 eval split。
- 不训练 Transformer，不打开 DADE，不增加显式纵向 skip，不重开 encoder 预训练。

## 4. Stage A：eval-only 三维动作接口审计

### 4.1 Authority sweep

固定 `P_RUD_075_COMP` checkpoint，将运行时 `actuator_residual_scale` 扫为：

```text
0.25 / 0.75 / 1.5 / 2.5
```

base policy 的 `residual_scale=0.25` 保持不变；只改变 wrapper 的 authority，
以便分离“head 输出”与“执行器权限”。

### 4.2 Channel sweep

| mode | `actuator_residual_channel_gains` |
|---|---|
| `rudder_only` | `[1.0, 1e-6, 1e-6]` |
| `stern_only` | `[1e-6, 1.0, 1e-6]` |
| `rudder_stern` | `[1.0, 1.0, 1e-6]` |
| `all` | `[1.0, 1.0, 1.0]` |

### 4.3 三方向小扰动

对每个采样 observation 做中心差分，并记录 raw head、policy residual、
actuator residual、applied action 和 total action：

| 方向 | 扰动输入 | 步长 |
|---|---|---:|
| cross-track | path cross error | `±0.10 m` |
| cross-track | path heading error | `±0.05 rad` |
| vertical | path vertical error | `±0.10 m` |
| vertical/pitch | path tangent pitch error | `±0.05 rad` |
| vertical/pitch | state `q` | `±0.05 rad/s` |
| vertical/pitch | state `w` | `±0.05 m/s` |
| forward/along-track | path progress ratio | `±0.02` |
| forward/along-track | path remaining ratio | `±0.02` |
| forward/along-track | endpoint proximity feature | `±0.10`（归一化特征） |
| forward/along-track | command speed | `±0.05 m/s` |
| forward/along-track | surge velocity `u` | `±0.05 m/s` |

### 4.4 必须报告的指标

每个 arm × level × radius × seed 记录：

- `applied_share`：`|applied residual| / (|base action| + |applied residual|)`，
  按通道和方向统计，报告中位数和 p90；
- `sign_consistency`：中心差分两侧方向一致的比例；
- `residual_abs_mean`、`residual_abs_p90`、`residual_to_action_std`；
- route cross p90、route vertical p90、整条 3D RMSE、五段 RMSE、20–90% keypoints；
- forward 代理指标：`path_completion_ratio_mean`、`final_path_progress_m`、
  `pre_goal_3d_rmse_m`、`miss_distance_at_plane_m`、
  `terminal_longitudinal_rmse_m`、`post_goal_overshoot_m`、
  `ground_route_speed_ratio_mean`；
- success、timeout、safety、flew-out、NaN/Inf、最大地速和 overspeed steps。

Stage A 的输出：

```text
artifacts/lowlevel/goal177_residual_head_action_interface_20261114/stage_a/
  run_manifest.json
  stage_a_summary.json
  stage_a_summary.md
```

## 5. Stage B：head-only 与 last-layer pilot

### 5.1 训练臂

| arm | encoder | 可训练部分 | residual 输出初始化 |
|---|---|---|---|
| `H0_head_only` | 全部冻结 | actor head / residual head | 非零，`0.01–0.03` |
| `H1_last_layer` | 除最后一层 state-encoder block 外冻结 | 最后一层 block + actor head | 非零，`0.01–0.03` |

控制臂：

- `zero_bc50`：同一 ALOS-COMP 基座、无 learned residual；
- `zero_alos_comp`：无 residual 的 ALOS-COMP 对照。

### 5.2 共享 PPO bundle

两个训练臂必须使用完全相同的 bundle，除可训练参数范围外不改变其他训练变量：

```json
{
  "num_learning_epochs": 2,
  "num_mini_batches": 32,
  "clip_param": 0.05,
  "schedule": "adaptive",
  "desired_kl": 0.005,
  "init_noise_std": 0.03,
  "learning_rate_actor": 0.0003
}
```

- 训练 seed：`86201`；
- 每个 arm：`1,024,000` transitions，40 iterations；
- 课程：沿用 Goal173/175 的冻结课程，`curriculum.enabled=true`，
  `promotion_enabled=false`，逐 episode 写 `curriculum_episodes.jsonl`；
- 不改变 reward、termination、success、radius、level mix 或 eval protocol。

Stage B 只在 `current10 × default/strict125`、dev seeds `20–59` 上做 pilot。
它不是正式 `arms × levels × radii × zero baselines` 矩阵；通过后必须另写正式 goal。

## 6. 执行计划

### 6.1 不可协商的运行契约

| 项目 | 强制要求 | 失败标签 |
|---|---|---|
| 训练课程 | `curriculum.enabled=true`；冻结 `stage_mix`；`promotion_enabled=false`；逐 episode `curriculum_episodes.jsonl` | `INVALID_NONCURRICULUM` |
| 训练速度 | 每个 stage 满足 `speed + current + variation <= 2.5 m/s` | `INVALID_SPEED_CONTRACT` |
| Stage A 评价 | 固定 checkpoint；`16 arms × 4 levels × 2 radii`，dev seeds `20–27` | `INVALID_INCOMPLETE_EVAL` |
| Stage B pilot | 两个训练臂 + 两个 zero control；`current10 × 2 radii`，dev seeds `20–59` | `INVALID_INCOMPLETE_EVAL` |
| 评价 seeds | 只用 dev seeds；held-out 关闭 | `INVALID_SEED_SPLIT` |
| 启动与结束 | 先跑协议检查器，再跑实验，最后跑产物审计器；任一失败非零退出 | `INVALID_PROTOCOL` |
| 主指标 | 三方向路线指标 + forward 代理 + success/timeout/safety，不能只看 success | 不得进入结论 |

- [x] Step 0：运行契约与矩阵检查；确认 checkpoint、config 和 SHA；确认 held-out 关闭。
- [x] Step 1：实现 `scripts/run_goal177_stage_a_interface_audit.py`、对应审计器、
      通道/authority override 和三方向 Jacobian 记录；先做 1 seed smoke。
- [x] Step 2：跑 Stage A `16 arms × 4 levels × 2 radii × 8 seeds`；
      生成 `stage_a_summary.json` / `.md`，并审计所有 cell、NaN、flew-out 和 speed。
- [x] Step 3：按 Gate A 判读。Gate A 失败就停止，不启动 Stage B 训练。
- [ ] Step 4：Gate A 通过后，生成 `H0_head_only`、`H1_last_layer` config 和
      `scripts/run_goal177_stage_b_pilot.py`；先 smoke，再各训 1M transitions。
- [ ] Step 5：运行 Stage B pilot 评价，生成 summary、曲线和三方向对比。
- [ ] Step 6：按 Gate B 判读；写 result md、ledger/decision；失败则写 blocker。
- [ ] Step 7：通过用户批准后，提交本次相关文件并 push；不自动 commit/push。

Stage A 执行记录：`144/144` cells 完成，正式 v3 head-only 审计为
`GATE_A_FAIL / RESIDUAL_TOO_SMALL_FOR_PPO`。maximum head residual/action std 为
`0.0737 < 0.20`；v2 曾把冻结 BC + PPO 的 composed authority 误当作 head
authority，已降级为诊断产物。Step 4–7 未执行，Stage B 未启动，held-out 保持关闭。
结果见 `doc/goals/goal177_result_20261114.md`。

### 6.2 每个 Step 的运行与提交边界

**Step 0**

```bash
python ../scripts/check_research_ledger.py \
  --goal Goal177 \
  --goal-doc doc/goals/goal177_residual_head_action_interface_3d_20261114.md \
  --strict
```

同时检查 `configs/governance/goal177_matrix.json` 的 arm 数、level 数、radius 数
和 seed 数；checkpoint SHA 必须与 Goal175 manifest 一致。

**Step 1–2**

```bash
python scripts/run_goal177_stage_a_interface_audit.py \
  --root artifacts/lowlevel/goal177_residual_head_action_interface_20261114 \
  --goal175-root /data/yhx/underwater-robotics/runs/auv-local/goal175/20261114-goal175-c186c74-a2 \
  --seed-start 20 --seed-count 8 --skip-existing
python scripts/audit_goal177_stage_a.py \
  --root artifacts/lowlevel/goal177_residual_head_action_interface_20261114
```

审计器必须验证：`16 × 4 × 2 = 128` 个 policy cell、`2 × 4 × 2 = 16` 个 zero
cell、每个 cell 8 个 seed、所有 trace 非空、无 NaN/Inf/flew-out、无 held-out seed。

**Step 4–5**

```bash
python scripts/run_goal177_stage_b_pilot.py \
  --root artifacts/lowlevel/goal177_residual_head_action_interface_20261114 \
  --seed 86201 --transitions 1024000 --skip-existing
python scripts/audit_goal177_stage_b.py \
  --root artifacts/lowlevel/goal177_residual_head_action_interface_20261114
```

训练启动器必须先调用协议检查器；训练结束后必须检查课程 provenance、checkpoint、
训练配置快照和 40 个 iteration records。评价只允许 `current10 × 2 radii × 40 seeds`。

**Step 7**

```bash
git add doc/goals/goal177_residual_head_action_interface_3d_20261114.md \
  doc/goals/goal177_result_20261114.md \
  configs/governance/goal177_matrix.json \
  scripts/run_goal177_stage_a_interface_audit.py \
  scripts/audit_goal177_stage_a.py \
  scripts/run_goal177_stage_b_pilot.py \
  scripts/audit_goal177_stage_b.py \
  research/LEDGER.jsonl
git diff --cached --check
```

禁止 `git add -A`；不提交 `artifacts/`、checkpoint、日志或缓存。是否 commit/push
必须由用户明确批准。

## 7. Gate

### Gate A：动作接口先决条件

| check | 阈值 | 失败标签 |
|---|---:|---|
| vertical 或 cross 的 applied share | `>= 1e-2`，或至少为 Goal176 同方向 baseline 的 `10x` | `ACTION_AUTHORITY_STILL_NULL` |
| forward/along-track 的 applied share | `>= 1e-3`，或至少为 Goal176 同方向 baseline 的 `10x` | `FORWARD_AUTHORITY_NULL` |
| 三方向 Jacobian 符号一致性 | `>= 0.90` | `ACTION_DIRECTION_UNSTABLE` |
| residual 幅度相对 action std | `residual_abs / action_std >= 0.20`，或 action std `<= 0.05` | `RESIDUAL_TOO_SMALL_FOR_PPO` |
| 稳定性 | 0 NaN/Inf、0 flew-out、0 safety 回归；timeout 不比 `zero_bc50` 增加 | `STABILITY_FAIL` |

Gate A 失败时：

1. 停止 Stage B 训练；
2. 按 channel、wrapper、plant 和 action parameterization 顺序检查；
3. 写 blocker 和至少两个可证伪 hypothesis；
4. 不把失败归因为 encoder，也不跳到预训练目标、DADE 或 Transformer。

### Gate B：最小 pilot 的行为收益

对比 `current10 × default/strict125` 下同场景 `zero_bc50`：

| check | 阈值 | 失败标签 |
|---|---:|---|
| vertical p90 | 改善 `>= 0.03 m` | `VERTICAL_PILOT_NO_GAIN` |
| cross-track p90 | 改善 `>= 0.08 m` 或 `>= 4%` | `CROSS_PILOT_NO_GAIN` |
| forward/along-track | completion ratio 回退 `<= 0.01`；pre-goal 3D RMSE 回退 `<= 0.02 m`；miss distance 不增加 `> 0.10 m` | `FORWARD_PILOT_REGRESSION` |
| 整条 3D route | 不差于 `zero_bc50` | `ROUTE_PILOT_REGRESSION` |
| success | 每 cell 下降 `<= 2/40` | `SUCCESS_PILOT_REGRESSION` |
| 稳定性 | 无 safety/timeout 增加；applied share 仍达到 Gate A 门槛 | `PILOT_STABILITY_FAIL` |

只有两个训练臂中至少一个同时满足 vertical、cross、forward 和 success 约束，才算
Stage B PASS。Stage B PASS 只说明值得进入正式矩阵，不等于正式结论。

## 8. 失败判读与下一步

| 观察 | 判读 | 下一步 |
|---|---|---|
| Stage A applied share 随 authority 单调上升，但闭环仍不动 | head 有输出，plant/动作方向或 credit 有问题 | 检查通道耦合和 reward/credit；不先改 encoder |
| 提高 authority 后 cross 改善、vertical/forward 不动 | 通道选择不对称 | 优先测试 stern/rpm 或分开的纵向通道；保持 cross guard |
| H0/H1 都无变化 | 冻结与否不是主因 | 检查 action interface、PPO advantage 和 termination/credit |
| H1 明显优于 H0 | 最后一层需要适配 | 另写正式矩阵 goal，不改本 Gate |
| probe 仍好但所有 applied share 都接近零 | 动作接口是主瓶颈 | 先修 residual output/wrapper；不打开 DADE |

## 9. 明确延后

- 正式完整矩阵（4 levels × 2 radii × 多 arms × 40+ seeds）不在本 goal 自动启动；
  Gate B 通过后另写 goal。
- Transformer actor、encoder 预训练、显式纵向 skip、DADE 历史编码全部延后；
  在有 Gate A/B 证据前不打开。
- 不改变 reward / termination / success / radius；不扩 seed 救指标。

## 10. Git 与完成检查

完成前必须运行：

```bash
python ../scripts/check_research_ledger.py \
  --goal Goal177 \
  --goal-doc doc/goals/goal177_residual_head_action_interface_3d_20261114.md \
  --strict
```

如果 checker 因路径或沙盒失败，可在沙盒外重试；不能用“命令没跑”代替通过。

## 11. Goal 提示词

```text
你现在执行 Goal177：Residual head 的三维动作接口与 head-only / last-layer pilot。

先读：
- settled_局部规划器_反应式/doc/goals/goal177_residual_head_action_interface_3d_20261114.md
- settled_局部规划器_反应式/doc/goals/goal176_result_20261114.md
- settled_局部规划器_反应式/doc/goals/goal175_result_20261114.md
- settled_局部规划器_反应式/configs/governance/goal177_matrix.json

硬约束：
1. 先只做 Stage A，eval-only，不训练；
2. Stage A 固定 Goal175 P_RUD_075_COMP checkpoint，dev seeds 20–27，四 levels，两 radii；
3. 同时检查 cross-track、vertical/pitch 和 forward/along-track 三个方向；
4. 记录 raw head -> policy residual -> actuator residual -> applied -> total 的全链路；
5. Gate A 失败立即停止，不启动 Stage B，不跳 DADE/预训练/Transformer；
6. 不打开 held-out 100–149，不改 reward/termination/success/radius，不扩 seed；
7. 不自动 commit/push；所有结果保留审计产物。

执行顺序：
- Step 0 检查矩阵、checkpoint 和 Research Contract；
- Step 1 实现 Stage A 启动器和审计器；
- Step 2 跑 1-seed smoke；
- Step 3 跑完整 Stage A，写 summary；
- Step 4 按 Gate A 判读并报告；只有 Gate A 通过才询问是否进入 Stage B。
```
