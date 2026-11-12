# S14：实际地速与执行器饱和奖励修正

记录日期：2026-11-04

关联任务：`doc/goals/goal9_odarrl_control_reproduction.md` 中的 PC 残差策略实验。

## 1. 背景

S10f 和 S13 已经给出两个稳定结论：

- DADE-pose 在严格留出的高速强流格 `grid_hs` 上曾出现配对正信号，但
  default/strict125 的绝对成功率和 H1/H3/H4 的实际材料饱和仍未过门禁；
- 同一组 DADE 配方继续增加 DAgger 轮次没有产生增量，`grid_hs` 的配对
  success delta 为负，因此不能再用“继续斗智斗勇”的方式调 DADE。

ALOS-R 的诊断进一步表明：命令速度下限和实际地速不是一回事。R1K/R4A 的
`guidance_route_speed_ratio_min=1.0`，但真实 NED 地速在路径切向上的最低比只有
`0.25–0.63`。现有 `pc_command_track` 奖励主要惩罚机体系纵荡地速误差和请求动作
饱和，没有直接奖励“沿参考路径前进的实际地速”，也没有惩罚真实的
`material_saturated`。因此策略可以在命令跟踪上看起来正常，却仍然在转向、海流和
执行器饱和组合下掉速。

本实验不新增第二个学习残差网络，不改变 actor 输入维度，不把真实海流送入 actor，
也不使用 `goal_distance` 调制速度。它只修改可选奖励项，让已有的 no-DADE 和
DADE-pose 残差在同一个闭环目标上被重新优化。

## 2. 技术栈

- 仿真：`auv_lowlevel` PC20 batch Torch + REMUS100；
- 策略：冻结 BC + 可学习残差；DADE-pose 复用现有历史 encoder 和部署可用的
  `current_estimate_body` 侧通道；
- 训练入口：`scripts/train_batch_lowlevel_ppo.py`；
- 闭环评测：`scripts/evaluate_pc_ppo.py`、`scripts/eval_pc_server_basics_20261103.sh`；
- 汇总：`scripts/summarize_pc_server_evals_basics_20261103.py`；
- 配对报告：新增 `scripts/summarize_pc_s14_capability_20261104.py`；
- 服务器：`yhx@152.136.183.51:43127`，Python
  `/data/yhx/underwater-robotics/envs/venvs/isaaclab/bin/python`；
- 本地解释器：`/home/bubble/水下机器人/.venv/bin/python`。

## 3. 训练域与留出域

训练 demos 继续只使用：

| 格子 | 命令速度 | 海流速度 | 用途 |
|---|---:|---:|---|
| `grid_lw` | `1.0–1.4 m/s` | `0.10–0.20 m/s` | 训练 |
| `grid_ls` | `1.0–1.4 m/s` | `0.30–0.45 m/s` | 训练 |
| `grid_hw` | `1.8–2.0 m/s` | `0.10–0.20 m/s` | 训练 |
| `grid_hs` | `1.8–2.0 m/s` | `0.30–0.45 m/s` | **held-out，零样本评测** |

S14 不把 `grid_hs` 加入 PPO、teacher、DAgger 或奖励统计。H1/H2/H3/H4 和 C4 也
继续只做保留或压力评测。

环境配置通过可选的 `domain.cells` 做**逐 episode 联合采样**：每个训练 episode
先按 seed 选择一个格子，再在同一个格子内同时采样命令速度和海流速度。这样不会
像“把速度和海流范围分别放宽”那样意外采到 `grid_hs`。未配置 `domain` 的旧环境
完全保持原采样逻辑。S14 配置只列 `grid_lw/grid_ls/grid_hw` 三个格子，`grid_hs`
只出现在评测标签中。

## 4. 奖励改动

新增字段全部默认关闭，旧配置行为保持不变：

| 字段 | 默认值 | 作用 |
|---|---:|---|
| `w_route_speed_pc` | `0.0` | 实际路径切向地速低于 75% 巡航速度时的惩罚权重 |
| `route_speed_floor_ratio` | `0.75` | 实际路径切向地速的软下限 |
| `w_material_sat_pc` | `0.0` | 实际 `material_saturated` 的每步惩罚权重 |

实际路径切向地速由环境使用真实 `eta`、`nu`、`current_ned_mps` 计算，不使用
`goal_distance`：

```text
v_ground_ned = body_to_world(ground_velocity_body(nu, current))
v_route = clamp(dot(v_ground_ned, path_tangent_ned), min=0)
route_shortfall = relu(0.75 * episode_cruise_speed - v_route)
reward_route_speed = -w_route_speed_pc * route_shortfall / 2.5
reward_material_sat = -w_material_sat_pc * material_saturated
```

S14 配置使用 `w_route_speed_pc=1.0`、`w_material_sat_pc=0.2`。奖励仍然乘
`pc_ppo_scale=0.01`，不改变 PPO 的外层尺度。

## 5. 控制层固定项

为了让奖励评价和当前 ALOS-R 结论一致，S14 新配置启用以下固定项；旧配置不设置时
保持原行为：

- `robust_terminal_enabled=true`；
- `terminal_speed_ratio=1.0`，终点捕获只改航向，不按距离减速；
- `terminal_alignment_min_m=terminal_alignment_max_m=20.0`；
- `minimum_speed_ratio=0.75`，命令速度在速率限制后再次钳位；
- `route_speed_floor_enabled=true`，使用 DVL 路径切向地速做反馈；
- `initial_speed_ratio=1.10`、`initial_propeller_rpm_ratio=0.25`：起步螺旋桨和
  初始纵荡速度允许非零，但使用温和值，避免重复 R4B/R4C 的过冲退化；
- 不启用 `terminal_braking_enabled`，不存在 `terminal_speed_mps=0`。

初始速度和预旋只影响起步，不是速度下限的实现手段；整条航线的命令速度下限仍由
`minimum_speed_ratio` 和 PCEnv 的最终钳位保证。

起步阶段允许螺旋桨和初始纵荡速度非零，但不会把 `goal_distance` 或时间衰减引入
速度指令。`initial_speed_ratio` 只作用于 reset 时的 `nu[:, 0]`；
`initial_propeller_rpm_ratio` 只作用于 reset 时的初始 RPM 请求。二者都不改变
actor 输入，也不改变终端捕获逻辑。

## 6. 实验臂与严格配对

| 臂 | 初始化 | 奖励 | DADE | 用途 |
|---|---|---|---|---|
| `S14-A` | S13 no-DADE `final.pt` | route floor + actual saturation | 否 | 奖励修正 |
| `S14-B` | S13 DADE-pose `final.pt` | route floor + actual saturation | 是 | 奖励修正 + DADE |

两个臂使用同一批 S13 训练 seed、同一冻结 BC、同一 PPO 预算、同一评估 seed 集和
同一套 tags。与 S13 `final.pt` 的同训练 seed、同评估 seed 比较，不用评估 seed
反选 checkpoint。

预注册 tags：

```text
grid_lw grid_ls grid_hw grid_hs default strict125 h1 h2 h3 h4 c4
```

## 7. 验收线

S14 的主判定分成三层：

1. **奖励修正是否有效**：`S14-A final` 对同 seed 的 S13 no-DADE `final`。
2. **DADE 是否仍有增量**：`S14-B final` 对同 seed 的 `S14-A final`。
3. **DADE 对 S13 的整体增量**：`S14-B final` 对同 seed 的 S13 DADE-pose `final`。

保留格 `grid_lw/grid_ls/grid_hw/default/strict125` 必须满足：

- success 下降不超过 `0.02`；
- 三维 RMSE 不超过参考的 `1.05×`；
- material saturation 增量不超过 `0.02`；
- 实际路径切向地速 episode-min 均值不下降超过 `0.02`。

held-out `grid_hs` 必须满足：

- success `>= 0.85`；
- 三维 RMSE `<= 6.0 m`；
- 相对同 seed 参考至少满足一项：success `+0.02` 或 RMSE `<= 0.95×`；
- saturation 增量不超过 `0.02`，实际地速最低比不下降超过 `0.02`。

绝对能力门禁继续报告：

- default：success `>= 0.70`，RMSE `<= 5.0 m`；
- strict125：success `>= 0.50`，RMSE `<= 6.5 m`，timeout 明显低于 `0.531`；
- H1：success `>= 0.95`，RMSE `<= 5.0 m`，saturation `<= 0.40`；
- H3：success `>= 0.95`，RMSE `<= 3.4 m`，saturation `<= 0.45`；
- H4：`RMSE <= 3.8 m`，saturation `<= 0.45`，压力测试，不单独决定总状态；
- C4：只报告，不作为主门禁。

必须同时报告 `success`、`timeout`、`path_three_d_rmse_m`、
`material_saturation_rate`、yaw/speed tracking、`ground_route_speed_ratio_episode_min_mean`、
命令速度比、`current_valid_rate`、crab angle 和 terminal speed ratio。

## 8. 执行计划

### 有界实验协议

S14 是一次有界实验，不允许无限迭代：

1. 先跑 1 train seed × 8 eval seed × 6 tags 的 smoke；
2. smoke 不通过（包括训练/评测/契约失败）立即停止，不进入全量；
3. smoke 通过后只跑一次 5 train seed × 32 eval seed × 11 tags；
4. 全量 fail 立即停止；
5. 全量 pass 只追加一次 32–63 个 held-out eval seed 复核；
6. 复核后无论结果如何都终止，不再调权重、不补训练 seed、不换 tag。

“smoke 通过”指两个 S14 arm 都完成训练和最小闭环评测，且没有 NaN、actor 输入
维度变化、`grid_hs` 训练泄漏或服务器基础设施错误；不要求 smoke 指标达到最终
验收线。

- [x] 先写 S14 背景、技术栈、训练/留出域、奖励公式和门禁；
- [x] 在 `BatchedPCRewardComputer` 增加可选的 route-speed 和 actual-saturation 项，默认 0；
- [x] 在 `RewardConfig` / `_parse_reward` 增加字段并校验非负性；
- [x] 在 `PCEnv.step()` 计算真实路径切向地速并传给奖励；
- [x] 增加无终端减速、route-speed 奖励、actual-saturation 奖励和旧配置兼容测试；
- [x] 生成 S14 环境配置、no-DADE 训练配置和 DADE-pose 训练配置；
- [x] 增加联合域配置与逐 episode 采样，训练只使用 `grid_lw/grid_ls/grid_hw`；
- [x] 增加 S14 队列脚本，复用 S13 supervised `final.pt` 做 PPO warm start；
- [x] 增加 S14 三方配对汇总脚本；
- [x] 本地定向测试和 ruff；
- [x] 同步到服务器独立目录 `settled_局部规划器_反应式_s14`；
- [x] 服务器 1 seed × 8 eval seed × 6 tags smoke；
- [x] 审核冻结 BC 漂移、warm start、`grid_hs` 未泄漏、无 NaN；
- [x] smoke 通过后启动 5 train seed × 32 eval seed × 11 tags；
- [x] 自动汇总、配对报告并回填 goal9；
- [x] 全量 fail 后按协议停止，不启动 32–63 held-out 复核；
- [x] 未经用户明确要求不 commit/push。

## 9. 运行命令

本地测试：

```bash
cd /home/bubble/水下机器人/settled_局部规划器_反应式
../.venv/bin/pytest -q tests/test_lowlevel_pc_reward_termination.py \
  tests/test_batch_alos_r.py tests/test_lowlevel_pc_env.py
../.venv/bin/ruff check src/auv_lowlevel/batch/pc.py \
  src/auv_lowlevel/batch/pc_env.py src/auv_lowlevel/config.py \
  scripts/run_pc_s14_capability_20261104.sh
```

服务器 smoke（1 个训练 seed、8 个评估 seed）：

```bash
cd /data/yhx/underwater-robotics/src/plan_p1/settled_局部规划器_反应式_s14
export PYTHONPATH="$PWD/src:$PWD/../settled_全局规划器/src"
TRAIN_ROOT="$PWD/artifacts/lowlevel/pc_s14_capability_smoke_ppo_20261104" \
EVAL_ROOT="$PWD/artifacts/lowlevel/pc_s14_capability_smoke_eval_20261104" \
REPORT_ROOT="$PWD/artifacts/lowlevel/pc_s14_capability_smoke_summary_20261104" \
TRAIN_LOG="$PWD/logs/pc_s14_capability_smoke_train_20261104" \
EVAL_LOG="$PWD/logs/pc_s14_capability_smoke_eval_20261104" \
QUEUE_LOG="$PWD/logs/pc_s14_capability_smoke_queue_20261104" \
SEEDS='94030' SEED_COUNT=8 TAGS='grid_hs default strict125 h1 h3 h4' \
  ITERATIONS=10 GPU_IDS='2' EVAL_WORKERS=4 \
  setsid nohup bash scripts/run_pc_s14_capability_20261104.sh \
  > logs/pc_s14_capability_smoke_20261104.nohup.log 2>&1 < /dev/null &
```

正式五 seed：

```bash
TRAIN_ROOT="$PWD/artifacts/lowlevel/pc_s14_capability_5x32_ppo_20261104" \
EVAL_ROOT="$PWD/artifacts/lowlevel/pc_s14_capability_5x32_eval_20261104" \
REPORT_ROOT="$PWD/artifacts/lowlevel/pc_s14_capability_5x32_summary_20261104" \
TRAIN_LOG="$PWD/logs/pc_s14_capability_5x32_train_20261104" \
EVAL_LOG="$PWD/logs/pc_s14_capability_5x32_eval_20261104" \
QUEUE_LOG="$PWD/logs/pc_s14_capability_5x32_queue_20261104" \
SEEDS='94030 94031 94032 94033 94034' SEED_COUNT=32 \
  TAGS='grid_lw grid_ls grid_hw grid_hs default strict125 h1 h2 h3 h4 c4' \
  ITERATIONS=100 GPU_IDS='2 4 5' EVAL_WORKERS=8 \
  setsid nohup bash scripts/run_pc_s14_capability_20261104.sh \
  > logs/pc_s14_capability_full_20261104.nohup.log 2>&1 < /dev/null &
```

常用监控：

```bash
pgrep -af 'run_pc_s14_capability|train_batch_lowlevel_ppo|evaluate_pc_ppo'
tail -f logs/pc_s14_capability_5x32_queue_20261104/queue.log
nvidia-smi --query-gpu=index,memory.free,utilization.gpu --format=csv
```

smoke 和正式 5×32 必须使用不同 `TRAIN_ROOT`，否则队列会把 10-iteration smoke
checkpoint 当成已完成的全量训练而跳过。

### smoke 记录

2026-11-04 的 smoke 使用 1 train seed × 8 eval seed × 6 tags。第一次失败是独立
目录缺少冻结 BC checkpoint（基础设施问题）；补齐只读软链接后重跑同一 smoke，
两个 arm 均完成 10 次迭代和 12 个评测，无 NaN、无 actor 维度变化、无 `grid_hs`
训练泄漏。smoke 指标只用于判断队列是否可运行，不作为最终结论：

| arm | tag | success | timeout | 3D RMSE | saturation |
|---|---|---:|---:|---:|---:|
| S14-A no-DADE | default | 0.7500 | 0.2500 | 3.444 | 0.1311 |
| S14-A no-DADE | strict125 | 0.3750 | 0.6250 | 6.515 | 0.1121 |
| S14-A no-DADE | grid_hs | 0.8750 | 0.1250 | 4.542 | 0.4444 |
| S14-B DADE-pose | default | 0.8750 | 0.1250 | 3.004 | 0.1169 |
| S14-B DADE-pose | strict125 | 0.5000 | 0.5000 | 5.396 | 0.0996 |
| S14-B DADE-pose | grid_hs | 1.0000 | 0.0000 | 1.795 | 0.4513 |

S14-A 的 H1/H3/H4 材料饱和仍高于门禁；S14-B 的 H1/H3/H4 也高于门禁。因此即使
smoke 契约通过，仍必须由唯一一次 5×32 决定是否 fail，不能据 8-seed 结果写正结论。

### 5×32 结果与终止结论

2026-11-04 23:42 全量队列正常结束：10/10 个训练任务、110/110 个
`train_seed × arm × tag` 评测和自动汇总均成功，无 NaN、无训练泄漏、无基础设施
失败。结果根目录为服务器 `pc_s14_capability_5x32_summary_20261104/`，本地同步副本
为 `artifacts/lowlevel/pc_s14_capability_5x32_summary_20261104/`。

门禁状态：**全量 fail**。S14-A 在 default、strict125、H1、H3、H4、held-out
`grid_hs` 全部失败；S14-B 只有 held-out `grid_hs` 绝对门禁通过，default、
strict125、H1、H3、H4 仍失败。按预注册的有界协议，未启动 32–63 held-out seed
复核，实验到此终止，不再调权重、不补训练 seed、不换 tag。

S14-A（no-DADE）：

| tag | success | timeout | 3D RMSE | saturation | route min | yaw err | speed err | current valid | current age | crab deg | terminal ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `grid_lw` | 0.9812 | 0.0187 | 2.196 | 0.1822 | 0.7040 | 0.2707 | 0.1489 | 0.1755 | 1.3406 | 0.0000 | 1.0000 |
| `grid_ls` | 0.6687 | 0.3312 | 4.826 | 0.1427 | 0.4704 | 0.5352 | 0.3863 | 0.1643 | 1.3552 | 0.0000 | 1.0000 |
| `grid_hw` | 0.9875 | 0.0125 | 2.478 | 0.4806 | 0.6835 | 0.2664 | 0.1975 | 0.1609 | 1.3398 | 0.0000 | 1.0000 |
| `grid_hs` | 0.8313 | 0.1688 | 6.278 | 0.4285 | 0.5512 | 0.4244 | 0.4104 | 0.1747 | 1.3237 | 0.0000 | 1.0000 |
| `default` | 0.6312 | 0.3688 | 5.233 | 0.1344 | 0.4958 | 0.5705 | 0.3571 | 0.1565 | 1.3807 | 0.0000 | 1.0000 |
| `strict125` | 0.4125 | 0.5875 | 6.598 | 0.1244 | 0.4372 | 0.7528 | 0.3606 | 0.1564 | 1.4135 | 0.0000 | 1.0000 |
| `h1` | 0.9000 | 0.1000 | 5.013 | 0.4473 | 0.5981 | 0.3544 | 0.3535 | 0.1727 | 1.3041 | 0.0000 | 1.0000 |
| `h2` | 0.9750 | 0.0250 | 3.490 | 0.4929 | 0.6583 | 0.2840 | 0.2583 | 0.1572 | 1.3375 | 0.0000 | 1.0000 |
| `h3` | 0.9625 | 0.0375 | 4.495 | 0.4940 | 0.6490 | 0.3015 | 0.2160 | 0.1579 | 1.3499 | 0.0000 | 1.0000 |
| `h4` | 0.9250 | 0.0750 | 6.767 | 0.4713 | 0.6165 | 0.3438 | 0.2611 | 0.1469 | 1.3695 | 0.0000 | 1.0000 |
| `c4` | 0.2062 | 0.7937 | 8.901 | 0.2335 | 0.4364 | 0.7558 | 0.4734 | 0.0003 | 2.7334 | 0.0000 | 1.0000 |

S14-B（DADE-pose）：

| tag | success | timeout | 3D RMSE | saturation | route min | yaw err | speed err | current valid | current age | crab deg | terminal ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `grid_lw` | 0.9625 | 0.0375 | 2.614 | 0.1636 | 0.7058 | 0.3283 | 0.1487 | 0.1782 | 1.3348 | 0.0000 | 1.0000 |
| `grid_ls` | 0.6562 | 0.3438 | 5.476 | 0.1325 | 0.5020 | 0.6255 | 0.3778 | 0.1735 | 1.3453 | 0.0000 | 1.0000 |
| `grid_hw` | 0.9875 | 0.0125 | 2.740 | 0.4526 | 0.6828 | 0.2993 | 0.1907 | 0.1714 | 1.3425 | 0.0000 | 1.0000 |
| `grid_hs` | 0.9125 | 0.0875 | 5.062 | 0.4147 | 0.5999 | 0.3770 | 0.3898 | 0.1939 | 1.3030 | 0.0000 | 1.0000 |
| `default` | 0.6250 | 0.3750 | 5.805 | 0.1216 | 0.5304 | 0.6383 | 0.3524 | 0.1658 | 1.3486 | 0.0000 | 1.0000 |
| `strict125` | 0.4062 | 0.5938 | 7.345 | 0.1132 | 0.4877 | 0.8265 | 0.3463 | 0.1640 | 1.3721 | 0.0000 | 1.0000 |
| `h1` | 0.9500 | 0.0500 | 4.198 | 0.4307 | 0.6264 | 0.3358 | 0.3400 | 0.2001 | 1.3194 | 0.0000 | 1.0000 |
| `h2` | 0.9625 | 0.0375 | 4.015 | 0.4681 | 0.6427 | 0.3352 | 0.2528 | 0.1736 | 1.3441 | 0.0000 | 1.0000 |
| `h3` | 0.9437 | 0.0563 | 5.280 | 0.4688 | 0.6454 | 0.3576 | 0.2143 | 0.1629 | 1.3708 | 0.0000 | 1.0000 |
| `h4` | 0.8562 | 0.1437 | 9.654 | 0.4332 | 0.5883 | 0.4570 | 0.2703 | 0.1725 | 1.3458 | 0.0000 | 1.0000 |
| `c4` | 0.2437 | 0.7562 | 8.771 | 0.2157 | 0.4241 | 0.7436 | 0.4786 | 0.0003 | 2.6997 | 0.0000 | 1.0000 |

配对结论：

- S14-A 相对 S13 no-DADE 几乎停在原地：default success `-0.0063`、RMSE
  `+0.032 m`；strict125 success `+0.0063`、RMSE `-0.033 m`；held-out `grid_hs`
  success `-0.0125`、RMSE `+0.172 m`。奖励修正没有解决 strict125 timeout 或
  高速饱和。
- S14-B 相对 S14-A 的 held-out `grid_hs` 有正信号：success `+0.0813`、RMSE
  `-1.216 m`；但 default、strict125、H3、H4 同时退化（success
  `-0.0063/-0.0063/-0.0187/-0.0688`，RMSE `+0.572/+0.747/+0.784/+2.887 m`）。
- S14-B 的 `grid_hs` 绝对门禁通过，但相对 S13 DADE-pose 只有 success
  `+0.0125`、RMSE `-0.163 m`，未达到预注册的 `+0.02` success 或 `0.95×` RMSE
  增量标准；不能把它登记为稳定的 DADE 正结论。
- 所有 arm 的 terminal speed ratio 都是 `1.0000`，不存在终端距离减速回归；
  strict125 仍主要由 timeout（`0.5875/0.5938`）和 3D RMSE 失败，而不是终点速度。

## 10. Git 提交与推送步骤

本实验不自动提交和推送。用户确认后，按文件粒度提交：

```bash
cd /home/bubble/水下机器人
git add \
  'settled_局部规划器_反应式/configs/lowlevel/odarrl_control_pc_s14_route_floor_reward.json' \
  'settled_局部规划器_反应式/configs/lowlevel/odarrl_control_ppo_pc_grid_gpu_frozen_bc_residual_nodade_s14_route_reward_lr3e-4_scale1p0_dz64_gamma995_warmup5_critic3e-4_fixed_rewardscale001.json' \
  'settled_局部规划器_反应式/configs/lowlevel/odarrl_control_ppo_pc_grid_gpu_frozen_bc_residual_dade_pose_s14_route_reward_lr3e-4_scale1p0_dz64_gamma995_warmup5_critic3e-4_fixed_rewardscale001.json' \
  'settled_局部规划器_反应式/src/auv_lowlevel/batch/pc.py' \
  'settled_局部规划器_反应式/src/auv_lowlevel/batch/pc_env.py' \
  'settled_局部规划器_反应式/src/auv_lowlevel/batch/pc_vec_env.py' \
  'settled_局部规划器_反应式/src/auv_lowlevel/config.py' \
  'settled_局部规划器_反应式/scripts/run_pc_s14_capability_20261104.sh' \
  'settled_局部规划器_反应式/scripts/summarize_pc_s14_capability_20261104.py' \
  'settled_局部规划器_反应式/scripts/run_pc_server_basics_train_20261103.sh' \
  'settled_局部规划器_反应式/scripts/eval_pc_server_basics_20261103.sh' \
  'settled_局部规划器_反应式/tests/test_lowlevel_pc_reward_termination.py' \
  'settled_局部规划器_反应式/tests/test_lowlevel_pc_env.py' \
  'settled_局部规划器_反应式/tests/test_lowlevel_pc_contract.py' \
  'settled_局部规划器_反应式/tests/test_pc_s14_capability_report.py' \
  'settled_局部规划器_反应式/doc/goals/goal15_capability_joint_domain.md' \
  'settled_局部规划器_反应式/doc/goals/goal9_pc_server_validation_plan.md'
git diff --cached --check
git commit -m "feat(pc): add route-speed and actuator-saturation reward experiment"
git push origin main
```

不要提交 `artifacts/`、`logs/`、checkpoint、TensorBoard event 或完整 rollout。

