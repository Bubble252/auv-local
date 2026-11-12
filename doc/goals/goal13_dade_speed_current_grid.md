# DADE 在解耦速度×海流域中的配对验证

## 背景

现有 no-DADE 残差已经在显式三格速度×海流训练域上完成五个训练 seed、每个 tag
`32` 个闭环评估 seed 的验证。训练格为低弱、低强、高弱，高强格严格留出；结果
仍然失败：高强格 final 为 `0.8313 / 6.54 m`，没有超过 pretrain initial 的
`0.8438 / 6.05 m`，default/strict125 也有退化。

已有的 multienv DADE-pose 实验证明历史机制被策略实际使用，但闭环验收失败。该
实验的训练域仍来自历史配置拼接，因此不能回答一个关键问题：**DADE 的收益是否
只在训练域缺少高速强流组合时被掩盖，还是 DADE 本身无法把当前估计转成闭环收益。**

本实验只做一组预注册的结构验证：复用同一个扩大域、同一批 teacher demos、同一个
冻结 BC、同一训练预算，比较 no-DADE 与 DADE-pose。它不继续搜索 DADE 超参数，
也不改变环境、奖励、动作包线或评估 seed。

## 实验问题

在三格速度×海流训练域上加入 DADE-pose 后，策略能否在高强留出格 `grid_hs` 上
相对严格配对的 no-DADE 取得可复现的闭环改善，同时保持 default、strict125、
H1、H3 和执行器饱和不明显退化？

## 技术栈

- 仿真环境：`auv_lowlevel` PC20 batch Torch 环境；
- teacher：PD/PI 控制器生成的闭环 demonstrations；
- 冻结 BC：`artifacts/lowlevel/pc_bc_c0_c1_c2_c3random_v1_dz64_20261103/pc_bc_actor.pt`；
- 训练入口：`scripts/pretrain_pc_residual.py`；
- 闭环评测：`scripts/evaluate_pc_ppo.py` 与
  `scripts/eval_pc_server_basics_20261103.sh`；
- 汇总：`scripts/summarize_pc_server_evals_basics_20261103.py`；
- 配对门禁：`scripts/summarize_pc_dade_grid_20261104.py`；
- 固定训练 seed：`94030 94031 94032 94033 94034`；
- 固定评估 seed：`0–31`，训练和留出格共用同一批评估 seed。

## 训练域与留出域

| 格子 | 命令速度 | 海流速度 | 用途 |
|---|---:|---:|---|
| `grid_lw` | `1.0–1.4 m/s` | `0.10–0.20 m/s` | 训练 |
| `grid_ls` | `1.0–1.4 m/s` | `0.30–0.45 m/s` | 训练 |
| `grid_hw` | `1.8–2.0 m/s` | `0.10–0.20 m/s` | 训练 |
| `grid_hs` | `1.8–2.0 m/s` | `0.30–0.45 m/s` | **held-out，零样本评测** |

训练 demos 只合并 `grid_lw`、`grid_ls`、`grid_hw`；`grid_hs` 只用于评测。default、
strict125、H1/H2/H3、C4/H4 作为保留和压力 tag，不参与训练。

## 两个严格配对的臂

两个臂使用相同训练 demos、相同冻结 BC、相同训练 seed、相同 `196,608` 个样本、
相同 batch size、epoch 上限、patience、学习率、输出 L2 和验证切分；唯一差异是
DADE 结构：

| 臂 | 配置 |
|---|---|
| no-DADE GPU 对照 | `odarrl_control_ppo_pc_grid_gpu_frozen_bc_residual_nodade_lr3e-4_scale1p0_*` |
| DADE-pose | `odarrl_control_ppo_pc_grid_gpu_frozen_bc_residual_dade_pose_lr3e-4_scale1p0_*` |

DADE-pose 保持既有结构化设置：`dade_enabled=true`、`dade_state_mode=pose`、
`actor_state_mode=pose`、isolated action head、`dade_warmstart_input_scale=0.03`、
当前估计辅助损失 `0.5`。不增加第二个学习残差网络，不改变 actor 输入维度，也不
把真实海流送入 actor。

## 产物

- no-DADE 训练：`artifacts/lowlevel/pc_dade_grid_nodade_supervised_residual_20261104/`；
- DADE 训练：`artifacts/lowlevel/pc_dade_grid_dade_supervised_residual_20261104/`；
- 两臂闭环评测：`artifacts/lowlevel/pc_dade_grid_{nodade,dade}_eval_20261104/`；
- 汇总：`artifacts/lowlevel/pc_dade_grid_summary_20261104/summary.{json,md}`；
- 配对报告：`artifacts/lowlevel/pc_dade_grid_report_20261104/report.{json,md}`；
- 队列日志：`logs/pc_dade_grid_20261104/`；
- 服务器队列入口：`scripts/run_pc_dade_grid_20261104.sh`。

## 执行计划

- [x] 生成 no-DADE 与 DADE-pose 的 GPU 训练配置；
- [x] 服务器 smoke：每个臂训练 1 个 seed，评估 8 个 seed 的 `grid_hs/default/strict125`；
- [x] smoke 审计 DADE 开关、冻结 BC 漂移、checkpoint 可加载和闭环无 NaN；
- [x] 五个训练 seed × 两个臂的 GPU 监督训练（队列 PID `392203`）；
- [x] 审计两臂都只使用 `lw/ls/hw` 三格，`hs` 未泄漏；
- [x] 两臂 `initial/final` 在 `grid_lw grid_ls grid_hw grid_hs default strict125 h1 h2 h3 h4 c4` 上闭环评测；
- [x] 用同一批 `32` 个评估 seed 生成配对汇总和预注册门禁；
- [x] 将结果回填 goal9 和服务器验证计划，保留所有负结果；
- [ ] 未经用户明确要求不自动 commit/push。

## 预注册门禁

所有比较均使用 DADE final 对同训练 seed、同评估 seed 的 no-DADE final。

- 训练格 `grid_lw/grid_ls/grid_hw`：成功率下降不超过 `0.02`，RMSE 不超过
  `1.10×`，material saturation 增量不超过 `0.05`；
- held-out `grid_hs`：success 至少 `+0.02` 或 RMSE 至少改善到 `0.95×`，且
  saturation 增量不超过 `0.05`；绝对线为 success `>=0.85`、RMSE `<=6.0 m`；
- default：success `>=0.70`、RMSE `<=5.0 m`、timeout 不得高于 no-DADE；
- strict125：success `>=0.50`、RMSE `<=6.5 m`、timeout `<=0.531`；
- H1：success `>=0.95`、RMSE `<=5.0 m`、saturation `<=0.40`；
- H3：success `>=0.95`、RMSE `<=3.4 m`、saturation `<=0.45`；
- H4：只做压力测试，报告 success、RMSE、saturation 和 timeout；
- C4：只做压力测试；
- 所有 tag 都必须报告 success、timeout、三维路径 RMSE、material saturation、
  yaw/speed tracking error，并保留每个 seed 的原始 summary。

如果 `grid_hs` 或 retention tag 未达到门禁，整体状态为 `fail`，不通过挑选 seed、
延长训练或追加同配方超参搜索改变结论。

## Smoke 结果（2026-11-04 18:14）

服务器 smoke 使用两个臂的同 seed `94030`、8 个评估 seed（`0–7`）。两个训练
manifest 均为 `completed`、`device=cuda`、只包含 `lw/ls/hw` 三格，高强格未泄漏，
冻结 BC 最大漂移为 `0.0`；DADE manifest 的 `dade_enabled=true`，no-DADE 为
`false`。6 个闭环 evaluation 全部 `rc=0`，没有 NaN 或安全终止异常。

smoke 仅用于结构验证，不能作为正式结论：

| Tag | DADE success | no-DADE success | DADE RMSE | no-DADE RMSE | DADE timeout | DADE saturation |
|---|---:|---:|---:|---:|---:|---:|
| default | `0.8750` | `0.7500` | `2.93 m` | `3.42 m` | `0.1250` | `0.1125` |
| grid_hs | `1.0000` | `0.8750` | `1.69 m` | `4.51 m` | `0.0000` | `0.4590` |
| strict125 | `0.3750` | `0.3750` | `6.51 m` | `6.47 m` | `0.6250` | `0.0917` |

`grid_hs` 的早期信号较大，但只有一个训练 seed、八个评估 seed，且 H1/H2/H3/H4
与完整 retention tag 尚未跑；不能据此宣告通过。正式五 seed × 32 seed 队列已启动，
训练 seed 为 `94030–94034`，GPU 为 `2/4/5`，日志位于
`logs/pc_dade_grid_20261104/queue.log`。

## 五 seed 训练审计（2026-11-04 18:21）

`10/10` 个监督训练 run 已完成，`audit.json` 的 `failures=[]`。两臂都只包含
`pc_grid_low_weak`、`pc_grid_low_strong`、`pc_grid_high_weak`，高强格没有泄漏；
所有 run 的冻结 BC 参数最大漂移均为 `0.0`，训练设备为 `cuda`，且都存在
`model_initial.pt`、`final.pt` 和 `training_result.json`。

| Arm | Seed | device | DADE | best epoch | epochs | best val loss |
|---|---:|---|---:|---:|---:|---:|
| no-DADE | `94030` | cuda | false | `3` | `34` | `0.02166` |
| no-DADE | `94031` | cuda | false | `32` | `63` | `0.01675` |
| no-DADE | `94032` | cuda | false | `50` | `81` | `0.01615` |
| no-DADE | `94033` | cuda | false | `33` | `64` | `0.01942` |
| no-DADE | `94034` | cuda | false | `9` | `40` | `0.02252` |
| DADE-pose | `94030` | cuda | true | `14` | `45` | `0.02422` |
| DADE-pose | `94031` | cuda | true | `14` | `45` | `0.01899` |
| DADE-pose | `94032` | cuda | true | `22` | `53` | `0.01767` |
| DADE-pose | `94033` | cuda | true | `19` | `50` | `0.02018` |
| DADE-pose | `94034` | cuda | true | `6` | `37` | `0.02539` |

DADE 的残差监督验证损失略高，但当前问题不是离线余差，而是闭环海流适应和执行器
饱和，因此不能仅凭该损失判定实验失败。审计文件为
`artifacts/lowlevel/pc_dade_grid_dade_supervised_residual_20261104/audit.json`。

## 五 seed × 32 评估 seed 正式结果（2026-11-04 18:37）

`220` 个正式闭环 summary 全部完成，所有 evaluation 的 `rc=0`；配对报告包含
`44` 个 aggregate，训练 seed 为 `94030–94034`，每个 tag 使用同一批评估 seed
`0–31`。报告脚本的早期版本把评估 seed 数误当成训练 seed 数，已修复为同时检查
`5` 个训练 seed 和每个 run 的 `32` 个评估 seed，随后在服务器重算；训练和评测均未
重跑。正式产物：

- 汇总：`artifacts/lowlevel/pc_dade_grid_summary_20261104/summary.json`；
- 报告：`artifacts/lowlevel/pc_dade_grid_report_20261104/report.{json,md}`。

表中 success、timeout、RMSE 和 saturation 均为五个训练 seed 的均值；Gate 使用
预注册门禁的实际判定，不做事后调整。

| Tag | Role | DADE success | no-DADE success | DADE RMSE | no-DADE RMSE | DADE timeout | DADE saturation | Gate |
|---|---|---:|---:|---:|---:|---:|---:|---|
| `grid_lw` | in-domain | `1.0000` | `0.9812` | `1.88 m` | `2.20 m` | `0.0000` | `0.1574` | `pass` |
| `grid_ls` | in-domain | `0.6813` | `0.6625` | `4.79 m` | `4.82 m` | `0.3187` | `0.1274` | `pass` |
| `grid_hw` | in-domain | `0.9750` | `0.9875` | `3.48 m` | `2.52 m` | `0.0250` | `0.4607` | **fail** |
| `grid_hs` | held-out | `0.9000` | `0.8438` | `4.90 m` | `6.11 m` | `0.1000` | `0.4382` | `pass` |
| `default` | retention | `0.6813` | `0.6375` | `4.96 m` | `5.20 m` | `0.3187` | `0.1171` | **fail** |
| `strict125` | retention | `0.4562` | `0.4062` | `6.36 m` | `6.63 m` | `0.5437` | `0.1074` | **fail** |
| `h1` | robustness | `0.9437` | `0.9250` | `3.76 m` | `4.36 m` | `0.0563` | `0.4455` | **fail** |
| `h2` | diagnostic | `0.9750` | `0.9875` | `3.39 m` | `2.69 m` | `0.0250` | `0.4827` | diagnostic |
| `h3` | robustness | `1.0000` | `0.9750` | `1.84 m` | `3.65 m` | `0.0000` | `0.5098` | **fail** |
| `h4` | pressure-only | `1.0000` | `0.9625` | `1.88 m` | `4.32 m` | `0.0000` | `0.5061` | pressure-only |
| `c4` | pressure-only | `0.2500` | `0.2062` | `8.27 m` | `8.83 m` | `0.7500` | `0.2155` | pressure-only |

配对差值的关键结论：

- `grid_hs`：相对同 seed 的 no-DADE，success `+0.0563`、RMSE 比 `0.8220`、
  saturation `+0.0105`；预注册的 transfer 条件、绝对 success `>=0.85` 和绝对
  RMSE `<=6.0 m` 均通过。这是明确的 held-out 正信号，但仍有 `0.10` timeout、
  `4.90 m` RMSE 和 `0.4382` saturation，不能表述为“高强格已解决”。
- `grid_lw`：success `+0.01875`、RMSE 比 `0.868`、saturation `-0.0264`，训练格
  保持并通过。
- `grid_ls`：success `+0.01875`、RMSE 比 `0.993`、saturation `-0.0156`，但绝对
  success 只有 `0.6813`、timeout `0.3187`，只是配对不退化，不能称为强流低速已达标。
- `grid_hw`：success 只下降 `0.0125`，但 RMSE 比 `1.523`（`2.52→3.48 m`），触发
  `RMSE <=1.10×` 门禁失败；说明 DADE 在高速度弱流训练格上引入了明显轨迹误差。
- `default`：success `+0.04375`、RMSE `-0.237 m`、timeout `-0.04375`，但绝对
  success `0.6813 < 0.70`，因此失败。这是“方向正确但未达线”，不是通过。
- `strict125`：success `+0.05`、RMSE `-0.268 m`、saturation `-0.0175`，但绝对
  success `0.4562 < 0.50`，timeout `0.5437 > 0.531`，仍失败。
- `h1`：success `0.9437 < 0.95`、saturation `0.4455 > 0.40`，失败；其 RMSE
  `3.76 m` 相对 no-DADE 的 `4.36 m` 有改善。
- `h3`：success `1.0000`、RMSE `1.84 m` 均过线，但 saturation `0.5098 > 0.45`，
  仍失败；这种提升主要依赖更大的执行器占用，不能当作干净收益。
- `h4/c4` 仅作压力测试：H4 success `1.0000`、RMSE `1.88 m`，但 saturation
  `0.5061`；C4 success 只有 `0.2500`、RMSE `8.27 m`、timeout `0.7500`，仍是明显
  的系统/教师边界。

**总判定：`overall_status=fail`。** DADE 在“训练中未出现的高速强流组合”上提供了
可复现的配对正信号，但训练格 `grid_hw` 的 RMSE 退化，default/strict125 没有过
绝对线，H1/H3/H4 的 saturation 仍超线。因此本轮只能支持“DADE 对 held-out 速度×
海流组合有迁移信号”，不能支持“DADE 已解决跨环境泛化或执行器饱和”。按预注册
规则，不通过挑选 seed、延长训练或继续同一 DADE-pose 配方改变结论。

后续优先级应是 actor 的实际地速跟踪、执行器权限/饱和奖励和闭环状态覆盖；DADE
不再作为主调参对象。若要继续 DADE，必须把它放进新的、预先注册的训练/奖励设计，
并继续使用同训练 seed、同评估 seed 的配对协议。

## 运行命令

smoke（1 个训练 seed、8 个评估 seed）：

```bash
SEEDS=94030 SEED_COUNT=8 TAGS='grid_hs default strict125' \
  EVAL_WORKERS=4 GPU_LIST='2' \
  bash scripts/run_pc_dade_grid_20261104.sh
```

正式五 seed（服务器 GPU 2/4/5）：

```bash
SEEDS='94030 94031 94032 94033 94034' SEED_COUNT=32 \
  TAGS='grid_lw grid_ls grid_hw grid_hs default strict125 h1 h2 h3 h4 c4' \
  EVAL_WORKERS=8 GPU_LIST='2 4 5' \
  setsid nohup bash scripts/run_pc_dade_grid_20261104.sh \
  > logs/pc_dade_grid_20261104/nohup.log 2>&1 < /dev/null &
```

## Git 提交与推送步骤

本实验不自动提交和推送。用户确认后，按文件粒度执行：

```bash
cd /home/bubble/水下机器人
git add \
  'settled_局部规划器_反应式/configs/lowlevel/odarrl_control_ppo_pc_grid_gpu_frozen_bc_residual_nodade_lr3e-4_scale1p0_dz64_gamma995_warmup5_critic3e-4_fixed_rewardscale001.json' \
  'settled_局部规划器_反应式/configs/lowlevel/odarrl_control_ppo_pc_grid_gpu_frozen_bc_residual_dade_pose_lr3e-4_scale1p0_dz64_gamma995_warmup5_critic3e-4_fixed_rewardscale001.json' \
  'settled_局部规划器_反应式/scripts/run_pc_dade_grid_20261104.sh' \
  'settled_局部规划器_反应式/scripts/summarize_pc_dade_grid_20261104.py' \
  'settled_局部规划器_反应式/scripts/eval_pc_server_basics_20261103.sh' \
  'settled_局部规划器_反应式/scripts/summarize_pc_server_evals_basics_20261103.py' \
  'settled_局部规划器_反应式/tests/test_pc_dade_grid_report.py' \
  'settled_局部规划器_反应式/doc/goals/goal13_dade_speed_current_grid.md' \
  'settled_局部规划器_反应式/doc/goals/goal9_odarrl_control_reproduction.md' \
  'settled_局部规划器_反应式/doc/goals/goal9_pc_server_validation_plan.md'
git diff --cached --check
git commit -m "feat(pc): evaluate DADE on held-out speed-current grid"
git push origin main
```

不要提交 `artifacts/`、`logs/`、checkpoint 或完整 rollout。
