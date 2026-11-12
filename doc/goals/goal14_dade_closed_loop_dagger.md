# DADE 闭环 DAgger 与高速强流留出验证

记录日期：2026-11-04

状态：**已完成，整体门禁 fail。** 五训练 seed × 32 评估 seed 的采集、合并、
监督训练、冻结 BC 审计、DADE 基线评测和 DAgger 配对评测全部完成；没有 NaN/Inf、
安全终止异常或缺失 checkpoint。DAgger 相对同 run 的零残差 `initial` 有学习效果，
但没有超过同 seed 的 DADE-pose 基线，`grid_hs` 留出收益没有复现。

## 背景

`goal13` 已经在显式速度×海流网格上完成 DADE-pose 与 no-DADE 的五 seed 配对。
两臂都只用 `grid_lw/grid_ls/grid_hw` 训练，`grid_hs`（高速 + 强流）严格留出。

现有 DADE-pose 结果给出一个可继续利用的信号：`grid_hs` 相对 no-DADE 的
success 为 `0.9000 / 0.8438`，RMSE 为 `4.90 / 6.11 m`，说明 DADE 对未见过的
速度-海流组合有迁移收益；但 `default`、`strict125`、H1/H3/H4 仍没有过绝对
门禁，执行器 saturation 也仍在上限附近。

上一轮 no-DADE DAgger 在 `grid_hs` 上只有弱正信号：success `0.8313→0.8500`、
RMSE `6.54→5.86 m`，没有达到预注册阈值。这个结果说明“补学生闭环状态标签”
方向有价值，但普通监督残差没有把 DADE 的估计能力带入动作。本实验把两者组合：
以 DADE-pose 学生为采集策略，补一轮闭环 DAgger，再看 held-out 高速强流和保留
环境是否同时改善。

## 实验问题

在同一个三格训练域、同一个冻结 BC、同一组 DADE 学生 checkpoint 和同一组评估
seed 下，给 DADE-pose 残差补一轮学生闭环教师标签，能否：

1. 在完全未见过的 `grid_hs` 上继续提高 success 或降低 RMSE；
2. 不破坏 `grid_lw/grid_ls/grid_hw` 的训练格；
3. 把 default/strict125 向绝对门禁推进；
4. 不进一步抬高 H1/H3/H4 的 material saturation。

## 技术路线

```text
DADE-pose 学生 final.pt
    -> 在 lw/ls/hw 三格闭环执行 student + 小幅动作噪声
    -> 在访问到的学生状态上采集 teacher 动作
    -> 与原 teacher demos 合并
    -> 用同一 DADE-pose 结构重新监督残差
    -> initial/final 固定 seed 闭环配对评测
```

这是 DAgger 的一轮闭环状态覆盖补充，不新增第二个残差网络，不改变 actor 输入
维度，不把真实海流送入 actor，也不训练新的环境。

## 固定协议

- 训练学生：`frozen_bc_residual_dade_grid_gpu_pose_lr3e-4_scale1p0` 五个 seed
  `94030–94034` 的 `final.pt`；
- 训练域：`grid_lw`、`grid_ls`、`grid_hw`，共三格；
- 留出域：`grid_hs`，只用于评测，严格不进入 DAgger 采集和训练；
- DAgger 采集：`16 env × 2048 step × 3 config × 5 seed`；
- 动作噪声：归一化动作 `action_noise_std=0.05`，用于扩大学生闭环状态覆盖；
- 新残差训练：与原 DADE-pose 配置相同的结构、学习率、验证切分和早停；
- 评估 seed：`0–31`；训练 seed：`94030–94034`；
- 对照：同训练 seed 的 DADE-pose grid `final.pt`，不是重新训练的 no-DADE；
- 同时保留 `initial/final`，用于检查 DAgger 相对同一初始化是否真的带来收益。

## 产物

- 配置：`configs/lowlevel/odarrl_control_ppo_pc_grid_dade_dagger_frozen_bc_residual_dade_pose_lr3e-4_scale1p0_dz64_gamma995_warmup5_critic3e-4_fixed_rewardscale001.json`
- 队列：`scripts/run_pc_dade_grid_dagger_20261104.sh`
- 采集数据：`artifacts/lowlevel/pc_dade_grid_dagger_demos_20261104/`
- 训练结果：`artifacts/lowlevel/pc_dade_grid_dagger_supervised_residual_20261104/`
- 闭环评测：`artifacts/lowlevel/pc_dade_grid_dagger_eval_20261104/`
- 汇总：`artifacts/lowlevel/pc_dade_grid_dagger_summary_20261104/`
- 配对报告：`artifacts/lowlevel/pc_dade_grid_dagger_report_20261104/`
- 日志：`logs/pc_dade_grid_dagger_20261104/`

## 预注册门禁

所有比较使用同训练 seed、同评估 seed 的 DADE-DAgger `final` 对 DADE-pose
`final`，并额外报告 DADE-DAgger `final` 对同 run `initial`。

### 高速强流留出格 `grid_hs`

- success 至少提高 `+0.02`，或 RMSE 不超过基线的 `0.95×`；
- 绝对 success `>=0.90`；
- 绝对 RMSE `<=5.0 m`；
- saturation 增量不超过 `+0.02`，绝对 saturation `<=0.45`；
- speed/yaw tracking ratio 不超过 `1.20×`。

### 训练格 `grid_lw/grid_ls/grid_hw`

- success 下降不超过 `0.02`；
- RMSE 不超过 `1.10×`；
- saturation 增量不超过 `+0.05`。

### 保留环境

- `default`：success `>=0.70`、RMSE `<=5.0 m`，且 timeout 不高于 DADE 基线；
- `strict125`：success `>=0.50`、RMSE `<=6.5 m`、timeout `<=0.531`；
- `H1`：success `>=0.95`、RMSE `<=5.0 m`、saturation `<=0.40`；
- `H3`：success `>=0.95`、RMSE `<=3.4 m`、saturation `<=0.45`。

`H2` 只作诊断，`H4`/`C4` 只作压力测试。若 `grid_hs` 没有相对 DADE 基线达到
`+0.02` 或 `0.95×`，该轮直接登记为失败，不追加同配方轮次、不反向挑选 seed。

## 执行顺序

- [x] 先在 GPU2 做 `94030 × 8 eval seed` 的 smoke，tags 为
   `grid_lw grid_ls grid_hw grid_hs default strict125`；
- [x] smoke 必须完成采集、合并、训练、冻结 BC 漂移审计和自动评测，且不能出现
   safety、NaN/Inf 或缺失 checkpoint；
- [x] smoke 通过后，在 GPU2/4/5 跑五 seed × 32 评估 seed，tags 为
   `grid_lw grid_ls grid_hw grid_hs default strict125 h1 h2 h3 h4 c4`；
- [x] 自动汇总后按上述门禁生成 `report.{json,md}`；即使门禁失败也保留全部结果。

Smoke：

```bash
cd /data/yhx/underwater-robotics/src/plan_p1/settled_局部规划器_反应式
SEEDS=94030 SEED_COUNT=8 RUN_SUFFIX=_smoke TAGS='grid_lw grid_ls grid_hw grid_hs default strict125' \
CHECKPOINTS='initial final' EVAL_WORKERS=4 GPU_LIST='2' ACTION_NOISE_STD=0.05 \
setsid nohup bash scripts/run_pc_dade_grid_dagger_20261104.sh \
  > logs/pc_dade_grid_dagger_20261104_smoke/nohup_smoke.log 2>&1 < /dev/null &
```

正式五 seed：

```bash
cd /data/yhx/underwater-robotics/src/plan_p1/settled_局部规划器_反应式
SEEDS='94030 94031 94032 94033 94034' SEED_COUNT=32 RUN_SUFFIX= \
TAGS='grid_lw grid_ls grid_hw grid_hs default strict125 h1 h2 h3 h4 c4' \
CHECKPOINTS='initial final' EVAL_WORKERS=8 GPU_LIST='2 4 5' ACTION_NOISE_STD=0.05 \
setsid nohup bash scripts/run_pc_dade_grid_dagger_20261104.sh \
  > logs/pc_dade_grid_dagger_20261104/nohup_full.log 2>&1 < /dev/null &
```

检查：

```bash
tail -f logs/pc_dade_grid_dagger_20261104/queue.log
pgrep -af 'run_pc_dade_grid_dagger|collect_pc_dagger|pretrain_pc_residual|evaluate_pc_ppo'
nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu --format=csv
```

## 执行结果（2026-11-04，5 seed × 32 eval）

训练审计 `failures=[]`，五个 run 的 `frozen_bc_parameter_max_abs_drift` 全为
`0.0`；每个 run 的唯一训练格都是 `lw/ls/hw`，每个格各有原始 demos 和 DAgger
demos 两路来源，`grid_hs` 未进入采集或训练。DADE 基线和 DAgger 各有 `110/110`
个闭环评测产物，汇总 `220` 行，报告与汇总均无非有限值。

| Tag | Role | DAgger success / RMSE / timeout / sat | DADE reference success / RMSE / timeout / sat | Gate |
|---|---|---:|---:|---|
| `grid_lw` | in-domain | `0.9750 / 2.16 m / 0.0250 / 0.1658` | `1.0000 / 1.88 m / 0.0000 / 0.1574` | fail |
| `grid_ls` | in-domain | `0.6438 / 4.94 m / 0.3563 / 0.1344` | `0.6813 / 4.79 m / 0.3187 / 0.1274` | fail |
| `grid_hw` | in-domain | `0.9938 / 2.19 m / 0.0063 / 0.4736` | `0.9750 / 3.48 m / 0.0250 / 0.4607` | pass |
| `grid_hs` | held-out | `0.8875 / 5.38 m / 0.1125 / 0.4428` | `0.9000 / 4.90 m / 0.1000 / 0.4382` | fail |
| `default` | retention | `0.6625 / 5.10 m / 0.3375 / 0.1244` | `0.6813 / 4.96 m / 0.3187 / 0.1171` | fail |
| `strict125` | retention | `0.4375 / 6.45 m / 0.5625 / 0.1150` | `0.4562 / 6.36 m / 0.5437 / 0.1074` | fail |
| `h1` | robustness | `0.9250 / 4.60 m / 0.0750 / 0.4483` | `0.9437 / 3.76 m / 0.0563 / 0.4455` | fail |
| `h2` | diagnostic | `0.9812 / 2.98 m / 0.0187 / 0.4954` | `0.9750 / 3.39 m / 0.0250 / 0.4827` | diagnostic |
| `h3` | robustness | `1.0000 / 1.83 m / 0.0000 / 0.5196` | `1.0000 / 1.84 m / 0.0000 / 0.5098` | fail |
| `h4` | pressure-only | `0.9938 / 2.61 m / 0.0063 / 0.5126` | `1.0000 / 1.88 m / 0.0000 / 0.5061` | pressure-only |
| `c4` | pressure-only | `0.2500 / 8.46 m / 0.7500 / 0.2275` | `0.2500 / 8.27 m / 0.7500 / 0.2155` | pressure-only |

配对差值：

- `grid_hs`：success `-0.0125`（配对范围 `-0.0625~+0.03125`）、RMSE ratio
  `1.1018`、saturation `+0.0046`；success 和 RMSE ratio 都没有达到
  `+0.02`/`0.95×` 的迁移门禁，绝对 success `0.8875 < 0.90`、绝对 RMSE
  `5.38 > 5.0 m`，因此 fail；
- 训练格：`grid_lw` success `-0.0250`、RMSE ratio `1.1488`，`grid_ls`
  success `-0.0375`、RMSE ratio `1.0312`；只有 `grid_hw` 的 success `+0.0188`
  且 RMSE ratio `0.6715` 通过；
- `default` success `-0.0188`、RMSE `+0.137 m`、timeout `+0.0188`；
  `strict125` success `-0.0188`、RMSE `+0.084 m`、timeout `+0.0188`，均未过
  绝对门禁；
- H1 success `-0.0188`、RMSE ratio `1.2451`，saturation `0.4483 > 0.40`；
  H3 success 和 RMSE 基本持平，但 saturation `0.5196 > 0.45`；
- 与同 run 的 `initial` 相比，DAgger 确实学到了残差：`grid_hs` success
  `+0.0438`、RMSE ratio `0.8885`，H3 success `+0.15625`、RMSE ratio
  `0.1828`。但这些收益在 DADE-pose 基线中已经存在，DAgger 没有提供增量。

判定：**S13 / DADE 闭环 DAgger fail。** 本轮没有在 `grid_hs` 上复现或扩大
DADE 的 held-out 收益，反而在训练格 `grid_lw/grid_ls`、default/strict125 和
H1 上退化；H3/H4 的 saturation 仍在 `0.51–0.52`，执行器门禁没有改善。按预注册
规则停止同配方轮次，不追加 DAgger 第二轮、不调动作噪声或 DADE dropout/head 来
挑 seed。下一步应转向实际地速跟踪、执行器权限/饱和奖励或新的控制层设计，而不是
继续重复“DADE + 监督 DAgger”。

产物：`artifacts/lowlevel/pc_dade_grid_dagger_{demos,supervised_residual,eval,summary,report}_20261104/`；
审计：`artifacts/lowlevel/pc_dade_grid_dagger_supervised_residual_20261104/audit.json`；
报告：`artifacts/lowlevel/pc_dade_grid_dagger_report_20261104/report.{json,md}`。

## Git 与提交规则

本实验不自动 commit/push。用户确认后按文件粒度提交：

```bash
cd /home/bubble/水下机器人
git add \
  'settled_局部规划器_反应式/configs/lowlevel/odarrl_control_ppo_pc_grid_dade_dagger_frozen_bc_residual_dade_pose_lr3e-4_scale1p0_dz64_gamma995_warmup5_critic3e-4_fixed_rewardscale001.json' \
  'settled_局部规划器_反应式/scripts/run_pc_dade_grid_dagger_20261104.sh' \
  'settled_局部规划器_反应式/scripts/eval_pc_server_basics_20261103.sh' \
  'settled_局部规划器_反应式/scripts/summarize_pc_dade_grid_20261104.py' \
  'settled_局部规划器_反应式/scripts/summarize_pc_server_evals_basics_20261103.py' \
  'settled_局部规划器_反应式/tests/test_pc_dade_grid_report.py' \
  'settled_局部规划器_反应式/tests/test_pc_server_eval_summary.py' \
  'settled_局部规划器_反应式/doc/goals/goal14_dade_closed_loop_dagger.md' \
  'settled_局部规划器_反应式/doc/goals/goal9_pc_server_validation_plan.md'
git diff --cached --check
git commit -m "feat(pc): add DADE closed-loop DAgger experiment and report"
git push origin main
```

不要提交 `artifacts/`、`logs/`、checkpoint 或完整 rollout。
