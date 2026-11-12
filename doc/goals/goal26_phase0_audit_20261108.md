# Goal26 Phase 0 审计与 zero-residual 证据

日期：2026-11-08<br>
状态：`PHASE0_PASS`；`PHASE1_ZERO_RESIDUAL_PASS`

## 1. success-first 冻结

本 Goal 以用户最新决定为准：

- 主门禁：success、safety、plane miss、timeout；
- 次级指标：pre-goal 3D RMSE 及完整路径诊断；
- 项目 `AGENTS.md` 中“主指标是 3D path RMSE”的历史文字不修改，只在本 Goal
  文档中声明 override；
- Goal25 的 D3 旧协议结论保持 FAIL，不追溯改判。

## 2. D3 只读基线

checkpoint：

```text
/dev/shm/pc_v1_goal25_d2_repair_20261108/bc_d3/pc_bc_actor.pt
SHA256 7430b26308b701565a52a92ed31c382ccca9cf1721d183ab856e3fafd9f4ab73
```

训练证据：

- D3 从 D0 初始化，在 D2 merged 数据上以 `2e-4` 微调；
- best epoch `10`，completed epochs `31`；
- actor state `12D`，path `8D`，path encoder `[64]`，`d_z=64`；
- path encoder alpha `1.0`；
- 无 command，无 current encoder，无 DADE；
- `frozen_bc_metadata` 在 Goal26 zero-residual smoke 中再次确认上述结构。

Goal25 的 40-seed dev 结果（`p_nav` success 口径）：

| scenario | success | pre-goal RMSE | full RMSE | plane miss | timeout | safety | saturation |
|---|---:|---:|---:|---:|---:|---:|---:|
| default | 0.7250 | 2.849170 | 2.763411 | 0.2750 | 0.0000 | 0.0000 | 0.191812 |
| strict125 | 0.5000 | 2.829135 | 2.752011 | 0.5000 | 0.0000 | 0.0000 | 0.191880 |

评估复用：

- matrix：`configs/lowlevel/eval/pc_v1_nominal_eval_matrix_goal25_d3_20261108.json`；
- matrix SHA256：`a59f5055ededa1d4ac681728da2f3b9f6819a0a58d5eef92cec1d9df58af21ff`；
- comparison SHA256：`03757899c968f0306ad45ab1e7fbf03fa3981bf0464e4e51144d1ec5478d40e8`；
- dev seeds：`20–59`；
- held-out：`100–149`，未使用；
- termination：`endpoint_stop`；
- sensor profile：`realistic_calibrated`；
- dtype：CPU/float64。

Goal25 原始产物未覆盖。

## 3. Goal26 配置

新增配置：

```text
configs/lowlevel/pc_goal26_env_default.json
configs/lowlevel/pc_goal26_env_strict125.json
configs/lowlevel/pc_goal26_env_default_p2.json
configs/lowlevel/pc_goal26_env_strict125_p2.json
configs/lowlevel/pc_goal26_ppo_p1_default.json
configs/lowlevel/pc_goal26_ppo_p2_default.json
```

配置 SHA256：

| file | SHA256 |
|---|---|
| `pc_goal26_env_default.json` | `220a27417c99f53c899b5fce8425585e4c13ba198b0d122c834dd2ccc883d0e4` |
| `pc_goal26_env_strict125.json` | `6e535f1633dc87fbde05b69fa82cbf4ca9bbe0560096234584e42ebc6cac0569` |
| `pc_goal26_ppo_p1_default.json` | 见运行 manifest（训练启动后记录） |
| `pc_goal26_ppo_p2_default.json` | 见运行 manifest（训练启动后记录） |

环境配置保持 `pc_control_mode=alos`，只新增默认关闭的
`reward.w_path_3d`。P1 为 `0.15`，P2 为 `0.30`；不切换
`path_learned_speed`，因此 D3 的 ALOS 环境保持完全可比。

代码扩展是向后兼容的：

- `RewardConfig.w_path_3d` 默认 `0.0`；
- ALOS 且 `w_path_3d>0` 时，在现有命令跟踪 reward 上追加
  `-w_path_3d * sqrt(cross^2 + vertical^2) / 5`；
- path-free 控制模式若设置 `w_path_3d>0` 会显式报错；
- 没有修改旧配置的默认 reward 行为。

验证：

```text
85 passed in 1.57s
```

覆盖 `test_lowlevel_pc_reward_termination.py`、`test_lowlevel_config.py`、
`test_lowlevel_pc_contract.py`、`test_lowlevel_pc_path_observations.py`。

## 4. 本地 reward 路径项 smoke

使用 Goal26 default 配置、4 env、1 step：

```text
path_cross:    [0.2049965, 0.5227055, 0.1212023, -0.1047228]
path_vertical: [0.3978432, 0.4407737, 0.0360, -0.3067643]
shaped_minus_unshaped: [-0.0134266, -0.0205122, -0.0037930, -0.0097244]
finite: true
```

说明 `w_path_3d=0.15` 确实进入 reward，且没有非有限值。

## 5. zero-residual 机制检查

本地 smoke：

```text
initial_residual_zero_applied = true
initial_residual_abs_max = 0.0
residual_output_zeroed = true
frozen_parameters_require_grad = false
path_encoder_alpha = 1.0
```

同 seed `20` 的 PPO evaluator 对照：

| action mode | success | pre-goal RMSE | residual abs max |
|---|---:|---:|---:|
| `frozen-base` | 1.0 | 0.610456 | 0.0 |
| `policy` | 1.0 | 0.610456 | 0.0 |

118 个可比较数值指标全部一致，`max_abs_diff=0.0`。这是一项机制检查，不是
40-seed 性能结论。

## 6. 服务器资源

服务器：

- host：`zhulab3-AS-4125GS-TNRT`；
- 8× RTX 4090；
- Goal26 使用隔离目录
  `/data/yhx/underwater-robotics/src/plan_p1/goal26_success_first_20261108/`；
- Python `/data/yhx/underwater-robotics/envs/venvs/isaaclab/bin/python`；
- 训练使用 `CUDA_VISIBLE_DEVICES=1`；
- 不与既有 GPU 0/2/3/4/6 任务冲突。

服务器 smoke：

```text
64 env × 8 steps: completed, 512 timesteps
4096 env × 128 steps: 1 iteration, 3.41 s, 153,756 steps/s collection
```

预估 200 轮约十几分钟，因此继续执行正式 P1 screen。

## 7. 裁决

- Phase 0：PASS；
- zero-residual：PASS；
- 可以启动 P1 PPO screen；
- 未使用 held-out；
- 未覆盖 Goal24/25 产物；
- 不自动 commit/push。
