# Goal9 S16-V0 正式 screen 报告

更新时间：2026-11-05

状态：**V0 fixed-speed / ALOS-free screen 失败。路径支路已打开，但 60 个评估 episode
全部 timeout，按预注册门禁 V0=`fail_v0`；V1 被阻断，不再用加预算、换 seed 或改门禁
抢救。**

## 1. 实验设置

| 项目 | 值 |
|---|---|
| training config | `configs/lowlevel/odarrl_control_ppo_pc_v0_fixed_speed_alpha1_lr1e-4_dz64.json` |
| config SHA-256 | `987f344566b3f05a8d77d6f175900353eb23d2f66f2526e1e333c49923f65b60` |
| actor command | `1D`，只含 `u_d` |
| critic command | `1D` |
| path / history path | `8 / 8` |
| history | short `10` / long `50`，stride `1` |
| DADE history mode | `encoded` |
| path encoder alpha | `1.0` |
| actor LR | `1e-4` |
| 训练预算 | `32 iterations × 128 envs × 256 steps = 1,048,576` env steps |
| 训练 seed | `91430` |
| 评估 seed | `1000–1019` |
| 评估 tags | `default`、`strict125`、`grid_hs` |
| S15 基线 | `A0_stride1`，同口径重新评估 |

基线重评估只补齐当前 evaluator 的 `time_to_goal_s` 和速度包线字段；success、timeout、
RMSE、saturation 与旧 A_screen summary 完全一致。

## 2. 结果

| tag | success delta | V0 success | 3D RMSE ratio | timeout delta | saturation delta | safety delta |
|---|---:|---:|---:|---:|---:|---:|
| default | `-0.75` | `0/20` | `2.7928` | `+0.75` | `-0.0361` | `0.0` |
| strict125 | `-0.35` | `0/20` | `1.6095` | `+0.35` | `-0.0148` | `0.0` |
| grid_hs | `-0.90` | `0/20` | `5.3778` | `+0.90` | `-0.2290` | `0.0` |

三组共 60 个评估 episode，success `0`、timeout `60`、safety termination `0`。

## 3. 路径通路检查

修正后的训练中路径支路确实打开：

| iteration | effective path output scale |
|---:|---:|
| 5 | `0.0113` |
| 10 | `0.0304` |
| 20 | `0.0677` |
| 31 | `0.1380` |

作为对照，失败 pre-opt 配置在第 31 iteration 只有 `1.55e-4`，基本关闭。因此本次
V0 失败不是“路径编码器仍为 0”导致的同一类实现问题。

## 4. 训练行为

- 训练中偶有批次 success 非零，但评估 seed 上完全不能泛化；
- 训练 reward 在后期仍约 `-30`，没有收敛到可部署的路径跟踪行为；
- material saturation 在训练后期约 `0.4–0.5`，但按用户最新取舍只作为风险记录，不参与
  V0 通过/否决；即使忽略 saturation，success/timeout/RMSE 仍远低于门禁。

## 4.1 失败原因复核（2026-11-06）

本节只补充诊断，不改变 `fail_v0` 的历史结论。

**更上游的主因：V0 的实验定义偏离了用户确认的 V 系列契约。** V 系列初始应保留 3D
命令 `[psi_d, theta_d, u_d]`，只在 `o_t` 中增加路径观测；执行版 V0 却直接改成
`[u_d]`、移除 `psi_d/theta_d`，并且采用完整策略。这个设计变更没有单独预注册，也没有
先建立对应输入契约的 BC 基座。

在这个错误契约下，直接机制是：原 PC BC checkpoint 的命令列是
`[psi_d, theta_d, u_d]`，而 V0 只保留 `[u_d]`。加载器只复制同名命令列，因此 V0 初始化
时直接丢掉了 BC 最依赖的航向和俯仰参考，见 `src/auv_lowlevel/batch/pc_bc.py:641`。

BC actor 首层命令列权重范数可以量化这个依赖：

| BC 命令 | 首层权重 L2 |
|---|---:|
| `psi_d` | `7.0345` |
| `theta_d` | `4.9599` |
| `u_d` | `1.3918` |

也就是说，BC 的主要方向控制来自 `psi_d/theta_d`，速度命令只占较小部分。V0 删除这两个
输入后，初始策略不再具备原来的转向/俯仰参考；路径编码器加载 BC 时又必须保持零输出，
见 `src/auv_lowlevel/batch/pc_bc.py:682`。因此 V0 一开始既缺方向命令，也没有已经学会
使用的路径特征，PPO 必须同时学会“用路径替代制导命令”和“保持闭环跟踪”。

这解释了训练后期的 `reward≈-30`、评估 60/60 timeout 和 RMSE ratio 大幅恶化。第 31
iteration 路径输出尺度 `0.1380` 只证明路径支路数值上打开了，不证明 actor 已经学会用它；
V0 也没有执行 `path-zero` 闭环消融来证明路径归因。

**非主因：**

- 不是奖励尺度：P 组 X3 已把 `pc_ppo_scale` 从 `0.01` 改到 `1.0`，闭环没有改善；
- 不是 saturation：忽略 saturation 后 success/timeout/RMSE 仍然失败；
- 不能只归因于“步数少”：32 iterations / 1,048,576 环境步是风险因素，但 V0 没有单独的
  预算消融，不能把无限加步数当成修复方案。

**修复方向（2026-11-06 最终确认）：** 直接以 `V-0D` 纯路径输入为主臂，不再要求先走
`V-3D → V-2D → V-1D`。正确路线是新增同契约的 `R-BC-0D`：
`state + 8D path -> action`、`command_dim=0`，训练后冻结，再执行
`R-Z0-0D → R-PPO-0D`。不能用现有 3D BC checkpoint 加 `command_mask` 冒充 0D，因为
冻结 BC 的 `base_action()` 仍读取 command，会把 `psi_d/theta_d/u_d` 一起清零。
`V-3D` 只保留为 0D 失败后的可选诊断控制，不是前置门禁。权威计划见
`doc/goals/goal9_v0d_direct_pure_path_plan_20261106.md`。旧 V0 的 1D + 完整策略不得
作为新 V 基线，也不能直接靠加训练步数抢救。

## 5. 判定

```text
stage = S16-V0
status = fail_v0
passed = false
best_success_gain = -0.35
best_rmse_improvement_m = -3.9264
timeout_increase = +0.75
material_saturation_policy = reported_not_gated
next_action = stop_v_branch_and_preserve_alos_external
```

## 6. V1/V1b/V2/V3

V0 未通过，按协议：

- V1 不启动；
- V1b/V2 不启动；
- V3 不启动；
- 当前最可靠的部署方案仍是 ALOS 外置制导 + 已冻结的 S15 A0 策略。

若以后要重新研究 ALOS-free，应新开预注册分支，明确改变课程/奖励/训练预算或网络容量，
不能把本次 V0 结果通过事后调参改写成成功。

## 7. 证据路径

服务器根目录：

```text
/data/yhx/underwater-robotics/src/plan_p1/settled_S16_copy/artifacts/lowlevel/pc_alos_free_v_20261105/
```

- 修正后 V0 manifest：
  `V0_fixed_speed/run_manifest.json`
- summary：
  `V0_fixed_speed/report/summary.{json,md}`
- checkpoint：
  `V0_fixed_speed/training/V0_fixed_speed_seed91430/final.pt`
- checkpoint SHA-256：
  `78f514cd9112baf32f03d5a8c97ca4828a4b10cd7cc91b318f45a786e7ad4cbc`
- 训练诊断：
  `V0_fixed_speed/training/V0_fixed_speed_seed91430/iteration_records.json`
- pre-opt 失败产物：
  `preopt/V0_fixed_speed/`、`preopt/smoke_V0_fixed_speed/`
- 同口径 S15 基线：
  `baseline_reval/screen/A0_stride1/`
