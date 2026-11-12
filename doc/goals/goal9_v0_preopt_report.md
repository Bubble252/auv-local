# Goal9 S16-V0 pre-opt 诊断报告

更新时间：2026-11-05

状态：**V0 首次 screen 失败，但原因是训练配置使路径支路几乎关闭，不能作为
“ALOS-free 不可行”的科学负结果。**

## 1. 首次 V0 screen

- 训练配置：
  `configs/lowlevel/odarrl_control_ppo_pc_v0_fixed_speed_dz64.json`
- 训练预算：`32 iterations × 128 envs × 256 steps = 1,048,576` env steps
- 训练 seed：`91430`
- 评估 seed：`1000–1019`
- 评估 tags：`default`、`strict125`、`grid_hs`

结果：

| tag | success delta | RMSE ratio | timeout delta | 结论 |
|---|---:|---:|---:|---|
| default | `-0.75` | `3.7261` | `+0.75` | 全崩 |
| strict125 | `-0.35` | `2.1474` | `+0.35` | 全崩 |
| grid_hs | `-0.85` | `7.8069` | `+0.85` | 全崩 |

## 2. 机制诊断

训练记录显示路径支路始终接近关闭：

| iteration | `path_encoder_alpha` | final weight norm | effective output scale |
|---:|---:|---:|---:|
| 0 | `0.01` | `0.0000` | `0.0` |
| 5 | `0.01` | `0.0033` | `3.26e-05` |
| 10 | `0.01` | `0.0111` | `1.11e-04` |
| 20 | `0.01` | `0.0137` | `1.37e-04` |
| 31 | `0.01` | `0.0155` | `1.55e-04` |

同时 actor LR 在 warmup 后仍是 `1e-5`。V0 删除了 ALOS 的 `psi_d/theta_d`，路径特征是
actor 获得航向修正的主要来源；路径支路有效尺度只有 `1e-4`，等价于策略几乎看不到路径，
因此全部超时/失败是配置导致的结构性失败，而不是 V0 假设被否定。

## 3. 修正

采用 P 组已经验证的优化结论，并只作为 V 组训练能力修正：

- `path_encoder_alpha: 0.01 -> 1.0`
- `learning_rate_actor: 1e-5 -> 1e-4`

V0/V1 同步采用相同设置，保持两者只差 `u_d` 输入：

```text
configs/lowlevel/odarrl_control_ppo_pc_v0_fixed_speed_alpha1_lr1e-4_dz64.json
configs/lowlevel/odarrl_control_ppo_pc_v1_learned_speed_alpha1_lr1e-4_dz64.json
```

runner 增加配置门禁：V0/V1 的 `path_encoder_alpha >= 1.0` 且
`learning_rate_actor >= 1e-4`，防止再次用“路径关闭”的配置做科学 screen。

## 4. 结果处理

- 首次 V0 smoke/screen 产物移动到
  `artifacts/lowlevel/pc_alos_free_v_20261105/preopt/`，不覆盖、不删除；
- 该结果登记为 `preopt_path_closed`，不计入 V0 科学结论；
- 评估 seeds、tags、S15 基线、门禁阈值均未改变；
- 修正后的 V0 重新使用 canonical `V0_fixed_speed` 路径，从 smoke 开始重跑。

## 5. 产物

服务器：

```text
/data/yhx/underwater-robotics/src/plan_p1/settled_S16_copy/artifacts/lowlevel/pc_alos_free_v_20261105/
  preopt/V0_fixed_speed/                 # alpha=0.01, actor_lr=1e-5, fail_v0
  preopt/smoke_V0_fixed_speed/           # 对应 smoke
  baseline_reval/                        # A0 同口径重评估
  V0_fixed_speed/                        # 修正后的 canonical V0
```
