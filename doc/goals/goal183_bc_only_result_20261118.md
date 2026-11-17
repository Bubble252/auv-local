# Goal183 BC_ONLY 结果（2026-11-18）

**Goal**：Goal183<br>
**状态**：`BC_RETRAIN_DRIFT_ON_OLD_DEMO / C_TEACHER_MATCH / PPO_SMOKE_PENDING`<br>
**完整协议**：`doc/goals/goal183_env_empirical_bc_20261117.md`

## 1. 结论

env-side empirical normalization 的实现和 PPO checkpoint 契约可用，但 empirical PE2E BC 在 BC_ONLY 闭环保真中明显退化。按预注册失败裁决，本轮停止在 BC_ONLY，不启动 72-iteration PPO，不通过 seed、LR、std 或 Gate 改动救结果。

## 2. BC_ONLY 对照

同一 C3 环境、seeds `20–23`、`path_dim=5`、`termination_mode=endpoint_stop`、`current_ablation=none`。success/timeout/pre-goal 3D 只是 telemetry。

| BC | success | 3D RMSE | horizontal | vertical | material saturation | pre-goal 3D (telemetry) |
|---|---:|---:|---:|---:|---:|---:|
| fixed PE2E current | 4/4 | 1.9841 m | 1.9060 m | 0.5514 m | 0.2091 | 2.0638 m |
| empirical PE2E current | 1/4 | 3.1209 m | 3.0474 m | 0.6735 m | 0.1774 | 3.3621 m |

fixed 对照使用当前代码在本地重跑，和 2026-11-17 服务器记录逐项一致，排除 fixed 路径回归。

## 3. 根因

- 两个 BC 使用同一 demo SHA、同一 `episode_id` split、同一网络和训练超参数；
- empirical checkpoint 的 mode、schema hash、statistics 均严格加载；
- 退化集中在 current 组：300 个 C3 teacher steps（1200 样本）中，current 特征有 `25.33%` 满足 `abs(z)>3`，state 为 `0.12%`，path 为 `0%`；
- PE2E demo 采集使用 `realistic_calibrated`，BC_ONLY/PPO 环境使用 `unbiased_small_noise_v2`；demo current 分布更窄，empirical std 把环境 current 放大成 OOD。

因此当前失败不是 normalization 位置或 checkpoint 加载错误，而是 demo current 统计与训练/评价环境 current 分布冲突。

## 4. Contract smoke

1536 envs、1 iteration：

| 指标 | 结果 | Gate |
|---|---:|---:|
| `losses_finite` | true | 必须 true |
| `approx_kl` | 0.005010 | `<= 0.02` |
| `clip_fraction` | 0.054077 | `<= 0.25` |
| `ratio_max` | 1.625438 | `<= 6.0` |
| peak reserved | 8438 MiB | 可用预算内 |

checkpoint `infos` 中的 normalization mode、position、actor/critic schema hash 和 statistics 均可审计；actor count 为 `354690 + 73728`，critic count 为 `73728`。

64-env smoke 只有 3072 samples，KL `0.023468`、clip `0.259115`，不作为 Gate 判读。

## 5. 证据

- empirical BC：`/home/bubble/水下机器人/_goal183_artifacts/pe2e_bc_empirical/pc_bc_actor.pt`，SHA-256 `551ab6be402256c1123788856eba8dd064687f5ed993652ce4e1cce64ed003e8`；
- BC_ONLY empirical：`/home/bubble/水下机器人/_goal183_artifacts/eval/pe2e_bc_empirical_c3_seeds20_23/summary.json`；
- BC_ONLY fixed control：`/home/bubble/水下机器人/_goal183_artifacts/eval/pe2e_bc_fixed_local_c3_seeds20_23/summary.json`；
- 1536-env smoke：`/data/yhx/underwater-robotics/runs/auv-local/goal183/20261118-goal183-env-empirical-bc/smoke/ppo_contract_1it_1536env/iteration_records.json`；
- ledger：`research/LEDGER.jsonl` 的 `2026-11-18 / Goal183` 条目。

## 6. 待定夺

1. 停止 env-side empirical 线，BCFT 保持 fixed normalization；
2. 保留 env-side empirical，但 current 组继续使用 fixed scale，或改用训练环境 rollout 估计 current 统计后重训 BC；
3. 重采与 `unbiased_small_noise_v2` 匹配的 PE2E demo；这与第一轮“不重采 demo”的约束冲突。

在方案确认前，不启动 72-iteration。

## 7. 用户决议：方案 C

2026-11-18 用户选择方案 C：重采与 `unbiased_small_noise_v2` 匹配的 C3 PE2E demo，再用原
empirical BC 配方重训。方案 B（fixed current / 训练环境 current 统计）不再执行。

C 的首个判据仍是同一 BC_ONLY 四 seed 对照；只有恢复后才进入 PPO smoke。若 C 仍失败，本线停止，
不通过 std、LR、seed、route 或 Gate 修改救结果。

## 8. 方案 C 结果（2026-11-18）

新 demo：`209` episodes / `262144` samples，`unbiased_small_noise_v2`，seeds `90000–90063`；
SHA-256 `c8e86cd36a3ca83f7b9d7eec5e82f62ca600e4676b9972d27ad425d79092ec01`。
新 checkpoint：SHA-256 `ce4290f90b8fb19d8e84984c4f335400076c5f02892f3cc26e2f33b37e708582`。

| 口径 | 3D RMSE | horizontal | vertical |
|---|---:|---:|---:|
| fixed / 旧 demo | 1.9841 m | 1.9060 m | 0.5514 m |
| empirical / 旧 demo | 3.1209 m | 3.0474 m | 0.6735 m |
| empirical / 新 demo | 2.4459 m | 2.4265 m | 0.3075 m |
| teacher / 同环境 | 2.4133 m | 2.3955 m | 0.2928 m |

新 demo 把 current `abs(z)>3` 从 `25.33%` 降到 `0.00%`；state `1.65%`、path `0.58%`。新 BC
与 teacher 对齐，但比 fixed 的 3D 差 `0.4618 m`。旧 demo 中 seed `20` 的一个 episode 曾进入
fixed BC training split，新 demo 不含 `20–23`，所以 fixed 不是干净同数据对照。判为
`SCREEN_PASS_WITH_REFERENCE_GAP`，只允许进入 1-iteration PPO smoke，不写成 formal PASS。

C 的 PPO warmstart 配置已另存为
`configs/lowlevel/goal183/train_c0_diagnostic_mod4_pe2e_bcft_env_empirical_5d_c_recollect_seed86201.json`；
旧配置未改写。PPO smoke 尚未启动。
