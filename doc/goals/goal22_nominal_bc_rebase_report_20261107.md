# Goal22：nominal 传感器域 BC 重采与重训报告

日期：2026-11-07

状态：**已完成执行；预注册门禁 FAIL；未进入 P3/P5/理想 actor 臂**

## 1. 结论

Goal22 的 N8/N12 数据采集、BC 训练、离线审计和 dev `20–39` 评估均已完成，但
N12 相对 N8 的 default 闭环门禁未通过。正式汇总结果：

```text
overall_pass = false
PASS = 9
FAIL = 4
```

失败项为：

- N12 default `pre_goal_3d_rmse_m` 未达到 `<= 0.95 × N8`；
- N12 default success 下降 `0.10`，超过允许的 `0.05`；
- N12 default goal capture 下降 `0.10`；
- N12 default `goal_plane_miss_rate` 增加 `0.10`，超过允许的 `0.05`。

按 Goal22/Goal23 的预注册停止线，**不启动 P3 定向 DAgger、不启动 P5 残差 PPO、
不启动理想 actor 完整对照臂、不评估 held-out**。N8 仅保留为未来新 Goal 的候选底座，
本次没有继续训练或扩 seed。

## 2. 固定实验条件

- 传感器域：`realistic_calibrated` nominal；
- `guidance.current_estimate_max_age_s=0.9`；
- 终止协议：`endpoint_stop`；
- default goal radius：`2.0 m`；
- strict125 goal radius：`1.25 m`；
- 教师目标：名义上沿用 `r2b current`，crab/current 研究因子保持不变；
- default 采集：`64 env × 4096 steps`，seeds `90000–90063`；
- strict125 采集：`32 env × 4096 steps`，seeds `90100–90131`；
- 评估：dev seeds `20–39`，CPU、float64；
- held-out `100–149`：未触碰。

2026-11-08 追加复核发现：新生成的 `v1_nominal_bc_*` 配置没有显式写入
`guidance.current_crab_enabled`、`guidance.robust_terminal_enabled` 和
`guidance.route_speed_floor_enabled`，配置加载器对三者默认取 `false`；旧
`v0d_teacher_r2b_current` 配置则分别为 `true/true/true`。因此新数据名义上是
`r2b current`，实际教师控制路径却与旧 g19 的 R2B current **不完全等价**。这不改变
N8 与 N12 的内部 8D/12D 对照（两者配置相同），但会污染新旧 BC 的跨代比较和失败
归因，后续必须修正或至少做显式配置对照后才能声称是干净的重基座。

## 3. 数据集与 checkpoint

两个 arm 均合并 default + strict125 数据，训练配方完全相同，只改变 path_dim。

| arm | path_dim | samples | episodes | dataset SHA256 | best epoch | best val loss | checkpoint SHA256 |
|---|---:|---:|---:|---|---:|---:|---|
| N8 | 8 | 393216 | 317 | `a4d48a989bf9f7b45e2497a6900924b4fe1765df71b24ac7650661a14daff9ad` | 131 | 0.0148185156 | `9dd416a121f471c5b1aadafb9d591e8259b9803a8bbdf6650ce14e6b49664e36` |
| N12 | 12 | 393216 | 317 | `f72e9addbc0a8b4e33ac684c88a40a4d28bd7f4bcbda163e40c53777f6de81f1` | 190 | 0.0094947722 | `57ecfc773971113ed34a1214440830de0065885e7cc524713bb257c102acbb34` |

固定 BC 配方：state `128,128`，actor `256,128`，`d_z=64`，batch `4096`，
max epochs `300`，patience `30`，lr `1e-3`，weight decay `1e-5`，seed `0`，
saturation weight `2.0`，threshold `0.9`，validation fraction `0.15`。

正式 checkpoint 使用 GPU 训练并在 CPU 评估；manifest 中的 checkpoint 路径指向
`bc_*_cuda`，规范副本位于 `/dev/shm/.../bc_N8/` 和 `/dev/shm/.../bc_N12/`，
SHA256 与上表一致。

## 4. dev `20–39` 主结果

`success/capture` 使用可部署的 `p_nav` 主口径；`success_only_3d_rmse_m` 是成功
episode 的 RMSE 离线派生值，只作诊断。`post_goal_overshoot_m` 与
`terminal_longitudinal_rmse_m` 在 `endpoint_stop` 下按协议为 `null`。

| arm | scenario | pre-goal RMSE | full RMSE | success-only RMSE | success | capture | plane miss | timeout | safety | saturation |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| N8 | default | 2.8237 | 2.7655 | 2.6784 | 0.650 | 0.650 | 0.350 | 0.000 | 0.000 | 0.1108 |
| N8 | strict125 | 2.8044 | 2.7494 | 3.1936 | 0.250 | 0.250 | 0.750 | 0.000 | 0.000 | 0.1090 |
| N12 | default | 2.8810 | 2.8072 | 2.9399 | 0.550 | 0.550 | 0.450 | 0.000 | 0.000 | 0.1035 |
| N12 | strict125 | 2.8664 | 2.7942 | 2.8888 | 0.350 | 0.350 | 0.650 | 0.000 | 0.000 | 0.1029 |

旧 `g19_dagger` nominal 参考（仅作护栏对照，不是本 Goal 的训练臂）：

| arm | scenario | pre-goal RMSE | full RMSE | success-only RMSE | success | capture | plane miss | timeout | safety | saturation |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| g19_dagger | default | 2.5927 | 2.5172 | 2.0424 | 0.700 | 0.700 | 0.300 | 0.000 | 0.000 | 0.1681 |
| g19_dagger | strict125 | 2.5708 | 2.5060 | 1.7144 | 0.400 | 0.400 | 0.600 | 0.000 | 0.000 | 0.1694 |

N8 default 相对 g19 的 pre-goal RMSE 比值为 `1.0891`，没有超过 `1.10` 护栏；
但 N8 的 success 和 success-only RMSE 仍弱于 g19，说明 nominal 传感器域存在可测量
的分布迁移，不能把新 N8 直接视为旧 BC 的等价替代。

## 5. 预注册门禁

| gate | actual | op | threshold | result |
|---|---:|---|---:|---|
| n12_default_pre_goal | 2.881010 | <= | 2.682545 | **FAIL** |
| n12_default_full_rmse | 2.807198 | <= | 2.903770 | PASS |
| n12_default_success_drop | 0.100000 | <= | 0.050000 | **FAIL** |
| n12_default_capture_drop | 0.100000 | <= | 0.000000 | **FAIL** |
| n12_default_plane_miss_increase | 0.100000 | <= | 0.050000 | **FAIL** |
| n12_strict_pre_goal | 2.866378 | <= | 3.084837 | PASS |
| n12_strict_success_floor | 0.350000 | >= | 0.100000 | PASS |
| n12_strict_safety_increase | 0.000000 | <= | 0.020000 | PASS |
| n8_default_vs_g19_pre_goal | 1.089118 | <= | 1.100000 | PASS |
| n8_default_path_zero_degradation | 4.521100 | >= | 1.050000 | PASS |
| n8_strict125_path_zero_degradation | 4.549357 | >= | 1.050000 | PASS |
| n12_default_path_zero_degradation | 7.226536 | >= | 1.050000 | PASS |
| n12_strict125_path_zero_degradation | 7.263424 | >= | 1.050000 | PASS |

## 6. path-zero 诊断

| arm | scenario | normal pre-goal RMSE | path-zero pre-goal RMSE | degradation ratio |
|---|---|---:|---:|---:|
| N8 | default | 2.8237 | 12.7664 | 4.5211× |
| N8 | strict125 | 2.8044 | 12.7582 | 4.5494× |
| N12 | default | 2.8810 | 20.8197 | 7.2265× |
| N12 | strict125 | 2.8664 | 20.8197 | 7.2634× |

path-zero 均显著退化，说明路径分支确实被策略使用，不是“路径输入完全无效”的实现错误。
N12 的 path-zero 退化更大，不能解释为 path encoder 没有接上。

离线 action audit 也支持策略确实在使用 path block：

| arm | normal normalized RMSE (roll/pitch/RPM) | path-zero normalized RMSE | predicted surface saturation (normal) | teacher surface saturation |
|---|---|---|---:|---:|
| N8 | `[0.1397, 0.1176, 0.0235]` | `[0.6000, 0.5832, 0.0678]` | 0.3398 | 0.4991 |
| N12 | `[0.1142, 0.0827, 0.0230]` | `[1.0866, 0.9393, 0.0644]` | 0.2679 | 0.4991 |

normal 下的 predicted reversal rate 两个 arm 均为 `0`；path-zero 下 N8 仅约
`3.82e-4 s^-1`，N12 为 `0`。训练/评估 summary 全部 finite，没有 NaN/Inf。

## 7. 失败原因与证据边界

1. **不是训练完全没收敛**：N12 的 validation loss（`0.00949`）和 normalized
   验证 RMSE 均优于 N8，但闭环 default success 反而低 `0.10`。这说明离线动作拟合
   更好没有转化为闭环成功率。
2. **失败集中在少数 seed**：default 的 success 差异来自 4 个 seed 翻转，净少 2 个
   成功 episode（seed 20/31/33 由成功变 plane miss，seed 30 反向成功）。在 20 个
   dev episode 上，这既可来自真实的闭环分布差异，也可能包含不可忽略的抽样噪声；
   但预注册门禁按点估计判定，不能事后放宽。
3. **strict125 不是均匀崩溃**：N12 strict125 success 为 `0.35`，反而高于 N8 的
   `0.25`，pre-goal RMSE 也在 `1.10×` 容忍范围内。当前阻断项是 default，而不是
   12D 终点上下文在所有场景都失效。
4. **终点上下文假设没有获得闭环支持**：N12 的离线误差更低，但 default 的
   capture/plane-miss 更差；现有证据只能说明“12D 终点块在该数据预算和教师分布下
   没有带来稳定的闭环收益”，不能据此断言终点特征本身无效，也不能用事后解释替代
   门禁结论。
5. **旧 g19 的“强”主要来自数据覆盖，而不是网络或 loss**：旧
   `g19_dagger` checkpoint 的合并数据为 `421,888` samples / `330` episodes，其中
   原始教师数据 `262,144`、一般 DAgger `131,072`、定向 DAgger `28,672`，DAgger
   合计 `159,744`，占 `37.9%`；新 N8/N12 是 `393,216` samples / `317` episodes
   的纯教师 BC，DAgger 为 `0`。旧数据的传感器统计也更严苛（nav position RMSE
   `0.716 m` vs 新数据 `0.390 m`，current estimate age `1.37 s` vs `0.41 s`，
   DVL bottom outage `2.70%` vs `0.85%`）。旧 BC 的 validation loss 反而更高
   （`0.1254` vs N8 `0.0148`），这说明 g19 的优势不是“拟合得更好”，而是训练
   分布包含更多失败恢复状态、且对更宽传感器误差更鲁棒。

## 8. 决策与后续

- Goal22 判定：**FAIL**；
- 不进入 P3 DAgger、P5 残差 PPO、理想 actor 完整对照臂；
- 不扩大 seed、不扫描超参数、不重新解释门禁、不碰 held-out；
- 若未来另开 Goal，可把 N8 作为下一阶段候选底座，但必须先明确新的假设和预注册
  门禁；本次不使用 N12；
- 在把 N8 送入 P3/P5 之前，先修正新 nominal 配置对 R2B current 三个教师开关的
  继承问题，并重采/重训，或至少做严格配置对照；否则下一步仍会混入教师分布差异；
- 所有正式数字以 `summary.json` 和 `comparison.json` 为准，不以 smoke 结果替代。

## 9. 产物与验证

根目录（同时通过仓库内
`artifacts/lowlevel/pc_v0d_nominal_bc_rebase_20261107` 符号链接访问）：

```text
/dev/shm/pc_v0d_nominal_bc_rebase_20261107/
  demos_N8_default/
  demos_N8_strict125/
  demos_N8_merged/
  demos_N12_default/
  demos_N12_strict125/
  demos_N12_merged/
  bc_N8/
  bc_N12/
  eval_goal22/
  audit_N8_normal.json
  audit_N8_path_zero.json
  audit_N12_normal.json
  audit_N12_path_zero.json
  comparison.json
  comparison.md
  logs/
```

验证记录：

- `uv run pytest -q tests/test_lowlevel_pc_bc.py tests/test_lowlevel_pc_env.py tests/test_lowlevel_pc_eval_contract.py`：`56 passed`；
- `uv run ruff check ...`：`All checks passed!`；
- 10 份评估 `summary.json` 全部 finite；
- 未启动 P3、P5、理想 actor 训练或 held-out 评估；
- 未 commit/push。

