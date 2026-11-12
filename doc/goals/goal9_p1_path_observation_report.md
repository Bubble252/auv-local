# Goal9 S15-P1 路径观测筛查报告

更新日期：2026-11-05

状态：**正式 screen 已完成；路径分支判为 `no_path_attribution`。**

对应协议：

- `doc/goals/goal9_odarrl_control_reproduction.md` 的 `14.4.18` P1 小节；
- `doc/goals/goal9_pc_server_validation_plan.md` 的 `S15.0` P1 小节。

本报告只记录 P1 的正式服务器结果，不把 Torch 预筛写成最终性能结论。P1 的路径分支已经
按预注册门禁停止；后续 P2 回到 P0 的 `A0_stride1` 基线，且不再继续增加路径变量。

## 1. 执行口径

- 主环境：`odarrl_control_pc_c3_random_realistic_v1_rewardscale001.json`；
- 训练 seed：`91430`；
- 评估 seeds：`1000–1019`，共 20 个固定 seed；
- 训练预算：`32 iterations × 128 envs × 256 steps = 1,048,576` 环境步；
- 评估 tag：`default`、`strict125`；
- 路径消融：`none`、`path-zero`、`path-geom-zero`、`path-context-zero`；
- 基线：P0 `A0_stride1`；
- 路径 schema：`path_geom` + `path_context`，维度 8，全部为 active；
- schema hash：`be882d27f3aa48fe7495c4bfaf27cd5062be588a9e350f495c8f3887c7dc4c64`。

训练预算、训练 seed、评估 seed 和门禁阈值均未因结果调整。

## 2. 权威产物

服务器根目录：

```text
/data/yhx/underwater-robotics/src/plan_p1/settled_局部规划器_反应式/artifacts/lowlevel/pc_abc_prework_20261105/P1_path_obs/
```

关键文件：

```text
run_manifest.json
report/P1_full/summary.json
report/P1_full/summary.md
report/stage_decision.json
P1_full_screen/training/P1_full_seed91430/final.pt
P1_full_screen/training_audit.json
```

- run manifest 状态：`completed_with_scientific_fail`；
- P1_full training audit：`passed=true`；
- config hash：`15d507845e4e111a332da33e45b6daaa542bd79928f23809965501a32d26af3a`；
- final checkpoint hash：`4f5cb105fef7218c2033bbf2ce5a3d51ebac006193891bcea52c7f7f4e080e7e`；
- BC checkpoint hash：`f56c359648a99e3fc8fde51f2e529af5c9916371450ff0d7a942c0f620d33936`；
- 正式 screen wall-clock：约 `344 s`。

## 3. P1 smoke 契约

P1 smoke 位于：

```text
artifacts/lowlevel/pc_abc_prework_20261105/P1_path_obs/smoke/
```

结果：

- `P1_full` 与 `P1_curriculum` 均完成 32 iterations；
- 两个训练 run 的 training audit 均为 `passed=true`；
- 二者最终路径 schema 相同；
- 课程记录依次包含 `state_base`、`state_base + path_geom`、
  `state_base + path_geom + path_context`；
- 四类路径消融齐全；
- smoke stage decision 为 `schema_equal=true`。

smoke 只证明训练、schema、消融和汇总链路可用，不作为科学结论。

## 4. P1 screen 结果

相对 P0 `A0_stride1` 的逐 seed 配对结果：

| tag | success delta | strict125 delta | 3D RMSE 改善 | timeout delta | safety delta | saturation delta |
|---|---:|---:|---:|---:|---:|---:|
| `default` | `+0.0500` | `+0.0500` | `0.4308 m` | `-0.0500` | `0.0000` | `+0.0053` |
| `strict125` | `0.0000` | `0.0000` | `0.0162 m` | `0.0000` | `0.0000` | `+0.0018` |

因此 P1 达到了当时的 primary signal 和 guardrail 要求：default 上 success 提升
`0.05`，3D RMSE 改善超过 `0.10 m`，safety 不退化，timeout 不增加，material
saturation 增幅低于 `0.02`。注意：2026-11-05 后续决策已把 saturation 统一降为风险
报告项，不再作为通过/否决门禁；本节保留历史门禁口径，不重写旧结果。

## 5. 路径归因消融

| tag | ablation | success 下降 | 3D RMSE 恶化 | timeout 变化 | action L2 mean | 判定 |
|---|---|---:|---:|---:|---:|---|
| `default` | `path-zero` | `-0.0000` | `0.0000 m` | `0.0000` | `4.70e-05` | 未检测到 |
| `default` | `path-geom-zero` | `-0.0000` | `-0.0000 m` | `0.0000` | 约 `0` | 未检测到 |
| `default` | `path-context-zero` | `-0.0000` | `0.0001 m` | `0.0000` | 约 `0` | 未检测到 |
| `strict125` | `path-zero` | `-0.0000` | `0.0001 m` | `0.0000` | 约 `0` | 未检测到 |
| `strict125` | `path-geom-zero` | `-0.0000` | `0.0000 m` | `0.0000` | 约 `0` | 未检测到 |
| `strict125` | `path-context-zero` | `-0.0000` | `0.0000 m` | `0.0000` | 约 `0` | 未检测到 |

`path-zero` 的 success 下降为 `-0.0`，3D RMSE 仅恶化 `3.13e-05 m`，动作 L2 平均变化
约 `4.70e-05`。这说明 P1 的指标改善不能归因于路径输入；当前 P1 路径编码分支没有被
actor 有效利用。

## 6. 最终判定

`stage_decision.json`：

```text
status = no_path_attribution
selected_arm = null
next_action = stop_path_branch_and_write_negative_result
```

P1 结论：

- 正信号：存在，default success 和 3D RMSE 达标；
- guardrails：通过；
- path attribution：不通过；
- 路径分支：fail，停止继续增加路径特征、改路径 scale 或换评估 seed；
- `P1_curriculum` screen：未运行，因为 `P1_full` 未通过路径归因门禁；这不是 curriculum
  训练失败，也不能记为 curriculum fail；
- 后续 P2：基线回到 P0 `A0_stride1`，不再继续加路径变量；比较 raw-history DADE 与
  base-state 编码历史 DADE；
- P3：默认不适用。只有新的 P2 证据明确显示路径信息被使用但仍有路径误差时，才允许
  按协议启动奖励分支。

## 7. 基础设施修复

本轮只修复了执行链路，没有改变科学预算：

1. smoke 与 screen 的 `run_manifest.json`、`report/` 分开，screen 不再覆盖 smoke；
2. smoke 自动使用 P0 smoke 的 seed `0–7` 基线，screen 使用 P0 screen 的 seed
   `1000–1019` 基线，避免 seed mismatch；
3. summarizer 忽略 P0 baseline 中的 `history-zero`、`history-reverse` 等历史消融目录，
   避免被误识别为多个 `(arm, tag, none)` 重复项；
4. P1 自动执行提示词中删除了 smoke 阶段显式传入 P0 screen baseline 的错误参数。

## 8. 下一步

下一步是 S15-P2，不是重跑 P1：

1. P2a 直接复用 P0 `A0_stride1` 的 `none` 与 `history-zero` 结果；
2. 实现 `dade_history_mode=raw|encoded`，P2b 只编码历史 token；
3. side channel 保持原始估计海流，current token 不参与 DADE 融合；
4. 在编码前执行 `history-zero`，并测量 p50/p95 决策时延；
5. P2b 无稳定增量时保留 raw-history DADE，停止 DADE-z 重构；
6. P2 结束后正式记录 P3 的“不适用/前置条件不满足”，然后进入正式 A/B/C。
