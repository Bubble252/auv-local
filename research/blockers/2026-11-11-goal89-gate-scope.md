# Blocker：Goal89 训练 gate 作用于 BC+PPO 总和，评估只 gate PPO

日期：2026-11-11<br>
Goal：Goal89<br>
状态：`CONFIRMED / OLD_ARMS_INVALID_GATE_SCOPE / CORRECTED_GOAL90_REQUIRED`

## 1. 现象

Goal89 `G20_5`/`G10_3` 的 1M 训练臂在 dev40 上退化：

| arm | default | strict125 |
|---|---:|---:|
| G20_5（3 seeds） | 23/40 | 14/40 |
| G10_3（3 seeds） | 22/40 | 13/40 |
| Goal88 S1.25 无 gate | 31/40 | 19–20/40 |

同时：

- 训练日志 `actuator_residual_gate_mean` 约 `0.32`（G20_5）和 `0.14`（G10_3）；
- 物理残差从无 gate 的约 `59` 降到 `17–33`；
- `policy` 与 `frozen-base` 在 12 组配对中 success 完全相同。

## 2. 根因

Goal86 P-A checkpoint 的语义是执行器残差：

```text
r_BC = normalize(D3_actuator - ALOS_actuator)
```

Goal89 旧训练使用 `base_policy.output_mode=composed` 和
`pc.actuator_terminal_gate_enabled=true`，因此闭环动作是：

```text
ALOS + scale * gate * (r_BC + r_PPO)
```

gate 在远处为 0，于是 `r_BC` 也被关掉，动作退化为纯 ALOS。评估器的
`gated-residual` 则是：

```text
ALOS + scale * (r_BC + gate * r_PPO)
```

训练/评估 gate scope 不一致，导致训练臂的 success 退化不能归因于 PPO 学习。

## 3. 证据

- `doc/goals/goal89_network_parity_and_gate_scope_audit_20261111.md`
- `artifacts/lowlevel/goal86/pa_residual_bc_path12_cmd3_current_20261110/pc_bc_actor.pt`
  - `residual_semantics.target = normalize(D3_actuator - ALOS_actuator)`
  - `residual_semantics.residual_scale = 1.0`
- `artifacts/lowlevel/goal89_gate_20261110/*_eval_*_dev40*/summary.json`
- `artifacts/lowlevel/goal89_gate_20261110/*_eval_*_dev40_frozenbase/summary.json`
- `artifacts/lowlevel/goal88_scale_20261110/scale1.25_seed*_eval_*_step7_*/summary.json`

网络结构对照：Goal89 PPO 的 22 个 `frozen_bc_*` state entry 与 Goal86 BC
checkpoint `max abs diff = 0.0`；不存在宽度、维度、reserved slot 或
path/current 接线错误。

## 4. 已排除项

- 不是 `d_z_total`/path/current/command 维度错位；
- 不是 frozen BC 权重加载错误；
- 不是 NoisyNet（这些 arm 没有 NoisyNet；评估为 deterministic mean）；
- 不是 reward/terminal/success/goal radius；
- 不是 `actuator` 接口本身错误。

## 5. 可证伪 hypothesis

### H89G-A：只 gate PPO 增量可以恢复 BC 的 success，同时保留 terminal 修正

- 支持：`ALOS + scale*(r_BC + gate*r_PPO)` 在 1M/dev40 上 default
  `>=31/40`、strict125 `>=19/40`，path 不差于 BC 超过 `0.01 m`。
- 否定：success 仍掉到 `<=25/40`，说明 gate 不是唯一原因。
- 最便宜证伪：Goal90 `PG20_5`/`PG10_3` 各 1 seed、1M。

### H89G-B：PPO 增量本身有正贡献

- 支持：在相同 corrected gate 下，`policy` 比 `frozen-base` 的 strict125
  至少 `+1` 且 path 不退化。
- 否定：`policy == frozen-base` 或 policy 更差，说明 PPO 分支没有有效增量。
- 最便宜证伪：同 checkpoint 的 `policy` vs `frozen-base` 配对。

### H89G-C：PPO 的瓶颈是探索噪声，不是网络结构

- 支持：训练噪声 std 约 `0.10`，确定性残差约 `0.019`，噪声/信号约 `5.3×`；
  降低 noise std 后确定性残差增大且 strict125 改善。
- 否定：降低噪声后残差仍被压小，或 success 不变。
- 最便宜证伪：Goal90 后追加 noise std `0.03` 的 1-seed screen。

## 6. 下一步

1. 旧 Goal89 `G20_5`/`G10_3` 标 `INVALID_GATE_SCOPE`，不据此判 FAIL。
2. `goal89_v2` 全部标 `INVALID_WRONG_BC_CHECKPOINT_CONTRACT`。
3. 开 Goal90：`output_mode=residual`、`action_interface=actuator`、
   `residual_terminal_gate_*`，只 gate PPO 增量。
4. 跑 corrected 1M/dev40/default+strict125，再决定是否进入探索 screen。
5. 不换 PPO/SAC 主算法族；不打开 held-out。
