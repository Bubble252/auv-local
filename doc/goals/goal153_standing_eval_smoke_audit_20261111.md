# Goal153 standing eval smoke 审计

**日期**：2026-11-11<br>
**状态**：`SMOKE_PASS / EVAL_PATH_REPAIRED / NOT_FORMAL`<br>
**范围**：只验证 standing-eval 数据通路与 trace 契约，不据此判定 reward arm 优劣。

## 1. 发现的问题

第一轮 smoke 的 `zero-residual` 只有 `0/4`，final distance 约 `18 m`，但同一 checkpoint
在 curriculum C1 eval 中并不差。根因不是策略，而是 C1 的 direct-command 训练脚手架：

- generated standing config 把 `curriculum.enabled` 设为 `false`；
- `BatchedPCLowLevelEnv` 在 C1 + `pc_control_mode=alos` 时重新把 `direct_commands` 设为 true；
- 因此 zero-residual 实际在跟踪随机 direct command，而不是沿参考路线飞。

这是评测配置错误，不能写成 `env_overfit` 或算法失败。

## 2. 修复

- `BatchedPCLowLevelEnv.direct_commands` 现在也尊重
  `curriculum_direct_commands_enabled=False`；
- standing runner 对 SAC / PPO / BC-residual 三类 evaluator 都强制传
  `--disable-curriculum-direct-command`；
- BC-residual evaluator 新增同名 CLI 参数；
- smoke trace 已验证包含 `policy_action_raw`、`policy_action_effective`、
  `actuator_saturated`、`material_saturated`。

## 3. 修复后 smoke

场景：`nominal / legacy_sine / 35 m / 1.2 m/s / default 2.0 m`，dev seeds `20–23`。

| tag | success | plane miss | timeout | path3d mean | vertical mean | final d | residual L2 ratio |
|---|---:|---:|---:|---:|---:|---:|---:|
| `zero` | `4/4` | 0 | 0 | `1.4520` | `0.8586` | `1.9884` | 0 |
| `R0_200k` | `4/4` | 0 | 0 | `1.4860` | `0.8332` | `1.9831` | `0.0744` |
| `R1_200k` | `4/4` | 0 | 0 | `1.4573` | `0.8433` | `1.9779` | `0.0530` |
| `R2_200k` | `3/4` | 1 | 0 | `1.5203` | `0.8988` | `2.1926` | `0.1368` |

补充观察：

- 所有 arm 的 policy-head pre-scale saturation 为 `0%`，residual post-scale saturation 为 `0%`；
- zero 的 base controller 自身有 actuator/material saturation，不能归因给 residual；
- 4 seeds 只能证明通路正确，不能判断 R1/R2 是否优于 R0。

## 4. 速度上界

REMUS100 最大速度为 `2.5 m/s`。standing suite 的速度轴已包含 `2.5`；训练侧
`CurriculumStageSpec.speed_range_mps` 现在硬校验 `<=2.5 m/s`，防止未来课程误采超包络速度。
冻结中的 Goal136/Goal143/Goal153 速度范围不被静默改写。

## 5. 下一步

- 等 `pilot_bc_1m_alos` 完成；
- 用 `100k / 200k / final` × `default/strict125` 跑 A_screen 40 seeds；
- A_screen 只作 screen，通过后再进 B/C tier 与 5M formal；
- 不因 smoke 的 R2 单次 plane miss 改算法或改 Gate。
