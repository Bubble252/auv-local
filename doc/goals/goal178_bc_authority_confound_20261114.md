# Goal178：BC authority 与 PPO head authority 的归因拆分

**日期**：2026-11-14<br>
**状态**：`RESULT_COMPLETE / BC_AUTHORITY_CONFOUND / HEAD_GAIN_NOT_CAUSAL`<br>
**上游**：Goal177 `GATE_A_FAIL / RESIDUAL_TOO_SMALL_FOR_PPO`

## Research Contract

```yaml
autoresearch:
  required: false
  triggers:
    - gate_fail_twice
    - no_metric_movement_after_3_fixes
    - heldout_collapse
    - seed_noise_ambiguous
  ledger: research/LEDGER.jsonl
  blockers_dir: research/blockers/
  reading_dir: research/paper_reading/
  max_outer_loops: 3
  check_command: python ../scripts/check_research_ledger.py --goal Goal178 --goal-doc doc/goals/goal178_bc_authority_confound_20261114.md --strict
```

- 这是冻结的 eval-only 归因诊断；在记录结果前不改动作定义、seed、场景或指标。

## 1. 背景与直接证据

Goal177 的 `A_S150_RUDDER_ONLY` 相对 `zero_bc50` 在 `current10/default`
出现约 `-0.083 m` 的 3D 改善，但该 arm 通过 actuator wrapper 同时放大了
`r_BC` 和 `r_PPO`。现有 trace 已证实：

```text
requested_action = base_action + residual_action
actuator_residual_applied ~= requested_action * authority * full_scale * gain
```

`A_S150_RUDDER_ONLY` 中 PPO 增量只占 rudder composed delta 的约 `0.31%`。
因此不能把该改善归因给 PPO head。

## 2. 假设

### H178-A：A_S150 的改善主要来自 BC authority

- 成立预期：固定同一 checkpoint、场景和 seeds，只用 `frozen-base` 把 BC
  authority 从 `0.75` 提到 `1.50`，3D/cross 改善与 `A_S150_RUDDER_ONLY`
  接近；PPO 增量为零仍能复现大部分行为变化。
- 失败预期：BC-only 1.5 与 `zero_bc50` 基本相同，而 `A_S150` 仍显著不同；
  此时才需要重新检查 PPO 与 BC 的耦合。
- 最便宜证伪：一个 eval-only cell，8 个 dev seeds。

### H178-B：authority scale 是混合干预，不是 head-only 剂量

- 成立预期：`A_S075` 与 `zero_bc50` 几乎重合，而 `A_S150` 的变化主要随 BC
  幅度而不是随 PPO residual 幅度变化。
- 失败预期：composed authority 与 head-only authority 在 trace 上可分离，
  且 BC-only 1.5 不产生同向路线变化。

## 3. 数据与隔离

- checkpoint：与 Goal177 相同的 `P_RUD_075_COMP` final checkpoint；
- 场景：`p-current10__r-legacy_sine__l35__v1p2__g-default`；
- dev seeds：`20–27`；
- held-out `100–149` 全程关闭；
- 不训练、不改 reward / termination / success / radius、不扩 seed。

## 4. 执行

只跑一个 eval-only cell：

```text
arm: BC_ONLY_S150
action_mode: frozen-base
actuator_residual_scale: 1.50
actuator_residual_channel_gains: [1.0, 1e-6, 1e-6]
```

与已有 `zero_bc50`（authority `0.75`、零 PPO）和
`A_S150_RUDDER_ONLY`（authority `1.50`、含 PPO）做三元配对。

## 5. Gate

| check | 判读 |
|---|---|
| BC-only 1.5 3D/cross 接近 A_S150 | `BC_AUTHORITY_CONFOUND` |
| BC-only 1.5 接近 zero_bc50 且 A_S150 不同 | `HEAD_EFFECT_REQUIRES_RECHECK` |
| 任一 cell 缺 seeds、timeout/safety/NaN 异常 | `INVALID` |

## 6. Git

不自动 commit/push；只提交本 goal 相关文件，禁止 `git add -A`。

## 7. 结果

设定：`current10 × default`，dev seeds `20–27`，固定同一 `P_RUD_075_COMP`
checkpoint；`BC_ONLY_S150` 使用 `action_mode=frozen-base`，PPO 残差严格为零，
只把 actuator authority 从 `0.75` 提到 `1.50`。

| arm | 3D RMSE | cross p90 | vertical p90 | success | Δ3D vs zero |
|---|---:|---:|---:|---:|---:|
| `zero_bc50` | `2.10591 m` | `3.40061 m` | `0.63767 m` | `8/8` | — |
| `BC_ONLY_S150` | `2.02103 m` | `3.21685 m` | `0.63299 m` | `8/8` | `-0.08488 m` |
| `A_S075_RUDDER_ONLY` | `2.10643 m` | `3.40144 m` | `0.63768 m` | `8/8` | `+0.00053 m` |
| `A_S150_RUDDER_ONLY` | `2.02286 m` | `3.21998 m` | `0.63300 m` | `8/8` | `-0.08305 m` |

`BC_ONLY_S150` 与 `A_S150` 的差异只有 `+0.00183 m` 3D（A_S150 略差），
`+0.00313 m` cross p90；两者的逐 seed 路线几乎重合。PPO 在 A_S150 的
rudder composed delta 中只占约 `0.31%`，相对 BC-only 的 actual rudder
absmean 差约 `0.00051 rad`。按 seed 计算，BC-only 1.5 与 A_S150 都只有
`4/8` seed 改善 3D/cross；均值改善来自 seed 间正负抵消，不是稳定收益。

判读：

```text
A_S150 的行为改善归因于 BC authority 0.75 -> 1.50，
不是 PPO residual head 的有效闭环收益。
```

因此 Goal177 的 authority sweep 是混合干预：它同时放大 `r_BC` 与 `r_PPO`。
这不是新的接线 bug；当前 actuator 契约本来就是
`ALOS + authority * (r_BC + r_PPO)`。问题是实验没有把两个 authority 分开，
所以不能把 composed 改善归因给 head。

证据：

- `/data/yhx/underwater-robotics/runs/auv-local/goal177/20261114-goal177-stage-a/stage_a/diagnostics/BC_ONLY_S150_current10_default/cell.json`
- 同目录 `trace.npz` 与 `summary.json`
