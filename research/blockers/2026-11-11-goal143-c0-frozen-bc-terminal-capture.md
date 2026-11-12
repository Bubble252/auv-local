# Goal143 C0 blocker：frozen-BC 残差拖累 ALOS

**日期**：2026-11-11<br>
**Goal**：Goal143<br>
**状态**：`C0_GATE_FAILED / FROZEN_BC_RESIDUAL_BOTTLENECK / TERMINAL_CAPTURE_NEAR_RADIUS`

## 现象

- `C-full-alos` 与 `C-no-current-alos` 的 PPO 1M pilot 在 C0 始终为
  default/strict `0.85`、plane miss `0.15`、timeout/safety `0`。
- 300k 连续两次无提升，判 `CURRICULUM_STALLED`；最终 `STAGE_BUDGET_EXHAUSTED`。
- 6/40 失败全是 `goal_plane_miss`，最终距离 `1.38–1.90 m`，goal radius `1.25 m`。

## 反事实

同 300k checkpoint、同 C0、同 dev20–59：

- `policy`（frozen-BC + PPO 残差）：`34/40 = 0.85`。
- `zero-residual`（frozen-BC + 零 PPO 残差）：`34/40 = 0.85`。
- `alos`（纯 ALOS，无 frozen-BC）：`37/40 = 0.925`。

`policy` 与 `zero-residual` 完全相同，说明 PPO 增量未改变行为；frozen-BC 残差
基座相对纯 ALOS 损失约 `7.5 pp`。这不是 direct command、current alpha、
curriculum sampler、reward、terminal 或 seed 噪声能解释的差异。

## 已排除

| 项 | 证据 |
|---|---|
| direct command | manifest `curriculum_direct_commands_enabled=false` |
| current encoder | C-full-alos 与 C-no-current-alos 同为 `0.85`；后者 alpha=0 |
| p_nav/eta | 失败 episode 的 `success_pnav_tight=false` 且 `success_eta_tight=false` |
| timeout/safety | 两者均为 0 |
| curriculum 门限 | 门限未改；pure ALOS 已经 `0.925` 过关 |
| Gate seed 选择 | 固定 dev20–59，未扫 seed、未打开 held-out |

## 可证伪假设

**H143-A：frozen-BC 残差是主因。** 去掉 frozen-BC、从纯 ALOS 学习残差，应至少
恢复到 `>=0.90`，且 zero-residual parity 不再与 policy 完全相同。最便宜证伪：
Goal144 的 `ALOS+PPO command residual` 在 C0 100k/200k dev40 Gate。若仍 `<0.90`
且 residual 非常小，才转向优化/探索归因。

**H143-B：末段 3D 捕获受动作权限/饱和限制。** 若失败集中在跨面水平+垂直同时
贴边，且残差限幅率高于成功集，则调整 terminal capture/限幅或动作尺度应能改善。
最便宜证伪：同 checkpoint 统计成功/失败 `actuator_residual_magnitude_clip_rate`、
跨面水平/垂直分解，并对 ALOS+RL 臂做限幅消融。当前初步支持：失败集约 `0.60`，
成功集约 `0.53`；seed 21 跨面水平 `1.34 m`、垂直 `1.33 m`。

**H143-C：reward/terminal 不是主因。** 若 pure ALOS 在同 reward/terminal 下达到
`0.925`，则无需改 reward 或 terminal 来解释 C0 失败。当前支持；若 H143-A 后
ALOS+RL 仍失败且失败模式改变，再单独审计 terminal reward credit。

## 决策

`pivot` 到 Goal144：保留 frozen-BC+PPO 组，新增纯 ALOS 基座上的 PPO/SAC 残差组；
同 C0/C1、同 dev40、同 `0.90` Gate。不换 PPO/SAC 主算法，不改 reward/terminal/
success/radius，不扫 seed，不打开 held-out。Goal140 继续 `P2`。

