# Goal161 Phase A 结果：cap 0.25 只能消害，不能在 current10 超过 zero

**日期**：2026-11-12<br>
**状态**：`PHASE_A_COMPLETE / DEPLOY_SCALE_CAP_NO_GAIN / PHASE_B_PENDING`<br>
**协议**：Goal160 full-authority B/C checkpoints，仅改部署 residual scale；不训练

## 1. 执行摘要

Phase A 共完成 `96/96` 个 summary：2 arms × 2 scales × 2 radii × 4 levels × 3 seeds。
`0 FAILED`，timeout/safety 均为 0，residual 活跃，风险项通过。

结论：`scale=0.25` 能把 Goal160 的 command residual 危害收回到 zero 附近，
但**没有产生超越 zero 的收益**，也没有通过 Goal161 的 +4 default / +2 strict 主门禁。

| radius | best arm | scale | current10 policy | zero | Δ vs zero | Δ vs A | 结论 |
|---|---|---:|---:|---:|---:|---:|---|
| default 2.0 m | B command | 0.25 | 90/120 | 90/120 | 0 | -1 | FAIL |
| strict 1.25 m | B command | 0.25 | 53/120 | 57/120 | -4 | 0 | FAIL |

## 2. 与 Goal160 的关系

Goal160 full authority 的 current10 default 为 B `83/120`、C `84/120`；
deploy `scale=0.25` 后回到 B `90/120`、C `89/120`。这支持“authority 过大是
Goal160 的危害来源”，但不支持“只把部署权限缩小就能获得正收益”。

## 3. 口径修正

先前 `diag_scale_20261112` 被误标为 current10：其 summary 实际记录
`curriculum_current_speed_range_override_mps=null`，环境默认洋流是
`0.25–0.45 m/s`。因此那里的 `95/96` 是默认洋流诊断，不是 current10。
本 Phase A 全部显式传入 `0.30–0.55 m/s + shear 1.0`，以本文件为准。

## 4. 其他层 guard

- default current07：B `99/120` vs zero `102/120`（-3）；C `97/120`（-5）。
- default current15：B `72/120` vs zero `66/120`（+6）；C `71/120`（+5）。
- nominal 基本持平或略低；strict 的 current10 仍为负。
- 因此不能把 current15 的正信号写成总体收益，其他层 guard 不通过。

## 5. 裁决与下一步

Phase A 不支持 `H161-A` 的“部署 cap 直接产生收益”。但 Phase B 仍按预注册执行，
因为 Phase B 会**在 cap 0.25 下重新训练方向**，而不是只缩小 full-authority
checkpoint 的输出。Phase B 若仍不能超过 zero，则判
`CAP_TRAINING_FAIL / AUTHORITY_DIRECTION_NOT_USEFUL`，再按 Research Contract
进入 blocker、channel restriction 和 terminal reward/credit 归因。

不换算法，不扩 seed，不打开 held-out，不改 reward/terminal/success/radius。

## 6. 产物

- Phase A summaries：`artifacts/lowlevel/goal161_command_authority_20261112/phaseA/`
- 汇总：`.../phaseA_analysis/phaseA_summary.{json,md}`
- 分析器：`scripts/analyze_pc_goal161_phaseA.py`
