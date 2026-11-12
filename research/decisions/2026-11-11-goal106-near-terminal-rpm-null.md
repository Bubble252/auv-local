# Goal106 decision：近终端 rpm 门控无效，pivot 到 reward/terminal credit 审计

**日期**：2026-11-11<br>
**状态**：`NO_SIGNAL / PIVOT_REWARD_TERMINAL / NO_TRAINING`

## 1. 触发

Goal104/105 已证明全局 rpm 0.50 可以把 strict 从 20 推到 23，但 default 掉 1。
Goal106 预注册检验“只在近终端降低 rpm”能否保住 default。

## 2. 结果

- C0 与 Goal105 逐元素完全一致；
- 4 个门控 profile 全部 strict 20/40；
- default 只有 3 个 profile 到 34/40，`G050_n2_f5` 仍为 33；
- strict gained 与 Goal104 `25/51/56/59` 重叠为 0；
- gate 按预注册降级为 `NO_SIGNAL`；
- 门控实际触发，near rpm delta 平均 206–282 rpm；
- 全局 U050 仍能让 4 个关键 strict seed 全部进入 1.25 m，说明收益不是近终端局部量。

## 3. 决策

```text
decision = pivot
```

立即停止 rpm 常数/距离门控扫值。新开 Goal107，只做离线 reward/terminal credit
审计：

- 用现有 C0/Goal106 traces 与一次带 reward breakdown 的只读复跑；
- 对齐 strict 失败 seeds 与 U050 成功 seeds；
- 量化 terminal reward、speed/path 项、advantage proxy 和最后一公里控制；
- 不训练、不改 success/goal radius、不打开 held-out、不换算法族。

只有审计证明 terminal reward/advantage 被压制，才允许写训练型 Goal108。

## 4. 证据

- `doc/goals/goal106_result_20261111.md`
- `research/blockers/2026-11-11-goal106-rpm-parameterization-null.md`
- `artifacts/lowlevel/goal106_near_terminal_rpm_20261111/phase0_summary.json`
