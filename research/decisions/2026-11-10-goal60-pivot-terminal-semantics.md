# Goal60 decision: pivot to terminal capture alignment

日期：2026-11-10<br>
决策：`pivot`

## 触发

Goal60 只读 trace 审计完成：

- H-C 强支持：default/strict125 的 success/capture discordance 分别为
  `0.381/0.269`，success_no_capture 的 true terminal distance 明显更大；
- H-B 弱/部分支持：path 正项被 command 负项淹没，rho CI 跨零；
- H-A 被否：saturation 是全局属性，failed-success 中位数差只有 `1e-5`；
- H-D/H-E 不支持：residual 非零且训练曲线平台化。

## 决策

停止把动作参数化/softsign 继续调参作为主线，也不启动 SAC/REDQ/DroQ。
下一 Goal 先预注册 terminal capture 对齐：

1. 在现有 trace 上比较 `p_nav` 与 `eta` 的 terminal distance/signed offset；
2. 确认 success 是否在 true capture 之前触发；
3. 只在需要时测试 termination/calibration 或 terminal reward composition 的
   1-seed falsification。

## 禁止

- 不重开 Goal27 的 PPO/reward/gamma/seed 搜索；
- 不打开 held-out `100–149`；
- 不把只读诊断写成性能 PASS；
- 不用换 seed、加预算或放宽阈值救结果。

## 证据

- `doc/goals/goal60_result_20261110.md`
- `research/blockers/2026-11-10-goal60-terminal-capture-discordance.md`
- `artifacts/lowlevel/goal60_trace_audit_20261110/goal60_trace_audit.json`
