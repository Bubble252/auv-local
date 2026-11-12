# Goal49 decision：feature coverage 不足，pivot 到 command-residual/guidance-timing audit

日期：2026-11-09<br>
Goal：Goal49<br>
诊断裁决：`INSUFFICIENT_FEATURE_VARIATION`<br>
Ledger decision：`pivot`

Goal49 的输入、reference 复现和 permutation 都有效，但冻结的 12 个 feedback 特征中
有 3 个在 strict125 late source window 中为常数：

```text
executor_residual_action_l2_mean      = 0.0  (C1 是 command-residual)
guidance_terminal_phase_fraction      = 1.0  (late 窗口全部 terminal phase)
guidance_route_floor_active_fraction  = 0.0  (late 窗口未激活)
```

因此只有 9/12 个特征可评估，低于预注册的 10 个 coverage Gate。可评估的 9 个特征也没
有一个在两个 strict125 block 同时满足 `rho <=0.30`、`drop >=0.20`；最大 candidate
drop 只有 `0.0911`，permutation global `p=1.0`。这不能写成“controller feedback
已被干净证伪”，只能写成“本轮 feature family 覆盖不足，且可评估子集没有支持信号”。

不再在 Goal49 中追加特征、换阈值或重跑不同窗口。下一步只允许预注册 Goal50
read-only command-residual/guidance-timing audit：

- 用 `command_residual_applied` / `command_corrected_physical` 替代 action-space
  `residual_action`；
- 用 episode-level terminal phase duration / time since entry、route-floor active
  duration 或 early/late activation 替代恒定 late fraction；
- 至少 10 个特征在两个 strict125 block 可评估；
- 保持 same-feature two-block Gate、non-discordant negative control 和 sigma
  permutation max-statistic；
- 不训练、不新增 replay、不改 controller/config/seed、不打开 held-out 100-149。

若 Goal50 仍不能解释，下一步才考虑新的 estimator/controller instrumentation replay；
不得直接恢复命令残差 PPO、supervised residual 或 reward/gamma/seed 扫描。

证据：

- `doc/goals/goal49_controller_feedback_discrimination_20261109.md`
- `doc/goals/goal49_result_20261109.md`
- `research/blockers/2026-11-09-goal49-feedback-feature-coverage.md`
- `research/paper_reading/2026-11-09-goal49-command-residual-feedback.md`
- `artifacts/lowlevel/goal49_controller_feedback_20261109/`
