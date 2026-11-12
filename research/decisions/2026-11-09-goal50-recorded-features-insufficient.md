# Goal50 decision：recorded features 不足，pivot 到 Goal51 instrumentation replay

日期：2026-11-09<br>
Goal：Goal50<br>
诊断裁决：`RECORDED_FEATURES_INSUFFICIENT`<br>
Ledger decision：`pivot`

Goal50 在 Goal47 四个 formal trace block 上完成只读分析。13 个冻结特征全部可评估，
coverage Gate 通过；但 terminal timing、route-floor timing 和 9 个 late-window
control/activity 特征都没有在两个 strict125 block 同时达到 `rho <=0.30`、
`drop >=0.20`。最强的 terminal-entry signal 只在 candidate 达到 drop `0.2432`，
但 rho 仍为 `0.6426`，baseline drop 只有 `0.0057`。

因此停止在现有 Goal47 traces 上增加 mean/std/timing 特征的路线。Goal49 的
`command_residual_applied` 建议也已被实际 trace 否定：四个 formal block 全为零。

下一步只允许预注册 Goal51 instrumentation replay，补齐当前 trace 缺失的：

- normalized USBL innovation、Kalman gain / posterior covariance；
- DVL scale/misalignment 或 bottom-track residual；
- command-to-actuator lag / rate-clipping。

Goal51 必须先冻结 replay equivalence Gate：同一 checkpoint、configs、seeds、
sensor profile、termination 和评估协议；旧字段 bitwise invariant；不训练、不扫
reward/gamma/seed、不恢复 command-residual PPO/supervised residual、不打开
held-out `100-149`。只有在 replay Gate 通过后，才允许做新的只读机制分析。

证据：

- `doc/goals/goal50_guidance_timing_audit_20261109.md`
- `doc/goals/goal50_result_20261109.md`
- `research/blockers/2026-11-09-goal50-recorded-features-insufficient.md`
- `research/paper_reading/2026-11-09-goal50-estimator-state-instrumentation.md`
- `artifacts/lowlevel/goal50_guidance_timing_audit_20261109/`
