# Goal50 blocker：recorded timing/control features 全部失败，strict125 sigma residual 仍未解释

**日期**：2026-11-09<br>
**Goal**：Goal50<br>
**触发**：Goal50 的 13 个冻结特征在两个 strict125 block 全部可评估，但三个 family
均为 `FALSIFIED`；coverage 不再是解释，现有 recorded features 不足。<br>
**Ledger decision**：`pivot`

## 1. 现象

Goal50 复用 Goal47 formal traces，输入完整且 Gate 有效：

```text
strict125 baseline partial rho  = 0.8336161586462849  (error 0.0)
strict125 candidate partial rho = 0.8857851797221404  (error 0.0)
records = 160, blocks = 4
evaluable features = 13 / 13
coverage gate = PASS
valid permutation draws = 9998 / 10000
global permutation p = 1.0
overall = RECORDED_FEATURES_INSUFFICIENT
```

三个 family：

```text
H1 terminal_timing    = FALSIFIED
H2 route_floor_timing = FALSIFIED
H3 late_control       = FALSIFIED
```

最强的 candidate 单特征信号是 `guidance_terminal_phase_entry_time_s`：
drop `0.2432`，但 partial rho 仍为 `0.6426`，且 baseline drop 只有 `0.0057`，
不满足 same-feature two-block Gate。所有其他 candidate drop 都小于 `0.20`。

## 2. 已排除项

按 `数据 → 标签 → 评测 → baseline → 架构 → 文献` 排查：

- **数据/哈希**：Goal47 diagnostic、per-seed/per-block CSV、Goal45/46 CSV 和四个
  formal trace SHA256 全部匹配；160 records、四 block × 40 seeds 完整；
- **标签/样本**：strict125 baseline/candidate discordant 数为 `13/7`，
  non-discordant 为 `27/33`，与 Goal47/48/49 一致；
- **评测/reference**：四个 block 的 reference partial rho 精确复现，error `0.0`；
- **baseline/重复性**：Goal49 的 9 个可评估 late-window controls 在 Goal50 中
  完全复现；没有特征因为输入或代码路径变化而翻转；
- **coverage**：Goal49 的 3 个常数特征已替换；Goal50 的 2 个 terminal timing 和
  2 个 route-floor timing 在 strict125 candidate discordant seeds 中均有变化；
- **数值/置换**：13 个特征全部 finite；valid permutation draws `9998/10000`，
  高于预注册 `9000`；
- **command residual**：`command_residual_applied` 在四个 Goal47 formal trace 中
  全为零；不能再把它当作未测试的 seed-level 候选；
- **non-discordant 混淆**：13 个特征都没有在两个 non-discordant block 同时 pass，
  因此不是全局 timing/control artifact。

## 3. 可证伪 hypotheses

### H-FILTER-CONSISTENCY：未记录的 filter covariance/gain 状态解释 residual

- 机制：`nav_drift_sigma` 可能通过 Kalman gain、posterior covariance 或 normalized
  innovation 的 seed-level 差异影响 late dynamic projection；当前 trace 只记录 raw
  `usbl_innovation` / `usbl_innovation_applied`，没有 gain 或 covariance。
- 如果成立：在 Goal51 replay 中记录 normalized innovation、gain 或 posterior
  covariance 后，控制其中一个预注册特征应使 strict125 two-block partial rho
  `<=0.30` 且 drop `>=0.20`。
- 如果失败：这些 filter-internal 特征仍不能降低 association；不能把它写成“OU 因果已识别”。
- 最便宜证伪：先做 trace-schema audit；已确认当前 Goal47 trace 没有
  `covariance`、`gain`、`scale` 或 `misalign` 字段。因此必须预注册 Goal51
  instrumentation replay，不能继续在 Goal50 事后加变量。

### H-DVL-CALIBRATION：DVL scale/misalignment 或 bottom-track 误差解释 residual

- 机制：DVL scale factor、安装角或 bottom-track 误差可能让 `nav_drift_sigma` 与
  trajectory-dependent dead-reckoning 误差共线；Goal46 只检查了 DVL outage
  exposure，Goal47 只检查了 cumulative DVL hold drift。
- 如果成立：Goal51 记录的 scale/misalignment 估计或 bottom-track residual 在
  strict125 two-block 通过同一 partial-rho Gate。
- 如果失败：DVL calibration 通道不能解释 residual，下一步应优先 filter state 或
  sensor geometry，而不是继续扫 DVL 参数。
- 最便宜证伪：Goal51 replay 只增加 trace instrumentation，不改变 policy/action；
  先验证 replay equivalence，再做 read-only association。

### H-ACTUATOR-LAG：高频 command-to-actuator lag 解释 residual

- 机制：Goal49/50 只使用 late-window mean/std/saturation；命令到舵/推进器的
  rate clipping 与 lag 可能携带未压缩的时序信息。
- 如果成立：预注册的 lag/rate-clipping 特征在两个 strict125 block 通过 same-feature
  Gate，且不被 non-discordant controls 解释。
- 如果失败：执行器的高频时序也不是 residual 通道；应停止在 recorded actuator
  feedback 上继续加特征。
- 最便宜证伪：在 Goal47 traces 上先做只读 command/actuator cross-correlation 与
  rate-clipping 审计；该实验必须在新的预注册 Goal 中执行，不能追加到 Goal50。

## 4. 文献调研

已使用 `paper-search` skill 检索并以 arXiv/Crossref 记录核验。Semantic Scholar
返回 HTTP 429，未把失败来源当作证据。直接相关文献和设计含义见：

`research/paper_reading/2026-11-09-goal50-estimator-state-instrumentation.md`

文献支持三个方向：USBL fix 的 temporal sparsity/geometry、filter tuning/covariance、
以及 sensor misalignment / DVL velocity uncertainty 都可能让 navigation error 与
trajectory 共同演化。当前 trace 没有 covariance、gain、scale 或 misalignment 字段，
所以这些机制不能在 Goal50 内被进一步区分。

## 5. 最便宜 falsification experiment

Goal50 的 formal read-only analysis 本身就是对 H1/H2/H3 的最便宜 falsification：
没有新增 replay，直接在已有 160 条 records 上重算 partial Spearman、negative
control 和 permutation。结果是 full coverage + 全部 FALSIFIED。

对 H-FILTER-CONSISTENCY / H-DVL-CALIBRATION 的最便宜可行检查是 trace-schema
audit。已执行并记录：

```text
trace keys containing covariance: []
trace keys containing gain: []
trace keys containing scale: []
trace keys containing misalign: []
innovation keys: usbl_innovation, usbl_innovation_applied
outage keys: dvl_bottom_outage, nav_dvl_outage_delta
```

因此当前 traces 不能直接测试这些机制。继续用现有字段构造代理变量只会再次变成
post-hoc feature sweep；正确下一步是 Goal51 instrumentation replay，而不是继续
加 Goal50 特征。

## 6. 裁决与下一步

```text
decision = pivot
```

放弃“已记录 timing/control/actuator mean 特征可以解释 strict125 sigma residual”的
当前主线。下一步只允许：

1. 预注册 Goal51 instrumentation replay；
2. 保持同一 checkpoint、configs、seeds、sensor profile、termination 和 evaluation
   protocol；
3. 只增加 opt-in trace instrumentation：normalized innovation、gain/covariance、
   DVL scale/misalignment、command-to-actuator lag/rate-clipping；
4. 先过 replay equivalence Gate，再做 read-only association；
5. 不训练、不扫 reward/gamma/seed、不恢复 command-residual PPO/supervised residual、
   不打开 held-out `100-149`。

## 7. 证据

- `doc/goals/goal50_guidance_timing_audit_20261109.md`
- `doc/goals/goal50_result_20261109.md`
- `research/paper_reading/2026-11-09-goal50-estimator-state-instrumentation.md`
- `artifacts/lowlevel/goal50_guidance_timing_audit_20261109/guidance_timing_diagnostic.json`
- `artifacts/lowlevel/goal50_guidance_timing_audit_20261109/guidance_timing_feature_results.csv`
