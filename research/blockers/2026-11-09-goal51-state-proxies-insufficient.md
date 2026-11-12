# Goal51 blocker：可记录 state proxies 全部失败，strict125 sigma residual 仍未解释

**日期**：2026-11-09<br>
**Goal**：Goal51<br>
**触发**：Goal51 的 13/14 个冻结 estimator/DVL/actuator 代理可评估，但三个 family
都没有特征在两个 strict125 block 通过 same-feature Gate；输入、reference 和
permutation 均有效。<br>
**Ledger decision**：`stop`

## 1. 现象

Goal51 只读分析 Goal47 formal traces，没有新增 replay、训练、policy load 或
environment step：

```text
records = 160, blocks = 4
strict125 baseline partial rho  = 0.8336161586462849  (error 0.0)
strict125 candidate partial rho = 0.8857851797221404  (error 0.0)
strict125 discordant/non-discordant = 13/27, 7/33
evaluable features = 13 / 14
required coverage  = 10 / 14
coverage gate      = PASS
valid permutation draws = 9999 / 10000
global permutation p = 1.0
overall = UNRECORDED_STATE_PROXIES_INSUFFICIENT
```

三个 family 的诊断裁决均为 `FALSIFIED`，但必须保留一个 coverage 限定：
`actuator_best_lag_corr_late` 在四个 block 的全部 160 个 seeds 上都不可评估。
原因是每个 strict125 late window 的 requested fin norm 只有一个 unique value
（`std = 0.0`），而 actual fin norm 正常变化；Pearson lag correlation 在常数输入上
没有定义。这不是数据损坏，也不能把它写成“lag correlation 已证伪”。

最强的 candidate evaluable proxy 是 `dvl_bottom_scale_proxy_late`：

```text
baseline:  rho = 0.8378, drop = -0.0042
candidate: rho = 0.8232, drop =  0.0626
```

candidate drop 仍远低于 `0.20`，baseline drop 为负，且 rho 仍高于 `0.30`。其他
特征的 candidate drop 在 `-0.0416` 到 `0.0527` 之间。所有 13 个可评估特征在两个
non-discordant block 的 pass 都是 `False`，没有被标记为全局 proxy artifact。

## 2. 输入与排除项

按 `数据 → 标签 → 评测 → baseline → 架构 → 文献` 排查：

- **数据/哈希**：Goal47 diagnostic/per-seed/per-block、Goal45/46 CSV、四个 formal
  trace SHA256 全部匹配冻结值；checkpoint SHA256 也匹配；
- **标签/样本**：160 records、四 block × 40 seeds；strict125 discordant 数为
  `13/7`，non-discordant 数为 `27/33`，与 Goal47-50 一致；
- **评测/reference**：四个 block 的 reference partial rho 精确复现，error `0.0`；
- **baseline/重复性**：Goal47 的 `ou_sigma_m`、`dynamic_projection_median_m`、
  `ou_projection_mean_m` 在 analyzer 内与 per-seed CSV 对齐；
- **coverage**：13/14 可评估，超过预注册 `10/14`；唯一不可评估特征是常数
  requested-fin-norm 导致的 lag correlation，不是整族 coverage 不足；
- **数值/置换**：除该不可评估特征外，所有 per-seed numeric 特征 finite；valid
  permutation draws `9999/10000`，高于 `9000`；
- **non-discordant 混淆**：13 个可评估特征都没有在两个 non-discordant block 同时
  pass，因此不是全局 sampler 或全局状态 artifact；
- **Stage B 条件**：coverage 没有低于 `10/14`；没有 PARTIAL/SUPPORTED 候选依赖
  精确未记录参数；family/global permutation 与 coverage 均有效。三个条件都不
  满足，不能启动 Stage B replay。

## 3. 可证伪 hypotheses

### H1：filter-consistency 代理解释 residual

- 机制：`nav_drift_sigma` 较大的 seed 会有更大幅值/更异常几何的 USBL innovation，
  控制 innovation 特征后 strict125 partial association 应下降。
- 结果：5 个 filter 特征在两个 strict125 block 都未通过
  `rho <=0.30 && drop >=0.20`。例如 baseline/candidate range 分别是
  `rho 0.7928-0.8592` / `0.8715-0.9270`，candidate drop 全部不超过 `0.0142`。
- 裁决：`FALSIFIED`（仅针对这 5 个 raw-innovation 代理）。

### H2：DVL-calibration 代理解释 residual

- 机制：DVL bottom residual、scale、misalignment 或 lag-1 autocorrelation 与
  `nav_drift_sigma` 共线，并携带 late dead-reckoning 误差。
- 结果：4 个 DVL 特征在两个 strict125 block 都未通过；最强 candidate drop 只有
  `0.0626`，且 baseline drop 为负。
- 裁决：`FALSIFIED`（仅针对这 4 个 trace-derived DVL 代理）。

### H3：actuator-response 代理解释 residual

- 机制：first-order lag、rate clipping、slew 或 fin reversal 携带 Goal49/50 的
  mean/std/saturation 特征没有捕获的高频信息。
- 结果：4 个可评估 actuator 特征都未通过；第 5 个
  `actuator_best_lag_corr_late` 因 requested fin norm 为常数而不可评估。
- 裁决：可评估子集 `FALSIFIED`；完整 H3 不能写成被完全证伪。

### H0：negative control

- 机制：若某个代理只是全局 correlation，它会在两个 non-discordant block 也 pass。
- 结果：没有可评估特征在两个 non-discordant block 同时 pass。
- 裁决：没有发现 `GLOBAL_STATE_PROXY_NONSPECIFIC` 证据；但也没有任何 feature 达到
  `discordant_specific`。

## 4. 最便宜 falsification experiment

Goal51 的 Stage A 本身就是最便宜实验：在冻结的 160 条 Goal47 rows 上直接计算
source-aligned partial Spearman、same-feature two-block Gate、non-discordant
negative control 和 10,000 次 within-block sigma permutation。没有 replay、训练、
policy load、environment step 或 held-out 访问。

额外的 schema/常数检查确认 `actuator_best_lag_corr_late` 的不可评估原因：所有
160 个 seeds 的 late requested fin norm 都是常数；因此不能靠改 threshold、换 seed
或补 replay 在 Goal51 内把它“救”成可评估。

现有 traces 上不存在更便宜、且能把 H1/H2/H3 区分开的实验：

- 再换一个 late-window proxy 是 post-hoc feature sweep；
- 精确未记录参数 replay 不满足预注册 Stage B 触发条件；
- 继续降低 rho/drop 阈值会改变 Gate，不允许。

因此本轮 null 是可解释的，不需要也不能通过收紧/放宽门禁来“继续”。

## 5. 文献调研

已使用 `paper-search` 检索 arXiv/Crossref，并核验关键条目的摘要或元数据；详见
`research/paper_reading/2026-11-09-goal51-state-proxy-null.md`。文献支持三点：

1. USBL fix 的 temporal/geometric availability 可能比单个 innovation 幅值更关键；
2. DVL history、misalignment 和 actuator saturation/rate state 可能是独立状态，
   Goal51 的 proxy null 不能直接否定这些机制；
3. 观测性 feature sweep 不能替代 intervention；在 delayed/overlapping effects 下，
   要识别机制需要新的预注册干预，而不是继续加 observational proxy。

这些文献只用于解释 null 的边界，不用于把未记录状态写成已证实机制。

## 6. 裁决与下一步

```text
decision = stop
```

停止“用当前 Goal47 trace 上的低维状态代理解释 strict125 sigma residual”的路线。
不追加 Goal51 特征、不改变窗口/阈值、不启动 Stage B、不恢复 command-residual
PPO/supervised residual、不扫 reward/gamma/seed、不打开 held-out `100-149`。

若未来继续，必须新开预注册 goal，并明确选择一种不同的研究问题：controlled
intervention、独立的 outcome-relevance 诊断，或显式记录一个具体 hidden state。
在用户决定新主线前，不再自动启动新的 proxy 搜索。

## 7. 证据

- `doc/goals/goal51_unrecorded_state_proxy_audit_20261109.md`
- `doc/goals/goal51_phase0_audit_20261109.md`
- `doc/goals/goal51_result_20261109.md`
- `research/paper_reading/2026-11-09-goal51-state-proxy-null.md`
- `artifacts/lowlevel/goal51_unrecorded_state_audit_20261109/unrecorded_state_diagnostic.json`
- `artifacts/lowlevel/goal51_unrecorded_state_audit_20261109/unrecorded_state_feature_results.csv`
- `artifacts/lowlevel/goal51_unrecorded_state_audit_20261109/unrecorded_state_per_seed.csv`
